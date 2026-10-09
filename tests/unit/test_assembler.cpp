/// Tests of op::DenseStrategy, op::assemble_rhs, op::assemble_diagonal and op::validate (WP9).
///
/// Geometry: sphere of radius 0.5 um (docs/05), plane wave k = +z, E along x (docs/06). Physics
/// checks use icosphere n = 1 (80 triangles, 2N = 240) at lambda = 1 um, which has the same
/// h / lambda as n = 2 at 500 nm and keeps the sanitizer build fast; structure and the
/// ICTF / MCTF symmetry use n = 0 (2N = 60), the PMCHWT symmetry n = 1 at 500 nm, the
/// right-hand side n = 2 at 500 nm (no assembly). The arbiter of all signs is the residual of
/// the Galerkin system for the exact Mie currents projected onto RWG functions
/// (fields_test_support.hpp), taken as max(E-row residual / |b_E|, H-row residual / |b_H|)
/// (assembler_test_support.hpp; the combined norm would not see the H rows): a sign error
/// leaves an O(1) residual that does not decrease with refinement. The Ag jump-sign case, the
/// n = 2 / n = 3 residual convergence, the dense Mie solves at 500 nm and the assembly timing
/// are in tests/validation/test_assembler_mie.cpp.
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/kernels/quadrature.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "assembler_test_support.hpp"

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace assembler_test;
using formulation::Kind;

namespace {

/// Fast unit-test wavelength (icosphere n = 1 at 1 um ~ n = 2 at 500 nm in h / lambda).
constexpr Real kLambdaFast = 1e-6;

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Cheap kernel options of the structural cases (the WP7 scheme: one Dunavant outer rule for
/// touching pairs, fixed near / far degrees 8 / 3). Symmetry, signs, determinism and the smoke
/// solve do not depend on the WP7b accuracy, and the graded touching rule with the degree
/// selection would make the sanitizer run of these cases 5x to 10x slower (64 s for the
/// determinism case).
kernels::OperatorOptions wp7_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_near = 8;
    return opt;
}

}  // namespace

TEST_CASE("assembler: structure, diagonal and Problem validation", "[operator]") {
    // Icosahedron (n = 0, 20 triangles, N = 30): the structure does not depend on the
    // resolution (nor on the kernel options: cheap WP7 options).
    SphereCase c(lossless_n15(), 0, Kind::PMCHWT);
    c.problem().kernel_options = wp7_options();
    const Index N = c.size();
    REQUIRE(N == 30);
    const auto t0 = std::chrono::steady_clock::now();
    const auto op = op::DenseStrategy().build(c.problem());
    WARN("dense assembly n = 0 (2N = 60): " << seconds_since(t0) << " s");
    const auto Z = std::dynamic_pointer_cast<op::DenseOperator>(op);
    REQUIRE(Z != nullptr);
    CHECK(op::DenseStrategy().name() == "dense");
    CHECK(Z->rows() == 2 * N);
    CHECK(Z->cols() == 2 * N);
    CHECK(Z->describe() == "dense 60x60");
    CHECK(Z->memory_bytes() == 16u * 60u * 60u);
    const MatrixXc& Zm = Z->matrix();
    CHECK(Zm.allFinite());

    const VectorXc d = op::assemble_diagonal(c.problem());
    REQUIRE(d.size() == 2 * N);
    const VectorXc zd = Zm.diagonal();
    // Same blocks and summation order as the full assembly: bitwise equal.
    CHECK((d.array() == zd.array()).all());

    // Invalid problems.
    op::Problem& p = c.problem();
    const op::Problem good = p;
    const op::DenseStrategy dense;
    {
        op::Problem bad = good;
        bad.space = nullptr;
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
        CHECK_THROWS_AS(dense.build(bad), std::invalid_argument);
        CHECK_THROWS_AS(op::assemble_rhs(bad), std::invalid_argument);
        CHECK_THROWS_AS(op::assemble_diagonal(bad), std::invalid_argument);
    }
    {
        op::Problem bad = good;
        bad.formulation = nullptr;
        CHECK_THROWS_AS(dense.build(bad), std::invalid_argument);
    }
    {
        op::Problem bad = good;
        bad.omega = 0.0;
        CHECK_THROWS_AS(dense.build(bad), std::invalid_argument);
        CHECK_THROWS_AS(op::assemble_diagonal(bad), std::invalid_argument);
        bad.omega = -good.omega;
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
        bad.omega = std::nan("");
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
        // Without an excitation, omega alone defines the frequency.
        bad.omega = 1.01 * good.omega;
        bad.excitation = nullptr;
        CHECK_NOTHROW(op::validate(bad));
        CHECK_THROWS_AS(op::assemble_rhs(bad), std::invalid_argument);  // no excitation
    }
    {
        op::Problem bad = good;
        bad.omega = 1.01 * good.omega;  // mismatch with the excitation
        CHECK_THROWS_AS(dense.build(bad), std::invalid_argument);
        CHECK_THROWS_AS(op::assemble_rhs(bad), std::invalid_argument);
    }
    {
        op::Problem bad = good;
        bad.exterior = lossless_n15();  // the plane wave lives in vacuum
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
    }
    {
        op::Problem bad = good;
        bad.kernel_options.quad_degree_near = 3;  // negative weights: not positive-interior
        CHECK_THROWS_AS(dense.build(bad), std::invalid_argument);
        CHECK_THROWS_AS(op::assemble_rhs(bad), std::invalid_argument);
        bad.kernel_options = good.kernel_options;
        bad.kernel_options.quad_degree_far = 21;
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
    }
    {
        // Material parameters must be finite and non-zero in both regions (checked before the
        // parallel assembly, which would otherwise fail with a misleading message).
        const Real nan = std::nan("");
        const Real inf = std::numeric_limits<Real>::infinity();
        op::Problem bad = good;
        bad.object.eps_r = Complex(nan, 0.0);
        CHECK_THROWS_AS(dense.build(bad), std::invalid_argument);
        CHECK_THROWS_AS(op::assemble_diagonal(bad), std::invalid_argument);
        bad.object.eps_r = Complex(0.0, 0.0);
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
        bad.object = good.object;
        bad.object.mu_r = Complex(1.0, inf);
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
        bad.object = good.object;
        bad.excitation = nullptr;  // so that the background check does not fire first
        bad.exterior.mu_r = Complex(0.0, 0.0);
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
        bad.exterior = good.exterior;
        bad.exterior.eps_r = Complex(inf, 0.0);
        CHECK_THROWS_AS(op::validate(bad), std::invalid_argument);
    }
    CHECK_NOTHROW(op::validate(good));
}

TEST_CASE("assembler: PMCHWT is complex-symmetric after negating the M rows", "[operator]") {
    // docs/01 / docs/05 criterion: S = diag(I, -I) Z_PMCHWT symmetric to < 1e-6 relative. The
    // raw Z is block-antisymmetric in the K blocks (Z_JM^T = -Z_MJ). Off-diagonal far / near
    // blocks are symmetric only up to rounding of the quadrature sums (both orderings use the
    // same rules); touching blocks are exactly symmetric when element_blocks averages them over
    // both orderings (always with the WP7 options used here; on the WP7b default path above
    // |k| h = 1, and identical L always) and otherwise symmetric to their raw asymmetry
    // (< 1e-9 of the block norm, kernels::OperatorOptions). Symmetry is a property of the
    // discretisation, not of its resolution: icosphere n = 1 (2N = 240) with the cheap WP7
    // options keeps the sanitizer build fast; the ICTF / MCTF case below uses the defaults.
    SphereCase c(material::silver_500nm(), 1, Kind::PMCHWT);
    c.problem().kernel_options = wp7_options();
    const MatrixXc Z = assemble(c.problem());
    const Real s = symmetry_defect(Z, 1.0, 1.0);
    // The raw matrix is not symmetric (the K blocks enter with opposite signs).
    const Real raw = symmetry_defect(Z, 1.0, -1.0);
    WARN("PMCHWT Ag, n = 1: |S - S^T| / |S| = " << s << ", raw Z " << raw);
    // Measured (WP9): S defect 2.7e-16 (n = 1), 3.9e-16 / 4.7e-16 (n = 2, n = 1.5 / Ag); raw Z
    // 3.7e-3 (the K blocks are small against the L blocks of the PMCHWT matrix).
    CHECK(s < 1e-6);
    CHECK(raw > 1e-3);
}

TEST_CASE("assembler: ICTF and MCTF are symmetric after diag(b1 eta1, -a1/eta1) scaling",
          "[operator]") {
    // For all three formulations of Table 1, a_i/eta_i and b_i eta_i do not depend on i, so the
    // jump terms 1/2 (a1/eta1 - a2/eta2) I and 1/2 (b2 eta2 - b1 eta1) I cancel and the K blocks
    // reduce to sums of principal values; S = diag(b1 eta1 I, -(a1/eta1) I) Z is then symmetric
    // (assembler.cpp). With the PMCHWT scaling diag(I, -I) the ICTF matrix is not symmetric: its
    // off-diagonal blocks differ by the factor (a/eta)/(b eta) = 4 / (eta1 + eta2)^2.
    // Icosahedron (n = 0): non-coplanar touching pairs, so K is non-trivial. ICTF with the
    // default kernel options (the WP7b graded path and degree selection; |k| h > 1 here, so
    // shared-edge / shared-vertex blocks are averaged), MCTF with the cheap WP7 options (both
    // on defaults would take ~6.5 s in the sanitizer build).
    const material::Material mat = material::silver_500nm();
    for (const Kind kind : {Kind::ICTF, Kind::MCTF}) {
        SphereCase c(mat, 0, kind);
        if (kind == Kind::MCTF) {
            c.problem().kernel_options = wp7_options();
        }
        const MatrixXc Z = assemble(c.problem());
        const Complex eta1 = c.problem().exterior.wave_impedance(c.problem().omega);
        const Complex eta2 = c.problem().object.wave_impedance(c.problem().omega);
        const formulation::Weights w = c.form->weights(eta1, eta2);
        const Real s = symmetry_defect(Z, w.b1 * eta1, w.a1 / eta1);
        const Real pmchwt_scaling = symmetry_defect(Z, 1.0, 1.0);
        WARN(c.form->name() << " (Ag): |S - S^T| / |S| = " << s << ", with diag(I, -I) "
                            << pmchwt_scaling);
        CHECK(s < 1e-6);
        CHECK(pmchwt_scaling > 1e-2);
    }
}

TEST_CASE("assembler: jump-term signs of each region against exact Mie currents", "[operator]") {
    // Each region's T-EFIE / T-MFIE alone (assembler_test_support.hpp), n = 1.5 sphere,
    // icosphere n = 1 at lambda = 1 um. With the jump sign of a region flipped the residual is
    // O(1); with the assembler's signs (K_1 = K^PV - I/2, K_2 = K^PV + I/2) it is at the level
    // of the projection error. The Ag case (n = 2 at 500 nm) is in the validation test. Cheap
    // WP7 kernel options (the sign test does not depend on the quadrature accuracy).
    SphereCase c(lossless_n15(), 1, Kind::PMCHWT, kLambdaFast);
    c.problem().kernel_options = wp7_options();
    const JumpSignResiduals r = jump_sign_residuals(c);
    WARN("n = 1.5, n = 1, 1 um: region 1 E/H rows "
         << r.ok1.e << " / " << r.ok1.h << " (flipped " << r.flipped1.e << " / " << r.flipped1.h
         << "), region 2 " << r.ok2.e << " / " << r.ok2.h << " (flipped " << r.flipped2.e << " / "
         << r.flipped2.h << ")");
    // Measured (WP9): region 1 E/H rows 0.143 / 0.151 (flipped 0.979 / 1.217), region 2
    // 0.194 / 0.282 (flipped 0.926 / 1.160).
    for (const RowResiduals& ok : {r.ok1, r.ok2}) {
        CHECK(ok.e < 0.4);
        CHECK(ok.h < 0.4);
    }
    for (const RowResiduals& bad : {r.flipped1, r.flipped2}) {
        CHECK(bad.e > 0.6);
        CHECK(bad.h > 0.6);
    }
}

TEST_CASE("assembler: right-hand side", "[operator]") {
    SphereCase c(lossless_n15(), 2, Kind::PMCHWT);
    op::Problem& p = c.problem();
    const Index N = c.size();
    p.kernel_options.quad_degree_rhs = 8;  // the right-hand-side rule (default 8 since WP7c)
    const VectorXc b8 = op::assemble_rhs(p);
    p.kernel_options.quad_degree_rhs = 4;
    const VectorXc b4 = op::assemble_rhs(p);
    p.kernel_options.quad_degree_rhs = 12;
    const VectorXc b12 = op::assemble_rhs(p);
    const Real d4 = (b4 - b12).norm() / b12.norm();
    const Real d8 = (b8 - b12).norm() / b12.norm();
    WARN("rhs: degree 4 vs 12: " << d4 << ", degree 8 vs 12: " << d8);
    // Measured (WP9): 3.2e-5 (degree 4) and 4.9e-10 (degree 8, the WP7 default). The plane wave
    // varies by k h ~ 2 rad over a triangle of this mesh, so degree 4 cannot reach 1e-6; degree 8
    // (the WP7c default of quad_degree_rhs; the WP7b right-hand side used degree 19) does.
    CHECK(d4 < 1e-4);
    CHECK(d8 < 1e-6);
    CHECK(d8 < 1e-3 * d4);

    // Independent evaluation: PMCHWT weights a1/eta1 = b1 eta1 = 1, so b = [<f, x e^{-jkz}>;
    // (1/eta1) <f, y e^{-jkz}>] (degree-12 Dunavant rule, RwgSpace::value).
    const geometry::TriangleMesh& mesh = c.setup.mesh;
    const kernels::TriangleRule& rule = kernels::triangle_rule(12);
    const Real k = k_vacuum();
    VectorXc ref = VectorXc::Zero(2 * N);
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const basis::RwgSpace::Support s = c.setup.space.support(t);
        const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
        const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
        const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
            const Complex phase = std::exp(-kJ * k * r.z()) * (rule.weights[q] * mesh.area(t));
            for (int a = 0; a < s.count; ++a) {
                const Vec3 f = c.setup.space.value(s.n[a], t, r);
                ref(s.n[a]) += f.x() * phase;
                ref(N + s.n[a]) += f.y() * phase / constants::eta0;
            }
        }
    }
    CHECK((b12.head(N) - ref.head(N)).norm() <= 1e-12 * ref.head(N).norm());
    CHECK((b12.tail(N) - ref.tail(N)).norm() <= 1e-12 * ref.tail(N).norm());

    // ICTF weights scale the two halves by a1/eta1 = 2/(eta1+eta2) and b1 eta1 = (eta1+eta2)/2.
    SphereCase ci(lossless_n15(), 2, Kind::ICTF);
    ci.problem().kernel_options.quad_degree_rhs = 12;
    const VectorXc bi = op::assemble_rhs(ci.problem());
    const Complex eta1 = constants::eta0;
    const Complex eta2 = constants::eta0 / 1.5;
    CHECK((bi.head(N) - 2.0 / (eta1 + eta2) * b12.head(N)).norm() <= 1e-12 * bi.head(N).norm());
    CHECK((bi.tail(N) - (eta1 + eta2) / 2.0 * b12.tail(N)).norm() <= 1e-12 * bi.tail(N).norm());

    p.excitation = nullptr;
    CHECK_THROWS_AS(op::assemble_rhs(p), std::invalid_argument);
}

TEST_CASE("assembler: OpenMP thread count does not change Z or b (bitwise)", "[operator]") {
#ifdef SPECKLEBEM_HAVE_OPENMP
    // Icosahedron (n = 0, 20 triangles), ICTF with cheap WP7-scheme options (near and touching
    // degree 4), and icosphere n = 1 with cheaper quadrature (degrees 1 / 2 / 2) and one region
    // only (determinism depends on neither; with the default options this case took 64 s in the
    // sanitizer build) to keep the single-thread sanitizer run short; the colouring splits each
    // colour over the 3 threads. element_blocks itself is deterministic on every path
    // (test_operators.cpp).
    const SingleRegion region1(1);
    for (const int n : {0, 1}) {
        SphereCase c(material::silver_500nm(), n, Kind::ICTF);
        c.problem().kernel_options = wp7_options();
        c.problem().kernel_options.quad_degree_near = 4;
        c.problem().kernel_options.quad_degree_sing = 4;
        if (n == 1) {
            c.problem().formulation = &region1;
            c.problem().kernel_options.quad_degree_far = 1;
            c.problem().kernel_options.quad_degree_near = 2;
            c.problem().kernel_options.quad_degree_sing = 2;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int saved = omp_get_max_threads();
        omp_set_num_threads(1);
        const MatrixXc Z1 = assemble(c.problem());
        const VectorXc b1 = op::assemble_rhs(c.problem());
        const VectorXc d1 = op::assemble_diagonal(c.problem());
        omp_set_num_threads(3);
        const MatrixXc Z3 = assemble(c.problem());
        const VectorXc b3 = op::assemble_rhs(c.problem());
        const VectorXc d3 = op::assemble_diagonal(c.problem());
        omp_set_num_threads(saved);
        WARN("determinism, icosphere n = " << n << ": " << seconds_since(t0) << " s");
        CHECK((Z1.array() == Z3.array()).all());
        CHECK((b1.array() == b3.array()).all());
        CHECK((d1.array() == d3.array()).all());
        // The diagonal is summed in the order of the full assembly.
        CHECK((d1.array() == Z1.diagonal().array()).all());
    }
#else
    SUCCEED("built without OpenMP: assembly is sequential");
#endif
}

TEST_CASE("assembler: dense PMCHWT solve of the n = 1.5 sphere against Mie (smoke)", "[operator]") {
    // Smoke check of the full chain (assembly, rhs, LU, far field) on icosphere n = 1 at
    // lambda = 1 um (docs/05 metric eps_rr with E_ref = sqrt(sigma), 37 angles in the
    // xz-plane), plus the row-wise residual of the projected exact Mie currents. The first
    // solver results at 500 nm (n = 2, 3, PMCHWT / ICTF / Ag) are in the validation test.
    // Cheap WP7 kernel options (the discretisation error of n = 1 dominates by far).
    SphereCase c(lossless_n15(), 1, Kind::PMCHWT, kLambdaFast);
    c.problem().kernel_options = wp7_options();
    const SolveResult res = solve_and_compare(c);
    WARN("PMCHWT n = 1.5, icosphere 1, 1 um: eps_rr = "
         << res.eps_rr << ", LU residual " << res.residual << ", exact-current residual "
         << res.exact_residual);
    // Measured (WP9): eps_rr 0.034, exact-current residual 0.219 (row-wise).
    CHECK(res.residual < 1e-10);
    CHECK(res.exact_residual < 0.45);
    CHECK(res.eps_rr < 0.1);
}

// ---------------------------------------------------------------------------------------------
// WP7c: Dunavant degree of the right-hand side (OperatorOptions::quad_degree_rhs).
// ---------------------------------------------------------------------------------------------

namespace {

/// <f_m, E_inc> (first N entries) and <f_m, H_inc> (last N) with the degree-d Dunavant rule,
/// written directly with RwgSpace::value (independent of op::assemble_rhs; d may be any degree:
/// the incident fields are smooth everywhere).
VectorXc rhs_moments(const basis::RwgSpace& space, const excitation::Excitation& exc, int d) {
    const geometry::TriangleMesh& mesh = space.mesh();
    const kernels::TriangleRule& rule = kernels::triangle_rule(d);
    const Index N = space.size();
    VectorXc b = VectorXc::Zero(2 * N);
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const basis::RwgSpace::Support s = space.support(t);
        const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
        const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
        const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
            const Real w = rule.weights[q] * mesh.area(t);
            const Vec3c E = exc.electric_field(r);
            const Vec3c H = exc.magnetic_field(r);
            for (int a = 0; a < s.count; ++a) {
                const Vec3 f = space.value(s.n[a], t, r);
                b(s.n[a]) += w * (f(0) * E(0) + f(1) * E(1) + f(2) * E(2));
                b(N + s.n[a]) += w * (f(0) * H(0) + f(1) * H(1) + f(2) * H(2));
            }
        }
    }
    return b;
}

/// max over the E and H halves of |a - ref| / |ref|.
Real rhs_rel(const VectorXc& a, const VectorXc& ref) {
    const Index N = ref.size() / 2;
    return std::max((a.head(N) - ref.head(N)).norm() / ref.head(N).norm(),
                    (a.tail(N) - ref.tail(N)).norm() / ref.tail(N).norm());
}

Real longest_mesh_edge(const geometry::TriangleMesh& m) {
    Real h = 0.0;
    for (Index t = 0; t < m.num_triangles(); ++t) {
        for (Index i = 0; i < 3; ++i) {
            const Vec3 a = m.vertices().row(m.triangles()(t, i)).transpose();
            const Vec3 b = m.vertices().row(m.triangles()(t, (i + 1) % 3)).transpose();
            h = std::max(h, (a - b).norm());
        }
    }
    return h;
}

/// Closed rough-surface box with the uniform box at the top-face spacing (no grading).
geometry::TriangleMesh rhs_rough_box(Real L, Real spacing, Real sigma, Real depth) {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = L;
    p.rms_roughness = sigma;
    p.correlation_length = 200e-9;
    p.mesh_size = spacing;
    p.seed = 20261009;
    p.box_depth = depth;
    p.box_mesh_size = spacing;
    return geometry::make_rough_surface_mesh(p);
}

/// Degree study on one mesh: for lambda = ratio h (h the longest edge), a plane wave (+z, E
/// along x) and a paraxial Gaussian beam (w0 = lambda, focus at the origin, p polarisation), the
/// relative RHS error of the given degrees against degree 20. Returns the worst error of each
/// degree (index = degree) and appends a table row per excitation.
std::array<Real, 21> rhs_degree_errors(const geometry::TriangleMesh& mesh,
                                       std::initializer_list<Real> ratios,
                                       std::initializer_list<int> degrees, std::string& table) {
    const basis::RwgSpace space(mesh);
    const Real h = longest_mesh_edge(mesh);
    std::array<Real, 21> worst{};
    for (const Real ratio : ratios) {
        const Real lambda = ratio * h;
        excitation::GaussianBeam::Params gp;
        gp.wavelength = lambda;
        gp.waist_radius = lambda;
        const excitation::PlaneWave pw(lambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0));
        const excitation::GaussianBeam gb(gp);
        for (const excitation::Excitation* exc :
             {static_cast<const excitation::Excitation*>(&pw),
              static_cast<const excitation::Excitation*>(&gb)}) {
            const VectorXc ref = rhs_moments(space, *exc, 20);
            table += "\n  lambda/" + std::to_string(static_cast<int>(ratio)) +
                     (exc == &pw ? " plane wave:" : " Gaussian beam:");
            for (const int d : degrees) {
                const Real e = rhs_rel(rhs_moments(space, *exc, d), ref);
                auto& w = worst[static_cast<std::size_t>(d)];
                w = std::max(w, e);
                std::ostringstream os;
                os << std::scientific << std::setprecision(1) << " d" << d << " " << e;
                table += os.str();
            }
        }
    }
    return worst;
}

/// Regular check at h = lambda / 10 (the coarser end of the meshing range): the error decreases
/// with the degree and the default quad_degree_rhs is within 1e-8 of degree 20. The relative
/// error depends on h / lambda, not on the number of triangles, so small meshes suffice.
void check_rhs_default(const geometry::TriangleMesh& mesh, const char* name) {
    const int d0 = kernels::OperatorOptions{}.quad_degree_rhs;
    std::string table;
    const std::array<Real, 21> e = rhs_degree_errors(mesh, {10.0}, {4, 6, d0}, table);
    WARN("rhs degree check, " << name << " (relative error vs degree 20):" << table);
    CHECK(e[4] > e[6]);
    CHECK(e[6] > e[static_cast<std::size_t>(d0)]);
    CHECK(e[static_cast<std::size_t>(d0)] <= 1e-8);
}

}  // namespace

TEST_CASE("assembler: right-hand-side degree convergence (icosphere)", "[operator]") {
    check_rhs_default(geometry::make_icosphere(0.5e-6, 0), "icosahedron");
}

TEST_CASE("assembler: right-hand-side degree convergence (rough box)", "[operator]") {
    check_rhs_default(rhs_rough_box(0.3e-6, 75e-9, 20e-9, 0.15e-6), "rough box");
}

// Hidden slow case, registered with ctest as "assembler: rhs degree study (slow)" (label slow;
// SKIPs in unoptimised builds): every positive-interior degree against degree 20 on the
// icosphere n = 3 and a rough box (L = 0.6 um, 50 nm, depth 0.4 um) at h = lambda/10 and
// lambda/27, plane wave and Gaussian beam (the study behind the default, ADR 0004).
TEST_CASE("assembler: right-hand-side degree study", "[operator][.slow][rhsstudy]") {
#ifndef NDEBUG
    SKIP("rhs degree study: optimised builds only");
#endif
    const std::initializer_list<int> all = {1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19};
    const auto d0 = static_cast<std::size_t>(kernels::OperatorOptions{}.quad_degree_rhs);
    std::string t1;
    const std::array<Real, 21> e1 =
        rhs_degree_errors(geometry::make_icosphere(0.5e-6, 3), {10.0, 27.0}, all, t1);
    WARN("rhs degree study, icosphere n = 3 (relative error vs degree 20):" << t1);
    CHECK(e1[d0] <= 1e-8);
    std::string t2;
    const std::array<Real, 21> e2 =
        rhs_degree_errors(rhs_rough_box(0.6e-6, 50e-9, 50e-9, 0.4e-6), {10.0, 27.0}, all, t2);
    WARN("rhs degree study, rough box (relative error vs degree 20):" << t2);
    CHECK(e2[d0] <= 1e-8);
}

TEST_CASE("assembler: quad_degree_rhs alone sets the right-hand-side rule", "[operator]") {
    // PMCHWT (a1/eta1 = b1 eta1 = 1): b = [<f, E_inc>; <f, H_inc>]. With quad_degree_rhs = 19 (the
    // WP7b right-hand side, which used quad_degree_near = 19) b is the direct degree-19 evaluation
    // and does not depend on quad_degree_near; the default (8) differs from it by < 1e-8 even on
    // this coarse mesh (k h ~ 2).
    SphereCase c(lossless_n15(), 2, Kind::PMCHWT);
    op::Problem& p = c.problem();
    p.kernel_options.quad_degree_rhs = 19;
    p.kernel_options.quad_degree_near = 19;
    const VectorXc b19 = op::assemble_rhs(p);
    p.kernel_options.quad_degree_near = 8;
    CHECK((op::assemble_rhs(p).array() == b19.array()).all());
    const VectorXc direct = rhs_moments(c.setup.space, *p.excitation, 19);
    CHECK(rhs_rel(b19, direct) <= 1e-13);
    p.kernel_options = kernels::OperatorOptions{};
    const Real d = rhs_rel(op::assemble_rhs(p), b19);
    WARN("default right-hand side (degree 8) vs degree 19, icosphere n = 2 at 500 nm: " << d);
    CHECK(d <= 1e-8);
    p.kernel_options.quad_degree_rhs = 3;  // negative weights: not positive-interior
    CHECK_THROWS_AS(op::assemble_rhs(p), std::invalid_argument);
}
