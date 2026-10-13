/// Tests of the stretch-aware dense kernels and assembly (WP-PML2, ADR 0009 decisions 2 and 3):
/// kernels::element_blocks(..., const geometry::CoordinateStretch*, ...), stretched_pair_info and
/// op::Problem::stretch.
///
///  1. Uniform-scaling identity: a profile of order 0 starting at 0 gives r~ = s r in the plane
///     (s = 1 - j sigma_max), R~ = s R on flat triangles, and L~(k) = s L(k s) with the unstretched
///     block at the complex wavenumber k s (eps -> eps s^2 keeps omega mu): Piola factors, R~ and
///     the singular route in the isotropic case against the subtraction-based kernels of ADR 0004.
///  2. Anisotropic stretch (sigma on x only, and the corner sigma_x != sigma_y): touching blocks
///     converge in stretched_touching_points against p = 16 and, for a moderate uniform
///     anisotropy, against an independent plain relative-coordinate reference (the WP7b
///     reference rules of test_operators.cpp with the stretched kernel); near / far blocks
///     against a composite Dunavant brute force written from the definitions with
///     CoordinateStretch::map / jacobian_diag.
///  3. Reciprocity: the stretched dense PMCHWT matrix is complex symmetric after diag(I, -I).
///  4. Decay: the coupling of a current outside the PML to test functions in the PML decays like
///     exp(-k Int sigma) (the outgoing-wave damping of the stretch), against the 1 / R decay
///     without stretch.
///  5. Refusals of the paths that do not support a stretch.
///  6. Bitwise regression: no stretch, an identity stretch and a stretch that misses the mesh give
///     the unstretched blocks and matrices bitwise; with a PML, entries away from it too.
/// Plus the jump term (Piola factors cancel), option validation and the non-horizontal guard.
/// Heavy convergence sweeps run in optimised builds only (SKIP otherwise); the debug build runs
/// reduced cases.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/kernels/singularity.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/postprocessing/fields.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;
using basis::RwgSpace;
using geometry::CoordinateStretch;
using geometry::StretchProfile;
using geometry::TriangleMesh;
using kernels::OperatorOptions;
using kernels::Proximity;
using kernels::RegionParams;
using Block = Eigen::Matrix<Complex, 3, 3>;

namespace {

constexpr Real kPi = constants::pi;
constexpr Real kLambda = 500e-9;
const Complex kJ(0.0, 1.0);

#ifdef NDEBUG
constexpr bool kOptimised = true;
#else
constexpr bool kOptimised = false;
#endif

struct NamedRegion {
    const char* name;
    RegionParams p;
};

RegionParams region_of(const material::Material& m, Real lambda = kLambda) {
    const Real omega = 2.0 * kPi * constants::c0 / lambda;
    return {m.wavenumber(omega), m.wave_impedance(omega), omega, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

std::vector<NamedRegion> all_regions() {
    return {NamedRegion{"vacuum", region_of(material::vacuum())},
            NamedRegion{"Si", region_of(material::silicon_500nm())},
            NamedRegion{"Ag", region_of(material::silver_500nm())}};
}

/// Flat structured grid at height z: nx x ny cells of size h from (x0, y0), each cell split along
/// (i, j)-(i+1, j+1) like the open surface (open_surface.hpp).
TriangleMesh flat_grid(Index nx, Index ny, Real h, Real x0, Real y0, Real z = 0.0) {
    Vertices v((nx + 1) * (ny + 1), 3);
    const auto id = [&](Index i, Index j) { return j * (nx + 1) + i; };
    for (Index j = 0; j <= ny; ++j) {
        for (Index i = 0; i <= nx; ++i) {
            v.row(id(i, j)) << x0 + static_cast<Real>(i) * h, y0 + static_cast<Real>(j) * h, z;
        }
    }
    Triangles f(2 * nx * ny, 3);
    Index t = 0;
    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            f.row(t++) << id(i, j), id(i + 1, j + 1), id(i + 1, j);
            f.row(t++) << id(i, j), id(i, j + 1), id(i + 1, j + 1);
        }
    }
    return TriangleMesh(v, f);
}

/// Stretch with the same profile on both axes (sigma_y = -1: y unstretched).
CoordinateStretch stretch_xy(Real start, Real width, Real sigma_x, Real order,
                             Real sigma_y = -1.0) {
    const StretchProfile px{start, width, sigma_x, order};
    const StretchProfile py{start, width, sigma_y < 0.0 ? 0.0 : sigma_y, order};
    return CoordinateStretch(px, py);
}

/// Uniform stretch r~ = s r in the plane (order 0 from 0: x~ = x - j sgn(x) sigma |x|).
CoordinateStretch uniform_stretch(Real sigma) {
    return stretch_xy(0.0, 1.0, sigma, 0.0, sigma);
}

struct Blocks {
    Block L = Block::Zero();
    Block K = Block::Zero();
};

Blocks blocks(const RwgSpace& space, Index t1, Index t2, const RegionParams& reg,
              const OperatorOptions& opt, const CoordinateStretch* stretch) {
    Blocks b;
    kernels::element_blocks(space, t1, t2, reg, opt, stretch, b.L, b.K);
    return b;
}

Real rel_diff(const Block& a, const Block& ref) {
    const Real n = ref.norm();
    return n > 0.0 ? (a - ref).norm() / n : a.norm();
}

bool bitwise_equal(const Block& a, const Block& b) {
    return (a.array() == b.array()).all();
}

/// First pair (t1 < t2) of class `cls`, preferring triangles near the grid centre.
std::pair<Index, Index> pair_of_class(const TriangleMesh& m, Proximity cls, Index t1) {
    for (Index t2 = 0; t2 < m.num_triangles(); ++t2) {
        if (kernels::classify(m, t1, t2) == cls) {
            return {t1, t2};
        }
    }
    throw std::logic_error("pair_of_class: no pair of the requested class");
}

/// Pair at a centroid distance closest to ratio * h from t1 (non-touching).
std::pair<Index, Index> pair_at_distance(const TriangleMesh& m, Index t1, Real d) {
    Index best = -1;
    Real err = 1e300;
    for (Index t2 = 0; t2 < m.num_triangles(); ++t2) {
        const Proximity c = kernels::classify(m, t1, t2);
        if (c != Proximity::near && c != Proximity::far) {
            continue;
        }
        const Real e = std::abs((m.centroid(t1) - m.centroid(t2)).norm() - d);
        if (e < err) {
            err = e;
            best = t2;
        }
    }
    return {t1, best};
}

/// Composite Dunavant points of triangle t (n_sub^2 congruent sub-triangles), weights with area.
struct QuadSet {
    std::vector<Vec3> r;
    std::vector<Real> w;
};

QuadSet composite_points(const TriangleMesh& m, Index t, int degree, int n_sub) {
    const Vec3 v0 = m.vertices().row(m.triangles()(t, 0)).transpose();
    const Vec3 v1 = m.vertices().row(m.triangles()(t, 1)).transpose();
    const Vec3 v2 = m.vertices().row(m.triangles()(t, 2)).transpose();
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const Vec3 a = (v1 - v0) / n_sub;
    const Vec3 b = (v2 - v0) / n_sub;
    const Real sub_area = 0.5 * a.cross(b).norm();
    QuadSet out;
    const auto add = [&](const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            out.r.push_back(l(0) * p0 + l(1) * p1 + l(2) * p2);
            out.w.push_back(rule.weights[q] * sub_area);
        }
    };
    for (int i = 0; i < n_sub; ++i) {
        for (int j = 0; i + j < n_sub; ++j) {
            const Vec3 p = v0 + i * a + j * b;
            add(p, p + a, p + b);
            if (i + j < n_sub - 1) {
                add(p + a, p + a + b, p + b);
            }
        }
    }
    return out;
}

/// Stretched Galerkin blocks written from the definitions (ADR 0009 decision 2) with
/// CoordinateStretch::map / jacobian_diag, RwgSpace::value / divergence and std::complex:
///   L = int int [j w mu (DS f_m) . (DS' f_n) + (div f_m)(div' f_n) / (j w eps)] G(R~),
///   K = int int (DS f_m) . (grad~' G x DS' f_n), grad~' G = (1 + j k R~) G (r~ - r~') / R~^2.
/// The cross product is written out (Eigen's cross conjugates complex vectors).
Blocks brute_stretched(const RwgSpace& space, Index t1, Index t2, const RegionParams& reg,
                       const CoordinateStretch& st, const QuadSet& outer, const QuadSet& inner) {
    const RwgSpace::Support s1 = space.support(t1);
    const RwgSpace::Support s2 = space.support(t2);
    const auto n1 = static_cast<std::size_t>(s1.count);
    const auto n2 = static_cast<std::size_t>(s2.count);
    const auto mapped = [&](const RwgSpace::Support& s, std::size_t n, Index t, const Vec3& r,
                            std::array<Vec3c, 3>& f) {
        const Vec3c jd = st.jacobian_diag(r);
        for (std::size_t a = 0; a < n; ++a) {
            const Vec3 v = space.value(s.n[a], t, r);
            f[a] = Vec3c(jd(0) * v(0), jd(1) * v(1), jd(2) * v(2));
        }
    };
    std::vector<std::array<Vec3c, 3>> fi(inner.r.size());
    std::vector<Vec3c> yi(inner.r.size());
    for (std::size_t j = 0; j < inner.r.size(); ++j) {
        mapped(s2, n2, t2, inner.r[j], fi[j]);
        yi[j] = st.map(inner.r[j]);
    }
    std::array<Real, 3> d1{}, d2{};
    for (std::size_t a = 0; a < n1; ++a) {
        d1[a] = space.divergence(s1.n[a], t1);
    }
    for (std::size_t b = 0; b < n2; ++b) {
        d2[b] = space.divergence(s2.n[b], t2);
    }
    const Complex c_vec = kJ * reg.omega * reg.mu;
    const Complex c_sca = 1.0 / (kJ * reg.omega * reg.eps);
    Blocks out;
    for (std::size_t i = 0; i < outer.r.size(); ++i) {
        std::array<Vec3c, 3> fo{};
        mapped(s1, n1, t1, outer.r[i], fo);
        const Vec3c xo = st.map(outer.r[i]);
        for (std::size_t j = 0; j < inner.r.size(); ++j) {
            const Vec3c d = xo - yi[j];
            const Complex R = std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2));
            const Complex G = std::exp(-kJ * reg.k * R) / (4.0 * kPi * R);
            const Complex cg = (1.0 + kJ * reg.k * R) * G / (R * R);
            const Real w = outer.w[i] * inner.w[j];
            for (std::size_t a = 0; a < n1; ++a) {
                for (std::size_t b = 0; b < n2; ++b) {
                    const Vec3c& m = fo[a];
                    const Vec3c& n = fi[j][b];
                    const Complex dot = m(0) * n(0) + m(1) * n(1) + m(2) * n(2);
                    const Vec3c gxn(d(1) * n(2) - d(2) * n(1), d(2) * n(0) - d(0) * n(2),
                                    d(0) * n(1) - d(1) * n(0));
                    const auto ai = static_cast<Eigen::Index>(a);
                    const auto bi = static_cast<Eigen::Index>(b);
                    out.L(ai, bi) += w * (c_vec * dot + c_sca * (d1[a] * d2[b])) * G;
                    out.K(ai, bi) += w * cg * (m(0) * gxn(0) + m(1) * gxn(1) + m(2) * gxn(2));
                }
            }
        }
    }
    return out;
}

/// Brute force with the same composite rule on both triangles.
Blocks brute_composite(const RwgSpace& space, Index t1, Index t2, const RegionParams& reg,
                       const CoordinateStretch& st, int degree, int n_sub) {
    return brute_stretched(space, t1, t2, reg, st,
                           composite_points(space.mesh(), t1, degree, n_sub),
                           composite_points(space.mesh(), t2, degree, n_sub));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// 6. Bitwise regression.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: no stretch, identity and a missing PML keep the blocks bitwise",
          "[stretched_kernels]") {
    // Kernel level: every pair of a small flat grid and of a rough box, three regions.
    const Real h = 30e-9;
    std::vector<TriangleMesh> meshes;
    meshes.push_back(flat_grid(3, 3, h, -1.4 * h, -1.6 * h));
    {
        geometry::RoughSurfaceParams rp;
        rp.mesh_size = h;
        rp.edge_length_L = 3.0 * h;
        rp.rms_roughness = 0.1 * h;
        rp.correlation_length = 2.0 * h;
        rp.seed = 7;
        rp.box_depth = 2.0 * h;
        rp.box_mesh_size = h;
        meshes.push_back(geometry::make_rough_surface_mesh(rp));
    }
    const CoordinateStretch identity = CoordinateStretch::identity();
    const CoordinateStretch zero_sigma = stretch_xy(0.0, 1e-6, 0.0, 2.0, 0.0);
    const CoordinateStretch missing = stretch_xy(1e-6, 1e-6, 4.0, 2.0, 4.0);  // beyond the mesh
    REQUIRE(zero_sigma.is_identity());
    REQUIRE_FALSE(missing.is_identity());
    OperatorOptions opt;
    if (!kOptimised) {  // cheap rules in the sanitizer build (the dispatch is what is tested)
        opt.outer_grading_levels = 0;
        opt.target_accuracy = 0.0;
        opt.quad_degree_near = 5;
    }
    for (const TriangleMesh& mesh : meshes) {
        const RwgSpace space(mesh);
        for (const NamedRegion& reg : all_regions()) {
            Index compared = 0;
            for (Index t1 = 0; t1 < mesh.num_triangles(); t1 += (kOptimised ? 1 : 5)) {
                for (Index t2 = 0; t2 < mesh.num_triangles(); t2 += (kOptimised ? 1 : 3)) {
                    Block L0;
                    Block K0;
                    kernels::element_blocks(space, t1, t2, reg.p, opt, L0, K0);
                    for (const CoordinateStretch* s :
                         {static_cast<const CoordinateStretch*>(nullptr), &identity, &zero_sigma,
                          &missing}) {
                        const Blocks b = blocks(space, t1, t2, reg.p, opt, s);
                        CHECK(bitwise_equal(b.L, L0));
                        CHECK(bitwise_equal(b.K, K0));
                    }
                    ++compared;
                }
            }
            INFO(reg.name << ": " << compared << " pairs");
            CHECK(compared > 0);
        }
    }
}

namespace {

/// Small open surface with a PML (ADR 0009 geometry): 2 x 2 patch cells, one taper cell, no gap,
/// one PML cell per side (6 x 6 cells, 72 triangles), fixed seed.
geometry::OpenSurfaceMesh small_open_surface(Real h, Index pml_cells = 1) {
    geometry::RoughSurfaceParams rp;
    rp.mesh_size = h;
    rp.edge_length_L = 4.0 * h;  // patch 2 h + 2 taper cells
    rp.rms_roughness = 0.1 * h;
    rp.correlation_length = 2.0 * h;
    rp.seed = 11;
    const geometry::HeightMap map = geometry::generate_gaussian_height_map(rp);
    geometry::OpenSurfaceParams op;
    op.patch_size = 2.0 * h;
    op.taper_width = h;
    op.gap_width = 0.0;
    op.pml_width = static_cast<Real>(pml_cells) * h;
    return geometry::make_open_surface_mesh(map, op);
}

/// Dense Z of a problem.
MatrixXc dense(const op::Problem& p) {
    const std::shared_ptr<op::LinearOperator> lo = op::DenseStrategy().build(p);
    auto* d = dynamic_cast<op::DenseOperator*>(lo.get());
    REQUIRE(d != nullptr);
    return std::move(d->matrix());
}

/// Cheap options for structural tests (the WP7 scheme for unstretched pairs, p = 4 for stretched
/// touching pairs, fixed near / far degrees).
OperatorOptions cheap_options() {
    OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_near = 5;
    opt.stretched_touching_points = 4;
    return opt;
}

struct OpenCase {
    geometry::OpenSurfaceMesh om;
    RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
    explicit OpenCase(Real h, formulation::Kind kind = formulation::Kind::PMCHWT)
        : om(small_open_surface(h)), space(om.mesh), form(formulation::make_formulation(kind)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = material::silver_500nm();
        problem.formulation = form.get();
        problem.omega = 2.0 * kPi * constants::c0 / kLambda;
        problem.kernel_options = cheap_options();
    }
};

/// Per basis function: true if one of its support triangles has a vertex in the PML zone.
std::vector<bool> touches_pml(const RwgSpace& space, const CoordinateStretch& pml) {
    const TriangleMesh& m = space.mesh();
    std::vector<bool> out(static_cast<std::size_t>(space.size()), false);
    for (Index n = 0; n < space.size(); ++n) {
        for (const Index t : {space.plus_triangle(n), space.minus_triangle(n)}) {
            for (Index i = 0; i < 3; ++i) {
                const Vec3 v = m.vertices().row(m.triangles()(t, i)).transpose();
                if (pml.in_pml(v)) {
                    out[static_cast<std::size_t>(n)] = true;
                }
            }
        }
    }
    return out;
}

/// Entries of Z (2N x 2N) whose two basis functions do not touch the PML must equal Z0 bitwise;
/// returns {entries away from the PML, changed entries elsewhere}.
std::pair<Index, Index> compare_away_from_pml(const MatrixXc& Z, const MatrixXc& Z0,
                                              const std::vector<bool>& touches) {
    const auto N = static_cast<Index>(touches.size());
    Index same = 0;
    Index changed = 0;
    for (Index a = 0; a < 2 * N; ++a) {
        for (Index b = 0; b < 2 * N; ++b) {
            const bool away = !touches[static_cast<std::size_t>(a % N)] &&
                              !touches[static_cast<std::size_t>(b % N)];
            if (away) {
                CHECK(Z(a, b) == Z0(a, b));
                ++same;
            } else if (Z(a, b) != Z0(a, b)) {
                ++changed;
            }
        }
    }
    return {same, changed};
}

}  // namespace

TEST_CASE("stretched kernels: dense Z of a rough box is bitwise unchanged without a stretch",
          "[stretched_kernels]") {
    // Rough box: nullptr, identity and a stretch that misses the mesh give the same matrix and
    // diagonal bitwise.
    const Real h = 50e-9;
    {
        geometry::RoughSurfaceParams rp;
        rp.mesh_size = h;
        rp.edge_length_L = 3.0 * h;
        rp.rms_roughness = 0.1 * h;
        rp.correlation_length = 2.0 * h;
        rp.seed = 3;
        rp.box_depth = 2.0 * h;
        rp.box_mesh_size = h;
        const TriangleMesh box = geometry::make_rough_surface_mesh(rp);
        const RwgSpace space(box);
        const auto form = formulation::make_formulation(formulation::Kind::PMCHWT);
        op::Problem p;
        p.space = &space;
        p.exterior = material::vacuum();
        p.object = material::silicon_500nm();
        p.formulation = form.get();
        p.omega = 2.0 * kPi * constants::c0 / kLambda;
        p.kernel_options = cheap_options();
        const MatrixXc Z0 = dense(p);
        const CoordinateStretch identity = CoordinateStretch::identity();
        const CoordinateStretch missing = stretch_xy(1e-6, 1e-6, 4.0, 2.0, 4.0);
        for (const CoordinateStretch* s : {&identity, &missing}) {
            p.stretch = s;
            const MatrixXc Z = dense(p);
            CHECK((Z.array() == Z0.array()).all());
            CHECK((op::assemble_diagonal(p).array() == Z0.diagonal().array()).all());
        }
    }
}

TEST_CASE("stretched kernels: dense Z of an open surface is bitwise unchanged away from the PML",
          "[stretched_kernels]") {
    // Open surface: the identity stretch gives the same matrix bitwise. With the PML, every entry
    // whose two basis functions have no stretched support triangle is bitwise the unstretched
    // entry, and the diagonal path agrees with the stretched matrix.
    const Real h = 50e-9;
    OpenCase c(h);
    const MatrixXc Z0 = dense(c.problem);
    const CoordinateStretch identity = CoordinateStretch::identity();
    c.problem.stretch = &identity;
    CHECK((dense(c.problem).array() == Z0.array()).all());
    const CoordinateStretch pml = geometry::make_pml_stretch(c.om, 3.0);
    c.problem.stretch = &pml;
    const MatrixXc Z = dense(c.problem);
    const auto [same, changed] = compare_away_from_pml(Z, Z0, touches_pml(c.space, pml));
    INFO("entries away from the PML: " << same << ", changed entries: " << changed);
    CHECK(same > 0);
    CHECK(changed > 0);
    // The diagonal path agrees with the dense matrix also with the stretch.
    CHECK((op::assemble_diagonal(c.problem).array() == Z.diagonal().array()).all());
}

TEST_CASE("stretched kernels: dense ICTF of an open surface with a PML (smoke test)",
          "[stretched_kernels]") {
    // ICTF (block-row scalings of PMCHWT, docs/03) on the small open surface with a PML of
    // sigma_max = 3: every entry is finite, and every entry whose two basis functions have no
    // stretched support triangle is bitwise the unstretched ICTF entry.
    const Real h = 50e-9;
    OpenCase c(h, formulation::Kind::ICTF);
    const MatrixXc Z0 = dense(c.problem);
    const CoordinateStretch pml = geometry::make_pml_stretch(c.om, 3.0);
    c.problem.stretch = &pml;
    const MatrixXc Z = dense(c.problem);
    CHECK(Z.allFinite());
    const auto [same, changed] = compare_away_from_pml(Z, Z0, touches_pml(c.space, pml));
    INFO("ICTF: entries away from the PML: " << same << ", changed entries: " << changed);
    CHECK(same > 0);
    CHECK(changed > 0);
}

// ---------------------------------------------------------------------------------------------
// 1. Uniform-scaling identity.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: uniform stretch, L~(k) = s L(k s) and K = 0 on flat pairs",
          "[stretched_kernels]") {
    // r~ = s r (order 0 from 0) on z = 0: R~ = s R, DS = s on in-plane vectors, so
    //   L~ = s^2 (1/s) L_vec(k s) + (1/s) L_sca(k s) = s L(k s) with eps -> eps s^2 (omega mu
    //   kept, (k s)^2 = omega^2 mu eps s^2). The reference is the unstretched element_blocks at
    //   the complex wavenumber k s (singularity subtraction, ADR 0004), touching remainder degree
    //   19 and near / far target 1e-10; the stretched side uses the relative-coordinate rule
    //   (isotropic: no ridge split) and the stretched degree selection.
    // h = lambda / 50 keeps |k s| h <= 1.6 (Ag, sigma = 4), where the reference is ~1e-8.
    const Real h = kLambda / 50.0;
    const TriangleMesh mesh = flat_grid(5, 5, h, -2.31 * h, -2.17 * h);
    const RwgSpace space(mesh);
    const Index tc = 24;  // a triangle near the centre
    std::vector<std::pair<const char*, std::pair<Index, Index>>> pairs = {
        {"identical", {tc, tc}},
        {"shared edge", pair_of_class(mesh, Proximity::shared_edge, tc)},
        {"shared vertex", pair_of_class(mesh, Proximity::shared_vertex, tc)},
        {"near", pair_at_distance(mesh, tc, 1.6 * h)},
        {"far", pair_at_distance(mesh, tc, 3.5 * h)},
    };
    OperatorOptions opt;
    opt.target_accuracy = 1e-10;
    OperatorOptions ref_opt = opt;
    ref_opt.quad_degree_sing = 19;
    const std::vector<Real> sigmas =
        kOptimised ? std::vector<Real>{0.5, 2.0, 4.0} : std::vector<Real>{2.0};
    for (const Real sigma : sigmas) {
        const CoordinateStretch st = uniform_stretch(sigma);
        const Complex s(1.0, -sigma);
        for (const NamedRegion& reg : all_regions()) {
            RegionParams rs = reg.p;
            rs.k = reg.p.k * s;
            rs.eps = reg.p.eps * s * s;
            for (const auto& [name, pr] : pairs) {
                const kernels::StretchedPairInfo info =
                    kernels::stretched_pair_info(mesh, pr.first, pr.second, reg.p.k, opt, st);
                REQUIRE(info.stretched);
                CHECK_FALSE(info.ridge_split);
                const Blocks b = blocks(space, pr.first, pr.second, reg.p, opt, &st);
                const Blocks r = blocks(space, pr.first, pr.second, rs, ref_opt, nullptr);
                const Real err = rel_diff(b.L, s * r.L);
                const bool touching = info.proximity == Proximity::identical ||
                                      info.proximity == Proximity::shared_edge ||
                                      info.proximity == Proximity::shared_vertex;
                WARN("uniform sigma = " << sigma << " " << reg.name << " " << name
                                        << ": |L~ - s L(ks)| / |s L| = " << err << " (|k s| h = "
                                        << std::abs(rs.k) * h << ", points " << info.points << ")");
                // Touching: the reference's own remainder error grows like (|k s| h)^4 (ADR
                // 0004; ~3e-9 at |k s| h <= 1, ~4e-7 at 2.2).
                const Real ksh = std::abs(rs.k) * h;
                CHECK(err < (touching ? (ksh <= 1.0 ? 2e-8 : 1e-6) : 1e-11));
                CHECK(b.K == Block::Zero());  // flat pair: in-plane triple products
                CHECK(r.K.norm() <= 1e-12 * r.L.norm() * std::abs(reg.p.eta));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Jump term.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: the Piola factors cancel in the jump term of horizontal triangles",
          "[stretched_kernels]") {
    // int (DS f_m) . (n x DS f_n) / det DS dS (det DS = s_x s_y, the Piola-mapped jump integrand)
    // equals jump_block for flat triangles, for a non-uniform anisotropic stretch.
    const Real h = 40e-9;
    const TriangleMesh mesh = flat_grid(4, 3, h, 0.5 * h, -1.5 * h);
    const RwgSpace space(mesh);
    const CoordinateStretch st = stretch_xy(1.0 * h, 2.0 * h, 4.0, 2.0, 1.5);
    const kernels::TriangleRule& rule = kernels::triangle_rule(10);
    Index stretched = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        Block I;
        kernels::jump_block(space, t, I);
        const RwgSpace::Support s = space.support(t);
        const Vec3 n = mesh.normal(t);
        const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
        const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
        const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
        Block P = Block::Zero();
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
            const Vec3c jd = st.jacobian_diag(r);
            const Complex det = jd(0) * jd(1) * jd(2);
            for (int a = 0; a < s.count; ++a) {
                const Vec3 fa = space.value(s.n[a], t, r);
                const Vec3c ma(jd(0) * fa(0), jd(1) * fa(1), jd(2) * fa(2));
                for (int b = 0; b < s.count; ++b) {
                    const Vec3 fb = space.value(s.n[b], t, r);
                    const Vec3c mb(jd(0) * fb(0), jd(1) * fb(1), jd(2) * fb(2));
                    const Vec3c nxm(n(1) * mb(2) - n(2) * mb(1), n(2) * mb(0) - n(0) * mb(2),
                                    n(0) * mb(1) - n(1) * mb(0));
                    P(a, b) += rule.weights[q] * mesh.area(t) *
                               (ma(0) * nxm(0) + ma(1) * nxm(1) + ma(2) * nxm(2)) / det;
                }
            }
        }
        if (st.in_pml(mesh.centroid(t))) {
            ++stretched;
        }
        CHECK((P - I).norm() <= 1e-14 * std::max(I.norm(), 1.0));
    }
    CHECK(stretched > 0);
}

// ---------------------------------------------------------------------------------------------
// 2. Anisotropic stretch: touching pairs.
// ---------------------------------------------------------------------------------------------

namespace {

/// Touching pairs of the central triangle of a grid (identical, all edge and vertex neighbours).
std::vector<std::pair<Index, Index>> touching_of(const TriangleMesh& m, Index t) {
    std::vector<std::pair<Index, Index>> out;
    for (Index s = 0; s < m.num_triangles(); ++s) {
        const Proximity c = kernels::classify(m, t, s);
        if (c == Proximity::identical || c == Proximity::shared_edge ||
            c == Proximity::shared_vertex) {
            out.emplace_back(t, s);
        }
    }
    return out;
}

const char* class_name(Proximity p) {
    switch (p) {
        case Proximity::identical:
            return "identical";
        case Proximity::shared_edge:
            return "shared edge";
        case Proximity::shared_vertex:
            return "shared vertex";
        case Proximity::near:
            return "near";
        case Proximity::far:
            break;
    }
    return "far";
}

/// Worst relative L error of the touching pairs of t at p points against p_ref; also checks
/// K = 0 (coplanar) and the bitwise transpose of the swapped ordering.
Real touching_convergence(const RwgSpace& space, Index t, const RegionParams& reg,
                          const CoordinateStretch& st, int p, int p_ref, const char* label) {
    OperatorOptions opt;
    opt.stretched_touching_points = p;
    OperatorOptions ref = opt;
    ref.stretched_touching_points = p_ref;
    Real worst = 0.0;
    const std::vector<std::pair<Index, Index>> tp = touching_of(space.mesh(), t);
    for (std::size_t ip = 0; ip < tp.size(); ip += (kOptimised ? 1 : 4)) {
        const auto [t1, t2] = tp[ip];
        const Blocks b = blocks(space, t1, t2, reg, opt, &st);
        const Blocks r = blocks(space, t1, t2, reg, ref, &st);
        const Blocks bs = blocks(space, t2, t1, reg, opt, &st);
        CHECK(bitwise_equal(bs.L, b.L.transpose()));
        CHECK(b.K == Block::Zero());
        const Real e = rel_diff(b.L, r.L);
        const kernels::StretchedPairInfo info =
            kernels::stretched_pair_info(space.mesh(), t1, t2, reg.k, opt, st);
        INFO(label << " " << class_name(info.proximity) << " (" << t1 << ", " << t2
                   << "): p = " << p << " vs " << p_ref << ": " << e << ", ridge width "
                   << info.ridge_width << ", points " << info.points);
        CHECK(info.stretched);
        worst = std::max(worst, e);
    }
    return worst;
}

/// Anisotropic touching sweep: strip of the PML on x only (sigma_y = 0, order-2 profile from 0,
/// W = 2 lambda, the central triangle of a 3 x 3 grid at sigma ~ 1, 2.6, 3.4; the grid stays
/// inside the profile, whose sigma' jumps at W) and the corner (sigma_x ~ 3.4, sigma_y ~ 1.5 and
/// 2): all touching pairs of the central triangle at h = lambda / 10, vacuum (|k s| h <= 2.2)
/// and Ag (|k s| h up to ~7, strongly damped; the k-aware point floor of the rule applies).
/// Self-convergence against p_ref ("a much higher level"); p = 10 is the default.
void anisotropic_sweep(bool full) {
    const Real h = kLambda / 10.0;
    const Real W = 2.0 * kLambda;
    struct Case {
        const char* name;
        Real frac_x;  // position of the centre in units of W from the start (x)
        Real frac_y;  // the same for y (< 0: y unstretched)
        Real tol10;   // bound at p = 10 (measured, with margin)
    };
    std::vector<Case> cases = {
        {"strip sigma 2.6", 0.8, -1.0, 2e-6},
        {"corner 3.4 / 1.5", 0.92, 0.61, 1e-7},
    };
    if (full) {
        cases = {
            {"strip sigma 1", 0.5, -1.0, 1e-7},    {"strip sigma 2.6", 0.8, -1.0, 2e-6},
            {"strip sigma 3.4", 0.92, -1.0, 2e-6}, {"corner 3.4 / 1.5", 0.92, 0.61, 1e-7},
            {"corner 3.4 / 2", 0.92, 0.71, 1e-7},
        };
    }
    const std::vector<int> ps = full ? std::vector<int>{6, 8, 10, 12} : std::vector<int>{10};
    const int p_ref = full ? 16 : 14;
    std::vector<NamedRegion> regs = {all_regions()[0]};
    if (full) {
        regs.push_back(all_regions()[2]);
    }
    for (const Case& c : cases) {
        const Real xc = c.frac_x * W;
        const Real yc = c.frac_y < 0.0 ? 0.0 : c.frac_y * W;
        const TriangleMesh mesh = flat_grid(3, 3, h, xc - 1.5 * h, yc - 1.5 * h);
        const RwgSpace space(mesh);
        const CoordinateStretch st =
            c.frac_y < 0.0 ? stretch_xy(0.0, W, 4.0, 2.0) : stretch_xy(0.0, W, 4.0, 2.0, 4.0);
        for (const NamedRegion& reg : regs) {
            std::ostringstream row;
            Real e10 = 0.0;
            for (const int p : ps) {
                const Real e = touching_convergence(space, 8, reg.p, st, p, p_ref, c.name);
                row << " p" << p << ": " << e;
                if (p == 10) {
                    e10 = e;
                }
            }
            WARN("anisotropic touching, " << c.name << ", " << reg.name << " (vs p = " << p_ref
                                          << "):" << row.str());
            CHECK(e10 < c.tol10);
        }
    }
}

}  // namespace

TEST_CASE("stretched kernels: anisotropic touching pairs converge in stretched_touching_points",
          "[stretched_kernels]") {
    if (!kOptimised) {
        // Reduced debug case: one strip position, p = 6 against p = 10.
        const Real h = kLambda / 10.0;
        const Real W = 2.0 * kLambda;
        const Real x_c = 0.8 * W;  // sigma = 4 * 0.8^2 = 2.56 at the centre
        const TriangleMesh mesh = flat_grid(3, 3, h, x_c - 1.5 * h, -1.5 * h);
        const RwgSpace space(mesh);
        const CoordinateStretch st = stretch_xy(0.0, W, 4.0, 2.0);
        const Real e = touching_convergence(space, 8, all_regions()[0].p, st, 6, 10, "debug");
        WARN("debug strip sigma ~ 2.6: p = 6 vs 10: " << e);
        CHECK(e < 5e-5);  // release: 1.1e-5 (p = 6 against p = 16)
        return;
    }
    anisotropic_sweep(false);
}

TEST_CASE("stretched kernels (full): anisotropic touching sweep",
          "[stretched_kernels][.slow][stretched_sweep]") {
    if (!kOptimised) {
        SKIP("optimised builds only");
    }
    anisotropic_sweep(true);
}

namespace {

/// Plain relative-coordinate reference (the WP7b rules of test_operators.cpp: reference_identical,
/// reference_shared_edge, reference_shared_vertex) with the stretched kernel of a UNIFORM
/// anisotropic stretch (the kernel depends on r - r' only, so the inner x / s rules of the
/// reference stay exact). Independent of the ridge split and of the production code.
struct StretchedSum {
    const RwgSpace* space;
    Index t1, t2;
    RegionParams reg;
    Complex sx, sy;  ///< uniform Jacobian factors
    Block L = Block::Zero();
    void add(const Vec3& x, const Vec3& y, Real w) {
        const RwgSpace::Support a = space->support(t1);
        const RwgSpace::Support b = space->support(t2);
        const Vec3c d((x(0) - y(0)) * sx, (x(1) - y(1)) * sy, x(2) - y(2));
        const Complex R = std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2));
        const Complex G = w * std::exp(-kJ * reg.k * R) / (4.0 * kPi * R);
        const Complex c_vec = kJ * reg.omega * reg.mu;
        const Complex c_sca = 1.0 / (kJ * reg.omega * reg.eps);
        for (int i = 0; i < a.count; ++i) {
            const Vec3 fa = space->value(a.n[i], t1, x);
            for (int j = 0; j < b.count; ++j) {
                const Vec3 fb = space->value(b.n[j], t2, y);
                const Complex dot =
                    sx * sx * fa(0) * fb(0) + sy * sy * fa(1) * fb(1) + fa(2) * fb(2);
                L(i, j) += (c_vec * dot +
                            c_sca * space->divergence(a.n[i], t1) * space->divergence(b.n[j], t2)) *
                           G;
            }
        }
    }
};

std::vector<std::pair<Real, Real>> gauss01(int n) {
    const kernels::LineRule g = kernels::gauss_legendre(n);
    std::vector<std::pair<Real, Real>> r;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        r.emplace_back(0.5 * (g.nodes[i] + 1.0), 0.5 * g.weights[i]);
    }
    return r;
}

Vec3 vtx(const TriangleMesh& m, Index v) {
    return m.vertices().row(v).transpose();
}

void ref_identical(const TriangleMesh& m, Index t, int n, StretchedSum& sum) {
    const Vec3 v0 = vtx(m, m.triangles()(t, 0));
    const Vec3 e1 = vtx(m, m.triangles()(t, 1)) - v0;
    const Vec3 e2 = vtx(m, m.triangles()(t, 2)) - v0;
    const Real jac = e1.cross(e2).norm();
    const std::array<std::array<Real, 2>, 6> hex = {
        {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {1.0, -1.0}}};
    const auto rule = gauss01(n);
    const kernels::TriangleRule& tri = kernels::triangle_rule(4);
    for (std::size_t i = 0; i < 6; ++i) {
        const auto& p = hex[i];
        const auto& q = hex[(i + 1) % 6];
        for (const auto& [rho, wr] : rule) {
            for (const auto& [u, wu] : rule) {
                const Real z1 = rho * (p[0] + u * (q[0] - p[0]));
                const Real z2 = rho * (p[1] + u * (q[1] - p[1]));
                const Real a = std::max(0.0, -z1);
                const Real b = std::max(0.0, -z2);
                const Real c = 1.0 - std::max(0.0, z1 + z2);
                const Real size = c - a - b;
                const Real w0 = jac * jac * rho * wr * wu * 0.5 * size * size;
                for (std::size_t k = 0; k < tri.weights.size(); ++k) {
                    const Vec3& l = tri.barycentric[k];
                    const Real s = l(0) * a + l(1) * (c - b) + l(2) * a;
                    const Real tt = l(0) * b + l(1) * b + l(2) * (c - a);
                    const Vec3 x = v0 + s * e1 + tt * e2;
                    sum.add(x, x + z1 * e1 + z2 * e2, w0 * tri.weights[k]);
                }
            }
        }
    }
}

void ref_shared_vertex(const Vec3& A, const Vec3& a1, const Vec3& a2, const Vec3& b1,
                       const Vec3& b2, int n, StretchedSum& sum) {
    const Vec3 da1 = a1 - A, da2 = a2 - A, db1 = b1 - A, db2 = b2 - A;
    const Real jac = da1.cross(da2).norm() * db1.cross(db2).norm();
    const auto rule = gauss01(n);
    for (const bool edge_on_test : {true, false}) {
        for (const auto& [lam, wl] : rule) {
            for (const auto& [sg, wsg] : rule) {
                for (const auto& [mu, wm] : rule) {
                    for (const auto& [nu, wn] : rule) {
                        const Real w = jac * lam * lam * lam * mu * wl * wsg * wm * wn;
                        const Real t0 = mu * (1.0 - nu);
                        const Real t1 = mu * nu;
                        if (edge_on_test) {
                            sum.add(A + lam * (sg * da1 + (1.0 - sg) * da2),
                                    A + lam * (t0 * db1 + t1 * db2), w);
                        } else {
                            sum.add(A + lam * (t0 * da1 + t1 * da2),
                                    A + lam * (sg * db1 + (1.0 - sg) * db2), w);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

TEST_CASE("stretched kernels: ridge-split touching rule against an independent plain reference",
          "[stretched_kernels]") {
    // Uniform anisotropic stretch sigma_x = 0.5, sigma_y = 0 (ridge width 0.8: the ridge split is
    // active), vacuum: identical and shared-vertex pairs against the plain relative-coordinate
    // reference with n = 24 (converged to ~1e-10 for this anisotropy in the WP-PML2 prototype
    // study; plain rules need no split for it, only many points).
    const Real h = kLambda / 10.0;
    const TriangleMesh mesh = flat_grid(3, 3, h, -1.37 * h, -1.41 * h);
    const RwgSpace space(mesh);
    const Real sigma = 0.5;
    const CoordinateStretch st = stretch_xy(0.0, 1.0, sigma, 0.0);
    const RegionParams reg = all_regions()[0].p;
    const Index t = 8;
    const int n_ref = kOptimised ? 24 : 14;
    OperatorOptions opt;
    for (const auto& [t1, t2] : touching_of(mesh, t)) {
        const Proximity c = kernels::classify(mesh, t1, t2);
        if (c == Proximity::shared_edge) {
            continue;  // the shared-edge reference needs the polytope split (covered above)
        }
        if (!kOptimised && c == Proximity::shared_vertex && t2 % 3 != 0) {
            continue;
        }
        StretchedSum sum{&space, t1, t2, reg, Complex(1.0, -sigma), Complex(1.0, 0.0)};
        if (c == Proximity::identical) {
            ref_identical(mesh, t1, n_ref, sum);
        } else {
            const auto& f = mesh.triangles();
            std::vector<Index> shared, only1, only2;
            for (Index i = 0; i < 3; ++i) {
                const Index v = f(t1, i);
                (v == f(t2, 0) || v == f(t2, 1) || v == f(t2, 2) ? shared : only1).push_back(v);
                const Index w = f(t2, i);
                if (!(w == f(t1, 0) || w == f(t1, 1) || w == f(t1, 2))) {
                    only2.push_back(w);
                }
            }
            ref_shared_vertex(vtx(mesh, shared[0]), vtx(mesh, only1[0]), vtx(mesh, only1[1]),
                              vtx(mesh, only2[0]), vtx(mesh, only2[1]), n_ref, sum);
        }
        const Blocks b = blocks(space, t1, t2, reg, opt, &st);
        const kernels::StretchedPairInfo info =
            kernels::stretched_pair_info(mesh, t1, t2, reg.k, opt, st);
        CHECK(info.ridge_split);
        const Real e = rel_diff(b.L, sum.L);
        WARN("independent reference, sigma_x = 0.5, " << class_name(c) << " (" << t1 << ", " << t2
                                                      << "): " << e);
        CHECK(e < (kOptimised ? 1e-8 : 1e-6));
    }
}

namespace {

using P3d = std::array<Real, 3>;

/// Splits a convex polygon by the plane n . p = 0 (through the origin) into its two sides (as
/// split_polygon in test_operators.cpp).
std::vector<std::vector<P3d>> split_polygon(const std::vector<P3d>& poly, const P3d& n) {
    const auto dot = [](const P3d& a, const P3d& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    std::vector<P3d> pos;
    std::vector<P3d> neg;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const P3d& a = poly[i];
        const P3d& b = poly[(i + 1) % poly.size()];
        const Real da = dot(n, a);
        const Real db = dot(n, b);
        if (da >= 0.0) {
            pos.push_back(a);
        }
        if (da <= 0.0) {
            neg.push_back(a);
        }
        if ((da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0)) {
            const Real s = da / (da - db);
            pos.push_back(
                {a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])});
            neg.push_back(pos.back());
        }
    }
    std::vector<std::vector<P3d>> out;
    for (std::vector<P3d>* side : {&pos, &neg}) {
        if (side->size() >= 3) {
            out.push_back(*side);
        }
    }
    return out;
}

/// The WP7b shared-edge reference rule (reference_shared_edge of test_operators.cpp) with the
/// stretched kernel of a UNIFORM stretch (StretchedSum) and one midpoint subdivision of every
/// cone base triangle (four cones per base): on whole facets the plain rule converges slowly
/// (~1e-7 at n = 10 even without stretch, WP-PML2 study), subdivided to ~1e-12. Shared edge AB,
/// x = A + s e + t a, y = A + (s + z) e + t' b (e = B - A, a = C - A, b = C' - A); the kernel
/// depends on (z, t, t') only, the integrand is quadratic in s on [max(0, -z), min(1 - t, 1 - t'
/// - z)] (2-point Gauss-Legendre, exact). (z, t, t') ranges over the cones from the origin over
/// the four outer facets of the polytope, split by z = 0 and t - t' - z = 0; every tetrahedron
/// (0, q0, q1, q2) by p = rho [q0 + u (q1 - q0) + u v (q2 - q1)], dp = |det(q0, q1, q2)| rho^2 u.
/// No ridge split: independent of the production rule.
void ref_shared_edge(const Vec3& A, const Vec3& B, const Vec3& C, const Vec3& Cp, int n,
                     StretchedSum& sum) {
    const Vec3 e = B - A;
    const Vec3 a = C - A;
    const Vec3 b = Cp - A;
    const Real jac = e.cross(a).norm() * e.cross(b).norm();
    std::vector<std::vector<P3d>> facets = {{{0, 1, 0}, {1, 1, 0}, {0, 1, 1}},
                                            {{-1, 0, 1}, {0, 0, 1}, {0, 1, 1}},
                                            {{1, 0, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}},
                                            {{-1, 0, 0}, {0, 1, 0}, {0, 1, 1}, {-1, 0, 1}}};
    for (const P3d& plane : {P3d{1, 0, 0}, P3d{-1, 1, -1}}) {
        std::vector<std::vector<P3d>> next;
        for (const auto& f : facets) {
            for (auto& piece : split_polygon(f, plane)) {
                next.push_back(std::move(piece));
            }
        }
        facets = std::move(next);
    }
    const auto rule = gauss01(n);
    const auto rule_s = gauss01(2);
    const auto cone = [&](const P3d& q0, const P3d& q1, const P3d& q2) {
        const Real det = std::abs(q0[0] * (q1[1] * q2[2] - q1[2] * q2[1]) -
                                  q0[1] * (q1[0] * q2[2] - q1[2] * q2[0]) +
                                  q0[2] * (q1[0] * q2[1] - q1[1] * q2[0]));
        for (const auto& [rho, wr] : rule) {
            for (const auto& [u, wu] : rule) {
                for (const auto& [v, wv] : rule) {
                    P3d p{};
                    for (std::size_t c = 0; c < 3; ++c) {
                        p[c] = rho * (q0[c] + u * (q1[c] - q0[c]) + u * v * (q2[c] - q1[c]));
                    }
                    const Real z = p[0];
                    const Real t = p[1];
                    const Real tp = p[2];
                    const Real lo = std::max(0.0, -z);
                    const Real hi = std::min(1.0 - t, 1.0 - tp - z);
                    if (hi <= lo) {
                        continue;
                    }
                    const Real w0 = jac * det * rho * rho * u * wr * wu * wv * (hi - lo);
                    for (const auto& [ss, ws] : rule_s) {
                        const Real s = lo + (hi - lo) * ss;
                        sum.add(A + s * e + t * a, A + (s + z) * e + tp * b, w0 * ws);
                    }
                }
            }
        }
    };
    const auto mid = [](const P3d& p, const P3d& q) {
        return P3d{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
    };
    for (const auto& f : facets) {
        for (std::size_t i = 1; i + 1 < f.size(); ++i) {
            const P3d& q0 = f[0];
            const P3d& q1 = f[i];
            const P3d& q2 = f[i + 1];
            const P3d m01 = mid(q0, q1);
            const P3d m12 = mid(q1, q2);
            const P3d m20 = mid(q2, q0);
            cone(q0, m01, m20);
            cone(m01, q1, m12);
            cone(m20, m12, q2);
            cone(m01, m12, m20);
        }
    }
}

}  // namespace

TEST_CASE("stretched kernels: ridge-split shared-edge pairs against an independent reference",
          "[stretched_kernels]") {
    // Uniform anisotropic stretch sigma_x = 0.5 and 1, sigma_y = 0 (the ridge split is active),
    // vacuum: the three shared-edge pairs of the central triangle against ref_shared_edge (plain,
    // no ridge split) with n = 24 (release; WP-PML2 review: agreement 1e-13 .. 1e-9). Bound 1e-8
    // as for the identical and shared-vertex pairs above; debug: one pair, n = 14, 1e-6.
    const Real h = kLambda / 10.0;
    const TriangleMesh mesh = flat_grid(3, 3, h, -1.37 * h, -1.41 * h);
    const RwgSpace space(mesh);
    const RegionParams reg = all_regions()[0].p;
    const Index t = 8;
    const int n_ref = kOptimised ? 24 : 14;
    const std::vector<Real> sigmas =
        kOptimised ? std::vector<Real>{0.5, 1.0} : std::vector<Real>{1.0};
    OperatorOptions opt;
    Index compared = 0;
    for (const Real sigma : sigmas) {
        const CoordinateStretch st = stretch_xy(0.0, 1.0, sigma, 0.0);
        Index done = 0;
        for (const auto& [t1, t2] : touching_of(mesh, t)) {
            if (kernels::classify(mesh, t1, t2) != Proximity::shared_edge ||
                (!kOptimised && done > 0)) {
                continue;
            }
            const auto& f = mesh.triangles();
            std::vector<Index> shared, only1, only2;
            for (Index i = 0; i < 3; ++i) {
                const Index v = f(t1, i);
                (v == f(t2, 0) || v == f(t2, 1) || v == f(t2, 2) ? shared : only1).push_back(v);
                const Index w = f(t2, i);
                if (!(w == f(t1, 0) || w == f(t1, 1) || w == f(t1, 2))) {
                    only2.push_back(w);
                }
            }
            REQUIRE(shared.size() == 2);
            StretchedSum sum{&space, t1, t2, reg, Complex(1.0, -sigma), Complex(1.0, 0.0)};
            ref_shared_edge(vtx(mesh, shared[0]), vtx(mesh, shared[1]), vtx(mesh, only1[0]),
                            vtx(mesh, only2[0]), n_ref, sum);
            const Blocks b = blocks(space, t1, t2, reg, opt, &st);
            const kernels::StretchedPairInfo info =
                kernels::stretched_pair_info(mesh, t1, t2, reg.k, opt, st);
            CHECK(info.ridge_split);
            CHECK(b.K == Block::Zero());
            const Real e = rel_diff(b.L, sum.L);
            WARN("independent shared-edge reference, sigma_x = " << sigma << " (" << t1 << ", "
                                                                 << t2 << "), ridge width "
                                                                 << info.ridge_width << ": " << e);
            CHECK(e < (kOptimised ? 1e-8 : 1e-6));
            ++done;
            ++compared;
        }
    }
    CHECK(compared == (kOptimised ? 6 : 1));
}

// ---------------------------------------------------------------------------------------------
// 2. Anisotropic stretch: near and far pairs.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: near and far pairs against the composite brute force",
          "[stretched_kernels]") {
    // Order-2 profile on x only (W = 2 lambda), corner with sigma_y too; pairs at D / h = 1.6 to
    // 8 in several directions (along the ridge and across it), vacuum and Si, and Ag in the deep
    // strip. The stretched rule with the default target 1e-5 must be within 1e-5 of the
    // composite degree-10 brute force on 4^2 sub-triangles whenever stretched_pair_info reports
    // target_reached; pairs that hit the degree cap are held to the regression bound 1e-3 and
    // reported. The brute force uses CoordinateStretch::map / jacobian_diag directly, so this
    // also checks the internal stretch evaluation.
    const Real h = kLambda / 10.0;
    const Real W = 2.0 * kLambda;
    const int n_sub = 4;
    struct Case {
        const char* name;
        Real xc, yc;
        bool corner;
    };
    const std::vector<Case> cases =
        kOptimised ? std::vector<Case>{{"strip sigma 1", 0.5 * W, 0.0, false},
                                       {"strip sigma 2.4-4", W - 4.6 * h, 0.0, false},
                                       {"corner", W - 4.6 * h, 0.7 * W, true}}
                   : std::vector<Case>{{"strip sigma 1.3-3.8", 0.75 * W, 0.0, false}};
    OperatorOptions opt;
    Index reached = 0;
    Index capped = 0;
    for (const Case& c : cases) {
        const TriangleMesh mesh = flat_grid(9, 9, h, c.xc - 4.5 * h, c.yc - 4.5 * h);
        const RwgSpace space(mesh);
        const CoordinateStretch st =
            c.corner ? stretch_xy(0.0, W, 4.0, 2.0, 4.0) : stretch_xy(0.0, W, 4.0, 2.0);
        const Index t0 = 80;  // centre cell
        // Debug (ASan, -O0; the brute force dominates): one near and one far partner.
        std::vector<Index> partners;
        bool have_near = false;
        bool have_far = false;
        for (Index s = 0; s < mesh.num_triangles(); s += (kOptimised ? 3 : 1)) {
            const Proximity cl = kernels::classify(mesh, t0, s);
            if (kOptimised && (cl == Proximity::near || cl == Proximity::far)) {
                partners.push_back(s);
            } else if (!kOptimised && cl == Proximity::near && !have_near) {
                partners.push_back(s);
                have_near = true;
            } else if (!kOptimised && cl == Proximity::far && !have_far && s % 37 == 0) {
                partners.push_back(s);
                have_far = true;
            }
        }
        std::vector<NamedRegion> regs = {all_regions()[0]};
        if (kOptimised) {
            regs.push_back(all_regions()[1]);
            if (!c.corner && c.xc > 0.6 * W) {
                regs.push_back(all_regions()[2]);  // Ag deep strip: most pairs at the cap
            }
        }
        for (const NamedRegion& reg : regs) {
            Real worst = 0.0;
            Real worst_capped = 0.0;
            Index capped_here = 0;
            for (const Index s : partners) {
                const kernels::StretchedPairInfo info =
                    kernels::stretched_pair_info(mesh, t0, s, reg.p.k, opt, st);
                REQUIRE(info.stretched);
                const Blocks b = blocks(space, t0, s, reg.p, opt, &st);
                const Blocks r = brute_composite(space, t0, s, reg.p, st, 10, n_sub);
                const Real e = rel_diff(b.L, r.L);
                INFO(c.name << " " << reg.name << " pair (" << t0 << ", " << s
                            << "), D / h = " << (mesh.centroid(t0) - mesh.centroid(s)).norm() / h
                            << ", degree " << info.degree << ", D_eff / D " << info.distance_factor
                            << ", error " << e);
                if (info.target_reached) {
                    CHECK(e < 1e-5);
                    worst = std::max(worst, e);
                    ++reached;
                } else {
                    // Degree cap 19 (the error model cannot meet the target; along the ridge of
                    // the deep strip): a regression bound, not an accuracy target (measured
                    // worst 4.9e-4, Ag at sigma 2.4-4; the fast PML ring WP-PML5 owns these).
                    CHECK(e < 1e-3);
                    worst_capped = std::max(worst_capped, e);
                    ++capped;
                    ++capped_here;
                }
                CHECK(b.K == Block::Zero());
            }
            WARN("near / far " << c.name << " " << reg.name << ": " << partners.size()
                               << " pairs, worst error " << worst << " (target reached), "
                               << worst_capped << " (degree cap, " << capped_here << " pairs)");
        }
    }
    WARN("near / far pairs: " << reached << " met the target, " << capped << " at the cap");
    CHECK(reached > 0);
}

// ---------------------------------------------------------------------------------------------
// 2. Non-flat pairs next to a flat PML: K, reciprocity without the transpose.
// ---------------------------------------------------------------------------------------------

namespace {

/// flat_grid whose vertices with x < x_flat are lifted by a smooth bump of amplitude ~amp h
/// (tilted, unstretched triangles); the vertices with x >= x_flat stay at z = 0 (the flat PML,
/// ADR 0009 decision 1). The last bumped column touches the PML start with gap 0.
TriangleMesh bumped_grid(Index nx, Index ny, Real h, Real x0, Real y0, Real x_flat, Real amp) {
    const TriangleMesh flat = flat_grid(nx, ny, h, x0, y0);
    Vertices v = flat.vertices();
    for (Index i = 0; i < v.rows(); ++i) {
        if (v(i, 0) < x_flat - 1e-9 * h) {
            const Real u = (v(i, 0) - x0) / h;
            const Real w = (v(i, 1) - y0) / h;
            v(i, 2) = amp * h * (0.6 * std::sin(1.3 * u + 0.4) + 0.4 * std::cos(0.9 * w + 1.1));
        }
    }
    return TriangleMesh(v, flat.triangles());
}

/// The mesh with the triangle order reversed (t -> F - 1 - t, rows and vertices unchanged).
/// element_blocks evaluates a stretched pair in the ordering (min, max) of the triangle indices
/// and transposes for the other one; in the reversed numbering the same geometric pair is
/// therefore integrated with test and source triangle exchanged, which bypasses that transpose.
TriangleMesh reversed_triangles(const TriangleMesh& m) {
    const Index F = m.num_triangles();
    Triangles f(F, 3);
    for (Index t = 0; t < F; ++t) {
        f.row(F - 1 - t) = m.triangles().row(t);
    }
    return TriangleMesh(m.vertices(), f);
}

/// Block B of the pair (t, s) with the RWG signs removed and the slots keyed by the position
/// (0..2 in the triangle's row) of the free vertex: independent of the edge numbering and of the
/// plus / minus assignment, so blocks of the original and the reversed mesh can be compared.
Block by_free_vertex(const RwgSpace& space, Index t, Index s, const Block& B) {
    const auto keys = [&](Index tri, std::array<Eigen::Index, 3>& pos, std::array<Real, 3>& sg) {
        const RwgSpace::Support sp = space.support(tri);
        for (int a = 0; a < sp.count; ++a) {
            const Index n = sp.n[a];
            const bool plus = space.plus_triangle(n) == tri;
            const Index fv = plus ? space.plus_free_vertex(n) : space.minus_free_vertex(n);
            const auto ua = static_cast<std::size_t>(a);
            for (Index i = 0; i < 3; ++i) {
                if (space.mesh().triangles()(tri, i) == fv) {
                    pos[ua] = static_cast<Eigen::Index>(i);
                }
            }
            sg[ua] = plus ? 1.0 : -1.0;
        }
        return sp.count;
    };
    std::array<Eigen::Index, 3> pt{}, ps{};
    std::array<Real, 3> st{}, ss{};
    const int nt = keys(t, pt, st);
    const int ns = keys(s, ps, ss);
    Block out = Block::Zero();
    for (int a = 0; a < nt; ++a) {
        for (int b = 0; b < ns; ++b) {
            const auto ua = static_cast<std::size_t>(a);
            const auto ub = static_cast<std::size_t>(b);
            out(pt[ua], ps[ub]) = (st[ua] * ss[ub]) * B(a, b);
        }
    }
    return out;
}

/// Test geometry of the non-flat cases: 7 x 4 cells at h = lambda / 10, the four columns x < x_p
/// bumped (amplitude 0.3 h; tilted, unstretched triangles), the three columns x >= x_p flat and
/// stretched by the order-2 profile on x from x_p = 10 h, W = 3 h, sigma_max = 4 (sigma up to
/// 0.44, 1.8 and 4 per column; the grid ends at x_p + W, inside the profile).
struct BumpedCase {
    static constexpr Index nx = 7;
    static constexpr Index ny = 4;
    Real h = kLambda / 10.0;
    Real xp = 10.0 * h;
    TriangleMesh mesh;
    TriangleMesh reversed;
    RwgSpace space;
    RwgSpace rspace;
    CoordinateStretch st;
    explicit BumpedCase(Real order = 2.0, Real sigma_max = 4.0)
        : mesh(bumped_grid(nx, ny, kLambda / 10.0, 6.0 * kLambda / 10.0, -2.0 * kLambda / 10.0,
                           10.0 * kLambda / 10.0, 0.3)),
          reversed(reversed_triangles(mesh)),
          space(mesh),
          rspace(reversed),
          st(stretch_xy(10.0 * kLambda / 10.0, 3.0 * kLambda / 10.0, sigma_max, order)) {}
    /// Triangle q (0, 1) of cell (i, j).
    [[nodiscard]] static Index tri(Index i, Index j, Index q) { return 2 * (j * nx + i) + q; }
    [[nodiscard]] Index rev(Index t) const { return mesh.num_triangles() - 1 - t; }
};

}  // namespace

TEST_CASE("stretched kernels: K of non-flat pairs next to a flat PML against the brute force",
          "[stretched_kernels]") {
    // BumpedCase: pairs (tilted test triangle, flat stretched source) are on the stretched path and
    // have K != 0. Production L and K against the composite degree-10 brute force on n_sub^2
    // sub-triangles (brute_stretched), near and far partners, vacuum and Ag (k_2 of
    // material::silver_500nm, eps_r ~ -9.4 - 1.2j):
    //  * target reached: relative block error < 1e-5 (target_accuracy) for L; K: see the known
    //    shortfall below (target checked without failing, regression bound 1e-4);
    //  * degree cap: regression bound 1e-3 (not an accuracy target);
    //  * reciprocity without the transpose: brute_stretched(s, t) (the swapped composite sets) is
    //    brute_stretched(t, s)^T to 1e-13, L and K (the integrand symmetry K_mn(t, s) = K_nm(s, t)
    //    of the file comment of stretched_operators.cpp), and the production blocks of the
    //    reversed triangle numbering (the other canonical ordering) agree to 1e-12 (the same
    //    point pairs in another order).
    const BumpedCase bc;
    const int n_sub = kOptimised ? 4 : 3;  // debug (ASan, -O0): the brute force dominates
    const std::vector<Index> tests =
        kOptimised ? std::vector<Index>{BumpedCase::tri(3, 1, 0), BumpedCase::tri(3, 2, 1),
                                        BumpedCase::tri(1, 2, 0)}
                   : std::vector<Index>{BumpedCase::tri(3, 1, 0)};
    const std::vector<NamedRegion> regs = {all_regions()[0], all_regions()[2]};
    OperatorOptions opt;
    Index reached = 0;
    Index capped = 0;
    Real worst_l = 0.0;
    Real worst_k = 0.0;
    Real worst_cap = 0.0;
    Real worst_recip = 0.0;
    Real worst_order = 0.0;
    Real worst_self = 0.0;
    for (const Index t0 : tests) {
        REQUIRE_FALSE(bc.st.in_pml(bc.mesh.centroid(t0)));
        // Partners: stretched triangles (columns 4..6), near and far; debug: one of each.
        std::vector<Index> partners;
        bool have_near = false;
        bool have_far = false;
        for (Index s = 0; s < bc.mesh.num_triangles(); ++s) {
            const Proximity cl = kernels::classify(bc.mesh, t0, s);
            if (!bc.st.in_pml(bc.mesh.centroid(s)) ||
                (cl != Proximity::near && cl != Proximity::far)) {
                continue;
            }
            if (kOptimised || (cl == Proximity::near && !have_near) ||
                (cl == Proximity::far && !have_far)) {
                partners.push_back(s);
                (cl == Proximity::near ? have_near : have_far) = true;
            }
        }
        REQUIRE(partners.size() >= 2);
        const QuadSet q0 = composite_points(bc.mesh, t0, 10, n_sub);
        for (const Index s : partners) {
            const QuadSet qs = composite_points(bc.mesh, s, 10, n_sub);
            for (const NamedRegion& reg : regs) {
                const kernels::StretchedPairInfo info =
                    kernels::stretched_pair_info(bc.mesh, t0, s, reg.p.k, opt, bc.st);
                REQUIRE(info.stretched);
                const Blocks b = blocks(bc.space, t0, s, reg.p, opt, &bc.st);
                const Blocks r = brute_stretched(bc.space, t0, s, reg.p, bc.st, q0, qs);
                REQUIRE(r.K.norm() > 0.0);
                const Real el = rel_diff(b.L, r.L);
                const Real ek = rel_diff(b.K, r.K);
                INFO(reg.name << " pair (" << t0 << ", " << s << "), D / h = "
                              << (bc.mesh.centroid(t0) - bc.mesh.centroid(s)).norm() / bc.h
                              << ", degree " << info.degree << ", D_eff / D "
                              << info.distance_factor << ": L " << el << ", K " << ek);
                if (kOptimised && s == partners.front()) {
                    // Brute force self-convergence (n_sub vs 2 n_sub; measured 5e-12).
                    const Blocks r2 = brute_composite(bc.space, t0, s, reg.p, bc.st, 10, 2 * n_sub);
                    const Real self = std::max(rel_diff(r.L, r2.L), rel_diff(r.K, r2.K));
                    CHECK(self < 1e-9);
                    worst_self = std::max(worst_self, self);
                }
                if (info.target_reached) {
                    CHECK(el < 1e-5);
                    // Non-flat pairs select their degree with the K margin kNonFlatLogMargin
                    // (without it one far pair, vacuum, D / h = 4.1 along the ridge, had K at
                    // 1.05e-5 with degree 9; degree 10 gives 8.7e-7).
                    CHECK(ek < 1e-5);
                    worst_l = std::max(worst_l, el);
                    worst_k = std::max(worst_k, ek);
                    ++reached;
                } else {
                    CHECK(el < 1e-3);
                    CHECK(ek < 1e-3);
                    worst_cap = std::max({worst_cap, el, ek});
                    ++capped;
                }
                // Reciprocity of the integrand (brute force, swapped roles; the first four
                // partners of each test triangle).
                if (s <= partners[kOptimised ? std::min<std::size_t>(3, partners.size() - 1) : 0]) {
                    const Blocks rs = brute_stretched(bc.space, s, t0, reg.p, bc.st, qs, q0);
                    const Real er =
                        std::max(rel_diff(rs.L.transpose(), r.L), rel_diff(rs.K.transpose(), r.K));
                    CHECK(er < 1e-13);
                    worst_recip = std::max(worst_recip, er);
                }
                // Production kernel in the other canonical ordering (reversed numbering).
                const Blocks br = blocks(bc.rspace, bc.rev(t0), bc.rev(s), reg.p, opt, &bc.st);
                const Real eo =
                    std::max(rel_diff(by_free_vertex(bc.rspace, bc.rev(t0), bc.rev(s), br.L),
                                      by_free_vertex(bc.space, t0, s, b.L)),
                             rel_diff(by_free_vertex(bc.rspace, bc.rev(t0), bc.rev(s), br.K),
                                      by_free_vertex(bc.space, t0, s, b.K)));
                CHECK(eo < 1e-12);
                worst_order = std::max(worst_order, eo);
            }
        }
    }
    WARN("non-flat near / far next to the PML: "
         << reached << " pairs met the target (worst L " << worst_l << ", K " << worst_k << "), "
         << capped << " at the cap (worst " << worst_cap << "); brute force self-convergence "
         << worst_self << "; reciprocity (brute) " << worst_recip
         << ", both orderings (production) " << worst_order);
    CHECK(reached > 0);
}

TEST_CASE("stretched kernels: touching pairs next to a flat PML agree in both orderings",
          "[stretched_kernels]") {
    // BumpedCase: the stretched touching pairs of a tilted triangle of the last bumped column
    // (non-coplanar shared edge / vertex at the PML start, K != 0, no ridge split) and of a PML
    // triangle in the middle PML column (coplanar, ridge split, sigma 0.4-1.8), evaluated in the
    // original and in the reversed triangle numbering: the relative-coordinate rules then
    // integrate the pair with test and source exchanged (bypassing the transpose of the
    // canonical ordering). Measured (release, p = 10): shared vertex <= 4e-14 (the rule is
    // symmetric under the exchange up to rounding), shared edge <= 1.5e-12 (coplanar) and
    // 6.7e-11 (K, non-coplanar); bound 1e-9 (debug p = 6: 1e-6).
    const BumpedCase bc;
    OperatorOptions opt;
    if (!kOptimised) {
        opt.stretched_touching_points = 6;
    }
    const std::vector<Index> tests = {BumpedCase::tri(3, 1, 0), BumpedCase::tri(5, 1, 1)};
    const RegionParams reg = all_regions()[0].p;
    Real worst_nc = 0.0;
    Real worst_c = 0.0;
    Index n_nc = 0;
    Index n_c = 0;
    for (const Index t0 : tests) {
        Index done = 0;
        for (const auto& [t1, t2] : touching_of(bc.mesh, t0)) {
            const kernels::StretchedPairInfo info =
                kernels::stretched_pair_info(bc.mesh, t1, t2, reg.k, opt, bc.st);
            if (!info.stretched || info.proximity == Proximity::identical ||
                (!kOptimised && done >= 2)) {
                continue;
            }
            const Blocks b = blocks(bc.space, t1, t2, reg, opt, &bc.st);
            const Blocks br = blocks(bc.rspace, bc.rev(t1), bc.rev(t2), reg, opt, &bc.st);
            const Block l0 = by_free_vertex(bc.space, t1, t2, b.L);
            const Block l1 = by_free_vertex(bc.rspace, bc.rev(t1), bc.rev(t2), br.L);
            const Real el = rel_diff(l1, l0);
            Real ek = 0.0;
            if (info.coplanar) {
                CHECK(b.K == Block::Zero());
                CHECK(br.K == Block::Zero());
                worst_c = std::max(worst_c, el);
                ++n_c;
            } else {
                REQUIRE(b.K.norm() > 0.0);
                ek = rel_diff(by_free_vertex(bc.rspace, bc.rev(t1), bc.rev(t2), br.K),
                              by_free_vertex(bc.space, t1, t2, b.K));
                worst_nc = std::max({worst_nc, el, ek});
                ++n_nc;
            }
            WARN("both orderings, " << class_name(info.proximity) << " (" << t1 << ", " << t2
                                    << "), " << (info.coplanar ? "coplanar" : "non-coplanar")
                                    << ", ridge width " << info.ridge_width << ": L " << el
                                    << ", K " << ek);
            CHECK(el < (kOptimised ? 1e-9 : 1e-6));
            CHECK(ek < (kOptimised ? 1e-9 : 1e-6));
            ++done;
        }
    }
    WARN("touching pairs in both orderings: " << n_nc << " non-coplanar (worst " << worst_nc
                                              << "), " << n_c << " coplanar (worst " << worst_c
                                              << ")");
    CHECK(n_nc > 0);
    CHECK(n_c > 0);
}

namespace {

/// Relative angular width eps = sqrt(lambda_min / lambda_max) of M = f_r f_r^T + f_i f_i^T for
/// F_+ = dy - c dx, c = j s_x / s_y (s = 1 - j sigma; operators.hpp, StretchedPairInfo), written
/// from the definition with std::complex.
Real ridge_eps(Real sigma_x, Real sigma_y) {
    const Complex c = kJ * Complex(1.0, -sigma_x) / Complex(1.0, -sigma_y);
    const Real m00 = std::norm(c);
    const Real m01 = -c.real();
    const Real m11 = 1.0;
    const Real tr = m00 + m11;
    const Real det = m00 * m11 - m01 * m01;
    const Real disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    const Real lmax = 0.5 * tr + disc;
    return std::sqrt(std::max(0.0, det / lmax) / lmax);
}

}  // namespace

TEST_CASE(
    "stretched kernels: stretched_pair_info reports the frozen-metric ridge of "
    "non-coplanar touching pairs",
    "[stretched_kernels]") {
    // Non-coplanar touching pairs (tilted triangle at the PML start, gap 0) are not ridge-split;
    // their ridge_width is the frozen-metric eps at the mid-point of the two centroids, the
    // smaller of the in-plane ridge (sigma_x, sigma_y) and the ridge between the stretched plane
    // and the unstretched z axis (max sigma, 0). Order-2 profile (sigma ~ 0 at the start: eps ~
    // 1) and a step profile of order 0 (sigma = 2 right at the start, the m < 1 case that WP-PML3
    // must exclude with W_g >= h: eps ~ 0.17).
    OperatorOptions opt;
    opt.stretched_touching_points = 4;  // eps does not depend on the points
    const RegionParams reg = all_regions()[0].p;
    for (const Real order : {2.0, 0.0}) {
        const BumpedCase bc(order, order == 0.0 ? 2.0 : 4.0);
        Index n = 0;
        Real eps_min = 1.0;
        for (Index t = 0; t < bc.mesh.num_triangles(); ++t) {
            for (Index s = 0; s < bc.mesh.num_triangles(); ++s) {
                const Proximity cl = kernels::classify(bc.mesh, t, s);
                if (cl != Proximity::shared_edge && cl != Proximity::shared_vertex) {
                    continue;
                }
                const kernels::StretchedPairInfo info =
                    kernels::stretched_pair_info(bc.mesh, t, s, reg.k, opt, bc.st);
                if (!info.stretched || info.coplanar) {
                    continue;
                }
                const Vec3 mid = 0.5 * (bc.mesh.centroid(t) + bc.mesh.centroid(s));
                if (order == 0.0 && std::abs(mid(0) - bc.xp) < 1e-6 * bc.h) {
                    continue;  // mid-point on the step of sigma: rounding decides the side
                }
                const Vec3c jd = bc.st.jacobian_diag(mid);
                const Real sx = -jd(0).imag();
                const Real sy = -jd(1).imag();
                const Real expected = std::min(ridge_eps(sx, sy), ridge_eps(std::max(sx, sy), 0.0));
                INFO("order " << order << ", pair (" << t << ", " << s << "): sigma " << sx
                              << ", ridge width " << info.ridge_width << ", expected " << expected);
                CHECK_FALSE(info.ridge_split);
                CHECK(std::abs(info.ridge_width - expected) <= 1e-12);
                eps_min = std::min(eps_min, info.ridge_width);
                ++n;
            }
        }
        WARN("order " << order << ": " << n << " non-coplanar stretched touching pairs, smallest "
                      << "ridge width " << eps_min);
        CHECK(n > 0);
        CHECK(eps_min < (order == 0.0 ? 0.5 : 1.0 - 1e-6));
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Reciprocity.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: the stretched PMCHWT matrix is complex symmetric",
          "[stretched_kernels]") {
    // Open surface with taper (non-flat triangles touching the PML, gap 0) and a PML of
    // sigma_max = 3: S = diag(I, -I) Z symmetric (stretched blocks are computed for one ordering
    // and transposed, so only the summation order differs).
    OpenCase c(50e-9);
    const CoordinateStretch pml = geometry::make_pml_stretch(c.om, 3.0);
    c.problem.stretch = &pml;
    const MatrixXc Z = dense(c.problem);
    const Index N = c.space.size();
    MatrixXc S = Z;
    S.bottomRows(N) *= -1.0;
    const Real defect = (S - S.transpose()).norm() / S.norm();
    WARN("stretched PMCHWT: |S - S^T| / |S| = " << defect << " (2N = " << 2 * N << ")");
    CHECK(defect < 1e-13);
    // Non-coplanar stretched touching pairs exist (taper triangle at the PML start).
    Index noncoplanar = 0;
    const TriangleMesh& m = c.om.mesh;
    for (Index t = 0; t < m.num_triangles(); ++t) {
        for (Index s = 0; s < m.num_triangles(); ++s) {
            const kernels::StretchedPairInfo info = kernels::stretched_pair_info(
                m, t, s, Complex(1.0, 0.0), c.problem.kernel_options, pml);
            if (info.stretched && !info.coplanar &&
                (info.proximity == Proximity::shared_edge ||
                 info.proximity == Proximity::shared_vertex)) {
                ++noncoplanar;
            }
        }
    }
    INFO("non-coplanar stretched touching pairs: " << noncoplanar);
    CHECK(noncoplanar > 0);
}

TEST_CASE("stretched kernels: dense assembly with a PML against the pair-by-pair reference",
          "[stretched_kernels]") {
    // WP-PML2c: DenseStrategy evaluates every unordered stretched pair once (one ordering, the
    // other by the sub-block transposes) and both regions in one call. Reference: Z assembled
    // pair by pair from the per-region element_blocks(..., &pml, ...) of every ordered pair, in a
    // plain loop. Small open surface, vacuum / Ag, sigma_max 2: at the outer PML vertices
    // |k_Ag| s h = 6.1 (h the cell diagonal) raises the Ag touching points (k-aware extra 1), so
    // both the shared pass and the separate passes of element_blocks_both_regions occur. Release:
    // the default near / far degree selection (the regions pick different degrees for some pairs);
    // debug: fixed degrees. Z_ref is summed from the two-region blocks, which are checked bitwise
    // against the single-region element_blocks pair by pair, so it is the per-region reference.
    // Release checks every ordered pair and every entry of Z. Debug (ASan, -O0; 187 s with the
    // full loops): the per-region check and the rule counts on every 7th ordered pair, and Z_ref
    // only for the test triangles of the lower band (centroid y in the lower 30 % of the mesh:
    // the y-PML strip, two PML corners, parts of the x-PML strips, the taper and the patch), Z
    // compared on the rows whose two support triangles both lie in the band (all columns).
    OpenCase c(50e-9);
    if (kOptimised) {
        c.problem.kernel_options.target_accuracy = OperatorOptions{}.target_accuracy;
        c.problem.kernel_options.quad_degree_near = OperatorOptions{}.quad_degree_near;
    }
    const OperatorOptions& opt = c.problem.kernel_options;
    const CoordinateStretch pml = geometry::make_pml_stretch(c.om, 2.0);
    c.problem.stretch = &pml;
    const RwgSpace& space = c.space;
    const TriangleMesh& m = space.mesh();
    const Index N = space.size();
    const Index F = m.num_triangles();
    const std::array<RegionParams, 2> reg = {region_of(c.problem.exterior),
                                             region_of(c.problem.object)};
    const formulation::Weights wt = c.form->weights(reg[0].eta, reg[1].eta);
    const std::array<Complex, 2> ce = {wt.a1 / reg[0].eta, wt.a2 / reg[1].eta};
    const std::array<Complex, 2> ch = {wt.b1 * reg[0].eta, wt.b2 * reg[1].eta};
    const std::array<Complex, 2> cm = {wt.b1 / reg[0].eta, wt.b2 / reg[1].eta};
    const std::array<Real, 2> jump = {-0.5, 0.5};
    MatrixXc Zref = MatrixXc::Zero(2 * N, 2 * N);
    MatrixXr scale = MatrixXr::Zero(2 * N, 2 * N);  // sum of |contributions| per entry
    Index stretched = 0;
    Index not_bitwise = 0;
    Index shared_rule = 0;
    Index separate_rule = 0;
    Index checked = 0;
    const Index stride = kOptimised ? 1 : 7;
    // Test triangles of Z_ref (debug: the lower band) and the rows of Z compared.
    std::vector<bool> in_band(static_cast<std::size_t>(F), true);
    if (!kOptimised) {
        std::vector<Real> cy(static_cast<std::size_t>(F));
        for (Index t = 0; t < F; ++t) {
            Real y = 0.0;
            for (Index i = 0; i < 3; ++i) {
                y += m.vertices()(m.triangles()(t, i), 1) / 3.0;
            }
            cy[static_cast<std::size_t>(t)] = y;
        }
        const auto [lo, hi] = std::minmax_element(cy.begin(), cy.end());
        const Real y_cut = *lo + 0.3 * (*hi - *lo);
        for (Index t = 0; t < F; ++t) {
            in_band[static_cast<std::size_t>(t)] = cy[static_cast<std::size_t>(t)] < y_cut;
        }
    }
    std::vector<bool> row_done(static_cast<std::size_t>(N));
    Index rows = 0;
    for (Index n = 0; n < N; ++n) {
        row_done[static_cast<std::size_t>(n)] =
            in_band[static_cast<std::size_t>(space.plus_triangle(n))] &&
            in_band[static_cast<std::size_t>(space.minus_triangle(n))];
        rows += row_done[static_cast<std::size_t>(n)] ? 1 : 0;
    }
    for (Index t = 0; t < F; ++t) {
        const RwgSpace::Support st = space.support(t);
        CHECK(kernels::is_stretched_triangle(pml, m, t) ==
              kernels::stretched_pair_info(m, t, t, reg[0].k, opt, pml).stretched);
        if (!in_band[static_cast<std::size_t>(t)]) {
            continue;
        }
        for (Index s = 0; s < F; ++s) {
            const RwgSpace::Support ss = space.support(s);
            std::array<Block, 2> Lb;
            std::array<Block, 2> Kb;
            kernels::element_blocks_both_regions(space, t, s, reg, {true, true}, opt, &pml, Lb, Kb);
            if ((t * F + s) % stride == 0) {
                ++checked;
                const kernels::StretchedPairInfo i0 =
                    kernels::stretched_pair_info(m, t, s, reg[0].k, opt, pml);
                if (i0.stretched) {
                    ++stretched;
                    const kernels::StretchedPairInfo i1 =
                        kernels::stretched_pair_info(m, t, s, reg[1].k, opt, pml);
                    ++(i0.points == i1.points && i0.degree == i1.degree ? shared_rule
                                                                        : separate_rule);
                }
                for (std::size_t i = 0; i < 2; ++i) {
                    const Blocks b = blocks(space, t, s, reg[i], opt, &pml);
                    if (!bitwise_equal(b.L, Lb[i]) || !bitwise_equal(b.K, Kb[i])) {
                        ++not_bitwise;
                    }
                }
            }
            Block I = Block::Zero();
            if (t == s) {
                kernels::jump_block(space, t, I);
            }
            for (std::size_t i = 0; i < 2; ++i) {
                for (int a = 0; a < st.count; ++a) {
                    for (int bb = 0; bb < ss.count; ++bb) {
                        const Complex l = Lb[i](a, bb);
                        const Complex k = Kb[i](a, bb) + jump[i] * I(a, bb);
                        const Index r = st.n[a];
                        const Index q = ss.n[bb];
                        const std::array<std::pair<Index, Index>, 4> at = {
                            {{r, q}, {r, N + q}, {N + r, q}, {N + r, N + q}}};
                        const std::array<Complex, 4> v = {ce[i] * l, -(ce[i] * k), ch[i] * k,
                                                          cm[i] * l};
                        for (std::size_t x = 0; x < 4; ++x) {
                            Zref(at[x].first, at[x].second) += v[x];
                            scale(at[x].first, at[x].second) += std::abs(v[x]);
                        }
                    }
                }
            }
        }
    }
    // One active region only: the other block pair is zero, the active one bitwise.
    {
        const Index t = 0;
        const Index s = F - 1;
        std::array<Block, 2> Lb;
        std::array<Block, 2> Kb;
        kernels::element_blocks_both_regions(space, t, s, reg, {false, true}, opt, &pml, Lb, Kb);
        const Blocks b = blocks(space, t, s, reg[1], opt, &pml);
        CHECK(bitwise_equal(Lb[1], b.L));
        CHECK(bitwise_equal(Kb[1], b.K));
        CHECK(Lb[0].isZero(0.0));
        CHECK(Kb[0].isZero(0.0));
    }
    const MatrixXc Z = dense(c.problem);
    Real worst = 0.0;
    for (Index a = 0; a < 2 * N; ++a) {
        if (!row_done[static_cast<std::size_t>(a % N)]) {
            continue;
        }
        for (Index b = 0; b < 2 * N; ++b) {
            if (scale(a, b) > 0.0) {
                worst = std::max(worst, std::abs(Z(a, b) - Zref(a, b)) / scale(a, b));
            } else {
                CHECK(Z(a, b) == Complex(0.0, 0.0));
            }
        }
    }
    WARN("PML dense vs pair-by-pair reference: "
         << checked << " ordered pairs checked per region, " << stretched << " of them stretched ("
         << shared_rule << " with the same rule in both regions, " << separate_rule
         << " with different rules); " << rows << " of " << N
         << " basis rows compared, max |Z - Z_ref| / sum |contributions| = " << worst);
    CHECK(rows > (kOptimised ? N - 1 : N / 5));
    CHECK(not_bitwise == 0);
    CHECK(shared_rule > 0);
    CHECK(separate_rule > 0);
    CHECK(worst < 1e-13);
    CHECK((op::assemble_diagonal(c.problem).array() == Z.diagonal().array()).all());
    // The stretched part is exactly complex symmetric per sub-block; the matrix up to the
    // unstretched blocks' rounding (as the reciprocity test above).
    MatrixXc S = Z;
    S.bottomRows(N) *= -1.0;
    CHECK((S - S.transpose()).norm() / S.norm() < 1e-13);
}

TEST_CASE("stretched kernels: OpenMP thread count does not change the dense Z with a PML (bitwise)",
          "[stretched_kernels]") {
#ifdef SPECKLEBEM_HAVE_OPENMP
    // WP-PML2c: DenseStrategy with a stretch evaluates every unordered pair once (pass 1, both
    // regions) and scatters the blocks and their transposes (pass 2). The result must not depend
    // on the thread count: 1 against 3 threads, all entries of Z and assemble_diagonal bitwise.
    // Small open surface (vacuum / Ag) with cheap options and low degrees (as the determinism
    // test of test_assembler.cpp: degrees 1 / 2 / 2); h = 20 nm keeps |k_Ag| s h below the
    // k-aware extra. The stretch is the x strip only (sigma_max 2, sigma_y = 0: the anisotropic
    // ridge-split rules, about half the triangles stretched; the debug cost is dominated by the
    // stretched touching pairs). The rule selection is not what is tested.
    OpenCase c(20e-9);
    c.problem.kernel_options.quad_degree_far = 1;
    c.problem.kernel_options.quad_degree_near = 2;
    c.problem.kernel_options.quad_degree_sing = 2;
    const StretchProfile px{c.om.pml_start, c.om.outer_half - c.om.pml_start, 2.0};
    StretchProfile py = px;
    py.sigma_max = 0.0;
    const CoordinateStretch pml(px, py);
    c.problem.stretch = &pml;
    // assemble_diagonal runs at 3 threads only and is compared with the diagonal of the 1-thread
    // Z (a 1-thread diagonal would add a third of the debug time; the pair-by-pair test above
    // checks assemble_diagonal against Z at the default thread count).
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const MatrixXc Z1 = dense(c.problem);
    omp_set_num_threads(3);
    const MatrixXc Z3 = dense(c.problem);
    const VectorXc d3 = op::assemble_diagonal(c.problem);
    omp_set_num_threads(saved);
    REQUIRE(Z1.rows() == Z3.rows());
    CHECK((Z1.array() == Z3.array()).all());
    CHECK((d3.array() == Z1.diagonal().array()).all());
    // The case has stretched pairs (otherwise the test would not probe the PML passes).
    Index n_stretched = 0;
    for (Index t = 0; t < c.space.mesh().num_triangles(); ++t) {
        n_stretched += kernels::is_stretched_triangle(pml, c.space.mesh(), t) ? 1 : 0;
    }
    CHECK(n_stretched > 0);
    CHECK(n_stretched < c.space.mesh().num_triangles());
#else
    SUCCEED("built without OpenMP: assembly is sequential");
#endif
}

TEST_CASE("stretched kernels: the fastmath kernel loop against the C library",
          "[stretched_kernels]") {
    // WP-PML2c: with fast_plain_kernel (default) stretched pairs with K = 0 evaluate exp(-j k R~)
    // with kernels::fastmath; pairs with K != 0 keep the C library (bitwise the arithmetic with
    // fast_plain_kernel = false). Every 3rd ordered stretched pair of the small open surface
    // (release; debug every 23rd with the cheap options), vacuum and Ag, sigma_max 2.
    OpenCase c(50e-9);
    if (kOptimised) {
        c.problem.kernel_options = OperatorOptions{};
    }
    OperatorOptions fast = c.problem.kernel_options;
    fast.fast_plain_kernel = true;
    OperatorOptions lib = fast;
    lib.fast_plain_kernel = false;
    const CoordinateStretch pml = geometry::make_pml_stretch(c.om, 2.0);
    const TriangleMesh& m = c.space.mesh();
    const Index F = m.num_triangles();
    const Index stride = kOptimised ? 3 : 23;
    const std::array<RegionParams, 2> reg = {region_of(c.problem.exterior),
                                             region_of(c.problem.object)};
    Index k_free = 0;
    Index with_k = 0;
    Index not_bitwise = 0;
    Real worst = 0.0;
    for (Index ip = 0; ip < F * F; ip += stride) {
        const Index t = ip / F;
        const Index s = ip % F;
        if (!kernels::is_stretched_triangle(pml, m, t) &&
            !kernels::is_stretched_triangle(pml, m, s)) {
            continue;
        }
        for (const RegionParams& r : reg) {
            const Blocks bf = blocks(c.space, t, s, r, fast, &pml);
            const Blocks bl = blocks(c.space, t, s, r, lib, &pml);
            if (bl.K == Block::Zero()) {
                ++k_free;
                CHECK(bf.K == Block::Zero());
                worst = std::max(worst, rel_diff(bf.L, bl.L));
            } else {
                ++with_k;
                if (!bitwise_equal(bf.L, bl.L) || !bitwise_equal(bf.K, bl.K)) {
                    ++not_bitwise;
                }
            }
        }
    }
    WARN("stretched fastmath vs C library: " << k_free << " K-free blocks, max relative L "
                                             << "difference " << worst << "; " << with_k
                                             << " blocks with K != 0");
    CHECK(k_free > 0);
    CHECK(with_k > 0);
    CHECK(not_bitwise == 0);
    CHECK(worst < 5e-13);  // measured 7e-14 (800-triangle surface, every 7th pair)

    // The fastmath choice is made per region in a shared two-region pass: a region beyond the
    // fastmath bound (|k| r_max > kFastSincosMax, here |k| = 1e14 / m) next to vacuum, on far
    // pairs of flat stretched triangles (K = 0) with fixed degrees (both regions share the rule),
    // gives each region its single-region blocks bitwise (vacuum fastmath, the other the C
    // library).
    RegionParams beyond = reg[0];
    beyond.k = Complex(1e14, -1e3);
    const std::array<RegionParams, 2> mixed = {reg[0], beyond};
    const OperatorOptions fixed = cheap_options();
    Index mixed_pairs = 0;
    Index fast_differs = 0;
    for (Index t = 0; t < F && mixed_pairs < 3; ++t) {
        for (Index s = t + 1; s < F && mixed_pairs < 3; ++s) {
            const kernels::StretchedPairInfo i0 =
                kernels::stretched_pair_info(m, t, s, mixed[0].k, fixed, pml);
            const kernels::StretchedPairInfo i1 =
                kernels::stretched_pair_info(m, t, s, mixed[1].k, fixed, pml);
            if (!i0.stretched || i0.proximity != Proximity::far || i0.degree != i1.degree ||
                !kernels::is_stretched_triangle(pml, m, t) ||
                !kernels::is_stretched_triangle(pml, m, s)) {
                continue;
            }
            std::array<Block, 2> Lb;
            std::array<Block, 2> Kb;
            kernels::element_blocks_both_regions(c.space, t, s, mixed, {true, true}, fixed, &pml,
                                                 Lb, Kb);
            for (std::size_t i = 0; i < 2; ++i) {
                const Blocks b = blocks(c.space, t, s, mixed[i], fixed, &pml);
                CHECK(bitwise_equal(b.L, Lb[i]));
                CHECK(bitwise_equal(b.K, Kb[i]));
                CHECK(b.K == Block::Zero());
            }
            // Sensitivity: the joint C-library choice would change the vacuum blocks.
            OperatorOptions fixed_lib = fixed;
            fixed_lib.fast_plain_kernel = false;
            if (!bitwise_equal(blocks(c.space, t, s, mixed[0], fixed_lib, &pml).L, Lb[0])) {
                ++fast_differs;
            }
            ++mixed_pairs;
        }
    }
    CHECK(mixed_pairs == 3);
    CHECK(fast_differs > 0);
}

// ---------------------------------------------------------------------------------------------
// 4. Decay into the PML.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: coupling into the PML decays like the stretched outgoing wave",
          "[stretched_kernels]") {
    // Flat strip along x (two cells wide), PML on x > x_p only (order 2, sigma_max 4, W = 1.5
    // lambda). Source: the RWG function at the strip centre (outside the PML); test functions
    // along the axis. The vacuum L entries of the column, |L~_mn| / |L_mn|, follow the damping of
    // the outgoing wave exp(-k Int_{x_p}^{x_m} sigma) up to the Piola factor |s_x(x_m)| <= 4.2 and
    // |R / R~| <= sqrt(1 + sigma^2)^(1/2) (bounds below), down to ~1e-8 at the PML end; without
    // the stretch the entries decay like 1 / R.
    const Real h = kLambda / 10.0;
    const Real xp = 0.5e-6;
    const Real W = 1.5 * kLambda;
    const Index nx = static_cast<Index>(std::lround((xp + W + 0.5e-6) / h));
    const TriangleMesh mesh = flat_grid(nx, 2, h, -0.5e-6, -h);
    const RwgSpace space(mesh);
    const CoordinateStretch st = stretch_xy(xp, W, 4.0, 2.0);
    const RegionParams reg = all_regions()[0].p;
    const Real k0 = reg.k.real();
    OperatorOptions opt;
    // Basis functions on the vertical edges of the upper row (edge from (x, 0) to (x, h): current
    // along x), sorted by x; the source is the one at x = 0.
    std::vector<std::pair<Real, Index>> axis;
    for (Index n = 0; n < space.size(); ++n) {
        const Index tp = space.plus_triangle(n);
        std::vector<Vec3> ev;
        for (Index i = 0; i < 3; ++i) {
            const Index v = mesh.triangles()(tp, i);
            if (v != space.plus_free_vertex(n)) {
                ev.push_back(mesh.vertices().row(v).transpose());
            }
        }
        const Vec3 mid = 0.5 * (ev[0] + ev[1]);
        if (std::abs(ev[0](0) - ev[1](0)) < 1e-6 * h && mid(1) > 0.0) {
            axis.emplace_back(mid(0), n);
        }
    }
    std::sort(axis.begin(), axis.end());
    REQUIRE(axis.size() > 20);
    Index n0 = axis[0].second;
    Real best = 1e300;
    for (const auto& [x, n] : axis) {
        if (std::abs(x) < best) {
            best = std::abs(x);
            n0 = n;
        }
    }
    const auto entry = [&](Index m, Index n, const CoordinateStretch* s) {
        Complex sum(0.0, 0.0);
        for (const Index t : {space.plus_triangle(m), space.minus_triangle(m)}) {
            const RwgSpace::Support st_ = space.support(t);
            int a = 0;
            while (st_.n[a] != m) {
                ++a;
            }
            for (const Index u : {space.plus_triangle(n), space.minus_triangle(n)}) {
                const RwgSpace::Support su = space.support(u);
                int b = 0;
                while (su.n[b] != n) {
                    ++b;
                }
                const Blocks bl = blocks(space, t, u, reg, opt, s);
                sum += bl.L(a, b);
            }
        }
        return sum;
    };
    Index checked = 0;
    Real last_ratio = 2.0;
    Real first_plain = 0.0;
    Real last_plain = 0.0;
    Real last_damping = 1.0;
    for (const auto& [x, m] : axis) {
        if (x <= xp + 0.5 * h || x > xp + W - 0.5 * h) {
            continue;
        }
        const Real damping = std::exp(-k0 * geometry::detail::stretch_integral(*st.profile_x(), x));
        const Complex plain = entry(m, n0, nullptr);
        const Real ratio = std::abs(entry(m, n0, &st)) / std::abs(plain);
        const Real sigma = geometry::detail::stretch_sigma(*st.profile_x(), x);
        const Real piola = std::sqrt(1.0 + sigma * sigma);
        INFO("x = " << x << ": |L~| / |L| = " << ratio << ", exp(-k Int sigma) = " << damping);
        // Measured: ratio / damping = 1.0 .. 2.3 (WP-PML2).
        CHECK(ratio >= 0.8 * damping);
        CHECK(ratio <= 1.2 * piola * damping);
        CHECK(ratio < last_ratio * 1.05);  // monotone decay
        last_ratio = ratio;
        if (checked == 0) {
            first_plain = std::abs(plain);
        }
        last_plain = std::abs(plain);
        last_damping = damping;
        ++checked;
    }
    WARN("decay test: " << checked << " test functions in the PML; deepest: |L~| / |L| = "
                        << last_ratio << ", exp(-k Int sigma) = " << last_damping
                        << "; unstretched entries decay by " << last_plain / first_plain);
    CHECK(checked >= 12);
    CHECK(last_ratio < 1e-3);
    CHECK(last_plain / first_plain > 0.1);  // without the stretch only ~1 / R
}

// ---------------------------------------------------------------------------------------------
// 5. Refusals, options, guards, info.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels: paths without stretch support refuse a PML stretch",
          "[stretched_kernels]") {
    OpenCase c(50e-9);
    const CoordinateStretch pml = geometry::make_pml_stretch(c.om, 3.0);
    const excitation::PlaneWave pw(kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0));
    c.problem.excitation = &pw;
    const Index N = c.space.size();
    op::BasisPattern pat;
    pat.group_of_row.assign(static_cast<std::size_t>(N), 0);
    pat.col_ptr = {0, 1};
    pat.cols = {0};
    // Without stretch (and with the identity) everything runs.
    const CoordinateStretch identity = CoordinateStretch::identity();
    for (const CoordinateStretch* s : {static_cast<const CoordinateStretch*>(nullptr), &identity}) {
        c.problem.stretch = s;
        CHECK_FALSE(op::has_stretch(c.problem));
        CHECK_NOTHROW(op::assemble_sparse(c.problem, pat));
        CHECK_NOTHROW(op::assemble_rhs(c.problem));
    }
    c.problem.stretch = &pml;
    CHECK(op::has_stretch(c.problem));
    CHECK_THROWS_AS(op::assemble_sparse(c.problem, pat), std::invalid_argument);
    std::vector<Index> row_ptr(static_cast<std::size_t>(N) + 1, 0);
    CHECK_THROWS_AS(op::assemble_region_sparse(c.problem, 0, row_ptr, {}), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::MlfmmStrategy().build(c.problem), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::MlfmmOperator(c.problem, mlfmm::MlfmmParams{}), std::invalid_argument);
    const post::SurfaceSolution sol{&c.problem, VectorXc::Zero(2 * N)};
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    Vertices dirs(1, 3);
    dirs << 0.0, 0.0, -1.0;
    CHECK_THROWS_AS(post::far_field(sol, dirs, F), std::invalid_argument);
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> E;
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> H;
    Vertices pts(1, 3);
    pts << 0.0, 0.0, -1e-6;
    CHECK_THROWS_AS(post::scattered_field(sol, pts, E, H), std::invalid_argument);
    // The dense paths accept it, and (WP-PML3a) the masked right-hand side; far_field needs a
    // current window (above: none given).
    CHECK_NOTHROW(op::assemble_diagonal(c.problem));
    CHECK_NOTHROW(op::assemble_rhs(c.problem));
}

TEST_CASE("stretched kernels: options, horizontal guard and rule information",
          "[stretched_kernels]") {
    const Real h = 40e-9;
    const TriangleMesh mesh = flat_grid(3, 3, h, 0.0, 0.0);
    const RwgSpace space(mesh);
    const RegionParams reg = all_regions()[0].p;
    const CoordinateStretch st = stretch_xy(1.0 * h, 2.0 * h, 4.0, 2.0);
    OperatorOptions opt;
    for (const int p : {3, 17}) {
        OperatorOptions bad = opt;
        bad.stretched_touching_points = p;
        CHECK_THROWS_AS(kernels::validate(bad), std::invalid_argument);
    }
    // Unstretched pair (x <= start): not on the stretched path.
    const kernels::StretchedPairInfo un = kernels::stretched_pair_info(mesh, 0, 0, reg.k, opt, st);
    CHECK_FALSE(un.stretched);
    // Stretched identical pair on the x strip: ridge split, coplanar.
    const Index tc = 17;  // cell (2, 1): x in [2h, 3h]
    const kernels::StretchedPairInfo id =
        kernels::stretched_pair_info(mesh, tc, tc, reg.k, opt, st);
    CHECK(id.stretched);
    CHECK(id.coplanar);
    CHECK(id.ridge_split);
    CHECK(id.ridge_width < 0.9);
    CHECK(id.points > 0);
    CHECK_THROWS_AS(kernels::stretched_pair_info(mesh, -1, 0, reg.k, opt, st), std::out_of_range);
    Block L;
    Block K;
    CHECK_THROWS_AS(kernels::element_blocks(space, 0, 99, reg, opt, &st, L, K), std::out_of_range);
    // A tilted triangle in the stretched zone is rejected.
    Vertices v(4, 3);
    v << 2.0 * h, 0.0, 0.0, 3.0 * h, 0.0, 0.2 * h, 3.0 * h, h, 0.0, 2.0 * h, h, 0.0;
    Triangles f(2, 3);
    f << 0, 1, 2, 0, 2, 3;
    const TriangleMesh tilted(v, f);
    const RwgSpace ts(tilted);
    CHECK_THROWS_AS(kernels::element_blocks(ts, 0, 1, reg, opt, &st, L, K), std::invalid_argument);
    CHECK_THROWS_AS(kernels::stretched_pair_info(tilted, 0, 0, reg.k, opt, st),
                    std::invalid_argument);
    // Without the stretch the tilted mesh is fine.
    CHECK_NOTHROW(kernels::element_blocks(ts, 0, 1, reg, opt, nullptr, L, K));
}

// ---------------------------------------------------------------------------------------------
// Cost study (hidden; release): dense assembly of an open surface with and without the PML.
// ---------------------------------------------------------------------------------------------

TEST_CASE("stretched kernels (full): dense assembly cost with a PML",
          "[stretched_kernels][.slow][stretched_cost]") {
    if (!kOptimised) {
        SKIP("optimised builds only");
    }
    // Open surface at h = lambda / 10: patch 0.5 um, taper 0.1 um, gap 0.1 um, PML 0.5 um (one
    // lambda), sigma_max 4, order 2; vacuum / Ag, PMCHWT, default kernel options. Reports the
    // assembly time without and with the stretch and the kernel evaluations per pair class.
    const Real h = 50e-9;
    geometry::RoughSurfaceParams rp;
    rp.mesh_size = h;
    rp.edge_length_L = 0.7e-6;
    rp.rms_roughness = 5e-9;
    rp.correlation_length = 100e-9;
    rp.seed = 5;
    const geometry::HeightMap map = geometry::generate_gaussian_height_map(rp);
    const geometry::OpenSurfaceMesh om =
        geometry::make_open_surface_mesh(map, {0.5e-6, 0.1e-6, 0.1e-6, 0.5e-6});
    const RwgSpace space(om.mesh);
    const auto form = formulation::make_formulation(formulation::Kind::PMCHWT);
    op::Problem p;
    p.space = &space;
    p.exterior = material::vacuum();
    p.object = material::silver_500nm();
    p.formulation = form.get();
    p.omega = 2.0 * kPi * constants::c0 / kLambda;
    const CoordinateStretch pml = geometry::make_pml_stretch(om, 4.0);
    const auto seconds = [&](const CoordinateStretch* s) {
        p.stretch = s;
        const auto t0 = std::chrono::steady_clock::now();
        const MatrixXc Z = dense(p);
        return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    };
    const Real t_plain = seconds(nullptr);
    const Real t_pml = seconds(&pml);
    // Kernel evaluations per class (region R1, one ordering per unordered pair); near / far
    // pairs at the degree cap in both regions (k_1 vacuum, k_2 Ag).
    std::array<Real, 5> points{};
    std::array<Real, 5> evals{};
    std::array<Real, 5> geometry{};
    Real evals_before = 0.0;
    std::array<Index, 5> pairs{};
    Index capped = 0;
    Index capped2 = 0;
    const TriangleMesh& m = om.mesh;
    const Complex k1 = p.exterior.wavenumber(p.omega);
    const Complex k2 = p.object.wavenumber(p.omega);
    for (Index t = 0; t < m.num_triangles(); ++t) {
        for (Index s = t; s < m.num_triangles(); ++s) {
            const kernels::StretchedPairInfo info =
                kernels::stretched_pair_info(m, t, s, k1, p.kernel_options, pml);
            if (!info.stretched) {
                continue;
            }
            const auto c = static_cast<std::size_t>(info.proximity);
            points[c] += static_cast<Real>(info.points);
            ++pairs[c];
            if (!info.target_reached) {
                ++capped;
            }
            const kernels::StretchedPairInfo info2 =
                kernels::stretched_pair_info(m, t, s, k2, p.kernel_options, pml);
            if ((info.proximity == Proximity::near || info.proximity == Proximity::far) &&
                !info2.target_reached) {
                ++capped2;
            }
            // Kernel evaluations (point pair x region) of the dense assembly: WP-PML2 evaluated
            // both orderings of t != s per region; WP-PML2c each unordered pair once, both
            // regions in one pass (one geometry evaluation per point pair) if their rules agree.
            const auto p1 = static_cast<Real>(info.points);
            const auto p2 = static_cast<Real>(info2.points);
            const bool same = info.points == info2.points && info.degree == info2.degree;
            evals[c] += p1 + p2;
            geometry[c] += same ? p1 : p1 + p2;
            evals_before += (t == s ? 1.0 : 2.0) * (p1 + p2);
        }
    }
    std::ostringstream os;
    Real evals_after = 0.0;
    Real geometry_after = 0.0;
    for (std::size_t c = 0; c < 5; ++c) {
        os << " " << class_name(static_cast<Proximity>(c)) << ": " << pairs[c] << " pairs, "
           << (pairs[c] > 0 ? points[c] / static_cast<Real>(pairs[c]) : 0.0)
           << " points/pair (k_1), " << evals[c] << " kernel evaluations, " << geometry[c]
           << " point-pair geometries (both regions);";
        evals_after += evals[c];
        geometry_after += geometry[c];
    }
    WARN("PML cost: " << m.num_triangles() << " triangles, 2N = " << 2 * space.size()
                      << ", assembly " << t_plain << " s without, " << t_pml
                      << " s with the stretch; stretched" << os.str()
                      << " near/far at the cap: " << capped << " (k_1), " << capped2
                      << " (k_2); kernel evaluations " << evals_before << " (WP-PML2, both "
                      << "orderings) -> " << evals_after << " (" << geometry_after
                      << " point-pair geometries)");
}
