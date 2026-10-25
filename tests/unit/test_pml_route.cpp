/// @file test_pml_route.cpp
/// WP-PML3a (ADR 0009 decisions 4 and 5): masked right-hand side in the PML (op::assemble_rhs,
/// op::pml_masked_functions), windowed far field (post::CurrentWindow, post::far_field) and the
/// PML constructor of Simulation (checks, default window, a tiny dense solve).
#include "specklebem/basis/rwg.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/simulation.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace specklebem;
using basis::RwgSpace;
using Catch::Matchers::ContainsSubstring;
using geometry::CoordinateStretch;
using geometry::OpenSurfaceMesh;
using geometry::SurfaceZone;

namespace {

constexpr Real kLambda = 500e-9;
using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

/// Release builds only (stretched dense assembly is minutes under the sanitizers at -O0).
constexpr bool kRelease =
#ifdef NDEBUG
    true;
#else
    false;
#endif

/// Flat (sigma = 0) height map of edge `edge` with spacing h (edge / h integer).
geometry::HeightMap flat_map(Real edge, Real h) {
    const auto cells = static_cast<Index>(std::llround(edge / h));
    geometry::HeightMap m;
    m.dx = edge / static_cast<Real>(cells);
    m.dy = m.dx;
    m.z = MatrixXr::Zero(cells + 1, cells + 1);
    return m;
}

/// Open surface with the given band widths (multiples of h); flat unless sigma > 0 (Gaussian
/// map, correlation length lc, fixed seed).
OpenSurfaceMesh open_surface(Real h, Real patch, Real taper, Real gap, Real pml, Real sigma = 0.0,
                             Real lc = 0.0) {
    geometry::OpenSurfaceParams op;
    op.patch_size = patch;
    op.taper_width = taper;
    op.gap_width = gap;
    op.pml_width = pml;
    if (sigma == 0.0)
        return geometry::make_open_surface_mesh(flat_map(patch + 2.0 * taper, h), op);
    geometry::RoughSurfaceParams rp;
    rp.mesh_size = h;
    rp.edge_length_L = patch + 2.0 * taper;
    rp.rms_roughness = sigma;
    rp.correlation_length = lc;
    rp.seed = 17;
    return geometry::make_open_surface_mesh(geometry::generate_gaussian_height_map(rp), op);
}

/// Open-surface problem (no excitation set) with its own space and formulation.
struct OpenProblem {
    OpenSurfaceMesh om;
    RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    CoordinateStretch pml;
    op::Problem problem;
    explicit OpenProblem(OpenSurfaceMesh m, Real sigma_max = 4.0)
        : om(std::move(m)),
          space(om.mesh),
          form(formulation::make_formulation(formulation::Kind::PMCHWT)),
          pml(geometry::make_pml_stretch(om, sigma_max)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = material::silicon_500nm();
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
    }
};

/// Patch 2 h, taper h, gap h, PML 2 h at h = 50 nm: 10 x 10 cells, 200 triangles, rough patch.
OpenSurfaceMesh rhs_surface() {
    const Real h = 50e-9;
    return open_surface(h, 2.0 * h, h, h, 2.0 * h, 0.1 * h, 2.0 * h);
}

/// Independent mask: a support triangle with a vertex strictly beyond the PML start.
std::vector<bool> reference_mask(const RwgSpace& space, Real start) {
    const geometry::TriangleMesh& m = space.mesh();
    std::vector<bool> out(static_cast<std::size_t>(space.size()), false);
    for (Index n = 0; n < space.size(); ++n) {
        for (const Index t : {space.plus_triangle(n), space.minus_triangle(n)}) {
            for (Index c = 0; c < 3; ++c) {
                const Vec3 v = m.vertices().row(m.triangles()(t, c)).transpose();
                if (std::abs(v.x()) > start || std::abs(v.y()) > start)
                    out[static_cast<std::size_t>(n)] = true;
            }
        }
    }
    return out;
}

bool bitwise_equal(Complex a, Complex b) {
    return a.real() == b.real() && a.imag() == b.imag();
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Masked right-hand side
// ---------------------------------------------------------------------------------------------

TEST_CASE("pml rhs: no stretch, identity and sigma_max = 0 leave the rhs bitwise unchanged",
          "[pml_rhs]") {
    OpenProblem c(rhs_surface());
    const excitation::PlaneWave pw(
        kLambda, Vec3(0.3, 0.0, 1.0).normalized(),
        Vec3c(Complex(1.0, 0.2), Complex(0.0, 0.5), Complex(-0.3, -0.06)));
    c.problem.excitation = &pw;
    const VectorXc b0 = op::assemble_rhs(c.problem);
    REQUIRE(b0.size() == 2 * c.space.size());
    const CoordinateStretch identity = CoordinateStretch::identity();
    const CoordinateStretch zero = geometry::make_pml_stretch(c.om, 0.0);
    REQUIRE(zero.is_identity());
    for (const CoordinateStretch* s : {&identity, &zero}) {
        c.problem.stretch = s;
        const VectorXc b = op::assemble_rhs(c.problem);
        CHECK((b.array() == b0.array()).all());
    }
    const std::vector<std::uint8_t> none = op::pml_masked_functions(c.space, identity);
    CHECK(static_cast<Index>(none.size()) == c.space.size());
    CHECK(std::all_of(none.begin(), none.end(), [](std::uint8_t m) { return m == 0; }));
}

TEST_CASE("pml rhs: masked entries are exactly zero, the others bitwise the unstretched rhs",
          "[pml_rhs]") {
    OpenProblem c(rhs_surface());
    const excitation::PlaneWave pw(kLambda, Vec3(0.0, 0.2, 1.0).normalized(),
                                   Vec3c(1.0, Complex(0.0, 0.5), Complex(0.0, -0.1)));
    c.problem.excitation = &pw;
    const VectorXc b0 = op::assemble_rhs(c.problem);
    c.problem.stretch = &c.pml;
    const VectorXc b = op::assemble_rhs(c.problem);
    const Index N = c.space.size();
    const std::vector<std::uint8_t> mask = op::pml_masked_functions(c.space, c.pml);
    const std::vector<bool> ref = reference_mask(c.space, c.om.pml_start);
    REQUIRE(static_cast<Index>(mask.size()) == N);
    Index masked = 0;
    Index unmasked_nonzero = 0;
    for (Index n = 0; n < N; ++n) {
        const auto i = static_cast<std::size_t>(n);
        CHECK((mask[i] != 0) == ref[i]);
        if (mask[i] != 0) {
            ++masked;
            CHECK(bitwise_equal(b(n), Complex(0.0, 0.0)));
            CHECK(bitwise_equal(b(N + n), Complex(0.0, 0.0)));
        } else {
            CHECK(bitwise_equal(b(n), b0(n)));
            CHECK(bitwise_equal(b(N + n), b0(N + n)));
            if (b0(n) != Complex(0.0, 0.0))
                ++unmasked_nonzero;
        }
    }
    INFO("masked " << masked << " of N = " << N);
    CHECK(masked > 0);
    CHECK(masked < N);
    CHECK(unmasked_nonzero > 0);
    CHECK(b0.cwiseAbs().maxCoeff() > 0.0);
}

TEST_CASE("pml rhs: a function straddling the Gap/Pml line is masked, one touching it is not",
          "[pml_rhs]") {
    OpenProblem c(rhs_surface());
    const geometry::TriangleMesh& m = c.om.mesh;
    const Real a2 = c.om.pml_start;
    const std::vector<std::uint8_t> mask = op::pml_masked_functions(c.space, c.pml);
    Index straddling = 0;
    Index touching = 0;
    for (Index n = 0; n < c.space.size(); ++n) {
        const SurfaceZone zp = c.om.zone[static_cast<std::size_t>(c.space.plus_triangle(n))];
        const SurfaceZone zm = c.om.zone[static_cast<std::size_t>(c.space.minus_triangle(n))];
        const bool gap_pml = (zp == SurfaceZone::Gap && zm == SurfaceZone::Pml) ||
                             (zp == SurfaceZone::Pml && zm == SurfaceZone::Gap);
        if (gap_pml) {
            ++straddling;
            CHECK(mask[static_cast<std::size_t>(n)] == 1);
        }
        if (zp == SurfaceZone::Gap && zm == SurfaceZone::Gap) {
            // Both triangles in the gap; a vertex on the start line, none beyond it.
            bool on_line = false;
            for (const Index t : {c.space.plus_triangle(n), c.space.minus_triangle(n)}) {
                for (Index k = 0; k < 3; ++k) {
                    const Vec3 v = m.vertices().row(m.triangles()(t, k)).transpose();
                    const Real tv = std::max(std::abs(v.x()), std::abs(v.y()));
                    on_line = on_line || std::abs(tv - a2) < 1e-9 * a2;
                }
            }
            if (on_line) {
                ++touching;
                CHECK(mask[static_cast<std::size_t>(n)] == 0);
            }
        }
    }
    CHECK(straddling > 0);
    CHECK(touching > 0);
}

// ---------------------------------------------------------------------------------------------
// Windowed far field
// ---------------------------------------------------------------------------------------------

TEST_CASE("pml far field: raised-cosine current window values, C1 continuity and checks",
          "[pml_far_field]") {
    const post::CurrentWindow w{1.0, 3.0};
    CHECK(post::current_window_value(w, 0.0) == 1.0);
    CHECK(post::current_window_value(w, 1.0) == 1.0);
    CHECK(std::abs(post::current_window_value(w, 2.0) - 0.5) < 1e-15);
    CHECK(std::abs(post::current_window_value(w, -2.0) - 0.5) < 1e-15);
    CHECK(post::current_window_value(w, 3.0) == 0.0);
    CHECK(post::current_window_value(w, 7.0) == 0.0);
    CHECK(std::abs(post::current_window_value(w, 1.5) -
                   0.5 * (1.0 + std::cos(0.25 * constants::pi))) < 1e-15);
    // Monotone non-increasing.
    Real prev = 1.0;
    for (int i = 0; i <= 400; ++i) {
        const Real v = post::current_window_value(w, 0.01 * i);
        CHECK(v <= prev);
        prev = v;
    }
    // C^1 at both ends: value and one-sided slopes match (slope 0 on both sides), the slope
    // vanishes like (pi / 2)^2 d / width^2 inside.
    for (const Real end : {1.0, 3.0}) {
        for (const Real d : {1e-3, 1e-4}) {
            const Real v0 = post::current_window_value(w, end);
            const Real slope_in =
                (post::current_window_value(w, end + (end == 1.0 ? d : -d)) - v0) / d;
            CHECK(std::abs(slope_in) <= 0.7 * d);  // pi^2 d / 16 = 0.62 d for width 2
        }
    }
    CHECK_THROWS_AS(post::current_window_value({-1.0, 1.0}, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(post::current_window_value({1.0, 1.0}, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(post::current_window_value({2.0, 1.0}, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(post::current_window_value({0.0, std::nan("")}, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(post::current_window_value(w, std::nan("")), std::invalid_argument);
}

namespace {

/// Random currents (fixed seed) on a space.
VectorXc random_currents(Index N, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    VectorXc x(2 * N);
    for (Index i = 0; i < 2 * N; ++i) x(i) = Complex(u(gen), u(gen));
    return x;
}

Vertices some_directions() {
    Vertices d(5, 3);
    d << 0.0, 0.0, -1.0, 0.3, 0.1, -0.9, -0.5, 0.4, -0.2, 0.2, -0.7, 0.4, 1.0, 0.0, 0.0;
    return d;
}

/// Independent windowed radiation integral (same Dunavant rule, per-point weights).
/// noinline: GCC 16 -O3 reports a false -Wnull-dereference in the test body when inlined.
[[gnu::noinline]] FieldMatrix reference_far_field(const RwgSpace& space, const VectorXc& x,
                                                  Real omega, const material::Material& ext,
                                                  const Vertices& dirs,
                                                  const post::CurrentWindow& w, int degree) {
    const geometry::TriangleMesh& m = space.mesh();
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const Complex k = ext.wavenumber(omega);
    const Complex eta = ext.wave_impedance(omega);
    const Index N = space.size();
    FieldMatrix F = FieldMatrix::Zero(dirs.rows(), 3);
    for (Index d = 0; d < dirs.rows(); ++d) {
        const Vec3 kh = dirs.row(d).transpose().normalized();
        Vec3c Jt = Vec3c::Zero();
        Vec3c Mt = Vec3c::Zero();
        for (Index t = 0; t < m.num_triangles(); ++t) {
            const Vec3 v0 = m.vertices().row(m.triangles()(t, 0)).transpose();
            const Vec3 v1 = m.vertices().row(m.triangles()(t, 1)).transpose();
            const Vec3 v2 = m.vertices().row(m.triangles()(t, 2)).transpose();
            const RwgSpace::Support sup = space.support(t);
            for (std::size_t q = 0; q < rule.weights.size(); ++q) {
                const Vec3& l = rule.barycentric[q];
                const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
                const Real wq =
                    rule.weights[q] * m.area(t) *
                    post::current_window_value(w, std::max(std::abs(r.x()), std::abs(r.y())));
                const Complex ph = std::exp(Complex(0.0, 1.0) * k * kh.dot(r)) * wq;
                for (int a = 0; a < sup.count; ++a) {
                    const Vec3 f = space.value(sup.n[a], t, r);
                    Jt += ph * x(sup.n[a]) * f.cast<Complex>();
                    Mt += ph * x(N + sup.n[a]) * f.cast<Complex>();
                }
            }
        }
        const auto cross = [](const Vec3& a, const Vec3c& b) {
            return Vec3c(a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(),
                         a.x() * b.y() - a.y() * b.x());
        };
        const Vec3c inner = eta * cross(kh, Jt) + Mt;
        F.row(d) = (Complex(0.0, 1.0) * k / (4.0 * constants::pi) * cross(kh, inner)).transpose();
    }
    return F;
}

}  // namespace

TEST_CASE("pml far field: a window equal to 1 everywhere leaves the far field bitwise unchanged",
          "[pml_far_field]") {
    // Closed mesh (icosphere) and open surface without stretch.
    const geometry::TriangleMesh sphere = geometry::make_icosphere(0.4e-6, 1);
    const RwgSpace ss(sphere);
    const std::unique_ptr<formulation::Formulation> form =
        formulation::make_formulation(formulation::Kind::PMCHWT);
    op::Problem p;
    p.space = &ss;
    p.exterior = material::vacuum();
    p.object = material::silicon_500nm();
    p.formulation = form.get();
    p.omega = 2.0 * constants::pi * constants::c0 / kLambda;
    const post::SurfaceSolution s{&p, random_currents(ss.size(), 5)};
    const Vertices dirs = some_directions();
    FieldMatrix F0, F1, F2;
    post::far_field(s, dirs, F0);
    post::FieldOptions fo;
    post::far_field(s, dirs, F1, fo);  // explicit, no window
    fo.window = post::CurrentWindow{1e-6, 2e-6};
    post::far_field(s, dirs, F2, fo);
    CHECK((F0.array() == F1.array()).all());
    CHECK((F0.array() == F2.array()).all());

    OpenProblem c(rhs_surface());
    const post::SurfaceSolution so{&c.problem, random_currents(c.space.size(), 6)};
    post::far_field(so, dirs, F0);
    fo.window = post::CurrentWindow{c.om.outer_half, 2.0 * c.om.outer_half};
    post::far_field(so, dirs, F1, fo);
    CHECK((F0.array() == F1.array()).all());
}

TEST_CASE("pml far field: windowed far field against an independent radiation integral",
          "[pml_far_field]") {
    OpenProblem c(rhs_surface());
    const post::SurfaceSolution s{&c.problem, random_currents(c.space.size(), 7)};
    const Vertices dirs = some_directions();
    const post::CurrentWindow w{c.om.taper_end, c.om.pml_start};
    post::FieldOptions fo;
    fo.quad_degree = 4;
    fo.window = w;
    FieldMatrix F;
    post::far_field(s, dirs, F, fo);
    const FieldMatrix R =
        reference_far_field(c.space, s.currents, c.problem.omega, c.problem.exterior, dirs, w, 4);
    const Real err = (F - R).norm() / R.norm();
    INFO("relative difference " << err);
    CHECK(err < 1e-12);
    // The window changes the result (the gap and PML currents are cut).
    FieldMatrix F0;
    post::FieldOptions plain;
    plain.quad_degree = 4;
    post::far_field(s, dirs, F0, plain);
    CHECK((F0 - F).norm() > 1e-3 * F0.norm());
    // With the stretch the windowed far field is the same (no PML current contributes).
    c.problem.stretch = &c.pml;
    FieldMatrix Fs;
    post::far_field(s, dirs, Fs, fo);
    CHECK((Fs.array() == F.array()).all());
}

TEST_CASE("pml far field: stretch and window rules of far_field and the near fields",
          "[pml_far_field]") {
    OpenProblem c(rhs_surface());
    const post::SurfaceSolution s{&c.problem, random_currents(c.space.size(), 8)};
    const Vertices dirs = some_directions();
    FieldMatrix F, E, H;
    Vertices pts(1, 3);
    pts << 0.0, 0.0, -1e-6;
    post::FieldOptions fo;
    fo.window = post::CurrentWindow{c.om.taper_end, c.om.pml_start};
    // Unstretched: near fields reject a window.
    CHECK_THROWS_WITH(post::scattered_field(s, pts, E, H, fo),
                      ContainsSubstring("window applies to far_field only"));
    CHECK_THROWS_AS(post::total_field(s, pts, E, H, fo), std::invalid_argument);
    CHECK_NOTHROW(post::scattered_field(s, pts, E, H));
    // Stretched.
    c.problem.stretch = &c.pml;
    CHECK_THROWS_WITH(post::far_field(s, dirs, F), ContainsSubstring("needs FieldOptions::window"));
    post::FieldOptions wide;
    wide.window = post::CurrentWindow{c.om.taper_end, c.om.pml_start * (1.0 + 1e-9)};
    CHECK_THROWS_WITH(post::far_field(s, dirs, F, wide),
                      ContainsSubstring("beyond the PML stretch start"));
    CHECK_NOTHROW(post::far_field(s, dirs, F, fo));  // outer_half == start is allowed
    CHECK(F.allFinite());
    CHECK_THROWS_AS(post::scattered_field(s, pts, E, H), std::invalid_argument);
    CHECK_THROWS_AS(post::total_field(s, pts, E, H), std::invalid_argument);
    post::FieldOptions bad;
    bad.window = post::CurrentWindow{2e-7, 1e-7};
    c.problem.stretch = nullptr;
    CHECK_THROWS_AS(post::far_field(s, dirs, F, bad), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Simulation with the PML termination
// ---------------------------------------------------------------------------------------------

namespace {

/// Flat open surface at h = 100 nm: patch 0.4, taper 0.2, gap 0.2, PML 0.2 um (16 x 16 cells).
OpenSurfaceMesh sim_surface(Real gap = 0.2e-6, Real taper = 0.2e-6) {
    return open_surface(100e-9, 0.4e-6, taper, gap, 0.2e-6);
}

std::shared_ptr<excitation::AngularSpectrumBeam> beam(Real waist, Real region_radius) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = waist;
    bp.region_radius = region_radius;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp);
}

SimulationConfig dense_config() {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::silicon_500nm();
    cfg.compression = "dense";
    return cfg;
}

/// The small surfaces have no dark band at 500 nm (pml_start 0.6 um < 3 w0 for any rigorous
/// waist); structural tests relax the tolerance.
PmlTermination relaxed() {
    PmlTermination t;
    t.dark_band_tolerance = 0.9;
    return t;
}

}  // namespace

TEST_CASE("pml simulation: construction, default window and accessors", "[pml_simulation]") {
    OpenSurfaceMesh om = sim_surface();
    const Real a1 = om.taper_end, a2 = om.pml_start;
    const Index triangles = om.mesh.num_triangles();
    const Index edges = om.mesh.num_edges();
    const Simulation sim(std::move(om), relaxed(), beam(0.4e-6, 0.9e-6), dense_config());
    CHECK(sim.has_pml());
    REQUIRE(sim.far_field_window().has_value());
    CHECK(sim.far_field_window()->inner_half == a1);
    CHECK(sim.far_field_window()->outer_half == a2);
    const post::FieldOptions fo = sim.far_field_options(8);
    CHECK(fo.quad_degree == 8);
    REQUIRE(fo.window.has_value());
    CHECK(fo.window->outer_half == a2);
    CHECK(sim.mesh().num_triangles() == triangles);
    CHECK(sim.num_unknowns() == 2 * edges);
    CHECK(static_cast<Index>(sim.zones().size()) == triangles);
    CHECK(sim.zone_half_widths()[2] == a2);
    CHECK(sim.problem().stretch == &sim.stretch());
    CHECK(op::has_stretch(sim.problem()));
    CHECK(sim.stretch().profile_x()->start == a2);
    CHECK(sim.pml_termination().sigma_max == 4.0);
    CHECK(sim.dark_band_ratio() > 0.0);
    CHECK(sim.dark_band_ratio() < 0.9);
    CHECK_THAT(sim.report(), ContainsSubstring("termination:    PML"));

    // Box route: no window, the PML accessors throw.
    const Simulation box(
        geometry::make_icosphere(0.3e-6, 1),
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0)),
        dense_config());
    CHECK_FALSE(box.has_pml());
    CHECK_FALSE(box.far_field_window().has_value());
    CHECK_FALSE(box.far_field_options().window.has_value());
    CHECK(box.problem().stretch == nullptr);
    CHECK_THROWS_AS(box.pml_termination(), std::logic_error);
    CHECK_THROWS_AS(box.stretch(), std::logic_error);
    CHECK_THROWS_AS(box.dark_band_ratio(), std::logic_error);
    CHECK_THAT(box.report(), !ContainsSubstring("termination"));
}

TEST_CASE("pml simulation: dark band, controlled region and excitation checks",
          "[pml_simulation]") {
    // Default tolerance 1e-4: the small surface has no dark band.
    CHECK_THROWS_WITH(
        Simulation(sim_surface(), PmlTermination{}, beam(0.4e-6, 0.9e-6), dense_config()),
        ContainsSubstring("no dark band") && ContainsSubstring("smaller beam waist"));
    // A plane wave never has one (ratio 1).
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(),
                                 std::make_shared<excitation::PlaneWave>(
                                     kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0)),
                                 dense_config()),
                      ContainsSubstring("no dark band"));
    // Controlled region: the unmasked vertices reach sqrt(2) 0.6 um = 0.85 um; the PML corners
    // (1.13 um) may lie outside.
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(), beam(0.4e-6, 0.8e-6), dense_config()),
                      ContainsSubstring("unmasked part of the mesh"));
    CHECK_NOTHROW(Simulation(sim_surface(), relaxed(), beam(0.4e-6, 0.86e-6), dense_config()));
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(), nullptr, dense_config()),
                      ContainsSubstring("excitation is null"));
    SimulationConfig wrong = dense_config();
    wrong.wavelength = 600e-9;
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(), beam(0.4e-6, 0.9e-6), wrong),
                      ContainsSubstring("wavelength"));
}

TEST_CASE("pml simulation: compression, gap, taper, orientation and termination checks",
          "[pml_simulation]") {
    const auto exc = beam(0.4e-6, 1.0e-6);
    // "mlfmm" is accepted since WP-PML5a (fast PML ring), with GMRES only.
    SimulationConfig mlfmm = dense_config();
    mlfmm.compression = "mlfmm";
    CHECK_NOTHROW(Simulation(sim_surface(), relaxed(), exc, mlfmm));
    // The local leaf rule (valid on the box route) is refused here, at construction.
    SimulationConfig local = mlfmm;
    local.mlfmm.leaf_radius_quantile = 0.9;
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(), exc, local),
                      ContainsSubstring("leaf_radius_quantile = 1"));
    mlfmm.solver = SolverKind::Direct;
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(), exc, mlfmm),
                      ContainsSubstring("direct solver needs the dense operator"));
    SimulationConfig unknown = dense_config();
    unknown.compression = "foo";
    CHECK_THROWS_WITH(Simulation(sim_surface(), relaxed(), exc, unknown),
                      ContainsSubstring("unknown compression"));
    SimulationConfig direct = dense_config();
    direct.solver = SolverKind::Direct;
    CHECK_NOTHROW(Simulation(sim_surface(), relaxed(), exc, direct));

    // No gap.
    CHECK_THROWS_WITH(Simulation(sim_surface(0.0), relaxed(), exc, dense_config()),
                      ContainsSubstring("gap W_g"));
    // Order < 1: gap 0.1 um < longest edge sqrt(2) 0.1 um; 0.2 um passes.
    PmlTermination low = relaxed();
    low.order = 0.5;
    CHECK_THROWS_WITH(Simulation(sim_surface(0.1e-6), low, exc, dense_config()),
                      ContainsSubstring("longest mesh edge"));
    CHECK_NOTHROW(Simulation(sim_surface(0.2e-6), low, exc, dense_config()));

    // Rough patch: Lc required, W_t >= Lc.
    const auto rough = [] {
        return open_surface(100e-9, 0.4e-6, 0.2e-6, 0.2e-6, 0.2e-6, 5e-9, 0.2e-6);
    };
    CHECK_THROWS_WITH(Simulation(rough(), relaxed(), exc, dense_config()),
                      ContainsSubstring("correlation_length"));
    PmlTermination lc = relaxed();
    lc.correlation_length = 0.3e-6;
    CHECK_THROWS_WITH(Simulation(rough(), lc, exc, dense_config()),
                      ContainsSubstring("narrower than the correlation length"));
    lc.correlation_length = 0.2e-6;
    CHECK_NOTHROW(Simulation(rough(), lc, exc, dense_config()));

    // Orientation and consistency of the open mesh.
    OpenSurfaceMesh flipped = sim_surface();
    flipped.mesh.flip_normals();
    CHECK_THROWS_WITH(Simulation(std::move(flipped), relaxed(), exc, dense_config()),
                      ContainsSubstring("n_z >= 0"));
    OpenSurfaceMesh short_zones = sim_surface();
    short_zones.zone.pop_back();
    CHECK_THROWS_WITH(Simulation(std::move(short_zones), relaxed(), exc, dense_config()),
                      ContainsSubstring("zone has"));

    // PmlTermination fields.
    for (const auto& [field, value] : std::vector<std::pair<int, Real>>{
             {0, 0.0}, {0, std::nan("")}, {1, -1.0}, {2, -1.0}, {3, 0.0}}) {
        PmlTermination t = relaxed();
        if (field == 0)
            t.sigma_max = value;
        else if (field == 1)
            t.order = value;
        else if (field == 2)
            t.correlation_length = value;
        else
            t.dark_band_tolerance = value;
        CHECK_THROWS_AS(Simulation(sim_surface(), t, exc, dense_config()), std::invalid_argument);
    }
}

TEST_CASE("pml simulation: far-field source, Huygens options and report", "[pml_simulation]") {
    const auto exc = beam(0.4e-6, 1.0e-6);
    // Default: the windowed currents (WP-PML6 switches only after 7f / 7g, ADR 0010).
    const Simulation cur(sim_surface(), relaxed(), exc, dense_config());
    CHECK(cur.far_field_source() == FarFieldSource::Currents);
    CHECK_FALSE(cur.far_field_options().huygens.has_value());
    const post::FieldOptions explicit_h = cur.far_field_options(6, FarFieldSource::Huygens);
    REQUIRE(explicit_h.huygens.has_value());
    CHECK_FALSE(explicit_h.huygens->z_h.has_value());
    CHECK(explicit_h.window->outer_half == 0.6e-6);
    CHECK_THAT(cur.report(), ContainsSubstring("far field:      windowed currents"));

    PmlTermination h = relaxed();
    h.far_field_source = FarFieldSource::Huygens;
    h.huygens.z_h = 0.75e-6;
    h.huygens.spacing = 100e-9;
    const Simulation hs(sim_surface(), h, exc, dense_config());
    CHECK(hs.far_field_source() == FarFieldSource::Huygens);
    const post::FieldOptions fo = hs.far_field_options(8);
    REQUIRE(fo.huygens.has_value());
    CHECK(*fo.huygens->z_h == 0.75e-6);
    CHECK(*fo.huygens->spacing == 100e-9);
    CHECK(fo.quad_degree == 8);
    CHECK(fo.window->inner_half == 0.4e-6);  // the plane window defaults to the current window
    CHECK_FALSE(hs.far_field_options(6, FarFieldSource::Currents).huygens.has_value());
    CHECK_THAT(hs.report(), ContainsSubstring("far field:      Huygens plane") &&
                                ContainsSubstring("z_h 7.5e-07") &&
                                ContainsSubstring("plane window [4e-07, 6e-07]"));

    // Option checks at construction (whatever the source).
    const auto refused = [&](const post::HuygensOptions& o) {
        PmlTermination t = relaxed();
        t.huygens = o;
        CHECK_THROWS_AS(Simulation(sim_surface(), t, exc, dense_config()), std::invalid_argument);
    };
    post::HuygensOptions o;
    o.z_h = 0.0;
    refused(o);
    o.z_h = std::nan("");
    refused(o);
    o = {};
    o.spacing = 0.0;
    refused(o);
    o.spacing = 0.26e-6;  // > lambda / 2
    refused(o);
    o = {};
    o.window = post::CurrentWindow{0.4e-6, 0.61e-6};  // beyond pml_start
    refused(o);
    o.window = post::CurrentWindow{0.5e-6, 0.4e-6};
    refused(o);
    o.window = post::CurrentWindow{0.2e-6, 0.6e-6};
    PmlTermination ok = relaxed();
    ok.huygens = o;
    CHECK_NOTHROW(Simulation(sim_surface(), ok, exc, dense_config()));

    // Box route: currents only.
    const Simulation box(
        geometry::make_icosphere(0.3e-6, 1),
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0)),
        dense_config());
    CHECK(box.far_field_source() == FarFieldSource::Currents);
    CHECK_THROWS_AS(box.far_field_options(6, FarFieldSource::Huygens), std::logic_error);
}

TEST_CASE("pml simulation: a tiny dense PML solve runs (direct and GMRES)", "[pml_simulation]") {
    // Patch 0.2, taper 0.1, gap 0.1, PML 0.1 um at h = 100 nm: 8 x 8 cells, 128 triangles.
    // Structural only (no dark band, PML one cell): masked rhs, finite currents, windowed far
    // field; physics is the validation-large smoke case (tests/validation_large) and WP-PML3b.
    // A low-index object (eps_r 2.25) and sigma_max 1 keep the stretched touching rule at its
    // base order (|k s| h small; Si at sigma_max 4 raises it to 16 points and costs ~25 s).
    const auto make = [] { return open_surface(100e-9, 0.2e-6, 0.1e-6, 0.1e-6, 0.1e-6); };
    SimulationConfig cfg = dense_config();
    cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    cfg.solver = SolverKind::Direct;
    PmlTermination term = relaxed();
    term.sigma_max = 1.0;
    // Cheap kernels (as the WP-PML2 structural tests): quick under the sanitizers.
    cfg.kernels.outer_grading_levels = 0;
    cfg.kernels.target_accuracy = 0.0;
    cfg.kernels.quad_degree_near = 5;
    cfg.kernels.stretched_touching_points = 4;
    Simulation sim(make(), term, beam(0.4e-6, 0.6e-6), cfg);
    const solver::GmresResult r = sim.solve();
    CHECK(r.converged);
    CHECK(r.true_relative_residual < 1e-10);
    const VectorXc& b = sim.rhs();
    const op::Problem& p = sim.problem();
    const Index N = p.space->size();
    const std::vector<std::uint8_t> mask = op::pml_masked_functions(*p.space, sim.stretch());
    Index masked = 0;
    for (Index n = 0; n < N; ++n) {
        if (mask[static_cast<std::size_t>(n)] != 0) {
            ++masked;
            CHECK(b(n) == Complex(0.0, 0.0));
            CHECK(b(N + n) == Complex(0.0, 0.0));
        }
    }
    CHECK(masked > 0);
    const VectorXc& x = sim.solution().currents;
    CHECK(x.allFinite());
    CHECK(x.cwiseAbs().maxCoeff() > 0.0);
    Vertices dirs(2, 3);
    dirs << 0.0, 0.0, -1.0, 0.5, 0.0, -0.8;
    FieldMatrix F;
    CHECK_THROWS_AS(post::far_field(sim.solution(), dirs, F), std::invalid_argument);
    post::far_field(sim.solution(), dirs, F, sim.far_field_options());
    CHECK(F.allFinite());
    CHECK(F.norm() > 0.0);
    // Huygens source through Simulation::far_field = post::huygens_far_field (WP-PML6); k_z > 0
    // rows are refused as for the currents.
    const post::FieldOptions fh = sim.far_field_options(6, FarFieldSource::Huygens);
    const FieldMatrix Fs = sim.far_field(dirs, fh);
    FieldMatrix Fh;
    post::huygens_far_field(sim.solution(), dirs, Fh, fh);
    CHECK((Fs.array() == Fh.array()).all());
    CHECK(Fs.allFinite());
    CHECK(Fs.norm() > 0.0);
    Vertices up(1, 3);
    up << 0.0, 0.3, 0.5;
    CHECK_THROWS_AS(sim.far_field(up, fh), std::invalid_argument);
    if (!kRelease)
        return;
    // GMRES (Jacobi per the recommendation) reproduces the direct solution.
    SimulationConfig g = cfg;
    g.solver = SolverKind::Gmres;
    g.gmres.tolerance = 1e-10;
    g.gmres.max_iter = 2000;
    Simulation sg(make(), term, beam(0.4e-6, 0.6e-6), g);
    const solver::GmresResult rg = sg.solve();
    CHECK(rg.converged);
    CHECK((sg.solution().currents - x).norm() < 1e-8 * x.norm());
}
