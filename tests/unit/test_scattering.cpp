// Unit tests of the absolute far-field observables (WP25, postprocessing/scattering.hpp and the
// Simulation conveniences): direction grids (weights, quadrature accuracy, layout), radiated
// power, DRC per steradian and its co/cross split, the absolute cylinder DRC and its approach to
// the far-field DRC, GaussianBeam::power(), Simulation::incident_power / drc / radiated_power /
// reflected_power / scattered_power (box and PML route) and the argument checks. The physical
// results (scattered power of a sphere against Mie, the flat-surface DRC integral against the
// Fresnel beam reflectance) are the validation tests tests/validation/test_drc_power.cpp and
// tests/validation_large/test_drc_fresnel_large.cpp.
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fields_test_support.hpp"

using namespace fields_test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr Real kInf = std::numeric_limits<Real>::infinity();

/// Sum of w f(k_hat) over a grid.
template <class F>
Real integrate(const post::SphereGrid& g, F&& f) {
    Real s = 0.0;
    for (Index r = 0; r < g.dirs.rows(); ++r) {
        s += g.weights(r) * f(Vec3(g.dirs.row(r).transpose()));
    }
    return s;
}

/// Directions (sin t, 0, -cos t) of the xz-plane cylinder (scattering angle t from -z).
Vertices cylinder_directions(const std::vector<Real>& angles) {
    Vertices d(static_cast<Index>(angles.size()), 3);
    for (std::size_t i = 0; i < angles.size(); ++i) {
        d.row(static_cast<Index>(i)) =
            Vec3(std::sin(angles[i]), 0.0, -std::cos(angles[i])).transpose();
    }
    return d;
}

/// Cheap quadrature of the driver tests (tests/unit/test_simulation.cpp).
kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions o;
    o.quad_degree_far = 1;
    o.quad_degree_near = 2;
    o.quad_degree_sing = 2;
    o.outer_grading_levels = 0;
    o.target_accuracy = 0.0;
    return o;
}

/// n = 1.5 sphere of radius 0.5 um, icosphere n = 1 (2N = 240) at lambda = 1 um, direct LU.
SimulationConfig sphere_config() {
    SimulationConfig c;
    c.wavelength = 1e-6;
    c.object = lossless_n15();
    c.kernels = cheap_options();
    c.solver = SolverKind::Direct;
    return c;
}

std::shared_ptr<excitation::AngularSpectrumBeam> sphere_beam() {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = 1e-6;
    bp.waist_radius = 1e-6;
    bp.region_radius = 1e-6;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp);
}

/// Flat open surface for the PML route (as tests/unit/test_pml_route.cpp, tiny solve).
geometry::OpenSurfaceMesh tiny_open_surface() {
    const Real h = 100e-9, patch = 0.2e-6, taper = 0.1e-6;
    const auto cells = static_cast<Index>(std::llround((patch + 2.0 * taper) / h));
    geometry::HeightMap m;
    m.dx = (patch + 2.0 * taper) / static_cast<Real>(cells);
    m.dy = m.dx;
    m.z = MatrixXr::Zero(cells + 1, cells + 1);
    geometry::OpenSurfaceParams op;
    op.patch_size = patch;
    op.taper_width = taper;
    op.gap_width = 0.1e-6;
    op.pml_width = 0.1e-6;
    return geometry::make_open_surface_mesh(m, op);
}

}  // namespace

TEST_CASE("scattering: hemisphere and sphere grids, layout and weight sums", "[scattering]") {
    // The weight sum is the Gauss-Legendre rule for int sin(theta): spectral in n_theta, at
    // rounding from n_theta = 7 (hemisphere) / 8 (sphere); n_theta = 1 is 11 % / 57 % off.
    CHECK_THAT(post::hemisphere_grid(-1.0, 1, 4).weights.sum(),
               WithinRel(2.0 * kPi * 1.1107207345, 1e-9));
    CHECK_THAT(post::hemisphere_grid(-1.0, 4, 4).weights.sum(),
               WithinRel(2.0 * kPi * (1.0 - 2.2802885e-8), 1e-14));
    for (const Real z_sign : {-1.0, 1.0}) {
        for (const auto& [nt, np] :
             std::vector<std::pair<int, int>>{{1, 4}, {7, 12}, {8, 5}, {90, 180}}) {
            const post::SphereGrid g = post::hemisphere_grid(z_sign, nt, np);
            const Index rows = static_cast<Index>(nt) * np;
            REQUIRE(g.dirs.rows() == rows);
            REQUIRE(g.weights.size() == rows);
            REQUIRE(g.theta.size() == rows);
            if (nt >= 7)
                CHECK_THAT(g.weights.sum(), WithinRel(2.0 * kPi, 1e-14));
            for (Index r = 0; r < rows; ++r) {
                CHECK_THAT(g.dirs.row(r).norm(), WithinAbs(1.0, 1e-15));
                CHECK(z_sign * g.dirs(r, 2) > 0.0);  // Gauss-Legendre: no node at the equator
                CHECK(g.theta(r) > 0.0);
                CHECK(g.theta(r) < 0.5 * kPi);
                CHECK_THAT(z_sign * g.dirs(r, 2), WithinAbs(std::cos(g.theta(r)), 1e-15));
                CHECK(g.weights(r) > 0.0);
            }
        }
    }
    // Row = i * n_phi + j: phi_j = j 2 pi / n_phi at fixed theta_i.
    const post::SphereGrid g = post::hemisphere_grid(-1.0, 3, 8);
    CHECK(g.theta(8) == g.theta(15));
    CHECK(g.theta(0) < g.theta(8));
    CHECK_THAT(std::atan2(g.dirs(10, 1), g.dirs(10, 0)), WithinAbs(2.0 * 2.0 * kPi / 8.0, 1e-14));
    CHECK_THAT(post::sphere_grid(1, 4).weights.sum(), WithinRel(2.0 * kPi * kPi, 1e-14));
    for (const auto& [nt, np] : std::vector<std::pair<int, int>>{{1, 4}, {9, 16}, {180, 360}}) {
        const post::SphereGrid s = post::sphere_grid(nt, np);
        REQUIRE(s.dirs.rows() == static_cast<Index>(nt) * np);
        if (nt >= 8)
            CHECK_THAT(s.weights.sum(), WithinRel(4.0 * kPi, 1e-14));
        CHECK(s.theta.minCoeff() > 0.0);
        CHECK(s.theta.maxCoeff() < kPi);
        for (Index r = 0; r < s.dirs.rows(); ++r) {
            CHECK_THAT(s.dirs(r, 2), WithinAbs(std::cos(s.theta(r)), 1e-15));
        }
    }
}

TEST_CASE("scattering: grids integrate low-order polynomials in cos theta to rounding",
          "[scattering]") {
    // Gauss-Legendre in theta is spectrally accurate for these entire integrands (not exact:
    // cos^m sin is no polynomial in theta); 16 nodes reach rounding. The trapezoid rule in phi is
    // exact for trigonometric polynomials of degree < n_phi. The sphere needs more nodes (its
    // theta interval is twice as long).
    const int nt = 16, nt_sphere = 24, np = 8;
    for (const Real z_sign : {-1.0, 1.0}) {
        const post::SphereGrid h = post::hemisphere_grid(z_sign, nt, np);
        for (int m = 0; m <= 6; ++m) {
            // int over the hemisphere around z_sign z of (k_z)^m = z_sign^m 2 pi / (m + 1).
            const Real ref = std::pow(z_sign, m) * 2.0 * kPi / (m + 1);
            const Real val = integrate(h, [m](const Vec3& k) { return std::pow(k.z(), m); });
            CHECK_THAT(val, WithinRel(ref, 1e-13));
        }
        // x^2 + 2 x y + x z: pi (2 / 3) + 0 + 0 over the hemisphere (exact in phi).
        const Real v = integrate(
            h, [](const Vec3& k) { return k.x() * k.x() + 2.0 * k.x() * k.y() + k.x() * k.z(); });
        CHECK_THAT(v, WithinRel(2.0 * kPi / 3.0, 1e-13));
    }
    const post::SphereGrid s = post::sphere_grid(nt_sphere, np);
    for (int m = 0; m <= 6; ++m) {
        const Real ref = (m % 2 == 0) ? 4.0 * kPi / (m + 1) : 0.0;
        const Real val = integrate(s, [m](const Vec3& k) { return std::pow(k.z(), m); });
        CHECK_THAT(val, WithinAbs(ref, 1e-13));
    }
    CHECK_THAT(integrate(s, [](const Vec3& k) { return k.y() * k.y(); }),
               WithinRel(4.0 * kPi / 3.0, 1e-13));
}

TEST_CASE("scattering: grid builders reject invalid arguments", "[scattering]") {
    CHECK_THROWS_AS(post::hemisphere_grid(0.5, 4, 8), std::invalid_argument);
    CHECK_THROWS_AS(post::hemisphere_grid(0.0, 4, 8), std::invalid_argument);
    CHECK_THROWS_AS(post::hemisphere_grid(std::nan(""), 4, 8), std::invalid_argument);
    CHECK_THROWS_AS(post::hemisphere_grid(-1.0, 0, 8), std::invalid_argument);
    CHECK_THROWS_AS(post::hemisphere_grid(-1.0, 4, 3), std::invalid_argument);
    CHECK_THROWS_AS(post::sphere_grid(0, 8), std::invalid_argument);
    CHECK_THROWS_AS(post::sphere_grid(4, 3), std::invalid_argument);
}

TEST_CASE("scattering: drc integrates to radiated_power / P_inc and splits into co + cross",
          "[scattering]") {
    // Projected Mie currents (no solve) of the n = 1.5 sphere, icosphere n = 1.
    const MieSetup m(lossless_n15(), 1);
    const Real p_inc = 3.7e-13;  // W, arbitrary
    for (const Real z_sign : {-1.0, 1.0}) {
        const post::SphereGrid g = post::hemisphere_grid(z_sign, 12, 24);
        const VectorXr d = post::drc(m.solution, g.dirs, p_inc);
        REQUIRE(d.size() == g.dirs.rows());
        CHECK((d.array() >= 0.0).all());
        const Real p = post::radiated_power(m.solution, g);
        CHECK(p > 0.0);
        CHECK_THAT(g.weights.dot(d), WithinRel(p / p_inc, 1e-13));
        // The (dirs, weights) overload is the grid overload.
        CHECK(post::radiated_power(m.solution, g.dirs, g.weights) == p);
        // D = |F|^2 / (2 eta_0 P_inc) per direction.
        Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
        post::far_field(m.solution, g.dirs, F);
        for (Index r = 0; r < F.rows(); r += 37) {
            CHECK_THAT(d(r),
                       WithinRel(F.row(r).squaredNorm() / (2.0 * constants::eta0 * p_inc), 1e-14));
        }
        // Co / cross with the incident polarisation x (any scale of the vector).
        const post::PolarizedIntensity pc =
            post::drc_polarized(m.solution, g.dirs, p_inc, Vec3(2.0, 0.0, 0.0));
        REQUIRE(pc.co.size() == d.size());
        CHECK(((pc.co + pc.cross - d).array().abs() <= 1e-12 * d.maxCoeff()).all());
        CHECK((pc.cross.array() >= 0.0).all());
        // On the axis the scattered far field of the x-polarised wave is (nearly: the icosphere
        // is not exactly symmetric) x-polarised; measured cross / co 1.8e-5 backward, 2e-8
        // forward.
        const post::PolarizedIntensity pole = post::drc_polarized(
            m.solution, Vertices(Vec3(0.0, 0.0, z_sign).transpose()), p_inc, Vec3::UnitX());
        CHECK(pole.cross(0) < 1e-4 * pole.co(0));
    }
    // Full sphere = both hemispheres (different rules, both converged for this smooth pattern).
    const Real full = post::radiated_power(m.solution, post::sphere_grid(48, 64));
    const Real halves = post::radiated_power(m.solution, post::hemisphere_grid(-1.0, 24, 64)) +
                        post::radiated_power(m.solution, post::hemisphere_grid(1.0, 24, 64));
    CHECK_THAT(full, WithinRel(halves, 1e-12));
    // A FieldOptions quadrature degree is passed through to far_field.
    post::FieldOptions fo;
    fo.quad_degree = 2;
    const post::SphereGrid g = post::hemisphere_grid(-1.0, 6, 12);
    CHECK(post::radiated_power(m.solution, g, fo) != post::radiated_power(m.solution, g));
}

TEST_CASE("scattering: scattered power of projected Mie currents approaches the Mie C_sca",
          "[scattering]") {
    // P_sca / I_inc with I_inc = |E_0|^2 / (2 eta_0) against the Mie cross section, for the
    // projected Mie currents of the n = 1.5 sphere at lambda = 1 um (ka = pi) on icosphere n = 2
    // (h ~ lambda / 6.6), which carry the RWG projection error (measured -4.6 %; the projection
    // at lambda = 500 nm, h ~ lambda / 3.3, gives -26 %). The quantitative check (solved
    // currents, Si at 500 nm, < 2 %) is the validation test.
    const MieSetup m(lossless_n15(), 2, 1e-6);
    const Real c_sca =
        post::radiated_power(m.solution, post::sphere_grid(32, 64)) * (2.0 * constants::eta0);
    const Real ref = m.mie.scattering_cross_section();
    WARN("projected Mie currents n = 2: C_sca " << c_sca << " m^2, Mie " << ref << ", rel "
                                                << c_sca / ref - 1.0);
    CHECK_THAT(c_sca, WithinRel(ref, 0.1));
}

TEST_CASE("scattering: radiated power and drc argument checks", "[scattering]") {
    const MieSetup m(lossless_n15(), 1);
    const post::SphereGrid g = post::hemisphere_grid(-1.0, 4, 8);
    // Incident power.
    for (const Real p : {0.0, -1.0, std::nan(""), kInf}) {
        CHECK_THROWS_AS(post::drc(m.solution, g.dirs, p), std::invalid_argument);
        CHECK_THROWS_AS(post::drc_polarized(m.solution, g.dirs, p, Vec3::UnitX()),
                        std::invalid_argument);
    }
    // Empty grids.
    const Vertices none(0, 3);
    CHECK_THROWS_WITH(post::drc(m.solution, none, 1.0), ContainsSubstring("empty"));
    CHECK_THROWS_AS(post::radiated_power(m.solution, none, VectorXr(0)), std::invalid_argument);
    CHECK_THROWS_AS(post::radiated_power(m.solution, post::SphereGrid{}), std::invalid_argument);
    CHECK_THROWS_AS(post::drc_polarized(m.solution, none, 1.0, Vec3::UnitX()),
                    std::invalid_argument);
    // Weights.
    CHECK_THROWS_AS(post::radiated_power(m.solution, g.dirs, g.weights.head(3)),
                    std::invalid_argument);
    VectorXr w = g.weights;
    w(2) = -1e-3;
    CHECK_THROWS_AS(post::radiated_power(m.solution, g.dirs, w), std::invalid_argument);
    w(2) = std::nan("");
    CHECK_THROWS_AS(post::radiated_power(m.solution, g.dirs, w), std::invalid_argument);
    // Polarisation vector.
    CHECK_THROWS_AS(post::drc_polarized(m.solution, g.dirs, 1.0, Vec3::Zero()),
                    std::invalid_argument);
    CHECK_THROWS_AS(post::drc_polarized(m.solution, g.dirs, 1.0, Vec3(std::nan(""), 0.0, 0.0)),
                    std::invalid_argument);
    // Zero direction: far_field's check.
    Vertices bad = g.dirs;
    bad.row(1).setZero();
    CHECK_THROWS_AS(post::drc(m.solution, bad, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(post::radiated_power(m.solution, bad, g.weights), std::invalid_argument);
    // Null problem.
    const post::SurfaceSolution no_problem{nullptr, m.solution.currents};
    CHECK_THROWS_AS(post::drc(no_problem, g.dirs, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(post::radiated_power(no_problem, g), std::invalid_argument);
    // Lossy (or non-positive) exterior: no far-field power.
    for (const material::Material bg :
         {material::Material{Complex(1.0, -0.01), Complex(1.0, 0.0)},
          material::Material{Complex(1.0, 0.0), Complex(1.0, -0.1)},
          material::Material{Complex(-2.0, 0.0), Complex(1.0, 0.0)}}) {
        op::Problem p = m.problem;
        p.exterior = bg;
        p.excitation = nullptr;
        const post::SurfaceSolution lossy{&p, m.solution.currents};
        CHECK_THROWS_WITH(post::radiated_power(lossy, g), ContainsSubstring("lossless"));
        CHECK_THROWS_WITH(post::drc(lossy, g.dirs, 1.0), ContainsSubstring("lossless"));
        CHECK_THROWS_WITH(post::drc_polarized(lossy, g.dirs, 1.0, Vec3::UnitX()),
                          ContainsSubstring("lossless"));
    }
}

TEST_CASE("scattering: absolute cylinder DRC scales the relative one and checks its arguments",
          "[scattering]") {
    const int nt = 3, ny = 2;
    FieldMatrix E(nt * ny, 3);
    for (Index r = 0; r < E.rows(); ++r) {
        E.row(r) << Complex(1.0 + static_cast<Real>(r), 0.5), Complex(0.0, -2.0), 0.0;
    }
    const VectorXr rel = post::differential_reflection_coefficient(E, nt, ny);
    const Real radius = 2e-3, p_inc = 1e-9;
    const material::Material glass{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    const VectorXr d = post::differential_reflection_coefficient(E, nt, ny, radius, p_inc, glass);
    const Real eta1 = constants::eta0 / 1.5;
    for (Index k = 0; k < nt; ++k) {
        CHECK_THAT(d(k), WithinRel(rel(k) * radius * radius / (2.0 * eta1 * p_inc), 1e-14));
    }
    const material::Material vac = material::vacuum();
    for (const Real bad : {0.0, -1.0, std::nan(""), kInf}) {
        CHECK_THROWS_AS(post::differential_reflection_coefficient(E, nt, ny, bad, p_inc, vac),
                        std::invalid_argument);
        CHECK_THROWS_AS(post::differential_reflection_coefficient(E, nt, ny, radius, bad, vac),
                        std::invalid_argument);
    }
    CHECK_THROWS_AS(post::differential_reflection_coefficient(E, nt, ny + 1, radius, p_inc, vac),
                    std::invalid_argument);
    CHECK_THROWS_WITH(
        post::differential_reflection_coefficient(
            E, nt, ny, radius, p_inc, material::Material{Complex(1.0, -1e-3), Complex(1.0, 0.0)}),
        ContainsSubstring("lossless"));
}

TEST_CASE("scattering: absolute cylinder DRC approaches the far-field DRC like 1 / r",
          "[scattering]") {
    // Scattered near field of projected Mie currents (n = 1.5 sphere of radius 0.5 um at
    // lambda = 1 um, icosphere n = 1) on xz-plane cylinders (n_y = 1, y = 0) of growing radius,
    // against drc() in the same directions (same currents and source quadrature, so the limit is
    // exact): the relative deviation falls like 1 / r (the documented O(1/(k r)), O(L / r) and
    // O(k L^2 / r) terms).
    const MieSetup m(lossless_n15(), 1, 1e-6);
    const Real p_inc = 2e-12;
    const std::vector<Real> angles = {0.1, 0.5, 1.0, 1.5, 2.2, 2.9};
    const auto nt = static_cast<int>(angles.size());
    const VectorXr d_far = post::drc(m.solution, cylinder_directions(angles), p_inc);
    std::vector<Real> radii, errors;
    for (Real r = 10e-6; r < 3e-3; r *= 2.0) {
        Vertices pts = r * cylinder_directions(angles);
        FieldMatrix E, H;
        post::scattered_field(m.solution, pts, E, H);
        const VectorXr d_cyl =
            post::differential_reflection_coefficient(E, nt, 1, r, p_inc, material::vacuum());
        radii.push_back(r);
        errors.push_back(((d_cyl - d_far).array().abs() / d_far.array()).maxCoeff());
    }
    std::string table;
    for (std::size_t i = 0; i < radii.size(); ++i) {
        table += "  r = " + std::to_string(radii[i] * 1e6) + " um: max rel " +
                 std::to_string(errors[i]) + ", r * err = " + std::to_string(radii[i] * errors[i]) +
                 " m\n";
    }
    WARN("cylinder DRC vs far-field DRC:\n" << table);
    for (std::size_t i = 1; i < errors.size(); ++i) {
        // Each doubling of r halves the deviation (1 / r), within a factor band.
        CHECK(errors[i] < 0.65 * errors[i - 1]);
        CHECK(errors[i] > 0.35 * errors[i - 1]);
    }
    CHECK(errors.back() < 1e-3);
}

TEST_CASE("scattering: GaussianBeam power is pi w0^2 / (4 eta)", "[scattering]") {
    excitation::GaussianBeam::Params p;
    p.wavelength = 500e-9;
    p.waist_radius = 2e-6;
    const excitation::GaussianBeam vac(p);
    CHECK_THAT(vac.power(), WithinRel(kPi * 4e-12 / (4.0 * constants::eta0), 1e-14));
    const excitation::GaussianBeam glass(p, material::Material{Complex(2.25, 0.0), Complex(1, 0)});
    CHECK_THAT(glass.power(), WithinRel(1.5 * vac.power(), 1e-14));
    // Numerical flux of 1/2 Re(E x H*) . k_hat of the model's fields through the waist plane and
    // a plane 10 z_R behind it (Gauss-Legendre in rho <= 5 w x trapezoid in phi).
    p.incidence_angle = 0.3;
    const excitation::GaussianBeam tilted(p);
    const Vec3 k(std::sin(0.3), 0.0, std::cos(0.3));
    const Vec3 u(std::cos(0.3), 0.0, -std::sin(0.3));
    const Vec3 v = Vec3::UnitY();
    const Real z_r = kPi * p.waist_radius * p.waist_radius / p.wavelength;
    const kernels::LineRule gl = kernels::gauss_legendre(48);
    for (const Real zeta : {0.0, 10.0 * z_r}) {
        const Real w = p.waist_radius * std::sqrt(1.0 + (zeta / z_r) * (zeta / z_r));
        const int n_phi = 16;
        const Real half = 2.5 * w;  // rho in [0, 5 w]: exp(-50) beyond
        Real flux = 0.0;
        for (std::size_t i = 0; i < gl.nodes.size(); ++i) {
            const Real rho = half * (gl.nodes[i] + 1.0);
            const Real wr = half * gl.weights[i] * rho * 2.0 * kPi / n_phi;
            for (int j = 0; j < n_phi; ++j) {
                const Real phi = 2.0 * kPi * j / n_phi;
                const Vec3 r = zeta * k + rho * (std::cos(phi) * u + std::sin(phi) * v);
                const Vec3c E = tilted.electric_field(r);
                const Vec3c H = tilted.magnetic_field(r);
                const Vec3c Hc = H.conjugate();
                // (E x H*) . k without Eigen's conjugating complex cross.
                const Vec3c s(E.y() * Hc.z() - E.z() * Hc.y(), E.z() * Hc.x() - E.x() * Hc.z(),
                              E.x() * Hc.y() - E.y() * Hc.x());
                flux += wr * 0.5 * (s.x() * k.x() + s.y() * k.y() + s.z() * k.z()).real();
            }
        }
        CHECK_THAT(flux, WithinRel(tilted.power(), 1e-10));
    }
}

TEST_CASE("scattering: GaussianBeam power is the paraxial limit of AngularSpectrumBeam power",
          "[scattering]") {
    // Both beams carry the unit on-axis amplitude E(focus) . e0_hat = 1 V/m, so their powers are
    // compared without renormalisation. With f = lambda / (pi w0) = 2 / (k w0), s = k_t / k and
    // the spectrum A = exp(-s^2 / f^2), the rigorous beam has
    //   E(focus) ~ int A (1 - s^2 cos^2 phi) d^2 s,
    //   P ~ int A^2 (1 - s^2 cos^2 phi) cos(alpha) d^2 s
    // (|p|^2 = 1 - (k_hat . e0_hat)^2, cos(alpha) = sqrt(1 - s^2)); with the moments
    // int A s^2 / int A = f^2 / 2 and int A^2 s^2 / int A^2 = f^2 / 4 this gives
    //   P_asb / P_gauss = 1 + f^2 / 2 + c f^4,  c = 0.313 (1-D integrals evaluated numerically),
    // where P_gauss = pi w0^2 / (4 eta) is the f -> 0 limit. At w0 = 10 lambda: f^2 = 1.01e-3,
    // so the two powers agree to 5.1e-4 relative, and the deviation from f^2 / 2 is O(f^4) =
    // 1e-6 (tolerance f^4, about 3 times the expected remainder; the angular-spectrum grid is
    // converged to its 1e-10 tolerance).
    for (const Real waist_in_lambda : {10.0, 5.0}) {
        const Real lambda = 500e-9;
        const Real w0 = waist_in_lambda * lambda;
        excitation::GaussianBeam::Params gp;
        gp.wavelength = lambda;
        gp.waist_radius = w0;
        const excitation::GaussianBeam gauss(gp);
        excitation::AngularSpectrumBeam::Params ap;
        ap.wavelength = lambda;
        ap.waist_radius = w0;
        ap.region_radius = w0;  // the power does not depend on it; keeps the grid small
        const excitation::AngularSpectrumBeam asb(ap);
        // Same amplitude convention: unit on-axis field at the focus for both.
        CHECK_THAT(gauss.electric_field(Vec3::Zero()).x().real(), WithinAbs(1.0, 1e-14));
        CHECK_THAT(asb.electric_field(Vec3::Zero()).x().real(), WithinAbs(1.0, 1e-9));
        const Real f2 = std::pow(lambda / (kPi * w0), 2);
        const Real dev = asb.power() / gauss.power() - 1.0;
        INFO("w0 = " << waist_in_lambda << " lambda: P_asb / P_gauss - 1 = " << dev
                     << ", f^2 / 2 = " << 0.5 * f2);
        CHECK(std::abs(dev) < f2);  // agreement to O(f^2): 1.0e-3 at w0 = 10 lambda
        CHECK_THAT(dev, WithinAbs(0.5 * f2, f2 * f2));
        if (waist_in_lambda == 10.0)
            CHECK(std::abs(dev) < 1e-3);
    }
}

TEST_CASE("scattering: Simulation incident power, drc and powers on the box route",
          "[scattering]") {
    // Plane wave: no finite power, but the radiated powers work.
    {
        Simulation sim(
            geometry::make_icosphere(kRadius, 1),
            std::make_shared<excitation::PlaneWave>(1e-6, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0)),
            sphere_config());
        CHECK_THROWS_WITH(sim.incident_power(), ContainsSubstring("plane wave"));
        CHECK_THROWS_AS(sim.incident_power(), std::logic_error);
        const post::SphereGrid g = post::hemisphere_grid(-1.0, 4, 8);
        CHECK_THROWS_AS(sim.reflected_power(4, 8), std::logic_error);  // before solve()
        CHECK_THROWS_AS(sim.scattered_power(4, 8), std::logic_error);
        CHECK_THROWS_AS(sim.radiated_power(g.dirs, g.weights), std::logic_error);
        CHECK_THROWS_AS(sim.drc(g.dirs), std::logic_error);
        sim.solve();
        CHECK_THROWS_AS(sim.drc(g.dirs), std::logic_error);  // incident_power()
        CHECK_THROWS_AS(sim.drc_polarized(g.dirs, Vec3::UnitX()), std::logic_error);
        const Real p_back = sim.reflected_power(4, 8);
        CHECK(p_back > 0.0);
        CHECK(sim.radiated_power(g.dirs, g.weights) == p_back);
        CHECK(sim.scattered_power(8, 16) > p_back);
    }
    // Gaussian beam: GaussianBeam::power().
    {
        excitation::GaussianBeam::Params gp;
        gp.wavelength = 1e-6;
        gp.waist_radius = 1.5e-6;
        const auto gb = std::make_shared<excitation::GaussianBeam>(gp);
        const Simulation sim(geometry::make_icosphere(kRadius, 1), gb, sphere_config());
        CHECK(sim.incident_power() == gb->power());
    }
    // Rigorous beam: AngularSpectrumBeam::power(), and the conveniences equal the post calls.
    const auto beam = sphere_beam();
    Simulation sim(geometry::make_icosphere(kRadius, 1), beam, sphere_config());
    CHECK(sim.incident_power() == beam->power());
    CHECK_THROWS_AS(sim.far_field(Vertices(Vec3::UnitZ().transpose())), std::logic_error);
    sim.solve();
    const post::SphereGrid back = post::hemisphere_grid(-1.0, 6, 12);
    const post::SphereGrid fwd = post::hemisphere_grid(1.0, 6, 12);
    const post::FieldOptions fo = sim.far_field_options(4);
    // Simulation::far_field: the box route accepts both hemispheres (k_z > 0 too).
    {
        Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
        post::far_field(sim.solution(), fwd.dirs, F, fo);
        CHECK(sim.far_field(fwd.dirs, 4) == F);
        CHECK(sim.far_field(fwd.dirs, fo) == F);
        CHECK(F.allFinite());
    }
    CHECK(sim.drc(back.dirs, 4) == post::drc(sim.solution(), back.dirs, beam->power(), fo));
    const post::PolarizedIntensity pc = sim.drc_polarized(back.dirs, Vec3::UnitX(), 4);
    const post::PolarizedIntensity pr =
        post::drc_polarized(sim.solution(), back.dirs, beam->power(), Vec3::UnitX(), fo);
    CHECK(pc.co == pr.co);
    CHECK(pc.cross == pr.cross);
    const Real p_refl = sim.reflected_power(6, 12, 4);
    CHECK(p_refl == post::radiated_power(sim.solution(), back, fo));
    CHECK(sim.radiated_power(back.dirs, back.weights, 4) == p_refl);
    CHECK_THAT(back.weights.dot(sim.drc(back.dirs, 4)), WithinRel(p_refl / beam->power(), 1e-13));
    // Box route: every direction has a far field; full sphere = both hemispheres.
    const Real p_fwd = sim.radiated_power(fwd.dirs, fwd.weights, 4);
    CHECK_THAT(sim.scattered_power(24, 48, 4),
               WithinRel(sim.reflected_power(12, 48, 4) +
                             sim.radiated_power(post::hemisphere_grid(1.0, 12, 48).dirs,
                                                post::hemisphere_grid(1.0, 12, 48).weights, 4),
                         1e-9));
    CHECK(p_fwd > 0.0);
    CHECK_THROWS_AS(sim.reflected_power(0, 8), std::invalid_argument);
    CHECK_THROWS_AS(sim.scattered_power(4, 3), std::invalid_argument);
    CHECK_THROWS_AS(sim.drc(back.dirs, 3), std::invalid_argument);  // not a positive rule
}

TEST_CASE("scattering: Simulation drc and powers on the PML route", "[scattering]") {
    // Tiny dense PML solve of tests/unit/test_pml_route.cpp (structural only): the window is
    // applied automatically, k_z > 0 directions and scattered_power are refused.
    SimulationConfig cfg;
    cfg.wavelength = 500e-9;
    cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    cfg.compression = "dense";
    cfg.solver = SolverKind::Direct;
    cfg.kernels.outer_grading_levels = 0;
    cfg.kernels.target_accuracy = 0.0;
    cfg.kernels.quad_degree_near = 5;
    cfg.kernels.stretched_touching_points = 4;
    PmlTermination term;
    term.sigma_max = 1.0;
    term.dark_band_tolerance = 0.9;
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = 500e-9;
    bp.waist_radius = 0.4e-6;
    bp.region_radius = 0.6e-6;
    const auto beam = std::make_shared<excitation::AngularSpectrumBeam>(bp);
    Simulation sim(tiny_open_surface(), term, beam, cfg);
    sim.solve();
    REQUIRE(sim.has_pml());
    const post::SphereGrid g = post::hemisphere_grid(-1.0, 6, 12);
    const post::FieldOptions fo = sim.far_field_options();
    REQUIRE(fo.window.has_value());
    const VectorXr d = sim.drc(g.dirs);
    CHECK(d == post::drc(sim.solution(), g.dirs, beam->power(), fo));
    CHECK(d.allFinite());
    CHECK(d.maxCoeff() > 0.0);
    // Without the window post::drc refuses the stretched problem.
    CHECK_THROWS_AS(post::drc(sim.solution(), g.dirs, beam->power()), std::invalid_argument);
    const Real p_refl = sim.reflected_power(6, 12);
    CHECK(p_refl == post::radiated_power(sim.solution(), g, fo));
    CHECK_THAT(g.weights.dot(d), WithinRel(p_refl / sim.incident_power(), 1e-13));
    // Grazing directions (k_z = 0) are accepted, k_z > 0 is refused.
    Vertices graze(2, 3);
    graze << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    CHECK(sim.drc(graze).allFinite());
    Vertices up = g.dirs;
    up(5, 2) = 1e-3;
    CHECK_THROWS_WITH(sim.drc(up), ContainsSubstring("k_z > 0"));
    CHECK_THROWS_WITH(sim.drc_polarized(up, Vec3::UnitX()), ContainsSubstring("k_z > 0"));
    CHECK_THROWS_WITH(sim.radiated_power(up, g.weights), ContainsSubstring("k_z > 0"));
    CHECK_THROWS_WITH(sim.scattered_power(4, 8), ContainsSubstring("PML route"));
    CHECK_THROWS_AS(sim.scattered_power(4, 8), std::logic_error);
    // Simulation::far_field (WP25 review S1): the same direction rule, windowed by default.
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(sim.solution(), g.dirs, F, fo);
    CHECK(sim.far_field(g.dirs) == F);
    CHECK(sim.far_field(g.dirs, fo) == F);
    CHECK(sim.far_field(graze).allFinite());
    CHECK_THROWS_WITH(sim.far_field(up), ContainsSubstring("k_z > 0"));
    CHECK_THROWS_AS(sim.far_field(up), std::invalid_argument);
    CHECK_THROWS_AS(sim.far_field(up, fo), std::invalid_argument);
    CHECK_THROWS_AS(sim.far_field(Vertices(Vec3::UnitZ().transpose())), std::invalid_argument);
    // The window is still required (an explicit FieldOptions without one is refused).
    CHECK_THROWS_AS(sim.far_field(g.dirs, post::FieldOptions{}), std::invalid_argument);
}

// Structural checks on a non-zero height map (WP25 review S3): D finite and >= 0, the hemisphere
// integral of D equals reflected_power / incident_power to rounding, co + cross = D. Fixed
// seeds, dense direct solves of a few hundred unknowns.
TEST_CASE("scattering: drc and powers on a tiny rough surface, box route", "[scattering]") {
    const Real h = 100e-9;
    {
        geometry::RoughSurfaceParams rp;
        rp.edge_length_L = 0.4e-6;
        rp.mesh_size = h;
        rp.rms_roughness = 20e-9;
        rp.correlation_length = 0.2e-6;
        rp.seed = 2510;
        const geometry::HeightMap hm = geometry::generate_gaussian_height_map(rp);
        REQUIRE(hm.z.cwiseAbs().maxCoeff() > 1e-9);  // really rough
        geometry::TriangleMesh mesh = geometry::make_mesh_from_height_map(hm, 0.3e-6, h);
        Real r_max = 0.0;
        for (Index v = 0; v < mesh.num_vertices(); ++v)
            r_max = std::max(r_max, mesh.vertices().row(v).norm());
        excitation::AngularSpectrumBeam::Params bp;
        bp.wavelength = 1e-6;
        bp.waist_radius = 0.3e-6;
        bp.region_radius = 1.05 * r_max;
        const auto beam = std::make_shared<excitation::AngularSpectrumBeam>(bp);
        SimulationConfig cfg = sphere_config();
        cfg.compression = "dense";
        Simulation sim(std::move(mesh), beam, cfg);
        sim.solve();
        REQUIRE(sim.num_unknowns() < 1000);
        const post::SphereGrid g = post::hemisphere_grid(-1.0, 8, 16);
        const VectorXr d = sim.drc(g.dirs);
        CHECK(d.allFinite());
        CHECK(d.minCoeff() >= 0.0);
        CHECK(d.maxCoeff() > 0.0);
        const Real p_refl = sim.reflected_power(8, 16);
        CHECK(p_refl == sim.radiated_power(g.dirs, g.weights));
        CHECK_THAT(g.weights.dot(d), WithinRel(p_refl / sim.incident_power(), 1e-13));
        const post::PolarizedIntensity p = sim.drc_polarized(g.dirs, Vec3::UnitX());
        CHECK(((p.co + p.cross - d).array().abs() <= 1e-12 * d.maxCoeff()).all());
        // The box route also radiates forward (through the box): the full sphere has more.
        CHECK(sim.scattered_power(16, 16) > p_refl);
    }
}

TEST_CASE("scattering: drc and powers on a tiny rough surface, PML route", "[scattering]") {
    const Real h = 100e-9;
    {
        // The tiny surface of the flat PML case (patch 0.2 um, taper, gap and PML 0.1 um: 8 x 8
        // cells) with random heights (uniform in +-10 nm, fixed seed, the same on every standard
        // library) on the 4 x 4 cells of patch and taper; the taper rule W_t >= Lc is met with
        // the nominal Lc = h of the white-noise map.
        geometry::HeightMap hm;
        hm.dx = h;
        hm.dy = h;
        hm.z = MatrixXr::Zero(5, 5);
        std::mt19937_64 rng(2511);
        for (Index i = 0; i < hm.z.rows(); ++i) {
            for (Index j = 0; j < hm.z.cols(); ++j) {
                const Real u = static_cast<Real>(rng() >> 11) * 0x1.0p-53;  // [0, 1)
                hm.z(i, j) = 10e-9 * (2.0 * u - 1.0);
            }
        }
        geometry::OpenSurfaceParams op;
        op.patch_size = 0.2e-6;
        op.taper_width = 0.1e-6;
        op.gap_width = 0.1e-6;
        op.pml_width = 0.1e-6;
        geometry::OpenSurfaceMesh om = geometry::make_open_surface_mesh(hm, op);
        REQUIRE(om.mesh.vertices().col(2).cwiseAbs().maxCoeff() > 1e-9);  // really rough
        SimulationConfig cfg;
        cfg.wavelength = 500e-9;
        cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
        cfg.compression = "dense";
        cfg.solver = SolverKind::Direct;
        cfg.kernels.outer_grading_levels = 0;
        cfg.kernels.target_accuracy = 0.0;
        cfg.kernels.quad_degree_near = 5;
        cfg.kernels.stretched_touching_points = 4;
        PmlTermination term;
        term.sigma_max = 1.0;
        term.dark_band_tolerance = 0.9;
        term.correlation_length = h;
        excitation::AngularSpectrumBeam::Params bp;
        bp.wavelength = 500e-9;
        bp.waist_radius = 0.4e-6;
        bp.region_radius = 0.6e-6;
        const auto beam = std::make_shared<excitation::AngularSpectrumBeam>(bp);
        Simulation sim(std::move(om), term, beam, cfg);
        sim.solve();
        REQUIRE(sim.has_pml());
        REQUIRE(sim.num_unknowns() < 1000);
        const post::SphereGrid g = post::hemisphere_grid(-1.0, 8, 16);
        const VectorXr d = sim.drc(g.dirs);
        CHECK(d.allFinite());
        CHECK(d.minCoeff() >= 0.0);
        CHECK(d.maxCoeff() > 0.0);
        const Real p_refl = sim.reflected_power(8, 16);
        CHECK_THAT(g.weights.dot(d), WithinRel(p_refl / sim.incident_power(), 1e-13));
        const post::PolarizedIntensity p = sim.drc_polarized(g.dirs, Vec3::UnitX());
        CHECK(((p.co + p.cross - d).array().abs() <= 1e-12 * d.maxCoeff()).all());
    }
}
