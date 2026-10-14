// Reference of the flat-interface validation (WP22c, docs/05 row "Flat box, tapered beam, 0 deg
// and 45 deg | Fresnel r_p, r_s"): plane-wave Fresnel coefficients in the exp(+jwt) convention,
// the beam reflectance R_beam of the rigorous angular-spectrum beam on an infinite interface, the
// beam power against the numerical flux through z = 0 and the edge loss of the ADR 0006 waist
// rule (tests/support/fresnel_flat_support.hpp). The simulated flat box (MLFMM, 2N >= 7 10^4) is
// the validation-large case tests/validation_large/test_flat_interface_fresnel_large.cpp;
// results: benchmarks/results/fresnel_flat.md. The last case checks on a tiny solved flat box
// that the public Simulation DRC / power API gives the support code's reflectance (WP25).
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>

#include "fresnel_flat_support.hpp"

using namespace fresnel_flat;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::shared_ptr<excitation::AngularSpectrumBeam> beam(Real w0, Real theta_deg,
                                                      excitation::Polarization pol,
                                                      Real region_radius = 0.0) {
    excitation::AngularSpectrumBeam::Params p;
    p.wavelength = kLambda;
    p.waist_radius = w0;
    p.incidence_angle = theta_deg * kDeg;
    p.polarization = pol;
    p.region_radius = region_radius;
    return std::make_shared<excitation::AngularSpectrumBeam>(p, material::vacuum());
}

}  // namespace

TEST_CASE("Fresnel coefficients: conventions and identities", "[validation][fresnel]") {
    const Complex eps_si = material::silicon_500nm().eps_r;
    const Complex eps_ag = material::silver_500nm().eps_r;
    SECTION("normal incidence: r_s = (1 - n) / (1 + n) = -r_p, Im(n) <= 0") {
        for (const Complex eps : {eps_si, eps_ag}) {
            const Complex n = std::sqrt(eps);
            REQUIRE(n.imag() <= 0.0);
            const FresnelCoefficients c = fresnel(1.0, eps, 1.0);
            const Complex r = (1.0 - n) / (1.0 + n);
            CHECK(std::abs(c.r_s - r) < 1e-14);
            CHECK(std::abs(c.r_p + r) < 1e-14);
        }
        // Ag at 500 nm: |r|^2 = 0.98165 (absorptance 1.8 %), Si: 0.38772.
        CHECK_THAT(std::norm(fresnel(1.0, eps_ag, 1.0).r_s), WithinAbs(0.98165, 5e-5));
        CHECK_THAT(std::norm(fresnel(1.0, eps_si, 1.0).r_s), WithinAbs(0.38772, 5e-5));
    }
    SECTION("Abeles identity r_p = r_s^2 at 45 deg") {
        for (const Complex eps : {eps_si, eps_ag, Complex(2.25, 0.0)}) {
            const FresnelCoefficients c = fresnel(1.0, eps, std::cos(0.25 * constants::pi));
            CHECK(std::abs(c.r_p - c.r_s * c.r_s) < 1e-14);
        }
    }
    SECTION("Brewster angle of a lossless dielectric, grazing incidence, passivity") {
        const Real n = 1.5;
        CHECK(std::abs(fresnel(1.0, Complex(n * n, 0.0), std::cos(std::atan(n))).r_p) < 1e-14);
        for (const Complex eps : {eps_si, eps_ag}) {
            CHECK_THAT(std::norm(fresnel(1.0, eps, 0.0).r_s), WithinAbs(1.0, 1e-14));
            for (int i = 0; i < 18; ++i) {  // 0 ... 85 deg
                const FresnelCoefficients c = fresnel(1.0, eps, std::cos(i * 5.0 * kDeg));
                CHECK(std::norm(c.r_s) < 1.0);
                CHECK(std::norm(c.r_p) < 1.0);
            }
        }
        // Same reflectance with the background index scaled with the object (n1 = 2).
        const FresnelCoefficients a = fresnel(1.0, eps_si, std::cos(0.6));
        const FresnelCoefficients b = fresnel(2.0, 4.0 * eps_si, std::cos(0.6));
        CHECK(std::abs(a.r_s - b.r_s) < 1e-13);
        CHECK(std::abs(a.r_p - b.r_p) < 1e-13);
    }
    SECTION("invalid input") {
        CHECK_THROWS_AS(fresnel(1.0, Complex(2.0, 0.1), 1.0), std::invalid_argument);
        CHECK_THROWS_AS(fresnel(0.0, eps_si, 1.0), std::invalid_argument);
        CHECK_THROWS_AS(fresnel(1.0, eps_si, 1.5), std::invalid_argument);
    }
}

TEST_CASE("Beam reflectance of an infinite interface tends to Fresnel as O((lambda / w0)^2)",
          "[validation][fresnel]") {
    const Complex eps_si = material::silicon_500nm().eps_r;
    for (const auto pol : {excitation::Polarization::P, excitation::Polarization::S}) {
        const BeamReflectance r1 = beam_reflectance(*beam(1e-6, 45.0, pol), 1.0, eps_si);
        const BeamReflectance r2 = beam_reflectance(*beam(2e-6, 45.0, pol), 1.0, eps_si);
        const BeamReflectance r4 = beam_reflectance(*beam(4e-6, 45.0, pol), 1.0, eps_si);
        const Real d1 = r1.reflectance / r1.fresnel_central - 1.0;
        const Real d2 = r2.reflectance / r2.fresnel_central - 1.0;
        const Real d4 = r4.reflectance / r4.fresnel_central - 1.0;
        INFO("pol " << (pol == excitation::Polarization::P ? "p" : "s") << ": R_beam / F - 1 = "
                    << d1 << " (w0 = 1 um), " << d2 << " (2 um), " << d4 << " (4 um)");
        // Measured: p -0.63 %, s +0.26 % at w0 = 1 um; second order in lambda / (pi w0).
        CHECK(std::abs(d1) > 1e-3);
        CHECK_THAT(d1 / d2, WithinRel(4.0, 0.05));
        CHECK_THAT(d2 / d4, WithinRel(4.0, 0.05));
        CHECK(std::abs(d4) < 5e-4);
        CHECK(r1.away_share < 1e-15);
    }
    // Normal incidence: |r|^2 is even in the angle and nearly flat for Ag.
    const BeamReflectance ag = beam_reflectance(*beam(1e-6, 0.0, excitation::Polarization::P), 1.0,
                                                material::silver_500nm().eps_r);
    CHECK(std::abs(ag.reflectance / ag.fresnel_central - 1.0) < 1e-5);
    CHECK_THAT(ag.s_share, WithinAbs(0.5, 0.01));
}

TEST_CASE("Beam power equals the numerical flux through z = 0; edge loss of the waist rule",
          "[validation][fresnel]") {
    for (const Real theta : {0.0, 45.0}) {
        // Region radius 5 w0 / cos(theta) (beam_region_radius): the flux square |x|, |y| <= R /
        // sqrt(2) lies in the controlled ball. Spacing lambda / 2.5: the intensity (products of
        // plane waves) is band-limited to transverse wavenumbers <= 2 k sin(alpha_max) < 2 k,
        // below the 2 pi / h = 2.5 k of the grid, so the trapezoid rule is exact up to the
        // truncation of the square (and the O(h^2) edge terms of the patch, where the beam is
        // ~1e-8 of its peak).
        const Real w0 = 1e-6;
        const Real R = 5.0 * w0 / std::cos(theta * kDeg);
        const auto b = beam(w0, theta, excitation::Polarization::P, R);
        const Real L = 4.0 * w0;
        const PlaneFlux f = incident_flux_z0(*b, L, R / std::sqrt(2.0), kLambda / 2.5);
        INFO("theta " << theta << ": flux / power - 1 = " << f.square / b->power() - 1.0
                      << ", edge loss " << 1.0 - f.patch / f.square);
        CHECK_THAT(f.square, WithinRel(b->power(), 1e-8));
        if (theta == 0.0) {
            // ADR 0006 amendment: w0 = L / 4 carries ~0.02 % past the patch edges.
            CHECK(1.0 - f.patch / f.square < 2e-4);
        } else {
            // The 45 deg footprint is 1 / cos(theta) longer: 0.6 % at w0 = L / 4.
            CHECK(1.0 - f.patch / f.square > 3e-3);
        }
    }
    // Paraxial limit of power(): pi w0^2 / (4 eta) (1 + f^2 / 2), f = lambda / (pi w0).
    const Real w0 = 4e-6;
    const Real f = kLambda / (constants::pi * w0);
    const Real eta = material::vacuum()
                         .wave_impedance(beam(w0, 0.0, excitation::Polarization::P)->omega())
                         .real();
    CHECK_THAT(beam(w0, 0.0, excitation::Polarization::P)->power(),
               WithinRel(constants::pi * w0 * w0 / (4.0 * eta) * (1.0 + 0.5 * f * f), 1e-4));
}

TEST_CASE("Hemisphere grid integrates the solid angle", "[validation][fresnel]") {
    for (const Real d : {0.5, 1.0, 5.0}) {
        const HemisphereGrid g = hemisphere_grid(-1.0, d);
        CHECK_THAT(g.weights.sum(), WithinRel(2.0 * constants::pi, 1e-12));
        CHECK(g.dirs.col(2).maxCoeff() < 0.0);
        // int cos^2(theta) dOmega = 2 pi / 3 (polynomial in the direction).
        CHECK_THAT(g.weights.dot(g.dirs.col(2).cwiseAbs2()),
                   WithinRel(2.0 * constants::pi / 3.0, 1e-12));
    }
    CHECK(hemisphere_grid(+1.0, 1.0, 2).dirs.rows() == 45 * 90);
}

TEST_CASE("Public DRC / power API reproduces the support-code reflectance of a flat box",
          "[validation][fresnel][drc_power]") {
    // CI coverage of the acceptance API (WP25 review W2): the flat-box reflectance of the
    // validation-large cases is R_sim = hemisphere_power(...).total / AngularSpectrumBeam::power()
    // (fresnel_flat::run); the public Simulation::reflected_power / incident_power / drc must give
    // the same number. Structural check on a tiny solved flat Si box (L = 0.4 um, uniform 100 nm
    // cells, depth 0.3 um, cheap quadrature, direct LU): the physics (R against Fresnel) is the
    // validation-large case; here only the agreement of the two evaluations.
    const Real h = 100e-9;
    const geometry::HeightMap hm = flat_height_map(0.4e-6, h);
    geometry::TriangleMesh mesh = geometry::make_mesh_from_height_map(hm, 0.3e-6, h);
    Real r_max = 0.0;
    for (Index v = 0; v < mesh.num_vertices(); ++v)
        r_max = std::max(r_max, mesh.vertices().row(v).norm());
    const auto b = beam(0.2e-6, 0.0, excitation::Polarization::P, 1.02 * r_max);
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::silicon_500nm();
    cfg.compression = "dense";
    cfg.solver = SolverKind::Direct;
    cfg.kernels.quad_degree_far = 1;
    cfg.kernels.quad_degree_near = 2;
    cfg.kernels.quad_degree_sing = 2;
    cfg.kernels.outer_grading_levels = 0;
    cfg.kernels.target_accuracy = 0.0;
    Simulation sim(std::move(mesh), b, cfg);
    sim.solve();

    // Support-code evaluation of fresnel_flat::run (grid of a case, Dunavant degree 6).
    const Real eta1 = material::vacuum().wave_impedance(b->omega()).real();
    const HemisphereGrid g = hemisphere_grid(-1.0, 5.0);  // 18 x 36 nodes
    REQUIRE(g.dirs.rows() == 18 * 36);
    const HemispherePower support =
        hemisphere_power(sim.solution(), g, 6, eta1, Vec3(0.0, 0.0, -1.0), 0.1);
    const Real r_support = support.total / b->power();
    REQUIRE(support.total > 0.0);
    INFO("2N = " << sim.num_unknowns() << ", R_support = " << r_support);

    // Same far fields, the same row-by-row summation w |F|^2 / (2 eta1) and bitwise the same
    // eta1 and P_inc: bitwise equal.
    CHECK(sim.incident_power() == b->power());
    const Real p_refl = sim.reflected_power(18, 36, 6);
    CHECK(p_refl == support.total);
    CHECK(sim.radiated_power(g.dirs, g.weights, 6) == support.total);
    CHECK(p_refl / sim.incident_power() == r_support);
    // sum w D: D is scaled by 1 / (2 eta1 P_inc) per row before the summation (another rounding
    // order), so equal to rounding only.
    const VectorXr d = sim.drc(g.dirs, 6);
    CHECK(d.allFinite());
    CHECK(d.minCoeff() >= 0.0);
    CHECK_THAT(g.weights.dot(d), WithinRel(r_support, 1e-13));
}
