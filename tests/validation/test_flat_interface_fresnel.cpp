// Reference of the flat-interface validation (WP22c, docs/05 row "Flat box, tapered beam, 0 deg
// and 45 deg | Fresnel r_p, r_s"): plane-wave Fresnel coefficients in the exp(+jwt) convention,
// the beam reflectance R_beam of the rigorous angular-spectrum beam on an infinite interface, the
// beam power against the numerical flux through z = 0 and the edge loss of the ADR 0006 waist
// rule (tests/support/fresnel_flat_support.hpp). The simulated flat box (MLFMM, 2N >= 7 10^4) is
// the validation-large case tests/validation_large/test_flat_interface_fresnel_large.cpp;
// results: benchmarks/results/fresnel_flat.md.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>

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
        // sqrt(2) lies in the controlled ball. lambda / 4 spacing: the intensity is band-limited
        // to 2 k sin(alpha_max) < 4 pi / (lambda / 4), so the trapezoid rule is exact up to the
        // truncation of the square.
        const Real w0 = 1e-6;
        const Real R = 5.0 * w0 / std::cos(theta * kDeg);
        const auto b = beam(w0, theta, excitation::Polarization::P, R);
        const Real L = 4.0 * w0;
        const PlaneFlux f = incident_flux_z0(*b, L, R / std::sqrt(2.0), kLambda / 4.0);
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
    for (const Real d : {0.5, 1.0, 2.0}) {
        const HemisphereGrid g = hemisphere_grid(-1.0, d, 2.0 * d);
        CHECK_THAT(g.weights.sum(), WithinRel(2.0 * constants::pi, 2e-4 * d * d));
        CHECK(g.dirs.col(2).maxCoeff() < 0.0);
    }
}
