#include "specklebem/material/material.hpp"
#include "specklebem/reference/mie.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

// Validation of the Mie reference solution itself (docs/05_validation.md: Mie is the root of
// the oracle chain). Nothing is written to disk.

using namespace specklebem;
using Catch::Matchers::WithinRel;
using reference::MieParams;
using reference::MieSolution;

namespace {
constexpr Real kPi = constants::pi;
constexpr Real kLambda = 500e-9;
}  // namespace

TEST_CASE("Mie validation: Ag sphere d = 1 um bistatic RCS in the xz and yz planes",
          "[validation][reference]") {
    const MieSolution s(MieParams{0.5e-6, kLambda, material::silver_500nm()});
    const Real geo = kPi * 0.5e-6 * 0.5e-6;
    for (const Real phi : {0.0, kPi / 2}) {
        for (int i = 0; i <= 180; ++i) {
            const Real theta = kPi * static_cast<Real>(i) / 180;
            const Real sigma = s.bistatic_rcs(theta, phi);
            REQUIRE(std::isfinite(sigma));
            CHECK(sigma > 0);
            // Mirror symmetry of the x-polarised problem: phi -> phi + pi.
            CHECK_THAT(s.bistatic_rcs(theta, phi + kPi), WithinRel(sigma, 1e-12));
            // Not a general law, but true for this sphere (x = 6.28): the diffraction-dominated
            // forward peak is the maximum of the pattern.
            CHECK(sigma <= s.bistatic_rcs(0.0, 0.0) * (1 + 1e-12));
        }
    }
    // Forward peak relative to the geometric cross section (|S(0)|^2 >= (Re S(0))^2 bound
    // from the optical theorem: sigma(0) >= k^2 C_ext^2 / (4 pi)).
    const Real k = 2 * kPi / kLambda;
    const Real cext = s.extinction_cross_section();
    CHECK(s.bistatic_rcs(0.0, 0.0) >= k * k * cext * cext / (4 * kPi) * (1 - 1e-12));
    CHECK(cext > geo);
    // Forward and backward scattering do not depend on the polarisation plane.
    CHECK_THAT(s.bistatic_rcs(0.0, kPi / 2), WithinRel(s.bistatic_rcs(0.0, 0.0), 1e-12));
    CHECK_THAT(s.bistatic_rcs(kPi, kPi / 2), WithinRel(s.bistatic_rcs(kPi, 0.0), 1e-12));
    // Power balance for the lossy sphere.
    CHECK(s.extinction_cross_section() > s.scattering_cross_section());
}

TEST_CASE("Mie validation: Rayleigh limit for Ag", "[validation][reference]") {
    const Real x = 0.01;
    const Real a = x * kLambda / (2 * kPi);
    const Real k = 2 * kPi / kLambda;
    const auto ag = material::silver_500nm();
    const MieSolution s(MieParams{a, kLambda, ag});
    const Complex alpha = (ag.eps_r - Real(1)) / (ag.eps_r + Real(2));
    CHECK_THAT(s.scattering_cross_section(),
               WithinRel(8 * kPi / 3 * std::pow(k, 4) * std::pow(a, 6) * std::norm(alpha), 1e-3));
    CHECK_THAT(s.extinction_cross_section() - s.scattering_cross_section(),
               WithinRel(4 * kPi * k * a * a * a * (-alpha.imag()), 1e-3));
}

TEST_CASE("Mie validation: Bohren-Huffman appendix A sample case", "[validation][reference]") {
    const MieParams p{0.525e-6, 0.6328e-6, {Complex(1.55 * 1.55, 0), Complex(1, 0)}};
    const MieSolution s(p);
    const Real geo = kPi * p.radius * p.radius;
    CHECK_THAT(s.scattering_cross_section() / geo, WithinRel(3.10543, 1e-4));
    CHECK_THAT(s.extinction_cross_section() / geo, WithinRel(3.10543, 1e-4));
    CHECK_THAT(s.bistatic_rcs(kPi, 0.0) / geo, WithinRel(2.92534, 1e-4));
}
