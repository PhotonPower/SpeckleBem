#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace specklebem;
using Catch::Matchers::WithinRel;

TEST_CASE("reference materials match Fu et al. 2023", "[material]") {
    const auto ag = material::silver_500nm();
    CHECK_THAT(ag.eps_r.real(), WithinRel(-9.794, 1e-12));
    CHECK_THAT(ag.eps_r.imag(), WithinRel(-0.313, 1e-12));
    const auto si = material::silicon_500nm();
    CHECK_THAT(si.eps_r.real(), WithinRel(18.478, 1e-12));
}

TEST_CASE("refractive index branch gives decaying waves for exp(+jwt)", "[material]") {
    const auto ag = material::silver_500nm();
    CHECK(ag.refractive_index().imag() <= 0.0);
    const auto si = material::silicon_500nm();
    CHECK(si.refractive_index().imag() <= 0.0);
    CHECK(si.refractive_index().real() > 4.0);
}

TEST_CASE("vacuum wavenumber", "[material]") {
    const Real lambda = 500e-9;
    const Real omega = 2 * constants::pi * constants::c0 / lambda;
    CHECK_THAT(material::vacuum().wavenumber(omega).real(),
               WithinRel(2 * constants::pi / lambda, 1e-12));
}

TEST_CASE("field decay length", "[material]") {
    const Real lambda = 500e-9;
    // Si at 500 nm: sqrt(18.478 - 0.606j) = 4.299182 - 0.0704785j -> delta = 1.129102 um.
    const Real d_si = material::field_decay_length(material::silicon_500nm().eps_r, lambda);
    CHECK_THAT(d_si, WithinRel(1.1291024613727742e-06, 1e-12));
    // Ag at 500 nm: sqrt(-9.794 - 0.313j) = 0.0500010 - 3.129936j -> delta = 25.42 nm.
    const Real d_ag = material::field_decay_length(material::silver_500nm().eps_r, lambda);
    CHECK_THAT(d_ag, WithinRel(2.5424631222083637e-08, 1e-12));
    // Only |Im| enters; delta scales with the wavelength.
    CHECK_THAT(material::field_decay_length(Complex(18.478, 0.606), lambda), WithinRel(d_si, 1e-14));
    CHECK_THAT(material::field_decay_length(Complex(18.478, -0.606), 2.0 * lambda),
               WithinRel(2.0 * d_si, 1e-14));
    // Lossless media: no decay.
    CHECK(material::field_decay_length(Complex(2.25, 0.0), lambda) ==
          std::numeric_limits<Real>::infinity());
    // Invalid input.
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    CHECK_THROWS_AS(material::field_decay_length(Complex(nan, 0.0), lambda), std::invalid_argument);
    CHECK_THROWS_AS(material::field_decay_length(Complex(1.0, inf), lambda), std::invalid_argument);
    for (const Real bad : {0.0, -lambda, nan, inf}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(material::field_decay_length(Complex(4.0, -0.1), bad),
                        std::invalid_argument);
    }
}
