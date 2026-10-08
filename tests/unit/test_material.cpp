#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
