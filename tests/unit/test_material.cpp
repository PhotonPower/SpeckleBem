#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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
    CHECK_THAT(material::field_decay_length(Complex(18.478, 0.606), lambda),
               WithinRel(d_si, 1e-14));
    CHECK_THAT(material::field_decay_length(Complex(18.478, -0.606), 2.0 * lambda),
               WithinRel(2.0 * d_si, 1e-14));
    // Lossless dielectric: no decay. Lossless metal (eps_r = -4): evanescent, n = -2j,
    // delta = lambda / (4 pi).
    CHECK(material::field_decay_length(Complex(2.25, 0.0), lambda) ==
          std::numeric_limits<Real>::infinity());
    CHECK_THAT(material::field_decay_length(Complex(-4.0, 0.0), lambda),
               WithinRel(lambda / (4.0 * constants::pi), 1e-14));
    // Material overload: n = sqrt(eps_r mu_r); equal to the eps_r form for mu_r = 1.
    CHECK_THAT(material::field_decay_length(material::silicon_500nm(), lambda),
               WithinRel(d_si, 1e-14));
    CHECK_THAT(material::field_decay_length(material::silver_500nm(), lambda),
               WithinRel(d_ag, 1e-14));
    // eps_r = 2.25 - 0.1j, mu_r = 4: n = 2 sqrt(eps_r), so delta halves.
    const material::Material magnetic{Complex(2.25, -0.1), Complex(4.0, 0.0)};
    CHECK_THAT(material::field_decay_length(magnetic, lambda),
               WithinRel(0.5 * material::field_decay_length(Complex(2.25, -0.1), lambda), 1e-12));
    CHECK(material::field_decay_length(material::vacuum(), lambda) ==
          std::numeric_limits<Real>::infinity());
    CHECK_THROWS_AS(material::field_decay_length(
                        material::Material{Complex(1.0, 0.0),
                                           Complex(std::numeric_limits<Real>::quiet_NaN(), 0.0)},
                        lambda),
                    std::invalid_argument);
    CHECK_THROWS_AS(material::field_decay_length(material::silicon_500nm(), -lambda),
                    std::invalid_argument);
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

namespace {

material::DispersiveMaterial make_table(VectorXr lambda, VectorXc n) {
    return material::DispersiveMaterial(std::move(lambda), std::move(n));
}

}  // namespace

TEST_CASE("dispersive material interpolates n with inclusive endpoints", "[material]") {
    const VectorXr lambda = (VectorXr(3) << 400e-9, 500e-9, 600e-9).finished();
    const VectorXc n =
        (VectorXc(3) << Complex(1.5, -0.1), Complex(2.0, -0.2), Complex(2.5, 0.0)).finished();
    const auto disp = make_table(lambda, n);
    const Complex n450 = 0.5 * (n(0) + n(1));
    CHECK(std::abs(disp.at_wavelength(450e-9).eps_r - n450 * n450) < 1e-14);
    CHECK(disp.at_wavelength(450e-9).mu_r == Complex(1, 0));
    // lambda_min, an interior node and lambda_max are valid and give the tabulated n^2.
    for (Index i = 0; i < 3; ++i) {
        CAPTURE(i);
        CHECK(std::abs(disp.at_wavelength(lambda(i)).eps_r - n(i) * n(i)) < 1e-14);
    }
    CHECK_THROWS_AS(disp.at_wavelength(std::nextafter(400e-9, 0.0)), std::out_of_range);
    CHECK_THROWS_AS(disp.at_wavelength(600.0001e-9), std::out_of_range);
    const Real inf = std::numeric_limits<Real>::infinity();
    for (const Real bad : {std::numeric_limits<Real>::quiet_NaN(), inf, -inf}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(disp.at_wavelength(bad), std::invalid_argument);
    }
}

TEST_CASE("dispersive material rejects invalid tables", "[material]") {
    const VectorXc n2 = VectorXc::Constant(2, Complex(1.5, -0.01));
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    CHECK_THROWS_AS(make_table(VectorXr::Constant(1, 500e-9), VectorXc::Ones(1)),
                    std::invalid_argument);  // fewer than 2 samples
    CHECK_THROWS_AS(make_table((VectorXr(3) << 4e-7, 5e-7, 6e-7).finished(), n2),
                    std::invalid_argument);  // size mismatch
    const VectorXr bad_lambdas[] = {
        (VectorXr(2) << 5e-7, 4e-7).finished(),  // decreasing
        (VectorXr(2) << 5e-7, 5e-7).finished(),  // repeated
        (VectorXr(2) << 0.0, 5e-7).finished(),  (VectorXr(2) << -4e-7, 5e-7).finished(),
        (VectorXr(2) << nan, 5e-7).finished(),  (VectorXr(2) << 4e-7, inf).finished()};
    for (const VectorXr& lambda : bad_lambdas) {
        CAPTURE(lambda.transpose());
        CHECK_THROWS_AS(make_table(lambda, n2), std::invalid_argument);
    }
    const VectorXr lambda = (VectorXr(2) << 4e-7, 5e-7).finished();
    // Messages print wavelengths in general format, not as 0.000000.
    CHECK_THROWS_WITH(make_table(lambda, n2).at_wavelength(7e-7),
                      Catch::Matchers::ContainsSubstring("7e-07") &&
                          Catch::Matchers::ContainsSubstring("[4e-07, 5e-07]"));
    CHECK_THROWS_AS(make_table(lambda, (VectorXc(2) << Complex(nan, 0.0), 1.0).finished()),
                    std::invalid_argument);
    // n + ik optics data: the message names the exp(+jwt) form and the conjugation.
    const VectorXc n_optics = (VectorXc(2) << Complex(1.5, 0.1), Complex(0.05, 3.0)).finished();
    CHECK_THROWS_AS(make_table(lambda, n_optics), std::invalid_argument);
    CHECK_THROWS_WITH(make_table(lambda, n_optics),
                      Catch::Matchers::ContainsSubstring("n - jk") &&
                          Catch::Matchers::ContainsSubstring("conjugated"));
    CHECK_NOTHROW(make_table(lambda, n_optics.conjugate()));
}
