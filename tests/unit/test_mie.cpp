#include "specklebem/material/material.hpp"
#include "specklebem/reference/mie.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace specklebem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using reference::MieParams;
using reference::MieSolution;

namespace {

constexpr Real kPi = constants::pi;
constexpr Real kLambda = 500e-9;

/// Sphere of diameter 1 um at lambda = 500 nm in vacuum.
MieParams sphere_d1um(const material::Material& mat, int n_max = 0) {
    return MieParams{0.5e-6, kLambda, mat, material::vacuum(), n_max};
}

material::Material lossless_n15() {
    return {Complex(1.5 * 1.5, 0), Complex(1, 0)};
}

Real omega_of(Real lambda) {
    return 2 * kPi * constants::c0 / lambda;
}

Vec3 direction(Real theta, Real phi) {
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}

/// Incident field of docs/06: E = x exp(-j k1 z), H = y exp(-j k1 z) / eta1 (vacuum here).
Vec3c incident_E(const Vec3& r, Real k1) {
    return {std::exp(Complex(0, -k1 * r.z())), 0, 0};
}
Vec3c incident_H(const Vec3& r, Real k1, Real eta1) {
    return {0, std::exp(Complex(0, -k1 * r.z())) / eta1, 0};
}

/// Central-difference curl of a vector field.
Vec3c curl_fd(const std::function<Vec3c(const Vec3&)>& f, const Vec3& r, Real h) {
    Vec3c d[3];  // d[k] = dF/dx_k
    for (Index k = 0; k < 3; ++k) {
        Vec3 e = Vec3::Zero();
        e(k) = h;
        d[k] = (f(r + e) - f(r - e)) / (2 * h);
    }
    return {d[1](2) - d[2](1), d[2](0) - d[0](2), d[0](1) - d[1](0)};
}

/// Tangential part of a complex vector with respect to the unit normal n.
Vec3c tangential(const Vec3c& v, const Vec3& n) {
    const Vec3c nc = n.cast<Complex>();
    return v - nc * nc.dot(v);  // dot() conjugates the (real) first argument only
}

/// 4 pi R^2 |E_s|^2 at distance R in direction (theta, phi).
Real near_field_rcs(const MieSolution& s, Real theta, Real phi, Real R) {
    return 4 * kPi * R * R * s.scattered_E(R * direction(theta, phi)).squaredNorm();
}

}  // namespace

TEST_CASE("Mie: Bohren-Huffman appendix A sample case", "[reference]") {
    // BH appendix A: m = 1.55, radius 0.525 um, lambda 0.6328 um (x = 5.213), lossless.
    // Q = C / (pi a^2); Q_back = sigma(theta = pi) / (pi a^2) with the radar (4 pi normalised)
    // cross section sigma = (4 pi / k^2) |S1(pi)|^2, i.e. Q_back = 4 |S1(pi)|^2 / x^2, which is
    // the normalisation that reproduces BH's printed value.
    const MieParams p{0.525e-6, 0.6328e-6, {Complex(1.55 * 1.55, 0), Complex(1, 0)}};
    const MieSolution s(p);
    const Real geo = kPi * p.radius * p.radius;
    CHECK_THAT(s.scattering_cross_section() / geo, WithinRel(3.10543, 1e-4));
    CHECK_THAT(s.extinction_cross_section() / geo, WithinRel(3.10543, 1e-4));
    CHECK_THAT(s.bistatic_rcs(kPi, 0.0) / geo, WithinRel(2.92534, 1e-4));
    // Backscattering is polarisation independent: same value for any phi.
    CHECK_THAT(s.bistatic_rcs(kPi, 1.234), WithinRel(s.bistatic_rcs(kPi, 0.0), 1e-12));
}

TEST_CASE("Mie: default n_max and override", "[reference]") {
    const MieParams p{0.525e-6, 0.6328e-6, {Complex(1.55 * 1.55, 0), Complex(1, 0)}};
    const Real x = 2 * kPi * p.radius / p.wavelength;  // 5.2128
    // Default (near-field criterion): n_max = ceil(x + 11 x^(1/3) + 1) = ceil(25.287) = 26.
    const auto expected = static_cast<Index>(std::ceil(x + 11 * std::cbrt(x) + 1));
    REQUIRE(expected == 26);
    // Wiscombe's far-field order: ceil(x + 4 x^(1/3) + 2) = ceil(14.149) = 15.
    const auto n_wiscombe = static_cast<Index>(std::ceil(x + 4 * std::cbrt(x) + 2));
    REQUIRE(n_wiscombe == 15);

    const MieSolution s(p);
    REQUIRE(s.a_n().size() == expected);
    REQUIRE(s.b_n().size() == expected);
    // Coefficients from the Wiscombe order up to n_max (inclusive) are negligible.
    for (Index n = n_wiscombe; n <= expected; ++n) {
        CHECK(std::abs(s.a_n()(n - 1)) < 1e-8);
        CHECK(std::abs(s.b_n()(n - 1)) < 1e-8);
    }

    // Override with the Wiscombe order: shared coefficients and cross sections agree.
    MieParams p_w = p;
    p_w.n_max = static_cast<int>(n_wiscombe);
    const MieSolution s_w(p_w);
    REQUIRE(s_w.a_n().size() == n_wiscombe);
    REQUIRE(s_w.b_n().size() == n_wiscombe);
    for (Index n = 0; n < n_wiscombe; ++n) {
        CHECK(std::abs(s_w.a_n()(n) - s.a_n()(n)) <= 1e-12 * std::abs(s.a_n()(0)));
        CHECK(std::abs(s_w.b_n()(n) - s.b_n()(n)) <= 1e-12 * std::abs(s.a_n()(0)));
    }
    CHECK_THAT(s_w.scattering_cross_section(), WithinRel(s.scattering_cross_section(), 1e-10));
    CHECK_THAT(s_w.extinction_cross_section(), WithinRel(s.extinction_cross_section(), 1e-10));
    CHECK_THAT(s_w.bistatic_rcs(kPi, 0.0), WithinRel(s.bistatic_rcs(kPi, 0.0), 1e-10));

    MieParams p_small = p;
    p_small.n_max = 4;
    CHECK(MieSolution(p_small).a_n().size() == 4);
}

TEST_CASE("Mie: large lossless sphere x = 100 (recurrence start indices)", "[reference]") {
    // Regression for the D_n / Miller start indices at large |m x| with no absorption.
    // Reference values from an independent 40-digit mpmath implementation (WP5 review).
    const Real x = 100;
    const MieParams p{x * kLambda / (2 * kPi), kLambda, {Complex(1.5 * 1.5, 0), Complex(1, 0)}};
    const MieSolution s(p);
    const Real geo = kPi * p.radius * p.radius;
    CHECK_THAT(s.extinction_cross_section() / geo, WithinRel(2.0943878, 1e-6));
    CHECK_THAT(s.scattering_cross_section() / geo, WithinRel(2.0943878, 1e-6));
    CHECK_THAT(s.bistatic_rcs(kPi, 0.0) / geo, WithinRel(1.7361931, 1e-6));
}

TEST_CASE("Mie: overflow is reported, never returned as non-finite values", "[reference]") {
    // Absurd truncation orders: y_n(x) exceeds the double range long before n = 2000 for
    // x = 6.28. The constructor must either succeed with finite coefficients and fields or
    // throw std::overflow_error.
    const material::Material strong_absorber{Complex(-100, -50), Complex(1, 0)};
    for (const auto& mat : {material::silver_500nm(), strong_absorber, lossless_n15()}) {
        for (const int n_max : {60, 200, 2000}) {
            const MieParams p = sphere_d1um(mat, n_max);
            try {
                const MieSolution s(p);
                CHECK(s.a_n().allFinite());
                CHECK(s.b_n().allFinite());
                CHECK(s.scattered_E(Vec3(0, 0.3e-6, 0.6e-6)).allFinite());
                CHECK(s.scattered_H(Vec3(0, 0.3e-6, 0.6e-6)).allFinite());
                CHECK(s.internal_E(Vec3(0.2e-6, 0, -0.3e-6)).allFinite());
                CHECK(std::isfinite(s.bistatic_rcs(1.0, 0.5)));
                CHECK(std::isfinite(s.extinction_cross_section()));
            } catch (const std::overflow_error&) {
                SUCCEED("overflow reported");
            }
        }
    }
    // n_max = 2000 at x = 6.28 cannot be represented: y_2000(6.28) ~ 3999!! / 6.28^2001.
    CHECK_THROWS_AS(MieSolution(sphere_d1um(material::silver_500nm(), 2000)), std::overflow_error);
}

TEST_CASE("Mie: Rayleigh limit with absorption (Ag)", "[reference]") {
    const Real x = 0.01;
    const Real a = x * kLambda / (2 * kPi);
    const Real k = 2 * kPi / kLambda;
    const auto ag = material::silver_500nm();
    const MieSolution s(MieParams{a, kLambda, ag});
    const Complex alpha = (ag.eps_r - Real(1)) / (ag.eps_r + Real(2));
    const Real csca_rayleigh = 8 * kPi / 3 * std::pow(k, 4) * std::pow(a, 6) * std::norm(alpha);
    // exp(+jwt): Im(eps_r) < 0  =>  Im(alpha) < 0  =>  C_abs > 0.
    const Real cabs_rayleigh = 4 * kPi * k * a * a * a * (-alpha.imag());
    const Real cabs = s.extinction_cross_section() - s.scattering_cross_section();
    CHECK_THAT(s.scattering_cross_section(), WithinRel(csca_rayleigh, 1e-3));
    CHECK_THAT(cabs, WithinRel(cabs_rayleigh, 1e-3));
    CHECK(cabs > 0);
}

TEST_CASE("Mie: power balance", "[reference]") {
    for (const auto& mat : {material::silver_500nm(), material::silicon_500nm()}) {
        const MieSolution s(sphere_d1um(mat));
        const Real cext = s.extinction_cross_section();
        const Real csca = s.scattering_cross_section();
        CHECK(cext >= csca);
        CHECK(cext - csca > 0);
    }
    const MieSolution s(sphere_d1um(lossless_n15()));
    const Real cext = s.extinction_cross_section();
    CHECK(std::abs(cext - s.scattering_cross_section()) / cext < 1e-12);
}

TEST_CASE("Mie: optical theorem", "[reference]") {
    for (const auto& mat : {material::silver_500nm(), lossless_n15()}) {
        const MieSolution s(sphere_d1um(mat));
        const Real k = 2 * kPi / kLambda;
        // Forward amplitude in our convention: S(0) = 1/2 sum (2n+1)(a_n + b_n) with the
        // conjugated coefficients (= conj of BH's S(0); Re S(0) is convention independent).
        Complex s0(0, 0);
        for (Index n = 1; n <= s.a_n().size(); ++n) {
            s0 += Real(0.5) * static_cast<Real>(2 * n + 1) * (s.a_n()(n - 1) + s.b_n()(n - 1));
        }
        // Self-consistency only (algebraically the same sum as extinction_cross_section()); the
        // independent optical-theorem check is the near-field forward amplitude below.
        CHECK_THAT(s.extinction_cross_section(), WithinRel(4 * kPi / (k * k) * s0.real(), 1e-12));

        // Same amplitude from the exact scattered field on the forward axis. For exp(+jwt),
        // E_s -> x_hat S(0) exp(-jkr) / (jkr), hence S(0) ~ E_x(r) * jkr * exp(+jkr).
        // The exact series at r = 1e4 lambda still carries the O(n^2/(kr)) ~ 1e-4 near-field
        // terms of h_n(kr); they are removed by Richardson extrapolation with r and 2r
        // (residual O((n^2/(kr))^2)).
        const auto amplitude = [&](Real r) {
            const Vec3c e = s.scattered_E(Vec3(0, 0, r));
            CHECK(std::abs(e.y()) <= 1e-12 * std::abs(e.x()));
            CHECK(std::abs(e.z()) <= 1e-12 * std::abs(e.x()));
            return e.x() * Complex(0, k * r) * std::exp(Complex(0, k * r));
        };
        const Real r = 1e4 * kLambda;
        const Complex s_r = amplitude(r);
        const Complex s_extrap = Real(2) * amplitude(2 * r) - s_r;
        CHECK(std::abs(s_r - s0) / std::abs(s0) < 1e-3);  // raw value: O(1/(kr)) away
        CHECK(std::abs(s_extrap - s0) / std::abs(s0) < 1e-5);
        CHECK_THAT(4 * kPi / (k * k) * s_extrap.real(),
                   WithinRel(s.extinction_cross_section(), 1e-5));
    }
}

TEST_CASE("Mie: bistatic RCS agrees with the exact scattered near field", "[reference]") {
    const MieSolution s(sphere_d1um(material::silver_500nm()));
    const Real r = 1e4 * kLambda;
    const std::vector<std::pair<Real, Real>> dirs = {{0.0, 0.0}, {0.7, 0.4},  {1.3, kPi / 2},
                                                     {1.9, 2.3}, {2.6, -1.2}, {kPi, 0.0}};
    for (const auto& [theta, phi] : dirs) {
        const Real sigma = s.bistatic_rcs(theta, phi);
        const Real f_r = near_field_rcs(s, theta, phi, r);
        const Real f_2r = near_field_rcs(s, theta, phi, 2 * r);
        // f(r) = sigma + O(1/r) exactly (h_n is a polynomial in 1/(kr) times exp(ikr)/(kr)).
        // At r = 1e4 lambda the O(1/r) term is 5e-5 .. 3e-3 relative (largest in the weak
        // backscattering direction); Richardson extrapolation with r and 2r removes it.
        CHECK(std::abs(f_r - sigma) / sigma < 1e-2);
        CHECK_THAT(2 * f_2r - f_r, WithinRel(sigma, 1e-5));
    }
}

TEST_CASE("Mie: boundary conditions on the sphere surface", "[reference]") {
    // The exact incident plane wave is compared with the truncated scattered and internal
    // series, so this also checks the default (near-field) n_max = 28 for x = 6.28: with
    // Wiscombe's far-field order (16) the truncation alone would leave ~5e-6 at |r| = a.
    const Real k = 2 * kPi / kLambda;
    const Real eta1 = constants::eta0;
    const Real omega = omega_of(kLambda);
    for (const auto& mat : {material::silver_500nm(), lossless_n15()}) {
        const MieParams p = sphere_d1um(mat);
        const MieSolution s(p);
        REQUIRE(s.a_n().size() == 28);
        const Real a = p.radius;
        const Real h = 1e-4 * a;
        const auto e_int = [&](const Vec3& r) { return s.internal_E(r); };
        const auto h_int_fd = [&](const Vec3& r) {
            return Vec3c(Complex(0, 1) / (omega * constants::mu0) * curl_fd(e_int, r, h));
        };
        const std::vector<std::pair<Real, Real>> dirs = {
            {0.0, 0.0},     {kPi, 0.0},         {0.4, 0.3},  {1.2, 2.0},
            {kPi / 2, 0.0}, {kPi / 2, kPi / 2}, {2.2, -0.8}, {2.9, 4.0}};
        for (const auto& [theta, phi] : dirs) {
            const Vec3 n = direction(theta, phi);
            const Vec3 r = a * n;
            const Vec3c e_out = incident_E(r, k) + s.scattered_E(r);
            const Vec3c e_in = s.internal_E(r);
            CHECK(tangential(e_out - e_in, n).norm() < 1e-6);  // |E_inc| = 1 V/m
            // Normal D: eps1 (E_inc + E_s).n = eps2 E_int.n (eps1 = 1).
            const Vec3c nc = n.cast<Complex>();
            CHECK(std::abs(nc.dot(e_out) - p.sphere.eps_r * nc.dot(e_in)) < 1e-6);

            // Tangential H: internal H from central differences of curl E_int, evaluated at
            // a - h and a - 2h (stencils stay inside the sphere) and extrapolated linearly to
            // the surface (error O((k |m| h)^2)).
            const Vec3c h_out = incident_H(r, k, eta1) + s.scattered_H(r);
            const Vec3c h_in = Real(2) * h_int_fd((a - h) * n) - h_int_fd((a - 2 * h) * n);
            CHECK(tangential(h_out - h_in, n).norm() < 1e-4 / eta1);  // |H_inc| = 1/eta1
        }
    }
}

TEST_CASE("Mie: scattered H is the curl of scattered E", "[reference]") {
    const MieSolution s(sphere_d1um(material::silver_500nm()));
    const Real omega = omega_of(kLambda);
    const auto e_s = [&](const Vec3& r) { return s.scattered_E(r); };
    for (const Vec3& r : {Vec3(0.3e-6, -0.2e-6, 0.7e-6), Vec3(-1.0e-6, 0.4e-6, -0.6e-6)}) {
        const Vec3c h_fd =
            Complex(0, 1) / (omega * constants::mu0) * curl_fd(e_s, r, 1e-4 * 0.5e-6);
        const Vec3c h_s = s.scattered_H(r);
        CHECK((h_fd - h_s).norm() < 1e-6 * h_s.norm());
    }
}

TEST_CASE("Mie: exp(+jwt) phase and decay conventions", "[reference]") {
    const auto ag = material::silver_500nm();
    const MieParams p = sphere_d1um(ag);
    const MieSolution s(p);
    const Real k = 2 * kPi / kLambda;

    // Outgoing wave exp(-jkr): along +z the phase decreases by k * delta.
    const Real r = 1e4 * kLambda;
    const Real delta = kLambda / 8;
    const Complex e1 = s.scattered_E(Vec3(0, 0, r)).x();
    const Complex e2 = s.scattered_E(Vec3(0, 0, r + delta)).x();
    const Real dphase = std::arg(e2 / e1);
    CHECK(dphase < 0);
    CHECK_THAT(dphase, WithinAbs(-k * delta, 1e-3));

    // Lossy sphere: Im(k2) <= 0 for exp(+jwt).
    CHECK(ag.wavenumber(omega_of(kLambda)).imag() < 0);

    // Skin effect: the internal field decays from the lit side (-z) into the Ag sphere.
    const Real a = p.radius;
    const Real e09 = s.internal_E(Vec3(0, 0, -0.9 * a)).norm();
    const Real e05 = s.internal_E(Vec3(0, 0, -0.5 * a)).norm();
    CHECK(e09 > e05);
}

TEST_CASE("Mie: fields are regular on the axis and at the origin", "[reference]") {
    for (const auto& mat : {material::silver_500nm(), lossless_n15()}) {
        const MieParams p = sphere_d1um(mat);
        const MieSolution s(p);
        const Real a = p.radius;
        for (const Real z : {a, -a, 2 * a, -2 * a}) {
            const Vec3c es = s.scattered_E(Vec3(0, 0, z));
            const Vec3c hs = s.scattered_H(Vec3(0, 0, z));
            CHECK(std::isfinite(es.norm()));
            CHECK(std::isfinite(hs.norm()));
            CHECK(es.norm() > 0);
            // Axial symmetry: on the z axis E_s is along x and H_s along y.
            CHECK(std::abs(es.y()) + std::abs(es.z()) <= 1e-12 * es.norm());
            CHECK(std::abs(hs.x()) + std::abs(hs.z()) <= 1e-12 * hs.norm());
        }
        for (const Real z : {0.0, 0.5 * a, -0.5 * a}) {
            const Vec3c ei = s.internal_E(Vec3(0, 0, z));
            CHECK(std::isfinite(ei.norm()));
            CHECK(std::abs(ei.y()) + std::abs(ei.z()) <= 1e-12 * ei.norm());
        }
        // The origin limit agrees with a neighbouring point.
        const Vec3c e0 = s.internal_E(Vec3::Zero());
        const Vec3c e_eps = s.internal_E(Vec3(1e-9 * a, 0, 0));
        CHECK((e0 - e_eps).norm() < 1e-6 * e0.norm());
    }
}

TEST_CASE("Mie: invalid input is rejected", "[reference]") {
    const auto ag = material::silver_500nm();
    CHECK_THROWS_AS(MieSolution(MieParams{0.0, kLambda, ag}), std::invalid_argument);
    CHECK_THROWS_AS(MieSolution(MieParams{0.5e-6, -1.0, ag}), std::invalid_argument);
    CHECK_THROWS_AS(MieSolution(MieParams{0.5e-6, kLambda, ag, material::vacuum(), -1}),
                    std::invalid_argument);
    // Lossy or magnetic background, magnetic sphere.
    CHECK_THROWS_AS(MieSolution(MieParams{0.5e-6, kLambda, ag, {Complex(1.5, -0.01), 1.0}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(MieSolution(MieParams{0.5e-6, kLambda, ag, {Complex(1.5, 0), 2.0}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(MieSolution(MieParams{0.5e-6, kLambda, {Complex(2.25, 0), 1.5}}),
                    std::invalid_argument);

    const MieSolution s(sphere_d1um(ag));
    CHECK_THROWS_AS(s.scattered_E(Vec3(0.2e-6, 0, 0)), std::domain_error);
    CHECK_THROWS_AS(s.scattered_H(Vec3(0, 0.2e-6, 0)), std::domain_error);
    CHECK_THROWS_AS(s.internal_E(Vec3(0, 0, 0.6e-6)), std::domain_error);
    CHECK_NOTHROW(s.scattered_E(Vec3(0, 0, 0.5e-6)));
    CHECK_NOTHROW(s.internal_E(Vec3(0, 0, 0.5e-6)));
}

TEST_CASE("Mie: surface tolerance band of the field evaluators", "[reference]") {
    const MieParams p = sphere_d1um(material::silver_500nm());
    const MieSolution s(p);
    const Real a = p.radius;
    const Vec3 n = direction(1.1, 0.7);
    // Within a relative 1e-10 of the surface both expansions are accepted.
    for (const Real f : {1 - 5e-11, 1 + 5e-11}) {
        CHECK_NOTHROW(s.scattered_E(f * a * n));
        CHECK_NOTHROW(s.scattered_H(f * a * n));
        CHECK_NOTHROW(s.internal_E(f * a * n));
    }
    // Beyond it, the wrong side throws while the right side is accepted.
    CHECK_THROWS_AS(s.scattered_E((1 - 1e-9) * a * n), std::domain_error);
    CHECK_THROWS_AS(s.scattered_H((1 - 1e-9) * a * n), std::domain_error);
    CHECK_THROWS_AS(s.internal_E((1 + 1e-9) * a * n), std::domain_error);
    CHECK_NOTHROW(s.internal_E((1 - 1e-9) * a * n));
    CHECK_NOTHROW(s.scattered_E((1 + 1e-9) * a * n));
}

TEST_CASE("Mie: a non-vacuum lossless background medium", "[reference]") {
    // Scaling law: a sphere of index n2 in a medium n1 at vacuum wavelength lambda behaves like
    // a sphere of index n2/n1 in vacuum at wavelength lambda/n1.
    const Real n1 = 1.33;
    const material::Material water{Complex(n1 * n1, 0), Complex(1, 0)};
    const auto si = material::silicon_500nm();
    const MieSolution s_med(MieParams{0.3e-6, kLambda, si, water});
    const material::Material si_rel{si.eps_r / (n1 * n1), Complex(1, 0)};
    const MieSolution s_vac(MieParams{0.3e-6, kLambda / n1, si_rel});
    CHECK_THAT(s_med.scattering_cross_section(),
               WithinRel(s_vac.scattering_cross_section(), 1e-12));
    CHECK_THAT(s_med.extinction_cross_section(),
               WithinRel(s_vac.extinction_cross_section(), 1e-12));
    CHECK_THAT(s_med.bistatic_rcs(1.0, 0.3), WithinRel(s_vac.bistatic_rcs(1.0, 0.3), 1e-12));
}
