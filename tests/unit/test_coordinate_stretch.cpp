#include "specklebem/geometry/coordinate_stretch.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

using namespace specklebem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using geometry::CoordinateStretch;
using geometry::StretchProfile;

namespace {

// PML of the ADR 0009 defaults at lambda_1 = 500 nm: start 3 um, width 2 lambda_1, m = 2.
constexpr Real kLambda = 500e-9;
constexpr Real kStart = 3e-6;
constexpr Real kWidth = 2.0 * kLambda;
constexpr Real kSigma = 4.0;

StretchProfile default_profile(Real order = 2.0) {
    return {kStart, kWidth, kSigma, order};
}

bool bit_equal(Real a, Real b) {
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

}  // namespace

TEST_CASE("CoordinateStretch: identity inside the start line", "[coordinate_stretch]") {
    // Asymmetric profiles on x and y, so a swap of the axes would show.
    const CoordinateStretch s(default_profile(), StretchProfile{2e-6, 1e-6, 3.0, 1.0});
    REQUIRE_FALSE(s.is_identity());
    std::mt19937_64 rng(20261013);
    std::uniform_real_distribution<Real> ux(-kStart, kStart);
    std::uniform_real_distribution<Real> uy(-2e-6, 2e-6);
    std::uniform_real_distribution<Real> uz(-1e-6, 1e-6);
    for (int k = 0; k < 200; ++k) {
        const Vec3 r(ux(rng), uy(rng), uz(rng));
        const Vec3c rt = s.map(r);
        for (int c = 0; c < 3; ++c) {
            CHECK(bit_equal(rt(c).real(), r(c)));
            CHECK(rt(c).imag() == 0.0);
        }
        const Vec3c jd = s.jacobian_diag(r);
        for (int c = 0; c < 3; ++c) {
            CHECK(jd(c) == Complex(1.0, 0.0));
        }
        CHECK_FALSE(s.in_pml(r));
    }
    // The start lines themselves are unstretched; just beyond them the point is in the PML.
    CHECK_FALSE(s.in_pml(Vec3(kStart, -2e-6, 0.0)));
    CHECK(s.in_pml(Vec3(std::nextafter(kStart, 1.0), 0.0, 0.0)));
    CHECK(s.in_pml(Vec3(0.0, -std::nextafter(2e-6, 1.0), 0.0)));
    // z is never stretched.
    const Vec3c far = s.map(Vec3(5e-6, -4e-6, 7e-7));
    CHECK(bit_equal(far(2).real(), 7e-7));
    CHECK(far(2).imag() == 0.0);
    CHECK(s.jacobian_diag(Vec3(5e-6, -4e-6, 7e-7))(2) == Complex(1.0, 0.0));
}

TEST_CASE("CoordinateStretch: analytic integral at the PML end, mirrored sign",
          "[coordinate_stretch]") {
    for (const Real m : {0.0, 1.0, 2.0, 3.0, 2.5}) {
        const CoordinateStretch s(default_profile(m), default_profile(m));
        const Real expected = -kSigma * kWidth / (m + 1.0);
        const Real end = kStart + kWidth;
        // x > 0: Im x~ = -sigma_max W / (m + 1); x < 0: the opposite sign.
        const Vec3c pos = s.map(Vec3(end, 0.5e-6, 0.0));
        const Vec3c neg = s.map(Vec3(-end, -0.5e-6, 0.0));
        CHECK_THAT(pos(0).imag(), WithinRel(expected, 1e-13));
        CHECK_THAT(neg(0).imag(), WithinRel(-expected, 1e-13));
        CHECK(bit_equal(pos(0).real(), end));
        CHECK(bit_equal(neg(0).real(), -end));
        // Same along y.
        const Vec3c py = s.map(Vec3(0.0, end, 0.0));
        const Vec3c ny = s.map(Vec3(0.0, -end, 0.0));
        CHECK_THAT(py(1).imag(), WithinRel(expected, 1e-13));
        CHECK_THAT(ny(1).imag(), WithinRel(-expected, 1e-13));
        // Odd symmetry: x~(-x) = -x~(x) (Re and Im both flip).
        for (const Real x : {kStart + 0.1 * kWidth, kStart + 0.7 * kWidth, kStart + 1.5 * kWidth}) {
            const Complex a = s.map(Vec3(x, 0.0, 0.0))(0);
            const Complex b = s.map(Vec3(-x, 0.0, 0.0))(0);
            CHECK(a == -b);
            CHECK(a.imag() < 0.0);
        }
        // Midpoint value Int_0^{W/2} sigma = sigma_max W (1/2)^(m+1) / (m + 1).
        const Real mid = s.map(Vec3(kStart + 0.5 * kWidth, 0.0, 0.0))(0).imag();
        CHECK_THAT(mid, WithinRel(-kSigma * kWidth * std::pow(0.5, m + 1.0) / (m + 1.0), 1e-13));
        // Beyond the PML end the stretch continues linearly with slope sigma_max.
        const Real beyond = s.map(Vec3(end + 0.3e-6, 0.0, 0.0))(0).imag();
        CHECK_THAT(beyond, WithinRel(expected - kSigma * 0.3e-6, 1e-13));
        CHECK(s.jacobian_diag(Vec3(end + 0.3e-6, 0.0, 0.0))(0) == Complex(1.0, -kSigma));
    }
}

TEST_CASE("CoordinateStretch: Jacobian equals the derivative of the map", "[coordinate_stretch]") {
    for (const Real m : {1.0, 2.0, 3.0}) {
        const CoordinateStretch s(default_profile(m), StretchProfile{kStart, kWidth, 5.0, m});
        // Central differences, step 1e-4 W: O(step^2 sigma'') truncation error.
        const Real step = 1e-4 * kWidth;
        for (const Real u : {0.05, 0.2, 0.5, 0.8, 0.95, 1.3}) {
            for (const Real sgn : {1.0, -1.0}) {
                const Real t = sgn * (kStart + u * kWidth);
                const Vec3 r(t, 0.9 * t, 0.0);
                const Vec3c jd = s.jacobian_diag(r);
                const Complex fx =
                    (s.map(Vec3(t + step, r.y(), 0.0))(0) - s.map(Vec3(t - step, r.y(), 0.0))(0)) /
                    (2.0 * step);
                const Complex fy = (s.map(Vec3(r.x(), r.y() + step, 0.0))(1) -
                                    s.map(Vec3(r.x(), r.y() - step, 0.0))(1)) /
                                   (2.0 * step);
                CHECK_THAT(std::abs(jd(0) - fx), WithinAbs(0.0, 1e-6));
                CHECK_THAT(std::abs(jd(1) - fy), WithinAbs(0.0, 1e-6));
                CHECK(jd(2) == Complex(1.0, 0.0));
                // dx~/dx = 1 - j sigma(|x|) on both sides (the sgn cancels).
                CHECK(jd(0).real() == 1.0);
                CHECK(jd(0).imag() < 0.0);
            }
        }
    }
}

TEST_CASE("CoordinateStretch: outgoing waves decay in both directions", "[coordinate_stretch]") {
    const CoordinateStretch s(default_profile(), default_profile());
    const Real k0 = 2.0 * constants::pi / kLambda;
    for (const Real k : {k0, 1.055 * k0}) {
        for (const Real sgn : {1.0, -1.0}) {
            // Outgoing along sgn x: exp(-j k sgn x~).
            Real prev = 1.0;
            for (int q = 1; q <= 20; ++q) {
                const Real x = sgn * (kStart + kWidth * static_cast<Real>(q) / 20.0);
                const Complex xt = s.map(Vec3(x, 0.0, 0.0))(0);
                const Real mag = std::abs(std::exp(Complex(0.0, -k * sgn) * xt));
                CHECK(mag < prev);
                prev = mag;
            }
            // At the PML end: exp(-k sigma_max W / (m + 1)).
            CHECK_THAT(prev, WithinRel(std::exp(-k * kSigma * kWidth / 3.0), 1e-12));
        }
        // An incoming wave (sgn reversed) would grow: the sign is the exp(+j w t) one.
        const Complex xt = s.map(Vec3(kStart + kWidth, 0.0, 0.0))(0);
        CHECK(std::abs(std::exp(Complex(0.0, k) * xt)) > 1.0);
    }
}

TEST_CASE("CoordinateStretch: product decay of an SPP-like wave in the corner",
          "[coordinate_stretch]") {
    const CoordinateStretch s(default_profile(), default_profile());
    const Real k0 = 2.0 * constants::pi / kLambda;
    const Real k_spp = 1.055 * k0;  // Ag at 500 nm, |k_t| of the WP-V3 surface wave
    // Oblique outgoing wave into the (+x, -y) corner: k_t = k_spp (cos a, -sin a).
    const Real a = 0.6;
    const Real kx = k_spp * std::cos(a);
    const Real ky = -k_spp * std::sin(a);
    const Real end = kStart + kWidth;
    const auto amplitude = [&](const Vec3& r) {
        const Vec3c rt = s.map(r);
        return std::abs(std::exp(Complex(0.0, -1.0) * (kx * rt(0) + ky * rt(1))));
    };
    const Real ax = amplitude(Vec3(end, 0.0, 0.0));
    const Real ay = amplitude(Vec3(0.0, -end, 0.0));
    const Real corner = amplitude(Vec3(end, -end, 0.0));
    CHECK_THAT(corner, WithinRel(ax * ay, 1e-12));
    CHECK_THAT(ax, WithinRel(std::exp(-kx * kSigma * kWidth / 3.0), 1e-12));
    CHECK_THAT(ay, WithinRel(std::exp(ky * kSigma * kWidth / 3.0), 1e-12));
    // Along the axis the one-way damping at W = 2 lambda, sigma_max = 4 is far below the PML
    // target 1e-4 (ADR 0009, decision 4).
    CHECK(std::exp(-k_spp * kSigma * kWidth / 3.0) < 1e-4);
    CHECK(corner < 1e-4);
}

TEST_CASE("CoordinateStretch: identity() is bit-identical", "[coordinate_stretch]") {
    const CoordinateStretch id = CoordinateStretch::identity();
    REQUIRE(id.is_identity());
    CHECK_FALSE(id.profile_x().has_value());
    CHECK_FALSE(id.profile_y().has_value());
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Real> u(-1e-3, 1e-3);
    for (int k = 0; k < 100; ++k) {
        const Vec3 r(u(rng), u(rng), u(rng));
        const Vec3c rt = id.map(r);
        for (int c = 0; c < 3; ++c) {
            CHECK(bit_equal(rt(c).real(), r(c)));
            CHECK(bit_equal(rt(c).imag(), 0.0));
        }
        CHECK(id.jacobian_diag(r) == Vec3c::Ones());
        CHECK_FALSE(id.in_pml(r));
    }
    // sigma_max = 0 on both axes is also the identity map (in_pml stays geometric).
    const CoordinateStretch zero(StretchProfile{1e-6, 1e-6, 0.0, 2.0},
                                 StretchProfile{1e-6, 1e-6, 0.0, 2.0});
    CHECK(zero.is_identity());
    const Vec3 r(1.7e-6, -2.4e-6, 3e-8);
    const Vec3c rt = zero.map(r);
    for (int c = 0; c < 3; ++c) {
        CHECK(bit_equal(rt(c).real(), r(c)));
        CHECK(bit_equal(rt(c).imag(), 0.0));
    }
    CHECK(zero.jacobian_diag(r) == Vec3c::Ones());
    CHECK(zero.in_pml(r));
    // One stretched axis is not the identity.
    CHECK_FALSE(
        CoordinateStretch(StretchProfile{1e-6, 1e-6, 0.0, 2.0}, default_profile()).is_identity());
}

TEST_CASE("CoordinateStretch: invalid profiles throw", "[coordinate_stretch]") {
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    const StretchProfile ok = default_profile();
    const auto bad = [&](StretchProfile p) {
        CHECK_THROWS_AS(CoordinateStretch(p, ok), std::invalid_argument);
        CHECK_THROWS_AS(CoordinateStretch(ok, p), std::invalid_argument);
    };
    bad({kStart, 0.0, kSigma, 2.0});      // width = 0
    bad({kStart, -1e-6, kSigma, 2.0});    // width < 0
    bad({kStart, kWidth, -0.1, 2.0});     // sigma_max < 0
    bad({kStart, kWidth, kSigma, -1.0});  // order < 0
    bad({-1e-6, kWidth, kSigma, 2.0});    // start < 0
    bad({nan, kWidth, kSigma, 2.0});
    bad({kStart, inf, kSigma, 2.0});
    bad({kStart, kWidth, nan, 2.0});
    bad({kStart, kWidth, kSigma, inf});
    CHECK_NOTHROW(CoordinateStretch(StretchProfile{0.0, kWidth, 0.0, 0.0}, ok));
}
