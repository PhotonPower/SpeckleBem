#include "specklebem/compression/mlfmm/interpolation.hpp"
#include "specklebem/compression/mlfmm/plane_wave.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace specklebem;
using mlfmm::SphereInterpolator;
using mlfmm::SphereSampling;

namespace {

constexpr Real kPi = constants::pi;
constexpr Real kK0 = 2.0 * kPi;  // wavelength 1 m: box sizes below are in wavelengths
const Complex kJ{0.0, 1.0};
#ifdef NDEBUG
constexpr Real kMaxBox = 4.0;  // largest box edge (wavelengths) of the addition-theorem sweeps
#else
constexpr Real kMaxBox = 1.0;  // unoptimised + sanitizers: the 2 and 4 lambda boxes take ~15 s
#endif

std::complex<long double> to_ld(Complex z) {
    return {static_cast<long double>(z.real()), static_cast<long double>(z.imag())};
}

/// h_l^{(2)}(z) = j^{l+1} e^{-jz}/z sum_k (l+k)!/(k!(l-k)!) (-j/(2z))^k (Bessel polynomial) in
/// long double; `cond` = sum |terms| / |sum| bounds the cancellation of the reference.
std::complex<long double> hankel_reference(int l, Complex z, long double& cond) {
    using C = std::complex<long double>;
    const C zl = to_ld(z);
    const C step = C(0.0L, -1.0L) / (2.0L * zl);
    C sum = 0.0L;
    C pw = 1.0L;
    long double coef = 1.0L;
    long double abs_sum = 0.0L;
    for (int k = 0; k <= l; ++k) {
        const C term = coef * pw;
        sum += term;
        abs_sum += std::abs(term);
        coef *= static_cast<long double>(l + k + 1) * static_cast<long double>(l - k) /
                static_cast<long double>(k + 1);
        pw *= step;
    }
    C jpow = 1.0L;
    for (int k = 0; k <= l % 4; ++k) {
        jpow *= C(0.0L, 1.0L);
    }
    cond = abs_sum / std::abs(sum);
    return jpow * std::exp(C(0.0L, -1.0L) * zl) / zl * sum;
}

/// exp(-jk khat.d) at every direction of s.
VectorXc plane_waves(Complex k, const SphereSampling& s, const Vec3& d) {
    const VectorXr phase = s.directions() * d;
    VectorXc w(s.size());
    for (Index q = 0; q < s.size(); ++q) {
        w.data()[q] = std::exp(-kJ * k * phase.data()[q]);
    }
    return w;
}

/// The statistical check of the ADR 0008 amendment (library, default pairs and seed).
mlfmm::ExpansionError random_check(Complex k, Real a, Real digits, int order = 0) {
    mlfmm::ExpansionErrorOptions opt;
    opt.order = order;
    opt.mode = mlfmm::ExpansionCheck::random;
    return mlfmm::expansion_error(k, a, digits, opt);
}

}  // namespace

TEST_CASE("mlfmm: truncation order follows the excess-bandwidth formula", "[plane_wave]") {
    const Real kd = kK0 * std::sqrt(3.0);
    CHECK(mlfmm::truncation_order(kK0, std::sqrt(3.0), 3.0) ==
          static_cast<int>(std::ceil(kd + 1.8 * std::pow(3.0, 2.0 / 3.0) * std::cbrt(kd))));
    // Only Re k sets the bandwidth.
    CHECK(mlfmm::truncation_order(Complex(kK0, -5.0), 1.0, 5.0) ==
          mlfmm::truncation_order(kK0, 1.0, 5.0));
    CHECK(mlfmm::truncation_order(kK0, 2.0, 3.0) > mlfmm::truncation_order(kK0, 1.0, 3.0));
    CHECK(mlfmm::truncation_order(kK0, 1.0, 5.0) > mlfmm::truncation_order(kK0, 1.0, 3.0));
    CHECK_THROWS_AS(mlfmm::truncation_order(Complex(kK0, 0.1), 1.0, 3.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::truncation_order(-kK0, 1.0, 3.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::truncation_order(kK0, 0.0, 3.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::truncation_order(kK0, 1.0, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::truncation_order(kK0, 1e9, 3.0), std::invalid_argument);
    CHECK_THROWS_AS(SphereSampling(-1), std::invalid_argument);
}

TEST_CASE("mlfmm: sphere sampling directions, frames and quadrature", "[plane_wave]") {
    const SphereSampling s(9);
    REQUIRE(s.num_theta() == 10);
    REQUIRE(s.num_phi() == 20);
    REQUIRE(s.size() == 200);
    CHECK(std::abs(s.weights().sum() - 4.0 * kPi) < 1e-13);
    for (Index q = 0; q < s.size(); ++q) {
        const Vec3 k = s.directions().row(q).transpose();
        const Vec3 th = s.theta_hat().row(q).transpose();
        const Vec3 ph = s.phi_hat().row(q).transpose();
        CHECK(std::abs(k.norm() - 1.0) < 1e-15);
        CHECK((th.cross(ph) - k).norm() < 1e-15);  // right-handed (khat, theta_hat, phi_hat)
        CHECK(std::abs(th.dot(k)) + std::abs(ph.dot(k)) < 1e-15);
    }
    for (int i = 1; i < s.num_theta(); ++i) {
        CHECK(s.theta()(i) > s.theta()(i - 1));
    }
    // Moments of degree <= 2L + 1 = 19 are exact: int x^2 y^4 z^12 dOmega = 4pi 1!!3!!11!!/19!!.
    Real m = 0.0;
    for (Index q = 0; q < s.size(); ++q) {
        const auto k = s.directions().row(q);
        m += s.weights()(q) * k(0) * k(0) * std::pow(k(1), 4) * std::pow(k(2), 12);
    }
    const Real exact = 4.0 * kPi * 3.0 * 10395.0 / 654729075.0;
    CHECK(std::abs(m - exact) < 1e-14 * exact);
}

TEST_CASE("mlfmm: Legendre polynomials", "[plane_wave]") {
    for (const Real x : {-1.0, -0.73, 0.0, 0.41, 1.0}) {
        const std::vector<Real> p = mlfmm::legendre_p(40, x);
        CHECK(std::abs(p[2] - 0.5 * (3 * x * x - 1)) < 1e-15);
        CHECK(std::abs(p[3] - 0.5 * (5 * x * x * x - 3 * x)) < 1e-15);
        CHECK(std::abs(p[4] - (35 * std::pow(x, 4) - 30 * x * x + 3) / 8) < 1e-15);
    }
    CHECK(std::abs(mlfmm::legendre_p(40, 1.0)[40] - 1.0) < 1e-14);
    CHECK(std::abs(mlfmm::legendre_p(41, -1.0)[41] + 1.0) < 1e-14);
    CHECK_THROWS_AS(mlfmm::legendre_p(3, 1.5), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::legendre_p(-1, 0.5), std::invalid_argument);
}

TEST_CASE("mlfmm: spherical Hankel closed forms for l = 0, 1, 2", "[plane_wave]") {
    for (const Complex z : {Complex(0.3, 0.0), Complex(2.5, 0.0), Complex(17.0, 0.0),
                            Complex(1.2, -0.4), Complex(0.2, -3.1), Complex(30.0, -2.0)}) {
        const std::vector<Complex> h = mlfmm::spherical_hankel2(2, z);
        // h_l^{(2)} = j_l - j y_l from the sin / cos forms of j_l, y_l (exp(+jwt): outgoing).
        const Complex s = std::sin(z);
        const Complex c = std::cos(z);
        const Complex j0 = s / z, y0 = -c / z;
        const Complex j1 = s / (z * z) - c / z, y1 = -c / (z * z) - s / z;
        const Complex j2 = (3.0 / (z * z * z) - 1.0 / z) * s - 3.0 * c / (z * z);
        const Complex y2 = -(3.0 / (z * z * z) - 1.0 / z) * c - 3.0 * s / (z * z);
        // The sin / cos forms cancel for large |Im z| (|h| << |j| + |y|): scale by |j| + |y|.
        CHECK(std::abs(h[0] - (j0 - kJ * y0)) < 1e-14 * (std::abs(j0) + std::abs(y0)));
        CHECK(std::abs(h[0] - kJ * std::exp(-kJ * z) / z) < 1e-15 * std::abs(h[0]));
        CHECK(std::abs(h[1] - (j1 - kJ * y1)) < 1e-14 * (std::abs(j1) + std::abs(y1)));
        CHECK(std::abs(h[2] - (j2 - kJ * y2)) < 1e-14 * (std::abs(j2) + std::abs(y2)));
    }
    CHECK_THROWS_AS(mlfmm::spherical_hankel2(3, Complex(1.0, 0.1)), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::spherical_hankel2(3, Complex(0.0, -1.0)), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::spherical_hankel2(400, Complex(0.1, 0.0)), std::overflow_error);
}

TEST_CASE("mlfmm: spherical Hankel underflow is an error, not zeros", "[plane_wave]") {
    // |h_0(10 - 740j)| = e^-740 / |z| ~ 1e-324: e^{-jz} underflows.
    CHECK_THROWS_AS(mlfmm::spherical_hankel2(5, Complex(10.0, -740.0)), std::underflow_error);
    CHECK_THROWS_AS(mlfmm::spherical_hankel2(0, Complex(1.0, -705.0)), std::underflow_error);
    // Inside the domain: e^-650 / |z| ~ 1e-285, all values normal and nonzero.
    const Complex z(10.0, -650.0);
    const std::vector<Complex> h = mlfmm::spherical_hankel2(20, z);
    for (const Complex& v : h) {
        CHECK(std::abs(v) >= std::numeric_limits<Real>::min());
        CHECK(std::isfinite(std::abs(v)));
    }
    CHECK(std::abs(h[0] - kJ * std::exp(-kJ * z) / z) < 1e-14 * std::abs(h[0]));
    // The translator inherits the domain (strongly lossy k, far offset).
    const SphereSampling s(4);
    CHECK_THROWS_AS(mlfmm::translator(Complex(1.0, -800.0), Vec3(1.0, 0.0, 0.0), s),
                    std::underflow_error);
}

TEST_CASE("mlfmm: spherical Hankel recurrence vs Bessel polynomial and Wronskian", "[plane_wave]") {
    const long double eps_ld = std::numeric_limits<long double>::epsilon();
    for (const Complex z :
         {Complex(0.05, 0.0), Complex(0.5, 0.0), Complex(1.0, 0.0), Complex(3.7, 0.0),
          Complex(10.0, 0.0), Complex(30.0, 0.0), Complex(80.0, 0.0), Complex(2.0, -0.5),
          Complex(10.0, -3.0), Complex(0.3, -3.0), Complex(0.06, -3.1), Complex(25.0, -0.4)}) {
        const int lmax = std::abs(z) < 0.1 ? 80 : 120;
        const std::vector<Complex> h = mlfmm::spherical_hankel2(lmax, z);
        // Rounding check: the same recurrence in long double.
        using CL = std::complex<long double>;
        const CL zl = to_ld(z);
        CL h_prev = CL(0.0L, 1.0L) * std::exp(CL(0.0L, -1.0L) * zl) / zl;
        CL h_cur = std::exp(CL(0.0L, -1.0L) * zl) * (CL(0.0L, 1.0L) / (zl * zl) - 1.0L / zl);
        Real worst_poly = 0.0, worst_ld = 0.0;
        for (int l = 0; l <= lmax; ++l) {
            const Complex hl = h[static_cast<std::size_t>(l)];
            long double cond = 0.0L;
            const CL ref = hankel_reference(l, z, cond);
            if (cond <= 1e6L) {  // reference trustworthy to <= 1e-13
                const Real err = std::abs(hl - Complex(static_cast<Real>(ref.real()),
                                                       static_cast<Real>(ref.imag()))) /
                                 static_cast<Real>(std::abs(ref));
                worst_poly = std::max(worst_poly, err);
                CHECK(err <= 1e-15 * (l + 1) + static_cast<Real>(64.0L * eps_ld * cond));
            }
            const CL ld = l == 0 ? h_prev : h_cur;
            if (l >= 1) {
                const CL next = static_cast<long double>(2 * l + 1) / zl * h_cur - h_prev;
                h_prev = h_cur;
                h_cur = next;
            }
            const Real err_ld = static_cast<Real>(std::abs(to_ld(hl) - ld) / std::abs(ld));
            worst_ld = std::max(worst_ld, err_ld);
            CHECK(err_ld <= 1e-15 * (l + 1));
        }
        INFO("z = " << z << " lmax " << lmax << " vs polynomial " << worst_poly
                    << " vs long double " << worst_ld);
        if (z.imag() == 0.0) {  // j_l y_{l-1} - j_{l-1} y_l = 1/z^2 where j_l is not tiny
            for (int l = 1; l <= std::min(lmax, static_cast<int>(z.real())); ++l) {
                const auto u = static_cast<std::size_t>(l);
                const Real jl = h[u].real(), yl = -h[u].imag();
                const Real jm = h[u - 1].real(), ym = -h[u - 1].imag();
                const Real w = (jl * ym - jm * yl) * z.real() * z.real();
                CHECK(std::abs(w - 1.0) < 1e-12);
            }
        }
    }
}

namespace {

using CL = std::complex<long double>;

/// j_l(z), l = 0..lmax, by Miller's downward recurrence (the minimal solution), normalised by
/// j_0 = sin z / z, in long double. The start index lies far beyond max(lmax, |z|).
std::vector<CL> bessel_j_miller(int lmax, CL z) {
    // Index arithmetic in std::size_t from a clamped lmax: with a signed int start, GCC 13 -O3
    // follows the path start = -2 (size 0, null data) into j[start] and reports a potential
    // null dereference in std::vector's allocation under -Wnull-dereference.
    const std::size_t start = static_cast<std::size_t>(std::max(lmax, 0)) + 60 +
                              2 * static_cast<std::size_t>(std::abs(z));
    std::vector<CL> j(start + 2, CL(0.0L, 0.0L));
    j[start] = CL(1e-30L, 0.0L);
    for (std::size_t u = start; u >= 1; --u) {
        j[u - 1] = static_cast<long double>(2 * u + 1) / z * j[u] - j[u + 1];
    }
    const CL scale = std::sin(z) / z / j[0];
    j.resize(static_cast<std::size_t>(lmax) + 1);
    for (CL& v : j) {
        v *= scale;
    }
    return j;
}

/// y_l(z), l = 0..lmax, by upward recurrence from the closed forms, in long double.
std::vector<CL> bessel_y_upward(int lmax, CL z) {
    std::vector<CL> y;
    y.push_back(-std::cos(z) / z);
    y.push_back(-std::cos(z) / (z * z) - std::sin(z) / z);
    for (int l = 1; l < lmax; ++l) {
        const auto u = static_cast<std::size_t>(l);
        y.push_back(static_cast<long double>(2 * l + 1) / z * y[u] - y[u - 1]);
    }
    return y;
}

}  // namespace

// Independent large-|z| reference for complex z (the Bessel-polynomial form cancels there):
// h_l = j_l - j y_l with j_l from Miller's downward recurrence and y_l upward, both in long
// double, themselves verified by the Wronskian j_l y_{l-1} - j_{l-1} y_l = z^-2. The reference
// loses ~e^{2|Im z|} to cancellation (|j|, |y| ~ e^{|Im z|}, |h| ~ e^{-|Im z|}): hence the
// (|j| + |y|) term of the tolerance.
TEST_CASE("mlfmm: spherical Hankel for complex z up to |z| = 80 (Miller + Wronskian)",
          "[plane_wave]") {
    const long double eps_ld = std::numeric_limits<long double>::epsilon();
    for (const Complex z : {Complex(80.0, 0.0), Complex(80.0, -0.5), Complex(79.5, -3.0),
                            Complex(50.0, -5.0), Complex(20.0, -4.0), Complex(3.0, -2.0)}) {
        constexpr int kLmax = 120;
        const std::vector<Complex> h = mlfmm::spherical_hankel2(kLmax, z);
        INFO("z = " << z);
        const CL zl = to_ld(z);
        const std::vector<CL> j = bessel_j_miller(kLmax, zl);
        const std::vector<CL> y = bessel_y_upward(kLmax, zl);
        Real worst = 0.0, worst_w = 0.0;
        for (int l = 0; l <= kLmax; ++l) {
            const auto u = static_cast<std::size_t>(l);
            if (l >= 1) {
                const CL a = j[u] * y[u - 1] * zl * zl;
                const CL b = j[u - 1] * y[u] * zl * zl;
                const long double werr = std::abs(a - b - 1.0L);
                worst_w = std::max(worst_w, static_cast<Real>(werr));
                CHECK(werr <= 1e-15L + 256.0L * eps_ld * (std::abs(a) + std::abs(b)));
            }
            const CL ref = j[u] - CL(0.0L, 1.0L) * y[u];
            const long double diff = std::abs(to_ld(h[u]) - ref);
            worst = std::max(worst, static_cast<Real>(diff / std::abs(ref)));
            // Double recurrence: O(l eps) relative (as in the long double recurrence check).
            CHECK(diff <= 1e-15L * (l + 1) * std::abs(ref) +
                              256.0L * eps_ld * (std::abs(j[u]) + std::abs(y[u])));
        }
        INFO("max relative error " << worst << ", Wronskian " << worst_w);
        CHECK(worst <= 1.4e-13);
    }
}

TEST_CASE("mlfmm: translator coefficients that overflow raise overflow_error", "[plane_wave]") {
    // |h_60(x)| ~ 119!!/x^61 crosses DBL_MAX near x = 3.7e-4: bisect (log scale) to the largest
    // h_60 that spherical_hankel2 still returns; then c_60 = 121 h_60 must overflow.
    constexpr int kL = 60;
    Real lo = 1e-5, hi = 1e-1;  // lo overflows, hi does not
    REQUIRE_THROWS_AS(mlfmm::spherical_hankel2(kL, lo), std::overflow_error);
    REQUIRE_NOTHROW(mlfmm::spherical_hankel2(kL, hi));
    for (int it = 0; it < 80; ++it) {
        const Real mid = std::sqrt(lo * hi);
        try {
            (void)mlfmm::spherical_hankel2(kL, mid);
            hi = mid;
        } catch (const std::overflow_error&) {
            lo = mid;
        }
    }
    const std::vector<Complex> h = mlfmm::spherical_hankel2(kL, hi);
    REQUIRE(std::abs(h.back()) > std::numeric_limits<Real>::max() / (2 * kL + 1));
    const SphereSampling s(kL);
    CHECK_THROWS_AS(mlfmm::translator(Complex(hi, 0.0), Vec3(1.0, 0.0, 0.0), s),
                    std::overflow_error);
    CHECK_THROWS_AS(mlfmm::translator(Complex(1.0, 0.0), Vec3(0.0, 0.0, 1e-4), s),
                    std::overflow_error);
    // A regular argument stays finite.
    CHECK(mlfmm::translator(Complex(1.0, 0.0), Vec3(0.0, 0.0, 40.0), s).allFinite());
}

TEST_CASE("mlfmm: translator matches the direct sum and its symmetry", "[plane_wave]") {
    const SphereSampling s(12);
    const Complex k(kK0 * 1.3, -0.2);
    const Vec3 r(0.9, -0.5, 1.4);
    const VectorXc t = mlfmm::translator(k, r, s);
    const VectorXc tm = mlfmm::translator(k, -r, s);
    const std::vector<Complex> h = mlfmm::spherical_hankel2(12, k * r.norm());
    for (Index q = 0; q < s.size(); q += 7) {
        const std::vector<Real> p =
            mlfmm::legendre_p(12, s.directions().row(q).dot(r.normalized().transpose()));
        Complex ref{0.0, 0.0};
        Complex mj{1.0, 0.0};
        for (std::size_t l = 0; l <= 12; ++l) {
            ref += mj * static_cast<Real>(2 * l + 1) * h[l] * p[l];
            mj *= -kJ;
        }
        CHECK(std::abs(t(q) - ref) < 1e-13 * std::abs(ref));
    }
    // T(k, -r, khat) = T(k, r, -khat); -khat is (theta -> pi - theta, phi -> phi + pi).
    for (int i = 0; i < s.num_theta(); ++i) {
        for (int j = 0; j < s.num_phi(); ++j) {
            const Index opp = s.index(s.num_theta() - 1 - i, (j + s.num_phi() / 2) % s.num_phi());
            CHECK(std::abs(tm(s.index(i, j)) - t(opp)) < 1e-12 * std::abs(t(opp)));
        }
    }
    CHECK_THROWS_AS(mlfmm::translator(k, Vec3::Zero(), s), std::invalid_argument);
}

// Measured at the formula order, a = 1/4, 1/2, 1, 2, 4 lambda. Statistical check (default 8
// pairs per offset, seed 20261009): d0 = 3: 1.6e-4 / 6.5e-6 / 2.2e-8 / 3.8e-11 / 3.1e-13;
// d0 = 5: 2.0e-5 / 9.0e-7 / 2.7e-9 / 1.9e-11 / 6.5e-12 (the WP18 test helper with other point
// sets saw up to 1.6e-3 / 2.4e-4 at lambda/4). Worst case (corners): d0 = 3: 8.6e-2 / 4.1e-2 /
// 1.7e-2 / 5.8e-3 / 1.2e-3; d0 = 5: 4.2e-2 / 2.7e-2 / 1.0e-2 / 1.7e-3 / 2.3e-4.
TEST_CASE("mlfmm: addition theorem for real k", "[plane_wave]") {
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0, 2.0, 4.0}) {
            if (a > kMaxBox) {
                continue;
            }
            const mlfmm::ExpansionError rnd = random_check(kK0, a, digits);
            const mlfmm::ExpansionError wc = mlfmm::expansion_error(kK0, a, digits);
            INFO("d0 = " << digits << ", a = " << a << ", L = " << wc.order << ": random "
                         << rnd.max_relative_error << ", worst case " << wc.max_relative_error);
            CHECK(rnd.order == wc.order);
            // a = lambda/4: the formula order loses up to one digit for random points; the ADR
            // 0008 amendment (WP18 measurements) therefore searches the order upwards
            // (search_truncation_order) and coarsens the leaf level where that fails.
            CHECK(rnd.max_relative_error <= std::pow(10.0, -digits) * (a < 0.5 ? 10.0 : 1.0));
            CHECK(rnd.max_relative_error <= wc.max_relative_error);  // corners are conservative
            CHECK(wc.max_relative_error < 0.1);
            CHECK(wc.worst_offset.isApprox(Vec3(2, 0, 0)));
        }
    }
    // A raised order passes the worst-case check (building block for an order search, WP21).
    CHECK(mlfmm::expansion_error(kK0, 2.0, 3.0, mlfmm::ExpansionErrorOptions{45, 0.0})
              .max_relative_error <= 1e-3);
}

TEST_CASE("mlfmm: statistical expansion check is seeded and deterministic", "[plane_wave]") {
    mlfmm::ExpansionErrorOptions opt;
    opt.mode = mlfmm::ExpansionCheck::random;
    const mlfmm::ExpansionError a = mlfmm::expansion_error(kK0, 0.5, 3.0, opt);
    const mlfmm::ExpansionError b = mlfmm::expansion_error(kK0, 0.5, 3.0, opt);
    CHECK(a.max_relative_error == b.max_relative_error);  // bitwise: same points, same sums
    CHECK(a.order == mlfmm::truncation_order(kK0, std::sqrt(3.0) * 0.5, 3.0));
    opt.seed = 4711;
    const mlfmm::ExpansionError c = mlfmm::expansion_error(kK0, 0.5, 3.0, opt);
    CHECK(c.max_relative_error != a.max_relative_error);
    CHECK(c.max_relative_error <= 1e-3);
    // More pairs: a larger sample, still within the target and below the corner worst case.
    opt.seed = 20261009;
    opt.pairs = 32;
    const mlfmm::ExpansionError d = mlfmm::expansion_error(kK0, 0.5, 3.0, opt);
    CHECK(d.max_relative_error <= 1e-3);
    CHECK(d.max_relative_error <= mlfmm::expansion_error(kK0, 0.5, 3.0).max_relative_error);
    // An enlarged diagonal (WP21: support radii) raises the order and keeps the target.
    opt.pairs = 8;
    opt.box_diagonal = 1.2 * std::sqrt(3.0) * 0.5;
    const mlfmm::ExpansionError e = mlfmm::expansion_error(kK0, 0.5, 3.0, opt);
    CHECK(e.order > a.order);
    opt.pairs = 0;
    CHECK_THROWS_AS(mlfmm::expansion_error(kK0, 0.5, 3.0, opt), std::invalid_argument);
    opt.pairs = 8;
    opt.order = -1;
    CHECK_THROWS_AS(mlfmm::expansion_error(kK0, 0.5, 3.0, opt), std::invalid_argument);
}

TEST_CASE("mlfmm: addition theorem for Si-like and Ag-like complex k", "[plane_wave]") {
    const Complex k_si = kK0 * Complex(4.3, -0.07);
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0}) {  // measured random <= 4.6e-6, worst case 1e-3..2e-2
            if (a > 0.5 * kMaxBox) {
                continue;
            }
            const Real err = random_check(k_si, a, digits).max_relative_error;
            const mlfmm::ExpansionError wc = mlfmm::expansion_error(k_si, a, digits);
            INFO("Si d0 = " << digits << ", a = " << a << ": random " << err << ", worst case "
                            << wc.max_relative_error);
            CHECK(err <= std::pow(10.0, -digits));
            CHECK(err <= wc.max_relative_error);
        }
    }
    CHECK((kMaxBox <= 1.0 ||  // release only (heavy under sanitizers)
           mlfmm::expansion_error(k_si, 0.5, 3.0, mlfmm::ExpansionErrorOptions{47, 0.0})
                   .max_relative_error <= 1e-3));
    // Ag: eps_r = -9.794 - 0.313j, n = sqrt(eps_r) with Im n <= 0. Measured 9.5, 2.6e2, 5.3e6,
    // 5.4e28 at a = 0.05, 0.1, 0.25, 1 lambda: the expansion is unusable.
    Complex n_ag = std::sqrt(Complex(-9.794, -0.313));
    n_ag = n_ag.imag() > 0.0 ? -n_ag : n_ag;
    for (const Real a : {0.05, 0.1, 0.25, 1.0}) {
        const Real e = mlfmm::expansion_error(kK0 * n_ag, a, 3.0).max_relative_error;
        CHECK(e > (a >= 0.25 ? 1e3 : 1.0));
    }
    CHECK_THROWS_AS(mlfmm::expansion_error(kK0, 0.0, 3.0), std::invalid_argument);
}

// Measured search_truncation_order (default options; full table in the slow sweep below):
// real k: the formula order meets 10^-d0 for a >= lambda/2 (d0 = 3, 5) and at lambda/4 for
// d0 = 3; lambda/4, d0 = 5: 11 -> 12 (9.3e-6); lambda/8: d0 = 3 at 6 (7.6e-4), d0 = 5: 8 -> 12
// (9.9e-6), d0 = 6 not achievable; lambda/4, d0 = 6: 12 -> 15 (4.0e-7), d0 = 7 not achievable.
// Si-like k (n = 4.3 - 0.07j), a = lambda_0/4 ... lambda_0, d0 = 3 and 5: the formula order
// (3.9e-8 ... 5.9e-12). Ag-like k, d0 = 3: a = 0.05 / 0.1 / 0.25 lambda_0: 2 -> 9, 2 -> 13,
// 3 -> 22 (<= 7.4e-4); a >= 0.5 lambda_0: breakdown (> 1e6); d0 = 5: not achievable (best
// 7.6e-5 / 4.4e-5 / 2.4e-4).
TEST_CASE("mlfmm: truncation order search (ADR 0008 amendment)", "[plane_wave]") {
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.5, 1.0, 2.0}) {
            const mlfmm::TruncationSearch r = mlfmm::search_truncation_order(kK0, a, digits);
            INFO("d0 = " << digits << ", a = " << a << ": formula " << r.formula_order << ", found "
                         << r.order << " (" << r.orders_tried << " tried), error " << r.error);
            CHECK(r.achievable);
            CHECK(r.order >= r.formula_order);
            CHECK(r.error <= std::pow(10.0, -digits));
            // The searched order is the formula order where the formula already suffices.
            CHECK(r.order <= r.formula_order + 2);
        }
    }
    // a = lambda/4: d0 = 3 at the formula order 8 (1.6e-4); d0 = 5 one order above the formula
    // (11 -> 12, 9.3e-6). The error minimum over L is 4.0e-7 at L = 15 (then the breakdown:
    // 3.1e-6, 1.5e-4, ... 2e11 at L = 30), so d0 = 7 is not achievable at lambda/4.
    {
        const mlfmm::TruncationSearch r3 = mlfmm::search_truncation_order(kK0, 0.25, 3.0);
        INFO("d0 = 3, lambda/4: formula " << r3.formula_order << ", order " << r3.order
                                          << ", error " << r3.error);
        CHECK(r3.achievable);
        CHECK(r3.error <= 1e-3);
        const mlfmm::TruncationSearch r5 = mlfmm::search_truncation_order(kK0, 0.25, 5.0);
        INFO("d0 = 5, lambda/4: formula " << r5.formula_order << ", order " << r5.order
                                          << ", error " << r5.error);
        CHECK(r5.achievable);
        CHECK(r5.order > r5.formula_order);  // the formula order alone misses 1e-5
        CHECK(r5.error <= 1e-5);
        const mlfmm::TruncationSearch r7 = mlfmm::search_truncation_order(kK0, 0.25, 7.0);
        INFO("d0 = 7, lambda/4: formula " << r7.formula_order << ", best " << r7.order << ", error "
                                          << r7.error);
        CHECK_FALSE(r7.achievable);
        CHECK(r7.breakdown);
        CHECK(r7.order > r7.formula_order);
        CHECK(r7.error < 1e-6);  // the best order seen is reported
        CHECK(r7.error > 1e-7);
    }
    // Si-like complex k at a = 0.5 lambda_0 (2.15 lambda in Si).
    {
        const Complex k_si = kK0 * Complex(4.3, -0.07);
        const mlfmm::TruncationSearch r = mlfmm::search_truncation_order(k_si, 0.5, 3.0);
        INFO("Si: formula " << r.formula_order << ", order " << r.order << ", error " << r.error);
        CHECK(r.achievable);
        CHECK(r.error <= 1e-3);
    }
    // Ag-like k (n = 0.05 - 3.13j): the formula order uses Re k and is far too small (2), but
    // small boxes reach d0 = 3 at a raised order (a = 0.1 lambda_0: L = 13, 7.3e-4); from
    // a = 0.5 lambda_0 the evanescent sampling is ill-conditioned (error > 1e6 at every L):
    // breakdown, i.e. the ADR 0008 §6 alternatives.
    Complex n_ag = std::sqrt(Complex(-9.794, -0.313));
    n_ag = n_ag.imag() > 0.0 ? -n_ag : n_ag;
    const mlfmm::TruncationSearch ag = mlfmm::search_truncation_order(kK0 * n_ag, 0.1, 3.0);
    INFO("Ag 0.1: formula " << ag.formula_order << ", order " << ag.order << ", error "
                            << ag.error);
    CHECK(ag.achievable);
    CHECK(ag.order > ag.formula_order + 5);
    CHECK(ag.error <= 1e-3);
    const mlfmm::TruncationSearch ag5 = mlfmm::search_truncation_order(kK0 * n_ag, 0.5, 3.0);
    INFO("Ag 0.5: formula " << ag5.formula_order << ", best " << ag5.order << ", error "
                            << ag5.error);
    CHECK_FALSE(ag5.achievable);
    CHECK(ag5.breakdown);
    CHECK(ag5.error > 1e3);
    // Argument checks.
    mlfmm::TruncationSearchOptions bad;
    bad.patience = 0;
    CHECK_THROWS_AS(mlfmm::search_truncation_order(kK0, 0.5, 3.0, bad), std::invalid_argument);
    bad.patience = 4;
    bad.max_order = 3;
    CHECK_THROWS_AS(mlfmm::search_truncation_order(kK0, 0.5, 3.0, bad), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::search_truncation_order(kK0, -1.0, 3.0), std::invalid_argument);
}

namespace {

/// Max abs error of child -> parent interpolation (orders from truncation_order for child edge a
/// and the parent edge 2a) of exp(-jk khat.d), |d| = half the child diagonal, four directions of
/// d. theta_component: interpolate theta_hat.u exp(-jk khat.d) (odd under pole reflection).
/// oversample_leaf: the child is a leaf sampled at leaf_sampling_order(L_c, p).
Real interpolation_error(Real a, Real digits, int p, bool theta_component = false,
                         mlfmm::PoleParity parity = mlfmm::PoleParity::even,
                         bool oversample_leaf = false) {
    const Real dc = std::sqrt(3.0) * a;
    const int lc = mlfmm::truncation_order(kK0, dc, digits);
    const SphereSampling child(oversample_leaf ? mlfmm::leaf_sampling_order(lc, p) : lc);
    const SphereSampling parent(mlfmm::truncation_order(kK0, 2.0 * dc, digits));
    const SphereInterpolator interp(child, parent, p);
    const Vec3 u(0.3, -0.5, 0.8);
    const auto f = [&](const SphereSampling& s, const Vec3& d) {
        VectorXc w = plane_waves(kK0, s, d);
        if (theta_component) {
            const VectorXr c = s.theta_hat() * u;
            for (Index q = 0; q < s.size(); ++q) {
                w.data()[q] *= c.data()[q];
            }
        }
        return w;
    };
    Real worst = 0.0;
    for (Vec3 d : {Vec3(1, 1, 1), Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0.3, -0.8, 0.5)}) {
        d *= 0.5 * dc / d.norm();
        const VectorXc fi = interp.interpolate(f(child, d), parity);
        worst = std::max(worst, (fi - f(parent, d)).cwiseAbs().maxCoeff());
    }
    return worst;
}

}  // namespace

TEST_CASE("mlfmm: interpolation order table and leaf oversampling", "[plane_wave]") {
    CHECK(mlfmm::interpolation_order(1.0) == 14);
    CHECK(mlfmm::interpolation_order(3.0) == 14);
    CHECK(mlfmm::interpolation_order(3.5) == 22);
    CHECK(mlfmm::interpolation_order(5.0) == 22);
    CHECK_THROWS_AS(mlfmm::interpolation_order(5.5), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::interpolation_order(0.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::interpolation_order(std::nan("")), std::invalid_argument);
    CHECK(mlfmm::leaf_sampling_order(8, 14) == 13);
    CHECK(mlfmm::leaf_sampling_order(30, 14) == 30);
    CHECK_THROWS_AS(mlfmm::leaf_sampling_order(-1, 14), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::leaf_sampling_order(8, 1), std::invalid_argument);
}

// Measured max error (real k) vs p, d0 = 3 at a = 1/2, 1, 2 lambda: p = 6: 8.8e-4, 3.5e-3, 9.6e-3;
// p = 10: 8.0e-5, 1.8e-4, 7.4e-4; p = 14: 1.5e-5, 3.3e-5, 3.1e-5. d0 = 5 at a = 1, 2 lambda:
// p = 6: 1.1e-3, 2.4e-3; p = 14: 7.5e-6, 2.4e-5; p = 22: 2.3e-7, 4.0e-7. p = 6 does not reach
// 0.1 x 10^-d0 (ADR 0008 §3); the target needs p ~ 12-14 (d0 = 3) and ~ 20-22 (d0 = 5), hence
// the table of interpolation_order (ADR 0008 amendment).
TEST_CASE("mlfmm: Lagrange interpolation of band-limited patterns", "[plane_wave]") {
    for (const Real a : {0.5, 1.0, 2.0}) {
        INFO("a = " << a);
        CHECK(interpolation_error(a, 3.0, a < 1.0 ? 12 : 14) <= 1e-4);
        const Real err6 = interpolation_error(a, 3.0, 6);
        CHECK(err6 <= 2e-2);
        CHECK(interpolation_error(a, 3.0, 10) < 0.25 * err6);
    }
    for (const Real a : {1.0, 2.0}) {
        INFO("a = " << a);
        CHECK(interpolation_error(a, 5.0, 22) <= 1e-6);
    }
    // theta / phi components are odd under the pole reflection.
    CHECK(interpolation_error(1.0, 3.0, 14, true, mlfmm::PoleParity::odd) <= 1e-4);
    CHECK(interpolation_error(1.0, 3.0, 14, true, mlfmm::PoleParity::even) > 1e-2);
}

// lambda/4 leaves (n_theta = 9 at d0 = 3, 12 at d0 = 5 without oversampling) with the table
// order and the oversampled leaf sampling L_leaf = max(L, p - 1) (ADR 0008 review amendment):
// d0 = 3 (p = 14): L_leaf = 13 equals the parent order 13, the interpolation is the identity
// (1.2e-15); without oversampling (n_theta = 9, stencil over the reflected nodes) 3.0e-5 /
// 9.8e-5 (odd), i.e. at the target. d0 = 5 (p = 22): L_leaf = 21 -> parent 15: 1.5e-12 /
// 7.5e-12 (odd); without oversampling 6.6e-8 / 3.4e-7. Also the lambda/2 level at d0 = 5
// (n_theta = 16 < p = 22, stencil over the pole-reflected nodes): 1.9e-7 / 5.3e-7.
TEST_CASE("mlfmm: interpolation from oversampled lambda/4 leaves meets 0.1 x 10^-d0",
          "[plane_wave]") {
    for (const Real digits : {3.0, 5.0}) {
        const int p = mlfmm::interpolation_order(digits);
        const Real target = 0.1 * std::pow(10.0, -digits);
        const Real leaf =
            interpolation_error(0.25, digits, p, false, mlfmm::PoleParity::even, true);
        const Real leaf_odd =
            interpolation_error(0.25, digits, p, true, mlfmm::PoleParity::odd, true);
        INFO("d0 = " << digits << ", p = " << p << ": leaf " << leaf << ", odd " << leaf_odd);
        CHECK(leaf <= target);
        CHECK(leaf_odd <= target);
    }
    const Real mid = interpolation_error(0.5, 5.0, 22);  // n_theta_src = 16 < 22
    INFO("d0 = 5, a = lambda/2, p = 22 without oversampling: " << mid);
    CHECK(mid <= 1e-6);
}

TEST_CASE("mlfmm: anterpolation is the exact transpose of interpolation", "[plane_wave]") {
    std::mt19937 rng(11);
    std::normal_distribution<Real> g(0.0, 1.0);
    const auto random_vector = [&](Index n) {
        VectorXc v(n);
        for (Index i = 0; i < n; ++i) {
            v(i) = Complex(g(rng), g(rng));
        }
        return v;
    };
    const SphereSampling coarse(9);
    const SphereSampling fine(17);
    // p = 16 and 20 > n_theta(coarse) = 10: stencils over the pole-reflected nodes.
    for (const auto& [src, tgt, p] :
         {std::tuple{&coarse, &fine, 6}, std::tuple{&fine, &coarse, 5},
          std::tuple{&coarse, &fine, 10}, std::tuple{&coarse, &fine, 16},
          std::tuple{&coarse, &fine, 20}}) {
        const SphereInterpolator interp(*src, *tgt, p);
        for (const auto parity : {mlfmm::PoleParity::even, mlfmm::PoleParity::odd}) {
            const VectorXc a = random_vector(interp.source_size());
            const VectorXc b = random_vector(interp.target_size());
            const VectorXc ia = interp.interpolate(a, parity);
            const VectorXc tb = interp.anterpolate(b, parity);
            const Complex lhs = (ia.array() * b.array()).sum();  // bilinear, no conjugation
            const Complex rhs = (a.array() * tb.array()).sum();
            CHECK(std::abs(lhs - rhs) <= 1e-13 * ia.norm() * b.norm());
        }
    }
    // Same sampling: the identity (target nodes coincide with source nodes).
    const SphereInterpolator same(fine, fine, 14);
    const VectorXc v = random_vector(fine.size());
    CHECK((same.interpolate(v) - v).norm() <= 1e-14 * v.norm());
    // Span interface with a larger workspace; size checks.
    const SphereInterpolator interp(coarse, fine, 6);
    const VectorXc a = random_vector(coarse.size());
    VectorXc out(fine.size());
    VectorXc ws(interp.workspace_size() + 3);
    interp.interpolate(std::span<const Complex>(a.data(), static_cast<std::size_t>(a.size())),
                       std::span<Complex>(out.data(), static_cast<std::size_t>(out.size())),
                       std::span<Complex>(ws.data(), static_cast<std::size_t>(ws.size())));
    CHECK((out - interp.interpolate(a)).norm() == 0.0);
    CHECK_THROWS_AS(interp.interpolate(VectorXc(fine.size())), std::invalid_argument);
    CHECK_THROWS_AS(interp.anterpolate(VectorXc(coarse.size())), std::invalid_argument);
    CHECK_THROWS_AS(SphereInterpolator(coarse, fine, 1), std::invalid_argument);
    CHECK_NOTHROW(SphereInterpolator(coarse, fine, 20));  // 2 n_theta: stencil fits exactly
    CHECK_THROWS_AS(SphereInterpolator(coarse, fine, 21), std::invalid_argument);
}

// Heavier measurement sweeps (label `slow`, registered as its own ctest entry; optimised builds
// only): the tables quoted in the comments above and in the report of the WP18 review round.
TEST_CASE("mlfmm: plane-wave accuracy sweep", "[plane_wave][.slow][mlfmm_sweep]") {
#ifndef NDEBUG
    SKIP("plane-wave sweep: optimised builds only");
#endif
    std::ostringstream os;
    os << "statistical check at the formula order (a in lambda, d0, L, error):\n";
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0, 2.0, 4.0}) {
            const mlfmm::ExpansionError e = random_check(kK0, a, digits);
            os << "  a " << a << " d0 " << digits << " L " << e.order << " err "
               << e.max_relative_error << "\n";
        }
    }
    os << "search_truncation_order (a, d0: formula -> order, error, achievable, breakdown, "
          "tried):\n";
    const auto search_row = [&os](const char* name, Complex k, Real a, Real digits) {
        const mlfmm::TruncationSearch r = mlfmm::search_truncation_order(k, a, digits);
        os << "  " << name << " a " << a << " d0 " << digits << ": " << r.formula_order << " -> "
           << r.order << " err " << r.error << " achievable " << r.achievable << " breakdown "
           << r.breakdown << " tried " << r.orders_tried << "\n";
    };
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0, 2.0, 4.0}) {
            search_row("real", kK0, a, digits);
        }
    }
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0}) {
            search_row("Si", kK0 * Complex(4.3, -0.07), a, digits);
        }
    }
    Complex n_ag = std::sqrt(Complex(-9.794, -0.313));
    n_ag = n_ag.imag() > 0.0 ? -n_ag : n_ag;
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.05, 0.1, 0.25, 0.5, 1.0}) {
            search_row("Ag", kK0 * n_ag, a, digits);
        }
    }
    for (const Real digits : {3.0, 5.0, 6.0, 7.0}) {
        for (const Real a : {0.125, 0.25}) {
            search_row("real", kK0, a, digits);
        }
    }
    os << "error vs L at lambda/4 (d0 = 5 points):";
    for (int l = 11; l <= 30; ++l) {
        os << " " << l << ":" << random_check(kK0, 0.25, 5.0, l).max_relative_error;
    }
    os << "\ninterpolation (a, d0, p, oversampled leaf: error even / odd):\n";
    for (const Real digits : {3.0, 5.0}) {
        const int p = mlfmm::interpolation_order(digits);
        for (const Real a : {0.25, 0.5, 1.0, 2.0}) {
            for (const bool over : {false, true}) {
                const Real dc = std::sqrt(3.0) * a;
                if (!over && 2 * (mlfmm::truncation_order(kK0, dc, digits) + 1) < p) {
                    continue;
                }
                os << "  a " << a << " d0 " << digits << " p " << p << " over " << over << ": "
                   << interpolation_error(a, digits, p, false, mlfmm::PoleParity::even, over)
                   << " / " << interpolation_error(a, digits, p, true, mlfmm::PoleParity::odd, over)
                   << "\n";
            }
        }
    }
    WARN(os.str());
    SUCCEED();
}
