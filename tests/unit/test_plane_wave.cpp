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

/// Worst relative error of the plane-wave addition theorem for `pairs` random source and
/// observer points uniform in their boxes (edge a), over the six nearest centre offsets.
Real addition_error_random(Complex k, Real a, Real digits, int pairs, std::mt19937& rng) {
    std::uniform_real_distribution<Real> u(-0.5, 0.5);
    const SphereSampling s(mlfmm::truncation_order(k, std::sqrt(3.0) * a, digits));
    const std::vector<Vec3> offsets = {Vec3(2, 0, 0), Vec3(2, 1, 0), Vec3(2, 1, 1),
                                       Vec3(2, 2, 0), Vec3(2, 2, 1), Vec3(2, 2, 2)};
    Real worst = 0.0;
    for (const Vec3& off : offsets) {
        const Vec3 x = off * a;
        const VectorXc wt = s.weights().cast<Complex>().cwiseProduct(mlfmm::translator(k, x, s));
        for (int n = 0; n < pairs; ++n) {
            const Vec3 o = a * Vec3(u(rng), u(rng), u(rng));   // o - C_o
            const Vec3 sp = a * Vec3(u(rng), u(rng), u(rng));  // s - C_s
            // exp(-jk khat.(o - C_o)) exp(-jk khat.(C_s - s)) = exp(-jk khat.(o - sp))
            const Complex approx =
                -kJ * k / (4.0 * kPi) * wt.cwiseProduct(plane_waves(k, s, o - sp)).sum();
            const Real r = (x + o - sp).norm();
            const Complex exact = std::exp(-kJ * k * r) / r;
            worst = std::max(worst, std::abs(approx - exact) / std::abs(exact));
        }
    }
    return worst;
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

// Measured (seed 20261009): random-point error for real k, d0 = 3: 3.6e-4 / 1.6e-5 / 5.1e-7 /
// 8.3e-10 / 2.1e-13 at a = 1/4, 1/2, 1, 2, 4 lambda; d0 = 5: 1.7e-5 / 2.1e-6 / 9.6e-8 / 1.2e-11 /
// 2.1e-11. Worst case (expansion_error): d0 = 3: 8.6e-2 / 4.1e-2 / 1.7e-2 / 5.8e-3 / 1.2e-3;
// d0 = 5: 4.2e-2 / 2.7e-2 / 1.0e-2 / 1.7e-3 / 2.3e-4. At a = lambda/4 the ADR order loses up to
// one digit for random points (other seeds: 1.6e-3 at d0 = 3).
TEST_CASE("mlfmm: addition theorem for real k", "[plane_wave]") {
    std::mt19937 rng(20261009);
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0, 2.0, 4.0}) {
            if (a > kMaxBox) {
                continue;
            }
            const Real err = addition_error_random(kK0, a, digits, 4, rng);
            const mlfmm::ExpansionError wc = mlfmm::expansion_error(kK0, a, digits);
            INFO("d0 = " << digits << ", a = " << a << ", L = " << wc.order << ": random " << err
                         << ", worst case " << wc.max_relative_error);
            CHECK(err <= std::pow(10.0, -digits) * (a < 0.5 ? 10.0 : 1.0));
            CHECK(err <= wc.max_relative_error);  // the corner check is conservative
            CHECK(wc.max_relative_error < 0.1);
            CHECK(wc.worst_offset.isApprox(Vec3(2, 0, 0)));
        }
    }
    // A raised order passes the worst-case check (building block for an order search, WP21).
    CHECK(mlfmm::expansion_error(kK0, 2.0, 3.0, mlfmm::ExpansionErrorOptions{45, 0.0})
              .max_relative_error <= 1e-3);
}

TEST_CASE("mlfmm: addition theorem for Si-like and Ag-like complex k", "[plane_wave]") {
    std::mt19937 rng(77);
    const Complex k_si = kK0 * Complex(4.3, -0.07);
    for (const Real digits : {3.0, 5.0}) {
        for (const Real a : {0.25, 0.5, 1.0}) {  // measured random <= 4.6e-6, worst case 1e-3..2e-2
            if (a > 0.5 * kMaxBox) {
                continue;
            }
            const Real err = addition_error_random(k_si, a, digits, 3, rng);
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

namespace {

/// Max abs error of child -> parent interpolation (orders from truncation_order for child edge a
/// and the parent edge 2a) of exp(-jk khat.d), |d| = half the child diagonal, four directions of
/// d. theta_component: interpolate theta_hat.u exp(-jk khat.d) (odd under pole reflection).
Real interpolation_error(Real a, Real digits, int p, bool theta_component = false,
                         mlfmm::PoleParity parity = mlfmm::PoleParity::even) {
    const Real dc = std::sqrt(3.0) * a;
    const SphereSampling child(mlfmm::truncation_order(kK0, dc, digits));
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

// Measured max error (real k) vs p, d0 = 3 at a = 1/2, 1, 2 lambda: p = 6: 8.8e-4, 3.5e-3, 9.6e-3;
// p = 10: 8.0e-5, 1.8e-4, 7.4e-4; p = 14: 1.5e-5, 3.3e-5, 3.1e-5. d0 = 5 at a = 1, 2 lambda:
// p = 6: 1.1e-3, 2.4e-3; p = 14: 7.5e-6, 2.4e-5; p = 22: 2.3e-7, 4.0e-7. The default p = 6 does not
// reach 0.1 x 10^-d0 (ADR 0008 §3); the target needs p ~ 12-14 (d0 = 3) and ~ 20-22 (d0 = 5).
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
    for (const auto& [src, tgt, p] : {std::tuple{&coarse, &fine, 6}, std::tuple{&fine, &coarse, 5},
                                      std::tuple{&coarse, &fine, 10}}) {
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
    const SphereInterpolator same(fine, fine);
    const VectorXc v = random_vector(fine.size());
    CHECK((same.interpolate(v) - v).norm() <= 1e-14 * v.norm());
    // Span interface with a larger workspace; size checks.
    const SphereInterpolator interp(coarse, fine);
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
    CHECK_THROWS_AS(SphereInterpolator(coarse, fine, 11), std::invalid_argument);
}
