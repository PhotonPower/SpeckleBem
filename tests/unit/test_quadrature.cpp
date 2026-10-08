#include "specklebem/kernels/quadrature.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <vector>

using namespace specklebem;
using kernels::gauss_legendre;
using kernels::LineRule;
using kernels::triangle_rule;
using kernels::triangle_rule_is_positive_interior;
using kernels::TriangleRule;

namespace {

constexpr int kMaxDegree = 20;

/// Point counts of Dunavant's rules, degree 1..20.
constexpr std::array<std::size_t, kMaxDegree> kDunavantPoints = {
    1, 3, 4, 6, 7, 12, 13, 16, 19, 25, 27, 33, 37, 42, 48, 52, 61, 70, 73, 79};

long double factorial(int n) {
    long double f = 1.0L;
    for (int k = 2; k <= n; ++k) {
        f *= static_cast<long double>(k);
    }
    return f;
}

/// Exact integral of x^a y^b over the reference triangle (0,0)-(1,0)-(0,1).
long double exact_monomial(int a, int b) {
    return factorial(a) * factorial(b) / factorial(a + b + 2);
}

long double ipow(long double x, int n) {
    long double r = 1.0L;
    for (int k = 0; k < n; ++k) {
        r *= x;
    }
    return r;
}

/// Rule applied to x^a y^b on the reference triangle: x = l2, y = l3, area = 1/2.
long double rule_monomial(const TriangleRule& rule, int a, int b) {
    long double s = 0.0L;
    for (std::size_t i = 0; i < rule.weights.size(); ++i) {
        const Vec3& l = rule.barycentric[i];
        s += static_cast<long double>(rule.weights[i]) * ipow(static_cast<long double>(l(1)), a) *
             ipow(static_cast<long double>(l(2)), b);
    }
    return 0.5L * s;
}

/// Powers l_j^e (e = 0..max_exp) of the barycentric coordinates of every rule point, so that
/// the many monomials of the symmetry test cost one multiply-add per point.
struct PowerTable {
    std::vector<std::array<std::vector<long double>, 3>> pw;  ///< pw[point][j][e]

    PowerTable(const TriangleRule& rule, int max_exp) : pw(rule.weights.size()) {
        for (std::size_t i = 0; i < rule.weights.size(); ++i) {
            for (Eigen::Index j = 0; j < 3; ++j) {
                auto& v = pw[i][static_cast<std::size_t>(j)];
                v.resize(static_cast<std::size_t>(max_exp) + 1);
                v[0] = 1.0L;
                for (std::size_t e = 1; e < v.size(); ++e) {
                    v[e] = v[e - 1] * static_cast<long double>(rule.barycentric[i](j));
                }
            }
        }
    }
};

/// Rule applied to l1^a l2^b l3^c (weights summing to 1, no area factor).
long double rule_barycentric_monomial(const TriangleRule& rule, const PowerTable& t, int a, int b,
                                      int c) {
    const auto ua = static_cast<std::size_t>(a);
    const auto ub = static_cast<std::size_t>(b);
    const auto uc = static_cast<std::size_t>(c);
    long double s = 0.0L;
    for (std::size_t i = 0; i < rule.weights.size(); ++i) {
        s += static_cast<long double>(rule.weights[i]) * t.pw[i][0][ua] * t.pw[i][1][ub] *
             t.pw[i][2][uc];
    }
    return s;
}

Real rel_err(long double got, long double exact) {
    return static_cast<Real>(std::abs(got - exact) / std::abs(exact));
}

}  // namespace

TEST_CASE("triangle rules: point counts, weight sums, barycentric sums", "[kernels]") {
    for (int d = 1; d <= kMaxDegree; ++d) {
        CAPTURE(d);
        const TriangleRule& rule = triangle_rule(d);
        REQUIRE(rule.barycentric.size() == rule.weights.size());
        CHECK(rule.weights.size() == kDunavantPoints[static_cast<std::size_t>(d - 1)]);
        long double wsum = 0.0L;
        for (const Real w : rule.weights) {
            wsum += static_cast<long double>(w);
        }
        CHECK(static_cast<Real>(std::abs(wsum - 1.0L)) < 1e-14);
        for (const Vec3& l : rule.barycentric) {
            CHECK(std::abs(l.sum() - 1.0) < 1e-14);
        }
    }
}

TEST_CASE("triangle rules integrate all monomials up to their degree exactly", "[kernels]") {
    for (int d = 1; d <= kMaxDegree; ++d) {
        const TriangleRule& rule = triangle_rule(d);
        Real worst = 0.0;
        for (int a = 0; a <= d; ++a) {
            for (int b = 0; a + b <= d; ++b) {
                worst = std::max(worst, rel_err(rule_monomial(rule, a, b), exact_monomial(a, b)));
            }
        }
        CAPTURE(d, worst);
        CHECK(worst < 1e-13);
    }
}

TEST_CASE("triangle rules are not exact one degree higher", "[kernels]") {
    // None of Dunavant's rules 1..20 is accidentally of higher degree, so for every degree
    // some monomial of total degree d + 1 must be integrated inexactly (well above rounding).
    for (int d = 1; d <= kMaxDegree; ++d) {
        const TriangleRule& rule = triangle_rule(d);
        Real worst = 0.0;
        for (int a = 0; a <= d + 1; ++a) {
            const int b = d + 1 - a;
            worst = std::max(worst, rel_err(rule_monomial(rule, a, b), exact_monomial(a, b)));
        }
        CAPTURE(d, worst);
        CHECK(worst > 1e-9);
    }
}

TEST_CASE("triangle_rule_is_positive_interior matches the rules and the documentation",
          "[kernels]") {
    // Documented in quadrature.hpp: negative weights (3, 7, 18, 20) and/or points outside the
    // triangle (11, 15, 16, 18, 20).
    const std::set<int> documented_exceptions = {3, 7, 11, 15, 16, 18, 20};
    for (int d = 1; d <= kMaxDegree; ++d) {
        CAPTURE(d);
        const TriangleRule& rule = triangle_rule(d);
        const Real min_w = *std::min_element(rule.weights.begin(), rule.weights.end());
        Real min_l = 1.0;
        for (const Vec3& l : rule.barycentric) {
            min_l = std::min(min_l, l.minCoeff());
        }
        CAPTURE(min_w, min_l);
        const bool pi = triangle_rule_is_positive_interior(d);
        // The helper agrees with an independent inspection of the points and weights ...
        CHECK(pi == (min_w > 0.0 && min_l > 0.0));
        // ... and with the documented exception list, whose degrees really violate it.
        CHECK(pi == (documented_exceptions.count(d) == 0));
        if (!pi) {
            CHECK((min_w < 0.0 || min_l < 0.0));
        }
    }
    CHECK_THROWS_AS((void)triangle_rule_is_positive_interior(0), std::invalid_argument);
    CHECK_THROWS_AS((void)triangle_rule_is_positive_interior(21), std::invalid_argument);
}

TEST_CASE("triangle rules are invariant under permutations of the barycentric coordinates",
          "[kernels]") {
    for (int d = 1; d <= kMaxDegree; ++d) {
        const TriangleRule& rule = triangle_rule(d);
        const PowerTable table(rule, d + 2);
        Real worst = 0.0;
        for (int a = 0; a <= d + 2; ++a) {
            for (int b = 0; a + b <= d + 2; ++b) {
                for (int c = 0; a + b + c <= d + 2; ++c) {
                    const long double ref = rule_barycentric_monomial(rule, table, a, b, c);
                    const std::array<std::array<int, 3>, 5> perms = {{
                        {a, c, b},
                        {b, a, c},
                        {b, c, a},
                        {c, a, b},
                        {c, b, a},
                    }};
                    for (const auto& p : perms) {
                        worst = std::max(
                            worst,
                            rel_err(rule_barycentric_monomial(rule, table, p[0], p[1], p[2]), ref));
                    }
                }
            }
        }
        CAPTURE(d, worst);
        CHECK(worst < 1e-13);
    }
}

TEST_CASE("triangle rules converge for a smooth function on a physical triangle", "[kernels]") {
    // f = exp(u) with u = x + 2y linear: the exact integral is 2 A times the second divided
    // difference of exp at the vertex values u_1, u_2, u_3.
    const Vec3 v1(0.1, 0.2, 0.0), v2(0.9, 0.3, 0.0), v3(0.4, 0.8, 0.0);
    const Real area =
        0.5 * std::abs((v2(0) - v1(0)) * (v3(1) - v1(1)) - (v3(0) - v1(0)) * (v2(1) - v1(1)));
    const auto u = [](const Vec3& p) { return static_cast<long double>(p(0) + 2.0 * p(1)); };
    const long double u1 = u(v1), u2 = u(v2), u3 = u(v3);
    const long double dd = std::exp(u1) / ((u1 - u2) * (u1 - u3)) +
                           std::exp(u2) / ((u2 - u1) * (u2 - u3)) +
                           std::exp(u3) / ((u3 - u1) * (u3 - u2));
    const auto exact = static_cast<Real>(2.0L * static_cast<long double>(area) * dd);

    std::array<Real, kMaxDegree> err{};
    for (int d = 1; d <= kMaxDegree; ++d) {
        const TriangleRule& rule = triangle_rule(d);
        Real s = 0.0;
        for (std::size_t i = 0; i < rule.weights.size(); ++i) {
            const Vec3& l = rule.barycentric[i];
            const Vec3 p = l(0) * v1 + l(1) * v2 + l(2) * v3;
            s += rule.weights[i] * area * std::exp(p(0) + 2.0 * p(1));
        }
        err[static_cast<std::size_t>(d - 1)] = std::abs(s - exact) / exact;
    }
    CAPTURE(err);
    CHECK(err[0] < 1e-1);
    CHECK(err[3] < err[0] * 1e-3);  // degree 4
    CHECK(err[7] < err[3] * 1e-3);  // degree 8
    CHECK(err[11] < 1e-12);         // degree 12: already at rounding level
    CHECK(err[19] < 1e-12);         // degree 20
}

TEST_CASE("triangle_rule caches rules and rejects invalid degrees", "[kernels]") {
    const TriangleRule& a = triangle_rule(7);
    const TriangleRule& b = triangle_rule(7);
    CHECK(&a == &b);
    CHECK(&triangle_rule(20) == &triangle_rule(20));
    CHECK_THROWS_AS((void)triangle_rule(0), std::invalid_argument);
    CHECK_THROWS_AS((void)triangle_rule(-3), std::invalid_argument);
    CHECK_THROWS_AS((void)triangle_rule(21), std::invalid_argument);
}

namespace {

/// Checks node count, ordering, symmetry, positivity, weight sum and exactness of a rule.
void check_gauss_legendre(int n) {
    CAPTURE(n);
    const LineRule rule = gauss_legendre(n);
    const auto un = static_cast<std::size_t>(n);
    REQUIRE(rule.nodes.size() == un);
    REQUIRE(rule.weights.size() == un);
    long double wsum = 0.0L;
    for (std::size_t i = 0; i < un; ++i) {
        CHECK(rule.nodes[i] > -1.0);
        CHECK(rule.nodes[i] < 1.0);
        if (i > 0) {
            CHECK(rule.nodes[i] > rule.nodes[i - 1]);
        }
        CHECK(std::abs(rule.nodes[i] + rule.nodes[un - 1 - i]) < 1e-15);
        CHECK(rule.weights[i] > 0.0);
        CHECK(std::abs(rule.weights[i] - rule.weights[un - 1 - i]) < 1e-15);
        wsum += static_cast<long double>(rule.weights[i]);
    }
    CHECK(static_cast<Real>(std::abs(wsum - 2.0L)) < 1e-14);

    Real worst = 0.0;
    for (int k = 0; k <= 2 * n - 1; ++k) {
        long double s = 0.0L;
        for (std::size_t i = 0; i < un; ++i) {
            s += static_cast<long double>(rule.weights[i]) *
                 ipow(static_cast<long double>(rule.nodes[i]), k);
        }
        // Odd powers integrate to 0 (absolute check), even powers to 2/(k+1) (relative).
        const long double exact = (k % 2 == 0) ? 2.0L / static_cast<long double>(k + 1) : 0.0L;
        const long double scale = (k % 2 == 0) ? exact : 1.0L;
        worst = std::max(worst, static_cast<Real>(std::abs(s - exact) / scale));
    }
    CAPTURE(worst);
    CHECK(worst < 1e-13);
}

}  // namespace

TEST_CASE("Gauss-Legendre rules n = 1..64 and n = 100", "[kernels]") {
    for (int n = 1; n <= 64; ++n) {
        check_gauss_legendre(n);
    }
    check_gauss_legendre(100);
}

TEST_CASE("Gauss-Legendre rules are not exact for x^(2n)", "[kernels]") {
    // The quadrature error for x^(2n) is E_n = 2^(2n+1) (n!)^4 / ((2n+1) ((2n)!)^2) > 0.
    // It drops below double rounding for large n (E_20 / exact ~ 5e-11), so the check runs
    // for n = 1..20 and compares the computed error with E_n.
    for (int n = 1; n <= 20; ++n) {
        CAPTURE(n);
        const LineRule rule = gauss_legendre(n);
        const int k = 2 * n;
        long double s = 0.0L;
        for (std::size_t i = 0; i < rule.nodes.size(); ++i) {
            s += static_cast<long double>(rule.weights[i]) *
                 ipow(static_cast<long double>(rule.nodes[i]), k);
        }
        const long double exact = 2.0L / static_cast<long double>(k + 1);
        const long double fn = factorial(n);
        const long double f2n = factorial(2 * n);
        const long double e_n = std::pow(2.0L, static_cast<long double>(2 * n + 1)) * fn * fn * fn *
                                fn / (static_cast<long double>(2 * n + 1) * f2n * f2n);
        const long double err = exact - s;
        CAPTURE(static_cast<Real>(err), static_cast<Real>(e_n));
        CHECK(err > 0.0L);
        CHECK(static_cast<Real>(std::abs(err - e_n) / e_n) < 1e-3);
    }
}

TEST_CASE("Gauss-Legendre n = 5 matches the closed form", "[kernels]") {
    const long double r = std::sqrt(10.0L / 7.0L);
    const long double x1 = std::sqrt(5.0L - 2.0L * r) / 3.0L;
    const long double x2 = std::sqrt(5.0L + 2.0L * r) / 3.0L;
    const long double s70 = std::sqrt(70.0L);
    const long double w0 = 128.0L / 225.0L;
    const long double w1 = (322.0L + 13.0L * s70) / 900.0L;
    const long double w2 = (322.0L - 13.0L * s70) / 900.0L;
    const std::array<long double, 5> xs = {-x2, -x1, 0.0L, x1, x2};
    const std::array<long double, 5> ws = {w2, w1, w0, w1, w2};
    const LineRule rule = gauss_legendre(5);
    REQUIRE(rule.nodes.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        CAPTURE(i);
        CHECK(std::abs(rule.nodes[i] - static_cast<Real>(xs[i])) < 1e-15);
        CHECK(std::abs(rule.weights[i] - static_cast<Real>(ws[i])) < 1e-15);
    }
}

TEST_CASE("gauss_legendre rejects n < 1", "[kernels]") {
    CHECK_THROWS_AS((void)gauss_legendre(0), std::invalid_argument);
    CHECK_THROWS_AS((void)gauss_legendre(-1), std::invalid_argument);
    CHECK(gauss_legendre(1).nodes == std::vector<Real>{0.0});
    CHECK(gauss_legendre(1).weights == std::vector<Real>{2.0});
}
