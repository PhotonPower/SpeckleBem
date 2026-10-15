// Zone and statistics logic of the WP24 near-field check (tests/support/ag_near_field_support.hpp,
// used by tests/validation/test_ag_near_field.cpp, tests/validation_large/
// test_ag_near_field_large.cpp and benchmarks/ag_near_field.cpp): the zone rule of docs/05
// (shadow = geometric shadow |x| < a, z > 0; lit = every other exterior point beyond d_min),
// nearest-rank percentiles, grid sizes and the shell points on the x-axis. No field evaluation:
// the zone statistics are fed with synthetic evaluations, so every case runs in milliseconds.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ag_near_field_support.hpp"

using namespace ag_near_field;

namespace {

constexpr Real kA = 0.5e-6;    // sphere radius [m]
constexpr Real kDMin = 10e-9;  // exclusion distance [m]
constexpr Real kEps = 1e-12;   // [m], << kDMin

/// Synthetic evaluation at `pts`: e = base everywhere, |E_Mie| = 1.
Evaluation flat_evaluation(const Vertices& pts, Real base) {
    Evaluation ev;
    const Index n = pts.rows();
    ev.e_sim = FieldMatrix::Zero(n, 3);
    ev.e_mie = FieldMatrix::Zero(n, 3);
    ev.e_mie.col(0).setConstant(Complex(1.0, 0.0));
    ev.rel = VectorXr::Constant(n, base);
    ev.abs = VectorXr::Constant(n, base);
    ev.mag_rel = VectorXr::Constant(n, 0.5 * base);
    return ev;
}

/// Point at signed surface distance d in direction psi from +z in the xz-plane (as shells()).
Vec3 at_distance(Real d, Real psi) {
    return (kA + d) * Vec3(std::sin(psi), 0.0, std::cos(psi));
}

}  // namespace

TEST_CASE("ag_near_field: points on the x-axis are lit whatever the rounding of z",
          "[ag_near_field]") {
    for (const Real z : {1e-22, -1e-22, 0.0}) {
        CHECK(classify(Vec3(kA + 2.0 * kDMin, 0.0, z), kA, kDMin) == Zone::lit);
        CHECK(classify(Vec3(-(kA + 2.0 * kDMin), 0.0, z), kA, kDMin) == Zone::lit);
        CHECK(classify(Vec3(3.0 * kA, 0.0, z), kA, kDMin) == Zone::lit);
    }
    // The shell point at psi = 90 deg: z = (a + d) cos(pi / 2) ~ 1e-22 m > 0.
    const Vertices sh = shells({2.0 * kDMin}, 360, kA);
    const Vec3 p90 = sh.row(90).transpose();
    REQUIRE(p90.z() > 0.0);
    REQUIRE(p90.z() < 1e-20);
    CHECK(classify(p90, kA, kDMin) == Zone::lit);
    CHECK(classify(sh.row(270).transpose(), kA, kDMin) == Zone::lit);
    CHECK(classify(sh.row(0).transpose(), kA, kDMin) == Zone::shadow);  // psi = 0: shadow pole
    CHECK(classify(sh.row(180).transpose(), kA, kDMin) == Zone::lit);   // lit pole
}

TEST_CASE("ag_near_field: near-surface band and interior at d = +-d_min +- eps",
          "[ag_near_field]") {
    for (const Real psi : {0.0, 0.3, 1.2, 0.5 * constants::pi, 2.0, constants::pi}) {
        CAPTURE(psi);
        CHECK(classify(at_distance(kDMin - kEps, psi), kA, kDMin) == Zone::band);
        CHECK(classify(at_distance(-(kDMin - kEps), psi), kA, kDMin) == Zone::band);
        CHECK(classify(at_distance(0.0, psi), kA, kDMin) == Zone::band);
        CHECK(classify(at_distance(-(kDMin + kEps), psi), kA, kDMin) == Zone::interior);
        const Zone outside = classify(at_distance(kDMin + kEps, psi), kA, kDMin);
        CHECK(outside != Zone::band);
        CHECK(outside != Zone::interior);
    }
    CHECK(classify(Vec3::Zero(), kA, kDMin) == Zone::interior);
    CHECK(classify(Vec3(0.0, 0.0, 0.5 * kA), kA, kDMin) == Zone::interior);  // not shadow
    // d_min = 0: no band.
    CHECK(classify(at_distance(kEps, 1.0), kA, 0.0) != Zone::band);
    CHECK(classify(at_distance(-kEps, 1.0), kA, 0.0) == Zone::interior);
}

TEST_CASE("ag_near_field: the shadow is the geometric shadow, the forward region beside it is lit",
          "[ag_near_field]") {
    // Geometric shadow |x| < a, z > 0 (outside the band).
    CHECK(classify(Vec3(0.0, 0.0, kA + 2.0 * kDMin), kA, kDMin) == Zone::shadow);
    CHECK(classify(Vec3(0.99 * kA, 0.0, 0.5 * kA), kA, kDMin) == Zone::shadow);
    CHECK(classify(Vec3(-0.5 * kA, 0.0, 5.0 * kA), kA, kDMin) == Zone::shadow);
    // Beside it: |x| >= a, z > 0.
    CHECK(classify(Vec3(kA, 0.0, 0.5 * kA), kA, kDMin) == Zone::lit);
    CHECK(classify(Vec3(-1.01 * kA, 0.0, 0.5 * kA), kA, kDMin) == Zone::lit);
    CHECK(classify(Vec3(2.0 * kA, 0.0, 5.0 * kA), kA, kDMin) == Zone::lit);
    // In front of the sphere (z < 0) everything is lit, also |x| < a.
    CHECK(classify(Vec3(0.0, 0.0, -(kA + 2.0 * kDMin)), kA, kDMin) == Zone::lit);
    CHECK(classify(Vec3(0.5 * kA, 0.0, -5.0 * kA), kA, kDMin) == Zone::lit);
    // Out of the plane the shadow is the cylinder x^2 + y^2 < a^2.
    CHECK(classify(Vec3(0.0, 0.9 * kA, 2.0 * kA), kA, kDMin) == Zone::shadow);
    CHECK(classify(Vec3(0.8 * kA, 0.8 * kA, 2.0 * kA), kA, kDMin) == Zone::lit);
    CHECK(in_geometric_shadow(Vec3(0.5 * kA, 0.0, 1e-30), kA));
    CHECK_FALSE(in_geometric_shadow(Vec3(0.5 * kA, 0.0, 0.0), kA));
}

TEST_CASE("ag_near_field: nearest-rank percentiles and the location of the maximum",
          "[ag_near_field]") {
    SECTION("100 values") {
        std::vector<Real> v;
        std::vector<Vec3> w;
        for (int i = 0; i < 100; ++i) {
            const int value = (37 * i) % 100 + 1;  // permutation of 1 ... 100
            v.push_back(static_cast<Real>(value));
            w.emplace_back(static_cast<Real>(value), 0.0, 0.0);
        }
        const Stats s = stats(v, w);
        CHECK(s.count == 100);
        CHECK(s.max == 100.0);
        CHECK(s.p99 == 99.0);
        CHECK(s.p95 == 95.0);
        CHECK(s.median == 50.0);
        CHECK(s.argmax.x() == 100.0);
    }
    SECTION("10 values: ranks ceil(q n)") {
        const std::vector<Real> v{3, 9, 1, 10, 5, 2, 8, 4, 7, 6};
        std::vector<Vec3> w;
        for (std::size_t i = 0; i < v.size(); ++i) w.emplace_back(static_cast<Real>(i), 0.0, 0.0);
        const Stats s = stats(v, w);
        CHECK(s.max == 10.0);
        CHECK(s.p99 == 10.0);    // rank ceil(9.9) = 10
        CHECK(s.p95 == 10.0);    // rank ceil(9.5) = 10
        CHECK(s.median == 5.0);  // rank ceil(5) = 5
        CHECK(s.argmax.x() == 3.0);
    }
    SECTION("one value and none") {
        const Stats one = stats({0.25}, {Vec3(1.0, 2.0, 3.0)});
        CHECK(one.count == 1);
        CHECK(one.max == 0.25);
        CHECK(one.p99 == 0.25);
        CHECK(one.median == 0.25);
        CHECK(one.argmax == Vec3(1.0, 2.0, 3.0));
        const Stats none = stats({}, {});
        CHECK(none.count == 0);
        CHECK(none.max == 0.0);
    }
}

TEST_CASE("ag_near_field: grid sizes are odd so that the grid contains both axes",
          "[ag_near_field]") {
    CHECK(grid_points(1.0, 0.5) == 5);                  // ceil(4) + 1 = 5
    CHECK(grid_points(1.0, 0.4) == 7);                  // ceil(5) + 1 = 6 -> 7
    CHECK(grid_points(1.0, 0.3) == 9);                  // ceil(6.67) + 1 = 8 -> 9
    CHECK(grid_points(1.5e-6, kLambda / 20.0) == 123);  // the record's 1 um grid
    CHECK(grid_points(6e-6, kLambda / 10.0) == 243);    // the record's 4 um grid
    for (const Real spacing : {0.31, 0.4, 0.5, 0.7}) {
        const Index n = grid_points(1.0, spacing);
        CHECK(n % 2 == 1);
        CHECK(2.0 / static_cast<Real>(n - 1) <= spacing);
        const Vertices g = xz_grid(1.0, n);
        REQUIRE(g.rows() == n * n);
        const Index mid = (n - 1) / 2;
        CHECK(g(mid * n + mid, 0) == 0.0);  // the centre (x_i outer, z_j inner)
        CHECK(g(mid * n + mid, 2) == 0.0);
        for (Index k = 0; k < n; ++k) {
            CHECK(g(k * n + mid, 2) == 0.0);  // the line z = 0
            CHECK(g(mid * n + k, 0) == 0.0);  // the line x = 0
        }
    }
}

TEST_CASE("ag_near_field: zone statistics apply the strict lit bound beside the shadow",
          "[ag_near_field]") {
    const Real d = 2.0 * kDMin;
    Vertices pts(8, 3);
    pts.row(0) = Vec3(0.0, 0.0, -(kA + d)).transpose();         // lit pole
    pts.row(1) = Vec3(kA + d, 0.0, 1e-22).transpose();          // x-axis, rounded z > 0
    pts.row(2) = Vec3(-(kA + d), 0.0, -1e-22).transpose();      // x-axis, rounded z < 0
    pts.row(3) = Vec3(1.2 * kA, 0.0, 1.5 * kA).transpose();     // forward, beside the shadow
    pts.row(4) = Vec3(0.0, 0.0, kA + d).transpose();            // shadow pole
    pts.row(5) = Vec3(0.5 * kA, 0.0, 2.0 * kA).transpose();     // geometric shadow
    pts.row(6) = Vec3(0.0, 0.0, kA + 0.5 * kDMin).transpose();  // band
    pts.row(7) = Vec3(0.0, 0.0, -0.5 * kA).transpose();         // interior
    Evaluation ev = flat_evaluation(pts, 0.01);
    ev.rel(5) = 0.5;  // allowed in the shadow
    ev.rel(6) = 5.0;  // band: ignored
    ev.abs(7) = 0.3;
    ZoneStats z = zone_stats(pts, ev, kA, kDMin);
    CHECK(z.band == 1);
    CHECK(z.lit.count == 4);
    CHECK(z.lit_x_axis.count == 2);
    CHECK(z.lit_forward.count == 1);
    CHECK(z.shadow.count == 2);
    CHECK(z.interior_abs.count == 1);
    CHECK(z.interior_abs.max == 0.3);
    CHECK(z.interior_max_mie == 1.0);
    CHECK(z.lit.max == 0.01);
    CHECK(z.shadow.max == 0.5);
    CHECK(z.shadow.argmax == Vec3(0.5 * kA, 0.0, 2.0 * kA));
    CHECK(z.lit_mag.max == 0.005);
    CHECK(passes(z));
    // An error of 0.2 beside the shadow fails (the old z > 0 half-space rule would accept it).
    ev.rel(3) = 0.2;
    z = zone_stats(pts, ev, kA, kDMin);
    CHECK(z.lit.max == 0.2);
    CHECK(z.lit_forward.max == 0.2);
    CHECK(z.shadow.max == 0.5);
    CHECK_FALSE(passes(z));
    // ... and so does one on the x-axis.
    ev.rel(3) = 0.01;
    ev.rel(1) = 0.1;
    z = zone_stats(pts, ev, kA, kDMin);
    CHECK(z.lit_x_axis.max == 0.1);
    CHECK_FALSE(passes(z));
    // Shadow errors up to < 1 pass.
    ev.rel(1) = 0.01;
    ev.rel(4) = 0.99;
    CHECK(passes(zone_stats(pts, ev, kA, kDMin)));
    ev.rel(4) = 1.0;
    CHECK_FALSE(passes(zone_stats(pts, ev, kA, kDMin)));
    // Sizes must match.
    Evaluation shorter = flat_evaluation(pts.topRows(7), 0.01);
    CHECK_THROWS_AS(zone_stats(pts, shorter, kA, kDMin), std::invalid_argument);
}

TEST_CASE("ag_near_field: zones of the default shells", "[ag_near_field]") {
    const ShellSet sh = default_shells(kA);
    REQUIRE(sh.n_angles == 360);
    REQUIRE(sh.d.size() == 34);
    REQUIRE(sh.points.rows() == 34 * 360);
    const Evaluation ev = flat_evaluation(sh.points, 0.01);
    // d_min below the innermost shells (lambda / 100): every point is counted.
    const ZoneStats z = zone_stats(sh.points, ev, kA, 1e-9);
    CHECK(z.band == 0);
    CHECK(z.interior_abs.count == 9 * 360);
    CHECK(z.lit.count + z.shadow.count == 25 * 360);
    CHECK(z.lit_x_axis.count == 25 * 2);  // psi = 90 and 270 deg on every exterior shell
    // Geometric shadow on a shell of radius a + d: z > 0 and (a + d) |sin psi| < a.
    Index shadow = 0;
    for (std::size_t k = 9; k < sh.d.size(); ++k) {
        for (Index i = 0; i < 360; ++i) {
            const Real psi = 2.0 * constants::pi * static_cast<Real>(i) / 360.0;
            const Real r = kA + sh.d[k];
            if (r * std::cos(psi) > 0.0 && r * std::abs(std::sin(psi)) < kA)
                ++shadow;
        }
    }
    CHECK(z.shadow.count == shadow);
    CHECK(z.lit_forward.count == 25 * 179 - shadow);  // z > 0 off the axis, outside the shadow
    // shell_stats: the psi = 90 / 270 deg points belong to the front half z <= 0.
    const ShellStats st = shell_stats(sh, ev, 9);
    CHECK(st.d == sh.d[9]);
    CHECK(st.front.count == 181);
    CHECK(st.back.count == 179);
}

TEST_CASE("ag_near_field: d_min scan entries", "[ag_near_field]") {
    CHECK(kDMinScanPerEdge[0] == 0.0);  // no exclusion beyond the evaluation cut-off
    CHECK(kDMinScanPerEdge[2] == kDMinPerEdge);
    // d_min = 0 counts the shells at |d| = lambda / 100 (no band at all).
    const ShellSet sh = default_shells(kA);
    const ZoneStats z = zone_stats(sh.points, flat_evaluation(sh.points, 0.01), kA, 0.0);
    CHECK(z.band == 0);
    CHECK(z.lit.count + z.shadow.count + z.interior_abs.count == sh.points.rows());
    CHECK_THAT(sh.d.front(), Catch::Matchers::WithinRel(-kLambda / 5.0, 1e-12));
    CHECK_THAT(sh.d[9], Catch::Matchers::WithinRel(kLambda / 100.0, 1e-12));
    CHECK(log_space(1.0, 100.0, 3)[1] == 10.0);
    CHECK_THROWS_AS(log_space(0.0, 1.0, 3), std::invalid_argument);
}
