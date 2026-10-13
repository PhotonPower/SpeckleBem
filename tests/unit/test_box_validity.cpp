// Unit tests of the pure parts of the closing-box validity study (WP-V1 / WP-V2,
// benchmarks/box_validity.cpp, tests/support/box_validity_support.hpp): far-field grids, the
// docs/05 sensitivity metric (L2 of |F|^2 over the reflection hemisphere with sin(theta)
// weights), eps_rr, the hemisphere power, the MLFMM memory guard and the command-line options of
// the MLFMM extension.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "box_validity_support.hpp"

using namespace box_validity;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

/// Random positive hemisphere intensities (fixed seed).
MatrixXr random_hemisphere(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(0.1, 2.0);
    MatrixXr I(kHemiTheta, kHemiPhi);
    for (Index i = 0; i < I.rows(); ++i)
        for (Index j = 0; j < I.cols(); ++j) I(i, j) = u(rng);
    return I;
}

FarField random_far_field(std::uint64_t seed) {
    FarField f;
    f.refl = random_hemisphere(seed);
    f.fwd = random_hemisphere(seed + 1);
    std::mt19937_64 rng(seed + 2);
    std::uniform_real_distribution<Real> u(0.1, 2.0);
    f.cut_xz.resize(kCut, 1);
    f.cut_yz.resize(kCut, 1);
    for (Index m = 0; m < kCut; ++m) {
        f.cut_xz(m, 0) = u(rng);
        f.cut_yz(m, 0) = u(rng);
    }
    return f;
}

}  // namespace

TEST_CASE("box_validity: far-field grids", "[box_validity]") {
    const Vertices refl = hemisphere_directions(-1.0);
    const Vertices fwd = hemisphere_directions(+1.0);
    REQUIRE(refl.rows() == kHemiTheta * kHemiPhi);
    REQUIRE(fwd.rows() == kHemiTheta * kHemiPhi);
    for (Index r = 0; r < refl.rows(); ++r) {
        CHECK_THAT(refl.row(r).norm(), WithinAbs(1.0, 1e-14));
        CHECK(refl(r, 2) < 0.0);  // reflection hemisphere: k_z < 0 (light comes from z < 0)
        CHECK(fwd(r, 2) == -refl(r, 2));
    }
    // Row i * kHemiPhi + j: theta_i = (i + 1/2) deg from -z, phi_j = 3 j deg.
    const Index row = 10 * kHemiPhi + 30;  // theta 10.5 deg, phi 90 deg
    CHECK_THAT(refl(row, 0), WithinAbs(0.0, 1e-14));
    CHECK_THAT(refl(row, 1), WithinAbs(std::sin(10.5 * kDeg), 1e-14));
    CHECK_THAT(refl(row, 2), WithinAbs(-std::cos(10.5 * kDeg), 1e-14));

    const Vertices xz = cut_directions(false);
    const Vertices yz = cut_directions(true);
    REQUIRE(xz.rows() == kCut);
    for (Index m = 0; m < kCut; ++m) {
        CHECK_THAT(xz.row(m).norm(), WithinAbs(1.0, 1e-14));
        CHECK(xz(m, 1) == 0.0);
        CHECK(yz(m, 0) == 0.0);
        CHECK(xz(m, 2) < 0.0);
    }
    CHECK_THAT(xz(0, 0), WithinAbs(-std::sin(89.5 * kDeg), 1e-14));
    CHECK_THAT(xz((kCut - 1) / 2, 0), WithinAbs(0.0, 1e-14));  // specular direction -z
}

TEST_CASE("box_validity: intensity arranges |F|^2 row-major", "[box_validity]") {
    FieldMatrix F(6, 3);
    for (Index r = 0; r < 6; ++r)
        F.row(r) << Complex(static_cast<Real>(r), 1.0), Complex(0.0, 0.0), Complex(0.0, 2.0);
    const MatrixXr I = intensity(F, 2, 3);
    CHECK(I(1, 2) == 25.0 + 1.0 + 4.0);  // row 1 * 3 + 2 = 5
    CHECK(I(0, 0) == 0.0 + 1.0 + 4.0);
    CHECK_THROWS_AS(intensity(F, 2, 2), std::invalid_argument);
}

TEST_CASE("box_validity: hemisphere L2 metric (docs/05)", "[box_validity]") {
    const MatrixXr R = random_hemisphere(7);
    CHECK(hemi_l2(R, R, 0.0) == 0.0);
    // A uniform relative change a gives exactly a.
    CHECK_THAT(hemi_l2(1.003 * R, R, 0.0), WithinRel(0.003, 1e-10));
    CHECK_THAT(hemi_l2(0.99 * R, R, kOffSpecularTheta), WithinRel(0.01, 1e-10));

    // Explicit formula sqrt(sum w (T - R)^2 / sum w R^2), w = sin(theta_i), on random data.
    const MatrixXr T = R + 0.01 * random_hemisphere(8);
    Real num = 0, den = 0, num30 = 0, den30 = 0;
    for (Index i = 0; i < R.rows(); ++i) {
        const Real th = (static_cast<Real>(i) + 0.5) * kHemiDTheta;
        const Real w = std::sin(th * kDeg);
        for (Index j = 0; j < R.cols(); ++j) {
            const Real d = T(i, j) - R(i, j);
            num += w * d * d;
            den += w * R(i, j) * R(i, j);
            if (th >= 30.0) {
                num30 += w * d * d;
                den30 += w * R(i, j) * R(i, j);
            }
        }
    }
    CHECK_THAT(hemi_l2(T, R, 0.0), WithinRel(std::sqrt(num / den), 1e-12));
    CHECK_THAT(hemi_l2(T, R, 30.0), WithinRel(std::sqrt(num30 / den30), 1e-12));

    // The sin(theta) weight: one perturbed cell near the pole counts less than one at grazing
    // angles.
    MatrixXr Tp = R, Tg = R;
    Tp(0, 5) += 0.1;
    Tg(80, 5) += 0.1;
    const Real ratio = hemi_l2(Tg, R, 0.0) / hemi_l2(Tp, R, 0.0);
    CHECK_THAT(ratio, WithinRel(std::sqrt(std::sin(80.5 * kDeg) / std::sin(0.5 * kDeg)), 1e-12));

    CHECK_THROWS_AS(hemi_l2(R, R.topRows(10), 0.0), std::invalid_argument);
    CHECK_THROWS_AS(hemi_l2(R, MatrixXr::Zero(kHemiTheta, kHemiPhi), 0.0), std::invalid_argument);
}

TEST_CASE("box_validity: hemisphere power and eps_rr", "[box_validity]") {
    // Constant intensity 1: 2 pi (midpoint rule in theta, error ~ (pi/180)^2 / 24).
    const MatrixXr one = MatrixXr::Ones(kHemiTheta, kHemiPhi);
    CHECK_THAT(hemi_power(one), WithinRel(2.0 * constants::pi, 2e-5));
    // cos(theta) intensity: pi.
    MatrixXr c(kHemiTheta, kHemiPhi);
    for (Index i = 0; i < kHemiTheta; ++i)
        c.row(i).setConstant(std::cos((static_cast<Real>(i) + 0.5) * kDeg));
    CHECK_THAT(hemi_power(c), WithinRel(constants::pi, 1e-4));

    MatrixXr r(4, 1), t(4, 1);
    r << 1.0, 2.0, 4.0, 1.0;
    t << 1.0, 2.0, 4.0, 3.0;
    // sqrt(mean (t - r)^2) / max r = sqrt(4 / 4) / 4.
    CHECK_THAT(eps_rr(t, r), WithinRel(0.25, 1e-14));
    CHECK_THROWS_AS(eps_rr(t, r.topRows(3)), std::invalid_argument);
    CHECK_THROWS_AS(eps_rr(t, MatrixXr::Zero(4, 1)), std::invalid_argument);
}

TEST_CASE("box_validity: compare combines all metrics", "[box_validity]") {
    const FarField R = random_far_field(11);
    FarField T = R;
    T.refl *= 1.002;
    T.fwd *= 0.95;
    const Metrics m = compare(T, R);
    CHECK_THAT(m.l2_refl, WithinRel(0.002, 1e-10));
    CHECK_THAT(m.l2_off30, WithinRel(0.002, 1e-10));
    CHECK_THAT(m.l2_fwd, WithinRel(0.05, 1e-10));
    CHECK_THAT(m.dP_refl, WithinRel(0.002, 1e-10));
    CHECK(m.eps_xz == 0.0);
    CHECK(m.eps_yz == 0.0);
    const Metrics same = compare(R, R);
    CHECK(same.l2_refl == 0.0);
    CHECK(same.dP_refl == 0.0);
    FarField bad = R;
    bad.cut_yz.conservativeResize(10, 1);  // (not x = x.topRows(): aliasing)
    CHECK_THROWS_AS(compare(bad, R), std::runtime_error);
}

TEST_CASE("box_validity: MLFMM memory guard", "[box_validity]") {
    // Krylov basis: (max_iter + 1) complex vectors of length 2N, in 1e9 bytes.
    CHECK_THAT(krylov_basis_gb(300, 510720), WithinRel(301.0 * 510720.0 * 16.0 / 1e9, 1e-14));
    CHECK_THAT(krylov_basis_gb(1, 1), WithinRel(32e-9, 1e-14));
    CHECK_THROWS_AS(krylov_basis_gb(0, 100), std::invalid_argument);
    CHECK_THROWS_AS(krylov_basis_gb(10, 0), std::invalid_argument);
    // Kept Hessenberg (--ritz): (k + 1) x k complex entries.
    CHECK_THAT(hessenberg_gb(4000), WithinRel(4001.0 * 4000.0 * 16.0 / 1e9, 1e-14));
    CHECK_THAT(hessenberg_gb(1), WithinRel(32e-9, 1e-14));
    CHECK_THROWS_AS(hessenberg_gb(0), std::invalid_argument);

    // Allowed: peak 40 + 5 = 45 <= 60 and setup 40 + 10 <= 100 available.
    CHECK_NOTHROW(check_memory_guard(40.0, 5.0, 100.0, 60.0, false));
    // Boundaries are inclusive: peak == max_gb, setup + headroom == available.
    CHECK_NOTHROW(check_memory_guard(50.0, 10.0, 60.0, 60.0, false));
    // Peak estimate above --max-gb (the Krylov basis tips it over).
    CHECK_THROWS_AS(check_memory_guard(50.0, 10.5, 200.0, 60.0, false), std::runtime_error);
    // Setup peak plus headroom above the available memory, peak itself within --max-gb.
    CHECK_THROWS_AS(check_memory_guard(40.0, 1.0, 49.9, 60.0, false), std::runtime_error);
    // Unknown available memory (<= 0): only the --max-gb test applies.
    CHECK_NOTHROW(check_memory_guard(40.0, 1.0, 0.0, 60.0, false));
    CHECK_NOTHROW(check_memory_guard(40.0, 1.0, -1.0, 60.0, false));
    CHECK_THROWS_AS(check_memory_guard(70.0, 1.0, 0.0, 60.0, false), std::runtime_error);
    // --force skips both tests ...
    CHECK_NOTHROW(check_memory_guard(500.0, 50.0, 20.0, 60.0, true));
    // ... but not the argument checks.
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    CHECK_THROWS_AS(check_memory_guard(nan, 1.0, 100.0, 60.0, true), std::invalid_argument);
    CHECK_THROWS_AS(check_memory_guard(1.0, -1.0, 100.0, 60.0, false), std::invalid_argument);
    CHECK_THROWS_AS(check_memory_guard(1.0, 1.0, 100.0, 0.0, false), std::invalid_argument);
    CHECK_THROWS_AS(check_memory_guard(1.0, 1.0, nan, 60.0, false), std::invalid_argument);
}

TEST_CASE("box_validity: options (dense defaults, MLFMM extension)", "[box_validity]") {
    const Options d = parse_options({});
    CHECK(d.compression == "dense");
    CHECK(d.material == "ag");
    CHECK(d.digits == 3.0);
    CHECK(d.leaf_quantile == 1.0);
    CHECK_FALSE(d.jacobi.has_value());
    CHECK(d.parts);
    CHECK_FALSE(d.estimate_only);
    CHECK(d.box_mesh_size == "400e-9");
    CHECK(d.tol == 1e-6);

    const Options m = parse_options({"--compression",
                                     "mlfmm",
                                     "--material",
                                     "si",
                                     "--L",
                                     "5e-6",
                                     "--L-gen",
                                     "6e-6",
                                     "--waist",
                                     "1e-6",
                                     "--depth",
                                     "11.3e-6",
                                     "--box-mesh-size",
                                     "uniform",
                                     "--fine-band",
                                     "--d0",
                                     "5",
                                     "--leaf-quantile",
                                     "0.8",
                                     "--jacobi",
                                     "1",
                                     "--tol",
                                     "1e-5",
                                     "--max-iter",
                                     "4000",
                                     "--estimate-only",
                                     "--force",
                                     "--parts",
                                     "0",
                                     "--history",
                                     "h.txt",
                                     "--tag",
                                     "si5",
                                     "--reference",
                                     "a",
                                     "--reference",
                                     "b"});
    CHECK(m.compression == "mlfmm");
    CHECK(m.material == "si");
    CHECK(m.L == 5e-6);
    REQUIRE(m.L_gen.has_value());
    CHECK(*m.L_gen == 6e-6);
    REQUIRE(m.waist.has_value());
    CHECK(*m.waist == 1e-6);
    REQUIRE(m.depth.has_value());
    CHECK(*m.depth == 11.3e-6);
    CHECK(m.box_mesh_size == "uniform");
    CHECK(m.fine_band);
    CHECK(m.digits == 5.0);
    CHECK(m.leaf_quantile == 0.8);
    REQUIRE(m.jacobi.has_value());
    CHECK(*m.jacobi);
    CHECK(m.tol == 1e-5);
    CHECK(m.max_iter == 4000);
    CHECK(m.estimate_only);
    CHECK(m.force);
    CHECK_FALSE(m.parts);
    CHECK(m.history == "h.txt");
    CHECK(m.tag == "si5");
    CHECK(m.references == std::vector<std::string>{"a", "b"});
    CHECK_FALSE(parse_options({"--jacobi", "auto"}).jacobi.has_value());
    CHECK_FALSE(*parse_options({"--jacobi", "0"}).jacobi);

    const Options c = parse_options({"--compare", "A", "B", "C", "--allow-unconverged"});
    CHECK(c.compare == std::vector<std::string>{"A", "B", "C"});
    CHECK(c.allow_unconverged);

    using V = std::vector<std::string>;
    CHECK_THROWS_AS(parse_options(V{"--compression", "aca"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--d0", "0"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--leaf-quantile", "1.5"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--jacobi", "yes"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--parts", "2"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--L", "4e-6", "--L-gen", "3e-6"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--eps-imag", "1"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--beam", "gauss"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--tol"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--compare", "A"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--frobnicate"}), std::invalid_argument);
}

TEST_CASE("box_validity: options of the WP-V3 diagnostics", "[box_validity]") {
    const Options d = parse_options({});
    CHECK(d.sigma == 50e-9);
    CHECK_FALSE(d.save_currents);
    CHECK_FALSE(d.mechanism);
    CHECK(d.ritz.empty());
    CHECK(d.load_currents.empty());
    CHECK(d.diff.empty());
    CHECK(d.depth_fit.empty());

    const Options s = parse_options(
        {"--sigma", "0", "--out", "run", "--save-currents", "--mechanism", "--ritz", "250,1000"});
    CHECK(s.sigma == 0.0);
    CHECK(s.save_currents);
    CHECK(s.mechanism);
    CHECK(s.ritz == std::vector<Index>{250, 1000});
    CHECK(parse_options({"--ritz", "300"}).ritz == std::vector<Index>{300});

    const Options l = parse_options({"--load-currents", "A", "--mechanism", "--diff", "B"});
    CHECK(l.load_currents == "A");
    CHECK(l.diff == "B");

    const Options f = parse_options({"--depth-fit", "A", "B", "C", "D"});
    CHECK(f.depth_fit == std::vector<std::string>{"A", "B", "C", "D"});

    using V = std::vector<std::string>;
    CHECK_THROWS_AS(parse_options(V{"--sigma", "-1e-9"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--ritz", "0"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--ritz", "10,x"}), std::invalid_argument);
    // A Krylov dimension beyond --max-iter is refused, not clamped.
    CHECK(parse_options(V{"--max-iter", "4000", "--ritz", "250,4000"}).ritz ==
          std::vector<Index>{250, 4000});
    CHECK_THROWS_AS(parse_options(V{"--max-iter", "4000", "--ritz", "250,100000"}),
                    std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--save-currents"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--load-currents", "A", "--out", "B"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--load-currents", "A", "--ritz", "10"}),
                    std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--diff", "B"}), std::invalid_argument);
    CHECK_THROWS_AS(parse_options(V{"--depth-fit", "A", "B"}), std::invalid_argument);
}
