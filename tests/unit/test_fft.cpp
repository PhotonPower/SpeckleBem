/// Unit tests of core::fft2 and core::next_smooth (WP-T3): normalisation and sign of the
/// transform as stated in core/fft.hpp, round trip, non-square shapes against a direct DFT,
/// determinism and the 5-smooth padding length. The bitwise guards of the two callers are the
/// pinned cases in test_rough_surface.cpp and test_speckle_statistics.cpp.
#include "specklebem/core/fft.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

using namespace specklebem;

namespace {

constexpr Real kTwoPi = 2.0 * constants::pi;

/// Uniform in [-1, 1) from the top 53 bits of std::mt19937_64 (fully specified by the
/// standard, unlike std::uniform_real_distribution).
MatrixXc random_matrix(Index rows, Index cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&rng] { return static_cast<Real>(rng() >> 11) * 0x1.0p-52 - 1.0; };
    MatrixXc a(rows, cols);
    for (Index c = 0; c < cols; ++c) {
        for (Index r = 0; r < rows; ++r) {
            const Real re = u();
            a(r, c) = Complex(re, u());
        }
    }
    return a;
}

/// Direct O((rows cols)^2) forward DFT with the convention of core/fft.hpp.
MatrixXc dft2_direct(const MatrixXc& a) {
    const Index rows = a.rows();
    const Index cols = a.cols();
    MatrixXc out(rows, cols);
    for (Index q = 0; q < cols; ++q) {
        for (Index p = 0; p < rows; ++p) {
            Complex s(0.0, 0.0);
            for (Index c = 0; c < cols; ++c) {
                for (Index r = 0; r < rows; ++r) {
                    // Reduce the phase index first so that the angle stays in [0, 2 pi).
                    const Real phase =
                        kTwoPi * (static_cast<Real>((p * r) % rows) / static_cast<Real>(rows) +
                                  static_cast<Real>((q * c) % cols) / static_cast<Real>(cols));
                    s += a(r, c) * std::polar(1.0, -phase);
                }
            }
            out(p, q) = s;
        }
    }
    return out;
}

Real max_abs(const MatrixXc& a) {
    return a.size() == 0 ? 0.0 : a.cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("fft2: delta and constant, normalisation and sign of the transform", "[core]") {
    for (const auto& [rows, cols] : std::vector<std::pair<Index, Index>>{{8, 8}, {6, 10}, {1, 5}}) {
        CAPTURE(rows, cols);
        const Real n = static_cast<Real>(rows * cols);
        // Delta at the origin -> constant 1 (forward unscaled).
        MatrixXc a = MatrixXc::Zero(rows, cols);
        a(0, 0) = 1.0;
        core::fft2(a, false);
        CHECK(max_abs(a - MatrixXc::Ones(rows, cols)) < 1e-14);
        // Inverse of the constant 1 -> delta of height 1 (inverse scaled by 1 / (rows cols)).
        core::fft2(a, true);
        MatrixXc delta = MatrixXc::Zero(rows, cols);
        delta(0, 0) = 1.0;
        CHECK(max_abs(a - delta) < 1e-14);
        // Shifted delta at (r0, c0) -> exp(-2 pi j (p r0 / rows + q c0 / cols)): the sign.
        const Index r0 = rows - 1;
        const Index c0 = cols > 2 ? 2 : 0;
        MatrixXc b = MatrixXc::Zero(rows, cols);
        b(r0, c0) = 1.0;
        core::fft2(b, false);
        for (Index q = 0; q < cols; ++q) {
            for (Index p = 0; p < rows; ++p) {
                const Real phase = kTwoPi * (static_cast<Real>(p * r0) / static_cast<Real>(rows) +
                                             static_cast<Real>(q * c0) / static_cast<Real>(cols));
                CHECK(std::abs(b(p, q) - std::polar(1.0, -phase)) < 1e-13);
            }
        }
        // Constant c -> c (rows cols) in bin (0, 0), zero elsewhere.
        MatrixXc k = MatrixXc::Constant(rows, cols, Complex(0.5, -2.0));
        core::fft2(k, false);
        CHECK(std::abs(k(0, 0) - Complex(0.5, -2.0) * n) < 1e-13 * n);
        k(0, 0) = 0.0;
        CHECK(max_abs(k) < 1e-13 * n);
    }
}

TEST_CASE("fft2: a plane wave goes to a single bin (non-square)", "[core]") {
    const Index rows = 6;
    const Index cols = 10;
    const Index p0 = 2;
    const Index q0 = 7;
    MatrixXc a(rows, cols);
    for (Index c = 0; c < cols; ++c) {
        for (Index r = 0; r < rows; ++r) {
            const Real phase = kTwoPi * (static_cast<Real>(p0 * r) / static_cast<Real>(rows) +
                                         static_cast<Real>(q0 * c) / static_cast<Real>(cols));
            a(r, c) = std::polar(1.0, phase);
        }
    }
    const MatrixXc plane = a;
    core::fft2(a, false);
    const Real n = static_cast<Real>(rows * cols);
    CHECK(std::abs(a(p0, q0) - n) < 1e-12 * n);
    a(p0, q0) = 0.0;
    CHECK(max_abs(a) < 1e-12 * n);
    // The inverse of the single bin of height 1 is the plane wave divided by rows cols.
    MatrixXc bin = MatrixXc::Zero(rows, cols);
    bin(p0, q0) = 1.0;
    core::fft2(bin, true);
    CHECK(max_abs(bin * n - plane) < 1e-13);
}

TEST_CASE("fft2: non-square and odd shapes equal the direct DFT", "[core]") {
    std::uint64_t seed = 3001;
    for (const auto& [rows, cols] :
         std::vector<std::pair<Index, Index>>{{5, 12}, {12, 5}, {7, 9}, {1, 1}, {1, 15}, {30, 2}}) {
        CAPTURE(rows, cols);
        const MatrixXc x = random_matrix(rows, cols, seed++);
        MatrixXc a = x;
        core::fft2(a, false);
        const MatrixXc ref = dft2_direct(x);
        CHECK(max_abs(a - ref) < 1e-12 * max_abs(ref));
    }
}

TEST_CASE("fft2: forward-inverse round trip", "[core]") {
    std::uint64_t seed = 3101;
    for (const auto& [rows, cols] : std::vector<std::pair<Index, Index>>{
             {8, 8}, {30, 30}, {15, 12}, {7, 11}, {64, 3}, {1, 9}, {13, 1}}) {
        CAPTURE(rows, cols);
        const MatrixXc x = random_matrix(rows, cols, seed++);
        MatrixXc a = x;
        core::fft2(a, false);
        core::fft2(a, true);
        CHECK(max_abs(a - x) < 1e-14 * max_abs(x) * std::log2(static_cast<Real>(rows * cols) + 2));
        // And the other order: inverse then forward.
        MatrixXc b = x;
        core::fft2(b, true);
        core::fft2(b, false);
        CHECK(max_abs(b - x) < 1e-14 * max_abs(x) * std::log2(static_cast<Real>(rows * cols) + 2));
    }
}

TEST_CASE("fft2: deterministic, two calls are bitwise equal", "[core]") {
    for (const bool inverse : {false, true}) {
        CAPTURE(inverse);
        const MatrixXc x = random_matrix(30, 24, 3201);
        MatrixXc a = x;
        MatrixXc b = x;
        core::fft2(a, inverse);
        core::fft2(b, inverse);
        REQUIRE(a.size() == b.size());
        CHECK(std::memcmp(a.data(), b.data(),
                          sizeof(Complex) * static_cast<std::size_t>(a.size())) == 0);
        CHECK(max_abs(a - x) > 0.0);  // it did transform
    }
}

TEST_CASE("fft2: matrices without entries are left unchanged", "[core]") {
    for (const auto& [rows, cols] : std::vector<std::pair<Index, Index>>{{0, 0}, {0, 4}, {3, 0}}) {
        CAPTURE(rows, cols);
        MatrixXc a(rows, cols);
        core::fft2(a, false);
        core::fft2(a, true);
        CHECK(a.rows() == rows);
        CHECK(a.cols() == cols);
    }
}

TEST_CASE("next_smooth: smallest 5-smooth length >= n", "[core]") {
    const std::vector<std::pair<Index, Index>> table = {
        {-3, 1},  {0, 1},   {1, 1},   {2, 2},    {7, 8},     {11, 12},    {13, 15},
        {28, 30}, {31, 32}, {49, 50}, {97, 100}, {121, 125}, {1025, 1080}};
    for (const auto& [n, m] : table) {
        CAPTURE(n);
        CHECK(core::next_smooth(n) == m);
    }
    const auto smooth = [](Index m) {
        for (const Index p : {Index{2}, Index{3}, Index{5}}) {
            while (m % p == 0) m /= p;
        }
        return m == 1;
    };
    for (Index n = 1; n <= 2000; ++n) {
        const Index m = core::next_smooth(n);
        CAPTURE(n, m);
        REQUIRE(m >= n);
        REQUIRE(smooth(m));
        for (Index k = n; k < m; ++k) REQUIRE_FALSE(smooth(k));
    }
}
