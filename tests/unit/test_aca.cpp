/// @file test_aca.cpp
/// WP-PML5b: the kernel-independent partially pivoted ACA (compression/aca/aca.hpp) on synthetic
/// matrices with fixed seeds: exact low rank, the smooth kernel 1 / |x - y| and the Helmholtz
/// kernel exp(-j k |x - y|) / |x - y| between separated clusters (against the dense matrix and its
/// SVD), zero rows, zero blocks, the rank cap, the transposed use of the factors and argument
/// errors.
#include "specklebem/compression/aca/aca.hpp"

#include <Eigen/SVD>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace specklebem;

namespace {

/// Generator over an explicit matrix; counts calls.
class DenseGenerator final : public aca::EntryGenerator {
public:
    explicit DenseGenerator(const MatrixXc& a) : a_(a) {}
    void row(Index i, Complex* out) const override {
        for (Index j = 0; j < a_.cols(); ++j) out[j] = a_(i, j);
    }
    void column(Index j, Complex* out) const override {
        for (Index i = 0; i < a_.rows(); ++i) out[i] = a_(i, j);
    }

private:
    const MatrixXc& a_;
};

MatrixXc random_matrix(Index m, Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    MatrixXc a(m, n);
    for (Index j = 0; j < n; ++j)
        for (Index i = 0; i < m; ++i) a(i, j) = Complex(u(rng), u(rng));
    return a;
}

/// Uniform points in [c, c + 1]^3 (x shifted by `shift`).
std::vector<Vec3> cloud(Index n, Real shift, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(0.0, 1.0);
    std::vector<Vec3> p(static_cast<std::size_t>(n));
    for (auto& x : p) x = Vec3(shift + u(rng), u(rng), u(rng));
    return p;
}

/// exp(-j k r) / r between two clouds (k = 0: the Laplace kernel 1 / r).
MatrixXc kernel_matrix(const std::vector<Vec3>& x, const std::vector<Vec3>& y, Real k) {
    MatrixXc a(static_cast<Index>(x.size()), static_cast<Index>(y.size()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (std::size_t j = 0; j < y.size(); ++j) {
            const Real r = (x[i] - y[j]).norm();
            a(static_cast<Index>(i), static_cast<Index>(j)) = std::exp(Complex(0.0, -k * r)) / r;
        }
    }
    return a;
}

/// Smallest rank r with sigma_{r+1} <= eps ||A||_F, i.e. the optimal rank for a relative
/// Frobenius error eps (up to the tail sum).
Index svd_rank(const MatrixXc& a, Real eps) {
    const Eigen::BDCSVD<MatrixXc> svd(a);
    const VectorXr s = svd.singularValues();
    const Real fro = a.norm();
    Real tail2 = s.squaredNorm();
    for (Index r = 0; r < s.size(); ++r) {
        if (std::sqrt(tail2) <= eps * fro)
            return r;
        tail2 -= s(r) * s(r);
    }
    return s.size();
}

}  // namespace

TEST_CASE("aca: exact low-rank matrix is reproduced at its rank", "[aca]") {
    const MatrixXc a = random_matrix(60, 5, 1) * random_matrix(40, 5, 2).transpose();
    const DenseGenerator gen(a);
    aca::CompressParams p;
    p.epsilon = 1e-8;
    const aca::LowRankBlock b = aca::compress(60, 40, gen, p);
    INFO("rank " << b.rank() << ", reason " << aca::to_string(b.reason));
    CHECK(b.converged());
    CHECK(b.rank() <= 5 + p.confirmations);  // the rank + the confirming noise terms
    CHECK(aca::frobenius_error(a, b) <= p.epsilon * a.norm());
    CHECK(b.U.rows() == 60);
    CHECK(b.V.rows() == 40);
    CHECK(std::abs(b.norm - a.norm()) <= 1e-8 * a.norm());
    CHECK(b.memory_bytes() == static_cast<std::size_t>(b.rank()) * 100 * sizeof(Complex));
    // Transposed use of the same factors: (U V^T)^T x = V (U^T x).
    const VectorXc x = random_matrix(60, 1, 3).col(0);
    const VectorXc yt = b.V * (b.U.transpose() * x);
    CHECK((yt - a.transpose() * x).norm() <= 1e-8 * (a.transpose() * x).norm());
}

TEST_CASE("aca: smooth kernels between separated clusters", "[aca]") {
    // Clusters of unit size at centre distance 3 (eta = diam / dist ~ 0.9): 1 / r and the
    // Helmholtz kernel with k = 6 (one wavelength per cluster).
    const std::vector<Vec3> x = cloud(120, 0.0, 11);
    const std::vector<Vec3> y = cloud(100, 3.0, 12);
    for (const Real k : {0.0, 6.0}) {
        const MatrixXc a = kernel_matrix(x, y, k);
        const DenseGenerator gen(a);
        for (const Real eps : {1e-4, 1e-6}) {
            aca::CompressParams p;
            p.epsilon = eps;
            const aca::LowRankBlock b = aca::compress(120, 100, gen, p);
            const Real err = aca::frobenius_error(a, b) / a.norm();
            const Index opt = svd_rank(a, eps);
            INFO("k " << k << ", eps " << eps << ": rank " << b.rank() << " (SVD rank " << opt
                      << "), error " << err << ", reason " << aca::to_string(b.reason) << ", rows "
                      << b.generated_rows << ", columns " << b.generated_columns);
            CHECK(b.reason == aca::StopReason::converged);
            CHECK(err <= eps);
            CHECK(b.rank() <= 2 * opt + 2);
            CHECK(b.rank() < 100);
            CHECK(b.generated_rows == b.rank());
            CHECK(b.generated_columns == b.rank());
            // The plain rule (one step) never stops later.
            p.confirmations = 1;
            const aca::LowRankBlock b1 = aca::compress(120, 100, gen, p);
            CHECK(b1.rank() <= b.rank());
        }
    }
}

TEST_CASE("aca: zero pivots, zero rows and zero blocks", "[aca]") {
    MatrixXc a = random_matrix(30, 4, 21) * random_matrix(25, 4, 22).transpose();
    SECTION("a zero first row: the next unused row is taken") {
        a.row(0).setZero();
        const DenseGenerator gen(a);
        aca::CompressParams p;
        p.epsilon = 1e-10;
        const aca::LowRankBlock b = aca::compress(30, 25, gen, p);
        CHECK(b.converged());
        CHECK(b.zero_rows >= 1);
        CHECK(b.rank() <= 4 + p.confirmations);
        CHECK(aca::frobenius_error(a, b) <= p.epsilon * a.norm());
    }
    SECTION("preferred rows are tried first") {
        a.row(3).setZero();
        a.row(7).setZero();
        const DenseGenerator gen(a);
        aca::CompressParams p;
        p.epsilon = 1e-10;
        p.preferred_rows = {3, 7, -1, 99, 12};
        const aca::LowRankBlock b = aca::compress(30, 25, gen, p);
        CHECK(b.zero_rows == 2);
        CHECK(b.converged());
        CHECK(aca::frobenius_error(a, b) <= p.epsilon * a.norm());
    }
    SECTION("leading zero rows beyond the limit: a probe column restarts the iteration") {
        a.topRows(5).setZero();
        const DenseGenerator gen(a);
        aca::CompressParams p;
        p.epsilon = 1e-10;
        p.max_zero_rows = 3;
        const aca::LowRankBlock b = aca::compress(30, 25, gen, p);
        CHECK(b.zero_rows == 3);
        CHECK(b.reason != aca::StopReason::zero_block);
        CHECK(b.rank() >= 4);
        CHECK(aca::frobenius_error(a, b) <= p.epsilon * a.norm());
    }
    SECTION("zero rows with k > 0: a probe column finds the unrepresented block") {
        // blockdiag(B, C): B = a b^T with powers of two (its residual rows vanish exactly after
        // one step), C random rank 3. Starting in B the zero rows would end the iteration with C
        // missing; the probe column (farthest from the pivot, in C) restarts it there.
        MatrixXc d = MatrixXc::Zero(30, 25);
        const Real pw[4] = {1.0, -2.0, 0.5, 4.0};
        for (Index i = 0; i < 15; ++i)
            for (Index j = 0; j < 12; ++j) d(i, j) = Complex(pw[i % 4] * pw[(j + 1) % 4], 0.0);
        d.block(15, 12, 15, 13) = random_matrix(15, 3, 23) * random_matrix(13, 3, 24).transpose();
        const DenseGenerator gen(d);
        aca::CompressParams p;
        p.epsilon = 1e-10;
        const aca::LowRankBlock b = aca::compress(30, 25, gen, p);
        INFO("rank " << b.rank() << ", reason " << aca::to_string(b.reason) << ", zero rows "
                     << b.zero_rows);
        CHECK(b.zero_rows >= p.max_zero_rows);
        CHECK(b.rank() >= 4);
        CHECK(aca::frobenius_error(d, b) <= p.epsilon * d.norm());
        // B alone: the probe column is reproduced as well, the iteration ends at rank 1.
        MatrixXc bonly = d;
        bonly.block(15, 12, 15, 13).setZero();
        const DenseGenerator gb(bonly);
        const aca::LowRankBlock bb = aca::compress(30, 25, gb, p);
        CHECK(bb.reason == aca::StopReason::zero_rows);
        CHECK(bb.rank() == 1);
        CHECK(aca::frobenius_error(bonly, bb) == 0.0);
    }
    SECTION("an identically zero block is rank 0 after the zero rows and one probe column") {
        a.setZero();
        const DenseGenerator gen(a);
        aca::CompressParams p;
        p.preferred_columns = {4};
        const aca::LowRankBlock b = aca::compress(30, 25, gen, p);
        CHECK(b.reason == aca::StopReason::zero_block);
        CHECK(b.rank() == 0);
        CHECK(b.converged());
        CHECK(b.generated_rows == p.max_zero_rows);
        CHECK(b.generated_columns == 1);
        CHECK(b.memory_bytes() == 0);
        CHECK(aca::frobenius_error(a, b) == 0.0);
    }
}

TEST_CASE("aca: rank cap, full rank and argument errors", "[aca]") {
    const MatrixXc a = random_matrix(20, 6, 31) * random_matrix(18, 6, 32).transpose();
    const DenseGenerator gen(a);
    aca::CompressParams p;
    p.epsilon = 1e-10;
    p.max_rank = 3;
    const aca::LowRankBlock capped = aca::compress(20, 18, gen, p);
    CHECK(capped.reason == aca::StopReason::rank_cap);
    CHECK_FALSE(capped.converged());
    CHECK(capped.rank() == 3);
    // A full-rank random matrix: every column pivoted, exact.
    const MatrixXc f = random_matrix(12, 9, 33);
    const DenseGenerator gf(f);
    p.max_rank = 0;
    const aca::LowRankBlock full = aca::compress(12, 9, gf, p);
    CHECK(full.converged());
    CHECK(full.rank() == 9);
    CHECK(aca::frobenius_error(f, full) <= 1e-12 * f.norm());
    // 1 x n and m x 1.
    const MatrixXc r1 = random_matrix(1, 7, 34);
    const aca::LowRankBlock b1 = aca::compress(1, 7, DenseGenerator(r1), {});
    CHECK(b1.rank() == 1);
    CHECK(aca::frobenius_error(r1, b1) <= 1e-14 * r1.norm());
    aca::CompressParams bad;
    CHECK_THROWS_AS(aca::compress(0, 5, gen, bad), std::invalid_argument);
    bad.epsilon = 0.0;
    CHECK_THROWS_AS(aca::compress(20, 18, gen, bad), std::invalid_argument);
    bad.epsilon = 1e-4;
    bad.max_rank = -1;
    CHECK_THROWS_AS(aca::compress(20, 18, gen, bad), std::invalid_argument);
    bad.max_rank = 0;
    bad.max_zero_rows = 0;
    CHECK_THROWS_AS(aca::compress(20, 18, gen, bad), std::invalid_argument);
    bad.max_zero_rows = 3;
    bad.confirmations = 0;
    CHECK_THROWS_AS(aca::compress(20, 18, gen, bad), std::invalid_argument);
    MatrixXc nan = a;
    nan(0, 0) = Complex(std::nan(""), 0.0);
    CHECK_THROWS_AS(aca::compress(20, 18, DenseGenerator(nan), {}), std::runtime_error);
    CHECK_THROWS_AS(aca::frobenius_error(f, capped), std::invalid_argument);
    CHECK(std::string(aca::to_string(aca::StopReason::rank_cap)) == "rank_cap");
}
