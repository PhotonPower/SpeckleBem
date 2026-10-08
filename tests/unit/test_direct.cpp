// Unit tests for solver::solve_direct.
//
// Regular tests use n <= 200 so that the unit suite stays fast (docs/05: unit tests < 1 s).
// The larger systems (n = 1000 and the 2000 x 2000 timing case) carry the hidden tag [.slow]:
// catch_discover_tests does not register them, run them explicitly with
//   build/release/tests/specklebem_unit_tests "[slow]"
// (the debug preset needs ~25 s for n = 1000 and ~150 s for n = 2000).
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/solver/direct.hpp"

#include <catch2/catch_test_macros.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

using namespace specklebem;

namespace {

MatrixXc random_matrix(Index rows, Index cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> n(0.0, 1.0);
    MatrixXc A(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            A(i, j) = Complex(n(rng), n(rng));
        }
    }
    return A;
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    return random_matrix(n, 1, seed).col(0);
}

Real relative_residual(const MatrixXc& A, const VectorXc& x, const VectorXc& b) {
    return (A * x - b).norm() / b.norm();
}

/// Counts the warnings logged while it is alive (logger level lowered to warn if needed and
/// restored afterwards).
class WarningCounter {
public:
    WarningCounter()
        : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64)),
          saved_level_(spdlog::default_logger()->level()) {
        sink_->set_level(spdlog::level::warn);
        spdlog::default_logger()->set_level(std::min(saved_level_, spdlog::level::warn));
        spdlog::default_logger()->sinks().push_back(sink_);
    }
    ~WarningCounter() {
        auto& sinks = spdlog::default_logger()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
        spdlog::default_logger()->set_level(saved_level_);
    }
    WarningCounter(const WarningCounter&) = delete;
    WarningCounter& operator=(const WarningCounter&) = delete;
    [[nodiscard]] std::size_t count() const { return sink_->last_formatted().size(); }

private:
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
    spdlog::level::level_enum saved_level_;
};

void check_random_system(Index n, Real tolerance) {
    const op::DenseOperator Z(random_matrix(n, n, 12345 + static_cast<std::uint64_t>(n)));
    const VectorXc b = random_vector(n, 777);
    solver::DirectSolveInfo info;
    const VectorXc x = solver::solve_direct(Z, b, &info);
    REQUIRE(x.size() == n);
    const Real res = relative_residual(Z.matrix(), x, b);
    INFO("n = " << n << ", residual = " << res << ", rcond = " << info.rcond);
    CHECK(res < tolerance);
    CHECK(std::abs(info.residual - res) <= 0.5 * res + 1e-16);  // reported vs recomputed
    CHECK(!info.ill_conditioned);
    CHECK(info.rcond > 1e-8);
}

}  // namespace

TEST_CASE("direct solver: random complex system", "[solver]") {
    check_random_system(200, 1e-12);
}

TEST_CASE("direct solver: random complex 1000 x 1000 system", "[solver][.slow]") {
    // Plain LU leaves |Zx - b|/|b| = 2.9e-12 here; iterative refinement gives ~5e-14.
    check_random_system(1000, 1e-12);
}

TEST_CASE("direct solver: identity and diagonal systems are exact", "[solver]") {
    const Index n = 50;
    const VectorXc b = random_vector(n, 1);
    const op::DenseOperator I(MatrixXc::Identity(n, n));
    solver::DirectSolveInfo info;
    CHECK(solver::solve_direct(I, b, &info) == b);
    CHECK(info.residual == 0.0);
    CHECK(info.refinement_steps == 0);
    CHECK(std::abs(info.rcond - 1.0) < 1e-12);

    // Power-of-two diagonal (real and imaginary): x_i = b_i / d_i without rounding (the
    // equilibration also scales by powers of two).
    VectorXc d(n);
    for (Index i = 0; i < n; ++i) {
        const Real mag = std::ldexp(1.0, static_cast<int>(i % 7) - 3);
        d(i) = (i % 2 == 0) ? Complex(mag, 0) : Complex(0, -mag);
    }
    const op::DenseOperator D(MatrixXc(d.asDiagonal()));
    const VectorXc x = solver::solve_direct(D, b);
    for (Index i = 0; i < n; ++i) {
        CHECK(x(i) == b(i) / d(i));
    }

    // General complex diagonal: exact up to the rounding of one division.
    const VectorXc g = random_vector(n, 2);
    const op::DenseOperator G(MatrixXc(g.asDiagonal()));
    const VectorXc y = solver::solve_direct(G, b);
    for (Index i = 0; i < n; ++i) {
        CHECK(std::abs(y(i) - b(i) / g(i)) <= 4e-16 * std::abs(b(i) / g(i)));
    }
}

TEST_CASE("direct solver: column scaling does not trigger the singularity test", "[solver]") {
    // Columns scaled geometrically from 1 down to 1e-16: the unequilibrated rcond is ~1e-19 but
    // the system is as well conditioned as the unscaled one (cf. the eta^2 ratio between the
    // J and M blocks of PMCHWT).
    const Index n = 200;
    MatrixXc A = random_matrix(n, n, 31);
    for (Index j = 0; j < n; ++j) {
        A.col(j) *= std::pow(10.0, -16.0 * static_cast<Real>(j) / static_cast<Real>(n - 1));
    }
    const op::DenseOperator Z(A);
    const VectorXc b = random_vector(n, 32);
    solver::DirectSolveInfo info;
    VectorXc x;
    REQUIRE_NOTHROW(x = solver::solve_direct(Z, b, &info));
    const Real res = relative_residual(A, x, b);
    INFO("residual = " << res << ", equilibrated rcond = " << info.rcond);
    CHECK(res < 1e-12);
    CHECK(!info.ill_conditioned);
    CHECK(info.rcond > 1e-8);
    // The solution is the unscaled one divided by the column scales.
    CHECK(std::abs(x(n - 1)) > 1e12);
}

TEST_CASE("direct solver: ill-conditioned system is flagged and logged", "[solver]") {
    const Index n = 60;
    MatrixXc A = random_matrix(n, n, 41);
    A.row(7) = A.row(23) + 1e-9 * random_matrix(1, n, 42);  // nearly dependent rows
    const op::DenseOperator Z(A);
    const VectorXc b = random_vector(n, 43);
    const WarningCounter counter;
    solver::DirectSolveInfo info;
    VectorXc x;
    REQUIRE_NOTHROW(x = solver::solve_direct(Z, b, &info));
    INFO("rcond = " << info.rcond << ", residual = " << info.residual);
    CHECK(info.ill_conditioned);
    CHECK(info.rcond >= std::numeric_limits<Real>::epsilon());
    CHECK(info.rcond < solver::kIllConditionedRcond);
    CHECK(counter.count() == 1);
    CHECK(std::isfinite(info.residual));
    CHECK(std::abs(info.residual - relative_residual(A, x, b)) <= 1e-3 * info.residual);
}

TEST_CASE("direct solver: singular matrices throw", "[solver]") {
    const Index n = 40;
    const VectorXc b = random_vector(n, 3);
    // Zero matrix.
    CHECK_THROWS_AS(solver::solve_direct(op::DenseOperator(MatrixXc::Zero(n, n)), b),
                    std::runtime_error);
    // Two identical rows.
    MatrixXc A = random_matrix(n, n, 4);
    A.row(7) = A.row(23);
    CHECK_THROWS_AS(solver::solve_direct(op::DenseOperator(A), b), std::runtime_error);
    // Rank n-1 product of random factors (pivot not exactly zero, rcond ~ machine precision).
    const MatrixXc R = random_matrix(n, n - 1, 5) * random_matrix(n - 1, n, 6);
    CHECK_THROWS_AS(solver::solve_direct(op::DenseOperator(R), b), std::runtime_error);
}

TEST_CASE("direct solver: invalid arguments throw", "[solver]") {
    const op::DenseOperator Z(random_matrix(10, 10, 7));
    CHECK_THROWS_AS(solver::solve_direct(Z, random_vector(9, 8)), std::invalid_argument);
    CHECK_THROWS_AS(solver::solve_direct(Z, VectorXc()), std::invalid_argument);
    const op::DenseOperator rect(random_matrix(10, 9, 9));
    CHECK_THROWS_AS(solver::solve_direct(rect, random_vector(10, 10)), std::invalid_argument);
    VectorXc b = random_vector(10, 11);
    b(3) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0);
    CHECK_THROWS_AS(solver::solve_direct(Z, b), std::invalid_argument);
    b(3) = Complex(0, std::numeric_limits<Real>::infinity());
    CHECK_THROWS_AS(solver::solve_direct(Z, b), std::invalid_argument);
    // An empty system is valid and has an empty solution.
    CHECK(solver::solve_direct(op::DenseOperator(MatrixXc(0, 0)), VectorXc()).size() == 0);
}

TEST_CASE("direct solver: 2000 x 2000 system", "[solver][.slow]") {
    const Index n = 2000;
    const op::DenseOperator Z(random_matrix(n, n, 2000));
    const VectorXc b = random_vector(n, 2001);
    const auto t0 = std::chrono::steady_clock::now();
    const VectorXc x = solver::solve_direct(Z, b);
    const Real seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    INFO("solve time " << seconds << " s");
    CHECK(relative_residual(Z.matrix(), x, b) < 1e-12);
#ifdef NDEBUG
    // Optimised builds only: the debug preset (-O0, ASan + UBSan) needs 125-150 s.
    CHECK(seconds < 20.0);
#endif
}
