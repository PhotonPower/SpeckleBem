// Unit tests for solver::solve_direct and solver::DenseLu.
//
// Regular tests use n <= 200 so that the unit suite stays fast (docs/05: unit tests < 1 s).
// The larger systems (n = 1000 and the 2000 x 2000 timing case) carry the hidden tag [.slow]:
// catch_discover_tests does not register them, run them explicitly with
//   build/release/tests/specklebem_unit_tests "[slow]"
// (the debug preset needs ~25 s for n = 1000 and ~150 s for n = 2000).
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/solver/direct.hpp"

#include <Eigen/LU>
#include <Eigen/QR>
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
#include <string>
#include <utility>
#include <vector>

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

// ---------------------------------------------------------------------------------------------
// solver::DenseLu (WP-LU1): reusable factorisation, LAPACK and Eigen backends.

namespace {

/// Backends to test: Eigen always, LAPACK when the library was built with it.
std::vector<solver::LuBackend> available_backends() {
    std::vector<solver::LuBackend> b{solver::LuBackend::Eigen};
    if (solver::lapack_available()) {
        b.push_back(solver::LuBackend::Lapack);
    }
    return b;
}

}  // namespace

TEST_CASE("dense LU: backend selection and reporting", "[solver]") {
    const MatrixXc A = random_matrix(5, 5, 101);
    const solver::DenseLu automatic(A);
    CHECK(automatic.backend() ==
          (solver::lapack_available() ? solver::LuBackend::Lapack : solver::LuBackend::Eigen));
    const solver::DenseLu eigen(A, solver::LuBackend::Eigen);
    CHECK(eigen.backend() == solver::LuBackend::Eigen);
    solver::DirectSolveInfo info;
    (void)eigen.solve(random_vector(5, 102), &info);
    CHECK(info.backend == solver::LuBackend::Eigen);
    if (solver::lapack_available()) {
        const solver::DenseLu lapack(A, solver::LuBackend::Lapack);
        CHECK(lapack.backend() == solver::LuBackend::Lapack);
        (void)lapack.solve(random_vector(5, 103), &info);
        CHECK(info.backend == solver::LuBackend::Lapack);
    } else {
        // No silent fallback: an explicit LAPACK request fails in a build without LAPACK.
        CHECK_THROWS_AS(solver::DenseLu(A, solver::LuBackend::Lapack), std::invalid_argument);
    }
    // solve_direct reports the Auto choice.
    (void)solver::solve_direct(op::DenseOperator(A), random_vector(5, 104), &info);
    CHECK(info.backend == automatic.backend());
    CHECK(std::string(solver::to_string(solver::LuBackend::Lapack)) == "lapack");
    CHECK(std::string(solver::to_string(solver::LuBackend::Eigen)) == "eigen");
    CHECK(std::string(solver::to_string(solver::LuBackend::Auto)) == "auto");
}

namespace {

/// Exact 1 / (|A D|_1 |(A D)^{-1}|_1) with the power-of-two column scales D of DenseLu.
Real exact_equilibrated_rcond(const MatrixXc& A) {
    MatrixXc AD = A;
    for (Index j = 0; j < A.cols(); ++j) {
        int e = 0;
        (void)std::frexp(A.col(j).cwiseAbs().maxCoeff(), &e);
        AD.col(j) *= std::ldexp(1.0, -e);
    }
    return 1.0 / (AD.cwiseAbs().colwise().sum().maxCoeff() *
                  AD.inverse().cwiseAbs().colwise().sum().maxCoeff());
}

/// Q1 diag(sigma) Q2^H with random unitary Q1, Q2 and sigma_i = 10^(-decades i / (n - 1)).
MatrixXc graded_matrix(Index n, Real decades, std::uint64_t seed) {
    const MatrixXc Q1 = Eigen::HouseholderQR<MatrixXc>(random_matrix(n, n, seed)).householderQ();
    const MatrixXc Q2 =
        Eigen::HouseholderQR<MatrixXc>(random_matrix(n, n, seed + 1)).householderQ();
    VectorXr sigma(n);
    for (Index i = 0; i < n; ++i) {
        sigma(i) = std::pow(10.0, -decades * static_cast<Real>(i) / static_cast<Real>(n - 1));
    }
    return Q1 * sigma.cast<Complex>().asDiagonal() * Q2.adjoint();
}

}  // namespace

TEST_CASE("dense LU: rcond estimate vs the exact equilibrated 1-norm rcond", "[solver]") {
    // Hager/Higham estimate a lower bound of |(A D)^{-1}|_1 (the norm of (A D)^{-1} applied to
    // some unit vector), so the reported rcond is an upper bound of the exact one; Higham (1988)
    // reports the estimate within a factor 3 in practice. Families: random (n = 1, 2, 17),
    // singular values graded over 6 decades, row scaling over 5 decades (not removed by the
    // column equilibration), and the upper triangular matrix with 1 on and -1 above the diagonal
    // (|U^{-1}|_1 = 2^(n-1)).
    std::vector<std::pair<std::string, MatrixXc>> cases;
    for (const Index n : {Index{1}, Index{2}, Index{17}}) {
        cases.emplace_back("random n = " + std::to_string(n),
                           random_matrix(n, n, 2024 + static_cast<std::uint64_t>(n)));
    }
    cases.emplace_back("graded singular values", graded_matrix(80, 6.0, 501));
    MatrixXc rows = random_matrix(80, 80, 503);
    for (Index i = 0; i < rows.rows(); ++i) {
        rows.row(i) *= std::pow(10.0, -5.0 * static_cast<Real>(i) / 79.0);
    }
    cases.emplace_back("scaled rows", rows);
    MatrixXc U = MatrixXc::Identity(25, 25);
    U.triangularView<Eigen::StrictlyUpper>().setConstant(Complex(-1, 0));
    cases.emplace_back("triangular, -1 above the diagonal", U);

    for (const auto& [label, A] : cases) {
        const Real exact = exact_equilibrated_rcond(A);
        for (const solver::LuBackend backend : available_backends()) {
            const solver::DenseLu lu(A, backend);
            INFO(label << ", backend " << solver::to_string(backend) << ": rcond " << lu.rcond()
                       << ", exact " << exact << ", ratio " << lu.rcond() / exact);
            CHECK(lu.rcond() >= exact * (1 - 1e-6));
            CHECK(lu.rcond() <= 3.0 * exact);
            CHECK(!lu.ill_conditioned());
            solver::DirectSolveInfo info;
            const VectorXc b = random_vector(A.rows(), 505);
            const VectorXc x = lu.solve(b, &info);
            CHECK(info.rcond == lu.rcond());
            // Backward stable: normwise backward error |r|_1 / (|A|_1 |x|_1 + |b|_1) at rounding
            // level (the relative residual itself grows with the condition number here).
            const Real backward =
                (b - A * x).lpNorm<1>() /
                (A.cwiseAbs().colwise().sum().maxCoeff() * x.lpNorm<1>() + b.lpNorm<1>());
            CHECK(backward <= 1e-14);
        }
    }
}

TEST_CASE("dense LU: LAPACK and Eigen backends agree", "[solver]") {
    if (!solver::lapack_available()) {
        SKIP("library built without LAPACK: only the Eigen backend exists");
    }
    for (const Index n : {Index{1}, Index{2}, Index{17}, Index{100}}) {
        const MatrixXc A = random_matrix(n, n, 2024 + static_cast<std::uint64_t>(n));
        const VectorXc b = random_vector(n, 99 + static_cast<std::uint64_t>(n));
        const MatrixXc B = random_matrix(n, 4, 199 + static_cast<std::uint64_t>(n));
        const solver::DenseLu lapack(A, solver::LuBackend::Lapack);
        const solver::DenseLu eigen(A, solver::LuBackend::Eigen);
        INFO("n = " << n << ": rcond lapack " << lapack.rcond() << ", eigen " << eigen.rcond());
        // Same pivot sequence and the same estimator on the same equilibrated matrix: the
        // estimates differ by rounding only.
        CHECK(std::abs(lapack.rcond() - eigen.rcond()) <= 1e-8 * eigen.rcond());
        solver::DirectSolveInfo info_l;
        solver::DirectSolveInfo info_e;
        const VectorXc x_l = lapack.solve(b, &info_l);
        const VectorXc x_e = eigen.solve(b, &info_e);
        CHECK(info_l.backend == solver::LuBackend::Lapack);
        CHECK(info_e.backend == solver::LuBackend::Eigen);
        CHECK(info_l.residual <= 1e-12);
        CHECK(info_e.residual <= 1e-12);
        CHECK((x_l - x_e).norm() <= 1e-12 * x_e.norm());
        const MatrixXc X_l = lapack.solve(B);
        const MatrixXc X_e = eigen.solve(B);
        CHECK((X_l - X_e).norm() <= 1e-12 * X_e.norm());
    }
}

TEST_CASE("dense LU: multiple right-hand sides equal column-by-column solves", "[solver]") {
    const Index n = 60;
    const MatrixXc A = random_matrix(n, n, 301);
    MatrixXc B = random_matrix(n, 5, 302);
    B.col(2).setZero();  // a zero right-hand side has the zero solution and residual 0
    for (const solver::LuBackend backend : available_backends()) {
        INFO("backend " << solver::to_string(backend));
        const solver::DenseLu lu(A, backend);
        solver::DirectSolveInfo block_info;
        const MatrixXc X = lu.solve(B, &block_info);
        REQUIRE(X.rows() == n);
        REQUIRE(X.cols() == B.cols());
        Real worst = 0;
        for (Index j = 0; j < B.cols(); ++j) {
            solver::DirectSolveInfo info;
            const VectorXc x = lu.solve(VectorXc(B.col(j)), &info);
            CHECK((X.col(j) - x).norm() <= 1e-13 * std::max(x.norm(), Real{1}));
            worst = std::max(worst, info.residual);
        }
        CHECK(X.col(2).isZero(0.0));
        CHECK(block_info.residual <= 1e-12);
        CHECK(block_info.residual <= 10.0 * worst + 1e-16);
        CHECK(block_info.refinement_steps <= 2);
        CHECK(block_info.rcond == lu.rcond());
        CHECK(block_info.backend == lu.backend());
        // The factorisation is reused: the same object solves again with the same result.
        CHECK(lu.solve(B) == X);
        // No right-hand sides.
        CHECK(lu.solve(MatrixXc(n, 0)).cols() == 0);
    }
}

TEST_CASE("dense LU: ill-conditioned and singular matrices", "[solver]") {
    const Index n = 60;
    MatrixXc A = random_matrix(n, n, 41);
    A.row(7) = A.row(23) + 1e-9 * random_matrix(1, n, 42);  // nearly dependent rows
    MatrixXc S = random_matrix(n, n, 4);
    S.row(7) = S.row(23);  // exactly dependent rows
    const MatrixXc R = random_matrix(n, n - 1, 5) * random_matrix(n - 1, n, 6);  // rank n - 1
    const MatrixXc zero = MatrixXc::Zero(n, n);
    for (const solver::LuBackend backend : available_backends()) {
        INFO("backend " << solver::to_string(backend));
        const WarningCounter counter;
        const solver::DenseLu lu(A, backend);
        CHECK(lu.ill_conditioned());
        CHECK(lu.rcond() >= std::numeric_limits<Real>::epsilon());
        CHECK(lu.rcond() < solver::kIllConditionedRcond);
        CHECK(counter.count() == 1);  // logged once per factorisation, not per solve
        solver::DirectSolveInfo info;
        const VectorXc b = random_vector(n, 43);
        const VectorXc x = lu.solve(b, &info);
        CHECK(info.ill_conditioned);
        CHECK(std::abs(info.residual - relative_residual(A, x, b)) <= 1e-3 * info.residual);
        CHECK(counter.count() == 1);
        CHECK_THROWS_AS(solver::DenseLu(S, backend), std::runtime_error);
        CHECK_THROWS_AS(solver::DenseLu(R, backend), std::runtime_error);
        CHECK_THROWS_AS(solver::DenseLu(zero, backend), std::runtime_error);
    }
}

TEST_CASE("dense LU: invalid arguments and the empty system", "[solver]") {
    for (const solver::LuBackend backend : available_backends()) {
        INFO("backend " << solver::to_string(backend));
        const MatrixXc rect = random_matrix(10, 9, 9);
        CHECK_THROWS_AS(solver::DenseLu(rect, backend), std::invalid_argument);
        MatrixXc bad = random_matrix(10, 10, 7);
        bad(4, 6) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0);
        CHECK_THROWS_AS(solver::DenseLu(bad, backend), std::invalid_argument);
        bad(4, 6) = Complex(0, -std::numeric_limits<Real>::infinity());
        CHECK_THROWS_AS(solver::DenseLu(bad, backend), std::invalid_argument);

        const MatrixXc A = random_matrix(10, 10, 7);
        const solver::DenseLu lu(A, backend);
        CHECK(lu.size() == 10);
        CHECK_THROWS_AS(lu.solve(random_vector(9, 8)), std::invalid_argument);
        CHECK_THROWS_AS(lu.solve(VectorXc()), std::invalid_argument);
        CHECK_THROWS_AS(lu.solve(random_matrix(11, 2, 8)), std::invalid_argument);
        VectorXc b = random_vector(10, 11);
        b(3) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0);
        CHECK_THROWS_AS(lu.solve(b), std::invalid_argument);
        MatrixXc B = random_matrix(10, 3, 12);
        B(5, 1) = Complex(0, std::numeric_limits<Real>::infinity());
        CHECK_THROWS_AS(lu.solve(B), std::invalid_argument);

        // An empty matrix is valid: rcond 1, empty solutions.
        const MatrixXc empty(0, 0);
        const solver::DenseLu e(empty, backend);
        CHECK(e.size() == 0);
        CHECK(e.rcond() == 1.0);
        CHECK(!e.ill_conditioned());
        solver::DirectSolveInfo info;
        CHECK(e.solve(VectorXc(), &info).size() == 0);
        CHECK(info.residual == 0.0);
        CHECK(info.backend == e.backend());
        const MatrixXc X = e.solve(MatrixXc(0, 4));
        CHECK(X.rows() == 0);
        CHECK(X.cols() == 4);
    }
    // solve_direct: a non-finite matrix entry is an invalid argument (before WP-LU1 it was
    // reported as a singular pivot, std::runtime_error).
    MatrixXc bad = random_matrix(10, 10, 13);
    bad(0, 0) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0);
    CHECK_THROWS_AS(solver::solve_direct(op::DenseOperator(bad), random_vector(10, 14)),
                    std::invalid_argument);
}

TEST_CASE("dense LU: 32-bit LAPACK integer guard", "[solver]") {
    const Index int_max = std::numeric_limits<int>::max();
    CHECK(solver::detail::checked_lapack_int(0, "n") == 0);
    CHECK(solver::detail::checked_lapack_int(11440, "n") == 11440);
    CHECK(solver::detail::checked_lapack_int(int_max, "n") == std::numeric_limits<int>::max());
    CHECK_THROWS_AS(solver::detail::checked_lapack_int(int_max + 1, "n"), std::invalid_argument);
    CHECK_THROWS_AS(solver::detail::checked_lapack_int(Index{1} << 40, "n"), std::invalid_argument);
    CHECK_THROWS_AS(solver::detail::checked_lapack_int(-1, "n"), std::invalid_argument);
}
