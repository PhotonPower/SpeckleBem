// Unit tests for solver::gmres and the preconditioners (WP13).
//
// Random systems use fixed seeds (std::mt19937_64). The BEM case solves the dense system of a
// small icosphere (subdivision 1 at lambda = 1 um, 2N = 240) against solve_direct.
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/solver/direct.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "assembler_test_support.hpp"

using namespace specklebem;
using solver::GmresParams;
using solver::GmresResult;
using solver::PreconditionerSide;

namespace {

MatrixXc random_matrix(Index rows, Index cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    MatrixXc A(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) A(i, j) = Complex(nd(rng), nd(rng));
    }
    return A;
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    return random_matrix(n, 1, seed).col(0);
}

/// Non-Hermitian matrix shift I + R / sqrt(2 n) (eigenvalues in a disk of radius ~1 about
/// `shift`; well conditioned for shift = 3).
MatrixXc well_conditioned(Index n, std::uint64_t seed, Real shift = 3.0) {
    MatrixXc A = random_matrix(n, n, seed) / std::sqrt(2.0 * static_cast<Real>(n));
    A.diagonal().array() += Complex(shift, 0.0);
    return A;
}

/// Complex diagonal with log-uniform magnitudes in [1e-3, 1e3] and random phases.
VectorXc wide_diagonal(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(0.0, 1.0);
    VectorXc s(n);
    for (Index i = 0; i < n; ++i) {
        const Real mag = std::pow(10.0, -3.0 + 6.0 * u(rng));
        s(i) = std::polar(mag, 2.0 * constants::pi * u(rng));
    }
    return s;
}

GmresParams params(Real tol, int max_iter = 1000, int restart = 0,
                   PreconditionerSide side = PreconditionerSide::Left) {
    GmresParams p;
    p.tolerance = tol;
    p.max_iter = max_iter;
    p.restart = restart;
    p.verbose = false;
    p.side = side;
    return p;
}

Real rel_diff(const VectorXc& a, const VectorXc& b) {
    return (a - b).norm() / b.norm();
}

bool finite(const GmresResult& r) {
    for (Index i = 0; i < r.x.size(); ++i) {
        if (!std::isfinite(r.x(i).real()) || !std::isfinite(r.x(i).imag()))
            return false;
    }
    for (const Real h : r.residual_history) {
        if (!std::isfinite(h))
            return false;
    }
    return std::isfinite(r.true_relative_residual);
}

const solver::IdentityPreconditioner kNone;

}  // namespace

TEST_CASE("gmres: full GMRES matches the LU solution on a well-conditioned system", "[gmres]") {
    const op::DenseOperator Z(well_conditioned(200, 11));
    const VectorXc b = random_vector(200, 12);
    const GmresResult r = solver::gmres(Z, b, kNone, params(1e-12));
    const VectorXc x_lu = solver::solve_direct(Z, b);
    INFO("iterations " << r.iterations);
    CHECK(r.converged);
    CHECK(r.residual_history.size() == static_cast<std::size_t>(r.iterations) + 1);
    CHECK(r.residual_history.front() == 1.0);
    CHECK(r.residual_history.back() <= 1e-12);
    CHECK(rel_diff(r.x, x_lu) <= 1e-10);
    CHECK(r.true_relative_residual <= 1e-11);
}

TEST_CASE("gmres: full GMRES terminates within n iterations", "[gmres]") {
    // Random Gaussian matrix: eigenvalues fill a disk about 0, so GMRES needs ~n iterations;
    // in exact arithmetic the Krylov space is complete after n steps.
    const Index n = 40;
    const op::DenseOperator Z(random_matrix(n, n, 21));
    const VectorXc b = random_vector(n, 22);
    const GmresResult r = solver::gmres(Z, b, kNone, params(1e-10));
    INFO("iterations " << r.iterations << ", true residual " << r.true_relative_residual);
    CHECK(r.converged);
    CHECK(r.iterations <= n);
    CHECK(finite(r));
    CHECK(rel_diff(r.x, solver::solve_direct(Z, b)) <= 1e-7);
}

TEST_CASE("gmres: happy breakdown terminates with the exact solution", "[gmres]") {
    const Index n = 30;
    SECTION("identity: exact breakdown after one iteration") {
        const op::DenseOperator Z(MatrixXc::Identity(n, n));
        const VectorXc b = random_vector(n, 31);
        const GmresResult r = solver::gmres(Z, b, kNone, params(1e-15));
        CHECK(r.iterations == 1);
        CHECK(r.converged);
        CHECK(finite(r));
        CHECK(rel_diff(r.x, b) <= 1e-15);
    }
    SECTION("I + rank 2: at most 3 iterations") {
        MatrixXc A = MatrixXc::Identity(n, n);
        A += random_vector(n, 32) * random_vector(n, 33).adjoint() / static_cast<Real>(n);
        A += random_vector(n, 34) * random_vector(n, 35).adjoint() / static_cast<Real>(n);
        const op::DenseOperator Z(A);
        const VectorXc b = random_vector(n, 36);
        const GmresResult r = solver::gmres(Z, b, kNone, params(1e-13));
        INFO("iterations " << r.iterations << ", monitored " << r.residual_history.back());
        CHECK(r.iterations <= 3);
        CHECK(r.converged);
        CHECK(finite(r));
        CHECK(r.true_relative_residual <= 1e-12);
    }
    SECTION("I + rank 2, tol 1e-300: stops at the happy breakdown, not converged") {
        // The tolerance is unreachable, so the loop can only end through the happy-breakdown
        // branch (h_{k+1,k} <= 100 eps |w_k|) after the Krylov space became invariant (dim 3).
        MatrixXc A = MatrixXc::Identity(n, n);
        A += random_vector(n, 32) * random_vector(n, 33).adjoint() / static_cast<Real>(n);
        A += random_vector(n, 34) * random_vector(n, 35).adjoint() / static_cast<Real>(n);
        const op::DenseOperator Z(A);
        const VectorXc b = random_vector(n, 36);
        const GmresResult r = solver::gmres(Z, b, kNone, params(1e-300));
        INFO("iterations " << r.iterations << ", monitored " << r.residual_history.back()
                           << ", true residual " << r.true_relative_residual);
        CHECK(r.iterations <= 3);
        CHECK_FALSE(r.converged);  // documented: converged follows the monitored residual
        CHECK(r.residual_history.size() == static_cast<std::size_t>(r.iterations) + 1);
        CHECK(finite(r));
        CHECK(r.true_relative_residual <= 1e-12);
    }
}

TEST_CASE("gmres: monotone history, restart and callback", "[gmres]") {
    // Eigenvalue disk closer to 0 (shift 1.5), so that restarting costs iterations.
    const op::DenseOperator Z(well_conditioned(200, 41, 1.5));
    const VectorXc b = random_vector(200, 42);
    std::vector<int> cb_iters;
    std::vector<Real> cb_values;
    const auto cb = [&](int k, Real res) {
        cb_iters.push_back(k);
        cb_values.push_back(res);
    };
    const GmresResult full = solver::gmres(Z, b, kNone, params(1e-10), cb);
    REQUIRE(full.converged);
    for (std::size_t k = 1; k < full.residual_history.size(); ++k) {
        CHECK(full.residual_history[k] <= full.residual_history[k - 1]);
    }
    REQUIRE(cb_values.size() == static_cast<std::size_t>(full.iterations));
    for (std::size_t k = 0; k < cb_values.size(); ++k) {
        CHECK(cb_iters[k] == static_cast<int>(k) + 1);
        CHECK(cb_values[k] == full.residual_history[k + 1]);
    }

    cb_iters.clear();
    cb_values.clear();
    const GmresResult restarted = solver::gmres(Z, b, kNone, params(1e-10, 1000, 5), cb);
    INFO("full " << full.iterations << ", GMRES(5) " << restarted.iterations);
    // History and callback agree entry for entry across restarts as well.
    REQUIRE(cb_values.size() == static_cast<std::size_t>(restarted.iterations));
    for (std::size_t k = 0; k < cb_values.size(); ++k) {
        CHECK(cb_iters[k] == static_cast<int>(k) + 1);
        CHECK(cb_values[k] == restarted.residual_history[k + 1]);
    }
    CHECK(restarted.converged);
    CHECK(restarted.iterations >= full.iterations);  // observed: 49 full, 55 GMRES(5)
    CHECK(restarted.residual_history.size() == static_cast<std::size_t>(restarted.iterations) + 1);
    CHECK(restarted.true_relative_residual <= 1e-9);
    CHECK(rel_diff(restarted.x, full.x) <= 1e-8);
}

TEST_CASE("gmres: left Jacobi GMRES is invariant under row scaling (issue #15)", "[gmres]") {
    const Index n = 200;
    const MatrixXc A = well_conditioned(n, 11);
    const VectorXc b = random_vector(n, 12);
    const VectorXc s = wide_diagonal(n, 51);
    const MatrixXc SA = s.asDiagonal() * A;
    const VectorXc Sb = s.cwiseProduct(b);
    const op::DenseOperator Z(A);
    const op::DenseOperator SZ(SA);
    const solver::DiagonalPreconditioner MZ(A.diagonal());
    const solver::DiagonalPreconditioner MSZ(SA.diagonal());

    const GmresResult r = solver::gmres(Z, b, MZ, params(1e-10));
    const GmresResult rs = solver::gmres(SZ, Sb, MSZ, params(1e-10));
    REQUIRE(r.converged);
    REQUIRE(rs.converged);
    CHECK(r.iterations == rs.iterations);
    REQUIRE(r.residual_history.size() == rs.residual_history.size());
    for (std::size_t k = 0; k < r.residual_history.size(); ++k) {
        // Absolute floor 1e-14 for the last entries, which are close to rounding level.
        CHECK(std::abs(r.residual_history[k] - rs.residual_history[k]) <=
              1e-10 * r.residual_history[k] + 1e-14);
    }
    CHECK(rel_diff(rs.x, r.x) <= 1e-9);

    // Unpreconditioned, the row scaling changes the Krylov space and the stopping criterion.
    // Observed: 21 iterations on (Z, b) (left Jacobi: 21 on both systems), 200 = n iterations
    // on (S Z, S b) without an iteration limit: the 1e-10 tolerance on |S b - S Z x| / |S b|
    // is only reached once the Krylov space is complete. The limit 3 u.iterations keeps the
    // sanitizer run short (observed residual after 63 iterations: 1.1e-2).
    const GmresResult u = solver::gmres(Z, b, kNone, params(1e-10));
    REQUIRE(u.converged);
    const GmresResult us = solver::gmres(SZ, Sb, kNone, params(1e-10, 3 * u.iterations));
    INFO("unpreconditioned: " << u.iterations << " vs row-scaled " << us.iterations << " (residual "
                              << us.residual_history.back() << "); left Jacobi: " << r.iterations);
    CHECK_FALSE(us.converged);
    CHECK(us.iterations == 3 * u.iterations);
}

TEST_CASE("gmres: right preconditioning monitors the true residual", "[gmres]") {
    const Index n = 200;
    const MatrixXc A = well_conditioned(n, 11);
    const VectorXc b = random_vector(n, 12);
    const VectorXc s = wide_diagonal(n, 51);
    const MatrixXc SA = s.asDiagonal() * A;
    const VectorXc Sb = s.cwiseProduct(b);
    const op::DenseOperator SZ(SA);
    const solver::DiagonalPreconditioner M(SA.diagonal());

    const GmresResult r =
        solver::gmres(SZ, Sb, M, params(1e-3, 1000, 0, PreconditionerSide::Right));
    INFO("iterations " << r.iterations << ", monitored " << r.residual_history.back() << ", true "
                       << r.true_relative_residual);
    CHECK(r.converged);
    CHECK(std::abs(r.residual_history.back() - r.true_relative_residual) <=
          1e-10 * r.true_relative_residual);

    // Tight tolerance: converges to the LU solution of the badly row-scaled system.
    const GmresResult t =
        solver::gmres(SZ, Sb, M, params(1e-12, 1000, 0, PreconditionerSide::Right));
    CHECK(t.converged);
    CHECK(t.true_relative_residual <= 1e-11);
    CHECK(rel_diff(t.x, solver::solve_direct(op::DenseOperator(A), b)) <= 1e-8);
}

TEST_CASE("gmres: restarted right preconditioning (GMRES(5), Jacobi) on a row-scaled system",
          "[gmres]") {
    const Index n = 200;
    const MatrixXc A = well_conditioned(n, 11);
    const VectorXc b = random_vector(n, 12);
    const VectorXc s = wide_diagonal(n, 51);
    const MatrixXc SA = s.asDiagonal() * A;
    const VectorXc Sb = s.cwiseProduct(b);
    const op::DenseOperator SZ(SA);
    const solver::DiagonalPreconditioner M(SA.diagonal());

    // The x update after each cycle goes through M^{-1}; the restart residual is the true one.
    // Tight tolerance: the solution matches LU of the unscaled system (observed 25 iterations,
    // |x - x_LU| / |x_LU| = 1.1e-12).
    const GmresResult t =
        solver::gmres(SZ, Sb, M, params(1e-12, 1000, 5, PreconditionerSide::Right));
    const VectorXc x_lu = solver::solve_direct(op::DenseOperator(A), b);
    INFO("tol 1e-12: iterations " << t.iterations << ", monitored " << t.residual_history.back()
                                  << ", true " << t.true_relative_residual
                                  << ", |x - x_LU| / |x_LU| " << rel_diff(t.x, x_lu));
    CHECK(t.converged);
    CHECK(t.iterations > 5);  // at least one restart happened
    CHECK(t.true_relative_residual <= 1e-11);
    CHECK(rel_diff(t.x, x_lu) <= 1e-8);

    // Monitored == true residual at the end, checked at the default tol 1e-3. The Arnoldi
    // estimate and the explicit residual differ by rounding, absolutely ~1e-17 .. 1e-16
    // (observed for tol 1e-2 .. 1e-6), so a 1e-10 relative check is only meaningful at
    // residuals >~ 1e-6 (at tol 1e-12: 9.36378e-13 vs 9.3637e-13). Observed at tol 1e-3:
    // 7 iterations (one restart), monitored 5.64e-4, difference 1.0e-16.
    const GmresResult r =
        solver::gmres(SZ, Sb, M, params(1e-3, 1000, 5, PreconditionerSide::Right));
    INFO("tol 1e-3: iterations " << r.iterations << ", monitored " << r.residual_history.back()
                                 << ", true " << r.true_relative_residual);
    CHECK(r.converged);
    CHECK(r.iterations > 5);
    CHECK(std::abs(r.residual_history.back() - r.true_relative_residual) <=
          1e-10 * r.true_relative_residual);
}

namespace {

/// Test double: y = A x, except that the `bad_call`-th application (1-based) returns NaN.
class NanAfterOperator final : public op::LinearOperator {
public:
    NanAfterOperator(MatrixXc A, int bad_call) : A_(std::move(A)), bad_call_(bad_call) {}
    [[nodiscard]] Index rows() const override { return A_.rows(); }
    [[nodiscard]] Index cols() const override { return A_.cols(); }
    void apply(const VectorXc& x, VectorXc& y) const override {
        ++calls_;
        y = A_ * x;
        if (calls_ >= bad_call_)
            y(0) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0.0);
    }
    [[nodiscard]] std::string describe() const override { return "NanAfterOperator"; }
    [[nodiscard]] std::size_t memory_bytes() const override { return 0; }
    [[nodiscard]] int calls() const { return calls_; }

private:
    MatrixXc A_;
    int bad_call_;
    mutable int calls_ = 0;
};

/// Test double: M^{-1} that returns +Inf in one entry.
class InfPreconditioner final : public solver::Preconditioner {
public:
    void apply(const VectorXc& r, VectorXc& z) const override {
        z = r;
        z(1) = Complex(0.0, std::numeric_limits<Real>::infinity());
    }
    [[nodiscard]] std::string name() const override { return "inf"; }
};

}  // namespace

TEST_CASE("gmres: non-finite operator or preconditioner output throws immediately", "[gmres]") {
    const Index n = 40;
    const MatrixXc A = well_conditioned(n, 61);
    const VectorXc b = random_vector(n, 62);

    SECTION("operator returns NaN at the 3rd application (left and right, full and restarted)") {
        for (const auto side : {PreconditionerSide::Left, PreconditionerSide::Right}) {
            for (const int restart : {0, 2}) {
                const NanAfterOperator Z(A, 3);
                CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(1e-14, 1000, restart, side)),
                                std::runtime_error);
                CHECK(Z.calls() == 3);  // stops at the first bad product, not after max_iter
            }
        }
    }
    SECTION("message names the iteration") {
        const NanAfterOperator Z(A, 2);
        CHECK_THROWS_WITH(
            solver::gmres(Z, b, kNone, params(1e-14)),
            "gmres: operator or preconditioner produced non-finite values at iteration 2");
    }
    SECTION("dense operator containing NaN") {
        MatrixXc An = A;
        An(5, 7) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0.0);
        CHECK_THROWS_AS(solver::gmres(op::DenseOperator(An), b, kNone, params(1e-14)),
                        std::runtime_error);
    }
    SECTION("preconditioner returns Inf") {
        const InfPreconditioner M;
        const op::DenseOperator Z(A);
        // Left: already M^{-1} b is non-finite (iteration 0); right: the first product.
        CHECK_THROWS_WITH(
            solver::gmres(Z, b, M, params(1e-14)),
            "gmres: operator or preconditioner produced non-finite values at iteration 0");
        CHECK_THROWS_WITH(
            solver::gmres(Z, b, M, params(1e-14, 1000, 0, PreconditionerSide::Right)),
            "gmres: operator or preconditioner produced non-finite values at iteration 1");
    }
}

TEST_CASE("gmres: max_iter, b = 0 and argument errors", "[gmres]") {
    const Index n = 40;
    const op::DenseOperator Z(random_matrix(n, n, 21));
    const VectorXc b = random_vector(n, 22);

    const GmresResult r = solver::gmres(Z, b, kNone, params(1e-10, 5));
    CHECK_FALSE(r.converged);
    CHECK(r.iterations == 5);
    CHECK(r.residual_history.size() == 6);
    CHECK(finite(r));
    // Restarted variant hits max_iter in the middle of a cycle.
    const GmresResult rr = solver::gmres(Z, b, kNone, params(1e-10, 7, 3));
    CHECK_FALSE(rr.converged);
    CHECK(rr.iterations == 7);
    CHECK(rr.residual_history.size() == 8);

    const GmresResult z = solver::gmres(Z, VectorXc::Zero(n), kNone, params(1e-6));
    CHECK(z.converged);
    CHECK(z.iterations == 0);
    CHECK(z.x.size() == n);
    CHECK(z.x.isZero(0.0));
    CHECK(z.true_relative_residual == 0.0);

    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    const op::DenseOperator rect(random_matrix(n, n + 1, 1));
    CHECK_THROWS_AS(solver::gmres(rect, b, kNone, params(1e-6)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, random_vector(n + 1, 2), kNone, params(1e-6)),
                    std::invalid_argument);
    VectorXc bad = b;
    bad(3) = Complex(nan, 0.0);
    CHECK_THROWS_AS(solver::gmres(Z, bad, kNone, params(1e-6)), std::invalid_argument);
    bad(3) = Complex(0.0, inf);
    CHECK_THROWS_AS(solver::gmres(Z, bad, kNone, params(1e-6)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(0.0)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(-1e-6)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(nan)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(inf)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(1e-6, 0)), std::invalid_argument);
    CHECK_THROWS_AS(solver::gmres(Z, b, kNone, params(1e-6, 10, -1)), std::invalid_argument);
}

TEST_CASE("gmres: DiagonalPreconditioner", "[gmres]") {
    const solver::DiagonalPreconditioner M(VectorXc{{Complex(2.0, 0.0), Complex(0.0, -4.0)}});
    CHECK(M.size() == 2);
    CHECK(M.name() == "diagonal");
    VectorXc z;
    M.apply(VectorXc{{Complex(1.0, 1.0), Complex(4.0, 0.0)}}, z);
    CHECK(std::abs(z(0) - Complex(0.5, 0.5)) <= 1e-15);
    CHECK(std::abs(z(1) - Complex(0.0, 1.0)) <= 1e-15);
    CHECK_THROWS_AS(M.apply(VectorXc::Ones(3), z), std::invalid_argument);

    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    CHECK_THROWS_AS(solver::DiagonalPreconditioner(VectorXc()), std::invalid_argument);
    CHECK_THROWS_AS(solver::DiagonalPreconditioner(VectorXc{{Complex(1.0, 0.0), Complex(0.0)}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(solver::DiagonalPreconditioner(VectorXc{{Complex(1.0, nan)}}),
                    std::invalid_argument);

    VectorXc zi;
    kNone.apply(VectorXc::Ones(3), zi);
    CHECK(zi == VectorXc::Ones(3));
    CHECK(kNone.name() == "none");
}

// Dense BEM systems of a sphere (two test cases, so each stays short under the sanitizers).
// Icosphere n = 1 (80 triangles, 2N = 240), r = 0.5 um, plane wave at lambda = 1 um: the same
// h / lambda as n = 2 at 500 nm (test_assembler.cpp), 16x cheaper to assemble (n = 2 took 36 s
// in win-debug). The 500 nm material constants are used at 1 um (this is a solver test).
// Measured at n = 2, lambda = 500 nm (2N = 960, tol 1e-8, win-release): Si ICTF 93 iterations,
// true residual 7.0e-9, |x - x_LU| / |x_LU| = 1.1e-7; Ag PMCHWT + left Jacobi 193 iterations,
// true residual 1.7e-7 (the stopping criterion is on the preconditioned residual), 6.7e-7.
// Observed at n = 1 (these tests): Si ICTF 53 iterations, true residual 8.0e-9,
// |x - x_LU| / |x_LU| = 1.1e-6; Ag PMCHWT + left Jacobi 46 iterations, true residual 8.1e-8,
// 1.1e-7.
namespace {
constexpr Real kLambdaBem = 1e-6;
}  // namespace

TEST_CASE("gmres: dense BEM system of a Si sphere (ICTF, no preconditioner) matches LU",
          "[gmres]") {
    using namespace assembler_test;
    SphereCase c(material::silicon_500nm(), 1, formulation::Kind::ICTF, kLambdaBem);
    const op::DenseOperator Z(assemble(c.problem()));
    const VectorXc b = op::assemble_rhs(c.problem());
    const GmresResult r = solver::gmres(Z, b, kNone, params(1e-8, 240));
    const VectorXc x_lu = solver::solve_direct(Z, b);
    INFO("Si ICTF: iterations " << r.iterations << ", true residual " << r.true_relative_residual
                                << ", |x - x_LU| / |x_LU| " << rel_diff(r.x, x_lu));
    CHECK(r.converged);
    CHECK(r.true_relative_residual <= 1e-6);
    CHECK(rel_diff(r.x, x_lu) <= 1e-5);
}

TEST_CASE("gmres: dense BEM system of an Ag sphere (PMCHWT, left Jacobi) matches LU", "[gmres]") {
    using namespace assembler_test;
    SphereCase c(material::silver_500nm(), 1, formulation::Kind::PMCHWT, kLambdaBem);
    const op::DenseOperator Z(assemble(c.problem()));
    const VectorXc b = op::assemble_rhs(c.problem());
    const solver::DiagonalPreconditioner M(op::assemble_diagonal(c.problem()));
    const GmresResult r = solver::gmres(Z, b, M, params(1e-8, 240));
    const VectorXc x_lu = solver::solve_direct(Z, b);
    INFO("Ag PMCHWT + Jacobi: iterations " << r.iterations << ", true residual "
                                           << r.true_relative_residual << ", |x - x_LU| / |x_LU| "
                                           << rel_diff(r.x, x_lu));
    CHECK(r.converged);
    CHECK(r.true_relative_residual <= 1e-6);
    CHECK(rel_diff(r.x, x_lu) <= 1e-5);
}
