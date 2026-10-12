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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "assembler_test_support.hpp"
#include "krylov_spectrum.hpp"

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

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

// --- Gram-Schmidt variants (WP-G1) -------------------------------------------------------------

namespace {

/// |I - V^H V|_F (an upper bound of the 2-norm).
Real orthogonality_loss(const MatrixXc& V) {
    return (MatrixXc::Identity(V.cols(), V.cols()) - V.adjoint() * V).norm();
}

GmresParams with(GmresParams p, solver::Orthogonalization o, bool keep_basis = false) {
    p.orthogonalization = o;
    p.keep_basis = keep_basis;
    return p;
}

/// Test double y = d .* x + U (W^H x): a cheap matvec, so that n can span several row blocks of
/// the parallel CGS2 products (4096 rows each), and deterministic (no BLAS).
class DiagonalPlusLowRank final : public op::LinearOperator {
public:
    DiagonalPlusLowRank(Index n, std::uint64_t seed)
        : d_(n), U_(random_matrix(n, 2, seed + 1)), W_(random_matrix(n, 2, seed + 2)) {
        // Eigenvalues of the diagonal part in the disk |z - 1| < 0.95: slow, steady convergence.
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<Real> u(0.0, 1.0);
        for (Index i = 0; i < n; ++i) {
            d_(i) = 1.0 + std::polar(0.95 * std::sqrt(u(rng)), 2.0 * constants::pi * u(rng));
        }
        U_ /= static_cast<Real>(n);
    }
    [[nodiscard]] Index rows() const override { return d_.size(); }
    [[nodiscard]] Index cols() const override { return d_.size(); }
    void apply(const VectorXc& x, VectorXc& y) const override {
        // Explicit dot products instead of a GEMV: a threaded BLAS GEMV may split the sums
        // over threads, which would make the operator depend on the thread count.
        const Complex c0 = W_.col(0).dot(x);
        const Complex c1 = W_.col(1).dot(x);
        y = d_.cwiseProduct(x) + c0 * U_.col(0) + c1 * U_.col(1);
    }
    [[nodiscard]] std::string describe() const override { return "DiagonalPlusLowRank"; }
    [[nodiscard]] std::size_t memory_bytes() const override { return 0; }
    [[nodiscard]] const VectorXc& diagonal() const { return d_; }

private:
    VectorXc d_;
    MatrixXc U_;
    MatrixXc W_;
};

constexpr Index kLargeN = 3 * 4096 + 123;  // three full row blocks and a partial one

}  // namespace

TEST_CASE("gmres: CGS2 and MGS on an ill-conditioned dense system (200 iterations)", "[gmres]") {
    // Badly row-scaled system (row scales 1e-3 .. 1e3), unpreconditioned: the residual decays
    // slowly, so all 200 iterations run and the basis has 201 columns.
    const Index n = 300;
    const MatrixXc A = wide_diagonal(n, 71).asDiagonal() * well_conditioned(n, 72);
    const op::DenseOperator Z(A);
    const VectorXc b = random_vector(n, 73);
    const GmresParams p = params(1e-14, 200);
    const GmresResult c =
        solver::gmres(Z, b, kNone, with(p, solver::Orthogonalization::CGS2, true));
    const GmresResult m = solver::gmres(Z, b, kNone, with(p, solver::Orthogonalization::MGS, true));
    REQUIRE(c.iterations == 200);
    REQUIRE(m.iterations == 200);
    REQUIRE(c.basis.cols() == 201);
    REQUIRE(c.basis.rows() == n);
    REQUIRE(m.basis.cols() == 201);
    const Real loss_c = orthogonality_loss(c.basis);
    const Real loss_m = orthogonality_loss(m.basis);
    Real max_hist = 0;
    for (std::size_t k = 0; k < c.residual_history.size(); ++k) {
        max_hist = std::max(max_hist, std::abs(c.residual_history[k] - m.residual_history[k]) /
                                          m.residual_history[k]);
    }
    VectorXc diff;
    Z.apply(c.x - m.x, diff);
    INFO("|I - V^H V|_F: CGS2 " << loss_c << ", MGS " << loss_m << "; re-orthogonalisations "
                                << c.reorthogonalizations << " / " << m.reorthogonalizations
                                << "; history rel. diff " << max_hist << "; residual "
                                << c.residual_history.back() << "; |A (x_c - x_m)| / |b| "
                                << diff.norm() / b.norm());
    CHECK_FALSE(c.converged);
    CHECK(loss_c <= 1e-12);
    CHECK(loss_m <= 1e-12);
    CHECK(max_hist <= 1e-8);
    CHECK(diff.norm() / b.norm() <= 1e-10);
    CHECK(c.reorthogonalizations == 200);  // CGS2 always makes two passes
    CHECK(m.reorthogonalizations <= 200);
    CHECK(c.orthogonalization_seconds > 0);
    CHECK(c.orthogonalization_seconds <= c.wall_seconds);
    CHECK(solver::gmres(Z, b, kNone, p).basis.size() == 0);  // keep_basis is off by default
}

TEST_CASE("gmres: CGS2 is bitwise deterministic across runs and thread counts", "[gmres]") {
    const DiagonalPlusLowRank Z(kLargeN, 81);
    const VectorXc b = random_vector(kLargeN, 82);
    const GmresParams p = with(params(1e-14, 60), solver::Orthogonalization::CGS2, true);
    const GmresResult r1 = solver::gmres(Z, b, kNone, p);
    const GmresResult r2 = solver::gmres(Z, b, kNone, p);
    REQUIRE(r1.iterations == 60);
    INFO("residual after 60 iterations " << r1.residual_history.back() << ", |I - V^H V|_F "
                                         << orthogonality_loss(r1.basis));
    CHECK(r1.x == r2.x);
    CHECK(r1.residual_history == r2.residual_history);
    CHECK(r1.basis == r2.basis);
    CHECK(orthogonality_loss(r1.basis) <= 1e-12);
#ifdef SPECKLEBEM_HAVE_OPENMP
    // The row-block partition and the order of the partial sums do not depend on the threads.
    const int threads = omp_get_max_threads();
    omp_set_num_threads(threads > 1 ? 1 : 3);
    const GmresResult r3 = solver::gmres(Z, b, kNone, p);
    omp_set_num_threads(threads);
    CHECK(r3.x == r1.x);
    CHECK(r3.residual_history == r1.residual_history);
#endif
}

TEST_CASE("gmres: CGS2 over several row blocks and basis panels (70 iterations)", "[gmres]") {
    // n spans three full row blocks and a partial one; 71 basis columns span a full panel of 64
    // and a remainder panel of 7 (full GMRES, m + 1 = 71).
    const DiagonalPlusLowRank Z(kLargeN, 101);
    const VectorXc b = random_vector(kLargeN, 102);
    const GmresParams p = params(1e-14, 70);
    const GmresResult c =
        solver::gmres(Z, b, kNone, with(p, solver::Orthogonalization::CGS2, true));
    const GmresResult m = solver::gmres(Z, b, kNone, with(p, solver::Orthogonalization::MGS, true));
    REQUIRE(c.iterations == 70);
    REQUIRE(m.iterations == 70);
    REQUIRE(c.basis.rows() == kLargeN);
    REQUIRE(c.basis.cols() == 71);
    REQUIRE(m.basis.cols() == 71);
    const Real loss_c = orthogonality_loss(c.basis);
    const Real loss_m = orthogonality_loss(m.basis);
    Real max_hist = 0;
    for (std::size_t k = 0; k < c.residual_history.size(); ++k) {
        max_hist = std::max(max_hist, std::abs(c.residual_history[k] - m.residual_history[k]) /
                                          m.residual_history[k]);
    }
    INFO("|I - V^H V|_F: CGS2 " << loss_c << ", MGS " << loss_m << "; residual "
                                << c.residual_history.back() << "; history rel. diff " << max_hist
                                << "; |x_c - x_m| / |x_m| " << rel_diff(c.x, m.x));
    CHECK_FALSE(c.converged);
    CHECK(loss_c <= 1e-12);
    CHECK(loss_m <= 1e-12);
    CHECK(max_hist <= 1e-10);
    CHECK(rel_diff(c.x, m.x) <= 1e-10);
    // A GMRES(66) cycle fills both panels exactly (64 + 3 columns) before the restart.
    const GmresResult r = solver::gmres(
        Z, b, kNone, with(params(1e-14, 70, 66), solver::Orthogonalization::CGS2, true));
    REQUIRE(r.iterations == 70);
    CHECK(r.basis.cols() == 5);  // last cycle: 4 iterations after the restart
    CHECK(orthogonality_loss(r.basis) <= 1e-12);
}

TEST_CASE("gmres: CGS2 matches MGS with left/right preconditioning and restart", "[gmres]") {
    const DiagonalPlusLowRank Z(kLargeN, 91);
    const VectorXc b = random_vector(kLargeN, 92);
    // Diagonal preconditioner |d| (magnitudes only; Jacobi on d would leave identity + rank 2):
    // the preconditioned eigenvalues lie on an arc of the unit circle.
    const solver::DiagonalPreconditioner M(Z.diagonal().cwiseAbs().cast<Complex>());
    for (const auto side : {PreconditionerSide::Left, PreconditionerSide::Right}) {
        for (const int restart : {0, 3}) {
            const GmresParams p = params(1e-10, 200, restart, side);
            const GmresResult c = solver::gmres(Z, b, M, with(p, solver::Orthogonalization::CGS2));
            const GmresResult m = solver::gmres(Z, b, M, with(p, solver::Orthogonalization::MGS));
            INFO((side == PreconditionerSide::Left ? "left" : "right")
                 << ", restart " << restart << ": iterations CGS2 " << c.iterations << ", MGS "
                 << m.iterations << "; true residual " << c.true_relative_residual
                 << ", |x_c - x_m| / |x_m| " << rel_diff(c.x, m.x));
            CHECK(c.converged);
            CHECK(m.converged);
            CHECK(std::abs(c.iterations - m.iterations) <= 2);
            CHECK(c.true_relative_residual <= 1e-9);
            CHECK(rel_diff(c.x, m.x) <= 1e-9);
            if (restart > 0)
                CHECK(c.iterations > restart);
        }
    }
}

// --- Hessenberg diagnostic hook (WP-A2) ----------------------------------------------------------

namespace {

using krylov_spectrum::eigenvalues;
using krylov_spectrum::harmonic_ritz_values;

/// Largest distance of an entry of `a` to its nearest entry of `b`, relative to max |b|.
Real spectral_distance(const VectorXc& a, const VectorXc& b) {
    Real worst = 0;
    for (Index i = 0; i < a.size(); ++i) {
        Real best = std::numeric_limits<Real>::infinity();
        for (Index j = 0; j < b.size(); ++j) best = std::min(best, std::abs(a(i) - b(j)));
        worst = std::max(worst, best);
    }
    return worst / b.cwiseAbs().maxCoeff();
}

/// |Op V_k - V H_bar|_F / |H_bar|_F for Op = M^{-1} A (left) or A M^{-1} (right), M^{-1} =
/// diag(inv_m). V holds V_{k+1}, or V_k after convergence (then the last row of H_bar is left
/// out of the product and the defect is taken over the first k - 1 columns; `tail_norm`
/// receives the norm of the last column of Op V_k - V_k H_k, which must equal |h_{k+1,k}|).
Real arnoldi_defect(const MatrixXc& A, const VectorXc& inv_m, PreconditionerSide side,
                    const MatrixXc& V, const MatrixXc& hbar, Real* tail_norm = nullptr) {
    const Index k = hbar.cols();
    const MatrixXc Vk = V.leftCols(k);
    const MatrixXc OpV = side == PreconditionerSide::Left ? MatrixXc(inv_m.asDiagonal() * A * Vk)
                                                          : MatrixXc(A * inv_m.asDiagonal() * Vk);
    const bool full = V.cols() == k + 1;
    const MatrixXc R = full ? MatrixXc(OpV - V * hbar) : MatrixXc(OpV - Vk * hbar.topRows(k));
    if (tail_norm != nullptr)
        *tail_norm = R.col(k - 1).norm();
    return (full ? R.norm() : R.leftCols(k - 1).norm()) / hbar.norm();
}

}  // namespace

TEST_CASE("gmres: keep_hessenberg is off by default and leaves the iterates bitwise unchanged",
          "[gmres]") {
    CHECK_FALSE(GmresParams{}.keep_hessenberg);
    const Index n = 60;
    const MatrixXc A = wide_diagonal(n, 101).asDiagonal() * well_conditioned(n, 102);
    const op::DenseOperator Z(A);
    const VectorXc b = random_vector(n, 103);
    const solver::DiagonalPreconditioner M(A.diagonal());
    for (const auto side : {PreconditionerSide::Left, PreconditionerSide::Right}) {
        for (const int restart : {0, 7}) {
            for (const auto o : {solver::Orthogonalization::CGS2, solver::Orthogonalization::MGS}) {
                CAPTURE(side == PreconditionerSide::Left, restart,
                        o == solver::Orthogonalization::CGS2);
                GmresParams p = with(params(1e-10, 50, restart, side), o);
                const GmresResult off = solver::gmres(Z, b, M, p);
                p.keep_hessenberg = true;
                const GmresResult on = solver::gmres(Z, b, M, p);
                CHECK(off.hessenberg.size() == 0);
                CHECK(on.hessenberg.size() > 0);
                CHECK(on.x == off.x);
                CHECK(on.residual_history == off.residual_history);
                CHECK(on.iterations == off.iterations);
                CHECK(on.true_relative_residual == off.true_relative_residual);
                CHECK(on.reorthogonalizations == off.reorthogonalizations);
            }
        }
    }
    // b = 0: no Krylov space, no Hessenberg.
    GmresParams p = params(1e-10);
    p.keep_hessenberg = true;
    CHECK(solver::gmres(Z, VectorXc::Zero(n), kNone, p).hessenberg.size() == 0);
}

namespace {

/// Test double: y = A x for the first `good_calls` applications, y = 0 afterwards (the
/// operator becomes singular on everything; exact zeros, so the breakdown test is not
/// rounding-dependent).
class ZeroAfterOperator final : public op::LinearOperator {
public:
    ZeroAfterOperator(MatrixXc A, int good_calls) : A_(std::move(A)), good_calls_(good_calls) {}
    [[nodiscard]] Index rows() const override { return A_.rows(); }
    [[nodiscard]] Index cols() const override { return A_.cols(); }
    void apply(const VectorXc& x, VectorXc& y) const override {
        ++calls_;
        y = calls_ <= good_calls_ ? VectorXc(A_ * x) : VectorXc::Zero(A_.rows());
    }
    [[nodiscard]] std::string describe() const override { return "ZeroAfterOperator"; }
    [[nodiscard]] std::size_t memory_bytes() const override { return 0; }

private:
    MatrixXc A_;
    int good_calls_;
    mutable int calls_ = 0;
};

}  // namespace

TEST_CASE("gmres: keep_hessenberg is empty when the first column of the last cycle is discarded",
          "[gmres]") {
    const MatrixXc A{{Complex(2.0, 0.0), Complex(0.0, 0.0)},
                     {Complex(0.0, 0.0), Complex(3.0, 0.0)}};
    const VectorXc b{{Complex(1.0, 0.0), Complex(1.0, 0.0)}};
    const auto both = {solver::Orthogonalization::CGS2, solver::Orthogonalization::MGS};
    SECTION("singular at the very first step (A b = 0)") {
        for (const auto o : both) {
            CAPTURE(o == solver::Orthogonalization::CGS2);
            GmresParams p = with(params(1e-14), o);
            p.keep_hessenberg = true;
            const MatrixXc A0{{Complex(0.0, 0.0), Complex(0.0, 0.0)},
                              {Complex(0.0, 0.0), Complex(1.0, 0.0)}};
            const VectorXc e0{{Complex(1.0, 0.0), Complex(0.0, 0.0)}};
            const GmresResult r = solver::gmres(op::DenseOperator(A0), e0, kNone, p);
            CHECK(r.iterations == 1);
            CHECK_FALSE(r.converged);
            CHECK(r.hessenberg.rows() == 0);
            CHECK(r.hessenberg.cols() == 0);
        }
    }
    SECTION("singular at the first step after a restart: earlier cycles are not kept") {
        for (const auto o : both) {
            CAPTURE(o == solver::Orthogonalization::CGS2);
            // GMRES(1): cycle 1 uses the first product (one kept column), the restart residual
            // the second (zero, so r = b), cycle 2 the third (zero: column discarded).
            GmresParams p = with(params(1e-14, 100, 1), o);
            p.keep_hessenberg = true;
            const GmresResult r = solver::gmres(ZeroAfterOperator(A, 1), b, kNone, p);
            CHECK(r.iterations == 2);
            CHECK_FALSE(r.converged);
            CHECK(r.hessenberg.rows() == 0);
            CHECK(r.hessenberg.cols() == 0);
            // Control: stopping in cycle 1 (max_iter = 1) returns its 2 x 1 matrix.
            GmresParams p1 = with(params(1e-14, 1, 1), o);
            p1.keep_hessenberg = true;
            const GmresResult r1 = solver::gmres(ZeroAfterOperator(A, 1), b, kNone, p1);
            CHECK(r1.hessenberg.rows() == 2);
            CHECK(r1.hessenberg.cols() == 1);
        }
    }
}

TEST_CASE("gmres: Ritz values of a full Krylov space reproduce the eigenvalues", "[gmres]") {
    // A = S D S^{-1} with a known spectrum D (fixed seed): eigenvalues spread over the annulus
    // 0.1 <= |z| <= 2 plus one small one (|d_0| = 0.05, a near-null mode), S well conditioned.
    // After n steps the Krylov space is the whole space (h_{n+1,n} at rounding level), so the
    // Ritz and the harmonic Ritz values both equal the eigenvalues of A.
    const Index n = 40;
    std::mt19937_64 rng(111);
    std::uniform_real_distribution<Real> u(0.0, 1.0);
    VectorXc d(n);
    for (Index i = 0; i < n; ++i)
        d(i) = std::polar(0.1 + 1.9 * u(rng), 2.0 * constants::pi * u(rng));
    d(0) = Complex(0.03, -0.04);
    const MatrixXc S = well_conditioned(n, 112);
    const MatrixXc A = S * d.asDiagonal() * S.inverse();
    const op::DenseOperator Z(A);
    const VectorXc b = random_vector(n, 113);
    GmresParams p = params(1e-300, static_cast<int>(n));
    p.keep_hessenberg = true;
    p.keep_basis = true;
    const GmresResult r = solver::gmres(Z, b, kNone, p);
    const Index k = r.hessenberg.cols();
    REQUIRE(k > 0);
    INFO("iterations " << r.iterations << ", Krylov dimension " << k << ", h_{k+1,k} "
                       << std::abs(r.hessenberg(k, k - 1)));
    REQUIRE(k == n);
    REQUIRE(r.hessenberg.rows() == n + 1);
    // Upper Hessenberg with a real non-negative subdiagonal (the Arnoldi norms).
    for (Index j = 0; j < k; ++j) {
        CHECK(r.hessenberg(j + 1, j).imag() == 0.0);
        CHECK(r.hessenberg(j + 1, j).real() >= 0.0);
        for (Index i = j + 2; i <= k; ++i) CHECK(r.hessenberg(i, j) == Complex(0.0));
    }
    CHECK(std::abs(r.hessenberg(k, k - 1)) <= 1e-12 * r.hessenberg.norm());
    const VectorXc ritz = krylov_spectrum::ritz_values(r.hessenberg);
    const VectorXc theta = harmonic_ritz_values(r.hessenberg);
    const Real dist_ritz = spectral_distance(ritz, d);
    const Real dist_back = spectral_distance(d, ritz);
    const Real dist_harm = spectral_distance(theta, d);
    INFO("Ritz " << dist_ritz << " / " << dist_back << ", harmonic " << dist_harm);
    CHECK(dist_ritz <= 1e-9);
    CHECK(dist_back <= 1e-9);
    CHECK(dist_harm <= 1e-9);
    // The smallest eigenvalue is found as the smallest harmonic Ritz value.
    Index imin = 0;
    theta.cwiseAbs().minCoeff(&imin);
    CHECK(std::abs(theta(imin) - d(0)) <= 1e-10);
    // H_bar = V_{n+1}^H A V_n with an orthonormal V_n: the singular values are those of A.
    const VectorXr sv = krylov_spectrum::singular_values(r.hessenberg);
    const VectorXr sv_ref = Eigen::BDCSVD<MatrixXc>(A).singularValues();
    CHECK((sv - sv_ref).cwiseAbs().maxCoeff() <= 1e-10 * sv_ref(0));
    // Arnoldi relation A V_k = V_{k+1} H_bar (V_k after the happy breakdown at k = n).
    REQUIRE(r.basis.cols() >= k);
    Real tail = 0;
    CHECK(arnoldi_defect(A, VectorXc::Ones(n), PreconditionerSide::Left, r.basis, r.hessenberg,
                         &tail) <= 1e-12);
    CHECK(tail <= 1e-11 * r.hessenberg.norm());
}

TEST_CASE("gmres: Hessenberg of the preconditioned operator, last cycle, Arnoldi relation",
          "[gmres]") {
    const Index n = 50;
    const MatrixXc A = wide_diagonal(n, 121).asDiagonal() * well_conditioned(n, 122);
    const op::DenseOperator Z(A);
    const VectorXc b = random_vector(n, 123);
    const solver::DiagonalPreconditioner M(A.diagonal());
    const VectorXc inv_m = A.diagonal().cwiseInverse();
    for (const auto side : {PreconditionerSide::Left, PreconditionerSide::Right}) {
        CAPTURE(side == PreconditionerSide::Left);
        {
            // Full Krylov space: the Ritz values are the eigenvalues of M^{-1} A (similar to
            // A M^{-1}).
            GmresParams p = params(1e-300, static_cast<int>(n), 0, side);
            p.keep_hessenberg = true;
            const GmresResult r = solver::gmres(Z, b, M, p);
            REQUIRE(r.hessenberg.cols() == n);
            // A M^{-1} = D (M^{-1} A) D^{-1} with the row scales D of 1e-3 .. 1e3 is strongly
            // non-normal: rounding of the right-preconditioned Arnoldi moves its eigenvalues
            // by ~1e-9 (left: ~1e-13).
            const Real tol = side == PreconditionerSide::Left ? 1e-10 : 1e-7;
            const VectorXc ref = eigenvalues(inv_m.asDiagonal() * A);
            CHECK(spectral_distance(eigenvalues(r.hessenberg.topRows(n)), ref) <= tol);
            CHECK(spectral_distance(harmonic_ritz_values(r.hessenberg), ref) <= tol);
        }
        {
            // Stopped at max_iter: V_{k+1} and the full relation.
            GmresParams p = params(1e-300, 20, 0, side);
            p.keep_hessenberg = true;
            p.keep_basis = true;
            const GmresResult r = solver::gmres(Z, b, M, p);
            REQUIRE(r.hessenberg.rows() == 21);
            REQUIRE(r.hessenberg.cols() == 20);
            REQUIRE(r.basis.cols() == 21);
            CHECK(arnoldi_defect(A, inv_m, side, r.basis, r.hessenberg) <= 1e-13);
            // sigma_min(H_bar) bounds sigma_min of the preconditioned operator from above.
            const MatrixXc op_matrix = side == PreconditionerSide::Left
                                           ? MatrixXc(inv_m.asDiagonal() * A)
                                           : MatrixXc(A * inv_m.asDiagonal());
            const VectorXr sv_op = Eigen::BDCSVD<MatrixXc>(op_matrix).singularValues();
            const VectorXr sv = krylov_spectrum::singular_values(r.hessenberg);
            CHECK(sv(sv.size() - 1) >= sv_op(sv_op.size() - 1) * (1.0 - 1e-12));
            CHECK(sv(0) <= sv_op(0) * (1.0 + 1e-12));
        }
        {
            // Restarted, cycles of 7, 7 and 3 steps: the Hessenberg of the last cycle only.
            GmresParams p = params(1e-300, 17, 7, side);
            p.keep_hessenberg = true;
            p.keep_basis = true;
            const GmresResult r = solver::gmres(Z, b, M, p);
            REQUIRE(r.iterations == 17);
            REQUIRE(r.hessenberg.rows() == 4);
            REQUIRE(r.hessenberg.cols() == 3);
            REQUIRE(r.basis.cols() == 4);
            CHECK(arnoldi_defect(A, inv_m, side, r.basis, r.hessenberg) <= 1e-13);
        }
        {
            // Converged: V_k; the last row holds the norm of the vector that was not formed.
            GmresParams p = params(1e-6, 200, 0, side);
            p.keep_hessenberg = true;
            p.keep_basis = true;
            const GmresResult r = solver::gmres(Z, b, M, p);
            REQUIRE(r.converged);
            const Index k = r.hessenberg.cols();
            REQUIRE(k == r.iterations);
            REQUIRE(r.basis.cols() == k);
            Real tail = 0;
            CHECK(arnoldi_defect(A, inv_m, side, r.basis, r.hessenberg, &tail) <= 1e-13);
            CHECK(std::abs(tail - std::abs(r.hessenberg(k, k - 1))) <= 1e-12 * r.hessenberg.norm());
            CHECK(std::abs(r.hessenberg(k, k - 1)) > 0.0);
        }
    }
}
