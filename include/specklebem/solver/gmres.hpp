#pragma once
/// @file gmres.hpp
/// Complex GMRES (Saad & Schultz) with optional restart. The paper reports
/// that restarts slow convergence for these problems, so the default is
/// full GMRES with a residual tolerance of 1e-3.
#include "specklebem/operator/linear_operator.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <functional>
#include <vector>

namespace specklebem::solver {

/// Side on which the preconditioner M^{-1} is applied (design decision, issue #15).
///
/// - Left (default, as in the paper): GMRES on M^{-1} A x = M^{-1} b. The monitored and
///   stopping quantity is the *preconditioned* relative residual
///   |M^{-1}(b - A x)| / |M^{-1} b| (the Arnoldi least-squares estimate). With a Jacobi
///   preconditioner it is invariant under any diagonal row scaling of A, so preconditioned
///   PMCHWT, ICTF and MCTF (block-row scalings of each other, docs/03) iterate identically.
/// - Right: GMRES on A M^{-1} y = b, x = M^{-1} y. The monitored quantity is the *true*
///   relative residual |b - A x| / |b| (again the Arnoldi estimate).
///
/// With IdentityPreconditioner both sides are the same unpreconditioned GMRES.
enum class PreconditionerSide { Left, Right };

struct GmresParams {
    /// Stopping tolerance on the monitored relative residual (see PreconditionerSide).
    Real tolerance = 1e-3;
    int max_iter = 2000;  ///< maximum number of iterations (matrix-vector products in Arnoldi)
    int restart = 0;      ///< 0 => no restart (full GMRES), m > 0 => GMRES(m)
    bool verbose = true;  ///< SBEM_INFO progress every 50 iterations and a summary line
    PreconditionerSide side = PreconditionerSide::Left;
};

struct GmresResult {
    VectorXc x;
    /// Monitored relative residual: entry 0 is the initial one (1 for x0 = 0, b != 0), entry k
    /// the value after k iterations; size iterations + 1. For b = 0 it is {0}.
    std::vector<Real> residual_history;
    int iterations = 0;
    bool converged = false;  ///< residual_history.back() <= tolerance
    double wall_seconds = 0;
    /// |b - A x| / |b| of the returned x, computed explicitly (one extra matvec) for both
    /// sides; 0 for b = 0.
    Real true_relative_residual = 0;
};

/// Called once per iteration with (k, residual_history[k]), k = 1 .. iterations.
using IterationCallback = std::function<void(int iter, Real residual)>;

/// Solve A x = b by preconditioned GMRES with the initial guess x0 = 0.
///
/// - Arnoldi with modified Gram-Schmidt and one re-orthogonalisation pass whenever the norm of
///   the new vector drops below 0.7 of its norm before orthogonalisation ("twice is enough");
///   Givens rotations on the Hessenberg matrix; the iterate is formed at convergence, at each
///   restart and at max_iter. At a restart the residual is recomputed explicitly.
/// - Happy breakdown: if h_{k+1,k} <= 100 eps |w_k| (w_k the new Arnoldi vector before
///   orthogonalisation) the Krylov space is invariant; the iteration stops with the
///   least-squares solution in it (exact up to rounding; converged if the monitored residual
///   is <= tolerance). If the rotated diagonal is below the same threshold, the operator is
///   singular on the Krylov space: the column is discarded, the iteration stops and a warning
///   is logged.
/// - Reaching max_iter is not an error: converged = false and a warning is logged.
/// - Memory: the Krylov basis dominates, (m + 1) n complex values (16 (m + 1) n bytes) with
///   m = restart, or m = iterations for full GMRES (the basis grows on demand); the
///   Hessenberg factor adds 16 m^2 / 2 bytes. Cost per iteration: one A and one M^{-1}
///   application plus O(k n) for the orthogonalisation.
/// - No OpenMP of its own (A and Eigen/BLAS parallelise the work).
/// @throws std::invalid_argument if A is not square, b.size() != A.rows(), b has non-finite
///         entries, tolerance is <= 0 or non-finite, max_iter < 1 or restart < 0.
/// @throws std::runtime_error if M^{-1} maps b != 0 to zero (singular preconditioner).
GmresResult gmres(const op::LinearOperator& A, const VectorXc& b, const Preconditioner& M,
                  const GmresParams& p, const IterationCallback& cb = {});

}  // namespace specklebem::solver
