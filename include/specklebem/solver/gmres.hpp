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

/// Gram-Schmidt variant of the Arnoldi step (WP-G1).
///
/// - CGS2 (default): classical Gram-Schmidt with re-orthogonalisation, always two passes
///   ("twice is enough": orthogonality to working precision, Giraud et al., Numer. Math. 101,
///   2005): h_i = V^H w, w -= V h_i for i = 1, 2, Hessenberg column h = h_1 + h_2. The products
///   are OpenMP-parallel over fixed blocks of rows (handed to the threads dynamically, one block
///   at a time), and the update of pass 1 is fused with the projection of pass 2 (three sweeps
///   over V per step). Each block's partial sums of V^H w depend on its rows only and are added
///   in block order, so results are bitwise reproducible from run to run and independent of the
///   number of threads and of their scheduling (for deterministic A and M^{-1}).
/// - MGS: modified Gram-Schmidt, serial, column by column, with a second pass when |w| drops
///   below 0.7 of its norm before the first (the algorithm before WP-G1). Kept for comparison
///   and as a fallback; never chosen automatically.
///
/// The variants differ by rounding only: their iterates are not bitwise equal, and iteration
/// counts can differ by one or two when the residual crosses the tolerance.
enum class Orthogonalization { CGS2, MGS };

struct GmresParams {
    /// Stopping tolerance on the monitored relative residual (see PreconditionerSide).
    Real tolerance = 1e-3;
    int max_iter = 2000;  ///< maximum number of iterations (matrix-vector products in Arnoldi)
    int restart = 0;      ///< 0 => no restart (full GMRES), m > 0 => GMRES(m)
    bool verbose = true;  ///< SBEM_INFO progress every 50 iterations and a summary line
    PreconditionerSide side = PreconditionerSide::Left;
    Orthogonalization orthogonalization = Orthogonalization::CGS2;
    /// Test/diagnostic hook: return the Krylov basis of the last cycle in GmresResult::basis
    /// (a copy, up to (m + 1) n complex values).
    bool keep_basis = false;
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
    /// Wall time of the Gram-Schmidt steps (projections, updates, norms and normalisation of
    /// the new basis vector); part of wall_seconds.
    double orthogonalization_seconds = 0;
    /// Number of Arnoldi steps that made a second Gram-Schmidt pass (CGS2: all of them).
    int reorthogonalizations = 0;
    /// With GmresParams::keep_basis: the Krylov vectors of the last cycle as columns
    /// (orthonormal up to rounding); empty otherwise.
    MatrixXc basis;
};

/// Called once per iteration with (k, residual_history[k]), k = 1 .. iterations.
using IterationCallback = std::function<void(int iter, Real residual)>;

/// Solve A x = b by preconditioned GMRES with the initial guess x0 = 0.
///
/// - Arnoldi with classical (CGS2, default) or modified Gram-Schmidt (see Orthogonalization);
///   Givens rotations on the Hessenberg matrix; the iterate is formed at convergence, at each
///   restart and at max_iter. At a restart the residual is recomputed explicitly and starts
///   the next cycle; it is not recorded: residual_history (and the callback) always hold the
///   Arnoldi estimates, entry for entry identical. If the recomputed residual is exactly zero,
///   the iteration stops with that x; residual_history.back() stays the estimate that was
///   reported, and converged follows it (true_relative_residual shows the exact residual).
/// - Happy breakdown: if h_{k+1,k} <= 100 eps |w_k| (w_k the new Arnoldi vector before
///   orthogonalisation) the Krylov space is invariant; the iteration stops with the
///   least-squares solution in it (exact up to rounding; converged if the monitored residual
///   is <= tolerance). If the rotated diagonal is below the same threshold, the operator is
///   singular on the Krylov space: the column is discarded, the iteration stops and a warning
///   is logged.
/// - Reaching max_iter is not an error: converged = false and a warning is logged.
/// - Memory: the Krylov basis dominates, at most (m + 1) n complex values (16 (m + 1) n bytes)
///   with m = restart, or m = max_iter for full GMRES. It grows on demand in contiguous
///   column-major panels of 64 columns, so it is never copied; the last panel holds only the
///   remainder of m + 1, so a filled basis (e.g. a GMRES(100) cycle: 64 + 37 columns) takes
///   exactly 16 (m + 1) n bytes, and a run that stops after k iterations allocates
///   min(m + 1, 64 ceil((k + 1) / 64)) columns. The Hessenberg factor adds 16 m^2 / 2 bytes. Cost
///   per iteration: one A and one M^{-1} application plus O(k n) for the orthogonalisation,
///   memory-bandwidth bound: CGS2 streams the k basis vectors three times per step, in parallel;
///   MGS once per pass (the second read of each vector mostly hits the cache), serially.
/// - OpenMP only in the CGS2 products and in forming the iterate V y; A and M^{-1} parallelise
///   their own work.
/// @throws std::invalid_argument if A is not square, b.size() != A.rows(), b has non-finite
///         entries, tolerance is <= 0 or non-finite, max_iter < 1 or restart < 0.
/// @throws std::runtime_error if M^{-1} maps b != 0 to zero (singular preconditioner), or if
///         A or M^{-1} produces non-finite values (checked on the norm of the initial residual,
///         of every new Arnoldi vector and of every recomputed restart residual; the solver
///         stops at the first such iteration instead of returning NaN).
GmresResult gmres(const op::LinearOperator& A, const VectorXc& b, const Preconditioner& M,
                  const GmresParams& p, const IterationCallback& cb = {});

}  // namespace specklebem::solver
