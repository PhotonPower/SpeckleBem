#pragma once
/// @file direct.hpp
/// Dense LU solves for reference solutions of small and medium problems.
///
/// Backends (WP-LU1): LAPACK zgetrf/zgetrs (multithreaded through the BLAS, e.g. OpenBLAS)
/// when CMake found LAPACK and a probe linking exactly these routines succeeded
/// (lapack_available()); Eigen's PartialPivLU otherwise. Both give the same results to rounding;
/// the backend used is reported in DirectSolveInfo::backend and logged once per process.
#include "specklebem/core/types.hpp"
#include "specklebem/operator/dense_operator.hpp"

#include <Eigen/Core>

#include <vector>

namespace specklebem::solver {

/// Equilibrated reciprocal condition number below which a solve is flagged (and logged).
inline constexpr Real kIllConditionedRcond = 1e-10;

/// Dense LU backend. Auto = LAPACK when lapack_available(), else Eigen.
enum class LuBackend {
    Auto,    ///< request only: LAPACK if available, else Eigen
    Lapack,  ///< LAPACK zgetrf / zgetrs (32-bit LAPACK integers)
    Eigen    ///< Eigen::PartialPivLU (available in every build)
};

/// "auto", "lapack" or "eigen".
[[nodiscard]] const char* to_string(LuBackend backend) noexcept;

/// True when the library was built with the LAPACK LU path (SPECKLEBEM_HAVE_LAPACK).
[[nodiscard]] bool lapack_available() noexcept;

/// Diagnostics of one solve (solve_direct or DenseLu::solve).
struct DirectSolveInfo {
    Real rcond = 0;                ///< 1-norm rcond estimate of the column-equilibrated matrix
    Real residual = 0;             ///< |b - Z x| / |b| of the returned x (|b - Z x| if b = 0)
    int refinement_steps = 0;      ///< accepted iterative-refinement steps (0..2)
    bool ill_conditioned = false;  ///< rcond < kIllConditionedRcond (also logged as a warning)
    LuBackend backend = LuBackend::Auto;  ///< backend that factorised (Lapack or Eigen)
};

/// Dense LU factorisation of a square complex matrix Z, reusable for any number of right-hand
/// sides (factor once, solve many).
///
/// - Equilibration: the columns of Z are scaled by powers of two to unit max-norm while they are
///   copied into the LU storage (Z D), so the singularity test is invariant to column scaling;
///   x = D (Z D)^{-1} b. Z itself is not modified.
/// - Factorisation: LU with partial pivoting of Z D, by LAPACK zgetrf (LuBackend::Lapack) or
///   Eigen's PartialPivLU (LuBackend::Eigen).
/// - rcond = 1 / (|Z D|_1 |(Z D)^{-1}|_1), with the Hager/Higham estimate of the inverse norm
///   (the algorithm of LAPACK zlacn2, which zgecon and Eigen's PartialPivLU::rcond() use) driven
///   by the backend's LU solves, 4-11 solves with one right-hand side. (zgecon itself is not
///   used: one estimator for both backends, and its level-2 zlatrs solves cost 2-4x the
///   zgetrs-driven estimate on a 2N = 11440 PML matrix, far more on a loaded machine;
///   benchmarks/results/dense_lu.md.)
/// - Refinement: every solve runs at most two steps of fixed-precision iterative refinement
///   against the original Z, each kept (per right-hand side) only if it reduces |b - Z x|.
/// - Memory: one n x n complex copy (16 n^2 bytes) for the LU factors, n pivot indices and O(n)
///   work vectors (O(n m) for m right-hand sides). Z is referenced, not copied: it must outlive
///   the DenseLu and stay unchanged (temporaries are rejected at compile time).
/// - Errors (constructor): std::invalid_argument if Z is not square or has non-finite entries,
///   if LuBackend::Lapack is requested but not available, or if n exceeds the 32-bit LAPACK
///   integer range on the LAPACK path; std::runtime_error if Z is singular, i.e. an exactly
///   zero (or non-finite) pivot or an equilibrated rcond below machine epsilon;
///   std::logic_error for an illegal LAPACK argument (info < 0, a bug). eps <= rcond <
///   kIllConditionedRcond logs a warning (once per factorisation) and sets ill_conditioned().
/// - An empty (0 x 0) matrix is valid (rcond 1); its solves return empty results.
class DenseLu {
public:
    /// Factorises Z (see the class documentation for errors).
    explicit DenseLu(const MatrixXc& Z, LuBackend backend = LuBackend::Auto);
    explicit DenseLu(const op::DenseOperator& Z, LuBackend backend = LuBackend::Auto);
    DenseLu(const MatrixXc&& Z, LuBackend backend = LuBackend::Auto) = delete;
    DenseLu(const op::DenseOperator&& Z, LuBackend backend = LuBackend::Auto) = delete;

    /// Solves Z x = b. Diagnostics (rcond, backend, ill_conditioned of the factorisation; the
    /// residual and refinement steps of this solve) go to *info when info is non-null.
    /// The two overloads take a VectorXc and a MatrixXc: an Eigen expression such as B.col(j)
    /// converts to both (ambiguous call) and must be passed as VectorXc(B.col(j)) or MatrixXc(...).
    /// @throws std::invalid_argument if b.size() != size() or b has non-finite entries.
    [[nodiscard]] VectorXc solve(const VectorXc& b, DirectSolveInfo* info = nullptr) const;
    /// Solves Z X = B for all columns of B at once (one zgetrs call / blocked triangular solves).
    /// Each column is refined independently as in solve(b); *info gets the largest relative
    /// residual and the largest number of accepted refinement steps over the columns.
    /// @throws std::invalid_argument if B.rows() != size() or B has non-finite entries.
    [[nodiscard]] MatrixXc solve(const MatrixXc& B, DirectSolveInfo* info = nullptr) const;

    [[nodiscard]] Index size() const noexcept { return n_; }
    /// Backend that factorised: LuBackend::Lapack or LuBackend::Eigen (never Auto).
    [[nodiscard]] LuBackend backend() const noexcept { return backend_; }
    [[nodiscard]] Real rcond() const noexcept { return rcond_; }
    [[nodiscard]] bool ill_conditioned() const noexcept { return rcond_ < kIllConditionedRcond; }
    /// Wall time of the factorisation (equilibration, LU, rcond) [s].
    [[nodiscard]] Real factor_seconds() const noexcept { return factor_seconds_; }

private:
    /// Unrefined x = D (Z D)^{-1} b, in place (vector: the same triangular-solve path as
    /// Eigen's PartialPivLU::solve for a vector; matrix: blocked solves).
    void apply_inverse(VectorXc& x) const;
    void apply_inverse(MatrixXc& X) const;
    /// x = (Z D)^{-1} x, or (Z D)^{-H} x if adjoint (no column scaling), in place.
    void lu_solve(VectorXc& x, bool adjoint) const;
    /// Hager/Higham lower-bound estimate of |(Z D)^{-1}|_1 (LAPACK zlacn2's algorithm).
    [[nodiscard]] Real inverse_norm1_estimate() const;
    /// Factorisation diagnostics (rcond, ill_conditioned, backend) into a fresh info.
    [[nodiscard]] DirectSolveInfo factor_info() const;

    const MatrixXc* Z_ = nullptr;  ///< original matrix (refinement)
    Index n_ = 0;
    LuBackend backend_ = LuBackend::Eigen;
    MatrixXc lu_;                     ///< LU factors of Z D (unit lower L below the diagonal)
    VectorXr scales_;                 ///< D (powers of two)
    std::vector<int> lapack_pivots_;  ///< zgetrf ipiv (1-based), LAPACK path
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> eigen_perm_;  ///< P (Eigen path)
    Real rcond_ = 1;
    Real factor_seconds_ = 0;
};

/// Solve Z x = b by dense LU with partial pivoting: DenseLu(Z).solve(b, info), with the solve
/// time logged. Equilibration, refinement, memory (one n x n copy), errors and diagnostics as
/// documented for DenseLu (LuBackend::Auto). An empty (0 x 0) system returns an empty vector.
VectorXc solve_direct(const op::DenseOperator& Z, const VectorXc& b,
                      DirectSolveInfo* info = nullptr);

namespace detail {
/// Value as a 32-bit LAPACK integer.
/// @throws std::invalid_argument if value < 0 or value > INT_MAX (what names the quantity).
[[nodiscard]] int checked_lapack_int(Index value, const char* what);
}  // namespace detail

}  // namespace specklebem::solver
