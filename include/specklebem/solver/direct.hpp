#pragma once
/// @file direct.hpp
/// Dense LU solve for reference solutions of small problems.
///
/// Implementation: Eigen PartialPivLU (LU with partial pivoting); its matrix products use BLAS
/// when CMake found BLAS/LAPACK (EIGEN_USE_BLAS). A LAPACK zgesv path is planned.
#include "specklebem/operator/dense_operator.hpp"

namespace specklebem::solver {

/// Equilibrated reciprocal condition number below which a solve is flagged (and logged).
inline constexpr Real kIllConditionedRcond = 1e-10;

/// Diagnostics of one solve_direct call.
struct DirectSolveInfo {
    Real rcond = 0;                ///< 1-norm rcond estimate of the column-equilibrated matrix
    Real residual = 0;             ///< |b - Z x| / |b| of the returned x (|b - Z x| if b = 0)
    int refinement_steps = 0;      ///< accepted iterative-refinement steps (0..2)
    bool ill_conditioned = false;  ///< rcond < kIllConditionedRcond (also logged as a warning)
};

/// Solve Z x = b by dense LU with partial pivoting.
///
/// - Equilibration: the columns of Z are scaled by powers of two to unit max-norm while they are
///   copied into the LU storage (Z D), so the singularity test is invariant to column scaling;
///   x = D (Z D)^{-1} b. Z itself is not modified.
/// - Refinement: at most two steps of fixed-precision iterative refinement against the original
///   Z, each kept only if it reduces |b - Z x|.
/// - Memory: one n x n complex copy (16 n^2 bytes) for the LU factors, plus O(n) vectors.
/// - Errors: std::invalid_argument if Z is not square, b.size() != Z.rows() or b has non-finite
///   entries; std::runtime_error if Z is singular, i.e. an exactly zero (or non-finite) pivot or
///   an equilibrated rcond below machine epsilon. eps <= rcond < kIllConditionedRcond logs a
///   warning and sets info->ill_conditioned.
/// - The solve time is logged; diagnostics are written to *info when info is non-null.
///   An empty (0 x 0) system returns an empty vector.
VectorXc solve_direct(const op::DenseOperator& Z, const VectorXc& b,
                      DirectSolveInfo* info = nullptr);

}  // namespace specklebem::solver
