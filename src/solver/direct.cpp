#include "specklebem/solver/direct.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specklebem::solver {

namespace {

constexpr int kMaxRefinementSteps = 2;

/// Power-of-two column scale d_j with max_i |Z_ij d_j| in [0.5, 1); 1 for zero / non-finite
/// columns (those are caught by the pivot check). Scaling by powers of two is exact.
VectorXr column_scales(const MatrixXc& A) {
    VectorXr d = VectorXr::Ones(A.cols());
    for (Index j = 0; j < A.cols(); ++j) {
        Real m = 0;
        for (Index i = 0; i < A.rows(); ++i) {
            m = std::max(m, std::abs(A(i, j)));
        }
        if (m > 0 && std::isfinite(m)) {
            int e = 0;
            (void)std::frexp(m, &e);
            d(j) = std::ldexp(1.0, -e);
        }
    }
    return d;
}

}  // namespace

VectorXc solve_direct(const op::DenseOperator& Z, const VectorXc& b, DirectSolveInfo* info) {
    const MatrixXc& A = Z.matrix();
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("solve_direct: matrix must be square (" +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                    ")");
    }
    if (b.size() != A.rows()) {
        throw std::invalid_argument("solve_direct: right-hand side has length " +
                                    std::to_string(b.size()) + ", expected " +
                                    std::to_string(A.rows()));
    }
    if (!b.allFinite()) {
        throw std::invalid_argument("solve_direct: right-hand side contains non-finite entries");
    }
    DirectSolveInfo local;
    DirectSolveInfo& out = info ? *info : local;
    out = DirectSolveInfo{};
    const Index n = A.rows();
    if (n == 0) {
        out.rcond = 1;
        return VectorXc();
    }

    ScopedTimer timer("solve_direct (dense LU, n = " + std::to_string(n) + ")");

    // Column equilibration A D (D = diag of powers of two) while copying into the LU storage:
    // the only n x n allocation. The singularity test below is then invariant to column scaling
    // (e.g. the eta^2 ratio between the J and M blocks of PMCHWT).
    const VectorXr d = column_scales(A);
    const Eigen::Array<Complex, Eigen::Dynamic, 1> dc = d.array().cast<Complex>();
    MatrixXc AD(n, n);
    for (Index j = 0; j < n; ++j) {
        AD.col(j) = A.col(j) * d(j);
    }
    const Eigen::PartialPivLU<Eigen::Ref<MatrixXc>> lu(AD);  // factorises AD in place

    // Singularity criterion: an exactly zero (or non-finite) pivot of U, or a reciprocal 1-norm
    // condition number estimate of the equilibrated matrix below machine epsilon.
    const auto& factors = lu.matrixLU();
    for (Index i = 0; i < n; ++i) {
        const Complex pivot = factors(i, i);
        if (pivot == Complex(0, 0) || !std::isfinite(pivot.real()) ||
            !std::isfinite(pivot.imag())) {
            throw std::runtime_error("solve_direct: matrix is singular (zero pivot in column " +
                                     std::to_string(i) + ")");
        }
    }
    const Real rcond = lu.rcond();
    out.rcond = rcond;
    if (!(rcond >= std::numeric_limits<Real>::epsilon())) {
        std::ostringstream msg;
        msg << "solve_direct: matrix is numerically singular (equilibrated rcond = "
            << std::scientific << rcond << " < machine epsilon)";
        throw std::runtime_error(msg.str());
    }
    out.ill_conditioned = rcond < kIllConditionedRcond;

    // x = D (A D)^{-1} b
    VectorXc x = (lu.solve(b).array() * dc).matrix();

    // Fixed-precision iterative refinement against the original Z (as LAPACK zgerfs): O(n^2) per
    // step. It brings the residual |b - Z x| / |b| of ill-conditioned random matrices from ~1e-12
    // down to ~1e-14 (n = 1000); a step is kept only if it reduces the residual.
    VectorXc r = b - A * x;
    Real r_norm = r.norm();
    for (int step = 0; step < kMaxRefinementSteps && r_norm > 0; ++step) {
        VectorXc x_new = x + (lu.solve(r).array() * dc).matrix();
        VectorXc r_new = b - A * x_new;
        const Real r_new_norm = r_new.norm();
        if (!(r_new_norm < r_norm)) {
            break;
        }
        x.swap(x_new);
        r.swap(r_new);
        r_norm = r_new_norm;
        ++out.refinement_steps;
    }
    const Real b_norm = b.norm();
    out.residual = b_norm > 0 ? r_norm / b_norm : r_norm;

    if (out.ill_conditioned) {
        SBEM_WARN(
            "solve_direct: ill-conditioned system (n = {}, equilibrated rcond = {:.3e} < {:.0e}); "
            "relative residual {:.3e} after {} refinement step(s)",
            n, rcond, kIllConditionedRcond, out.residual, out.refinement_steps);
    }
    SBEM_DEBUG("solve_direct: n = {}, rcond = {:.3e}, relative residual = {:.3e}, {} refinement", n,
               rcond, out.residual, out.refinement_steps);
    return x;
}

}  // namespace specklebem::solver
