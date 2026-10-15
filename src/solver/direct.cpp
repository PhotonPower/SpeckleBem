#include "specklebem/solver/direct.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef SPECKLEBEM_HAVE_LAPACK
extern "C" {
// LAPACK, Fortran interface with LP64 (32-bit) integers as in OpenBLAS, reference LAPACK and MKL
// LP64; the trailing std::size_t is the hidden length of the character argument (gfortran >= 8;
// an f2c/C implementation ignores it). The link is verified by cmake/Dependencies.cmake.
void zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv,
             int* info);
void zgetrs_(const char* trans, const int* n, const int* nrhs, const std::complex<double>* a,
             const int* lda, const int* ipiv, std::complex<double>* b, const int* ldb, int* info,
             std::size_t trans_len);
}
#endif

namespace specklebem::solver {

namespace {

constexpr int kMaxRefinementSteps = 2;

static_assert(std::is_same_v<Complex, std::complex<double>>,
              "the LAPACK z-routines need Complex = std::complex<double>");

/// Power-of-two column scale d_j with max_i |Z_ij d_j| in [0.5, 1); 1 for zero columns (those
/// are caught by the pivot check). Scaling by powers of two is exact.
/// @throws std::invalid_argument for a non-finite entry.
VectorXr column_scales(const MatrixXc& A) {
    VectorXr d = VectorXr::Ones(A.cols());
    for (Index j = 0; j < A.cols(); ++j) {
        Real m = 0;
        bool finite = true;
        for (Index i = 0; i < A.rows(); ++i) {
            const Complex a = A(i, j);
            finite = finite && std::isfinite(a.real()) && std::isfinite(a.imag());
            m = std::max(m, std::abs(a));
        }
        if (!finite) {
            throw std::invalid_argument("dense LU: matrix has non-finite entries (column " +
                                        std::to_string(j) + ")");
        }
        if (m > 0) {
            int e = 0;
            (void)std::frexp(m, &e);
            d(j) = std::ldexp(1.0, -e);
        }
    }
    return d;
}

LuBackend resolve_backend(LuBackend requested) {
    switch (requested) {
        case LuBackend::Auto:
            return lapack_available() ? LuBackend::Lapack : LuBackend::Eigen;
        case LuBackend::Lapack:
            if (!lapack_available()) {
                throw std::invalid_argument(
                    "dense LU: LuBackend::Lapack requested but the library was built without "
                    "LAPACK (zgetrf_/zgetrs_ not found or not linkable at configure time)");
            }
            return LuBackend::Lapack;
        case LuBackend::Eigen:
            return LuBackend::Eigen;
    }
    throw std::invalid_argument("dense LU: unknown LuBackend value");
}

/// Logs the backend at info level once per process and backend.
void log_backend_once(LuBackend backend) {
    static std::once_flag lapack_flag;
    static std::once_flag eigen_flag;
    if (backend == LuBackend::Lapack) {
        std::call_once(lapack_flag, [] { SBEM_INFO("dense LU backend: LAPACK (zgetrf/zgetrs)"); });
    } else {
        std::call_once(eigen_flag, [] { SBEM_INFO("dense LU backend: Eigen PartialPivLU"); });
    }
}

void check_rhs(const MatrixXc& B, Index n) {
    if (B.rows() != n) {
        throw std::invalid_argument("dense LU: right-hand side has length " +
                                    std::to_string(B.rows()) + ", expected " + std::to_string(n));
    }
    if (!B.allFinite()) {
        throw std::invalid_argument("dense LU: right-hand side contains non-finite entries");
    }
}

void check_rhs(const VectorXc& b, Index n) {
    if (b.size() != n) {
        throw std::invalid_argument("dense LU: right-hand side has length " +
                                    std::to_string(b.size()) + ", expected " + std::to_string(n));
    }
    if (!b.allFinite()) {
        throw std::invalid_argument("dense LU: right-hand side contains non-finite entries");
    }
}

#ifdef SPECKLEBEM_HAVE_LAPACK
/// zgetrs on the columns of X (in place): A X = B (trans 'N') or A^H X = B (trans 'C').
void lapack_getrs(const MatrixXc& lu, const std::vector<int>& ipiv, Complex* x, Index n, Index nrhs,
                  char trans = 'N') {
    const int ni = detail::checked_lapack_int(n, "matrix dimension");
    const int mi = detail::checked_lapack_int(nrhs, "number of right-hand sides");
    const int ld = std::max(ni, 1);
    int info = 0;
    zgetrs_(&trans, &ni, &mi, lu.data(), &ld, ipiv.data(), x, &ld, &info, 1);
    if (info != 0) {
        throw std::logic_error("dense LU: zgetrs reported an illegal argument, info = " +
                               std::to_string(info));
    }
}
#endif

}  // namespace

const char* to_string(LuBackend backend) noexcept {
    switch (backend) {
        case LuBackend::Auto:
            return "auto";
        case LuBackend::Lapack:
            return "lapack";
        case LuBackend::Eigen:
            return "eigen";
    }
    return "unknown";
}

bool lapack_available() noexcept {
#ifdef SPECKLEBEM_HAVE_LAPACK
    return true;
#else
    return false;
#endif
}

int detail::checked_lapack_int(Index value, const char* what) {
    if (value < 0 || value > static_cast<Index>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("dense LU: ") + what + " " + std::to_string(value) +
                                    " is outside the 32-bit LAPACK integer range [0, " +
                                    std::to_string(std::numeric_limits<int>::max()) + "]");
    }
    return static_cast<int>(value);
}

DenseLu::DenseLu(const op::DenseOperator& Z, LuBackend backend) : DenseLu(Z.matrix(), backend) {}

DenseLu::DenseLu(const MatrixXc& Z, LuBackend backend) : Z_(&Z), n_(Z.rows()) {
    if (Z.rows() != Z.cols()) {
        throw std::invalid_argument("dense LU: matrix must be square (" + std::to_string(Z.rows()) +
                                    "x" + std::to_string(Z.cols()) + ")");
    }
    backend_ = resolve_backend(backend);
    if (backend_ == LuBackend::Lapack) {
        (void)detail::checked_lapack_int(n_, "matrix dimension");
    }
    const auto t0 = std::chrono::steady_clock::now();
    // Column equilibration Z D (D = diag of powers of two) while copying into the LU storage:
    // the only n x n allocation. The singularity test below is then invariant to column scaling
    // (e.g. the eta^2 ratio between the J and M blocks of PMCHWT).
    scales_ = column_scales(Z);
    if (n_ == 0) {
        rcond_ = 1;
        log_backend_once(backend_);
        return;
    }
    lu_.resize(n_, n_);
    for (Index j = 0; j < n_; ++j) {
        lu_.col(j) = Z.col(j) * scales_(j);
    }
    // 1-norm of Z D for the rcond estimate (the quantity of LAPACK zgecon and Eigen's
    // PartialPivLU::rcond()).
    const Real anorm = lu_.cwiseAbs().colwise().sum().maxCoeff();

    if (backend_ == LuBackend::Lapack) {
#ifdef SPECKLEBEM_HAVE_LAPACK
        const int n = static_cast<int>(n_);
        lapack_pivots_.resize(static_cast<std::size_t>(n_));
        int info = 0;
        zgetrf_(&n, &n, lu_.data(), &n, lapack_pivots_.data(), &info);
        if (info < 0) {
            throw std::logic_error("dense LU: zgetrf reported an illegal argument, info = " +
                                   std::to_string(info));
        }
        if (info > 0) {
            throw std::runtime_error("dense LU: matrix is singular (zero pivot in column " +
                                     std::to_string(info - 1) + ")");
        }
#endif
    } else {
        const Eigen::PartialPivLU<Eigen::Ref<MatrixXc>> lu(lu_);  // factorises lu_ in place
        eigen_perm_ = lu.permutationP();
    }

    // Singularity criterion: an exactly zero (or non-finite) pivot of U, or a reciprocal 1-norm
    // condition number estimate of the equilibrated matrix below machine epsilon.
    for (Index i = 0; i < n_; ++i) {
        const Complex pivot = lu_(i, i);
        if (pivot == Complex(0, 0) || !std::isfinite(pivot.real()) ||
            !std::isfinite(pivot.imag())) {
            throw std::runtime_error(
                "dense LU: matrix is singular (zero or non-finite pivot in "
                "column " +
                std::to_string(i) + ")");
        }
    }
    // rcond = 1 / (|Z D|_1 |(Z D)^{-1}|_1) with the Hager/Higham lower-bound estimate of the
    // inverse norm (the algorithm of LAPACK zlacn2, used by zgecon and by Eigen's
    // PartialPivLU::rcond()), driven by the LU solves of the backend (zgetrs / Eigen's blocked
    // triangular solves). zgecon itself is not used: its zlatrs solves are level-2 code calling
    // BLAS-1/2 routines, 3-6 s on a 2N = 11440 PML matrix (63 s under heavy load) against
    // 1.5 s for this loop on zgetrs (benchmarks/results/dense_lu.md).
    const Real inv_norm = inverse_norm1_estimate();
    rcond_ = (anorm > 0 && inv_norm > 0) ? (Real{1} / anorm) / inv_norm : Real{0};
    if (!(rcond_ >= std::numeric_limits<Real>::epsilon())) {
        std::ostringstream msg;
        msg << "dense LU: matrix is numerically singular (equilibrated rcond = " << std::scientific
            << rcond_ << " < machine epsilon)";
        throw std::runtime_error(msg.str());
    }
    factor_seconds_ = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    log_backend_once(backend_);
    if (ill_conditioned()) {
        SBEM_WARN(
            "dense LU: ill-conditioned system (n = {}, equilibrated rcond = {:.3e} < {:.0e}); up "
            "to {:.0f} digits may be lost, the residual of each solve is in DirectSolveInfo",
            n_, rcond_, kIllConditionedRcond, -std::log10(rcond_));
    }
    SBEM_DEBUG("dense LU: n = {}, backend {}, rcond = {:.3e}, factorisation {:.3f} s", n_,
               to_string(backend_), rcond_, factor_seconds_);
}

DirectSolveInfo DenseLu::factor_info() const {
    DirectSolveInfo info;
    info.rcond = rcond_;
    info.ill_conditioned = ill_conditioned();
    info.backend = backend_;
    return info;
}

void DenseLu::lu_solve(VectorXc& x, bool adjoint) const {
    if (backend_ == LuBackend::Lapack) {
#ifdef SPECKLEBEM_HAVE_LAPACK
        lapack_getrs(lu_, lapack_pivots_, x.data(), n_, 1, adjoint ? 'C' : 'N');
#endif
    } else if (!adjoint) {
        // Eigen's PartialPivLU::solve: x = U^{-1} L^{-1} P b.
        x = eigen_perm_ * x;
        lu_.triangularView<Eigen::UnitLower>().solveInPlace(x);
        lu_.triangularView<Eigen::Upper>().solveInPlace(x);
    } else {
        // (P^T L U)^H x = b: x = P^T L^{-H} U^{-H} b (Eigen's _solve_impl_transposed).
        lu_.triangularView<Eigen::Upper>().adjoint().solveInPlace(x);
        lu_.triangularView<Eigen::UnitLower>().adjoint().solveInPlace(x);
        x = eigen_perm_.transpose() * x;
    }
}

Real DenseLu::inverse_norm1_estimate() const {
    // Hager (1984) / Higham (1988) estimate of |B|_1, B = (Z D)^{-1}, from products with B and
    // B^H: LAPACK zlacn2 (complex version, at most 5 iterations, alternating-sign check).
    constexpr int kMaxIterations = 5;
    const Index n = n_;
    const Real safe_min = std::numeric_limits<Real>::min();
    const auto complex_sign = [safe_min](VectorXc& v) {
        for (Index i = 0; i < v.size(); ++i) {
            const Real a = std::abs(v(i));
            v(i) = a > safe_min ? v(i) / a : Complex(1, 0);
        }
    };
    VectorXc x = VectorXc::Constant(n, Complex(Real{1} / static_cast<Real>(n), 0));
    lu_solve(x, false);
    if (n == 1) {
        return std::abs(x(0));
    }
    Real est = x.cwiseAbs().sum();
    complex_sign(x);
    lu_solve(x, true);
    Index j = 0;
    (void)x.cwiseAbs().maxCoeff(&j);
    for (int iter = 2; iter <= kMaxIterations; ++iter) {
        x.setZero();
        x(j) = Complex(1, 0);
        lu_solve(x, false);  // column j of B: |B e_j|_1 is a lower bound
        const Real est_old = est;
        // zlacn2 takes the new value; keeping the larger one departs from it deliberately (both
        // are valid lower bounds of |B|_1).
        est = std::max(est_old, x.cwiseAbs().sum());
        if (!(est > est_old)) {
            break;
        }
        complex_sign(x);
        lu_solve(x, true);
        const Index j_last = j;
        (void)x.cwiseAbs().maxCoeff(&j);
        if (std::abs(x(j_last)) == std::abs(x(j))) {
            break;
        }
    }
    // Alternating-sign test vector x_i = (-1)^i (1 + i / (n - 1)) guards against the
    // estimator's worst cases.
    for (Index i = 0; i < n; ++i) {
        const Real mag = 1 + static_cast<Real>(i) / static_cast<Real>(n - 1);
        x(i) = Complex(i % 2 == 0 ? mag : -mag, 0);
    }
    lu_solve(x, false);
    const Real alt = 2 * x.cwiseAbs().sum() / (3 * static_cast<Real>(n));
    return std::max(est, alt);
}

void DenseLu::apply_inverse(VectorXc& x) const {
    lu_solve(x, false);
    x.array() *= scales_.array().cast<Complex>();
}

void DenseLu::apply_inverse(MatrixXc& X) const {
    if (X.cols() == 0) {
        return;
    }
    if (backend_ == LuBackend::Lapack) {
#ifdef SPECKLEBEM_HAVE_LAPACK
        lapack_getrs(lu_, lapack_pivots_, X.data(), n_, X.cols());
#endif
    } else {
        X = eigen_perm_ * X;
        lu_.triangularView<Eigen::UnitLower>().solveInPlace(X);
        lu_.triangularView<Eigen::Upper>().solveInPlace(X);
    }
    X = scales_.cast<Complex>().asDiagonal() * X;
}

VectorXc DenseLu::solve(const VectorXc& b, DirectSolveInfo* info) const {
    check_rhs(b, n_);
    DirectSolveInfo local;
    DirectSolveInfo& out = info ? *info : local;
    out = factor_info();
    if (n_ == 0) {
        return VectorXc();
    }
    const MatrixXc& A = *Z_;
    VectorXc x = b;
    apply_inverse(x);

    // Fixed-precision iterative refinement against the original Z (as LAPACK zgerfs): O(n^2) per
    // step. It brings the residual |b - Z x| / |b| of ill-conditioned random matrices from ~1e-12
    // down to ~1e-14 (n = 1000); a step is kept only if it reduces the residual.
    VectorXc r = b - A * x;
    Real r_norm = r.norm();
    for (int step = 0; step < kMaxRefinementSteps && r_norm > 0; ++step) {
        VectorXc dx = r;
        apply_inverse(dx);
        VectorXc x_new = x + dx;
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
    SBEM_DEBUG("dense LU solve: n = {}, relative residual = {:.3e}, {} refinement step(s)", n_,
               out.residual, out.refinement_steps);
    return x;
}

MatrixXc DenseLu::solve(const MatrixXc& B, DirectSolveInfo* info) const {
    check_rhs(B, n_);
    DirectSolveInfo local;
    DirectSolveInfo& out = info ? *info : local;
    out = factor_info();
    const Index m = B.cols();
    if (n_ == 0 || m == 0) {
        return MatrixXc(n_, m);
    }
    const MatrixXc& A = *Z_;
    MatrixXc X = B;
    apply_inverse(X);

    // Refinement as in solve(b), column by column: a column leaves the refinement at its first
    // step that does not reduce its residual. The active columns are refined as one block.
    MatrixXc R = B;
    R.noalias() -= A * X;
    VectorXr r_norm = R.colwise().norm().transpose();
    std::vector<Index> active;
    for (Index j = 0; j < m; ++j) {
        if (r_norm(j) > 0) {
            active.push_back(j);
        }
    }
    std::vector<int> steps(static_cast<std::size_t>(m), 0);
    for (int step = 0; step < kMaxRefinementSteps && !active.empty(); ++step) {
        const auto k = static_cast<Index>(active.size());
        MatrixXc X_new(n_, k);
        MatrixXc R_new(n_, k);
        for (Index c = 0; c < k; ++c) {
            R_new.col(c) = R.col(active[static_cast<std::size_t>(c)]);
        }
        apply_inverse(R_new);  // corrections
        for (Index c = 0; c < k; ++c) {
            const Index j = active[static_cast<std::size_t>(c)];
            X_new.col(c) = X.col(j) + R_new.col(c);
            R_new.col(c) = B.col(j);
        }
        R_new.noalias() -= A * X_new;
        std::vector<Index> still_active;
        for (Index c = 0; c < k; ++c) {
            const Index j = active[static_cast<std::size_t>(c)];
            const Real r_new_norm = R_new.col(c).norm();
            if (r_new_norm < r_norm(j)) {
                X.col(j) = X_new.col(c);
                R.col(j) = R_new.col(c);
                r_norm(j) = r_new_norm;
                ++steps[static_cast<std::size_t>(j)];
                if (r_new_norm > 0) {
                    still_active.push_back(j);
                }
            }
        }
        active.swap(still_active);
    }
    for (Index j = 0; j < m; ++j) {
        const Real b_norm = B.col(j).norm();
        out.residual = std::max(out.residual, b_norm > 0 ? r_norm(j) / b_norm : r_norm(j));
        out.refinement_steps = std::max(out.refinement_steps, steps[static_cast<std::size_t>(j)]);
    }
    SBEM_DEBUG(
        "dense LU solve: n = {}, {} right-hand sides, largest relative residual = {:.3e}, up to {} "
        "refinement step(s)",
        n_, m, out.residual, out.refinement_steps);
    return X;
}

VectorXc solve_direct(const op::DenseOperator& Z, const VectorXc& b, DirectSolveInfo* info) {
    const MatrixXc& A = Z.matrix();
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("solve_direct: matrix must be square (" +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                    ")");
    }
    check_rhs(b, A.rows());  // before the O(n^3) factorisation
    DirectSolveInfo local;
    DirectSolveInfo& out = info ? *info : local;
    out = DirectSolveInfo{};
    if (A.rows() == 0) {
        const DenseLu lu(A);
        return lu.solve(b, &out);
    }
    const ScopedTimer timer("solve_direct (dense LU, n = " + std::to_string(A.rows()) + ")");
    const DenseLu lu(A);
    VectorXc x = lu.solve(b, &out);
    SBEM_DEBUG(
        "solve_direct: n = {}, backend {}, rcond = {:.3e}, relative residual = {:.3e}, {} "
        "refinement",
        A.rows(), to_string(out.backend), out.rcond, out.residual, out.refinement_steps);
    return x;
}

}  // namespace specklebem::solver
