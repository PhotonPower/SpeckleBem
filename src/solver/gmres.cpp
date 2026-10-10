#include "specklebem/solver/gmres.hpp"

#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace specklebem::solver {

namespace {

/// Re-orthogonalise when |w| drops below this fraction of its norm before the MGS pass.
constexpr Real kReorthFactor = 0.7;
/// Breakdown threshold relative to |w| before orthogonalisation.
constexpr Real kBreakdownFactor = 100 * std::numeric_limits<Real>::epsilon();
/// Progress log interval (iterations) when verbose.
constexpr int kLogInterval = 50;

bool all_finite(const VectorXc& v) {
    for (Index i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v(i).real()) || !std::isfinite(v(i).imag()))
            return false;
    }
    return true;
}

void check_arguments(const op::LinearOperator& A, const VectorXc& b, const GmresParams& p) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres: operator is not square (" + std::to_string(A.rows()) +
                                    " x " + std::to_string(A.cols()) + ")");
    }
    if (b.size() != A.rows()) {
        throw std::invalid_argument("gmres: right-hand side size " + std::to_string(b.size()) +
                                    " != operator rows " + std::to_string(A.rows()));
    }
    if (!all_finite(b)) {
        throw std::invalid_argument("gmres: right-hand side has non-finite entries");
    }
    if (!(p.tolerance > 0) || !std::isfinite(p.tolerance)) {
        throw std::invalid_argument("gmres: tolerance must be finite and > 0");
    }
    if (p.max_iter < 1) {
        throw std::invalid_argument("gmres: max_iter must be >= 1");
    }
    if (p.restart < 0) {
        throw std::invalid_argument("gmres: restart must be >= 0");
    }
}

/// Complex Givens rotation G = [c s; -conj(s) c] (c real) with G [a; b] = [r; 0].
struct Givens {
    Real c = 1;
    Complex s{0, 0};

    /// Builds the rotation for (a, b) and returns r (|r| = hypot(|a|, |b|)).
    Complex make(Complex a, Complex b) {
        const Real abs_a = std::abs(a);
        const Real abs_b = std::abs(b);
        if (abs_b == 0) {
            c = 1;
            s = Complex(0, 0);
            return a;
        }
        const Real rho = std::hypot(abs_a, abs_b);
        if (abs_a == 0) {
            c = 0;
            s = std::conj(b) / abs_b;
            return rho;
        }
        const Complex phase = a / abs_a;
        c = abs_a / rho;
        s = phase * std::conj(b) / rho;
        return phase * rho;
    }

    void apply(Complex& x, Complex& y) const {
        const Complex t = c * x + s * y;
        y = -std::conj(s) * x + c * y;
        x = t;
    }
};

/// One preconditioned operator application w = M^{-1} A v (left) or w = A M^{-1} v (right).
class PreconditionedOperator {
public:
    PreconditionedOperator(const op::LinearOperator& A, const Preconditioner& M,
                           PreconditionerSide side)
        : A_(A), M_(M), side_(side) {}

    void apply(const VectorXc& v, VectorXc& w) {
        if (side_ == PreconditionerSide::Left) {
            A_.apply(v, tmp_);
            M_.apply(tmp_, w);
        } else {
            M_.apply(v, tmp_);
            A_.apply(tmp_, w);
        }
    }

    /// Residual of the GMRES system for the iterate x of A x = b: M^{-1}(b - A x) (left) or
    /// b - A x (right).
    void residual(const VectorXc& b, const VectorXc& x, VectorXc& r) {
        A_.apply(x, tmp_);
        tmp_ = b - tmp_;
        if (side_ == PreconditionerSide::Left) {
            M_.apply(tmp_, r);
        } else {
            r = tmp_;
        }
    }

private:
    const op::LinearOperator& A_;
    const Preconditioner& M_;
    PreconditionerSide side_;
    VectorXc tmp_;
};

/// Throws if a norm computed from an operator or preconditioner output is not finite.
void check_finite_norm(Real norm, int iteration) {
    if (!std::isfinite(norm)) {
        throw std::runtime_error(
            "gmres: operator or preconditioner produced non-finite values at iteration " +
            std::to_string(iteration));
    }
}

const char* side_name(PreconditionerSide side) {
    return side == PreconditionerSide::Left ? "left" : "right";
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

/// Rows per block of the parallel CGS2 products. The block partition (and so the order of the
/// partial sums) depends on n only, never on the thread count.
constexpr Index kRowBlock = 4096;
/// Columns per panel of the Krylov basis.
constexpr Index kPanelCols = 64;

/// Krylov basis V (n x size()), stored as contiguous column-major panels of equal width so
/// that it grows without copying. Panels are kept across restarts.
class KrylovBasis {
public:
    KrylovBasis(Index n, Index max_cols)
        : n_(n), panel_cols_(std::max<Index>(1, std::min(kPanelCols, max_cols))) {}

    [[nodiscard]] Index size() const { return size_; }
    void clear() { size_ = 0; }

    void append(const VectorXc& v) {
        if (size_ == static_cast<Index>(panels_.size()) * panel_cols_)
            panels_.emplace_back(n_, panel_cols_);
        col(size_) = v;
        ++size_;
    }

    /// h(0 .. k-1) = V(:, 0 .. k-1)^H w: per row block a vector of partial dot products, added
    /// in block order.
    void project(const VectorXc& w, Index k, VectorXc& h) {
        const Index blocks = (n_ + kRowBlock - 1) / kRowBlock;
        partial_.resize(k, blocks);
#pragma omp parallel for schedule(static) if (blocks > 1)
        for (Index b = 0; b < blocks; ++b) {
            const Index r0 = b * kRowBlock;
            const Index len = std::min(kRowBlock, n_ - r0);
            const auto ws = w.segment(r0, len);
            for (Index j = 0; j < k; ++j) partial_(j, b) = col(j).segment(r0, len).dot(ws);
        }
        h = partial_.col(0);
        for (Index b = 1; b < blocks; ++b) h += partial_.col(b);
    }

    /// w -= V(:, 0 .. k-1) h(0 .. k-1), per row block (four columns per sweep over the block).
    void subtract(VectorXc& w, Index k, const VectorXc& h) {
        const Index blocks = (n_ + kRowBlock - 1) / kRowBlock;
#pragma omp parallel for schedule(static) if (blocks > 1)
        for (Index b = 0; b < blocks; ++b) {
            const Index r0 = b * kRowBlock;
            const Index len = std::min(kRowBlock, n_ - r0);
            auto ws = w.segment(r0, len);
            Index j = 0;
            for (; j + 4 <= k; j += 4) {
                ws -= h(j) * col(j).segment(r0, len) + h(j + 1) * col(j + 1).segment(r0, len) +
                      h(j + 2) * col(j + 2).segment(r0, len) +
                      h(j + 3) * col(j + 3).segment(r0, len);
            }
            for (; j < k; ++j) ws -= h(j) * col(j).segment(r0, len);
        }
    }

    /// One modified Gram-Schmidt pass of w against V(:, 0 .. k-1), accumulated into h.
    void mgs_pass(VectorXc& w, Index k, VectorXc& h) {
        for (Index j = 0; j < k; ++j) {
            const Complex hj = col(j).dot(w);  // v_j^H w
            h(j) += hj;
            w -= hj * col(j);
        }
    }

    [[nodiscard]] MatrixXc to_matrix() const {
        MatrixXc out(n_, size_);
        for (Index j = 0; j < size_; ++j) out.col(j) = col(j);
        return out;
    }

private:
    [[nodiscard]] MatrixXc::ColXpr col(Index j) {
        return panels_[static_cast<std::size_t>(j / panel_cols_)].col(j % panel_cols_);
    }
    [[nodiscard]] MatrixXc::ConstColXpr col(Index j) const {
        return panels_[static_cast<std::size_t>(j / panel_cols_)].col(j % panel_cols_);
    }

    Index n_;
    Index panel_cols_;
    Index size_ = 0;
    std::vector<MatrixXc> panels_;
    MatrixXc partial_;
};

}  // namespace

GmresResult gmres(const op::LinearOperator& A, const VectorXc& b, const Preconditioner& M,
                  const GmresParams& p, const IterationCallback& cb) {
    check_arguments(A, b, p);
    const auto t0 = std::chrono::steady_clock::now();
    const Index n = b.size();

    GmresResult res;
    res.x = VectorXc::Zero(n);
    const Real b_norm = b.norm();
    if (b_norm == 0) {
        res.residual_history = {0.0};
        res.converged = true;
        res.true_relative_residual = 0;
        return res;
    }

    PreconditionedOperator op(A, M, p.side);
    VectorXc r;
    if (p.side == PreconditionerSide::Left) {
        M.apply(b, r);
    } else {
        r = b;
    }
    Real beta = r.norm();
    const Real norm0 = beta;
    check_finite_norm(norm0, 0);
    if (!(norm0 > 0)) {
        throw std::runtime_error("gmres: the preconditioner maps the right-hand side to zero");
    }

    const int m = p.restart > 0 ? std::min(p.restart, p.max_iter) : p.max_iter;
    if (p.verbose) {
        SBEM_INFO("gmres: n = {}, {} preconditioner '{}', tol = {:.1e}, max_iter = {}, {}", n,
                  side_name(p.side), M.name(), p.tolerance, p.max_iter,
                  p.restart > 0 ? "restart = " + std::to_string(p.restart)
                                : std::string("full (no restart)"));
    }

    // Krylov basis V (grows on demand), Hessenberg columns H[k] (rotated in place, size k + 2),
    // rotations and the rotated right-hand side g of the least-squares problem.
    KrylovBasis V(n, static_cast<Index>(m) + 1);
    std::vector<VectorXc> H;
    std::vector<Givens> rot;
    VectorXc g;
    VectorXc w;
    VectorXc h_pass;
    VectorXc v;
    res.residual_history.push_back(1.0);
    int total = 0;
    bool stop = false;
    const bool cgs = p.orthogonalization == Orthogonalization::CGS2;
    using Clock = std::chrono::steady_clock;

    while (true) {
        // v: the newest basis vector v_k, also stored as a plain vector for the operator.
        v = r / beta;
        V.clear();
        V.append(v);
        H.clear();
        rot.clear();
        // (A scalar store into the freshly allocated g trips a GCC -Wnull-dereference false
        // positive.)
        g = VectorXc::Unit(m + 1, 0) * beta;
        int k = 0;  // Krylov dimension of this cycle (columns of H)

        while (k < m && total < p.max_iter) {
            op.apply(v, w);
            const auto t_orth = Clock::now();
            const Real w_norm0 = w.norm();
            check_finite_norm(w_norm0, total + 1);
            VectorXc h = VectorXc::Zero(k + 2);
            // Gram-Schmidt against v_0 .. v_k, plus a second pass if |w| dropped by more than
            // 0.7 (or always). CGS2 accumulates the Hessenberg column as h = h_1 + h_2.
            const Index kv = static_cast<Index>(k) + 1;
            Real h_next = w_norm0;
            for (int pass = 0; pass < 2; ++pass) {
                const Real before = h_next;
                if (cgs) {
                    V.project(w, kv, h_pass);
                    V.subtract(w, kv, h_pass);
                    h.head(kv) += h_pass;
                } else {
                    V.mgs_pass(w, kv, h);
                }
                h_next = w.norm();
                if (pass == 1)
                    ++res.reorthogonalizations;
                else if (!p.always_reorthogonalize && h_next >= kReorthFactor * before)
                    break;
            }
            h(k + 1) = h_next;
            res.orthogonalization_seconds += seconds_since(t_orth);
            for (int j = 0; j < k; ++j) {
                rot[static_cast<std::size_t>(j)].apply(h(j), h(j + 1));
            }
            Givens gk;
            const Complex r_kk = gk.make(h(k), h(k + 1));
            ++total;
            if (std::abs(r_kk) <= kBreakdownFactor * w_norm0) {
                // A (or M^{-1}) is singular on the Krylov space: discard the column.
                SBEM_WARN(
                    "gmres: singular Hessenberg matrix at iteration {} (operator or "
                    "preconditioner singular on the Krylov space); stopping",
                    total);
                res.residual_history.push_back(res.residual_history.back());
                if (cb)
                    cb(total, res.residual_history.back());
                stop = true;
                break;
            }
            h(k) = r_kk;
            h(k + 1) = 0;
            gk.apply(g(k), g(k + 1));
            rot.push_back(gk);
            H.push_back(std::move(h));
            ++k;

            const Real rel = std::abs(g(k)) / norm0;
            res.residual_history.push_back(rel);
            if (cb)
                cb(total, rel);
            if (p.verbose && total % kLogInterval == 0) {
                SBEM_INFO("gmres: iteration {}: relative residual {:.3e}", total, rel);
            }
            if (rel <= p.tolerance) {
                stop = true;
                break;
            }
            if (h_next <= kBreakdownFactor * w_norm0) {
                SBEM_DEBUG("gmres: happy breakdown at iteration {} (h = {:.3e}, |w| = {:.3e})",
                           total, h_next, w_norm0);
                stop = true;
                break;
            }
            const auto t_norm = Clock::now();
            v = w / h_next;
            V.append(v);
            res.orthogonalization_seconds += seconds_since(t_norm);
        }

        // Solve the k x k upper-triangular system R y = g(0..k-1) and update the iterate.
        if (k > 0) {
            VectorXc y = g.head(k);
            for (int i = k - 1; i >= 0; --i) {
                for (int j = i + 1; j < k; ++j) {
                    y(i) -= H[static_cast<std::size_t>(j)](i) * y(j);
                }
                y(i) /= H[static_cast<std::size_t>(i)](i);
            }
            // u = V(:, 0 .. k-1) y, as u -= V (-y) with the parallel row-block product.
            VectorXc u = VectorXc::Zero(n);
            V.subtract(u, k, -y);
            if (p.side == PreconditionerSide::Left) {
                res.x += u;
            } else {
                VectorXc z;
                M.apply(u, z);
                res.x += z;
            }
        }
        if (stop || total >= p.max_iter)
            break;
        // Restart: recompute the residual of the GMRES system explicitly.
        op.residual(b, res.x, r);
        beta = r.norm();
        check_finite_norm(beta, total);
        SBEM_DEBUG("gmres: restart after {} iterations, recomputed relative residual {:.3e}", total,
                   beta / norm0);
        if (beta == 0) {
            // Exact solution at the restart: stop. residual_history keeps the Arnoldi estimate
            // already passed to the callback (see gmres.hpp).
            SBEM_DEBUG("gmres: exact solution at the restart after {} iterations", total);
            break;
        }
    }

    res.iterations = total;
    res.converged = res.residual_history.back() <= p.tolerance;
    if (p.keep_basis)
        res.basis = V.to_matrix();
    VectorXc ax;
    A.apply(res.x, ax);
    res.true_relative_residual = (b - ax).norm() / b_norm;
    res.wall_seconds = seconds_since(t0);

    if (!res.converged) {
        SBEM_WARN(
            "gmres: not converged after {} iterations: relative residual {:.3e} > tol "
            "{:.1e} (true residual {:.3e})",
            res.iterations, res.residual_history.back(), p.tolerance, res.true_relative_residual);
    } else if (p.verbose) {
        SBEM_INFO(
            "gmres: converged in {} iterations, {:.2f} s ({} orthogonalisation {:.2f} s, {} "
            "re-orthogonalisations): relative residual {:.3e} ({}), true residual {:.3e}",
            res.iterations, res.wall_seconds, cgs ? "CGS2" : "MGS", res.orthogonalization_seconds,
            res.reorthogonalizations, res.residual_history.back(),
            p.side == PreconditionerSide::Left ? "preconditioned" : "true",
            res.true_relative_residual);
    }
    return res;
}

}  // namespace specklebem::solver
