#include "specklebem/solver/gmres.hpp"

#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

const char* side_name(PreconditionerSide side) {
    return side == PreconditionerSide::Left ? "left" : "right";
}

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
    if (!(norm0 > 0) || !std::isfinite(norm0)) {
        throw std::runtime_error("gmres: the preconditioner maps the right-hand side to " +
                                 std::string(norm0 > 0 ? "a non-finite vector" : "zero"));
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
    std::vector<VectorXc> V;
    std::vector<VectorXc> H;
    std::vector<Givens> rot;
    VectorXc g;
    VectorXc w;
    res.residual_history.push_back(1.0);
    int total = 0;
    bool stop = false;

    while (true) {
        V.resize(1);
        V[0] = r / beta;
        H.clear();
        rot.clear();
        // (A scalar store into the freshly allocated g trips a GCC -Wnull-dereference false
        // positive.)
        g = VectorXc::Unit(m + 1, 0) * beta;
        int k = 0;  // Krylov dimension of this cycle (columns of H)

        while (k < m && total < p.max_iter) {
            op.apply(V[static_cast<std::size_t>(k)], w);
            const Real w_norm0 = w.norm();
            VectorXc h = VectorXc::Zero(k + 2);
            // Modified Gram-Schmidt, plus a second pass if |w| dropped by more than 0.7.
            for (int pass = 0; pass < 2; ++pass) {
                const Real before = w.norm();
                for (int j = 0; j <= k; ++j) {
                    const VectorXc& vj = V[static_cast<std::size_t>(j)];
                    const Complex hj = vj.dot(w);  // vj^H w
                    h(j) += hj;
                    w -= hj * vj;
                }
                if (w.norm() >= kReorthFactor * before)
                    break;
            }
            const Real h_next = w.norm();
            h(k + 1) = h_next;
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
            V.push_back(w / h_next);
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
            VectorXc u = VectorXc::Zero(n);
            for (int j = 0; j < k; ++j) {
                u += y(j) * V[static_cast<std::size_t>(j)];
            }
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
        SBEM_DEBUG("gmres: restart after {} iterations, recomputed relative residual {:.3e}", total,
                   beta / norm0);
        if (beta == 0) {  // exact solution found at the restart
            res.residual_history.back() = 0;
            break;
        }
    }

    res.iterations = total;
    res.converged = res.residual_history.back() <= p.tolerance;
    VectorXc ax;
    A.apply(res.x, ax);
    res.true_relative_residual = (b - ax).norm() / b_norm;
    res.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!res.converged) {
        SBEM_WARN(
            "gmres: not converged after {} iterations: relative residual {:.3e} > tol "
            "{:.1e} (true residual {:.3e})",
            res.iterations, res.residual_history.back(), p.tolerance, res.true_relative_residual);
    } else if (p.verbose) {
        SBEM_INFO(
            "gmres: converged in {} iterations, {:.2f} s: relative residual {:.3e} "
            "({}), true residual {:.3e}",
            res.iterations, res.wall_seconds, res.residual_history.back(),
            p.side == PreconditionerSide::Left ? "preconditioned" : "true",
            res.true_relative_residual);
    }
    return res;
}

}  // namespace specklebem::solver
