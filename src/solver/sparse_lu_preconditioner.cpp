/// @file sparse_lu_preconditioner.cpp
/// solver::SparseLuPreconditioner (WP-SC1, ADR 0012).
#include "specklebem/solver/sparse_lu_preconditioner.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/system_memory.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseLU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

namespace specklebem::solver {

namespace {

using ColMatrix = Eigen::SparseMatrix<Complex, Eigen::ColMajor, Index>;
using EigenLu = Eigen::SparseLU<ColMatrix, Eigen::COLAMDOrdering<Index>>;

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/// Eigen's SparseLU with access to two protected settings: the fill factor of the initial
/// reservation of the factor arrays (Eigen's default 20 reserves 20 nnz(A) entries for each of
/// L's and U's value arrays up front: 64 GB of values per 10^8 input entries), and the release of
/// Eigen's permuted copy of the input after the factorisation (only its dimensions are used by
/// the solve).
///
/// The protected members used here (m_info, m_factorizationIsOk, m_perfv.fillfactor, m_mat) are
/// not part of Eigen's public API; checked against Eigen 3.4.0 (the fetched version) and 5.0.1
/// (the MSYS2 package).
class Lu final : public EigenLu {
public:
    // Eigen leaves m_info uninitialised until a factorisation sets it (and does not set it when
    // its initial memory reservation fails).
    Lu() {
        this->m_info = Eigen::InvalidInput;
        this->m_factorizationIsOk = false;
    }
    [[nodiscard]] bool factorization_ok() const { return this->m_factorizationIsOk; }
    void set_initial_fill(Index f) { this->m_perfv.fillfactor = f; }
    void release_input_copy() {
        ColMatrix empty(this->m_mat.rows(), this->m_mat.cols());
        this->m_mat.swap(empty);
    }
};

/// Sets the calling thread's OpenMP thread count to 1 for its lifetime (the dense kernels inside
/// Eigen's SparseLU and an OpenMP-built BLAS follow it): bitwise thread-count independence.
/// Only the OpenMP setting is clamped: a caller's Eigen::setNbThreads(n > 1) takes precedence
/// over omp_get_max_threads() in Eigen's dense products and bypasses the clamp.
class SingleThreadScope {
public:
    SingleThreadScope() {
#ifdef SPECKLEBEM_HAVE_OPENMP
        saved_ = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
    }
    ~SingleThreadScope() {
#ifdef SPECKLEBEM_HAVE_OPENMP
        omp_set_num_threads(saved_);
#endif
    }
    SingleThreadScope(const SingleThreadScope&) = delete;
    SingleThreadScope& operator=(const SingleThreadScope&) = delete;

#ifdef SPECKLEBEM_HAVE_OPENMP
private:
    int saved_ = 1;  // the thread count restored by the destructor
#endif
};

std::string megabytes(std::size_t bytes) {
    char buf[64];
    if (bytes == std::numeric_limits<std::size_t>::max())
        return "unlimited";
    std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / 1e6);
    return buf;
}

}  // namespace

struct SparseLuPreconditioner::Impl {
    Lu lu;
};

SparseLuPreconditioner::SparseLuPreconditioner(const op::SparseOperator::Matrix& A,
                                               std::optional<std::size_t> budget_bytes,
                                               Real expected_fill)
    : impl_(std::make_unique<Impl>()), expected_fill_(expected_fill) {
    const char* who = "SparseLuPreconditioner";
    if (A.rows() == 0 || A.cols() == 0)
        throw std::invalid_argument(std::string(who) + ": empty matrix");
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(std::string(who) + ": the matrix is " +
                                    std::to_string(A.rows()) + " x " + std::to_string(A.cols()) +
                                    ", expected a square matrix");
    }
    if (!(std::isfinite(expected_fill) && expected_fill > 0.0))
        throw std::invalid_argument(std::string(who) + ": expected_fill must be finite and > 0");
    n_ = A.rows();
    nnz_a_ = A.nonZeros();
    for (Index j = 0; j < A.outerSize(); ++j) {
        for (op::SparseOperator::Matrix::InnerIterator it(A, j); it; ++it) {
            if (!std::isfinite(it.value().real()) || !std::isfinite(it.value().imag())) {
                throw std::invalid_argument(std::string(who) + ": non-finite entry (" +
                                            std::to_string(it.row()) + ", " +
                                            std::to_string(it.col()) + ")");
            }
        }
    }
    if (budget_bytes) {
        budget_ = *budget_bytes;
    } else {
        const std::size_t avail = core::available_memory_bytes();
        if (avail == 0) {
            SBEM_WARN("{}: the available memory is unknown, the factor memory is not limited", who);
            budget_ = std::numeric_limits<std::size_t>::max();
        } else {
            budget_ = avail / 2;
        }
    }
    const Real est = expected_fill * static_cast<Real>(nnz_a_) * static_cast<Real>(kBytesPerEntry);
    estimate_ = est >= static_cast<Real>(std::numeric_limits<std::size_t>::max())
                    ? std::numeric_limits<std::size_t>::max()
                    : static_cast<std::size_t>(std::ceil(est));
    if (estimate_ > budget_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3g", expected_fill);
        throw std::runtime_error(std::string(who) + ": the estimated factor memory " +
                                 megabytes(estimate_) + " (expected fill " + buf + " x " +
                                 std::to_string(nnz_a_) + " entries x " +
                                 std::to_string(kBytesPerEntry) + " B) exceeds the budget " +
                                 megabytes(budget_) + "; not factorised");
    }

    Lu& lu = impl_->lu;
    const SingleThreadScope single;
    const auto t0 = std::chrono::steady_clock::now();
    {
        ColMatrix C = A;  // RowMajor -> ColMajor (Eigen's SparseLU needs column storage)
        C.makeCompressed();
        lu.set_initial_fill(static_cast<Index>(std::clamp(std::ceil(expected_fill), 1.0, 20.0)));
        lu.analyzePattern(C);
        analysis_s_ = seconds_since(t0);
        const auto t1 = std::chrono::steady_clock::now();
        lu.factorize(C);
        factor_s_ = seconds_since(t1);
    }
    lu.release_input_copy();
    if (!lu.factorization_ok() || lu.info() != Eigen::Success) {
        std::string msg = lu.lastErrorMessage();
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == ' ')) msg.pop_back();
        throw std::runtime_error(
            std::string(who) + ": the sparse LU failed (" +
            (lu.info() == Eigen::NumericalIssue ? "numerical issue" : "memory or input") +
            "): " + msg);
    }
    nnz_l_ = lu.nnzL();
    nnz_u_ = lu.nnzU();
    if (memory_bytes() > budget_) {
        throw std::runtime_error(
            std::string(who) + ": the factor memory " + megabytes(memory_bytes()) +
            " (nnz(L + U) " + std::to_string(factor_nonzeros()) + ") exceeds the budget " +
            megabytes(budget_) + " (estimate was " + megabytes(estimate_) + "); factors released");
    }
}

SparseLuPreconditioner::~SparseLuPreconditioner() = default;
SparseLuPreconditioner::SparseLuPreconditioner(SparseLuPreconditioner&&) noexcept = default;
SparseLuPreconditioner& SparseLuPreconditioner::operator=(SparseLuPreconditioner&&) noexcept =
    default;

void SparseLuPreconditioner::apply(const VectorXc& r, VectorXc& z) const {
    if (r.size() != n_) {
        throw std::invalid_argument("SparseLuPreconditioner::apply: vector size " +
                                    std::to_string(r.size()) + " != " + std::to_string(n_));
    }
    // A separate result vector: r and z may be the same object.
    const SingleThreadScope single;
    VectorXc x = impl_->lu.solve(r);
    z.swap(x);
}

Real SparseLuPreconditioner::fill_factor() const {
    return nnz_a_ > 0 ? static_cast<Real>(factor_nonzeros()) / static_cast<Real>(nnz_a_) : 0.0;
}

std::size_t SparseLuPreconditioner::memory_bytes() const {
    // Factor entries (value + index), the row and column permutations and the column pointer
    // arrays of L and U (8 bytes per column each).
    return static_cast<std::size_t>(factor_nonzeros()) * kBytesPerEntry +
           4 * static_cast<std::size_t>(n_) * sizeof(Index);
}

std::string SparseLuPreconditioner::describe() const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "sparse LU: %lld unknowns, nnz(A) %lld, nnz(L + U) %lld (L %lld, U %lld), fill "
                  "%.2f, %s (estimate %s, budget %s), setup %.2f s (analysis %.2f s, "
                  "factorisation %.2f s)",
                  static_cast<long long>(n_), static_cast<long long>(nnz_a_),
                  static_cast<long long>(factor_nonzeros()), static_cast<long long>(nnz_l_),
                  static_cast<long long>(nnz_u_), fill_factor(), megabytes(memory_bytes()).c_str(),
                  megabytes(estimate_).c_str(), megabytes(budget_).c_str(), setup_seconds(),
                  analysis_s_, factor_s_);
    return buf;
}

}  // namespace specklebem::solver
