#pragma once
/// @file sparse_lu_preconditioner.hpp
/// Exact sparse LU of a sparse matrix as a preconditioner (WP-SC1, ADR 0012 decision 1): the
/// near-field-inverse ceiling M^{-1} = Z_n^{-1} of every near-field-based preconditioner. A
/// measurement tool, never a default (make_near_field_lu_preconditioner in simulation.hpp builds
/// it from the near parts of an MLFMM operator).
#include "specklebem/core/types.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace specklebem::solver {

/// M = A for a sparse n x n matrix A: z = A^{-1} r by one forward / back substitution with the
/// LU factors of Eigen::SparseLU (supernodal, COLAMD column ordering, partial pivoting with
/// Eigen's default threshold 1, complex double, 64-bit indices). The RowMajor input is converted
/// to ColMajor (Eigen's SparseLU needs it); the conversion and Eigen's own copy of the matrix are
/// released after the factorisation, only the factors and the permutations are kept.
///
/// Memory budget. Eigen does not expose the fill-in before the numeric factorisation (the
/// symbolic and the numeric factorisation are interleaved), so the budget is checked twice:
///  (a) before the factorisation against the estimate expected_fill x nnz(A) x
///      kBytesPerEntry (kBytesPerEntry = 24: a 16-byte value and an 8-byte index per entry);
///      the caller passes the fill factor nnz(L + U) / nnz(A) measured on a smaller case of the
///      same family (default 1: the input's own size, i.e. no fill);
///  (b) after the factorisation against the actual factor memory memory_bytes(), throwing then
///      (the factors are freed with the object), so the failure is explicit even when the
///      estimate was optimistic.
/// Both checks count the factor entries actually used (nnz(L + U) x kBytesPerEntry + the
/// permutations and column pointers), not the capacity of Eigen's factor arrays. Eigen's memInit
/// reserves about ceil(expected_fill) x nnz(A) x 42 B up front (the value arrays lusup and ucol,
/// 16 B per entry each, and the index arrays usub, 8 B, and lsub, a quarter of that count x 8 B),
/// grows an array by 1.5x (old and new copy coexisting) whenever it overflows, and never
/// shrinks it; the kept factors are views into these arrays, so the memory held after the
/// factorisation is the capacity, not memory_bytes(). On Windows the whole reservation counts
/// against the commit limit when it is allocated, touched or not.
/// The budget therefore bounds the factors that are kept, not the process peak: the input, its
/// ColMajor copy, Eigen's permuted copy, the over-reserved and growing factor arrays and the
/// work arrays coexist during the factorisation (measured on the 16 um rough-Ag near field,
/// factors 66.3 GB: a peak of about 1.3x the factors plus the input copies, 84.7 GB working set
/// in the run refused by the post-check under a 40 GB budget and 86.4 GB in the run with a
/// 75 GB budget). A factorisation can thus exhaust the machine before the post-check fires; size
/// the budget and expected_fill with that peak in mind.
///
/// Deterministic: Eigen's SparseLU is a sequential algorithm, but its supernodal kernels call
/// Eigen's dense products (and, with EIGEN_USE_BLAS, the BLAS), which split over the OpenMP
/// threads and then sum in a thread-count dependent order (measured: the solve differed between
/// 1 and 24 threads). The factorisation and apply() therefore run with the calling thread's
/// OpenMP thread count set to 1 (restored afterwards; an OpenMP-built BLAS such as the MSYS2 /
/// usual Linux OpenBLAS follows it, a pthreads-built BLAS needs OPENBLAS_NUM_THREADS=1), so setup
/// and apply() are bitwise identical for any number of OpenMP threads (tested). Call them from
/// outside parallel regions (as GMRES does).
class SparseLuPreconditioner final : public Preconditioner {
public:
    /// Bytes per stored entry in the estimate and in memory_bytes(): a complex<double> value
    /// and one 64-bit index (U stores one row index per entry; for the supernodal L this is an
    /// upper bound, L stores one row-index list per supernode).
    static constexpr std::size_t kBytesPerEntry = 24;

    /// Converts A to ColMajor, runs Eigen's analyzePattern (COLAMD) and factorize (all timed).
    /// @param A square, non-empty, finite entries (not necessarily compressed).
    /// @param budget_bytes limit of the used factor memory (memory_bytes(), checked before and
    ///        after the factorisation; not a bound on the process peak, see the class doc);
    ///        empty: half of the physical memory available now (core::available_memory_bytes();
    ///        unlimited if that is unknown, i.e. 0, with an SBEM_WARN).
    /// @param expected_fill expected nnz(L + U) / nnz(A) (finite, > 0); sets the pre-check
    ///        estimate and Eigen's initial reservation of the factor arrays (Eigen's fill-factor
    ///        parameter ceil(expected_fill) clamped to [1, 20]; 20 is Eigen's default; about
    ///        ceil(expected_fill) x nnz(A) x 42 B are reserved up front, see the class doc).
    /// @throws std::invalid_argument for an empty or non-square A, a non-finite entry, or
    ///         expected_fill not finite and > 0; std::runtime_error when the estimate exceeds the
    ///         budget (before the factorisation), when Eigen reports a failure (info() !=
    ///         Success: a structurally or numerically singular matrix, or memory; the message
    ///         contains Eigen's lastErrorMessage()), or when the actual factor memory exceeds
    ///         the budget (after the factorisation).
    explicit SparseLuPreconditioner(const op::SparseOperator::Matrix& A,
                                    std::optional<std::size_t> budget_bytes = std::nullopt,
                                    Real expected_fill = 1.0);
    ~SparseLuPreconditioner() override;
    SparseLuPreconditioner(const SparseLuPreconditioner&) = delete;
    SparseLuPreconditioner& operator=(const SparseLuPreconditioner&) = delete;
    SparseLuPreconditioner(SparseLuPreconditioner&&) noexcept;
    SparseLuPreconditioner& operator=(SparseLuPreconditioner&&) noexcept;

    /// z = A^{-1} r (r and z may be the same vector). Sequential.
    /// @throws std::invalid_argument if r.size() != size().
    void apply(const VectorXc& r, VectorXc& z) const override;
    [[nodiscard]] std::string name() const override { return "sparse_lu"; }

    [[nodiscard]] Index size() const { return n_; }
    /// Stored entries of A (nnz(A); explicit zeros count).
    [[nodiscard]] Index input_nonzeros() const { return nnz_a_; }
    /// Entries of L (supernodal, incl. the dense supernode blocks) and of U (Eigen's nnzL(),
    /// nnzU()).
    [[nodiscard]] Index nonzeros_l() const { return nnz_l_; }
    [[nodiscard]] Index nonzeros_u() const { return nnz_u_; }
    /// nnz(L) + nnz(U).
    [[nodiscard]] Index factor_nonzeros() const { return nnz_l_ + nnz_u_; }
    /// factor_nonzeros() / input_nonzeros(): the expected_fill to pass for a larger case.
    [[nodiscard]] Real fill_factor() const;
    /// Used factor memory: factor_nonzeros() x kBytesPerEntry + the two permutations and the
    /// per-column pointer arrays (8 bytes per column each). Counts the used entries, not the
    /// capacity Eigen reserved for them (which the object holds and which is larger; see the
    /// class doc).
    [[nodiscard]] std::size_t memory_bytes() const;
    /// The pre-check estimate expected_fill x nnz(A) x kBytesPerEntry [bytes] (used entries,
    /// like memory_bytes()).
    [[nodiscard]] std::size_t estimated_bytes() const { return estimate_; }
    /// The budget both checks used [bytes] (SIZE_MAX when unlimited).
    [[nodiscard]] std::size_t budget_bytes() const { return budget_; }
    [[nodiscard]] Real expected_fill() const { return expected_fill_; }
    /// ColMajor conversion + analyzePattern (COLAMD, elimination tree) [s].
    [[nodiscard]] double analysis_seconds() const { return analysis_s_; }
    /// Numeric factorisation [s].
    [[nodiscard]] double factorization_seconds() const { return factor_s_; }
    [[nodiscard]] double setup_seconds() const { return analysis_s_ + factor_s_; }
    /// "sparse LU: n unknowns, nnz(A) a, nnz(L + U) f (L l, U u), fill F, M MB (estimate E MB,
    /// budget B MB), setup t s (analysis x s, factorisation y s)" (MB = 10^6 bytes).
    [[nodiscard]] std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Index n_ = 0;
    Index nnz_a_ = 0;
    Index nnz_l_ = 0;
    Index nnz_u_ = 0;
    Real expected_fill_ = 1.0;
    std::size_t estimate_ = 0;
    std::size_t budget_ = 0;
    double analysis_s_ = 0;
    double factor_s_ = 0;
};

}  // namespace specklebem::solver
