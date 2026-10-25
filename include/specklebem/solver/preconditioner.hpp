#pragma once
/// @file preconditioner.hpp
/// Preconditioners M^{-1} for the iterative solver, usable on either side (default left; see
/// solver::PreconditionerSide in gmres.hpp).
#include "specklebem/operator/linear_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <Eigen/LU>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace specklebem::solver {

class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void apply(const VectorXc& r, VectorXc& z) const = 0;  ///< z = M^{-1} r
    [[nodiscard]] virtual std::string name() const = 0;
};

class IdentityPreconditioner final : public Preconditioner {
public:
    void apply(const VectorXc& r, VectorXc& z) const override { z = r; }
    [[nodiscard]] std::string name() const override { return "none"; }
};

/// Diagonal (Jacobi) preconditioner: the one that gave the fastest convergence
/// for Ag surfaces in the paper. Typically built from op::assemble_diagonal(problem).
class DiagonalPreconditioner final : public Preconditioner {
public:
    /// Stores 1 / diag; apply() is z = r .* (1 / diag), O(n).
    /// @throws std::invalid_argument if diag is empty or has zero or non-finite entries.
    explicit DiagonalPreconditioner(VectorXc diag);
    /// z = M^{-1} r. @throws std::invalid_argument if r.size() != size().
    void apply(const VectorXc& r, VectorXc& z) const override;
    [[nodiscard]] Index size() const { return inv_diag_.size(); }
    [[nodiscard]] std::string name() const override { return "diagonal"; }

private:
    VectorXc inv_diag_;
};

/// Partition of the unknowns [0, n) into the diagonal blocks of a BlockDiagonalPreconditioner
/// (e.g. the J and M unknowns of the basis functions of one octree box per block,
/// mlfmm::level_unknown_blocks). Every index in [0, n) belongs to exactly one block; the order of
/// the indices in a block is the row / column order of the block matrix.
struct BlockPartition {
    std::vector<std::vector<Index>> blocks;
    /// One label per block for messages and reports (e.g. the octree box index); empty: the
    /// block number.
    std::vector<Index> ids;
};

/// The diagonal blocks D_k = S(I_k, I_k) of S = the sum of the sparse `sources` (all n x n, I_k =
/// partition.blocks[k]). Every same-block entry must be stored in exactly one source (the
/// near-field contract of ADR 0011: Z_rr's near part and the PML ring's near part partition the
/// near pattern); the values are copied, never added. O(nnz of the block rows) per block,
/// OpenMP-parallel over the blocks.
/// @throws std::invalid_argument for an invalid partition (an index outside [0, n), twice or
///         missing; ids of the wrong size), no source, a null, non-square or differently sized
///         source, and for a same-block entry stored in two sources or in none (the message names
///         the block id and the global row and column).
[[nodiscard]] std::vector<MatrixXc> extract_diagonal_blocks(
    const BlockPartition& partition, const std::vector<const op::SparseOperator::Matrix*>& sources);
/// The diagonal blocks D_k = A(I_k, I_k) of a dense n x n matrix (tests, the dense route).
/// @throws std::invalid_argument for an invalid partition or a non-square A.
[[nodiscard]] std::vector<MatrixXc> extract_diagonal_blocks(const BlockPartition& partition,
                                                            const MatrixXc& A);

/// Block-diagonal (block Jacobi) preconditioner M = diag(D_1, ..., D_K) on a partition of the
/// unknowns (WP-PC1, ADR 0011): z = M^{-1} r solves D_k z(I_k) = r(I_k) for every block k.
/// Simulation builds it from the near-field blocks of the octree boxes on one level (the leaves
/// or one level up; on the PML route from Z_rr's near part plus the near ring), or from the dense
/// matrix (make_block_preconditioner in simulation.hpp).
///
/// Each block is factorised with Eigen::PartialPivLU, in parallel over the blocks (one task per
/// block; the BLAS calls of the LU nested in the parallel region run single-threaded with an
/// OpenMP-built OpenBLAS, as in the MSYS2 and the usual Linux packages; a pthreads-built BLAS
/// needs OPENBLAS_NUM_THREADS=1 for the bitwise thread-count independence). apply() runs the
/// permutation and the forward / back substitutions with explicit loops in a fixed order, one
/// block per task, and writes block-disjoint parts of z, so z is bitwise identical for any
/// number of OpenMP threads. Left preconditioning as for Jacobi: a row scaling of A that is
/// constant on the rows of each block (e.g. PMCHWT -> ICTF / MCTF, which scale the J-test and
/// the M-test rows by two constants, with blocks holding both J and M unknowns) leaves
/// M^{-1} A unchanged up to rounding.
class BlockDiagonalPreconditioner final : public Preconditioner {
public:
    /// Default smallest accepted reciprocal condition number of a block (Eigen's 1-norm
    /// estimate, PartialPivLU::rcond()).
    static constexpr Real kDefaultMinReciprocalCondition = 1e-13;

    /// Factorises blocks[k] (the matrix of partition.blocks[k]); extraction_seconds() = 0.
    /// @throws std::invalid_argument for an invalid partition, blocks.size() != the number of
    ///         blocks, a block of the wrong size or min_rcond not finite and >= 0.
    /// @throws std::runtime_error for a block with non-finite entries or an estimated
    ///         reciprocal condition number below min_rcond (singular or ill-conditioned; the
    ///         message names the block id, its size and the estimate). No fallback.
    BlockDiagonalPreconditioner(BlockPartition partition, std::vector<MatrixXc> blocks,
                                Real min_rcond = kDefaultMinReciprocalCondition);
    /// extract_diagonal_blocks(partition, sources), then factorised (both timed).
    /// @throws as extract_diagonal_blocks and the constructor above.
    BlockDiagonalPreconditioner(BlockPartition partition,
                                const std::vector<const op::SparseOperator::Matrix*>& sources,
                                Real min_rcond = kDefaultMinReciprocalCondition);
    /// extract_diagonal_blocks(partition, A), then factorised (both timed).
    /// @throws as extract_diagonal_blocks and the first constructor.
    BlockDiagonalPreconditioner(BlockPartition partition, const MatrixXc& A,
                                Real min_rcond = kDefaultMinReciprocalCondition);

    /// z = M^{-1} r (r and z may be the same vector). O(sum_k n_k^2).
    /// @throws std::invalid_argument if r.size() != size().
    void apply(const VectorXc& r, VectorXc& z) const override;
    [[nodiscard]] std::string name() const override { return "block_diagonal"; }

    [[nodiscard]] Index size() const { return n_; }
    [[nodiscard]] Index num_blocks() const { return static_cast<Index>(lu_.size()); }
    [[nodiscard]] Index min_block_size() const { return min_block_; }
    [[nodiscard]] Index max_block_size() const { return max_block_; }
    [[nodiscard]] const BlockPartition& partition() const { return partition_; }
    /// The LU factorisation of block k. @throws std::out_of_range
    [[nodiscard]] const Eigen::PartialPivLU<MatrixXc>& block_lu(Index k) const;
    /// Smallest estimated reciprocal condition number over the blocks, and the id of its block.
    [[nodiscard]] Real min_reciprocal_condition() const { return min_rcond_seen_; }
    [[nodiscard]] Index worst_block_id() const { return worst_id_; }
    /// LU factors (16 n_k^2 bytes per block), pivots and the index lists [bytes].
    [[nodiscard]] std::size_t memory_bytes() const;
    [[nodiscard]] double extraction_seconds() const { return extract_s_; }
    [[nodiscard]] double factorization_seconds() const { return factor_s_; }
    /// Extraction + factorisation [s].
    [[nodiscard]] double setup_seconds() const { return extract_s_ + factor_s_; }
    /// "block-diagonal: K blocks of n_min / mean / n_max unknowns, M MB, setup t s (extraction
    /// a s, LU b s), min rcond r (block id)" (MB = 10^6 bytes).
    [[nodiscard]] std::string describe() const;

private:
    void factorize(std::vector<MatrixXc> blocks, Real min_rcond);

    BlockPartition partition_;
    Index n_ = 0;
    Index min_block_ = 0;
    Index max_block_ = 0;
    std::vector<Eigen::PartialPivLU<MatrixXc>> lu_;
    Real min_rcond_seen_ = 0;
    Index worst_id_ = -1;
    double extract_s_ = 0;
    double factor_s_ = 0;
};

/// Schur-complement preconditioner (Ergül, Malas & Gürel 2011) for metals
/// (planned; Phase 8).
class SchurComplementPreconditioner;

}  // namespace specklebem::solver
