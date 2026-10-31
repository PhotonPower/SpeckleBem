#pragma once
/// @file schur_preconditioner.hpp
/// Near-field Schur-complement preconditioner (WP-SC2, ADR 0012 decision 2; Ergül, Malas &
/// Gürel 2011). An explicit option, never a default (make_schur_preconditioner in simulation.hpp
/// builds it from the near parts of an MLFMM operator).
#include "specklebem/core/types.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <Eigen/LU>

#include <cstddef>
#include <string>
#include <vector>

namespace specklebem::solver {

/// Parameters of SchurComplementPreconditioner.
struct SchurParams {
    /// Smallest accepted estimated reciprocal condition number (Eigen's 1-norm estimate,
    /// PartialPivLU::rcond()) of every J block of A and every M block of the Schur complement;
    /// finite and >= 0.
    Real min_rcond = BlockDiagonalPreconditioner::kDefaultMinReciprocalCondition;
};

/// Approximate block LU of the near-field matrix Z_n = [A B; C D] (2N x 2N; rows = the J-test
/// and M-test equations, columns = [J; M], so A = Z_n[J, J], B = Z_n[J, M], C = Z_n[M, J],
/// D = Z_n[M, M], each N x N; docs/06, docs/03):
///
///   M = [Ã 0; C I] [I Ã^{-1} B; 0 S̃_b],  i.e.  M = [Ã  B; C  S̃_b + C Ã^{-1} B],
///
/// with
///  * Ã = blockdiag_k A(I_k, I_k) over the J blocks I_k (`j_blocks`; for the near field: the J
///    unknowns of one octree box), each factorised by Eigen::PartialPivLU;
///  * S̃ = D - C Ã^{-1} B restricted to the pattern of D (entries where D stores nothing are
///    zero), and S̃_b = blockdiag_b S̃(R_b, R_b) over the M blocks R_b (`m_blocks`), each
///    factorised by Eigen::PartialPivLU. Only these diagonal blocks of S̃ are formed: the
///    preconditioner never uses the entries of S̃ outside them;
///  * B and C exact (the whole near pattern, including the coupling between boxes).
///
/// apply(): y1 = Ã^{-1} r1; y2 = S̃_b^{-1} (r2 - C y1); x1 = y1 - Ã^{-1} (B y2); x2 = y2 (a
/// fixed linear operator, usable on either GMRES side). With one J block and one M block and a
/// pattern of D that holds every entry of C A^{-1} B (e.g. a dense D), M = Z_n exactly.
///
/// Formation of S̃_b (setup): for each M block b (in parallel) and each J block k coupled to it
/// both ways (C(R_b, I_k) != 0 and B(I_k, R_b) != 0), in increasing k: the dense non-zero parts
/// C_kb = C(R_kb, I_k), B_kb = B(I_k, Q_kb) (R_kb, Q_kb the rows / columns of R_b with a
/// non-zero), T = Ã_k^{-1} B_kb, S̃(R_kb, Q_kb) -= C_kb T on the pattern of D. Work: sum over
/// the pairs of n_k^2 |Q_kb| + |R_kb| n_k |Q_kb| complex multiply-adds (formation_macs()). Each
/// S̃_b is accumulated by one task in a fixed order, so the setup is bitwise independent of the
/// thread count (as for BlockDiagonalPreconditioner: the BLAS calls nested in the parallel
/// region run single-threaded with an OpenMP-built OpenBLAS; a pthreads-built BLAS needs
/// OPENBLAS_NUM_THREADS=1). apply() uses explicit loops in a fixed order (block solves as
/// BlockDiagonalPreconditioner::apply, sparse products row by row, sources in order), so z is
/// bitwise identical for any number of OpenMP threads.
/// The LU factors of the owning and the borrowing constructor are bitwise equal; apply() is not
/// bitwise equal across a different split of the same Z_n into sources (the B and C row sums run
/// source by source), only to rounding.
///
/// Left preconditioning: a row scaling diag(alpha I_N, beta I_N) of Z (PMCHWT -> ICTF / MCTF)
/// scales A, B by alpha and C, D by beta, hence Ã by alpha, S̃ by beta and M by the same row
/// scaling: M^{-1} Z is unchanged up to rounding (the ADR 0011 invariance, tested).
///
/// Storage: the LU factors of the J and M blocks (16 n^2 bytes per block), and B, C as the
/// sparse matrices of the input: the owning constructor keeps a copy of Z_n's off-diagonal
/// parts B and C (counted in memory_bytes()), the borrowing one reads them from the given
/// sources (not counted; they must outlive the preconditioner). Setup needs in addition a
/// column-wise copy of B (nnz(B) x 24 bytes) and the diagonal blocks of S̃ before their LU
/// (16 m^2 bytes per M block), both released before the constructor returns.
class SchurComplementPreconditioner final : public Preconditioner {
public:
    using Matrix = op::SparseOperator::Matrix;

    /// Owning constructor: Z_n (2N x 2N, RowMajor; need not be compressed), copied or moved in;
    /// after the setup only its off-diagonal parts B and C are kept.
    /// @param N half of the size (the number of J and of M unknowns), >= 1.
    /// @param j_blocks partition of the J unknowns [0, N): every index exactly once, no index
    ///        >= N; the order inside a block is the row / column order of its matrix; ids empty
    ///        (block number) or one per block.
    /// @param m_blocks partition of the M unknowns [N, 2N) (global indices): every index exactly
    ///        once, no index < N.
    /// @throws std::invalid_argument for N < 1, Z_n not 2N x 2N, an invalid partition (an index
    ///         missing, twice, outside its half or outside [0, 2N); an empty block; ids of the
    ///         wrong size), params.min_rcond not finite and >= 0, or a row of the matrix with
    ///         unsorted column indices; std::runtime_error for a J block of A or an M block of
    ///         S̃ with non-finite entries or an estimated reciprocal condition number below
    ///         params.min_rcond or exactly singular (a zero pivot on the U diagonal, refused
    ///         independently of the estimate; the message names the block id, its size and the
    ///         estimate, taken as 0 for a zero pivot). No fallback.
    SchurComplementPreconditioner(Matrix Zn, Index N, BlockPartition j_blocks,
                                  BlockPartition m_blocks, SchurParams params = {});
    /// Borrowing constructor: Z_n = the sum of the sparse `sources` (each 2N x 2N, RowMajor,
    /// compressed, with sorted column indices in every row; an entry stored in several sources
    /// is summed, the near-field contract of ADR 0011 stores each in exactly one), e.g. Z_rr's
    /// near part and the PML near ring of an MLFMM operator. The sources are read during setup
    /// and by apply() (B and C), never copied: they must outlive the preconditioner.
    /// @throws as the owning constructor, plus std::invalid_argument for no source, a null,
    ///         non-square, differently sized or uncompressed source.
    SchurComplementPreconditioner(std::vector<const Matrix*> sources, Index N,
                                  BlockPartition j_blocks, BlockPartition m_blocks,
                                  SchurParams params = {});
    SchurComplementPreconditioner(const SchurComplementPreconditioner&) = delete;
    SchurComplementPreconditioner& operator=(const SchurComplementPreconditioner&) = delete;
    SchurComplementPreconditioner(SchurComplementPreconditioner&&) = delete;
    SchurComplementPreconditioner& operator=(SchurComplementPreconditioner&&) = delete;
    ~SchurComplementPreconditioner() override = default;

    /// z = M^{-1} r (r and z may be the same vector). O(sum n_k^2 + sum m_b^2 + nnz(B) +
    /// nnz(C)): two J-block solves, one M-block solve, one product with B and one with C.
    /// @throws std::invalid_argument if r.size() != size().
    void apply(const VectorXc& r, VectorXc& z) const override;
    [[nodiscard]] std::string name() const override { return "schur"; }

    /// 2N.
    [[nodiscard]] Index size() const { return 2 * n_; }
    /// N.
    [[nodiscard]] Index half_size() const { return n_; }
    [[nodiscard]] Index num_j_blocks() const { return static_cast<Index>(lu_a_.size()); }
    [[nodiscard]] Index num_m_blocks() const { return static_cast<Index>(lu_s_.size()); }
    [[nodiscard]] Index min_j_block_size() const { return min_j_; }
    [[nodiscard]] Index max_j_block_size() const { return max_j_; }
    [[nodiscard]] Index min_m_block_size() const { return min_m_; }
    [[nodiscard]] Index max_m_block_size() const { return max_m_; }
    /// The partitions as given (m_blocks with global indices in [N, 2N)).
    [[nodiscard]] const BlockPartition& j_partition() const { return j_part_; }
    [[nodiscard]] const BlockPartition& m_partition() const { return m_part_; }
    /// LU of the J block k of A, Ã_k = A(I_k, I_k). @throws std::out_of_range
    [[nodiscard]] const Eigen::PartialPivLU<MatrixXc>& j_block_lu(Index k) const;
    /// LU of the M block b of the Schur complement, S̃(R_b, R_b) (local order of
    /// m_partition().blocks[b]; reconstructedMatrix() returns the block). @throws
    /// std::out_of_range
    [[nodiscard]] const Eigen::PartialPivLU<MatrixXc>& m_block_lu(Index b) const;
    /// Smallest estimated reciprocal condition number over the J blocks of A / the M blocks of
    /// S̃, and the id of its block.
    [[nodiscard]] Real min_reciprocal_condition_a() const { return min_rcond_a_; }
    [[nodiscard]] Index worst_j_block_id() const { return worst_a_id_; }
    [[nodiscard]] Real min_reciprocal_condition_s() const { return min_rcond_s_; }
    [[nodiscard]] Index worst_m_block_id() const { return worst_s_id_; }
    /// Entries of the J blocks of A that no source stores (zero in Ã).
    [[nodiscard]] Index missing_entries_a() const { return missing_a_; }
    /// Entries of the M blocks of S̃ outside the pattern of D (zero in S̃_b).
    [[nodiscard]] Index masked_entries_s() const { return masked_s_; }
    /// sum_k n_k^2 and sum_b m_b^2 (the entries of the J blocks of A and of the M blocks of S̃).
    [[nodiscard]] Index j_block_entries() const;
    [[nodiscard]] Index m_block_entries() const;
    /// Stored entries of B and of C (in the sources; explicit zeros count).
    [[nodiscard]] Index nonzeros_b() const { return nnz_b_; }
    [[nodiscard]] Index nonzeros_c() const { return nnz_c_; }
    /// Number of (M block, J block) pairs coupled both ways (one dense product each in the
    /// formation of S̃_b).
    [[nodiscard]] Index coupling_pairs() const { return pairs_; }
    /// Complex multiply-adds of the formation of S̃_b: sum over the pairs of n_k^2 |Q_kb| (the
    /// LU solve) + |R_kb| n_k |Q_kb| (the product).
    [[nodiscard]] double formation_macs() const { return macs_; }
    /// True for the borrowing constructor (B and C read from the sources, not counted).
    [[nodiscard]] bool borrows_sources() const { return owned_.rows() == 0; }
    /// LU factors and pivots of both block sets, the index lists, the per-row offsets of the
    /// B / C parts of each source, and (owning constructor) the kept copy of B and C [bytes].
    [[nodiscard]] std::size_t memory_bytes() const;
    /// Extraction + LU of the J blocks of A [s].
    [[nodiscard]] double a_factorization_seconds() const { return a_s_; }
    /// Column-wise copy of B, extraction of the M blocks of D and formation of S̃_b [s].
    [[nodiscard]] double schur_formation_seconds() const { return form_s_; }
    /// LU of the M blocks of S̃ [s].
    [[nodiscard]] double schur_factorization_seconds() const { return s_s_; }
    /// Validation and row offsets + the three above [s].
    [[nodiscard]] double setup_seconds() const { return prep_s_ + a_s_ + form_s_ + s_s_; }
    /// "Schur complement: J blocks K (n_min / mean / n_max), M blocks K (...), nnz(B) b, nnz(C)
    /// c (borrowed | owned), P coupling pairs (F GFlop), M MB, setup t s (A LU a s, S̃ formation
    /// f s, S̃ LU l s), min rcond A r (block id), S̃ r (block id)" (MB = 10^6 bytes, GFlop = 8
    /// formation_macs() / 10^9), followed by the missing / masked counts when > 0.
    [[nodiscard]] std::string describe() const;

private:
    void setup(std::vector<const Matrix*> sources, BlockPartition j_blocks, BlockPartition m_blocks,
               const SchurParams& params);

    Index n_ = 0;
    BlockPartition j_part_;
    BlockPartition m_part_;
    /// m_part_ with local indices (global - N).
    std::vector<std::vector<Index>> m_local_;
    std::vector<const Matrix*> sources_;
    /// Per source, per row r: the position of the first entry of row r with column >= N (the
    /// J rows' B part starts there, the M rows' C part ends there).
    std::vector<std::vector<Index>> split_;
    Matrix owned_;  ///< owning constructor: B and C of Z_n (A and D removed after the setup)
    std::vector<Eigen::PartialPivLU<MatrixXc>> lu_a_;
    std::vector<Eigen::PartialPivLU<MatrixXc>> lu_s_;
    Index min_j_ = 0, max_j_ = 0, min_m_ = 0, max_m_ = 0;
    Real min_rcond_a_ = 0, min_rcond_s_ = 0;
    Index worst_a_id_ = -1, worst_s_id_ = -1;
    Index missing_a_ = 0, masked_s_ = 0;
    Index nnz_b_ = 0, nnz_c_ = 0, pairs_ = 0;
    double macs_ = 0;
    double prep_s_ = 0, a_s_ = 0, form_s_ = 0, s_s_ = 0;
};

}  // namespace specklebem::solver
