#pragma once
/// @file incomplete_lu_preconditioner.hpp
/// Incomplete LU (ILUT) of the near-field blocks A = Z_n[J, J] and D = Z_n[M, M] as a
/// preconditioner M^{-1} = diag(Ã^{-1}, D̃^{-1}) (WP-SC3, ADR 0012 decision 6). An explicit
/// option, never a default (make_incomplete_lu_preconditioner in simulation.hpp builds it from the
/// near parts of an MLFMM operator).
///
/// Own implementation, not Eigen::IncompleteLUT. Eigen's class (identical in the fetched 3.4.0
/// and the MSYS2 5.0.1-dev sources) was checked and rejected: (1) it replaces an exactly zero
/// pivot by sqrt(droptol) x the row norm without telling (a silent fallback, forbidden here) and
/// never checks tiny pivots; (2) it drops an L multiplier by its absolute size |l_ik| <= droptol
/// (dimensionless) but U entries relative to the row's 2-norm, and its fill limit is an integer
/// (fill_in = fillfactor x nnz(A) / n + 1 entries per row, half of them for L, half for U, one
/// global count for all rows); (3) it selects the next pivot row by a linear minimum search over
/// the remaining lower part of the work row, O(len^2) per row, prohibitive for rows with
/// hundreds to thousands of entries. Its ordering (AMD on the pattern of A + A^T, no numerical
/// pivoting) is deterministic and is reused here (Eigen::AMDOrdering).
#include "specklebem/core/types.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace specklebem::solver {

/// Symmetric ordering of a block before its incomplete factorisation (applied to rows and
/// columns alike, so the diagonal stays on the diagonal; ILUT does no numerical pivoting).
enum class IlutOrdering {
    /// Approximate minimum degree on the pattern of B + B^T (Eigen::AMDOrdering; the ordering of
    /// Eigen's IncompleteLUT). The default.
    Amd,
    /// The given order of the unknowns (for the near field: the RWG function order of the mesh).
    Natural
};

/// "amd" or "natural".
[[nodiscard]] const char* to_string(IlutOrdering ordering);

/// Parameters of IncompleteLuPreconditioner (dual-threshold ILUT, Saad 1994).
struct IlutParams {
    /// Drop tolerance tau (finite, >= 0): in row i an entry w_j of the work row (an L entry
    /// before its division by the pivot, i.e. |l_ij| |u_jj|, or a U entry) is dropped when
    /// |w_j| <= tau ||b_i||_2, the 2-norm of the row of the block (relative to the row norm, so
    /// the factorisation is invariant under a scaling of the block by a constant). 0: drop only
    /// exact zeros.
    Real drop_tol = 1e-4;
    /// Fill limit (finite, > 0): row i of L + U keeps at most max(1, round(fill_factor x nnz_i))
    /// entries, nnz_i = the stored entries of the block's row i; the diagonal is always kept and
    /// the off-diagonal L and U entries compete for the remaining places by |w_j| (the
    /// pre-division magnitude above; ties by the smaller column). Hence nnz(L + U) <=
    /// fill_factor x nnz(B) + n per block: fill_ratio() <= fill_factor up to the rounding.
    Real fill_factor = 3.0;
    /// Pivot guard (finite, >= 0): a pivot with |u_ii| <= min_pivot_rel x ||b_i||_2 (also an
    /// exactly zero or non-finite one) stops the setup with std::runtime_error naming the block
    /// and the row; never replaced. Default 1e-10: the triangular solves would amplify by more
    /// than 10^10 relative to the row, i.e. the factorisation has broken down.
    Real min_pivot_rel = 1e-10;
    /// Limit of the factor memory [bytes], > 0 when set; checked before the factorisation
    /// against the bound that the fill limit implies (bound_bytes(): every row filled to its
    /// limit, the capacity reserved for the factors) and after it against memory_bytes(). Empty:
    /// half of the physical memory available now (core::available_memory_bytes(); unlimited if
    /// unknown, with an SBEM_WARN).
    std::optional<std::size_t> budget_bytes;
    IlutOrdering ordering = IlutOrdering::Amd;
};

/// Checks IlutParams as the IncompleteLuPreconditioner constructors do.
/// @throws std::invalid_argument for drop_tol not finite and >= 0, fill_factor not finite and
///         > 0, min_pivot_rel not finite and >= 0, budget_bytes = 0, or an unknown ordering.
void validate_ilut_params(const IlutParams& params);

/// M = diag(Ã, D̃) with Ã = L_A U_A ≈ P_A A P_A^T and D̃ = L_D U_D ≈ P_D D P_D^T the dual-threshold
/// incomplete LU factorisations (ILUT, Saad 1994: row-wise IKJ elimination, drop rule relative
/// to the row norm, fill limit per row; no numerical pivoting) of the two N x N diagonal blocks
/// of a 2N x 2N system [A B; C D] (x = [J; M], docs/06). B and C are not used: on the rough
/// PML systems the near J-M coupling is weak and the near-field-inverse gain sits in the
/// inter-box coupling inside A and D (WP-SC2, ADR 0012 amendment).
///
/// Elimination of row i (in the permuted order): the row is scattered into a dense work vector;
/// its lower-part columns are visited in increasing order through a binary min-heap (fill-in
/// columns below i are pushed as they appear); a column k whose work entry passes the drop rule
/// gives the multiplier l_ik = w_k / u_kk and the update w -= l_ik u_k,(k+1:n), otherwise it is
/// zeroed and skipped. Then the pivot is checked (min_pivot_rel), the U entries are filtered by
/// the drop rule, the L and U candidates are cut to the row's fill limit (nth_element with a
/// total order: deterministic), sorted by column and appended to one CSR array per block (L part
/// then U part per row; the diagonal separately). With drop_tol = 0 and a fill limit that keeps
/// every entry this is the exact LU without pivoting.
///
/// Deterministic: each block is factorised sequentially by one thread (A and D in parallel,
/// OpenMP), apply() runs the forward / back substitution of each block sequentially (the two
/// blocks in parallel) with explicit loops in a fixed order; no BLAS and no threaded Eigen
/// kernel is involved, so setup and apply() are bitwise identical whether their OpenMP regions
/// run with one thread or two (tested: a run nested in an outer parallel region, i.e. with
/// teams of one, against a top-level run with teams of two). The ordering is Eigen's
/// sequential AMD.
///
/// Left preconditioning: a row scaling diag(alpha I_N, beta I_N) of Z (PMCHWT -> ICTF / MCTF)
/// scales A by alpha and D by beta. The drop rule, the fill selection and the pivot guard are
/// relative to the row norm and the ordering depends only on the pattern, so Ã scales by alpha
/// and D̃ by beta: M^{-1} Z is unchanged up to rounding (the ADR 0011 invariance, tested).
///
/// Memory: per block the factor entries (16 B value + 8 B column index = kBytesPerEntry), the
/// diagonal (16 B per row), two row-pointer arrays and the permutation (8 B per row each). The
/// setup reserves the bound of the fill limit up front and shrinks the arrays to their size at the
/// end, so memory_bytes() (computed from the capacities) is what the object holds. The setup needs
/// in addition per block a copy of the block (24 B per entry), the AMD work (Eigen: the
/// symmetrised pattern and its copies, about 3 x 16 B per entry of B + B^T) and O(n) work arrays,
/// released before the constructor returns; with A and D in parallel both coexist.
class IncompleteLuPreconditioner final : public Preconditioner {
public:
    using Matrix = op::SparseOperator::Matrix;
    /// Bytes per stored off-diagonal factor entry (complex<double> value + 64-bit column index).
    static constexpr std::size_t kBytesPerEntry = 24;

    /// From the two blocks (N x N each, RowMajor, need not be compressed; duplicates of an
    /// uncompressed matrix are summed).
    /// @throws std::invalid_argument for an empty or non-square block, blocks of different
    ///         sizes, a non-finite entry, or invalid params (drop_tol not finite and >= 0,
    ///         fill_factor not finite and > 0, min_pivot_rel not finite and >= 0, budget_bytes
    ///         = 0, an unknown ordering); std::runtime_error for a zero row, a zero, tiny or
    ///         non-finite pivot (the message names the block "A" or "D", the row in the block's
    ///         original numbering and |u_ii| / ||b_i||), or the bound / factor memory above the
    ///         budget (with the numbers). No fallback.
    IncompleteLuPreconditioner(const Matrix& A, const Matrix& D, const IlutParams& params = {});
    /// A = Z[0:N, 0:N] and D = Z[N:2N, N:2N] of Z = the sum of the sparse `sources` (each 2N x 2N,
    /// RowMajor; an entry stored in several sources is summed, the near-field contract of ADR
    /// 0011 stores each in exactly one), e.g. Z_rr's near part and the PML near ring of an MLFMM
    /// operator. Only the entries of A and D are read (B and C are ignored); the sources are not
    /// referenced after the constructor.
    /// @throws as the first constructor, plus std::invalid_argument for no source, a null source,
    ///         N < 1, or a source that is not 2N x 2N.
    IncompleteLuPreconditioner(const std::vector<const Matrix*>& sources, Index N,
                               const IlutParams& params = {});

    /// z = M^{-1} r: z(0:N) = P_A^T U_A^{-1} L_A^{-1} P_A r(0:N), the same with D for (N:2N)
    /// (r and z may be the same vector). O(nnz(L + U)).
    /// @throws std::invalid_argument if r.size() != size().
    void apply(const VectorXc& r, VectorXc& z) const override;
    [[nodiscard]] std::string name() const override { return "ilut"; }

    /// 2N.
    [[nodiscard]] Index size() const { return 2 * n_; }
    /// N.
    [[nodiscard]] Index half_size() const { return n_; }
    [[nodiscard]] const IlutParams& params() const { return params_; }
    /// Stored entries of A and of D (after summing duplicates; explicit zeros count).
    [[nodiscard]] Index input_nonzeros_a() const { return a_.input_nnz; }
    [[nodiscard]] Index input_nonzeros_d() const { return d_.input_nnz; }
    /// nnz(L + U) of the block: strictly lower L entries + strictly upper U entries + the
    /// diagonal (L's unit diagonal is not stored and not counted).
    [[nodiscard]] Index nonzeros_a() const { return a_.factor_nnz(); }
    [[nodiscard]] Index nonzeros_d() const { return d_.factor_nnz(); }
    /// nnz(L + U) / nnz(block).
    [[nodiscard]] Real fill_ratio_a() const { return a_.fill_ratio(); }
    [[nodiscard]] Real fill_ratio_d() const { return d_.fill_ratio(); }
    /// Smallest |u_ii| / ||b_i||_2 over the rows of the block (the pivot guard's quantity).
    [[nodiscard]] Real min_pivot_ratio_a() const { return a_.min_pivot_ratio; }
    [[nodiscard]] Real min_pivot_ratio_d() const { return d_.min_pivot_ratio; }
    /// Stability indicator of the block's triangular factors (Chow & Saad 1997), made
    /// scale-free: condest = ||(L U)^{-1} e||_inf ||B||_inf with e = (1, ..., 1) (one forward /
    /// back substitution) and ||B||_inf the largest absolute row sum of the block. Since
    /// ||(L U)^{-1} e||_inf <= ||(L U)^{-1}||_inf ||e||_inf = ||(L U)^{-1}||_inf, it is a lower
    /// bound of ||B||_inf ||(L U)^{-1}||_inf (the norm of the block times the norm of the inverse
    /// of its incomplete factors; neither kappa_inf(B) nor kappa_inf(L U), ||L U||_inf is not
    /// computed). Values far above the condition number of the block itself mean unstable
    /// triangular solves; a non-finite value stops the setup.
    [[nodiscard]] Real condest_a() const { return a_.condest; }
    [[nodiscard]] Real condest_d() const { return d_.condest; }
    /// Complex multiply-adds of the elimination (the row updates w -= l_ik u_k) of the block:
    /// the work measure of the setup.
    [[nodiscard]] double factorization_macs_a() const { return a_.macs; }
    [[nodiscard]] double factorization_macs_d() const { return d_.macs; }
    /// Extraction + ordering + factorisation of the block [s] (measured in its thread; A and D
    /// run in parallel, so setup_seconds() is about the larger of the two).
    [[nodiscard]] double setup_seconds_a() const { return a_.extract_s + a_.order_s + a_.factor_s; }
    [[nodiscard]] double setup_seconds_d() const { return d_.extract_s + d_.order_s + d_.factor_s; }
    /// Ordering (AMD) part of setup_seconds_a/d [s].
    [[nodiscard]] double ordering_seconds_a() const { return a_.order_s; }
    [[nodiscard]] double ordering_seconds_d() const { return d_.order_s; }
    /// Wall time of the whole setup (validation, both blocks) [s].
    [[nodiscard]] double setup_seconds() const { return setup_s_; }
    /// Memory held by the object [bytes], from the capacities of the factor arrays (equal to their
    /// sizes after the final shrink): per block nnz_off x kBytesPerEntry + 16 n (diagonal) + 8 (n
    /// + 1) x 2 (row pointers L / U split) + 8 n (permutation).
    [[nodiscard]] std::size_t memory_bytes() const;
    /// The pre-check bound [bytes]: the same count with every row filled to its fill limit.
    [[nodiscard]] std::size_t bound_bytes() const { return bound_; }
    /// The budget both checks used [bytes] (SIZE_MAX when unlimited).
    [[nodiscard]] std::size_t budget_bytes() const { return budget_; }
    /// "ILUT (fill F, drop T, ordering O): 2 x n unknowns, A nnz a -> L+U f (fill r, min pivot p,
    /// condest c), D ..., M MB (bound B MB, budget X MB), setup t s (A a s incl. ordering o s, D
    /// d s)"
    /// (MB = 10^6 bytes).
    [[nodiscard]] std::string describe() const;

    /// Factors of one block (public for the implementation's helpers; read-only use).
    struct Factor {
        Index n = 0;
        Index input_nnz = 0;
        /// perm[k] = the original index of the k-th unknown in the elimination order.
        std::vector<Index> perm;
        /// Row k of the factors: entries [ptr[k], mid[k]) are L (columns < k), [mid[k],
        /// ptr[k+1]) are U (columns > k), both sorted by column; ptr has n + 1 entries, mid n.
        std::vector<Index> ptr;
        std::vector<Index> mid;
        std::vector<Index> col;
        std::vector<Complex> val;
        std::vector<Complex> diag;  ///< u_kk
        Real min_pivot_ratio = 0;
        Real norm_inf = 0;  ///< ||B||_inf, the largest absolute row sum of the block
        double macs = 0;    ///< complex multiply-adds of the row updates
        Real condest = 0;   ///< ||(L U)^{-1} e||_inf ||B||_inf, e = ones
        double extract_s = 0, order_s = 0, factor_s = 0;

        [[nodiscard]] Index factor_nnz() const { return static_cast<Index>(col.size()) + n; }
        [[nodiscard]] Real fill_ratio() const;
        [[nodiscard]] std::size_t memory_bytes() const;
    };

    /// The factors of block A / D (read-only; for tests and diagnostics).
    [[nodiscard]] const Factor& factors_a() const { return a_; }
    [[nodiscard]] const Factor& factors_d() const { return d_; }

private:
    void setup(const std::vector<const Matrix*>& sources_a, Index offset_a,
               const std::vector<const Matrix*>& sources_d, Index offset_d, Index N);

    Index n_ = 0;
    IlutParams params_;
    Factor a_;
    Factor d_;
    std::size_t bound_ = 0;
    std::size_t budget_ = 0;
    double setup_s_ = 0;
};

}  // namespace specklebem::solver
