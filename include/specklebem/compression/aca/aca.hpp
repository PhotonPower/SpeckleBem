#pragma once
/// @file aca.hpp
/// Partially pivoted adaptive cross approximation (ACA, Bebendorf 2000) of a matrix that is
/// known only through its rows and columns: kernel-independent low-rank compression of one
/// admissible block (WP-PML5b: the far blocks of the PML ring, ADR 0008 amendment 2026-10-18
/// decision 3b; Phase 8 reuses it for AcaStrategy, docs/09). Depends on core and Eigen only.
///
/// compress() builds A ~ S_k = U V^T = sum_l u_l v_l^T (plain transpose, no conjugation: a
/// complex symmetric block is applied transposed from the same factors, (U V^T)^T x = V (U^T x)):
///   1. row i_k of the residual: r = A(i_k, :) - sum_l U(i_k, l) V(:, l)^T;
///   2. column pivot j_k = argmax over the unused columns of |r_j| (first index on ties),
///      v_k = r / r_{j_k};
///   3. column j_k of the residual: u_k = A(:, j_k) - sum_l V(j_k, l) U(:, l);
///   4. ||S_k||_F^2 = ||S_{k-1}||_F^2 + 2 Re sum_{l<k} (u_l^H u_k)(v_l^H v_k) + ||u_k||^2 ||v_k||^2
///      (exact update of the Frobenius norm of the approximation);
///   5. stop (StopReason::converged) once ||u_k|| ||v_k|| <= epsilon ||S_k||_F, the standard
///      stopping rule, has held in CompressParams::confirmations consecutive steps (default 2);
///      otherwise the next row i_{k+1} = argmax over the unused rows of |u_k(i)|.
/// Zero pivots (a residual row that is exactly zero, e.g. a row already reproduced exactly or a
/// sub-block that vanishes identically): the row is marked used and the next unused row is
/// taken (first the unused CompressParams::preferred_rows in their order, then the unused rows in
/// ascending order after the last one tried). After max_zero_rows consecutive zero rows, with
/// k > 0, one probe column is generated first (once per compression: the unused column farthest
/// in index from every used one, i.e. with spatially sorted columns the least like the pivots):
/// if its residual u, taken as typical, leaves ||u|| sqrt(cols) > epsilon ||S_k||_F, the
/// iteration restarts at its largest unused-row entry (rows the approximation reproduces
/// exactly need not mean the block is done, e.g. a block-diagonal matrix); otherwise it ends
/// as StopReason::zero_rows (every sampled residual row vanishes and the probe column is
/// within the tolerance); with k = 0 the probe columns (unused preferred_columns,
/// else column 0) are generated first, and a non-zero probe column restarts the iteration at its
/// largest entry; only if every probe is zero the block is returned as rank 0
/// (StopReason::zero_block). All rows used: StopReason::exhausted (the approximation reproduces
/// every row: exact). k reaching min(rows, cols) without the rule firing: exhausted as well (all
/// rows or all columns pivoted: exact up to rounding). k reaching a smaller max_rank:
/// StopReason::rank_cap, the only unconverged outcome (the caller falls back, e.g. to the exact
/// block).
///
/// The result depends only on the generator's values (deterministic: no randomness, no threads).
/// Cost: k rows and k columns of A (plus zero rows and probes) and O(k^2 (rows + cols)) flops.
/// Memory 16 k (rows + cols) bytes for the factors plus two scratch vectors.
#include "specklebem/core/types.hpp"

#include <cstddef>
#include <vector>

namespace specklebem::aca {

/// Access to the entries of A (rows x cols) by rows and columns. Implementations may cache;
/// compress() calls them from one thread.
class EntryGenerator {
public:
    virtual ~EntryGenerator() = default;
    /// Row i of A into out[0 .. cols).
    virtual void row(Index i, Complex* out) const = 0;
    /// Column j of A into out[0 .. rows).
    virtual void column(Index j, Complex* out) const = 0;
};

struct CompressParams {
    /// Relative tolerance of the stopping rule ||u_k|| ||v_k|| <= epsilon ||S_k||_F, in (0, 1).
    Real epsilon = 1e-4;
    /// Largest rank (> 0); 0 = min(rows, cols). Reaching it without convergence returns
    /// StopReason::rank_cap.
    Index max_rank = 0;
    /// Consecutive steps that must meet the stopping rule before it ends the iteration (>= 1).
    /// 1 is the plain rule; the default 2 guards against a single small increment of the partial
    /// pivoting (measured: the plain rule left 1.04 epsilon on a 200 x 150 k = 6 Helmholtz block
    /// of two unit clusters 3 apart, eps 1e-6) at the price of a few more terms.
    Index confirmations = 2;
    /// Consecutive zero residual rows that end the iteration (>= 1).
    Index max_zero_rows = 3;
    /// Rows tried first: the initial pivot row and the replacements after zero rows (e.g. rows
    /// the generator already holds in a cache). Entries outside [0, rows) are ignored.
    std::vector<Index> preferred_rows;
    /// Probe columns of the zero-block check (k = 0 only). Entries outside [0, cols) are ignored.
    std::vector<Index> preferred_columns;
};

enum class StopReason { converged, zero_rows, zero_block, exhausted, rank_cap };

/// "converged", "zero_rows", "zero_block", "exhausted" or "rank_cap".
[[nodiscard]] const char* to_string(StopReason r);

/// A ~ U V^T (rows x cols); rank() = U.cols() = V.cols().
struct LowRankBlock {
    Index rows = 0;
    Index cols = 0;
    MatrixXc U;  ///< rows x k
    MatrixXc V;  ///< cols x k
    StopReason reason = StopReason::converged;
    Real norm = 0;                ///< ||U V^T||_F (step 4, the running estimate of ||A||_F)
    Real last_increment = 0;      ///< ||u_k|| ||v_k|| of the last term (0 for rank 0)
    Index generated_rows = 0;     ///< EntryGenerator::row calls
    Index generated_columns = 0;  ///< EntryGenerator::column calls
    Index zero_rows = 0;          ///< zero residual rows met

    [[nodiscard]] Index rank() const { return U.cols(); }
    /// False only for StopReason::rank_cap.
    [[nodiscard]] bool converged() const { return reason != StopReason::rank_cap; }
    /// 16 rank (rows + cols): the factors.
    [[nodiscard]] std::size_t memory_bytes() const;
};

/// ACA of the rows x cols matrix given by gen (file comment).
/// @throws std::invalid_argument for rows or cols < 1, epsilon outside (0, 1), max_rank < 0 or
///         max_zero_rows < 1; std::runtime_error if the generator returns a non-finite pivot or
///         column entry.
[[nodiscard]] LowRankBlock compress(Index rows, Index cols, const EntryGenerator& gen,
                                    const CompressParams& params);

/// ||A - U V^T||_F for an explicit A (tests and diagnostics).
/// @throws std::invalid_argument if the sizes differ.
[[nodiscard]] Real frobenius_error(const MatrixXc& A, const LowRankBlock& b);

}  // namespace specklebem::aca
