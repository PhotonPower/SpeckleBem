#pragma once
/// @file pml_ring_far.hpp
/// Internal (not installed): the far part of the PML ring shared by the exact path of WP-PML5a
/// and the ACA path of WP-PML5b (pml_ring.hpp, "ACA far ring"): triangle colouring, the far
/// partition into column clusters, the block entry generator over PairBlockEvaluator and the
/// ACA far ring itself.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/operator/assembler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace specklebem::mlfmm::detail {

/// Greedy triangle colouring of the dense assembler (triangles sharing a basis function get
/// different colours; colour[t] = the smallest colour not used by the other support triangle of
/// any basis function on t with a smaller index). The summation order of every far-ring entry
/// (test triangle by colour, source triangle ascending) follows it.
[[nodiscard]] std::vector<int> greedy_triangle_colours(const basis::RwgSpace& space);

/// Axis-aligned box (lo, hi) of support vertices; empty: lo > hi.
struct Extent {
    std::array<Real, 3> lo{1e300, 1e300, 1e300};
    std::array<Real, 3> hi{-1e300, -1e300, -1e300};
    void add(const Extent& o);
    void add_support(const basis::RwgSpace& space, Index basis);
    [[nodiscard]] Real diameter() const;  ///< diagonal (0 if empty)
};
/// Distance between two non-empty extents (0 if they overlap).
[[nodiscard]] Real extent_distance(const Extent& a, const Extent& b);

/// The far partition of one row cluster R (a box with stretched functions: a leaf, or with
/// PmlRingParams::row_clusters any level).
struct FarRowLeaf {
    Index row_box = -1;       ///< index into tree.boxes()
    std::vector<Index> rows;  ///< stretched basis indices of R, ascending position
    Extent extent;            ///< support extent of the rows
    /// ascending `first`, disjoint, covering every far column of A; geometry and admissible
    /// filled by partition_far_ring, kind / rank / region / bound by AcaFarRing
    std::vector<PmlRingFarBlock> blocks;
};

/// The far blocks (pml_ring.hpp, "ACA far ring") grouped by row cluster: without row_clusters
/// for every leaf with stretched functions (leaf order of boxes_at_level(leaf)) the column
/// descent (coarse as PmlRingParams::coarse_columns); with row_clusters (and use_aca) the dual
/// descent over same-level box pairs, row clusters sorted by level and first position. eta as
/// PmlRingParams. With use_aca = false every far leaf is an exact (non-admissible) block.
/// @throws std::invalid_argument for eta <= 0 (use_aca) or a tree not built on space.
[[nodiscard]] std::vector<FarRowLeaf> partition_far_ring(const Octree& tree,
                                                         const basis::RwgSpace& space,
                                                         const std::vector<std::uint8_t>& stretched,
                                                         bool use_aca, Real eta, bool coarse,
                                                         bool row_clusters);

/// The partition of partition_far_ring rebuilt from its flattened blocks (plan_pml_ring_far,
/// PmlRingPlan::far_blocks; WP-PML5c plan sharing): each run of consecutive blocks with one row
/// box forms one FarRowLeaf, its rows the stretched functions of that box in position order and
/// their support extent, i.e. exactly what partition_far_ring produced.
/// @throws std::invalid_argument if a block's row box is not a box of `tree` or its row count
///         differs from the box's stretched functions (a plan made on another tree /
///         classification).
[[nodiscard]] std::vector<FarRowLeaf> far_partition_from_blocks(
    const Octree& tree, const basis::RwgSpace& space, const std::vector<std::uint8_t>& stretched,
    std::vector<PmlRingFarBlock> blocks);

/// The evaluators of the far ring by region mask (bit 0 R1, bit 1 R2: index 1 = R1 only, 2 = R2
/// only, 3 = both) and the per-triangle-pair magnitude bound: region i of a triangle pair whose
/// vertex extents are farther apart than drop_distance[i] is dropped (the ADR 0008 §6 bound
/// applied per pair, as the WP21 exact parts do per basis pair); a pair with no region left is
/// skipped. tri_ext empty: no per-pair dropping.
struct PairEvaluators {
    std::array<const op::PairBlockEvaluator*, 4> eval{nullptr, nullptr, nullptr, nullptr};
    std::array<Real, 2> drop_distance{1e300, 1e300};
    std::vector<Extent> tri_ext;
};

/// Exact far-ring entries of a block rows x [c0, c0 + n) (positions) from triangle-pair blocks,
/// with the summation order of the WP-PML5a exact path (test triangle by colour, then source
/// triangle ascending): row(i) / column(j) / full() give bitwise the same entries. Rows and
/// columns are cached (each gives all four sub-blocks). Not thread-safe (one per thread).
class BlockSource {
public:
    /// @param rows basis indices of the rows; columns: perm[c0 .. c0 + n)
    /// @param mask the regions kept for the block (1, 2 or 3; see PairEvaluators)
    BlockSource(const basis::RwgSpace& space, const PairEvaluators& eval, int mask,
                const std::vector<int>& colour, const std::vector<Index>& perm,
                const std::vector<Index>& rows, Index c0, Index n);
    [[nodiscard]] Index rows() const { return m_; }
    [[nodiscard]] Index cols() const { return n_; }
    /// Row i: [JJ(i, :), JM(i, :), MJ(i, :), MM(i, :)], 4 n entries (cached).
    const Complex* row(Index i);
    /// Column j: [JJ(:, j), JM(:, j), MJ(:, j), MM(:, j)], 4 m entries (cached).
    const Complex* column(Index j);
    /// The whole block in the 5a layout: per column j the 4 m entries [JJ, JM, MJ, MM]
    /// (recomputed from the triangle pairs; the cache is not used).
    void full(std::vector<Complex>& w) const;
    /// Frees the cached rows and columns (WP-PML5b review W3: before full() allocates the exact
    /// block of a rank-cap fallback, so the transient per block is max(cache, block), not their
    /// sum); cached_rows() / cached_columns() and the generated counts are kept. Call it last
    /// (row() / column() after it would generate and count the entries again).
    void release_cache();
    /// Bytes of the cached rows and columns (64 n per row, 64 m per column).
    [[nodiscard]] std::size_t cache_bytes() const;
    [[nodiscard]] const std::vector<Index>& cached_rows() const { return row_order_; }
    [[nodiscard]] const std::vector<Index>& cached_columns() const { return col_order_; }
    [[nodiscard]] Index generated_rows() const { return static_cast<Index>(row_order_.size()); }
    [[nodiscard]] Index generated_columns() const { return static_cast<Index>(col_order_.size()); }
    [[nodiscard]] Index pair_evaluations() const { return evaluations_; }
    /// Triangle pairs with a region part dropped by the per-pair bound (incl. skipped pairs).
    [[nodiscard]] Index pair_reductions() const { return reductions_; }
    /// Triangle pairs skipped (no region left after the per-pair bound; part of the reductions).
    [[nodiscard]] Index pair_skips() const { return skips_; }

private:
    struct Tri {
        Index t = -1;
        std::array<Index, 3> local{-1, -1, -1};  ///< local row / column index per support slot
    };
    /// Evaluates the pair with the regions left after the per-pair bound; false: none left.
    bool evaluate(Index t, Index s, op::TrianglePairBlocks& out) const;
    const PairEvaluators& eval_;
    int mask_ = 3;
    mutable Index evaluations_ = 0, reductions_ = 0, skips_ = 0;
    Index m_ = 0, n_ = 0;
    std::vector<Tri> test_;  ///< test triangles of the rows, by (colour, index)
    std::vector<Tri> src_;   ///< source triangles of the columns, ascending
    std::vector<std::array<Index, 2>> row_tris_;  ///< row i -> its entries of test_ (colour order)
    std::vector<std::array<Index, 2>> col_tris_;  ///< column j -> its entries of src_ (ascending)
    std::vector<std::vector<Complex>> row_cache_, col_cache_;
    std::vector<Index> row_order_, col_order_;
};

/// One row-leaf chunk of the exact far apply (the WP-PML5a inner loop): columns p of the
/// position range [p0, p1) with stored column index j0 + (p - p0) of the block w (4 ns complex
/// per column), direct products into acc (4 ns reals: [Jr, Ji, Mr, Mi] blocks), transposed
/// products into u (4 reals per position) for the real positions.
void apply_exact_columns(const Complex* w, std::size_t ns, const Real* xs, const Real* xr, Index p0,
                         Index p1, std::size_t j0, Real* acc, Real* u,
                         const std::uint8_t* stretched_pos);

/// The ACA far ring (pml_ring.hpp): assembled in the constructor, applied like the exact ring
/// (fixed chunks of row leaves, thread-count independent).
class AcaFarRing {
public:
    /// @param partition from partition_far_ring(..., use_aca = true, ...)
    /// @param epsilon ACA tolerance (in (0, 1))
    /// @param drop_tolerance magnitude-bound threshold (in [0, 1); 0: nothing dropped)
    AcaFarRing(const op::Problem& problem, const std::vector<int>& colour,
               std::vector<FarRowLeaf> partition, const std::vector<Index>& perm,
               const std::vector<Index>& inv_perm, const std::vector<std::uint8_t>& stretched_pos,
               const PmlRingParams& params, Real epsilon, Real drop_tolerance);
    ~AcaFarRing();
    AcaFarRing(const AcaFarRing&) = delete;
    AcaFarRing& operator=(const AcaFarRing&) = delete;
    /// y += Z_far_ring x (x, y in basis order, size 2N).
    void apply(const VectorXc& x, VectorXc& y) const;
    [[nodiscard]] const PmlRingFarStats& stats() const;
    [[nodiscard]] const std::vector<PmlRingFarBlock>& blocks() const;
    [[nodiscard]] std::size_t bytes() const;  ///< factors + exact blocks + per-leaf data

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace specklebem::mlfmm::detail
