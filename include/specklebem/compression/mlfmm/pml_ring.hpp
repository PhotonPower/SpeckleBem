#pragma once
/// @file pml_ring.hpp
/// Exact PML ring Z_ring of the fast PML route (WP-PML5a; ADR 0008 amendment 2026-10-18,
/// decisions 1-3a; ADR 0009 decisions 2, 4 and 6).
///
/// With a non-identity PML stretch (op::Problem::stretch) the system matrix splits into
/// Z = Z_rr + Z_ring by the classification of the basis functions:
///  * a function is *stretched* iff one of its support triangles has a vertex strictly inside the
///    stretched zone (classify_stretched = op::pml_masked_functions, the right-hand-side mask:
///    the masked test functions are exactly the stretched ones); all others are *real*. A pair of
///    two real functions never touches a stretched triangle, so its entries are bitwise the
///    unstretched ones (WP-PML2) and go through the existing MLFMM (MlfmmOperator: near part and
///    expansion restricted to the real functions);
///  * Z_ring holds every pair (m, n) with m or n stretched: the rows of the stretched test
///    functions over all columns and the real rows over the stretched columns.
///
/// Organisation (ADR 0008 amendment decision 3): by the leaf boxes of the MLFMM octree (real
/// coordinates; the stretch is a map on the real mesh). A pair is *near* iff the leaves of m and n
/// coincide or touch (integer leaf coordinates differ by at most 1 per axis: the leaf and its
/// near_list, the rule of near_pattern), otherwise *far*.
///  * Near ring: the exact Galerkin entries of all four blocks (op::assemble_sparse_stretched on
///    the near pattern restricted to the pairs with a stretched member, split_near_pattern), with
///    the jump terms of the dense assembler (identical triangles); stored in an
///    op::SparseOperator (4 x 24 bytes per ordered pair).
///  * Far ring (WP-PML5a: exact, the correctness baseline; WP-PML5b replaces it by ACA): per leaf
///    A with stretched functions A_s, a dense block of the four sub-blocks JJ, JM, MJ, MM over the
///    rows A_s and the columns of every leaf B that is not near A (all of B's functions, in
///    octree order), i.e. one block per (A, B) box pair, stored contiguously per A (4 x 16 bytes
///    per stored pair, no indices). Far pairs share no triangle (r_max < leaf edge, checked by the
///    MLFMM far operator), so they carry no jump term and each sub-block of Z is complex
///    symmetric on them (L and K^PV are; the stretched kernels evaluate the canonical ordering and
///    transpose, kernels::element_blocks): the entries of a real row m over a stretched column n
///    are the stored entries of row n, column m, applied transposed. Each far pair is therefore
///    represented once; memory O(N_s N) (N_s stretched functions).
///  * Entries: kernels::element_blocks with the stretch through op::PairBlockEvaluator (the dense
///    assembler's per-triangle-pair blocks), summed per basis pair over the four triangle pairs in
///    the order (test triangle by colour, source triangle ascending). The stored entries equal
///    the DenseStrategy entries up to rounding (summation order). A transposed entry (real row
///    m, stretched column n, taken from the stored row n) equals the dense Z(m, n) up to
///    rounding where the triangle pairs of m and n share no vertex; a far pair can share a
///    vertex only if 2 r_max exceeds the leaf edge a (r_max = max_support_radius: shared
///    vertex => midpoint distance <= 2 r_max, far leaves => > a), i.e. with
///    automatic_leaf_size = false or under the d0 = 3 leaf rule (r_max / a up to 0.6; not at
///    d0 = 5, r_max / a <= 0.3), and its touching-pair rules are not exactly symmetric in the
///    two orderings: relative asymmetry < 1e-9 there. Assembly: OpenMP over the test triangles of a
///    greedy colouring (no two triangles of one colour share a stretched function),
///    schedule(dynamic, 1); the result does not depend on the thread count. Each triangle pair of a
///    stretched test triangle is evaluated in the ordering of its row: stretched-stretched pairs
///    whose both orderings are needed are evaluated twice (WP-PML5c caches the canonical blocks).
///  * Budget: the far bytes are estimated before any allocation (estimate_pml_ring) and checked
///    against PmlRingParams::max_far_bytes; exceeding it throws std::runtime_error.
///
/// apply(): y = Z_near_ring x + Z_far_ring x; the near part per row in storage order, the far
/// part in one pass over the stored blocks. The row leaves are split into at most 64 chunks of
/// consecutive leaves, balanced by stored entries (fixed by the ring's structure, not by the
/// thread count); OpenMP schedule(dynamic, 1) over the chunks: each row leaf's direct products
/// go to its own (stretched) rows, summed over its columns in storage order, and its transposed
/// products into the chunk's buffer (4N reals); the real rows then sum the chunk buffers in
/// chunk order. Every entry of y has a fixed summation order, so apply() is **thread-count
/// independent** (bitwise equal results for any number of OpenMP threads; unit test 1 / 2 / 3 /
/// max threads). Scratch per call: chunks x 32 N bytes (23 MB at N = 11 408). WP-PML5a review
/// W4 replaced per-thread buffers (deterministic only for a fixed thread count); a two-pass
/// variant (transposed products over the column leaves) cost 1.2-1.5x the ring apply, the
/// chunked one-pass matches the per-thread version (benchmarks/results/pml_ring.md). Safe for
/// concurrent calls (all scratch is per call).
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/linear_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specklebem::mlfmm {

/// Stretched basis functions (ADR 0008 amendment 2026-10-18 decision 1): entry n is 1 iff a support
/// triangle of n has a vertex strictly inside the stretched zone, else 0 (size N). By definition
/// identical to op::pml_masked_functions(space, stretch), the PML mask of the right-hand side.
[[nodiscard]] std::vector<std::uint8_t> classify_stretched(
    const basis::RwgSpace& space, const geometry::CoordinateStretch& stretch);

/// The MLFMM near pattern (near_pattern(tree, space)) split by a classification.
struct SplitNearPattern {
    /// Near pairs with both functions real (Z_rr's near part); stretched rows have no columns.
    op::BasisPattern real;
    /// Near pairs with a stretched member (the near ring): stretched rows with every near column,
    /// real rows with the stretched near columns.
    op::BasisPattern ring;
    Index real_pairs = 0;  ///< ordered basis pairs of `real`
    Index ring_pairs = 0;  ///< ordered basis pairs of `ring`
};

/// Splits near_pattern(tree, space) into its real-real and ring parts (two groups per near-pattern
/// group: the real rows and the stretched rows). The two parts partition the near pattern.
/// @param stretched size N, non-zero = stretched (classify_stretched)
/// @throws std::invalid_argument as near_pattern, or for a classification of the wrong size.
[[nodiscard]] SplitNearPattern split_near_pattern(const Octree& tree, const basis::RwgSpace& space,
                                                  const std::vector<std::uint8_t>& stretched);

struct PmlRingParams {
    /// Budget of the exact far ring blocks [bytes], checked against the pre-assembly estimate
    /// (estimate_pml_ring; memory model 64 N_s N_far bytes, N_s stretched functions, N_far the
    /// functions outside a row leaf's 27-leaf neighbourhood) before any allocation; exceeding it
    /// throws std::runtime_error. 0 = automatic: the dense matrix 16 (2N)^2 bytes. The exact
    /// ring of WP-PML5a stores at most N_s x N pairs at 64 bytes = 16 (2N) (2 N_s) bytes, never
    /// more than the dense matrix, so the automatic budget never fires: it only guards against
    /// inconsistent use, and an explicit value bounds the run. Independently of the budget,
    /// MlfmmOperator warns (warn_pml_ring_memory) when the estimate exceeds the available
    /// physical memory. SIZE_MAX disables the guard.
    std::size_t max_far_bytes = 0;
};

/// Pre-assembly sizes of the PML ring (estimate_pml_ring), also reported by PmlRingOperator.
struct PmlRingEstimate {
    Index n = 0;              ///< N
    Index stretched = 0;      ///< N_s
    Index row_leaves = 0;     ///< leaves with at least one stretched function
    Index near_pairs = 0;     ///< ordered near pairs with a stretched member
    Index far_stored = 0;     ///< stored far pairs: stretched rows x far columns
    Index far_pairs = 0;      ///< ordered far pairs with a stretched member (stored + transposed)
    Index rr_near_pairs = 0;  ///< ordered near pairs of two real functions (Z_rr's near part)
    std::size_t near_bytes = 0;   ///< near ring SparseOperator (4 x 24 bytes per pair + rows)
    std::size_t far_bytes = 0;    ///< far ring blocks (64 bytes per stored pair + per-leaf data)
    std::size_t budget = 0;       ///< far budget in effect (PmlRingParams::max_far_bytes resolved)
    std::size_t dense_bytes = 0;  ///< 16 (2N)^2, the dense matrix for comparison
};

/// The sizes of the ring of (tree, space, stretched) without assembling; O(N + leaves x 27).
/// @throws std::invalid_argument for a classification of the wrong size, a tree not built on
///         space, or a tree with elevated functions (local leaf rule: not supported by the ring).
[[nodiscard]] PmlRingEstimate estimate_pml_ring(const Octree& tree, const basis::RwgSpace& space,
                                                const std::vector<std::uint8_t>& stretched,
                                                const PmlRingParams& params = {});

/// @throws std::runtime_error if e.far_bytes > e.budget (message with the estimate and the
///         remedies); nothing otherwise.
void check_pml_ring_budget(const PmlRingEstimate& e, const PmlRingParams& params);

/// Logs one SBEM_WARN (a warning, not an error: the automatic budget is the dense matrix, so it
/// does not bound the run by the machine) if the estimated far ring bytes e.far_bytes exceed
/// `available` (MlfmmOperator passes core::available_memory_bytes(), before any assembly; 0 =
/// unknown: no warning). Returns true iff it warned.
bool warn_pml_ring_memory(const PmlRingEstimate& e, std::size_t available);

class PmlRingOperator final : public op::LinearOperator {
public:
    /// Classifies (classify_stretched), estimates and checks the far budget, assembles the near
    /// ring (assemble_sparse_stretched) and the far ring blocks. The tree is used during
    /// construction only (its permutation and leaf structure are copied).
    /// @param tree the MLFMM octree over problem.space (MlfmmOperator's), without elevated
    ///        functions (leaf_radius_quantile = 1)
    /// @throws std::invalid_argument for an invalid Problem (op::validate), a Problem without a
    ///         non-identity stretch, a tree not built on problem.space or with elevated functions;
    ///         std::runtime_error if the far estimate exceeds the budget (message with the
    ///         estimate and PmlRingParams::max_far_bytes); as kernels::element_blocks.
    PmlRingOperator(const op::Problem& problem, const Octree& tree,
                    const PmlRingParams& params = {});
    ~PmlRingOperator() override;

    [[nodiscard]] Index rows() const override;
    [[nodiscard]] Index cols() const override;
    /// y = Z_ring x (see the file comment; thread-count independent).
    /// @throws std::invalid_argument if x has the wrong size.
    void apply(const VectorXc& x, VectorXc& y) const override;
    /// Classification, near / far pairs, bytes and times.
    [[nodiscard]] std::string describe() const override;
    /// near_bytes() + far_bytes() + the copied octree data.
    [[nodiscard]] std::size_t memory_bytes() const override;

    [[nodiscard]] std::size_t near_bytes() const;  ///< near ring SparseOperator
    [[nodiscard]] std::size_t far_bytes() const;   ///< far ring blocks (allocated)
    [[nodiscard]] Index stretched_count() const;   ///< N_s
    [[nodiscard]] Index near_pairs() const;        ///< ordered near ring pairs
    [[nodiscard]] Index far_pairs() const;         ///< ordered far ring pairs (stored + transposed)
    [[nodiscard]] Index far_stored_pairs() const;  ///< stored far pairs
    [[nodiscard]] Real near_seconds() const;       ///< near ring assembly time [s]
    [[nodiscard]] Real far_seconds() const;        ///< far ring assembly time [s]
    [[nodiscard]] const PmlRingEstimate& estimate() const;
    /// The classification (size N, 1 = stretched).
    [[nodiscard]] const std::vector<std::uint8_t>& stretched() const;
    /// The near ring (2N x 2N; pattern = SplitNearPattern::ring).
    [[nodiscard]] const op::SparseOperator& near_operator() const;
    /// True iff the far ring represents the pair (m, n) (basis indices in [0, N)): m stretched and
    /// the leaves of m and n not near, or m real, n stretched and the leaves not near.
    /// @throws std::out_of_range for indices outside [0, N).
    [[nodiscard]] bool far_contains(Index m, Index n) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace specklebem::mlfmm
