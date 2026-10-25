#pragma once
/// @file pml_ring.hpp
/// PML ring Z_ring of the fast PML route (WP-PML5a exact, WP-PML5b ACA far blocks; ADR 0008
/// amendment 2026-10-18, decisions 1-3; ADR 0009 decisions 2, 4 and 6).
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
///    op::SparseOperator (4 x 24 bytes per ordered pair). WP-PML5c: each unordered stretched
///    triangle pair of the pattern is evaluated once (canonical-pair cache of
///    assemble_sparse_stretched, parallel over the pairs) and the touching pairs of the PML
///    strips once per translation class (op::StretchedPairEvaluation::translation_classes):
///    bitwise the per-ordering result of WP-PML5a; near_stats() reports the counts and phase
///    times.
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
///    whose both orderings are needed are evaluated twice in the far ring (the near ring evaluates
///    each unordered pair once since WP-PML5c).
///  * Budget: the far bytes are estimated before any allocation (estimate_pml_ring, once per
///    operator through plan_pml_ring together with the classification, the split near pattern and
///    the ACA far partition) and checked against PmlRingParams::max_far_bytes; exceeding it throws
///    std::runtime_error.
///  * ACA far ring (WP-PML5b, ADR 0008 amendment decision 3b; PmlRingParams::use_aca, the
///    default; use_aca = false is the exact far ring above). The far pairs are partitioned into
///    blocks (R, C) of a row cluster R (an octree box; rows R_s = its stretched functions) and a
///    column cluster C (an octree box, all its functions; detail::partition_far_ring,
///    plan_pml_ring_far). A pair is *admissible* if max(diam R_s, diam C) <= eta dist(R_s, C)
///    (PmlRingParams::eta; diam and dist of the axis-aligned bounding boxes of the support
///    vertices, real coordinates). With row_clusters (default) the partition is the H-matrix one: a
///    dual descent over same-level box pairs from the root; a pair of non-adjacent boxes (no two
///    of their leaves near) is a block if admissible or if both are leaves (then exact, as above),
///    otherwise its child pairs are visited; adjacent pairs are subdivided down to the leaves,
///    where they are near (near ring). Without row_clusters every row cluster is a leaf A and its
///    far columns are partitioned by a column descent (the coarsest admissible box, or a far leaf;
///    coarse_columns = false: leaves only). Every far pair (m in R_s, n) lies in exactly one
///    block; the columns of a block are contiguous in octree order. Admissible blocks: each of
///    the four sub-blocks JJ, JM, MJ, MM over R_s x C is compressed by aca::compress (partial
///    pivoting, relative tolerance epsilon = PmlRingParams::aca_epsilon or 10^-(d0+1)), rows and
///    columns generated from the same PairBlockEvaluator triangle-pair blocks and in the same
///    summation order as the exact path (detail::BlockSource: a block stored exactly holds the
///    5a entries bitwise unless the per-pair bound below drops a region part of one of its
///    triangle pairs, which happens only with a lossy region; without drops, with leaf rows
///    and no admissible block the ACA path's product is bitwise the exact ring's, unit test).
///    The four ACAs share one cache of generated function rows / columns (one
///    row gives all four sub-block rows) and start from the pivots of the previous ones. Rank cap
///    per sub-block: the break-even rank floor(m n / (m + n)) (m = |R_s|, n = |C|; or
///    PmlRingParams::max_rank); a sub-block that reaches it unconverged turns the whole block
///    exact (counted; its cached rows and columns are freed before the exact block is
///    allocated, and the exact entries are recomputed). With the automatic cap, a block whose
///    break-even rank is below kPmlRingMinAcaRank is stored exactly without an ACA attempt
///    (PmlRingFarBlock::aca_skipped). Triangle-pair memo (WP-PML5d, PmlRingParams::
///    pair_memo_bytes): the triangle-pair blocks evaluated for a generated row or column are kept
///    per block (within the budget) and reused by the other rows and columns that meet the same
///    pair and by a fallback's exact generation; no entry's summation order changes, so the ring
///    is bitwise the one without the memo (1.5x fewer kernel evaluations at the 7c size, 1.7x for
///    Si). Transient memory of the assembly (WP-PML5b review W3):
///    per thread, the block being built: the shared row / column cache of its ACAs (64 bytes per
///    generated row entry and column entry; at the break-even cap up to about the block's exact
///    bytes) plus the factors, or after a fallback the exact block alone, plus the pair memo
///    (at most pair_memo_bytes). At the 7c size the
///    fallback blocks are all <= 0.2 MB (pml_ring.md, 5b review); a large coarse block reaching
///    its cap would need up to its exact bytes once per thread (e.g. 5000 x 37 500 rows x
///    columns: 12 GB). Each sub-block is complex symmetric on far pairs as above, so the real
///    rows of C are served transposed from the same factors: (U V^T)^T x = V (U^T x).
///    Worst case (max_rank = 0): no block takes more than its exact bytes (plus ~0.5 kB of block
///    bookkeeping), so the ACA far ring never exceeds the exact estimate far_bytes; only the
///    non-admissible blocks (far_exact_bytes) are known before assembly and checked against the
///    budget. Magnitude bound (ADR 0008 §6, WP21 form): per block and region i, delta_i =
///    F (1 + alpha R*) exp(-alpha kappa R*) with alpha = -Im k_i, R* = max(d, (1 - kappa) /
///    (alpha kappa)), d = dist(R_s, C), kappa = 2 / sqrt(4 + C_s^2), F = (1 + C_s^2)^2 / kappa^3,
///    C_s = the largest sigma_max of the stretch (pml_ring_decay_bound): with the stretch the
///    complex distance has Re R~ >= kappa R (the secant slopes of the stretch lie in [0, C_s];
///    unit-tested by sampling), so |G~| and |grad G~| (with the Jacobian and Piola factors, each
///    <= sqrt(1 + C_s^2)) stay below delta_i times their undamped bounds; delta_i <= 10^-(d0+1)
///    drops region i of the block (both regions: the block is dropped), counted and logged with
///    the bound (SBEM_DEBUG per block, one SBEM_INFO summary); a vacuum region (alpha = 0) is
///    never dropped. The same bound per triangle pair (vertex extents, as the WP21 exact parts per
///    basis pair): inside a block that keeps region i, the region-i part of a triangle pair
///    farther apart than PmlRingFarStats::drop_distance[i] is not evaluated (a pair left without
///    a region is skipped), counted (pair_reductions). Assembly: OpenMP over the row clusters
///    (schedule(dynamic, 1)), each cluster's blocks serially: the result does not depend on the
///    thread count. Ranks, bytes and drops: far_stats(), far_blocks(), describe().
///
/// apply(): y = Z_near_ring x + Z_far_ring x; the near part per row in storage order, the far
/// part in one pass over the stored blocks (ACA path: per row cluster its blocks in column order,
/// exact blocks as below, low-rank sub-blocks as U (V^T x_C) directly and V (U^T x_R)
/// transposed; its direct products go through the chunk buffers too, since a stretched row
/// belongs to row clusters of several levels, and every position sums the chunk buffers in chunk
/// order). The row leaves are split into at most 64 chunks of
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

#include <array>
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

/// Smallest break-even rank floor(m n / (m + n)) for which the ACA far ring attempts a
/// compression under the automatic rank cap (PmlRingParams::max_rank = 0; WP-PML5b review W3).
/// With CompressParams::confirmations = 2 an ACA capped at rank 3 converges only for sub-blocks
/// of numerical rank <= 1, and JJ / MM never are at the ring's tolerances: at the 7c size all 884
/// admissible blocks with break-even rank 2-3 ended at the cap (attempt + exact generation),
/// while 176 of 300 with break-even 4-7 compressed. Smaller blocks are stored exactly
/// directly; the stored ring is unchanged where every such attempt failed.
inline constexpr Index kPmlRingMinAcaRank = 4;

struct PmlRingParams {
    /// Budget of the exact far ring blocks [bytes], checked against the pre-assembly estimate
    /// (estimate_pml_ring; memory model 64 N_s N_far bytes, N_s stretched functions, N_far the
    /// functions outside a row leaf's 27-leaf neighbourhood) before any allocation; exceeding it
    /// throws std::runtime_error. 0 = automatic: the dense matrix 16 (2N)^2 bytes. The exact
    /// ring of WP-PML5a stores at most N_s x N pairs at 64 bytes = 16 (2N) (2 N_s) bytes, never
    /// more than the dense matrix, so the automatic budget never fires: it only guards against
    /// inconsistent use, and an explicit value bounds the run. Independently of the budget,
    /// MlfmmOperator warns (warn_pml_ring_memory) when the estimated peak of the ring known before
    /// assembly (near ring, its assembly cache, the far blocks stored exactly) exceeds the
    /// available physical memory. SIZE_MAX disables the guard. With use_aca the budget bounds the
    /// exact (non-admissible) blocks known before assembly (PmlRingEstimate::far_exact_bytes); the
    /// ACA blocks are bounded by their rank caps (never more than their exact bytes); if the
    /// assembled blocks (exact + ACA factors) end above the budget, one SBEM_WARN says so.
    std::size_t max_far_bytes = 0;
    /// WP-PML5b: true (default) compresses the admissible far blocks by ACA; false keeps the
    /// exact far ring of WP-PML5a (the correctness baseline of the comparison tests).
    bool use_aca = true;
    /// Relative ACA tolerance per sub-block, in (0, 1); 0 = automatic 10^-(d0 + 1) (d0 = the
    /// MLFMM accuracy digits). The magnitude-bound drop threshold stays 10^-(d0 + 1).
    Real aca_epsilon = 0.0;
    /// Admissibility max(diam R_s, diam C) <= eta dist(R_s, C) (> 0; file comment). Default 2.0
    /// (benchmarks/results/pml_ring.md, "5b": at the 7c size eta 0.5 / 1 / 2 / 3 give 1260 / 819 /
    /// 571 / 524 MB of far ring, max rank 19 / 28 / 37 / 39; 2 keeps the ranks moderate).
    Real eta = 2.0;
    /// Rank cap per sub-block (>= 0); 0 = the break-even rank floor(m n / (m + n)) of each block,
    /// so that no ACA block takes more memory than its exact storage (the worst case of the file
    /// comment). A positive value replaces it (diagnostics and tests: above the break-even a block
    /// can exceed its exact bytes). A sub-block that reaches the cap unconverged makes its block
    /// exact (a rank-cap fallback).
    Index max_rank = 0;
    /// true: column clusters are the coarsest admissible octree boxes; false: single leaves
    /// (row leaves only, row_clusters = false).
    bool coarse_columns = true;
    /// true: row clusters of every level (an H-matrix block partition of the ring rows: dual
    /// descent over same-level box pairs, see the file comment); false: row leaves only (each
    /// leaf's stretched functions against column clusters).
    bool row_clusters = true;
    /// Diagnostics of a standalone PmlRingOperator only (comparison runs): false skips the near
    /// ring (apply() is the far ring only, near_operator() empty). MlfmmOperator (and hence
    /// Simulation) refuses false with std::invalid_argument: an operator without its near ring
    /// solves a different system and has no stretched diagonal (WP-PML5b review W1).
    bool assemble_near = true;
    /// WP-PML5d: bytes of the triangle-pair memo per ACA far block (one block per assembling
    /// thread at a time, so at most this per thread; detail::BlockSource): the evaluated pair
    /// blocks are kept in strips per test / source triangle while they fit, so that rows sharing
    /// a test triangle, columns sharing a source triangle, row / column overlaps and the exact
    /// generation of a rank-cap fallback reuse them instead of evaluating the pair again. The
    /// stored ring is bitwise the same for every value (unit test). 0 disables the memo. Default
    /// 16 MiB (benchmarks/results/pml_ring.md, "5d: pair memo": at the 7c size it holds every
    /// block of the two finest far levels completely (largest 9.9 MB) and the coarsest ones in
    /// part: 13.0 M kernel evaluations instead of 18.9 M, an unlimited memo 12.7 M with blocks of
    /// up to 46 MB; at 2N = 59 600 41.4 M instead of 59.1 M, unlimited 39.3 M with up to 131 MB).
    std::size_t pair_memo_bytes = std::size_t{16} << 20;
};

/// One block (R, C) of the ACA far ring (PmlRingOperator::far_blocks; WP-PML5b): rows R_s (the
/// stretched functions of the row cluster R, an octree box: a leaf, or with
/// PmlRingParams::row_clusters any level), columns all functions of the octree box C.
struct PmlRingFarBlock {
    enum class Kind { aca, exact, fallback, dropped };
    Index row_box = -1;  ///< row cluster R (index into Octree::boxes()): a leaf A, or a coarser box
    int row_level = 0;   ///< octree level of R
    Index box = -1;      ///< column cluster C (index into Octree::boxes())
    int level = 0;       ///< octree level of C
    Index first = 0;     ///< first column position of C (octree order)
    Index count = 0;     ///< |C|
    Index rows = 0;      ///< |R_s|
    bool admissible = false;
    Real distance = 0;      ///< dist(R_s, C) of the support extents [m]
    Real row_diameter = 0;  ///< diam R_s [m]
    Real col_diameter = 0;  ///< diam C [m]
    /// aca: four low-rank sub-blocks; exact: non-admissible, stored exactly; fallback:
    /// admissible but stored exactly (a sub-block reached its rank cap, or, under the automatic
    /// cap, the break-even rank is below kPmlRingMinAcaRank: aca_skipped); dropped: every region
    /// below the magnitude bound, not stored.
    Kind kind = Kind::exact;
    std::array<Index, 4> rank = {0, 0, 0, 0};   ///< ACA ranks of JJ, JM, MJ, MM (kind aca)
    std::array<bool, 2> region = {true, true};  ///< region parts kept (false: dropped)
    std::array<Real, 2> bound = {1.0, 1.0};     ///< magnitude bound delta_i per region
    Index generated_rows = 0;     ///< function rows generated by the ACAs (admissible blocks)
    Index generated_columns = 0;  ///< function columns generated by the ACAs
    Index pair_evaluations = 0;   ///< kernel evaluations of triangle pairs for the block
    /// WP-PML5d: triangle-pair blocks taken from the pair memo instead of being evaluated again
    /// (pair_evaluations + pair_memo_hits = the evaluations without the memo)
    Index pair_memo_hits = 0;
    /// distinct triangle pairs evaluated (a pair met by several rows, columns or the exact
    /// generation counts once; = pair_evaluations if the memo held every strip of the block)
    Index pair_distinct = 0;
    /// distinct triangle pairs with a region dropped by the per-pair bound (incl. pair_skips)
    Index pair_reductions = 0;
    Index pair_skips = 0;        ///< of them, distinct pairs with no region left (not evaluated)
    std::size_t memo_bytes = 0;  ///< bytes of the block's pair memo (<= pair_memo_bytes)
    bool memo_limited = false;   ///< a memo strip did not fit into pair_memo_bytes
    /// Kind::fallback without an ACA attempt: the automatic break-even rank cap is below
    /// kPmlRingMinAcaRank (incl. 0 for a single row).
    bool aca_skipped = false;
};

/// The magnitude bound of the ACA far ring (file comment, "Magnitude bound"): delta =
/// F (1 + alpha R*) exp(-alpha kappa R*), kappa = 2 / sqrt(4 + sigma_max^2), F = (1 +
/// sigma_max^2)^2 / kappa^3, R* = max(distance, (1 - kappa) / (alpha kappa)); >= 1 (never below
/// a drop threshold) for alpha <= 0 or distance <= 0.
/// @param alpha -Im k of the region [1/m]
/// @param distance lower bound of |r - r'| over the block (real coordinates) [m]
/// @param sigma_max the largest sigma_max of the stretch profiles (>= 0)
[[nodiscard]] Real pml_ring_decay_bound(Real alpha, Real distance, Real sigma_max);

/// Far ring statistics after assembly (PmlRingOperator::far_stats; WP-PML5b).
struct PmlRingFarStats {
    bool aca = false;           ///< PmlRingParams::use_aca
    Real epsilon = 0;           ///< ACA tolerance in effect (ACA path)
    Real drop_tolerance = 0;    ///< magnitude-bound threshold 10^-(d0 + 1) (ACA path)
    Real eta = 0;               ///< admissibility parameter (ACA path)
    Index aca_blocks = 0;       ///< admissible blocks stored as four low-rank sub-blocks
    Index exact_blocks = 0;     ///< non-admissible far leaf blocks (exact; 5a: one per row leaf)
    Index fallback_blocks = 0;  ///< admissible blocks stored exactly (Kind::fallback)
    /// of them, stored exactly without an ACA attempt (break-even rank < kPmlRingMinAcaRank)
    Index skipped_attempts = 0;
    Index dropped_blocks = 0;  ///< blocks dropped entirely (every region below the bound)
    std::array<Index, 2> region_dropped = {0, 0};  ///< blocks with region i dropped (incl. above)
    Real max_dropped_bound = 0;                    ///< largest delta_i of a dropped region part
    Index aca_pairs = 0;                           ///< stored pairs represented by ACA blocks
    Index exact_pairs = 0;        ///< stored pairs held exactly (incl. fallback blocks)
    Index dropped_pairs = 0;      ///< stored pairs of dropped blocks
    std::size_t aca_bytes = 0;    ///< low-rank factors (16 k (m + n) per sub-block)
    std::size_t exact_bytes = 0;  ///< exact blocks (64 bytes per pair)
    /// Ranks of all ACA sub-blocks (4 per ACA block): min, median, max; mean per sub-block
    /// (JJ, JM, MJ, MM); histogram over [0], [1], [2, 3], [4, 7], ... (bin b >= 1: [2^(b-1),
    /// 2^b - 1]).
    Index rank_min = 0, rank_median = 0, rank_max = 0;
    std::array<Real, 4> rank_mean = {0, 0, 0, 0};
    std::vector<Index> rank_histogram;
    Index generated_rows = 0;     ///< ACA function rows generated (all four sub-blocks each)
    Index generated_columns = 0;  ///< ACA function columns generated
    /// Per-triangle-pair bound (file comment): region i of a triangle pair farther apart than
    /// drop_distance[i] [m] is dropped (1e300: never, e.g. vacuum).
    std::array<Real, 2> drop_distance = {1e300, 1e300};
    Index pair_evaluations = 0;  ///< kernel evaluations of triangle pairs in the far ring assembly
    /// WP-PML5d: pair blocks served by the pair memo (PmlRingParams::pair_memo_bytes) instead of
    /// a kernel evaluation; pair_evaluations + pair_memo_hits = the evaluations without the memo.
    Index pair_memo_hits = 0;
    /// Distinct triangle pairs evaluated, summed over the blocks (WP-PML5d step 0); the
    /// evaluations without the memo / pair_distinct is the repetition of the ACA entry generation
    /// (rows sharing a test triangle, columns sharing a source triangle, row / column overlaps,
    /// rank-cap fallbacks; 1.5 at the 7c size).
    Index pair_distinct = 0;
    /// Distinct triangle pairs with a region dropped by the per-pair bound: the evaluated ones
    /// with one region left plus the skipped ones (pair_skips). Distinct visited pairs =
    /// pair_distinct + pair_skips. Logged (SBEM_INFO summary) and in Simulation::report().
    Index pair_reductions = 0;
    Index pair_skips = 0;         ///< distinct triangle pairs with no region left (not evaluated)
    std::size_t memo_budget = 0;  ///< PmlRingParams::pair_memo_bytes in effect
    std::size_t memo_bytes_max = 0;  ///< largest pair memo of a block (<= memo_budget)
    Index memo_limited_blocks = 0;   ///< blocks whose memo hit the budget (some strips not kept)
    /// bound of the memo bytes alive at once during the assembly: min(threads, row clusters) x
    /// memo_bytes_max (PmlRingOperator::peak_bytes)
    std::size_t memo_peak_bytes = 0;
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
    std::size_t near_bytes = 0;  ///< near ring SparseOperator (4 x 24 bytes per pair + rows)
    /// far ring blocks stored exactly (64 bytes per stored pair + per-leaf data): the WP-PML5a
    /// ring, and the upper bound of the ACA ring (every ACA block <= its exact bytes)
    std::size_t far_bytes = 0;
    /// the part known to be exact before assembly, checked against the budget: far_bytes without
    /// ACA, the non-admissible far leaf blocks with PmlRingParams::use_aca
    std::size_t far_exact_bytes = 0;
    Index far_aca_blocks = 0;    ///< use_aca: admissible blocks
    Index far_exact_blocks = 0;  ///< use_aca: non-admissible far leaf blocks
    Index far_aca_pairs = 0;     ///< use_aca: stored pairs inside admissible blocks
    /// Transient bytes of the near-ring assembly (WP-PML5c review S1): the canonical-pair cache of
    /// op::assemble_sparse_stretched (576 bytes per unordered stretched triangle pair of the ring
    /// pattern + index, order and translation-class data; op::estimate_sparse_stretched_cache, an
    /// upper bound of SparseStretchedStats::cache_bytes; 311 MB at the 7c size). Needs the
    /// stretch: set by plan_pml_ring (and hence by every PmlRingOperator), 0 from
    /// estimate_pml_ring alone. Freed before the far ring is allocated.
    std::size_t near_assembly_bytes = 0;
    /// Transient bytes of the triangle-pair memo of the ACA far-ring assembly (WP-PML5d): at most
    /// PmlRingParams::pair_memo_bytes per block being built, i.e. per assembling thread
    /// (omp_get_max_threads(), at most the number of row clusters); 0 without ACA. Alive while the
    /// far blocks are assembled (after the near-ring cache is freed).
    std::size_t far_memo_bytes = 0;
    std::size_t budget = 0;       ///< far budget in effect (PmlRingParams::max_far_bytes resolved)
    std::size_t dense_bytes = 0;  ///< 16 (2N)^2, the dense matrix for comparison
};

/// The sizes of the ring of (tree, space, stretched) without assembling; O(N + leaves x 27).
/// With params.use_aca the far partition (admissible / exact blocks) is computed as well
/// (O(row leaves x visited boxes)).
/// @throws std::invalid_argument for a classification of the wrong size, a tree not built on
///         space, or a tree with elevated functions (local leaf rule: not supported by the ring),
///         or invalid ACA parameters (eta <= 0, aca_epsilon outside [0, 1), max_rank < 0).
[[nodiscard]] PmlRingEstimate estimate_pml_ring(const Octree& tree, const basis::RwgSpace& space,
                                                const std::vector<std::uint8_t>& stretched,
                                                const PmlRingParams& params = {});

/// The block partition of the ACA far ring without assembling (WP-PML5b): every block's geometry
/// and admissibility (kind aca if admissible, exact otherwise; ranks, regions and bounds unset),
/// in the order of PmlRingOperator::far_blocks. Empty with params.use_aca = false.
/// @throws as estimate_pml_ring.
[[nodiscard]] std::vector<PmlRingFarBlock> plan_pml_ring_far(
    const Octree& tree, const basis::RwgSpace& space, const std::vector<std::uint8_t>& stretched,
    const PmlRingParams& params = {});

/// @throws std::runtime_error if e.far_exact_bytes > e.budget (message with the estimate and the
///         remedies); nothing otherwise.
void check_pml_ring_budget(const PmlRingEstimate& e, const PmlRingParams& params);

/// Peak bytes of the ring during its construction as far as known before assembly: the near ring
/// plus the larger of its transient assembly cache and the far blocks stored exactly with the
/// far-ring pair memo (the cache is freed before the far blocks are allocated): e.near_bytes +
/// max(e.near_assembly_bytes, e.far_exact_bytes + e.far_memo_bytes). Without ACA far_exact_bytes =
/// far_bytes (the WP-PML5c form); with ACA the compressed factors (at most far_bytes -
/// far_exact_bytes, known only after assembly) come on top of far_exact_bytes;
/// PmlRingOperator::peak_bytes() reports the measured peak.
[[nodiscard]] std::size_t pml_ring_peak_bytes(const PmlRingEstimate& e);

/// Logs one SBEM_WARN (a warning, not an error: the automatic budget is the dense matrix, so it
/// does not bound the run by the machine) if the estimated peak of the ring known before assembly,
/// pml_ring_peak_bytes(e) (near ring, near-ring assembly cache, far blocks stored exactly:
/// e.far_exact_bytes, = e.far_bytes without ACA), exceeds `available` (MlfmmOperator passes
/// core::available_memory_bytes(), before any assembly; 0 = unknown: no warning). Returns true
/// iff it warned. (With ACA the compressed blocks are not known before assembly; their upper
/// bound far_bytes - far_exact_bytes is logged by the operator, SBEM_INFO.)
bool warn_pml_ring_memory(const PmlRingEstimate& e, std::size_t available);

/// Identity of the octree a PmlRingPlan was made on (WP-PML5c review W3): the near-ring pattern
/// of the plan must match the far ring's near ranges, which PmlRingOperator takes from its tree.
struct PmlRingTreeFingerprint {
    int levels = 0;          ///< Octree::levels()
    Index leaves = 0;        ///< boxes on the leaf level
    Real leaf_size = 0.0;    ///< Octree::box_size(leaf_level())
    std::uint64_t hash = 0;  ///< FNV-1a of the permutation, root centre and per leaf (ijk, range,
                             ///< near_list)
    bool operator==(const PmlRingTreeFingerprint&) const = default;
};

/// The fingerprint of `tree`; O(N + leaves x 27).
[[nodiscard]] PmlRingTreeFingerprint pml_ring_tree_fingerprint(const Octree& tree);

/// The pattern computations of the PML ring, done once per operator and shared by MlfmmOperator
/// (budget check, real-real near part) and PmlRingOperator (near ring, far blocks) (WP-PML5c;
/// with the ACA far partition of WP-PML5b).
struct PmlRingPlan {
    std::vector<std::uint8_t> stretched;  ///< classify_stretched(space, stretch)
    SplitNearPattern split;               ///< split_near_pattern(tree, space, stretched)
    PmlRingEstimate estimate;             ///< estimate_pml_ring(tree, space, stretched, params)
    /// plan_pml_ring_far(tree, space, stretched, params), from the same pass as the estimate
    /// (empty with use_aca = false): the block partition of the ACA far ring.
    std::vector<PmlRingFarBlock> far_blocks;
    /// The parameters the plan was made with; PmlRingOperator refuses a plan whose far-partition
    /// parameters (use_aca, eta, coarse_columns, row_clusters) differ from its own.
    PmlRingParams params;
    PmlRingTreeFingerprint tree;  ///< pml_ring_tree_fingerprint(tree)
};

/// classify_stretched, split_near_pattern, estimate_pml_ring and (with params.use_aca)
/// plan_pml_ring_far in one call (each once; the far partition and the estimate in one pass), and
/// estimate.near_assembly_bytes from op::estimate_sparse_stretched_cache on the ring pattern (an
/// index pass over its triangle pairs, no kernel evaluation).
/// @throws as these.
[[nodiscard]] PmlRingPlan plan_pml_ring(const Octree& tree, const basis::RwgSpace& space,
                                        const geometry::CoordinateStretch& stretch,
                                        const PmlRingParams& params = {});

class PmlRingOperator final : public op::LinearOperator {
public:
    /// The ring from a precomputed plan (WP-PML5c; MlfmmOperator passes the plan it made for its
    /// budget check and real-real near part): the classification, the near-ring pattern
    /// plan.split.ring, the estimate and the ACA far partition plan.far_blocks are taken over
    /// (plan.split.real is not used and may be empty); the budget is resolved from `params` and
    /// checked as below. Otherwise as the constructor below, with identical results.
    /// @throws std::invalid_argument as below, and if plan.stretched differs from
    ///         classify_stretched(*problem.space, *problem.stretch), the plan's estimate was
    ///         made for another space (estimate.n != N), the plan was made on another octree
    ///         (plan.tree != pml_ring_tree_fingerprint(tree)) or with other far-partition
    ///         parameters (plan.params vs params: use_aca, eta, coarse_columns, row_clusters), or
    ///         its far partition does not match the tree; std::runtime_error as below.
    PmlRingOperator(const op::Problem& problem, const Octree& tree, PmlRingPlan plan,
                    const PmlRingParams& params = {}, Real accuracy_digits = 3.0);
    /// Classifies (classify_stretched), estimates and checks the far budget, assembles the near
    /// ring (assemble_sparse_stretched) and the far ring blocks (exact or ACA). The tree is used
    /// during construction only (its permutation and leaf structure are copied).
    /// @param tree the MLFMM octree over problem.space (MlfmmOperator's), without elevated
    ///        functions (leaf_radius_quantile = 1)
    /// @param accuracy_digits d0 of the MLFMM (> 0): the automatic ACA tolerance and drop
    ///        threshold 10^-(d0 + 1) (MlfmmOperator passes MlfmmParams::accuracy_digits)
    /// @throws std::invalid_argument for an invalid Problem (op::validate), a Problem without a
    ///         non-identity stretch, a tree not built on problem.space or with elevated functions,
    ///         invalid ACA parameters or accuracy_digits; std::runtime_error if the far estimate
    ///         exceeds the budget (message with the estimate and PmlRingParams::max_far_bytes);
    ///         as kernels::element_blocks.
    PmlRingOperator(const op::Problem& problem, const Octree& tree,
                    const PmlRingParams& params = {}, Real accuracy_digits = 3.0);
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
    /// far ring blocks (allocated: exact blocks + ACA factors + per-leaf data)
    [[nodiscard]] std::size_t far_bytes() const;
    /// Blocks, ranks, drops and bytes of the far ring (ACA path; exact path: exact bytes only).
    [[nodiscard]] const PmlRingFarStats& far_stats() const;
    /// Every block of the ACA far ring, grouped by row cluster: with row_clusters the clusters by
    /// octree level (coarsest first), then by first position; without, the row leaves in leaf
    /// order; each cluster's blocks by ascending first column. Empty on the exact path.
    [[nodiscard]] const std::vector<PmlRingFarBlock>& far_blocks() const;
    [[nodiscard]] Index stretched_count() const;   ///< N_s
    [[nodiscard]] Index near_pairs() const;        ///< ordered near ring pairs
    [[nodiscard]] Index far_pairs() const;         ///< ordered far ring pairs (stored + transposed)
    [[nodiscard]] Index far_stored_pairs() const;  ///< stored far pairs
    [[nodiscard]] Real near_seconds() const;       ///< near ring assembly time [s]
    [[nodiscard]] Real far_seconds() const;        ///< far ring assembly time [s]
    /// Counters and phase times of the near-ring assembly (op::assemble_sparse_stretched).
    [[nodiscard]] const op::SparseStretchedStats& near_stats() const;
    /// Measured peak of the ring's storage during construction: near_bytes() plus the larger of
    /// the near-ring assembly cache (near_stats().cache_bytes) and far_bytes() (exact blocks + ACA
    /// factors) plus the far-ring pair memos alive at once (far_stats().memo_peak_bytes,
    /// WP-PML5d); the post-assembly counterpart of pml_ring_peak_bytes(estimate()).
    [[nodiscard]] std::size_t peak_bytes() const;
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
