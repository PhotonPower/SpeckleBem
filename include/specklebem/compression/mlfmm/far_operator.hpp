#pragma once
/// @file far_operator.hpp
/// Multilevel fast multipole far part Z_far of the combined 2N x 2N system (WP20a, WP21,
/// docs/04_theory_mlfmm.md "Passes per matvec", ADR 0008 and its amendments). Z_far holds every
/// leaf pair that is neither the same leaf nor in its near list (octree.hpp); WP20b adds the exact
/// near part.
///
/// Blocks (docs/03, operator/assembler.hpp) with e_i = a_i/eta_i, h_i = b_i eta_i, m_i =
/// b_i/eta_i and the far-field forms of patterns.hpp (c_L = w mu_i k_i / 16 pi^2, c_K =
/// k_i^2 / 16 pi^2):
///   y_J = sum_i e_i (L_i x_J - K_i x_M),   y_M = sum_i (h_i K_i x_J + m_i L_i x_M).
/// L and K differ only in the source pattern (V vs W = khat x V), and khat x commutes with
/// interpolation, phase shifts and translation, so each region aggregates two vector fields,
/// F_J = sum_n x_J,n V_n and F_M = sum_n x_M,n V_n, and applies khat x at reception:
///   y_J,m += sum_q R_m . (e c_L G_J - e c_K khat x G_M),
///   y_M,m += sum_q R_m . (h c_K khat x G_J + m c_L G_M),
/// with G_J, G_M the incoming (translated) fields of the leaf box of m.
///
/// Passes per region (levels 2 ... leaf; levels 0 and 1 have no interaction lists):
///  1. Leaf aggregation: F_box = sum over the box's bases of x_n V_n (relative to the leaf centre).
///  2. Upward: F_parent = sum_children e^{+jk khat.(c_child - c_parent)} I F_child, I the Lagrange
///     interpolation child -> parent sampling (odd pole parity for the theta/phi components;
///     identity if both samplings coincide). Sign: the radiation pattern carries
///     e^{+jk khat.(r' - c)} and r' - c_parent = (r' - c_child) + (c_child - c_parent).
///  3. Translation: G_A = sum_{B in interaction_list(A)} w_q T_L(k, c_A - c_B, khat_q) F_B: the
///     incoming fields are *weighted* (they carry the quadrature weights w_q of their level).
///  4. Downward: G_child += I^T (e^{-jk khat.(c_child - c_parent)} G_parent). The receiving pattern
///     carries e^{-jk khat.(r - c)}, so the shift to the child centre is e^{-jk khat.(c_child -
///     c_parent)} at the parent directions, and sum_p w_p g_p (I R)_p = sum_c R_c (I^T (w g))_c:
///     the parent weights are already in G (interpolation.hpp weight convention), no child weights.
///  5. Reception: the unweighted sum over the leaf directions above.
///
/// Orders per region and level: search_truncation_order with the enlarged diagonal sqrt(3) a +
/// 2 max_support_radius (leaf level: leaf_sampling, sampled at max(L, p - 1)); translators are the
/// order-truncated T_L at the level's sampling, premultiplied by the weights, one per integer
/// offset present in the interaction lists. A region whose weights are all zero is skipped.
///
/// Region policy (ADR 0008 §6 with its amendments, WP21), per active region i and level l, from
/// the leaf upwards; once a level does not use the expansion, no coarser level does:
///  1. expansion: the order search (statistical pre-screen) is achievable and the block check
///     passes. Block check: sampled far blocks (basis_patterns about the level's box centres,
///     the level's sampling and order) of up to kBlockCheckPairs box pairs of the nearest
///     interaction offset classes against exact element_blocks entries (target 0.01 x 10^-d0,
///     no decay-aware relaxation); per box the bases nearest to the partner box (the facing,
///     corner-near ones, the worst case of WP19b) and a spread of the rest; error = max over the
///     pairs of the relative Frobenius errors
///     of the L and K blocks. It always runs and is reported; it decides only for lossy regions
///     (Im k < 0): accepted if <= block_check_tolerance(d0) = 10^-d0, or if <=
///     kLossDegradationFactor x the block error of the lossless analogue (k = Re k_i, its own
///     order search, which must be achievable) — the loss must not spoil the expansion beyond
///     what a lossless medium of the same geometry has. For lossless regions the ADR 0008
///     WP20a/WP20b amendments make the docs/05 matvec criterion the acceptance test (the nearest
///     corner-near blocks exceed 10^-d0 there, e.g. 9e-4 / 4e-4 at lambda/2 leaves for d0 =
///     3 / 5, while the matvec errors are 2e-4 / 4e-6).
///  2. otherwise, per ordered box pair (A, B in A's interaction list) of the level, with alpha =
///     -Im k_i and d = the distance between the bounding boxes of the supports (all triangle
///     vertices of the boxes' basis functions; a lower bound of |r - r'| over the pair):
///       delta(A, B) = (1 + alpha d) exp(-alpha d)   (1 for d <= 0),
///     the factor by which |G| and |grad G| of region i stay below their undamped bounds
///     1 / (4 pi R) and (1 + |Re k| R) / (4 pi R^2) on the pair (the WP-P2 reasoning of
///     kernels::OperatorOptions::decay_aware_target; (1 + x) e^-x is decreasing). Every region-i
///     entry of the pair is at most delta times the undamped bound U of the same entry, so
///     dropping the pair changes each block by at most delta ||U||:
///       truncation if delta <= 10^-(d0+1): a documented numerical truncation one digit below the
///         error a lossless expansion of the same geometry may have (10^-d0 ||B|| <= 10^-d0 ||U||);
///       exact otherwise: the pair's region-i entries L_i, K_i are assembled exactly
///         (op::assemble_region_sparse) into a per-region sparse correction (exact_part(i),
///         40 bytes per basis pair: L_i and K_i, combined with the block weights e_i, h_i, m_i in
///         apply()). Inside an exact box pair the same bound is applied per basis pair (d =
///         distance of the two supports' bounding boxes): basis pairs with delta <= 10^-(d0+1)
///         are truncated as well (counted in describe()).
///     The level's decision is FarDecision::truncation if no pair is exact, else
///     FarDecision::exact (with the counts of both kinds).
/// The exact fallback needs strong decay. A pair is truncated once alpha d >= x*(d0), the root of
/// (1 + x) e^-x = 10^-(d0+1) (truncation_decay_exponent: 11.76 for d0 = 3, 16.65 for d0 = 5), so
/// the exact pairs reach x* / alpha in support distance. The fallback is allowed only if x* /
/// alpha <= kExactFallbackBoxEdges a_l at the first level l without expansion (the exact pairs
/// then stay within a few boxes of that level); otherwise the constructor throws
/// TruncationOrderError (causes lossy_region / digits / mesh_or_leaf_size). The size of the exact
/// parts is estimated before any allocation: per region the upper bound sum n_A n_B over the
/// exact box pairs (n = bases per box), logged; if the bound of both regions exceeds the budget
/// (MlfmmParams::max_exact_far_bytes, 40 bytes per pair), the per-basis-pair count (the exact
/// nnz, without allocating) decides, and TruncationOrderError with cause exact_part_too_large is
/// thrown if it exceeds the budget too. MlfmmParams::exact_far_regions forces the exact decision
/// on every level of a region (budget included). Each leaf pair is far through exactly one level,
/// so per region the near part, the expansion levels, the exact part and the truncated pairs
/// partition the basis pairs. Every non-expansion decision is logged (SBEM_INFO) with its bound
/// and counts, and describe() lists every level's decision, search error, block error, bound and
/// counts: a pair is dropped only with delta <= 10^-(d0+1), never silently.
///
/// Jump terms: Z_far has none (the dense K carries -/+ 1/2 on coincident triangles only), which
/// needs r_max < a_leaf: far pairs have midpoints more than a_leaf apart, bases sharing a triangle
/// at most r_max (checked by the constructor).
///
/// apply() is safe for concurrent calls (WP20b): the pass storage (outgoing and incoming fields of
/// every level, workspace_bytes()) comes from a mutex-guarded pool of workspaces; a call takes a
/// free one or allocates a new one and returns it afterwards, so serial calls (GMRES) reuse one
/// workspace and k concurrent calls hold k. The storage is not zero-filled up front: every pass
/// zeroes the box segments it accumulates into, by the thread that owns the box. OpenMP over boxes
/// in every pass, each box written by one thread in a fixed order: the result is bitwise identical
/// for any thread count.
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/linear_operator.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace specklebem::op {
class RegionSparseOperator;
}

namespace specklebem::mlfmm {

/// Box pairs of a level used by the block check (one per nearest offset class).
inline constexpr int kBlockCheckPairs = 3;
/// Basis functions per box in the block check: all if the box has at most this many, else the
/// half nearest to the partner box (edge midpoint to the partner cube: the facing, corner-near
/// bases, the worst case of WP19b) and an evenly spaced selection of the rest.
inline constexpr int kBlockCheckBases = 24;

/// x*(d0): the root of (1 + x) e^-x = 10^-(d0+1), i.e. the decay exponent alpha d beyond which
/// the region policy truncates a pair (11.76 for d0 = 3, 16.65 for d0 = 5).
/// @throws std::invalid_argument for digits <= 0 or non-finite digits.
[[nodiscard]] Real truncation_decay_exponent(Real digits);
/// The exact fallback is allowed only if x*(d0) / alpha <= this many box edges of the first level
/// without expansion (file comment).
inline constexpr Real kExactFallbackBoxEdges = 4.0;

/// Pre-assembly estimate and result of a region's exact far part (file comment).
struct ExactPartInfo {
    int first_level = -1;       ///< first (finest) level without expansion; -1: none
    Index box_pairs = 0;        ///< ordered exact box pairs
    Index pair_bound = 0;       ///< sum n_A n_B over them (upper bound of the basis pairs)
    Index counted_pairs = -1;   ///< per-basis-pair count before assembly; -1: not needed
    Index pairs = 0;            ///< assembled basis pairs (0 before assembly / none)
    Index truncated_pairs = 0;  ///< basis pairs of exact box pairs dropped by the bound
    Real truncated_bound = 0;   ///< their largest delta
    std::size_t bytes = 0;      ///< memory of the assembled part
    Real seconds = 0;           ///< assembly time
};

/// Acceptance threshold of the block check of lossy regions for d0 digits: 10^-d0.
[[nodiscard]] Real block_check_tolerance(Real digits);
/// Lossy regions: largest accepted ratio of the block error to that of the lossless analogue.
inline constexpr Real kLossDegradationFactor = 2.0;

/// Decision of the region policy for one region and level (file comment).
enum class FarDecision { expansion, truncation, exact };

/// "expansion", "truncation" or "exact".
[[nodiscard]] const char* to_string(FarDecision d);

/// Per region and level (2 ... leaf) setup data, reported by describe().
struct FarLevelInfo {
    int level = 0;
    Index boxes = 0;
    Real box_size = 0;  ///< box edge [m]
    FarDecision decision = FarDecision::expansion;
    int truncation_order = 0;        ///< L of the translators (expansion levels)
    int sampling_order = 0;          ///< L of the sampling (leaf: max(L, p - 1))
    Index directions = 0;            ///< 2 (sampling_order + 1)^2
    bool search_achievable = false;  ///< order search result (false: failed or not run)
    Real search_error = 0;           ///< statistical check of search_truncation_order at L
    Real block_error = -1;           ///< block check (file comment); -1: not run ("not run")
    Real reference_error = -1;       ///< lossless analogue's block error; -1: not needed / none
    Index block_pairs = 0;           ///< box pairs of the block check
    Real decay_bound = 0;            ///< non-expansion levels: largest delta of a truncated pair
    Real exact_bound = 0;            ///< non-expansion levels: smallest delta of an exact pair
    Index truncated_pairs = 0;       ///< ordered box pairs dropped (delta <= 10^-(d0+1))
    Index exact_pairs = 0;           ///< ordered box pairs evaluated exactly
    Index translators = 0;           ///< distinct interaction offsets (expansion levels)
    std::size_t pattern_bytes = 0;   ///< per-apply outgoing + incoming fields of the level
    Real setup_seconds = 0;          ///< search, block check, decisions, translators, shifts
};

/// No expansion order meets the accuracy for a region and level and the region's decay is too
/// weak for the truncation / exact fallback of ADR 0008 §6, or the exact fallback would exceed
/// MlfmmParams::max_exact_far_bytes (file comment). Thrown by the far operator (and
/// MlfmmOperator) before any expensive setup or allocation; Simulation turns it into advice per
/// cause.
class TruncationOrderError : public std::runtime_error {
public:
    enum class Cause {
        /// r_max / a > max_support_ratio(d0) at the failing level: finer mesh, larger leaves
        mesh_or_leaf_size,
        /// alpha D >= 1 (D the enlarged diagonal: the loss spoils the expansion) but x*(d0) /
        /// alpha > kExactFallbackBoxEdges a (too little decay for the exact fallback)
        lossy_region,
        /// otherwise: fewer digits or larger leaves
        digits,
        /// the exact parts' estimate exceeds MlfmmParams::max_exact_far_bytes (level() = the
        /// region's first level without expansion; the message gives the estimate)
        exact_part_too_large
    };
    TruncationOrderError(const std::string& what, int region, int level, Cause cause)
        : std::runtime_error(what), region_(region), level_(level), cause_(cause) {}
    [[nodiscard]] int region() const { return region_; }  ///< 0 = R1, 1 = R2
    [[nodiscard]] int level() const { return level_; }
    [[nodiscard]] Cause cause() const { return cause_; }

private:
    int region_;
    int level_;
    Cause cause_;
};

class MlfmmFarOperator final : public op::LinearOperator {
public:
    /// @param tree octree over problem.space (non-owning; must outlive the operator)
    /// @param params accuracy_digits = d0 in (0, 5] (interpolation_order); truncation_L must be 0
    ///        (per-level search), precompute_translators true, use_fft_interpolation false;
    ///        params.octree is not used (the tree is given).
    /// @throws std::invalid_argument for an invalid Problem (op::validate), unsupported params, a
    ///         tree not built on problem.space or (trees with >= 3 levels) a largest support
    ///         radius r_max >= the leaf edge; TruncationOrderError (file comment);
    ///         std::overflow_error from the plane-wave functions.
    MlfmmFarOperator(const op::Problem& problem, const Octree& tree, const MlfmmParams& params);
    MlfmmFarOperator(const op::Problem&, const Octree&&, const MlfmmParams&) = delete;
    ~MlfmmFarOperator() override;

    [[nodiscard]] Index rows() const override;
    [[nodiscard]] Index cols() const override;
    /// y = Z_far x. @throws std::invalid_argument if x has the wrong size.
    void apply(const VectorXc& x, VectorXc& y) const override;
    [[nodiscard]] std::string describe() const override;
    /// Stored tables (leaf patterns, translators, phase shifts, interpolators), the exact parts
    /// and the pooled apply workspaces, counted as at least one (the one a serial solve needs).
    [[nodiscard]] std::size_t memory_bytes() const override;
    /// Bytes of one apply workspace (the fields of all levels and active regions; 0 without far
    /// levels), excluding the per-thread scratch.
    [[nodiscard]] std::size_t workspace_bytes() const;
    /// Number of apply workspaces allocated so far (pooled or in use): 0 before the first apply,
    /// 1 after serial applies, at most the peak number of concurrent apply() calls.
    [[nodiscard]] std::size_t workspaces() const;

    [[nodiscard]] const Octree& octree() const;
    /// Region 0 = R1 (exterior), 1 = R2 (object). @throws std::out_of_range for other indices.
    [[nodiscard]] bool region_active(int region) const;
    /// Levels 2 ... leaf of an active region with their decisions (empty if the tree has fewer
    /// than 3 levels or the region is inactive).
    [[nodiscard]] const std::vector<FarLevelInfo>& levels(int region) const;
    /// Leaf radiation patterns of a region that uses the expansion on the leaf level.
    /// @throws std::invalid_argument otherwise (inactive region, fewer than 3 levels or no
    ///         expansion level).
    [[nodiscard]] const RadiationPatterns& patterns(int region) const;
    /// Exact region-i entries (L_i, K_i per basis pair) of the box pairs with decision exact (file
    /// comment) as a 2N x 2N operator; nullptr if the region has none.
    [[nodiscard]] const op::RegionSparseOperator* exact_part(int region) const;
    /// Estimate and result of the region's exact part (zeros / -1 if it has none).
    [[nodiscard]] const ExactPartInfo& exact_info(int region) const;
    /// The exact-part budget in effect (MlfmmParams::max_exact_far_bytes, automatic value
    /// resolved) [bytes].
    [[nodiscard]] std::size_t exact_budget() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Near-field bytes of the octree's leaf level as mlfmm::assemble_near stores them (4 blocks x
/// (16 + 8) bytes per near basis pair + row pointers), computed from the box counts without
/// assembling: the reference of the automatic exact-part budget.
[[nodiscard]] std::size_t estimate_near_bytes(const Octree& tree);

/// Test-only hooks, not part of the API (no stability guarantee; used by tests/unit).
namespace testing {
/// Workspace-pool exception safety (WP21): the next `count` apply workspaces that
/// MlfmmFarOperator::apply would allocate throw std::bad_alloc instead (process-wide counter,
/// not for production use).
void fail_next_workspace_allocations(int count);
}  // namespace testing

}  // namespace specklebem::mlfmm
