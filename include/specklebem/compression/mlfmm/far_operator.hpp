#pragma once
/// @file far_operator.hpp
/// Multilevel fast multipole far part Z_far of the combined 2N x 2N system (WP20a,
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
/// offset present in the interaction lists. If the order search finds no order meeting 10^-d0
/// for an active region at some level, the constructor throws (the lossy-region policy of ADR
/// 0008 §6 is WP21). A region whose weights are all zero is skipped.
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
#include <string>
#include <vector>

namespace specklebem::mlfmm {

/// Per region and level (2 ... leaf) setup data, reported by describe().
struct FarLevelInfo {
    int level = 0;
    Index boxes = 0;
    Real box_size = 0;              ///< box edge [m]
    int truncation_order = 0;       ///< L of the translators
    int sampling_order = 0;         ///< L of the sampling (leaf: max(L, p - 1))
    Index directions = 0;           ///< 2 (sampling_order + 1)^2
    Real search_error = 0;          ///< statistical check of search_truncation_order at L
    Index translators = 0;          ///< distinct interaction offsets
    std::size_t pattern_bytes = 0;  ///< per-apply outgoing + incoming fields of the level
    Real setup_seconds = 0;         ///< order search, translators, interpolator, shifts
};

class MlfmmFarOperator final : public op::LinearOperator {
public:
    /// @param tree octree over problem.space (non-owning; must outlive the operator)
    /// @param params accuracy_digits = d0 in (0, 5] (interpolation_order); truncation_L must be 0
    ///        (per-level search), precompute_translators true, use_fft_interpolation false;
    ///        params.octree is not used (the tree is given).
    /// @throws std::invalid_argument for an invalid Problem (op::validate), unsupported params, a
    ///         tree not built on problem.space or (trees with >= 3 levels) a largest support
    ///         radius r_max >= the leaf edge; std::runtime_error if no truncation order meets
    ///         10^-d0 for an active region at some level (lossy region, WP21);
    ///         std::underflow_error / std::overflow_error from the plane-wave functions.
    MlfmmFarOperator(const op::Problem& problem, const Octree& tree, const MlfmmParams& params);
    MlfmmFarOperator(const op::Problem&, const Octree&&, const MlfmmParams&) = delete;
    ~MlfmmFarOperator() override;

    [[nodiscard]] Index rows() const override;
    [[nodiscard]] Index cols() const override;
    /// y = Z_far x. @throws std::invalid_argument if x has the wrong size.
    void apply(const VectorXc& x, VectorXc& y) const override;
    [[nodiscard]] std::string describe() const override;
    /// Stored tables (leaf patterns, translators, phase shifts, interpolators) plus the pooled
    /// apply workspaces, counted as at least one (the one a serial solve needs).
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
    /// Levels 2 ... leaf of an active region (empty if the tree has fewer than 3 levels).
    [[nodiscard]] const std::vector<FarLevelInfo>& levels(int region) const;
    /// Leaf radiation patterns of an active region. @throws std::invalid_argument if the region is
    /// inactive or the tree has fewer than 3 levels.
    [[nodiscard]] const RadiationPatterns& patterns(int region) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace specklebem::mlfmm
