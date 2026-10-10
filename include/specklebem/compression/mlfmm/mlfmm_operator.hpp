#pragma once
/// @file mlfmm_operator.hpp
/// Multilevel fast multipole operator Z = Z_near + Z_far (WP20b, ADR 0008).
///
/// Pipeline per matrix-vector product (Song, Lu & Chew 1997):
///   1. Aggregation   : radiation patterns of leaf boxes (plane-wave expansion
///                      sampled on the unit sphere, L_m Gauss-Legendre points in
///                      theta x 2 L_m uniform points in phi), upward pass with
///                      interpolation + phase shift.
///   2. Translation   : interaction-list transfer with diagonal translators
///                      T_L(k, r_ij, k_hat).
///   3. Disaggregation: downward pass with anterpolation + phase shift,
///                      receive at leaves.
///
/// Z_near is handled by SparseOperator. Both regions (k_1, k_2) are expanded;
/// the lossy interior wavenumber of metals limits the usable depth (see
/// docs/04_theory_mlfmm.md).
///
/// MlfmmOperator owns the octree (built from params.octree with the wavelength of R1,
/// lambda_1 = 2 pi / Re k_1, as the reference of min_box_size_lambda), the exact near field
/// Z_near (assemble_near: basis pairs in the same or adjacent leaves, all four blocks, bitwise
/// equal to the dense entries) and the multilevel far part Z_far (MlfmmFarOperator: every other
/// leaf pair, through exactly one interaction list). The octree lists make the two parts
/// complementary, so apply() = Z_near x + Z_far x counts every pair once. apply() is safe for
/// concurrent calls (both parts are).
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/linear_operator.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace specklebem::op {
class SparseOperator;
}

namespace specklebem::mlfmm {

class MlfmmFarOperator;

struct MlfmmParams {
    OctreeParams octree;
    int truncation_L = 0;        ///< 0 => automatic from excess bandwidth formula
    Real accuracy_digits = 3.0;  ///< d0 in L = kD + 1.8 d0^{2/3} (kD)^{1/3}
    bool precompute_translators = true;
    bool use_fft_interpolation = false;  ///< later: FFT / Lagrange interpolation
    /// Leaf rule of ADR 0008 (WP20b amendment), applied by MlfmmOperator: the octree's leaf edge
    /// floor is raised to max(octree.min_box_size_lambda, a_min(d0), r_max / 0.3) with a_min =
    /// lambda1 / 4 for d0 <= 3 and lambda1 / 2 for d0 > 3 (r_max = max_support_radius, lambda1 the
    /// exterior wavelength); octree.min_box_size_lambda acts as a lower bound. The accuracy
    /// statements of ADR 0008 / docs/05 assume it. false: the octree parameters are used as given
    /// (tests and experiments with deliberately small trees).
    bool automatic_leaf_size = true;
    /// Per region (0 = R1, 1 = R2): true evaluates every far interaction of that region exactly
    /// (the near-field fallback of ADR 0008 §6 on every far level) instead of the
    /// expansion / truncation policy. Rigorous, but O(N^2) memory for that region; meant for
    /// diagnostics and the complementarity tests.
    std::array<bool, 2> exact_far_regions = {false, false};
};

/// Leaf rule of ADR 0008 (MlfmmParams::automatic_leaf_size): leaf edge >= kLeafMinLambdaD3 lambda
/// for d0 <= 3, >= kLeafMinLambdaD5 lambda for d0 > 3 (measured: d0 = 5 needs leaves >= lambda/2),
/// and r_max / a <= kLeafMaxSupportRatio (the order search fails for larger ratios at d0 = 5).
inline constexpr Real kLeafMinLambdaD3 = 0.25;
inline constexpr Real kLeafMinLambdaD5 = 0.5;
inline constexpr Real kLeafMaxSupportRatio = 0.3;

/// Octree parameters after the leaf rule: params.octree with min_box_size_lambda raised to
/// max(given, a_min(d0), r_max / (kLeafMaxSupportRatio (1 - kMinBoxSizeTolerance) wavelength))
/// (the tolerance factor keeps r_max / a <= 0.3 despite the octree's floor tolerance);
/// params.octree unchanged if params.automatic_leaf_size is false.
/// @param wavelength the octree's reference wavelength lambda1 [m]
/// @throws std::invalid_argument for an empty space, accuracy_digits <= 0 or a non-positive
///         wavelength.
[[nodiscard]] OctreeParams leaf_rule_params(const basis::RwgSpace& space, Real wavelength,
                                            const MlfmmParams& params);

class MlfmmOperator final : public op::LinearOperator {
public:
    /// Builds the octree (octree params after leaf_rule_params), then the far part (its order
    /// searches and policy decisions reject unusable configurations before the near-field
    /// assembly), then the near field.
    /// @throws std::invalid_argument for an invalid Problem (op::validate), invalid octree or
    ///         MLFMM params, or a mesh too coarse for the leaves (r_max >= leaf edge);
    ///         TruncationOrderError (a std::runtime_error) if a region has neither a usable
    ///         expansion nor enough decay for the ADR 0008 §6 fallback, see MlfmmFarOperator.
    MlfmmOperator(const op::Problem& problem, const MlfmmParams& params);
    ~MlfmmOperator() override;

    [[nodiscard]] Index rows() const override;
    [[nodiscard]] Index cols() const override;
    /// y = Z_near x + Z_far x. @throws std::invalid_argument if x has the wrong size.
    void apply(const VectorXc& x, VectorXc& y) const override;
    /// Octree, near nnz / memory / time, far per-region and per-level data, totals.
    [[nodiscard]] std::string describe() const override;
    /// Near field + far tables and pooled apply workspaces (MlfmmFarOperator::memory_bytes) +
    /// octree.
    [[nodiscard]] std::size_t memory_bytes() const override;

    [[nodiscard]] const Octree& octree() const;
    [[nodiscard]] const op::SparseOperator& near_operator() const;
    [[nodiscard]] const MlfmmFarOperator& far_operator() const;

private:
    [[nodiscard]] std::size_t octree_bytes() const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MlfmmStrategy final : public op::CompressionStrategy {
public:
    explicit MlfmmStrategy(MlfmmParams p = {}) : params_(p) {}
    /// A new MlfmmOperator(p, params). @throws as MlfmmOperator.
    [[nodiscard]] std::shared_ptr<op::LinearOperator> build(const op::Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "mlfmm"; }

private:
    MlfmmParams params_;
};

}  // namespace specklebem::mlfmm
