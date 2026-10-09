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
};

class MlfmmOperator final : public op::LinearOperator {
public:
    /// Builds the octree, then the far part (its cheap order searches reject unusable
    /// configurations before the near-field assembly), then the near field.
    /// @throws std::invalid_argument for an invalid Problem (op::validate), invalid octree or
    ///         MLFMM params, or a mesh too coarse for the leaves (r_max >= leaf edge);
    ///         std::runtime_error if no truncation order meets 10^-d0 for a region and level
    ///         (e.g. a lossy metal interior: the ADR 0008 §6 policy is WP21), see
    ///         MlfmmFarOperator.
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
