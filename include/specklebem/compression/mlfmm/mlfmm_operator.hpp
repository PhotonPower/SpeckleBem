#pragma once
/// @file mlfmm_operator.hpp
/// Multilevel fast multipole operator for the far-field part of Z.
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
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/linear_operator.hpp"

#include <memory>

namespace specklebem::mlfmm {

struct MlfmmParams {
    OctreeParams octree;
    int truncation_L = 0;        ///< 0 => automatic from excess bandwidth formula
    Real accuracy_digits = 3.0;  ///< d0 in L = kD + 1.8 d0^{2/3} (kD)^{1/3}
    bool precompute_translators = true;
    bool use_fft_interpolation = false;  ///< later: FFT / Lagrange interpolation
};

class MlfmmOperator final : public op::LinearOperator {
public:
    MlfmmOperator(const op::Problem& problem, const MlfmmParams& params);
    ~MlfmmOperator() override;

    [[nodiscard]] Index rows() const override;
    [[nodiscard]] Index cols() const override;
    void apply(const VectorXc& x, VectorXc& y) const override;
    [[nodiscard]] std::string describe() const override;
    [[nodiscard]] std::size_t memory_bytes() const override;

    [[nodiscard]] const Octree& octree() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MlfmmStrategy final : public op::CompressionStrategy {
public:
    explicit MlfmmStrategy(MlfmmParams p = {}) : params_(p) {}
    [[nodiscard]] std::shared_ptr<op::LinearOperator> build(const op::Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "mlfmm"; }

private:
    MlfmmParams params_;
};

}  // namespace specklebem::mlfmm
