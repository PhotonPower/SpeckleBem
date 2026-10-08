#pragma once
/// @file aca_operator.hpp
/// Placeholder for Phase 8: Adaptive Cross Approximation of admissible
/// cluster-pair blocks (kernel-independent low-rank compression). Shares the
/// Octree / cluster tree with MLFMM. See docs/09_compression_methods.md.
#include "specklebem/operator/assembler.hpp"

namespace specklebem::aca {

struct AcaParams {
    Real epsilon = 1e-4;  ///< relative block accuracy
    Real admissibility_eta = 2.0;
    int leaf_size = 64;
};

class AcaStrategy final : public op::CompressionStrategy {
public:
    explicit AcaStrategy(AcaParams p = {}) : params_(p) {}
    [[nodiscard]] std::shared_ptr<op::LinearOperator> build(const op::Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "aca"; }

private:
    AcaParams params_;
};

}  // namespace specklebem::aca
