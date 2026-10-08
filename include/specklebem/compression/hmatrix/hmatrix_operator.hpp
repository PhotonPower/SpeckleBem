#pragma once
/// @file hmatrix_operator.hpp
/// Placeholder for Phase 8: hierarchical (H / H^2) matrix representation,
/// enabling approximate LU as a strong preconditioner for metallic surfaces.
#include "specklebem/operator/assembler.hpp"

namespace specklebem::hmatrix {

class HMatrixStrategy final : public op::CompressionStrategy {
public:
    [[nodiscard]] std::shared_ptr<op::LinearOperator> build(const op::Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "hmatrix"; }
};

}  // namespace specklebem::hmatrix
