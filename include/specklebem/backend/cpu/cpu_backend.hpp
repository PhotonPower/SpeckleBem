#pragma once
#include "specklebem/backend/backend.hpp"

namespace specklebem::backend {

class CpuBackend final : public Backend {
public:
    [[nodiscard]] Device device() const override { return Device::cpu; }
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int num_threads() const override;
    void gemv(const MatrixXc& A, const VectorXc& x, VectorXc& y, Complex alpha,
              Complex beta) const override;
};

}  // namespace specklebem::backend
