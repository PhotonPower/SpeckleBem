#pragma once
/// @file cuda_backend.hpp
/// CUDA backend (Phase 7). Compiled only with SPECKLEBEM_ENABLE_CUDA.
/// Planned kernels, in order of expected payoff:
///   1. near-field element assembly (embarrassingly parallel over triangle pairs)
///   2. MLFMM translation stage (batched complex multiply-accumulate)
///   3. leaf aggregation / disaggregation and interpolation
///   4. post-processing field evaluation on large observation grids
#include "specklebem/backend/backend.hpp"

namespace specklebem::backend {

class CudaBackend final : public Backend {
public:
    explicit CudaBackend(int device_id = 0);
    [[nodiscard]] Device device() const override { return Device::cuda; }
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int num_threads() const override;
    void gemv(const MatrixXc& A, const VectorXc& x, VectorXc& y, Complex alpha,
              Complex beta) const override;

private:
    int device_id_;
};

}  // namespace specklebem::backend
