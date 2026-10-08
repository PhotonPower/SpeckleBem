// CUDA backend - Phase 7 placeholder. Compiled only when SPECKLEBEM_ENABLE_CUDA=ON.
#include "specklebem/backend/gpu/cuda_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace specklebem::backend {

CudaBackend::CudaBackend(int device_id) : device_id_(device_id) {
    if (cudaSetDevice(device_id_) != cudaSuccess)
        throw std::runtime_error("cudaSetDevice failed");
}
std::string CudaBackend::name() const {
    return "cuda:" + std::to_string(device_id_);
}
int CudaBackend::num_threads() const {
    return 0;
}

void CudaBackend::gemv(const MatrixXc&, const VectorXc&, VectorXc&, Complex, Complex) const {
    throw std::logic_error("CudaBackend::gemv not implemented yet (Phase 7)");
}

}  // namespace specklebem::backend
