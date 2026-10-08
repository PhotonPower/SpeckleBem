#pragma once
/// @file backend.hpp
/// Execution backend abstraction (ADR 0005).
///
/// Performance-critical kernels (near-field assembly, MLFMM translation and
/// interpolation, dense matvec, field evaluation at many points) are written
/// against this interface so that the CPU (OpenMP + BLAS) and the GPU (CUDA)
/// implementations can be exchanged without touching the algorithmic layers.
/// Data ownership is explicit: buffers live in a `DeviceBuffer` and are moved
/// across the host/device boundary only at well-defined points.
#include "specklebem/core/types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace specklebem::backend {

enum class Device { cpu, cuda };

class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual Device device() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual int num_threads() const = 0;

    /// y = alpha A x + beta y, A dense complex (BLAS zgemv / cuBLAS).
    virtual void gemv(const MatrixXc& A, const VectorXc& x, VectorXc& y, Complex alpha = 1,
                      Complex beta = 0) const = 0;
};

std::shared_ptr<Backend> make_backend(Device d);
std::shared_ptr<Backend> default_backend();
void set_default_backend(std::shared_ptr<Backend> b);

}  // namespace specklebem::backend
