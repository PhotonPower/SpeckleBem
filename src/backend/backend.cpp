#include "specklebem/backend/backend.hpp"

#include "specklebem/backend/cpu/cpu_backend.hpp"

#include <stdexcept>
#ifdef SPECKLEBEM_HAVE_CUDA
#include "specklebem/backend/gpu/cuda_backend.hpp"
#endif

namespace specklebem::backend {

namespace {
std::shared_ptr<Backend>& current() {
    static std::shared_ptr<Backend> b = std::make_shared<CpuBackend>();
    return b;
}
}  // namespace

std::shared_ptr<Backend> make_backend(Device d) {
    switch (d) {
        case Device::cpu:
            return std::make_shared<CpuBackend>();
        case Device::cuda:
#ifdef SPECKLEBEM_HAVE_CUDA
            return std::make_shared<CudaBackend>();
#else
            throw std::runtime_error("SpeckleBem was built without CUDA support");
#endif
    }
    throw std::logic_error("unknown device");
}

std::shared_ptr<Backend> default_backend() {
    return current();
}
void set_default_backend(std::shared_ptr<Backend> b) {
    current() = std::move(b);
}

}  // namespace specklebem::backend
