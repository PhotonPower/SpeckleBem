#include "specklebem/core/timer.hpp"

#include "specklebem/core/logging.hpp"

#include <utility>

namespace specklebem {

ScopedTimer::ScopedTimer(std::string label)
    : label_(std::move(label)), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    SBEM_INFO("[timer] {}: {:.3f} s", label_, elapsed_seconds());
}

double ScopedTimer::elapsed_seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

}  // namespace specklebem
