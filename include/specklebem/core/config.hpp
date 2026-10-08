#pragma once
/// @file config.hpp
/// Version information and feature flags.

#define SPECKLEBEM_VERSION_MAJOR 0
#define SPECKLEBEM_VERSION_MINOR 1
#define SPECKLEBEM_VERSION_PATCH 0

namespace specklebem {
inline constexpr const char* version_string() {
    return "0.1.0";
}
bool has_openmp();
bool has_cuda();
bool has_hdf5();
bool has_blas_lapack();
}  // namespace specklebem
