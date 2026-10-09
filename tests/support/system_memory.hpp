#pragma once
/// @file system_memory.hpp
/// Physical memory of the machine and peak resident set size of the test process, for the
/// memory guards of the dense validation cases (mie_dense_test_support.hpp). Implemented in
/// system_memory.cpp so that the platform headers (<windows.h> defines the macros `near` and
/// `far`, which collide with kernels::Proximity) stay out of the test translation units.
#include "specklebem/core/types.hpp"

namespace system_memory {

/// Total physical memory of the machine in bytes (0 if unknown).
specklebem::Real physical_memory_bytes();

/// Peak resident set size (peak working set on Windows) of this process in bytes (0 if
/// unknown). catch_discover_tests runs every test case in its own process.
specklebem::Real peak_rss_bytes();

}  // namespace system_memory
