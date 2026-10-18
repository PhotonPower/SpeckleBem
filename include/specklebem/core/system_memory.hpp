#pragma once
/// @file system_memory.hpp
/// Physical memory currently available to new allocations, for the memory warnings of large
/// operators (WP-PML5a review W3: the exact PML ring far blocks). Platform headers stay in
/// system_memory.cpp (<windows.h> defines the macros `near` and `far`).
#include <cstddef>

namespace specklebem::core {

/// Physical memory currently available to new allocations in bytes; 0 if unknown. On a shared
/// machine this, not the installed memory, decides whether a large allocation fits without
/// swapping.
///  * Windows: ullAvailPhys of GlobalMemoryStatusEx;
///  * Linux: MemAvailable of /proc/meminfo (free plus reclaimable memory), falling back to the
///    free pages sysconf(_SC_AVPHYS_PAGES) if MemAvailable is missing;
///  * other POSIX systems with _SC_AVPHYS_PAGES: the free pages;
///  * anything else (e.g. macOS): 0.
/// Never throws.
[[nodiscard]] std::size_t available_memory_bytes() noexcept;

}  // namespace specklebem::core
