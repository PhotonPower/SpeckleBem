/// @file system_memory.cpp
/// core::available_memory_bytes (system_memory.hpp).
#include "specklebem/core/system_memory.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif __has_include(<unistd.h>)
#include <unistd.h>
#define SPECKLEBEM_HAVE_UNISTD 1
#endif

#if defined(__linux__)
#include <fstream>
#include <sstream>
#include <string>
#endif

namespace specklebem::core {

namespace {

#if defined(__linux__)
/// MemAvailable of /proc/meminfo in bytes, 0 if absent (kernels before 3.14) or unreadable.
std::size_t linux_mem_available() {
    std::ifstream in("/proc/meminfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("MemAvailable:", 0) != 0)
            continue;
        std::istringstream fields(line.substr(13));
        unsigned long long kib = 0;
        if (fields >> kib)
            return static_cast<std::size_t>(kib) * 1024U;  // reported in kB (= KiB)
        return 0;
    }
    return 0;
}
#endif

}  // namespace

std::size_t available_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status) != 0)
            return static_cast<std::size_t>(status.ullAvailPhys);
#else
#if defined(__linux__)
        if (const std::size_t avail = linux_mem_available(); avail > 0)
            return avail;
#endif
#if defined(SPECKLEBEM_HAVE_UNISTD) && defined(_SC_AVPHYS_PAGES)
        // Free pages only (no reclaimable page cache): an underestimate.
        const long pages = sysconf(_SC_AVPHYS_PAGES);
        const long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0)
            return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
#endif
#endif
    } catch (...) {  // stream errors (bad_alloc) on Linux: unknown
    }
    return 0;
}

}  // namespace specklebem::core
