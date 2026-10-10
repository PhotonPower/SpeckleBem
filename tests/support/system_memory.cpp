#include "system_memory.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// windows.h must precede psapi.h.
#include <psapi.h>
#elif __has_include(<sys/resource.h>) && __has_include(<unistd.h>)
#include <sys/resource.h>
#include <unistd.h>
#define SPECKLEBEM_TEST_HAVE_POSIX_MEMORY 1
#endif

#if defined(__linux__)
#include <fstream>
#include <sstream>
#include <string>
#endif

namespace system_memory {

using specklebem::Real;

namespace {

#if defined(__linux__)
/// MemAvailable of /proc/meminfo in bytes (the kernel's estimate of the memory available to new
/// allocations without swapping: free pages plus reclaimable page cache), 0 if absent (kernels
/// before 3.14) or unreadable.
Real linux_mem_available() {
    std::ifstream in("/proc/meminfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("MemAvailable:", 0) != 0)
            continue;
        std::istringstream fields(line.substr(13));
        Real kib = 0.0;
        if (fields >> kib && kib > 0.0)
            return 1024.0 * kib;  // reported in kB (= KiB)
        return 0.0;
    }
    return 0.0;
}
#endif

}  // namespace

Real physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0)
        return static_cast<Real>(status.ullTotalPhys);
#elif defined(SPECKLEBEM_TEST_HAVE_POSIX_MEMORY)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<Real>(pages) * static_cast<Real>(page_size);
#endif
    return 0.0;
}

Real available_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0)
        return static_cast<Real>(status.ullAvailPhys);
#else
#if defined(__linux__)
    if (const Real avail = linux_mem_available(); avail > 0.0)
        return avail;
#endif
#if defined(SPECKLEBEM_TEST_HAVE_POSIX_MEMORY) && defined(_SC_AVPHYS_PAGES)
    // Free pages only (no reclaimable page cache): an underestimate, used without MemAvailable.
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<Real>(pages) * static_cast<Real>(page_size);
#endif
#endif
    return 0.0;
}

Real peak_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) != 0)
        return static_cast<Real>(counters.PeakWorkingSetSize);
#elif defined(SPECKLEBEM_TEST_HAVE_POSIX_MEMORY)
    // ru_maxrss is in KiB on Linux.
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return 1024.0 * static_cast<Real>(ru.ru_maxrss);
#endif
    return 0.0;
}

}  // namespace system_memory
