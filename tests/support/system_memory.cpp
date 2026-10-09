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

namespace system_memory {

using specklebem::Real;

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
