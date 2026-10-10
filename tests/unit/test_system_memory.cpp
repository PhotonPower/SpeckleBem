/// Tests of the memory queries of the test support library (tests/support/system_memory.hpp)
/// that guard the large validation cases (WP22a-f).
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "system_memory.hpp"

using specklebem::Real;

TEST_CASE("system_memory: physical, available and peak memory are consistent", "[system_memory]") {
    const Real total = system_memory::physical_memory_bytes();
    const Real avail = system_memory::available_memory_bytes();
    const Real peak = system_memory::peak_rss_bytes();
    INFO("physical " << total / 1e9 << " GB, available " << avail / 1e9 << " GB, peak RSS "
                     << peak / 1e9 << " GB");
    CHECK(total >= 0.0);
    CHECK(avail >= 0.0);
    CHECK(peak >= 0.0);
#if defined(_WIN32) || defined(__linux__)
    // Known on the supported platforms (macOS returns 0 for the available memory, documented).
    CHECK(total > 0.0);
    CHECK(avail > 0.0);
    CHECK(peak > 0.0);
#endif
    if (total > 0.0)
        CHECK(avail <= total);
    if (total > 0.0 && peak > 0.0)
        CHECK(peak <= total);
}
