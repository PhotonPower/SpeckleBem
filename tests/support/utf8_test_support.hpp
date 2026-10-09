#pragma once
/// @file utf8_test_support.hpp
/// Helpers for the UTF-8 path tests (ADR 0007): UTF-8 std::string from a u8"" literal and the
/// non-ASCII directory name used by the round-trip tests.
#include <string>
#include <string_view>

namespace specklebem::test {

/// Bytes of a C++20 u8"" literal (char8_t) as a UTF-8 std::string.
inline std::string utf8(std::u8string_view s) {
    std::string out;
    out.reserve(s.size());
    for (const char8_t c : s) out.push_back(static_cast<char>(c));
    return out;
}

/// Non-ASCII directory name (Latin-1, Greek micro sign, CJK, and a non-BMP character, which
/// needs a UTF-16 surrogate pair on Windows).
inline std::string non_ascii_dir_name() {
    return utf8(u8"sbem_Jürgen_µm_路径_\U0001F642");
}

}  // namespace specklebem::test
