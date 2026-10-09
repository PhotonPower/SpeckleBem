/// @file path.cpp
/// UTF-8 <-> std::filesystem::path conversion (ADR 0007).
#include "specklebem/core/path.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace specklebem::core {

std::size_t find_invalid_utf8(std::string_view s) noexcept {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    std::size_t k = 0;
    while (k < s.size()) {
        const unsigned char c = byte(k);
        std::size_t n = 0;  // number of continuation bytes
        std::uint32_t cp = 0;
        if (c < 0x80U) {
            ++k;
            continue;
        }
        if ((c & 0xe0U) == 0xc0U) {
            n = 1;
            cp = c & 0x1fU;
        } else if ((c & 0xf0U) == 0xe0U) {
            n = 2;
            cp = c & 0x0fU;
        } else if ((c & 0xf8U) == 0xf0U) {
            n = 3;
            cp = c & 0x07U;
        } else {
            return k;  // continuation byte or 0xf8-0xff as lead byte
        }
        if (k + n >= s.size())
            return k;  // truncated sequence
        for (std::size_t m = 1; m <= n; ++m) {
            if ((byte(k + m) & 0xc0U) != 0x80U)
                return k;
            cp = (cp << 6) | (byte(k + m) & 0x3fU);
        }
        constexpr std::array<std::uint32_t, 4> kMin = {0, 0x80U, 0x800U, 0x10000U};
        if (cp < kMin[n] || cp > 0x10ffffU || (cp >= 0xd800U && cp <= 0xdfffU))
            return k;
        k += n + 1;
    }
    return std::string_view::npos;
}

std::filesystem::path path_from_utf8(std::string_view utf8) {
    if (const std::size_t bad = find_invalid_utf8(utf8); bad != std::string_view::npos) {
        throw std::invalid_argument("path_from_utf8: path is not valid UTF-8 (byte " +
                                    std::to_string(bad) + ")");
    }
    if (const std::size_t nul = utf8.find('\0'); nul != std::string_view::npos) {
        throw std::invalid_argument("path_from_utf8: path contains a NUL byte (byte " +
                                    std::to_string(nul) + ")");
    }
    std::u8string u8(utf8.size(), u8'\0');
    std::transform(utf8.begin(), utf8.end(), u8.begin(),
                   [](char c) { return static_cast<char8_t>(c); });
    return std::filesystem::path(u8);
}

std::string path_to_utf8(const std::filesystem::path& p) {
    const std::u8string u8 = p.u8string();
    std::string out(u8.size(), '\0');
    std::transform(u8.begin(), u8.end(), out.begin(),
                   [](char8_t c) { return static_cast<char>(c); });
    return out;
}

}  // namespace specklebem::core
