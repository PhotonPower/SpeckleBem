#pragma once
/// @file path.hpp
/// UTF-8 file paths (ADR 0007).
///
/// Every `std::string` that denotes a file or directory path in the public C++ API is UTF-8 on
/// every platform. On Windows, `std::filesystem::path(std::string)` and `path::string()`
/// interpret narrow strings in the active ANSI code page, which garbles (or throws on)
/// non-ASCII UTF-8 paths. Library code therefore converts user-supplied path strings only with
/// path_from_utf8 and turns paths back into strings (error messages, logging) only with
/// path_to_utf8; it never uses `path(std::string)` or `path::string()` on user paths and opens
/// streams with `std::filesystem::path` objects.
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

namespace specklebem::core {

/// Byte offset of the first malformed UTF-8 sequence in s (truncated or invalid lead /
/// continuation bytes, overlong forms, surrogates U+D800-U+DFFF, code points above U+10FFFF),
/// or std::string_view::npos if s is well-formed UTF-8.
[[nodiscard]] std::size_t find_invalid_utf8(std::string_view s) noexcept;

/// Path from a UTF-8 string (via std::u8string, so independent of the ANSI code page).
/// @throws std::invalid_argument if utf8 is not well-formed UTF-8 or contains a NUL byte
///         (which would silently truncate the path in operating-system calls); the message
///         names the byte offset, not the malformed bytes.
[[nodiscard]] std::filesystem::path path_from_utf8(std::string_view utf8);

/// UTF-8 string of a path (via path::u8string(), native separators as stored), the inverse of
/// path_from_utf8: path_to_utf8(path_from_utf8(s)) == s for every valid s.
[[nodiscard]] std::string path_to_utf8(const std::filesystem::path& p);

}  // namespace specklebem::core
