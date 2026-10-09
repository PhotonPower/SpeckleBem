#include "specklebem/core/path.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "utf8_test_support.hpp"

using specklebem::core::find_invalid_utf8;
using specklebem::core::path_from_utf8;
using specklebem::core::path_to_utf8;
using specklebem::test::utf8;
namespace fs = std::filesystem;

TEST_CASE("path: ASCII and non-ASCII UTF-8 round trips", "[core][path]") {
    for (const std::string& s :
         {std::string(), std::string("mesh.stl"), std::string("C:/data/run 1/sphere.obj"),
          std::string("rel/dir/"), utf8(u8"J\u00fcrgen_\u00b5m_\u8def\u5f84/\u00fc.stl"),
          utf8(u8"\U0001F642/caf\u00e9.npy"), specklebem::test::non_ascii_dir_name()}) {
        INFO("path '" << s << "'");
        CHECK(find_invalid_utf8(s) == std::string_view::npos);
        CHECK(path_to_utf8(path_from_utf8(s)) == s);
    }
    // Components and extensions are those of the Unicode path.
    const fs::path p = path_from_utf8(utf8(u8"J\u00fcrgen_\u00b5m_\u8def\u5f84/\u00fc.stl"));
    CHECK(path_to_utf8(p.filename()) == utf8(u8"\u00fc.stl"));
    CHECK(path_to_utf8(p.parent_path()) == utf8(u8"J\u00fcrgen_\u00b5m_\u8def\u5f84"));
    CHECK(path_to_utf8(p.extension()) == ".stl");
    CHECK(p == fs::path(u8"J\u00fcrgen_\u00b5m_\u8def\u5f84/\u00fc.stl"));
#ifdef _WIN32
    // Independent of the ANSI code page: the native (UTF-16) name holds the code points.
    CHECK(path_from_utf8(utf8(u8"J\u00fc\u8def\U0001F642")).native() ==
          std::wstring(L"J\u00fc\u8def\U0001F642"));
#endif
}

TEST_CASE("path: malformed UTF-8 and NUL bytes throw", "[core][path]") {
    struct Bad {
        std::string bytes;
        std::size_t offset;
    };
    for (const Bad& b :
         {Bad{"\xff", 0}, Bad{"ab\x80", 2}, Bad{"x\xc3", 1}, Bad{"\xe2\x82", 0}, Bad{"\xc0\xaf", 0},
          Bad{"\xe0\x80\xaf", 0}, Bad{"ok/\xed\xa0\x80", 3}, Bad{"\xf4\x90\x80\x80", 0},
          Bad{"\xc3\x28", 0}, Bad{"\xf8\x88\x80\x80", 0}}) {
        INFO("offset " << b.offset);
        CHECK(find_invalid_utf8(b.bytes) == b.offset);
        CHECK_THROWS_AS(path_from_utf8(b.bytes), std::invalid_argument);
    }
    CHECK(find_invalid_utf8("\xf4\x8f\xbf\xbf") == std::string_view::npos);  // U+10FFFF
    CHECK_THROWS_AS(path_from_utf8(std::string("a\0b", 3)), std::invalid_argument);
}
