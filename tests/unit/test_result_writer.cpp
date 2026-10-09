#include "specklebem/core/path.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/io/result_writer.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "utf8_test_support.hpp"

using namespace specklebem;
using Catch::Matchers::ContainsSubstring;
namespace fs = std::filesystem;

namespace {

/// Unique temporary directory, removed with its contents on destruction.
class TempDir {
public:
    TempDir() {
        static std::atomic<int> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() / ("specklebem_result_writer_" + std::to_string(stamp) +
                                             "_" + std::to_string(counter++));
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    [[nodiscard]] const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in.good());
    const auto size = static_cast<std::size_t>(fs::file_size(p));
    std::string bytes(size, '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(size));
    REQUIRE(in.good());
    return bytes;
}

/// Number of leftover temporary files ("*~tmp") below dir.
int count_temp_files(const fs::path& dir) {
    int n = 0;
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        const std::string f = core::path_to_utf8(e.path().filename());
        n += f.size() >= 4 && f.ends_with("~tmp") ? 1 : 0;
    }
    return n;
}

/// Runs f, which must throw std::runtime_error whose message contains needle.
template <class F>
void check_runtime_error(F&& f, const std::string& needle) {
    try {
        f();
        FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        CHECK_THAT(e.what(), ContainsSubstring(needle));
    }
}

/// Independent NPY 1.0 reader: checks the preamble, returns the header dict (padding and
/// newline stripped) and the raw data bytes.
struct Npy {
    std::string dict;
    std::string data;
};
Npy read_npy(const fs::path& p) {
    const std::string bytes = read_file(p);
    REQUIRE(bytes.size() >= 10);
    REQUIRE(bytes.substr(0, 6) == std::string("\x93NUMPY", 6));
    REQUIRE(bytes[6] == '\x01');
    REQUIRE(bytes[7] == '\x00');
    const std::size_t hlen = static_cast<unsigned char>(bytes[8]) +
                             (std::size_t{static_cast<unsigned char>(bytes[9])} << 8);
    REQUIRE((10 + hlen) % 64 == 0);
    REQUIRE(bytes.size() >= 10 + hlen);
    std::string header = bytes.substr(10, hlen);
    REQUIRE(header.back() == '\n');
    header.pop_back();
    const std::size_t last = header.find_last_not_of(' ');
    return {header.substr(0, last + 1), bytes.substr(10 + hlen)};
}

std::uint64_t le64(const std::string& data, std::size_t k) {
    std::uint64_t bits = 0;
    for (std::size_t b = 0; b < 8; ++b) {
        bits |= std::uint64_t{static_cast<unsigned char>(data[8 * k + b])} << (8 * b);
    }
    return bits;
}
std::uint64_t bits(Real x) {
    return std::bit_cast<std::uint64_t>(x);
}

}  // namespace

TEST_CASE("result_writer: vectors and matrices are bitwise exact, C order", "[io]") {
    TempDir tmp;
    const auto w = io::open_npy_directory(core::path_to_utf8(tmp.path() / "a" / "b"));
    std::mt19937_64 rng(20261009);
    std::normal_distribution<Real> nd(0.0, 1e3);
    VectorXc v(7);
    MatrixXr mr(3, 5);  // non-square: catches C / Fortran order mix-ups
    MatrixXc mc(4, 2);
    for (Index k = 0; k < v.size(); ++k) v(k) = {nd(rng), nd(rng)};
    for (Index k = 0; k < mr.size(); ++k) mr.data()[k] = nd(rng);
    for (Index k = 0; k < mc.size(); ++k) mc.data()[k] = {nd(rng), nd(rng)};
    w->write_vector("current", v);
    w->write_matrix("real", mr);
    w->write_matrix("fields/complex", mc);  // group -> sub-directory
    w->write_vector("empty", VectorXc());
    w->write_matrix("no_rows", MatrixXr(0, 3));

    const fs::path root = tmp.path() / "a" / "b";
    const Npy nv = read_npy(root / "current.npy");
    CHECK(nv.dict == "{'descr': '<c16', 'fortran_order': False, 'shape': (7,), }");
    REQUIRE(nv.data.size() == 7 * 16);
    for (Index k = 0; k < 7; ++k) {
        const auto ku = static_cast<std::size_t>(k);
        CHECK(le64(nv.data, 2 * ku) == bits(v(k).real()));
        CHECK(le64(nv.data, 2 * ku + 1) == bits(v(k).imag()));
    }
    const Npy nr = read_npy(root / "real.npy");
    CHECK(nr.dict == "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 5), }");
    REQUIRE(nr.data.size() == 15 * 8);
    for (Index i = 0; i < 3; ++i) {
        for (Index j = 0; j < 5; ++j) {
            CHECK(le64(nr.data, static_cast<std::size_t>(5 * i + j)) == bits(mr(i, j)));
        }
    }
    const Npy nc = read_npy(root / "fields" / "complex.npy");
    CHECK(nc.dict == "{'descr': '<c16', 'fortran_order': False, 'shape': (4, 2), }");
    REQUIRE(nc.data.size() == 8 * 16);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 2; ++j) {
            const auto k = static_cast<std::size_t>(2 * (2 * i + j));
            CHECK(le64(nc.data, k) == bits(mc(i, j).real()));
            CHECK(le64(nc.data, k + 1) == bits(mc(i, j).imag()));
        }
    }
    const Npy ne = read_npy(root / "empty.npy");
    CHECK(ne.dict == "{'descr': '<c16', 'fortran_order': False, 'shape': (0,), }");
    CHECK(ne.data.empty());
    CHECK(read_npy(root / "no_rows.npy").dict ==
          "{'descr': '<f8', 'fortran_order': False, 'shape': (0, 3), }");

    // Overwriting replaces the file completely.
    w->write_vector("current", VectorXc::Constant(2, Complex(1.0, -2.0)));
    const Npy nv2 = read_npy(root / "current.npy");
    CHECK(nv2.dict == "{'descr': '<c16', 'fortran_order': False, 'shape': (2,), }");
    CHECK(nv2.data.size() == 2 * 16);
    CHECK(count_temp_files(tmp.path()) == 0);  // temporary files renamed over the targets
}

TEST_CASE("result_writer: mesh vertices and 0-based triangles", "[io]") {
    TempDir tmp;
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 1);
    io::open_npy_directory(core::path_to_utf8(tmp.path()))->write_mesh("geometry/sphere", mesh);
    const fs::path dir = tmp.path() / "geometry" / "sphere";
    const Npy nv = read_npy(dir / "vertices.npy");
    CHECK(nv.dict == "{'descr': '<f8', 'fortran_order': False, 'shape': (42, 3), }");
    REQUIRE(nv.data.size() == 42 * 3 * 8);
    const Npy nt = read_npy(dir / "triangles.npy");
    CHECK(nt.dict == "{'descr': '<i8', 'fortran_order': False, 'shape': (80, 3), }");
    REQUIRE(nt.data.size() == 80 * 3 * 8);
    for (Index i = 0; i < 42; ++i) {
        for (Index j = 0; j < 3; ++j) {
            CHECK(le64(nv.data, static_cast<std::size_t>(3 * i + j)) ==
                  bits(mesh.vertices()(i, j)));
        }
    }
    for (Index i = 0; i < 80; ++i) {
        for (Index j = 0; j < 3; ++j) {
            CHECK(le64(nt.data, static_cast<std::size_t>(3 * i + j)) ==
                  static_cast<std::uint64_t>(mesh.triangles()(i, j)));
        }
    }
}

TEST_CASE("result_writer: attributes.json escaping and Real round trip", "[io]") {
    TempDir tmp;
    const fs::path json = tmp.path() / "attributes.json";
    {
        const auto w = io::open_npy_directory(core::path_to_utf8(tmp.path()));
        w->write_attribute("title", "say \"hi\" \\ C:\\dir\nline\ttab\x01 \xc3\xa9 \xe2\x82\xac");
        w->write_attribute("wavelength", 500e-9);
        w->write_attribute("count", 3.0);
        w->write_attribute("title", "replaced");  // keeps its position
        w->write_attribute("neg_zero", -0.0);
        CHECK(fs::exists(json));  // written on every call, not only on destruction
    }
    CHECK(read_file(json) ==
          "{\n  \"title\": \"replaced\",\n  \"wavelength\": 5e-07,\n  \"count\": 3.0,\n"
          "  \"neg_zero\": -0.0\n}\n");

    TempDir tmp2;
    const auto w = io::open_npy_directory(core::path_to_utf8(tmp2.path()));
    w->write_attribute(
        "s", "say \"hi\" \\ C:\\dir\nline\ttab\x01\x1f\r \xc3\xa9 \xe2\x82\xac \xf0\x9f\x98\x80");
    CHECK(read_file(tmp2.path() / "attributes.json") ==
          "{\n  \"s\": \"say \\\"hi\\\" \\\\ C:\\\\dir\\nline\\ttab\\u0001\\u001f\\u000d \xc3\xa9 "
          "\xe2\x82\xac \xf0\x9f\x98\x80\"\n}\n");

    // Shortest round-trip form parses back to the identical double.
    std::mt19937_64 rng(42);
    for (int k = 0; k < 200; ++k) {
        Real x = 0.0;
        do {
            x = std::bit_cast<Real>(rng());
        } while (!std::isfinite(x));
        w->write_attribute("x", x);
        const std::string text = read_file(tmp2.path() / "attributes.json");
        const std::size_t pos = text.find("\"x\": ") + 5;
        const std::size_t end = text.find('\n', pos);
        Real back = 0.0;
        const auto res = std::from_chars(text.data() + pos, text.data() + end, back);
        REQUIRE(res.ec == std::errc());
        REQUIRE(res.ptr == text.data() + end);
        CHECK(bits(back) == bits(x));
    }
    CHECK_THROWS_AS(w->write_attribute("nan", std::numeric_limits<Real>::quiet_NaN()),
                    std::invalid_argument);
    CHECK_THROWS_AS(w->write_attribute("inf", std::numeric_limits<Real>::infinity()),
                    std::invalid_argument);
    for (const char* bad :
         {"\xff", "\xc0\xaf", "\xed\xa0\x80", "\xe2\x82", "a\x80", "\xf4\x90\x80\x80"}) {
        CHECK_THROWS_AS(w->write_attribute("bad", std::string(bad)), std::invalid_argument);
    }
}

TEST_CASE("result_writer: invalid names and paths throw", "[io]") {
    TempDir tmp;
    const auto w = io::open_npy_directory(core::path_to_utf8(tmp.path()));
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1.0, 0);
    for (const char* bad :
         {"",      "/abs", "a/",   "a//b",     ".",        "..",    "a/../b", "a.",
          "a b",   "a\\b", "C:x",  "\xc3\xa9", "a\nb",     "CON",   "nul",    "Aux",
          "prn.x", "COM1", "com9", "LPT1",     "lpt9.a.b", "g/CON", "Com5/x", "a/NUL.tar"}) {
        INFO("name '" << bad << "'");
        CHECK_THROWS_AS(w->write_vector(bad, VectorXc(1)), std::invalid_argument);
        CHECK_THROWS_AS(w->write_matrix(bad, MatrixXr(1, 1)), std::invalid_argument);
        CHECK_THROWS_AS(w->write_matrix(bad, MatrixXc(1, 1)), std::invalid_argument);
        CHECK_THROWS_AS(w->write_mesh(bad, mesh), std::invalid_argument);
        CHECK_THROWS_AS(w->write_attribute(bad, 1.0), std::invalid_argument);
        CHECK_THROWS_AS(w->write_attribute(bad, std::string("v")), std::invalid_argument);
    }
    w->write_vector("a.b-c_D/x.y", VectorXc(1));  // all allowed characters
    CHECK(fs::exists(tmp.path() / "a.b-c_D" / "x.y.npy"));
    // Device names only as the whole stem: these are ordinary names.
    for (const char* ok : {"CONX", "xcon", "COM10", "LPT", "a.con", "NULL", "aux_1"}) {
        INFO("name '" << ok << "'");
        CHECK_NOTHROW(w->write_vector(ok, VectorXc(1)));
        CHECK_NOTHROW(w->write_attribute(ok, 1.0));
    }
    CHECK(fs::exists(tmp.path() / "a.con.npy"));

    CHECK_THROWS_AS(io::open_npy_directory(""), std::invalid_argument);
    // A regular file where the directory should be: runtime_error naming the path.
    std::ofstream(tmp.path() / "plain_file") << "x";
    check_runtime_error(
        [&] { (void)io::open_npy_directory(core::path_to_utf8(tmp.path() / "plain_file")); },
        "plain_file");
}

TEST_CASE("result_writer: a target that is a directory throws and leaves no temporary file",
          "[io]") {
    TempDir tmp;
    const auto w = io::open_npy_directory(core::path_to_utf8(tmp.path()));
    fs::create_directories(tmp.path() / "x.npy");
    fs::create_directories(tmp.path() / "g" / "m.npy");
    fs::create_directories(tmp.path() / "mesh" / "triangles.npy");
    fs::create_directories(tmp.path() / "attributes.json");
    check_runtime_error([&] { w->write_vector("x", VectorXc(2)); }, "x.npy");
    check_runtime_error([&] { w->write_matrix("x", MatrixXr(2, 2)); }, "x.npy");
    check_runtime_error([&] { w->write_matrix("g/m", MatrixXc(2, 2)); }, "m.npy");
    check_runtime_error([&] { w->write_mesh("mesh", geometry::make_icosphere(1.0, 0)); },
                        "triangles.npy");
    check_runtime_error([&] { w->write_attribute("a", 1.0); }, "attributes.json");
    CHECK(fs::is_directory(tmp.path() / "x.npy"));
    CHECK(fs::is_directory(tmp.path() / "attributes.json"));
    CHECK(count_temp_files(tmp.path()) == 0);
}

TEST_CASE("result_writer: open_hdf5 is unavailable", "[io]") {
    try {
        (void)io::open_hdf5("out.h5");
        FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        CHECK_THAT(e.what(), ContainsSubstring("out.h5"));
#ifndef SPECKLEBEM_HAVE_HDF5
        CHECK_THAT(e.what(), ContainsSubstring("SPECKLEBEM_ENABLE_HDF5"));
#endif
    }
}

TEST_CASE("result_writer: .npy directory under a non-ASCII directory name (UTF-8 paths)", "[io]") {
    TempDir tmp;
    const std::string dir = core::path_to_utf8(tmp.path()) + "/" + test::non_ascii_dir_name() +
                            "/" + test::utf8(u8"Ergebnis_\u00e4");
    INFO("dir " << dir);
    VectorXc v(3);
    v << Complex(1.0, -2.0), Complex(0.5, 0.25), Complex(-3.0, 1e-9);
    {
        const auto w = io::open_npy_directory(dir);
        w->write_vector("currents", v);
        w->write_mesh("mesh", geometry::make_icosphere(1.0, 0));
        w->write_attribute("material", test::utf8(u8"Silber \u00b5m"));
        w->write_attribute("wavelength", 500e-9);
    }
    // The directory exists under the intended Unicode names (no code-page garbling).
    const fs::path root = core::path_from_utf8(dir);
    REQUIRE(fs::is_directory(root));
    std::vector<std::string> entries;
    for (const auto& e : fs::directory_iterator(tmp.path())) {
        entries.push_back(core::path_to_utf8(e.path().filename()));
    }
    CHECK(entries == std::vector<std::string>{test::non_ascii_dir_name()});
    const Npy nv = read_npy(root / "currents.npy");
    CHECK(nv.dict == "{'descr': '<c16', 'fortran_order': False, 'shape': (3,), }");
    REQUIRE(nv.data.size() == 48);
    for (std::size_t k = 0; k < 3; ++k) {
        CHECK(le64(nv.data, 2 * k) == bits(v(static_cast<Index>(k)).real()));
        CHECK(le64(nv.data, 2 * k + 1) == bits(v(static_cast<Index>(k)).imag()));
    }
    CHECK(fs::exists(root / "mesh" / "vertices.npy"));
    CHECK(fs::exists(root / "mesh" / "triangles.npy"));
    CHECK(read_file(root / "attributes.json") == "{\n  \"material\": \"" +
                                                     test::utf8(u8"Silber \u00b5m") +
                                                     "\",\n  \"wavelength\": 5e-07\n}\n");

    // Error messages name the exact UTF-8 path; invalid UTF-8 is rejected.
    const std::string blocked = dir + "/" + test::utf8(u8"Datei_\u00f6");
    std::ofstream(core::path_from_utf8(blocked)) << "x";
    check_runtime_error([&] { (void)io::open_npy_directory(blocked); }, blocked);
    CHECK_THROWS_AS(io::open_npy_directory("out_\xff"), std::invalid_argument);
}
