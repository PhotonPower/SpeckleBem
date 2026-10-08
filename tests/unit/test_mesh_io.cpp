#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/mesh_io.hpp"
#include "specklebem/geometry/sphere.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

using namespace specklebem;
using Catch::Matchers::ContainsSubstring;
using geometry::make_icosphere;
using geometry::TriangleMesh;
namespace io = geometry::io;
namespace fs = std::filesystem;

#ifndef SPECKLEBEM_TEST_DATA_DIR
#error "SPECKLEBEM_TEST_DATA_DIR must be defined by tests/CMakeLists.txt"
#endif

namespace {

std::string fixture(const std::string& name) {
    return std::string(SPECKLEBEM_TEST_DATA_DIR) + "/" + name;
}

/// Unique file in the system temp directory, removed on destruction.
class TempFile {
public:
    explicit TempFile(const std::string& extension) {
        static std::atomic<unsigned> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto salt = std::random_device{}();  // uniqueness only, not test input
        path_ = (fs::temp_directory_path() /
                 ("specklebem_mesh_io_" + std::to_string(stamp) + "_" + std::to_string(salt) + "_" +
                  std::to_string(counter++) + extension))
                    .string();
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    ~TempFile() {
        std::error_code ec;
        fs::remove(path_, ec);
        fs::remove(path_ + ".tmp", ec);  // writer's temporary file, should never remain
    }
    [[nodiscard]] const std::string& path() const { return path_; }

private:
    std::string path_;
};

void write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    REQUIRE(out.good());
}

/// Message of the std::runtime_error thrown by f, or a marker if none / another type.
std::string runtime_error_message(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::runtime_error& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("<other exception: ") + e.what() + ">";
    }
    return "<no exception>";
}

/// Largest vertex distance from the origin (length scale for relative tolerances).
Real vertex_scale(const TriangleMesh& m) {
    Real s = 0.0;
    for (Index i = 0; i < m.num_vertices(); ++i) s = std::max(s, m.vertices().row(i).norm());
    return s;
}

/// Largest deviation of corner k of triangle t between the two meshes, relative to the
/// vertex scale of `ref`; +inf if the triangle counts differ.
Real max_corner_deviation(const TriangleMesh& ref, const TriangleMesh& other) {
    if (ref.num_triangles() != other.num_triangles())
        return std::numeric_limits<Real>::infinity();
    Real dev = 0.0;
    for (Index t = 0; t < ref.num_triangles(); ++t) {
        for (Index k = 0; k < 3; ++k) {
            const auto a = ref.vertices().row(ref.triangles()(t, k));
            const auto b = other.vertices().row(other.triangles()(t, k));
            dev = std::max(dev, (a - b).norm());
        }
    }
    return dev / vertex_scale(ref);
}

struct Format {
    const char* name;
    const char* extension;
    std::function<void(const TriangleMesh&, const std::string&)> write;
    Real tolerance;  ///< documented acceptance number
    bool bit_exact;  ///< shortest round-trip doubles: expect zero deviation
};

}  // namespace

TEST_CASE("mesh_io: icosphere round trip preserves triangles, vertices and orientation",
          "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1e-6, 2);
    REQUIRE(sphere.num_triangles() == 320);
    REQUIRE(sphere.num_vertices() == 162);

    // Binary STL stores float32 (relative rounding ~6e-8): 1e-6 is the accepted deviation
    // from the 1e-12 criterion. The text formats write shortest round-trip doubles.
    const std::array<Format, 4> formats{{
        {"binary STL", ".stl",
         [](const TriangleMesh& m, const std::string& p) { io::write_mesh(m, p); }, 1e-6, false},
        {"ASCII STL", ".stl",
         [](const TriangleMesh& m, const std::string& p) {
             io::write_stl(m, p, io::StlFormat::ascii);
         },
         1e-12, true},
        {"OBJ", ".obj", [](const TriangleMesh& m, const std::string& p) { io::write_mesh(m, p); },
         1e-12, true},
        {"Gmsh 4.1", ".msh",
         [](const TriangleMesh& m, const std::string& p) { io::write_mesh(m, p); }, 1e-12, true},
    }};
    const auto& fmt = formats[GENERATE(0U, 1U, 2U, 3U)];
    INFO("format: " << fmt.name);

    TempFile file(fmt.extension);
    fmt.write(sphere, file.path());
    const TriangleMesh read = io::read_mesh(file.path());

    CHECK(read.num_triangles() == sphere.num_triangles());
    CHECK(read.num_vertices() == sphere.num_vertices());  // welded for STL
    CHECK(max_corner_deviation(sphere, read) <= fmt.tolerance);
    if (fmt.bit_exact) {
        CHECK(max_corner_deviation(sphere, read) == 0.0);
        CHECK(read.signed_volume() == sphere.signed_volume());
    }
    CHECK(read.is_closed());
    CHECK(read.is_consistently_oriented());
    CHECK(std::abs(read.signed_volume() - sphere.signed_volume()) <=
          fmt.tolerance * std::abs(sphere.signed_volume()));
    CHECK(read.signed_volume() > 0.0);
    CHECK(read.num_boundary_edges() == 0);
}

TEST_CASE("mesh_io: write_mesh with .stl writes binary STL", "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1e-6, 1);
    TempFile file(".stl");
    io::write_mesh(sphere, file.path());
    CHECK(fs::file_size(file.path()) ==
          84 + 50 * static_cast<std::uintmax_t>(sphere.num_triangles()));
    std::ifstream in(file.path(), std::ios::binary);
    std::string head(5, '\0');
    in.read(head.data(), 5);
    CHECK(head != "solid");
}

TEST_CASE("mesh_io: tetrahedron fixtures agree across formats", "[geometry][io]") {
    const TriangleMesh stl = io::read_mesh(fixture("tetra_ascii.stl"));
    const TriangleMesh obj = io::read_mesh(fixture("tetra.obj"));
    const TriangleMesh msh = io::read_mesh(fixture("tetra_v41.msh"));
    const Real expected_volume = 1e-9 / 6.0;  // legs of 1 mm
    for (const TriangleMesh* m : {&stl, &obj, &msh}) {
        CHECK(m->num_vertices() == 4);  // Gmsh: unreferenced point node dropped
        CHECK(m->num_triangles() == 4);
        CHECK(m->is_closed());
        CHECK(m->signed_volume() > 0.0);
        CHECK(std::abs(m->signed_volume() - expected_volume) <= 1e-12 * expected_volume);
    }
    CHECK(max_corner_deviation(obj, stl) <= 1e-12);
    CHECK(max_corner_deviation(obj, msh) <= 1e-12);
    // OBJ keeps the file's vertex numbering and the in-triangle vertex order.
    CHECK(obj.triangles()(0, 0) == 0);
    CHECK(obj.triangles()(0, 1) == 2);
    CHECK(obj.triangles()(0, 2) == 1);
}

TEST_CASE("mesh_io: extension dispatch and file errors", "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1.0, 0);
    const TempFile foo(".foo");
    const TempFile no_extension("");
    const TempFile empty_out(".obj");
    CHECK_THROWS_AS(io::read_mesh(foo.path()), std::invalid_argument);
    CHECK_THROWS_AS(io::read_mesh(no_extension.path()), std::invalid_argument);
    CHECK_THROWS_AS(io::write_mesh(sphere, foo.path()), std::invalid_argument);
    CHECK_THROWS_AS(io::write_mesh(TriangleMesh(), empty_out.path()), std::invalid_argument);
    CHECK_FALSE(fs::exists(foo.path()));
    CHECK_FALSE(fs::exists(empty_out.path()));

    const std::string missing =
        (fs::temp_directory_path() / "specklebem_missing_mesh.stl").string();
    CHECK_THAT(runtime_error_message([&] { (void)io::read_mesh(missing); }),
               ContainsSubstring("specklebem_missing_mesh.stl"));
    CHECK_THROWS_AS(io::read_obj(missing), std::runtime_error);
    CHECK_THROWS_AS(io::read_gmsh(missing), std::runtime_error);

    // Upper-case extensions are dispatched too.
    TempFile file(".OBJ");
    io::write_mesh(sphere, file.path());
    CHECK(io::read_mesh(file.path()).num_triangles() == 20);
}

TEST_CASE("mesh_io: malformed fixtures throw runtime_error naming the file", "[geometry][io]") {
    const std::string stl_msg =
        runtime_error_message([] { (void)io::read_mesh(fixture("broken_count.stl")); });
    CHECK_THAT(stl_msg, ContainsSubstring("broken_count.stl"));
    CHECK_THAT(stl_msg, ContainsSubstring("triangle count 5"));

    const std::string obj_msg =
        runtime_error_message([] { (void)io::read_mesh(fixture("broken_face.obj")); });
    CHECK_THAT(obj_msg, ContainsSubstring("broken_face.obj"));
    CHECK_THAT(obj_msg, ContainsSubstring("line 9"));
    CHECK_THAT(obj_msg, ContainsSubstring("out of range"));

    const std::string msh_msg =
        runtime_error_message([] { (void)io::read_mesh(fixture("broken_version.msh")); });
    CHECK_THAT(msh_msg, ContainsSubstring("broken_version.msh"));
    CHECK_THAT(msh_msg, ContainsSubstring("2.2"));
}

TEST_CASE("mesh_io: OBJ face forms, polygons and errors", "[geometry][io]") {
    const std::string verts = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n";

    SECTION("a/b/c face form") {
        TempFile file(".obj");
        write_text(file.path(), verts + "f 1/1/1 2/2/2 3/3/3\n");
        const TriangleMesh m = io::read_obj(file.path());
        CHECK(m.num_triangles() == 1);
        CHECK(m.num_vertices() == 4);  // unreferenced OBJ vertices are kept
        CHECK(m.triangles()(0, 0) == 0);
        CHECK(m.triangles()(0, 1) == 1);
        CHECK(m.triangles()(0, 2) == 2);
    }
    SECTION("quad is fan-triangulated") {
        TempFile file(".obj");
        write_text(file.path(), verts + "f 1 2 3 4\n");
        const TriangleMesh m = io::read_obj(file.path());
        REQUIRE(m.num_triangles() == 2);
        CHECK(m.triangles()(0, 0) == 0);
        CHECK(m.triangles()(0, 1) == 1);
        CHECK(m.triangles()(0, 2) == 2);
        CHECK(m.triangles()(1, 0) == 0);
        CHECK(m.triangles()(1, 1) == 2);
        CHECK(m.triangles()(1, 2) == 3);
        CHECK(m.num_boundary_edges() == 4);
    }
    SECTION("negative index") {
        TempFile file(".obj");
        write_text(file.path(), verts + "f -4 -3 -2\n");
        CHECK_THAT(runtime_error_message([&] { (void)io::read_obj(file.path()); }),
                   ContainsSubstring("line 5"));
    }
    SECTION("zero index, short face, bad number, no faces") {
        for (const std::string& body :
             {verts + "f 0 1 2\n", verts + "f 1 2\n", "v 0 0 x\n" + verts + "f 1 2 3\n", verts}) {
            TempFile file(".obj");
            write_text(file.path(), body);
            CHECK_THROWS_AS(io::read_obj(file.path()), std::runtime_error);
        }
    }
    SECTION("degenerate triangle is reported with the file name") {
        TempFile file(".obj");
        write_text(file.path(), verts + "f 1 2 2\n");
        CHECK_THAT(runtime_error_message([&] { (void)io::read_obj(file.path()); }),
                   ContainsSubstring(fs::path(file.path()).filename().string()));
    }
    SECTION("inconsistent winding is repaired by the TriangleMesh constructor") {
        TempFile file(".obj");
        write_text(file.path(),
                   "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                   "f 1 2 3\nf 1 2 4\nf 1 4 3\nf 2 3 4\n");  // first face flipped
        const TriangleMesh m = io::read_obj(file.path());
        CHECK(m.is_closed());
        CHECK(m.is_consistently_oriented());
        CHECK(m.signed_volume() > 0.0);
    }
}

TEST_CASE("mesh_io: Gmsh element types and format checks", "[geometry][io]") {
    const std::string head = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n";
    const std::string nodes =
        "$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n";

    SECTION("tetrahedron element") {
        TempFile file(".msh");
        write_text(file.path(),
                   head + nodes + "$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n");
        const std::string msg = runtime_error_message([&] { (void)io::read_gmsh(file.path()); });
        CHECK_THAT(msg, ContainsSubstring("element type 4"));
        CHECK_THAT(msg, ContainsSubstring("tetrahedron"));
    }
    SECTION("binary MSH 4.1 is rejected with the version in the message") {
        TempFile file(".msh");
        write_text(file.path(), "$MeshFormat\n4.1 1 8\n$EndMeshFormat\n");
        CHECK_THAT(runtime_error_message([&] { (void)io::read_gmsh(file.path()); }),
                   ContainsSubstring("4.1 1 8"));
    }
    SECTION("unknown node tag") {
        TempFile file(".msh");
        write_text(file.path(),
                   head + nodes + "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 7\n$EndElements\n");
        CHECK_THAT(runtime_error_message([&] { (void)io::read_gmsh(file.path()); }),
                   ContainsSubstring("unknown node tag 7"));
    }
    SECTION("truncated file") {
        TempFile file(".msh");
        write_text(file.path(), head + nodes + "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2\n");
        CHECK_THROWS_AS(io::read_gmsh(file.path()), std::runtime_error);
    }
}

TEST_CASE("mesh_io: binary STL whose header starts with 'solid' is read as binary",
          "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1e-6, 2);
    TempFile file(".stl");
    io::write_stl(sphere, file.path(), io::StlFormat::binary);
    {
        std::fstream f(file.path(), std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0);
        f.write("solid", 5);
        REQUIRE(f.good());
    }
    const TriangleMesh read = io::read_stl(file.path());
    CHECK(read.num_triangles() == 320);
    CHECK(read.num_vertices() == 162);
    CHECK(max_corner_deviation(sphere, read) <= 1e-6);
}

TEST_CASE("mesh_io: STL import welds shared vertices exactly", "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1e-6, 2);
    const auto format = GENERATE(io::StlFormat::binary, io::StlFormat::ascii);
    TempFile file(".stl");
    io::write_stl(sphere, file.path(), format);
    const TriangleMesh read = io::read_stl(file.path());
    CHECK(read.num_vertices() == 162);
    CHECK(read.num_edges() == sphere.num_edges());
    CHECK(read.is_closed());
}

TEST_CASE("mesh_io: ASCII STL errors carry the line number", "[geometry][io]") {
    TempFile file(".stl");
    write_text(file.path(),
               "solid s\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n"
               "   vertex 1 0\n  endloop\n endfacet\nendsolid s\n");
    const std::string msg = runtime_error_message([&] { (void)io::read_stl(file.path()); });
    CHECK_THAT(msg, ContainsSubstring(fs::path(file.path()).filename().string()));
    CHECK_THAT(msg, ContainsSubstring("line 6"));
}

TEST_CASE("mesh_io: 20480-triangle binary STL round trip", "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1e-6, 5);
    REQUIRE(sphere.num_triangles() == 20480);
    TempFile file(".stl");
    io::write_mesh(sphere, file.path());
    const TriangleMesh read = io::read_mesh(file.path());
    CHECK(read.num_triangles() == 20480);
    CHECK(read.num_vertices() == sphere.num_vertices());
    CHECK(read.num_edges() == sphere.num_edges());
    CHECK(read.is_closed());
    CHECK(max_corner_deviation(sphere, read) <= 1e-6);
}

namespace {

/// Little-endian binary STL with the given stored count and `records` copies of a triangle.
std::string binary_stl(std::uint32_t stored_count, std::uint32_t records,
                       const std::string& header = "test header") {
    std::string data = header;
    data.resize(80, ' ');
    for (unsigned i = 0; i < 4; ++i)
        data.push_back(static_cast<char>((stored_count >> (8 * i)) & 0xFFU));
    const std::array<float, 12> rec{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                    1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    for (std::uint32_t r = 0; r < records; ++r) {
        for (const float f : rec) {
            std::uint32_t u = 0;
            std::memcpy(&u, &f, sizeof u);
            for (unsigned i = 0; i < 4; ++i)
                data.push_back(static_cast<char>((u >> (8 * i)) & 0xFFU));
        }
        data.append(2, '\0');
    }
    return data;
}

const std::string kTetraStl =
    "solid t\n"
    "facet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex 1 0 0\nendloop\nendfacet\n"
    "facet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 0 1\nendloop\nendfacet\n"
    "facet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 0 1\nvertex 0 1 0\nendloop\nendfacet\n"
    "facet normal 0 0 0\nouter loop\nvertex 1 0 0\nvertex 0 1 0\nvertex 0 0 1\nendloop\nendfacet\n"
    "endsolid t\n";
const std::string kTetraObj =
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n";
const std::string kTetraMsh =
    "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
    "$Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
    "$Elements\n1 4 1 4\n2 1 2 4\n1 1 3 2\n2 1 2 4\n3 1 4 3\n4 2 3 4\n$EndElements\n";

/// Replaces every "\n" by `eol`.
std::string with_line_endings(const std::string& text, const std::string& eol) {
    std::string out;
    for (const char c : text) {
        if (c == '\n') {
            out += eol;
        } else {
            out += c;
        }
    }
    return out;
}

/// Reads `text` written to a temporary file with the given extension.
TriangleMesh read_text(const std::string& extension, const std::string& text) {
    TempFile file(extension);
    write_text(file.path(), text);
    return io::read_mesh(file.path());
}

void check_unit_tetra(const TriangleMesh& m) {
    CHECK(m.num_vertices() == 4);
    CHECK(m.num_triangles() == 4);
    CHECK(m.is_closed());
    CHECK(std::abs(m.signed_volume() - 1.0 / 6.0) <= 1e-12);
}

}  // namespace

TEST_CASE("mesh_io: empty files are rejected", "[geometry][io]") {
    const std::string ext = GENERATE(as<std::string>{}, ".stl", ".obj", ".msh");
    INFO("extension " << ext);
    TempFile file(ext);
    write_text(file.path(), "");
    CHECK_THAT(runtime_error_message([&] { (void)io::read_mesh(file.path()); }),
               ContainsSubstring(fs::path(file.path()).filename().string()));
}

TEST_CASE("mesh_io: binary STL with an implausible triangle count", "[geometry][io]") {
    SECTION("count 0 with no records") {
        TempFile file(".stl");
        write_text(file.path(), binary_stl(0, 0));
        CHECK_THAT(runtime_error_message([&] { (void)io::read_stl(file.path()); }),
                   ContainsSubstring("no triangles"));
    }
    SECTION("count 0xFFFFFFFF") {
        TempFile file(".stl");
        write_text(file.path(), binary_stl(0xFFFFFFFFU, 1));
        CHECK_THAT(runtime_error_message([&] { (void)io::read_stl(file.path()); }),
                   ContainsSubstring("triangle count 4294967295"));
    }
    SECTION("truncated binary STL whose header starts with 'solid'") {
        TempFile file(".stl");
        write_text(file.path(), binary_stl(4, 3, "solid binary header"));
        const std::string msg = runtime_error_message([&] { (void)io::read_stl(file.path()); });
        CHECK_THAT(msg, ContainsSubstring(fs::path(file.path()).filename().string()));
        CHECK_THAT(msg, ContainsSubstring("triangle count 4"));
        CHECK_THAT(msg, ContainsSubstring("234 bytes"));
    }
    SECTION("valid binary records") {
        TempFile file(".stl");
        write_text(file.path(), binary_stl(1, 1));
        const TriangleMesh m = io::read_stl(file.path());
        CHECK(m.num_triangles() == 1);
        CHECK(m.num_vertices() == 3);
    }
}

TEST_CASE("mesh_io: ASCII STL missing endfacet reports a line number", "[geometry][io]") {
    TempFile file(".stl");
    write_text(file.path(),
               "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
               "vertex 0 1 0\nendloop\nendsolid s\n");
    const std::string msg = runtime_error_message([&] { (void)io::read_stl(file.path()); });
    CHECK_THAT(msg, ContainsSubstring("line 8"));
    CHECK_THAT(msg, ContainsSubstring("endfacet"));
}

TEST_CASE("mesh_io: CRLF, CR-only and missing final newline are accepted", "[geometry][io]") {
    const std::string eol = GENERATE(as<std::string>{}, "\n", "\r\n", "\r");
    const bool final_newline = GENERATE(true, false);
    INFO("CR count in eol: " << std::count(eol.begin(), eol.end(), '\r')
                             << ", final newline: " << final_newline);
    const std::array<std::pair<std::string, std::string>, 3> cases{
        {{".stl", kTetraStl}, {".obj", kTetraObj}, {".msh", kTetraMsh}}};
    for (const auto& [ext, text] : cases) {
        INFO("format " << ext);
        std::string body = with_line_endings(text, eol);
        if (!final_newline)
            body.resize(body.size() - eol.size());
        check_unit_tetra(read_text(ext, body));
    }
}

TEST_CASE("mesh_io: CR-only files report correct line numbers", "[geometry][io]") {
    TempFile file(".obj");
    write_text(file.path(), "v 0 0 0\rv 1 0 0\rv 0 1 0\rf 1 2 9\r");
    CHECK_THAT(runtime_error_message([&] { (void)io::read_obj(file.path()); }),
               ContainsSubstring("line 4"));
}

TEST_CASE("mesh_io: ASCII STL with a UTF-8 byte-order mark", "[geometry][io]") {
    check_unit_tetra(read_text(".stl", "\xEF\xBB\xBF" + kTetraStl));
}

TEST_CASE("mesh_io: OBJ comment directly after a face index", "[geometry][io]") {
    check_unit_tetra(
        read_text(".obj",
                  "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2# c\nf 1 2 4 #c\nf 1 4 3#\n"
                  "f 2/1 3/1 4/1#comment 9 9\n"));
}

TEST_CASE("mesh_io: OBJ indices beyond the integer range are rejected", "[geometry][io]") {
    const std::string index = GENERATE(as<std::string>{}, "99999999999999999999",
                                       "9223372036854775807", "-9223372036854775808");
    INFO("index " << index);
    TempFile file(".obj");
    write_text(file.path(), "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 " + index + "\n");
    CHECK_THAT(runtime_error_message([&] { (void)io::read_obj(file.path()); }),
               ContainsSubstring("line 4"));
}

TEST_CASE("mesh_io: Gmsh with a negative node count is rejected", "[geometry][io]") {
    TempFile file(".msh");
    write_text(file.path(), "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 -4 1 4\n$EndNodes\n");
    CHECK_THAT(runtime_error_message([&] { (void)io::read_gmsh(file.path()); }),
               ContainsSubstring("negative node count"));
}

TEST_CASE("mesh_io: failed write throws and leaves no partial file", "[geometry][io]") {
    const TriangleMesh sphere = make_icosphere(1.0, 1);
    const TempFile dir_name("");
    const std::string ext = GENERATE(as<std::string>{}, ".stl", ".obj", ".msh");
    const std::string target = (fs::path(dir_name.path()) / ("mesh" + ext)).string();
    INFO("target " << target);
    REQUIRE_FALSE(fs::exists(dir_name.path()));  // parent directory does not exist
    CHECK_THAT(runtime_error_message([&] { io::write_mesh(sphere, target); }),
               ContainsSubstring(target));
    CHECK_FALSE(fs::exists(target));
    CHECK_FALSE(fs::exists(target + ".tmp"));
}

TEST_CASE("mesh_io: writing replaces an existing file and removes the temporary",
          "[geometry][io]") {
    TempFile file(".obj");
    write_text(file.path(), "stale content that is longer than nothing\n");
    const TriangleMesh sphere = make_icosphere(1.0, 0);
    io::write_mesh(sphere, file.path());
    CHECK_FALSE(fs::exists(file.path() + ".tmp"));
    CHECK(io::read_mesh(file.path()).num_triangles() == 20);
}
