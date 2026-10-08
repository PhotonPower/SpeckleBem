#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace specklebem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using geometry::generate_gaussian_height_map;
using geometry::HeightMap;
using geometry::make_mesh_from_height_map;
using geometry::make_rough_surface_mesh;
using geometry::RoughSurfaceParams;
using geometry::TriangleMesh;

namespace {

RoughSurfaceParams make_params(Real L, Real mesh_size, Real sigma, Real lc, std::uint64_t seed) {
    RoughSurfaceParams p;
    p.edge_length_L = L;
    p.mesh_size = mesh_size;
    p.rms_roughness = sigma;
    p.correlation_length = lc;
    p.seed = seed;
    return p;
}

/// Value of an integer line "label : value" in the quality report, or -1 if absent.
long report_value(const std::string& report, const std::string& label) {
    const std::regex re(label + R"(\s*:\s*(-?\d+))");
    std::smatch m;
    if (!std::regex_search(report, m, re))
        return -1;
    return std::stol(m[1].str());
}

/// Unnormalised normal (v1 - v0) x (v2 - v0) of row t of the raw arrays.
Vec3 raw_normal(const Vertices& v, const Triangles& f, Index t) {
    const Vec3 a = v.row(f(t, 0)).transpose();
    const Vec3 b = v.row(f(t, 1)).transpose();
    const Vec3 c = v.row(f(t, 2)).transpose();
    return (b - a).cross(c - a);
}

Vec3 raw_centroid(const Vertices& v, const Triangles& f, Index t) {
    return (v.row(f(t, 0)) + v.row(f(t, 1)) + v.row(f(t, 2))).transpose() / 3.0;
}

/// Per-face orientation tallies of a box mesh around the height map h.
struct FaceCheck {
    Index top = 0, bottom = 0, wall = 0;
    bool top_ok = true, bottom_ok = true, wall_ok = true;
};

/// Classifies the triangles of the raw box arrays by centroid (walls: on the patch rim;
/// bottom: at z = depth; top: everything else) and checks the normal directions.
FaceCheck check_faces(const Vertices& v, const Triangles& f, const HeightMap& h, Real depth) {
    const Real hx = 0.5 * static_cast<Real>(h.z.rows() - 1) * h.dx;
    const Real hy = 0.5 * static_cast<Real>(h.z.cols() - 1) * h.dy;
    const Real tol = 1e-6 * std::min(h.dx, h.dy);
    const Vec3 box_centre(0.0, 0.0, 0.5 * depth);
    FaceCheck fc;
    for (Index t = 0; t < f.rows(); ++t) {
        const Vec3 c = raw_centroid(v, f, t);
        const Vec3 n = raw_normal(v, f, t).normalized();
        if (std::abs(c.x()) > hx - tol || std::abs(c.y()) > hy - tol) {
            ++fc.wall;
            fc.wall_ok = fc.wall_ok && std::abs(n.z()) < 1e-12 && n.dot(c - box_centre) > 0.0;
        } else if (c.z() == depth) {
            ++fc.bottom;
            fc.bottom_ok = fc.bottom_ok && n.z() > 0.0;
        } else {
            ++fc.top;
            fc.top_ok = fc.top_ok && n.z() < 0.0 && c.z() < h.z.maxCoeff() + tol;
        }
    }
    return fc;
}

/// True if every grid point (x_i, y_j) carries a vertex and the lowest vertex there has
/// z = h.z(i, j) exactly (the top face reproduces the height map).
bool top_face_reproduces(const TriangleMesh& mesh, const HeightMap& h) {
    const Index nx = h.z.rows();
    const Index ny = h.z.cols();
    const Real x0 = -0.5 * static_cast<Real>(nx - 1) * h.dx;
    const Real y0 = -0.5 * static_cast<Real>(ny - 1) * h.dy;
    const Real tol = 1e-9 * std::min(h.dx, h.dy);
    std::map<std::pair<Index, Index>, Real> zmin;
    for (Index k = 0; k < mesh.num_vertices(); ++k) {
        const Real x = mesh.vertices()(k, 0);
        const Real y = mesh.vertices()(k, 1);
        const auto i = static_cast<Index>(std::llround((x - x0) / h.dx));
        const auto j = static_cast<Index>(std::llround((y - y0) / h.dy));
        if (i < 0 || i >= nx || j < 0 || j >= ny)
            return false;
        if (std::abs(x - (x0 + static_cast<Real>(i) * h.dx)) > tol ||
            std::abs(y - (y0 + static_cast<Real>(j) * h.dy)) > tol)
            return false;
        const auto key = std::make_pair(i, j);
        const Real z = mesh.vertices()(k, 2);
        const auto it = zmin.find(key);
        if (it == zmin.end())
            zmin.emplace(key, z);
        else
            it->second = std::min(it->second, z);
    }
    if (static_cast<Index>(zmin.size()) != nx * ny)
        return false;
    for (const auto& [key, z] : zmin) {
        if (z != h.z(key.first, key.second))
            return false;
    }
    return true;
}

/// Expected counts of the closed box (see rough_surface.cpp): n_rim = 2(n_x-1) + 2(n_y-1),
/// V = 2 n_x n_y + n_rim (n_z - 1), F = 4 (n_x-1)(n_y-1) + 2 n_rim n_z, E = 3F/2.
struct BoxCounts {
    Index v, f, e;
};
BoxCounts box_counts(Index nx, Index ny, Index nz) {
    const Index n_rim = 2 * (nx - 1) + 2 * (ny - 1);
    const Index f = 4 * (nx - 1) * (ny - 1) + 2 * n_rim * nz;
    return {2 * nx * ny + n_rim * (nz - 1), f, 3 * f / 2};
}

/// Regression reference (docs/05): seed 42, L = 10 um, mesh 50 nm, sigma 50 nm, Lc 500 nm.
RoughSurfaceParams regression_params() {
    return make_params(10e-6, 50e-9, 50e-9, 500e-9, 42);
}
constexpr std::array<Index, 4> kRegressionLattice = {0, 67, 133, 200};

/// Full-map checksums: sum z, sum z^2 and sum (i + 2 j) z (the weight makes the last one
/// sensitive to transposed or permuted maps).
struct Checksums {
    Real sum_z = 0.0;
    Real sum_z2 = 0.0;
    Real sum_ij_z = 0.0;
};
Checksums checksums(const HeightMap& h) {
    Checksums c;
    for (Index j = 0; j < h.z.cols(); ++j) {
        for (Index i = 0; i < h.z.rows(); ++i) {
            const Real z = h.z(i, j);
            c.sum_z += z;
            c.sum_z2 += z * z;
            c.sum_ij_z += static_cast<Real>(i + 2 * j) * z;
        }
    }
    return c;
}

/// Content of tests/data/rough_surface_seed42_L10um.txt for the current generator.
std::string regression_reference_text() {
    const HeightMap h = generate_gaussian_height_map(regression_params());
    const Checksums c = checksums(h);
    std::string out =
        "# SpeckleBem rough-surface regression reference (docs/05_validation.md).\n"
        "# generate_gaussian_height_map with seed = 42, edge_length_L = 10e-6 m, mesh_size = "
        "50e-9 m,\n"
        "# rms_roughness = 50e-9 m, correlation_length = 500e-9 m, use_fft = true -> " +
        std::to_string(h.z.rows()) + " x " + std::to_string(h.z.cols()) +
        " grid.\n"
        "# Values in SI units (metres), 15 significant digits. Lines: 'rms <HeightMap::rms()>',\n"
        "# 'correlation_length <HeightMap::estimated_correlation_length()>', full-map checksums\n"
        "# 'sum_z <sum z>', 'sum_z2 <sum z^2>', 'sum_ij_z <sum (i + 2 j) z>', and 'z <i> <j> "
        "<z(i,j)>'.\n"
        "# A generator change that alters these numbers must be deliberate. Regenerate (from the\n"
        "# repository root) with\n"
        "#   SPECKLEBEM_REGEN_OUTPUT=tests/data/rough_surface_seed42_L10um.txt "
        "build/release/tests/specklebem_unit_tests \"[.regen]\"\n";
    const auto line = [&out](const char* key, Real value) {
        std::array<char, 64> buf{};
        std::snprintf(buf.data(), buf.size(), "%.14e", value);
        out += std::string(key) + " " + buf.data() + "\n";
    };
    line("rms", h.rms());
    line("correlation_length", h.estimated_correlation_length());
    line("sum_z", c.sum_z);
    line("sum_z2", c.sum_z2);
    line("sum_ij_z", c.sum_ij_z);
    for (const Index i : kRegressionLattice) {
        for (const Index j : kRegressionLattice) {
            line(("z " + std::to_string(i) + " " + std::to_string(j)).c_str(), h.z(i, j));
        }
    }
    return out;
}

}  // namespace

TEST_CASE("rough surface: FFT and direct convolution agree", "[geometry]") {
    // 32 x 32 cells (33 x 33 points), kernel radius 4 Lc = 16 cells.
    RoughSurfaceParams p = make_params(1.6e-6, 50e-9, 20e-9, 200e-9, 7);
    p.use_fft = true;
    const HeightMap fft = generate_gaussian_height_map(p);
    p.use_fft = false;
    const HeightMap direct = generate_gaussian_height_map(p);
    REQUIRE(fft.z.rows() == 33);
    REQUIRE(fft.z.cols() == 33);
    REQUIRE(direct.z.rows() == 33);
    REQUIRE(direct.z.cols() == 33);
    CHECK_THAT(fft.dx, WithinRel(50e-9, 1e-14));
    CHECK(fft.dx == fft.dy);
    const Real max_diff = (fft.z - direct.z).cwiseAbs().maxCoeff();
    CHECK(max_diff / p.rms_roughness < 1e-10);
    // Not trivially zero.
    CHECK(fft.rms() > 0.3 * p.rms_roughness);
}

TEST_CASE("rough surface: height statistics over 20 seeds (docs/05)", "[geometry]") {
    // docs/05_validation.md: rms within 2 %, Lc within 5 % (20 seeds). L = 20 um,
    // mesh 100 nm (201 x 201 grid, Lc / dx = 5, L / Lc = 40).
    const Real sigma = 50e-9;
    const Real lc = 500e-9;
    Real sum_rms2 = 0.0;
    Real sum_lc = 0.0;
    Real sum_mean = 0.0;
    Real max_rms_dev = 0.0;
    Real max_abs_mean = 0.0;
    constexpr int kSeeds = 20;
    for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
        const HeightMap h =
            generate_gaussian_height_map(make_params(20e-6, 100e-9, sigma, lc, seed));
        REQUIRE(h.z.rows() == 201);
        REQUIRE(h.z.cols() == 201);
        const Real rms = h.rms();
        const Real mean = h.z.mean();
        sum_rms2 += rms * rms;
        sum_lc += h.estimated_correlation_length();
        sum_mean += mean;
        max_rms_dev = std::max(max_rms_dev, std::abs(rms / sigma - 1.0));
        max_abs_mean = std::max(max_abs_mean, std::abs(mean));
    }
    const Real ensemble_rms = std::sqrt(sum_rms2 / kSeeds);
    const Real mean_lc = sum_lc / kSeeds;
    const Real ensemble_mean = sum_mean / kSeeds;
    CAPTURE(ensemble_rms / sigma, mean_lc / lc, max_rms_dev, ensemble_mean / sigma,
            max_abs_mean / sigma);
    CHECK_THAT(ensemble_rms, WithinRel(sigma, 0.02));
    CHECK_THAT(mean_lc, WithinRel(lc, 0.05));
    CHECK(max_rms_dev < 0.10);
    CHECK(std::abs(ensemble_mean) < 0.1 * sigma);
    CHECK(max_abs_mean < 0.1 * sigma);
}

TEST_CASE("rough surface: determinism of the seed", "[geometry]") {
    const RoughSurfaceParams p = make_params(2e-6, 50e-9, 30e-9, 300e-9, 12345);
    const HeightMap a = generate_gaussian_height_map(p);
    const HeightMap b = generate_gaussian_height_map(p);
    CHECK(a.z == b.z);
    RoughSurfaceParams q = p;
    q.seed = 12346;
    const HeightMap c = generate_gaussian_height_map(q);
    CHECK((a.z - c.z).cwiseAbs().maxCoeff() > 0.1 * p.rms_roughness);
    q.seed = 0;
    const HeightMap d = generate_gaussian_height_map(q);
    const HeightMap e = generate_gaussian_height_map(q);
    CHECK((d.z - e.z).cwiseAbs().maxCoeff() > 0.1 * p.rms_roughness);
}

TEST_CASE("rough surface: regression for seed 42, L = 10 um (docs/05)", "[geometry]") {
    // Reference written by the hidden "[.regen]" test case below.
    const std::string path =
        std::string(SPECKLEBEM_TEST_DATA_DIR) + "/rough_surface_seed42_L10um.txt";
    std::ifstream in(path);
    REQUIRE(in.good());
    std::map<std::string, Real> ref;
    std::vector<std::pair<std::pair<Index, Index>, Real>> ref_z;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "z") {
            Index i = 0;
            Index j = 0;
            Real z = 0.0;
            ls >> i >> j >> z;
            ref_z.push_back({{i, j}, z});
        } else {
            Real value = 0.0;
            ls >> value;
            ref[key] = value;
        }
        REQUIRE_FALSE(ls.fail());
    }
    for (const char* key : {"rms", "correlation_length", "sum_z", "sum_z2", "sum_ij_z"}) {
        CAPTURE(key);
        REQUIRE(ref.count(key) == 1);
    }
    REQUIRE(ref_z.size() == 16);

    const RoughSurfaceParams p = regression_params();
    const Real sigma = p.rms_roughness;
    const HeightMap h = generate_gaussian_height_map(p);
    REQUIRE(h.z.rows() == 201);
    REQUIRE(h.z.cols() == 201);
    CHECK_THAT(h.rms(), WithinRel(ref["rms"], 1e-12));
    CHECK_THAT(h.estimated_correlation_length(), WithinRel(ref["correlation_length"], 1e-12));
    const Checksums c = checksums(h);
    CHECK_THAT(c.sum_z, WithinRel(ref["sum_z"], 1e-12));
    CHECK_THAT(c.sum_z2, WithinRel(ref["sum_z2"], 1e-12));
    CHECK_THAT(c.sum_ij_z, WithinRel(ref["sum_ij_z"], 1e-12));
    for (const auto& [ij, z] : ref_z) {
        CAPTURE(ij.first, ij.second);
        // Relative to max(|z|, sigma): a height close to zero is compared on the sigma scale.
        CHECK_THAT(h.z(ij.first, ij.second), WithinAbs(z, 1e-12 * std::max(std::abs(z), sigma)));
    }
}

// Hidden (not run by ctest): regenerates the regression reference after a deliberate
// generator change. Writes to the path in SPECKLEBEM_REGEN_OUTPUT, or prints to stdout
// (run from the repository root):
//   SPECKLEBEM_REGEN_OUTPUT=tests/data/rough_surface_seed42_L10um.txt
//   build/release/tests/specklebem_unit_tests "[.regen]"   (one command line)
TEST_CASE("rough surface: regenerate the seed-42 regression reference", "[.regen]") {
    const std::string text = regression_reference_text();
    const char* out = std::getenv("SPECKLEBEM_REGEN_OUTPUT");
    if (out != nullptr && *out != '\0') {
        std::ofstream file(out);
        REQUIRE(file.good());
        file << text;
        REQUIRE(file.good());
    } else {
        std::cout << text;
    }
}

TEST_CASE("rough surface mesh: closed box, counts, orientation, top face", "[geometry]") {
    // L = 2 um, mesh 100 nm: n = 21 points per axis; default depth 2 um -> n_z = 20 wall
    // rows. n_rim = 80, V = 2 * 441 + 80 * 19 = 2402, F = 4 * 400 + 2 * 80 * 20 = 4800,
    // E = 7200, V - E + F = 2.
    const RoughSurfaceParams p = make_params(2e-6, 100e-9, 50e-9, 500e-9, 3);
    const Real depth = 2e-6;
    const HeightMap h = generate_gaussian_height_map(p);
    REQUIRE(h.z.rows() == 21);
    const TriangleMesh mesh = make_rough_surface_mesh(p);

    const BoxCounts expected = box_counts(21, 21, 20);
    CHECK(expected.v == 2402);
    CHECK(expected.f == 4800);
    CHECK(mesh.num_vertices() == expected.v);
    CHECK(mesh.num_triangles() == expected.f);
    CHECK(mesh.num_edges() == expected.e);
    CHECK(mesh.num_boundary_edges() == 0);
    CHECK(mesh.is_closed());
    CHECK(mesh.num_components() == 1);
    CHECK(mesh.is_consistently_oriented());
    CHECK(mesh.signed_volume() > 0.0);
    CHECK(mesh.num_vertices() - mesh.num_edges() + mesh.num_triangles() == 2);
    CHECK(report_value(mesh.quality_report(), "Euler characteristic") == 2);

    // The generator emits the final arrays itself: the constructor changed nothing.
    const auto [v, f] = geometry::detail::rough_box_arrays(h, depth);
    CHECK(mesh.vertices() == v);
    CHECK(mesh.triangles() == f);

    // Normal directions on the raw arrays (identical to the mesh normals).
    const FaceCheck fc = check_faces(v, f, h, depth);
    CHECK(fc.top == 2 * 20 * 20);
    CHECK(fc.bottom == 2 * 20 * 20);
    CHECK(fc.wall == 2 * 80 * 20);
    CHECK(fc.top_ok);
    CHECK(fc.bottom_ok);
    CHECK(fc.wall_ok);
    bool normals_match = true;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        normals_match =
            normals_match && (mesh.normal(t) - raw_normal(v, f, t).normalized()).norm() < 1e-12;
    }
    CHECK(normals_match);

    CHECK(top_face_reproduces(mesh, h));
    const auto [lo, hi] = mesh.bounding_box();
    CHECK_THAT(hi.z(), WithinAbs(depth, 0.0));
    CHECK_THAT(lo.z(), WithinAbs(h.z.minCoeff(), 0.0));
    CHECK_THAT(lo.x(), WithinRel(-1e-6, 1e-12));
    CHECK_THAT(hi.x(), WithinRel(1e-6, 1e-12));
    CHECK_THAT(lo.y(), WithinRel(-1e-6, 1e-12));
    CHECK_THAT(hi.y(), WithinRel(1e-6, 1e-12));
}

TEST_CASE("rough surface mesh: box depth handling", "[geometry]") {
    RoughSurfaceParams p = make_params(1e-6, 100e-9, 20e-9, 200e-9, 5);

    SECTION("explicit depth is honoured") {
        p.box_depth = 0.75e-6;
        const TriangleMesh mesh = make_rough_surface_mesh(p);
        CHECK(mesh.bounding_box().second.z() == 0.75e-6);
        // n = 11, n_z = ceil(0.75 um / 100 nm) = 8.
        const BoxCounts expected = box_counts(11, 11, 8);
        CHECK(mesh.num_vertices() == expected.v);
        CHECK(mesh.num_triangles() == expected.f);
        CHECK(mesh.is_closed());
    }

    SECTION("too small or invalid depth throws") {
        const HeightMap h = generate_gaussian_height_map(p);
        const Real zmax = h.z.maxCoeff();
        CHECK_THROWS_AS(make_mesh_from_height_map(h, zmax + 0.5 * h.dx), std::invalid_argument);
        CHECK_THROWS_AS(make_mesh_from_height_map(h, zmax + h.dx), std::invalid_argument);
        CHECK_THROWS_AS(make_mesh_from_height_map(h, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(make_mesh_from_height_map(h, -1e-6), std::invalid_argument);
        CHECK_NOTHROW(make_mesh_from_height_map(h, zmax + 1.01 * h.dx));
        p.box_depth = 1e-9;
        CHECK_THROWS_AS(make_rough_surface_mesh(p), std::invalid_argument);
    }

    SECTION("user height map round-trips, default depth") {
        // Non-square map with dx != dy and an analytic profile.
        HeightMap h;
        h.dx = 100e-9;
        h.dy = 150e-9;
        h.z.resize(6, 4);
        for (Index i = 0; i < 6; ++i) {
            for (Index j = 0; j < 4; ++j) {
                h.z(i, j) = 40e-9 * std::sin(0.7 * static_cast<Real>(i) + 1.3) *
                            std::cos(0.9 * static_cast<Real>(j) - 0.2);
            }
        }
        const TriangleMesh mesh = make_mesh_from_height_map(h);
        CHECK(top_face_reproduces(mesh, h));
        CHECK(mesh.bounding_box().second.z() == 2e-6);
        // n_z = ceil(2 um / min(dx, dy)) = 20.
        const BoxCounts expected = box_counts(6, 4, 20);
        CHECK(mesh.num_vertices() == expected.v);
        CHECK(mesh.num_triangles() == expected.f);
        CHECK(mesh.is_closed());
        CHECK(mesh.num_components() == 1);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6);
        CHECK(mesh.triangles() == f);
        const FaceCheck fc = check_faces(v, f, h, 2e-6);
        CHECK(fc.top == 2 * 5 * 3);
        CHECK(fc.bottom == 2 * 5 * 3);
        CHECK(fc.wall == 2 * 16 * 20);
        CHECK(fc.top_ok);
        CHECK(fc.bottom_ok);
        CHECK(fc.wall_ok);
    }

    SECTION("degenerate grids and parameters throw") {
        HeightMap h;
        h.dx = 100e-9;
        h.dy = 100e-9;
        h.z = MatrixXr::Zero(1, 5);
        CHECK_THROWS_AS(make_mesh_from_height_map(h), std::invalid_argument);
        h.z = MatrixXr::Zero(5, 1);
        CHECK_THROWS_AS(make_mesh_from_height_map(h), std::invalid_argument);
        h.z = MatrixXr::Zero(3, 3);
        h.dx = 0.0;
        CHECK_THROWS_AS(make_mesh_from_height_map(h), std::invalid_argument);

        RoughSurfaceParams q = p;
        q.mesh_size = 3.0 * q.edge_length_L;  // rounds to a single grid point
        CHECK_THROWS_AS(generate_gaussian_height_map(q), std::invalid_argument);
        q = p;
        q.correlation_length = 0.0;
        CHECK_THROWS_AS(generate_gaussian_height_map(q), std::invalid_argument);
        q = p;
        q.rms_roughness = -1e-9;
        CHECK_THROWS_AS(generate_gaussian_height_map(q), std::invalid_argument);
        q = p;
        q.edge_length_L = std::nan("");
        CHECK_THROWS_AS(generate_gaussian_height_map(q), std::invalid_argument);
        // Grid cap: more than 2^20 points per axis is rejected before allocating.
        q = p;
        q.mesh_size = q.edge_length_L / 2e6;
        CHECK_THROWS_AS(generate_gaussian_height_map(q), std::invalid_argument);
        CHECK_THROWS_AS(make_rough_surface_mesh(q), std::invalid_argument);
    }
}

TEST_CASE("rough surface: under-resolved correlation length is rejected", "[geometry]") {
    // dx = 100 nm: Lc must be at least 2 * dx = 200 nm.
    RoughSurfaceParams p = make_params(1e-6, 100e-9, 20e-9, 190e-9, 9);
    CHECK_THROWS_AS(generate_gaussian_height_map(p), std::invalid_argument);
    CHECK_THROWS_AS(make_rough_surface_mesh(p), std::invalid_argument);
    p.correlation_length = 20e-9;  // < dx / 4: the kernel would be a single point
    CHECK_THROWS_WITH(generate_gaussian_height_map(p),
                      Catch::Matchers::ContainsSubstring("under-resolved"));
    p.correlation_length = 200e-9;  // exactly 2 dx is accepted
    CHECK_NOTHROW(generate_gaussian_height_map(p));
    p.correlation_length = 250e-9;
    CHECK(generate_gaussian_height_map(p).rms() > 0.0);
}

TEST_CASE("rough surface: estimators on a known map", "[geometry]") {
    HeightMap h;
    h.dx = 1.0;
    h.dy = 1.0;
    h.z.resize(2, 2);
    h.z << 1.0, -1.0, 3.0, 1.0;  // mean 1, deviations 0, -2, 2, 0
    CHECK_THAT(h.rms(), WithinRel(std::sqrt(2.0), 1e-15));
    HeightMap flat;
    flat.dx = 1.0;
    flat.dy = 1.0;
    flat.z = MatrixXr::Constant(4, 4, 3.0);
    CHECK(flat.rms() == 0.0);
    CHECK_THROWS_AS(flat.estimated_correlation_length(), std::runtime_error);
    CHECK_THROWS_AS(HeightMap{}.rms(), std::logic_error);
}

TEST_CASE("rough surface mesh: generation time for L = 10 um, 50 nm (docs/01)", "[geometry]") {
    // 201 x 201 grid, n_z = 40: F = 4 * 200^2 + 2 * 800 * 40 = 224000 triangles.
    const RoughSurfaceParams p = make_params(10e-6, 50e-9, 50e-9, 500e-9, 11);
    const auto start = std::chrono::steady_clock::now();
    const TriangleMesh mesh = make_rough_surface_mesh(p);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CAPTURE(seconds);
    CHECK(mesh.num_triangles() == box_counts(201, 201, 40).f);
    CHECK(mesh.num_triangles() == 224000);
    CHECK(mesh.is_closed());
#ifdef NDEBUG
    CHECK(seconds < 5.0);
#else
    CHECK(seconds < 60.0);
#endif
}
