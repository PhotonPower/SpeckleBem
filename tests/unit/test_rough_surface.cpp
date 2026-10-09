#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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
Index report_value(const std::string& report, const std::string& label) {
    const std::regex re(label + R"(\s*:\s*(-?\d+))");
    std::smatch m;
    if (!std::regex_search(report, m, re))
        return -1;
    return static_cast<Index>(std::stoll(m[1].str()));
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

// ---- Graded box (WP2b) helpers ------------------------------------------------------

/// Counts the warnings (or the messages of at least the given level) logged while it is
/// alive (logger level lowered if needed and restored afterwards).
class WarningCounter {
public:
    explicit WarningCounter(spdlog::level::level_enum level = spdlog::level::warn)
        : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64)),
          saved_level_(spdlog::default_logger()->level()) {
        sink_->set_level(level);
        spdlog::default_logger()->set_level(std::min(saved_level_, level));
        spdlog::default_logger()->sinks().push_back(sink_);
    }
    ~WarningCounter() {
        auto& sinks = spdlog::default_logger()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
        spdlog::default_logger()->set_level(saved_level_);
    }
    WarningCounter(const WarningCounter&) = delete;
    WarningCounter& operator=(const WarningCounter&) = delete;
    [[nodiscard]] std::size_t count() const { return sink_->last_formatted().size(); }
    /// Number of the logged messages that contain the given text.
    [[nodiscard]] std::size_t count(const std::string& text) const {
        const std::vector<std::string> lines = sink_->last_formatted();
        return static_cast<std::size_t>(
            std::count_if(lines.begin(), lines.end(),
                          [&](const std::string& l) { return l.find(text) != std::string::npos; }));
    }

private:
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
    spdlog::level::level_enum saved_level_;
};

/// Height map with exactly representable inputs: heights a * ((7 i + 13 j) mod 41 - 20)
/// (pre-WP2b checksums below were computed on this map).
HeightMap pattern_map(Index nx, Index ny, Real dx, Real dy, Real a = 1e-9) {
    HeightMap h;
    h.dx = dx;
    h.dy = dy;
    h.z.resize(nx, ny);
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            h.z(i, j) = a * static_cast<Real>((i * 7 + j * 13) % 41 - 20);
        }
    }
    return h;
}

/// Pattern map multiplied by a bump that vanishes on the rim (rim heights exactly 0).
HeightMap bump_map(Index nx, Index ny, Real dx, Real dy, Real a) {
    HeightMap h = pattern_map(nx, ny, dx, dy, a / 20.0);
    const auto cx = static_cast<Real>(nx - 1);
    const auto cy = static_cast<Real>(ny - 1);
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            const auto u = static_cast<Real>(i) / cx;
            const auto w = static_cast<Real>(j) / cy;
            h.z(i, j) *= 16.0 * u * (1.0 - u) * w * (1.0 - w);
        }
    }
    return h;
}

/// Expected graded counts from the resolved layout: V = n_x n_y + (n_cx + 1)(n_cy + 1) +
/// relaxation nodes + sum over interior rows of the ring sizes (rim and bottom ring are
/// shared); F = 2 (n_x - 1)(n_y - 1) + sum over strips (R_k + R_(k+1)) + 2 relaxation
/// nodes + 2 n_cx n_cy.
Index graded_vertex_count(Index nx, Index ny, const geometry::detail::BoxGrading& g) {
    Index v = nx * ny + (g.coarse_cells_x + 1) * (g.coarse_cells_y + 1) + g.relaxation_vertices;
    for (std::size_t k = 1; k + 1 < g.row_ring_sizes.size(); ++k) v += g.row_ring_sizes[k];
    return v;
}
Index graded_triangle_count(Index nx, Index ny, const geometry::detail::BoxGrading& g) {
    Index f = 2 * (nx - 1) * (ny - 1) + 2 * g.coarse_cells_x * g.coarse_cells_y +
              2 * g.relaxation_vertices;
    for (std::size_t k = 0; k + 1 < g.row_ring_sizes.size(); ++k) {
        f += g.row_ring_sizes[k] + g.row_ring_sizes[k + 1];
    }
    return f;
}

/// FNV-1a hash of the triangle index array (row by row).
std::uint64_t triangle_hash(const Triangles& f) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (Index t = 0; t < f.rows(); ++t) {
        for (Index q = 0; q < 3; ++q) {
            hash ^= static_cast<std::uint64_t>(f(t, q));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

/// Weighted vertex checksum sum_k (k + 1) (|x_k| + 2 |y_k| + 3 |z_k|).
Real vertex_checksum(const Vertices& v) {
    Real s = 0.0;
    for (Index k = 0; k < v.rows(); ++k) {
        s += static_cast<Real>(k + 1) *
             (std::abs(v(k, 0)) + 2.0 * std::abs(v(k, 1)) + 3.0 * std::abs(v(k, 2)));
    }
    return s;
}

/// True if every directed edge occurs exactly once and its reverse occurs too: the raw
/// arrays describe a closed, edge-manifold, consistently oriented surface.
bool closed_and_consistent(const Triangles& f) {
    std::vector<std::pair<Index, Index>> directed;
    directed.reserve(static_cast<std::size_t>(3 * f.rows()));
    for (Index t = 0; t < f.rows(); ++t) {
        for (Index q = 0; q < 3; ++q) {
            const Index a = f(t, q);
            const Index b = f(t, (q + 1) % 3);
            directed.emplace_back(a, b);
        }
    }
    std::sort(directed.begin(), directed.end());
    if (std::adjacent_find(directed.begin(), directed.end()) != directed.end())
        return false;
    for (const auto& [a, b] : directed) {
        if (!std::binary_search(directed.begin(), directed.end(), std::make_pair(b, a)))
            return false;
    }
    return true;
}

/// Aspect ratio R / (2 r) = a b c s / (8 A^2) (as in TriangleMesh::quality_report()).
Real aspect_ratio(const Vertices& v, const Triangles& f, Index t) {
    const Vec3 p0 = v.row(f(t, 0)).transpose();
    const Vec3 p1 = v.row(f(t, 1)).transpose();
    const Vec3 p2 = v.row(f(t, 2)).transpose();
    const Real a = (p1 - p0).norm();
    const Real b = (p2 - p1).norm();
    const Real c = (p0 - p2).norm();
    const Real s = 0.5 * (a + b + c);
    const Real area = 0.5 * (p1 - p0).cross(p2 - p0).norm();
    return a * b * c * s / (8.0 * area * area);
}

/// Face tallies and checks of a (graded or uniform) box, classified by vertices: bottom if
/// all three vertices lie at z = depth, wall if all three lie on one side plane, top
/// otherwise.
struct BoxCheck {
    Index top = 0, wall = 0, bottom = 0;
    bool top_ok = true, wall_ok = true, bottom_ok = true;
    Real max_aspect_top = 0.0;  ///< top face (depends on the height map)
    Real max_aspect_box = 0.0;  ///< closing box: walls and bottom plate
    Real min_area = std::numeric_limits<Real>::infinity();
};
BoxCheck check_box(const Vertices& v, const Triangles& f, const HeightMap& h, Real depth) {
    const Real hx = 0.5 * static_cast<Real>(h.z.rows() - 1) * h.dx;
    const Real hy = 0.5 * static_cast<Real>(h.z.cols() - 1) * h.dy;
    const Real tol = 1e-9 * std::min(h.dx, h.dy);
    const Vec3 box_centre(0.0, 0.0, 0.5 * depth);
    BoxCheck bc;
    for (Index t = 0; t < f.rows(); ++t) {
        bool bottom = true;
        std::array<bool, 4> side{true, true, true, true};  // x = -hx, x = hx, y = -hy, y = hy
        for (Index q = 0; q < 3; ++q) {
            const Index k = f(t, q);
            bottom = bottom && v(k, 2) == depth;
            side[0] = side[0] && std::abs(v(k, 0) + hx) < tol;
            side[1] = side[1] && std::abs(v(k, 0) - hx) < tol;
            side[2] = side[2] && std::abs(v(k, 1) + hy) < tol;
            side[3] = side[3] && std::abs(v(k, 1) - hy) < tol;
        }
        const bool wall = side[0] || side[1] || side[2] || side[3];
        const Vec3 raw = raw_normal(v, f, t);
        const Vec3 n = raw.normalized();
        const Vec3 c = raw_centroid(v, f, t);
        const Real aspect = aspect_ratio(v, f, t);
        if (wall || bottom)
            bc.max_aspect_box = std::max(bc.max_aspect_box, aspect);
        else
            bc.max_aspect_top = std::max(bc.max_aspect_top, aspect);
        if (wall) {
            ++bc.wall;
            bc.wall_ok = bc.wall_ok && n.z() == 0.0 && n.dot(c - box_centre) > 0.0 &&
                         std::abs(std::abs(n.x()) + std::abs(n.y()) - 1.0) < 1e-12;
        } else if (bottom) {
            ++bc.bottom;
            bc.bottom_ok = bc.bottom_ok && n.z() > 1.0 - 1e-12;
        } else {
            ++bc.top;
            bc.top_ok = bc.top_ok && n.z() < 0.0 && c.z() < h.z.maxCoeff() + tol;
        }
        bc.min_area = std::min(bc.min_area, 0.5 * raw.norm());
    }
    return bc;
}

/// Largest aspect ratio of the walls and bottom of the uniform WP2 box on the same map
/// (per-triangle checks: used by the release build only).
[[maybe_unused]] Real uniform_box_aspect(const HeightMap& h, Real depth) {
    const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, std::min(h.dx, h.dy));
    return check_box(v, f, h, depth).max_aspect_box;
}

/// Aspect bound of the graded box on a rough rim (safety net in rough_surface.hpp): at most
/// twice the walls of the uniform WP2 box (whose first row the rim steps shear), and 4 is
/// always allowed. Uses the reference computed by box_grading() (cheap in the sanitizer
/// build); it is checked against the uniform arrays in the WP2c acceptance test.
Real rough_rim_aspect_bound(const HeightMap& h, Real depth, std::optional<Real> box_mesh_size = {},
                            std::optional<Real> fine_depth = {}) {
    const geometry::detail::BoxGrading g =
        geometry::detail::box_grading(h, depth, box_mesh_size, fine_depth);
    return std::max(4.0, 2.0 * g.uniform_wall_aspect);
}

/// Full set of graded-box checks on the raw arrays and on the constructed mesh. Rough rims
/// pass max_aspect = rough_rim_aspect_bound().
void check_graded_box(const HeightMap& h, Real depth, std::optional<Real> box_mesh_size,
                      Index expected_levels, Real max_aspect = 4.0,
                      std::optional<Real> fine_depth = {}) {
    const Index nx = h.z.rows();
    const Index ny = h.z.cols();
    const geometry::detail::BoxGrading g =
        geometry::detail::box_grading(h, depth, box_mesh_size, fine_depth);
    CAPTURE(nx, ny, h.dx, h.dy, depth, g.levels, g.coarse_cells_x, g.coarse_cells_y,
            g.target_spacing, g.relaxation_rows, g.fine_rows);
    CHECK(g.levels == expected_levels);
    const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, box_mesh_size, fine_depth);
    CHECK(v.rows() == graded_vertex_count(nx, ny, g));
    CHECK(f.rows() == graded_triangle_count(nx, ny, g));
    CHECK(closed_and_consistent(f));

    const BoxCheck bc = check_box(v, f, h, depth);
    CAPTURE(bc.top, bc.wall, bc.bottom, bc.max_aspect_top, bc.max_aspect_box, bc.min_area);
    CHECK(bc.top == 2 * (nx - 1) * (ny - 1));
    CHECK(bc.bottom == 2 * g.coarse_cells_x * g.coarse_cells_y);
    CHECK(bc.top + bc.wall + bc.bottom == f.rows());
    CHECK(bc.top_ok);
    CHECK(bc.wall_ok);
    CHECK(bc.bottom_ok);
    CHECK(bc.max_aspect_box <= max_aspect);
    // Smallest triangle well above zero: at least a quarter of the smallest top-face half
    // cell (the fine wall row has height ~h, the cells grow with depth).
    CHECK(bc.min_area > 0.25 * 0.5 * h.dx * h.dy * std::min(h.dx, h.dy) / std::max(h.dx, h.dy));

    // The constructor accepts the arrays unchanged (no orientation repair).
    const TriangleMesh mesh = make_mesh_from_height_map(h, depth, box_mesh_size, fine_depth);
    CHECK(mesh.vertices() == v);
    CHECK(mesh.triangles() == f);
    CHECK(mesh.is_closed());
    CHECK(mesh.num_boundary_edges() == 0);
    CHECK(mesh.num_components() == 1);
    CHECK(mesh.is_consistently_oriented());
    CHECK(mesh.signed_volume() > 0.0);
    CHECK(mesh.num_edges() * 2 == mesh.num_triangles() * 3);
    CHECK(mesh.num_vertices() - mesh.num_edges() + mesh.num_triangles() == 2);
    CHECK(report_value(mesh.quality_report(), "Euler characteristic") == 2);
    CHECK(top_face_reproduces(mesh, h));
    const auto [lo, hi] = mesh.bounding_box();
    CHECK(hi.z() == depth);
    CHECK(lo.z() == h.z.minCoeff());
}

/// Longest vertical wall edges (both ends in one rim column) of a box: between the rim and
/// the anchor row of the column (BoxGrading::anchor_z; empty for the uniform box), with the
/// upper end above z_f, and overall.
struct VerticalEdges {
    Real above_anchor = 0.0;
    Real above_fine = 0.0;
    Real all = 0.0;
};
VerticalEdges vertical_wall_edges(const Vertices& v, const Triangles& f, const HeightMap& h,
                                  const std::vector<Real>& anchor_z, std::optional<Real> zf) {
    const Index nx = h.z.rows();
    const Index ny = h.z.cols();
    // Rim walk of BoxGrading::anchor_z: column index of every rim grid node.
    std::vector<Index> column(static_cast<std::size_t>(nx * ny), -1);
    Index c = 0;
    for (Index i = 0; i + 1 < nx; ++i) column[static_cast<std::size_t>(i * ny)] = c++;
    for (Index j = 0; j + 1 < ny; ++j) column[static_cast<std::size_t>((nx - 1) * ny + j)] = c++;
    for (Index i = nx - 1; i > 0; --i) column[static_cast<std::size_t>(i * ny + ny - 1)] = c++;
    for (Index j = ny - 1; j > 0; --j) column[static_cast<std::size_t>(j)] = c++;
    const Real x0 = -0.5 * static_cast<Real>(nx - 1) * h.dx;
    const Real y0 = -0.5 * static_cast<Real>(ny - 1) * h.dy;
    const Real tol = 1e-9 * std::min(h.dx, h.dy);
    VerticalEdges e;
    for (Index t = 0; t < f.rows(); ++t) {
        for (Index q = 0; q < 3; ++q) {
            const Index a = f(t, q);
            const Index b = f(t, (q + 1) % 3);
            if (std::abs(v(a, 0) - v(b, 0)) > tol || std::abs(v(a, 1) - v(b, 1)) > tol)
                continue;
            const Real upper = std::min(v(a, 2), v(b, 2));
            const Real lower = std::max(v(a, 2), v(b, 2));
            const Real len = lower - upper;
            e.all = std::max(e.all, len);
            if (zf.has_value() && upper < *zf - tol)
                e.above_fine = std::max(e.above_fine, len);
            if (anchor_z.empty())
                continue;
            const auto i = static_cast<Index>(std::llround((v(a, 0) - x0) / h.dx));
            const auto j = static_cast<Index>(std::llround((v(a, 1) - y0) / h.dy));
            const Index col = column[static_cast<std::size_t>(i * ny + j)];
            REQUIRE(col >= 0);
            if (lower <= anchor_z[static_cast<std::size_t>(col)] + tol)
                e.above_anchor = std::max(e.above_anchor, len);
        }
    }
    return e;
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
        std::ofstream file(out, std::ios::binary);  // LF line endings on every platform
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
    // Uniform box (WP2): box_mesh_size = top-face spacing.
    RoughSurfaceParams p = make_params(2e-6, 100e-9, 50e-9, 500e-9, 3);
    p.box_mesh_size = p.mesh_size;
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
    const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, p.box_mesh_size);
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
    // 201 x 201 grid, default depth 2 um, automatic graded box (WP2b): 80000 top-face
    // triangles plus a closing box of about 5750 (see the reference-case test below); the
    // uniform WP2 box had 224000 triangles.
    const RoughSurfaceParams p = make_params(10e-6, 50e-9, 50e-9, 500e-9, 11);
    const auto start = std::chrono::steady_clock::now();
    const TriangleMesh mesh = make_rough_surface_mesh(p);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CAPTURE(seconds);
    const geometry::detail::BoxGrading g =
        geometry::detail::box_grading(generate_gaussian_height_map(p), 2e-6);
    CHECK(g.levels == 3);
    CHECK(mesh.num_triangles() == graded_triangle_count(201, 201, g));
    CHECK(mesh.num_vertices() == graded_vertex_count(201, 201, g));
    CHECK(mesh.num_triangles() < 90000);
    // Backlog WP2b acceptance on a generated (seeded) map: closing box <= 20 % of the
    // 80000 top-face triangles.
    const Real ratio = static_cast<Real>(mesh.num_triangles() - 80000) / static_cast<Real>(80000);
    CAPTURE(ratio);
    CHECK(ratio <= 0.20);
    CHECK(mesh.is_closed());
    CHECK(mesh.num_components() == 1);
#ifdef NDEBUG
    CHECK(seconds < 5.0);
#else
    CHECK(seconds < 60.0);
#endif
}

// ---- Graded box (WP2b) ---------------------------------------------------------------

TEST_CASE("rough surface mesh: graded box, reference case L = 10 um, 50 nm, depth 2 um",
          "[geometry]") {
    // 201 x 201 grid (n - 1 = 200 = 8 * 25), dx = 50 nm, rim heights 0, depth 2 um.
    // Automatic rule: h_c = min(depth / 2, L / 8, 10 h) = min(1 um, 1.25 um, 500 nm) = 500 nm,
    // M = round(log2(10)) = 3 -> 25 x 25 bottom cells of 400 nm.
    // Wall rows: rim, fine row (50 nm), transitions 50 / 100 / 200 nm (levels 1, 2, 3),
    // 4 coarse rows of 400 nm. Rings R_m = 800 / 2^m.
    // Walls: 2 * 800 + (800 + 400) + (400 + 200) + (200 + 100) + 4 * 2 * 100 = 4500,
    // bottom 2 * 25^2 = 1250, top 2 * 200^2 = 80000: closing box 5750 = 7.19 % of the top
    // face (uniform WP2 box: 80000 bottom + 64000 wall = 180 %).
    const HeightMap h = bump_map(201, 201, 50e-9, 50e-9, 50e-9);
    const Real depth = 2e-6;
    const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, depth);
    CHECK_THAT(g.target_spacing, WithinRel(500e-9, 1e-12));
    CHECK(g.levels == 3);
    CHECK(g.coarse_cells_x == 25);
    CHECK(g.coarse_cells_y == 25);
    CHECK(g.row_ring_sizes == std::vector<Index>{800, 800, 400, 200, 100, 100, 100, 100, 100});

    // Counts from the resolved layout (cheap; the arrays below are checked against them).
    const Index n_top = 2 * 200 * 200;
    const Index n_all = graded_triangle_count(201, 201, g);
    const Index n_box = n_all - n_top;
    const Real ratio = static_cast<Real>(n_box) / static_cast<Real>(n_top);
    CAPTURE(n_box, ratio);
    CHECK(n_all == 85750);
    CHECK(n_box == 5750);
    CHECK(ratio <= 0.20);  // backlog WP2b acceptance (<= ~20 %)
    CHECK(graded_vertex_count(201, 201, g) == 201 * 201 + 26 * 26 + 800 + 400 + 200 + 4 * 100);

    const auto [v, f] = geometry::detail::rough_box_arrays(h, depth);
    CHECK(f.rows() == n_all);
    CHECK(v.rows() == graded_vertex_count(201, 201, g));
#ifdef NDEBUG
    // Per-triangle checks of the 85750 triangles: release only (the per-test time limit in
    // the sanitizer build); the debug build runs the same checks on the L = 2 um analogue.
    CHECK(closed_and_consistent(f));
    const BoxCheck bc = check_box(v, f, h, depth);
    CAPTURE(bc.max_aspect_top, bc.max_aspect_box);
    CHECK(bc.top == n_top);
    CHECK(bc.wall == 4500);
    CHECK(bc.bottom == 1250);
    CHECK(bc.top_ok);
    CHECK(bc.wall_ok);
    CHECK(bc.bottom_ok);
    CHECK(bc.max_aspect_box <= 4.0);
    CHECK(bc.max_aspect_top <= 4.0);
#endif

    // Uniform fallback on the same grid: the WP2 box, 4 * 200^2 + 2 * 800 * 40 = 224000.
    const auto [vu, fu] = geometry::detail::rough_box_arrays(h, depth, 50e-9);
    CHECK(fu.rows() == box_counts(201, 201, 40).f);
    CHECK(fu.rows() == 224000);
    CHECK(vu.rows() == box_counts(201, 201, 40).v);
}

TEST_CASE("rough surface mesh: graded box, closed and outward (L = 2 um, 100 nm, depth 1 um)",
          "[geometry]") {
    // Debug-sized analogue of the reference case. Automatic rule: h_c = min(500 nm, 250 nm,
    // 1 um) = 250 nm -> M = round(log2(2.5)) = 1. This small patch cannot reach the 20 %
    // ratio (the L / 8 cap alone makes the bottom plate 25 % of the top face); the ratio is
    // checked on the reference grid above.
    RoughSurfaceParams p = make_params(2e-6, 100e-9, 50e-9, 500e-9, 21);
    p.box_depth = 1e-6;
    const HeightMap h = generate_gaussian_height_map(p);
    REQUIRE(h.z.rows() == 21);

    SECTION("automatic coarse spacing") {
        check_graded_box(h, 1e-6, std::nullopt, 1);
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 1e-6);
        CHECK_THAT(g.target_spacing, WithinRel(250e-9, 1e-12));
        CHECK(g.coarse_cells_x == 10);
        CHECK(g.row_ring_sizes.front() == 80);
        CHECK(g.row_ring_sizes.back() == 40);
        // make_rough_surface_mesh uses the same layout.
        const TriangleMesh mesh = make_rough_surface_mesh(p);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, 1e-6);
        CHECK(mesh.vertices() == v);
        CHECK(mesh.triangles() == f);
    }
    SECTION("user coarse spacing, two levels") {
        // h_c = 400 nm -> M = 2 (fits: 1.5 * 4 * 100 nm <= 1 um).
        check_graded_box(h, 1e-6, 400e-9, 2);
        p.box_mesh_size = 400e-9;
        const TriangleMesh mesh = make_rough_surface_mesh(p);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, 1e-6, 400e-9);
        CHECK(mesh.triangles() == f);
        CHECK(mesh.vertices() == v);
    }
    SECTION("rounding to the nearest power of two") {
        // h_c / h = 2.8 -> log2 = 1.49 -> M = 1; 2.9 -> 1.54 -> M = 2.
        CHECK(geometry::detail::box_grading(h, 1e-6, 280e-9).levels == 1);
        CHECK(geometry::detail::box_grading(h, 1e-6, 290e-9).levels == 2);
        // 2^M <= min(n_x - 1, n_y - 1) = 20 caps M at 4; the depth caps it further.
        CHECK(geometry::detail::box_grading(h, 10e-6, 1.0).levels == 4);
        CHECK(geometry::detail::box_grading(h, 1e-6, 1.0).levels == 2);
    }
}

TEST_CASE("rough surface mesh: uniform fallback is bit-identical to WP2", "[geometry]") {
    // Checksums of detail::rough_box_arrays(h, 2e-6) computed with the WP2 implementation
    // (commit 5d52618) before the grading was added.
    SECTION("21 x 21, dx = dy") {
        const HeightMap h = pattern_map(21, 21, 100e-9, 100e-9);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6, 100e-9);
        CHECK(v.rows() == box_counts(21, 21, 20).v);
        CHECK(f.rows() == box_counts(21, 21, 20).f);
        CHECK(v.rows() == 2402);
        CHECK(f.rows() == 4800);
        CHECK(triangle_hash(f) == 9102135350366221715ULL);
        CHECK_THAT(vertex_checksum(v), WithinRel(1.54021067259999853e+01, 1e-14));
        // Same arrays through the public mesh function (no repair).
        const TriangleMesh mesh = make_mesh_from_height_map(h, 2e-6, 100e-9);
        CHECK(mesh.vertices() == v);
        CHECK(mesh.triangles() == f);
        // A box_mesh_size below h / sqrt(2) also gives the uniform box, with one SBEM_WARN
        // per mesh built (no throw); the detail:: inspection functions stay silent.
        WarningCounter warnings;
        const auto [v2, f2] = geometry::detail::rough_box_arrays(h, 2e-6, 30e-9);
        CHECK(v2 == v);
        CHECK(f2 == f);
        CHECK(geometry::detail::box_grading(h, 2e-6, 30e-9).levels == 0);
        CHECK(warnings.count() == 0);
        TriangleMesh small;
        CHECK_NOTHROW(small = make_mesh_from_height_map(h, 2e-6, 30e-9));
        CHECK(warnings.count() == 1);
        CHECK(small.vertices() == v);
        CHECK(small.triangles() == f);
        // h / sqrt(2) <= h_c < sqrt(2) h: uniform box without a warning.
        const TriangleMesh near = make_mesh_from_height_map(h, 2e-6, 75e-9);
        CHECK(warnings.count() == 1);
        CHECK(near.triangles() == f);
    }
    SECTION("6 x 4, dx != dy (automatic rule gives the uniform box)") {
        const HeightMap h = pattern_map(6, 4, 100e-9, 150e-9);
        // Automatic h_c = min(1 um, 450 nm / 8, 1 um) = 56 nm < min(dx, dy): M = 0.
        CHECK(geometry::detail::box_grading(h, 2e-6).levels == 0);
        for (const std::optional<Real> hc : {std::optional<Real>{}, std::optional<Real>{100e-9}}) {
            const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6, hc);
            CHECK(v.rows() == box_counts(6, 4, 20).v);
            CHECK(f.rows() == box_counts(6, 4, 20).f);
            CHECK(triangle_hash(f) == 16555332754961717385ULL);
            CHECK_THAT(vertex_checksum(v), WithinRel(2.24080899999999889e-01, 1e-14));
        }
    }
}

TEST_CASE("rough surface mesh: graded box with non-power-of-two rim counts and dx != dy",
          "[geometry]") {
    SECTION("33 x 17 (power-of-two cell counts), ring sizes") {
        // n - 1 = 32 and 16, h_c = 4 h -> M = 2: bottom 8 x 4 cells, rings 96 / 48 / 24.
        const HeightMap h = pattern_map(33, 17, 80e-9, 80e-9);
        check_graded_box(h, 2e-6, 320e-9, 2);
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6, 320e-9);
        CHECK(g.coarse_cells_x == 8);
        CHECK(g.coarse_cells_y == 4);
        REQUIRE(g.row_ring_sizes.size() >= 5);
        CHECK(g.row_ring_sizes[0] == 96);
        CHECK(g.row_ring_sizes[1] == 96);
        CHECK(g.row_ring_sizes[2] == 48);
        CHECK(g.row_ring_sizes[3] == 24);
        CHECK(g.row_ring_sizes.back() == 24);
    }
    SECTION("37 x 25, dx = dy, three levels") {
        // n - 1 = 36 -> ceil(36 / 8) = 5 coarse cells (7, 7, 7, 7, 8 fine cells);
        // n - 1 = 24 -> 3 coarse cells of 8.
        const HeightMap h = pattern_map(37, 25, 80e-9, 80e-9);
        check_graded_box(h, 2e-6, 640e-9, 3);
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6, 640e-9);
        CHECK(g.coarse_cells_x == 5);
        CHECK(g.coarse_cells_y == 3);
        // Automatic rule: h_c = min(1 um, 1.92 um / 8 = 240 nm, 800 nm) -> M = round(1.58) = 2.
        check_graded_box(h, 2e-6, std::nullopt, 2);
    }
    SECTION("37 x 25, dx = 60 nm, dy = 90 nm") {
        // Levels from h_b = max(dx, dy) = 90 nm: 720 / 90 = 8 -> M = 3, 360 / 90 = 4 -> M = 2,
        // 480 / 90 = 5.3 -> log2 = 2.4 -> M = 2.
        const HeightMap h = pattern_map(37, 25, 60e-9, 90e-9, 0.25e-9);
        check_graded_box(h, 2e-6, 720e-9, 3);
        check_graded_box(h, 1.5e-6, 360e-9, 2);
        check_graded_box(h, 2e-6, 480e-9, 2);
        // Automatic: min(1 um, 2.16 um / 8 = 270 nm, 600 nm) / 90 nm = 3 -> M = 2.
        check_graded_box(h, 2e-6, std::nullopt, 2);
    }
    SECTION("odd cell counts, dy > dx") {
        // n - 1 = 23 and 19: no power of two divides them; dy = 1.5 dx, h_b = 150 nm.
        const HeightMap h = pattern_map(24, 20, 100e-9, 150e-9, 0.25e-9);
        check_graded_box(h, 3e-6, 1200e-9, 3);
        check_graded_box(h, 3e-6, 300e-9, 1);
        // 200 nm / 150 nm = 1.33 < sqrt(2): uniform box.
        CHECK(geometry::detail::box_grading(h, 3e-6, 200e-9).levels == 0);
    }
    SECTION("anisotropic map, dx / dy = 4 (41 x 101, 200 nm x 50 nm) and transposed") {
        // Levels from h_b = max(dx, dy) = 200 nm, row heights from h_g = sqrt(dx dy) =
        // 100 nm. With min(dx, dy) row heights the 2:1 cells on the walls along x were 8:1
        // flat (aspect ratio 8.6); with max(dx, dy) the unevenly halved cells on the walls
        // along y (100 = 13 coarse cells of 7 or 8) were 1:8 needles (4.6). Now: 2.65 on a
        // flat map (2:1 cells 400 nm x 100 nm), 3.8 with the +-20 nm pattern rim used here.
        for (const bool transposed : {false, true}) {
            CAPTURE(transposed);
            const HeightMap h = transposed ? pattern_map(101, 41, 50e-9, 200e-9)
                                           : pattern_map(41, 101, 200e-9, 50e-9);
            // Automatic: min(1 um, 5 um / 8 = 625 nm, 10 * 50 nm) / 200 nm = 2.5 -> M = 1.
            check_graded_box(h, 2e-6, std::nullopt, 1);
            check_graded_box(h, 2e-6, 800e-9, 2);
            check_graded_box(h, 3e-6, 1600e-9, 3);
            const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 3e-6, 1600e-9);
            CHECK(g.coarse_cells_x == (transposed ? 13 : 5));
            CHECK(g.coarse_cells_y == (transposed ? 5 : 13));
            // 1.5 * 2^M h_g <= H: depth 1.1 um allows M = 2 only (1.2 um needed for M = 3).
            CHECK(geometry::detail::box_grading(h, 1.1e-6, 1600e-9).levels == 2);
            check_graded_box(h, 1.1e-6, 1600e-9, 2);
        }
    }
}

TEST_CASE("rough surface mesh: shallow box degrades to the uniform mesh", "[geometry]") {
    // 21 x 21 grid, h = 100 nm, heights within +-5 nm.
    const HeightMap h = pattern_map(21, 21, 100e-9, 100e-9, 0.25e-9);
    SECTION("depth < 2 h: no transition fits") {
        const Real depth = 190e-9;
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, depth, 400e-9);
        CHECK(g.levels == 0);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, 400e-9);
        const auto [vu, fu] = geometry::detail::rough_box_arrays(h, depth, 100e-9);
        CHECK(v == vu);
        CHECK(f == fu);
        // n_z = ceil(190 nm / 100 nm) = 2.
        CHECK(f.rows() == box_counts(21, 21, 2).f);
        CHECK(v.rows() == box_counts(21, 21, 2).v);
        const TriangleMesh mesh = make_mesh_from_height_map(h, depth, 400e-9);
        CHECK(mesh.is_closed());
        CHECK(mesh.triangles() == f);
    }
    SECTION("the depth limits the number of levels") {
        // 1.5 * 2^M h <= H: depth 0.5 um allows M = 1 only, 0.65 um allows M = 2.
        CHECK(geometry::detail::box_grading(h, 0.5e-6, 800e-9).levels == 1);
        CHECK(geometry::detail::box_grading(h, 0.65e-6, 800e-9).levels == 2);
        check_graded_box(h, 0.5e-6, 800e-9, 1);
        check_graded_box(h, 0.65e-6, 800e-9, 2);
    }
}

TEST_CASE("rough surface mesh: rim pits are absorbed by the relaxation band", "[geometry]") {
    // 21 x 21 flat map, h = 100 nm, depth 2 um: the automatic rule gives M = 1 (250 nm).
    // A pit on the rim at the middle node of the first 2:1 cell (i = 1, j = 0). Under WP2b
    // (wall rows following the rim) a 300 nm pit inverted that cell and forced the uniform
    // box; now the rows below the rim follow the level-M interpolation of the rim.
    HeightMap h;
    h.dx = 100e-9;
    h.dy = 100e-9;
    h.z = MatrixXr::Zero(21, 21);
    CHECK(geometry::detail::box_grading(h, 2e-6).levels == 1);
    CHECK(geometry::detail::box_grading(h, 2e-6).relaxation_rows == 0);
    SECTION("moderate pit: no relaxation node") {
        // Excess 50 nm: the rim-to-anchor gaps are 100 .. 150 nm <= 2 h_g, one segment each.
        h.z(1, 0) = 50e-9;
        check_graded_box(h, 2e-6, std::nullopt, 1);
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
        CHECK(g.relaxation_rows == 0);
        CHECK(g.relaxation_vertices == 0);
    }
    SECTION("deep pit: grading kept with local relaxation nodes") {
        h.z(1, 0) = 300e-9;
        WarningCounter warnings;
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
        CHECK(g.levels == 1);
        CHECK(g.levels_unreduced == 1);
        // Node offsets D = 300 nm at the level-1 nodes i = 0 and 2, so z_S = 400 nm there:
        // gaps of 400 nm (two segments) at i = 0 and 2, 100 nm at the pit, and 250 nm at the
        // neighbouring columns i = 3 and (0, 1) where z_S falls back to 100 nm: four
        // relaxation nodes, at most one per column. Rows: rim, anchor (level 0), level 1.
        CHECK(g.relaxation_rows == 1);
        CHECK(g.relaxation_vertices == 4);
        REQUIRE(g.row_ring_sizes.size() >= 4);
        CHECK(g.row_ring_sizes[1] == 80);
        CHECK(g.row_ring_sizes[2] == 40);
        REQUIRE(g.anchor_z.size() == 80);
        CHECK_THAT(g.anchor_z[0], WithinRel(400e-9, 1e-12));
        CHECK_THAT(g.anchor_z[1], WithinRel(400e-9, 1e-12));
        CHECK_THAT(g.anchor_z[3], WithinRel(250e-9, 1e-12));
        CHECK_THAT(g.anchor_z[79], WithinRel(250e-9, 1e-12));
        check_graded_box(h, 2e-6, std::nullopt, 1, rough_rim_aspect_bound(h, 2e-6));
        const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6);
        CHECK(vertical_wall_edges(v, f, h, g.anchor_z, std::nullopt).above_anchor <=
              2.0 * h.dx * (1.0 + 1e-9));
        CHECK(warnings.count() == 0);
    }
    SECTION("deep pits, two levels requested: both levels kept") {
        // M = 2 (400 nm). WP2b: a pit at i = 1 gave M = 0, a pit at i = 2 gave M = 1.
        for (const Index i : {Index{1}, Index{2}}) {
            CAPTURE(i);
            h.z.setZero();
            h.z(i, 0) = 300e-9;
            WarningCounter warnings;
            check_graded_box(h, 2e-6, 400e-9, 2, rough_rim_aspect_bound(h, 2e-6, 400e-9));
            // Gaps up to 400 nm: at most one relaxation node per column.
            CHECK(geometry::detail::box_grading(h, 2e-6, 400e-9).relaxation_rows == 1);
            CHECK(warnings.count() == 0);
        }
    }
    SECTION("safety net: a pit next to the bottom plate gives the uniform box") {
        // 1.85 um pit, depth 2 um (valid: depth > max(xi) + h). The anchor row
        // z_S = z~ + D + h_g reaches 1.95 um at the neighbouring level-1 nodes, so the
        // columns there keep less than a quarter of the mean height below the anchor row.
        h.z(1, 0) = 1.85e-6;
        WarningCounter warnings;
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
        CHECK(g.levels == 0);
        CHECK(g.levels_unreduced == 1);
        CHECK(g.anchor_z.empty());
        CHECK(g.wall_aspect == 0.0);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6);
        CHECK(f.rows() == box_counts(21, 21, 20).f);
        CHECK(closed_and_consistent(f));
        CHECK(warnings.count() == 0);  // detail:: inspection functions are silent
        const TriangleMesh mesh = make_mesh_from_height_map(h, 2e-6);
        CHECK(warnings.count() == 1);  // one warning per mesh built
        CHECK(mesh.triangles() == f);
        CHECK(mesh.is_closed());
    }
}

TEST_CASE("rough surface mesh: generated rough rims keep M = 3 (WP2c acceptance)", "[geometry]") {
    // Backlog WP2c: default rough surfaces with sigma <= 250 nm and Lc >= 100 nm at 50 nm
    // spacing keep M = 3. L = 4 um (81 x 81), depth 2 um: automatic h_c = min(1 um, 500 nm,
    // 500 nm) = 500 nm -> M = 3. Under WP2b sigma = 250 nm, Lc = 100 nm gave the uniform box.
    // (The closing box fraction scales like 1 / L; it is checked at L = 10 um below.)
    // Layout checks on all maps in both builds; the per-triangle checks of these 81 x 81
    // maps in the release build only (time limit of the sanitizer build, which runs the
    // small smooth-rim case below and the rough rims of the tests above per triangle).
    const Real depth = 2e-6;
    for (const auto& [sigma, lc] :
         std::vector<std::pair<Real, Real>>{{250e-9, 100e-9}, {50e-9, 500e-9}}) {
        for (const std::uint64_t seed : {1ULL, 2ULL, 3ULL}) {
            CAPTURE(sigma, lc, seed);
            const HeightMap h =
                generate_gaussian_height_map(make_params(4e-6, 50e-9, sigma, lc, seed));
            REQUIRE(h.z.rows() == 81);
            const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, depth);
            CAPTURE(g.relaxation_rows, g.relaxation_vertices, g.wall_aspect, g.uniform_wall_aspect);
            CHECK(g.levels == 3);
            CHECK(g.levels_unreduced == 3);
            // Smooth rims need (almost) no relaxation nodes, rough ones many (measured:
            // <= 3 of 320 columns for sigma = 50 nm, Lc = 500 nm; ~1000 for 250 nm, 100 nm).
            if (sigma < 100e-9)
                CHECK(g.relaxation_vertices <= 8);
            else
                CHECK(g.relaxation_vertices > 320);
            CHECK(g.wall_aspect <= std::max(4.0, 2.0 * g.uniform_wall_aspect));
#ifdef NDEBUG
            // The plan's reference equals the uniform arrays (walls dominate the bottom).
            const Real uniform = uniform_box_aspect(h, depth);
            CHECK_THAT(g.uniform_wall_aspect, WithinRel(uniform, 1e-9));
            WarningCounter warnings;
            // Smooth rims: the explicit bound 4; rough rims: the safety-net bound.
            check_graded_box(h, depth, std::nullopt, 3,
                             sigma < 100e-9 ? 4.0 : std::max(4.0, 2.0 * uniform));
            CHECK(warnings.count() == 0);
#endif
        }
    }
    SECTION("small smooth-rim case, per triangle in both builds") {
        // sigma = 50 nm, Lc = 500 nm on L = 2 um (41 x 41); h_c = 400 nm gives M = 3.
        const HeightMap h =
            generate_gaussian_height_map(make_params(2e-6, 50e-9, 50e-9, 500e-9, 1));
        REQUIRE(h.z.rows() == 41);
        WarningCounter warnings;
        check_graded_box(h, depth, 400e-9, 3, 4.0);
        CHECK(warnings.count() == 0);
    }
}

TEST_CASE("rough surface mesh: relaxation band keeps vertical wall edges <= 2 h_g (WP2c)",
          "[geometry]") {
    // Review C1: under a rough rim every vertical wall edge between the rim and the anchor
    // row is at most 2 h_g long (h_g = 50 nm), with and without a fine band, and with a fine
    // band (depth 4 um, z_f = 3 um) every vertical wall edge above z_f is at most 2 h_g and
    // at most twice the longest one of the uniform box (~58 nm). Before the fix the
    // relaxation strips under sigma = 250 nm, Lc = 100 nm rims were 500-620 nm tall.
    // Seeds 1-3 in the release build, seed 1 in the sanitizer build (time limit).
    const Real hs = 50e-9;
    const Real bound = 2.0 * hs * (1.0 + 1e-9);
#ifdef NDEBUG
    const std::vector<std::uint64_t> seeds{1, 2, 3};
#else
    const std::vector<std::uint64_t> seeds{1};
#endif
    for (const auto& [sigma, lc] :
         std::vector<std::pair<Real, Real>>{{250e-9, 100e-9}, {50e-9, 100e-9}}) {
        for (const std::uint64_t seed : seeds) {
            CAPTURE(sigma, lc, seed);
            const HeightMap h =
                generate_gaussian_height_map(make_params(4e-6, hs, sigma, lc, seed));
            REQUIRE(h.z.rows() == 81);
            {
                // No fine band: depth 2 um, automatic M = 3.
                const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
                CHECK(g.levels == 3);
                CHECK(g.relaxation_rows >= 1);
                const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6);
                CHECK(closed_and_consistent(f));
                const VerticalEdges e = vertical_wall_edges(v, f, h, g.anchor_z, std::nullopt);
                CAPTURE(e.above_anchor, g.relaxation_rows, g.relaxation_vertices);
                CHECK(e.above_anchor > hs);
                CHECK(e.above_anchor <= bound);
            }
            {
                // Fine band: depth 4 um, z_f = 3 um.
                const Real depth = 4e-6;
                const Real zf = 3e-6;
                const geometry::detail::BoxGrading g =
                    geometry::detail::box_grading(h, depth, std::nullopt, zf);
                CHECK(g.levels == 3);
                CHECK(g.fine_rows >= 59);
                const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, std::nullopt, zf);
                CHECK(closed_and_consistent(f));
                const VerticalEdges e = vertical_wall_edges(v, f, h, g.anchor_z, zf);
                const auto [vu, fu] = geometry::detail::rough_box_arrays(h, depth, hs);
                const VerticalEdges eu = vertical_wall_edges(vu, fu, h, {}, zf);
                CAPTURE(e.above_anchor, e.above_fine, eu.above_fine);
                CHECK(e.above_anchor <= bound);
                CHECK(e.above_fine <= bound);
                CHECK(e.above_fine <= 2.0 * eu.above_fine);
            }
        }
    }
}

TEST_CASE("rough surface mesh: very rough rims and the shape safety net", "[geometry]") {
    SECTION("shallow box under a very rough rim keeps its levels") {
        // A 1 um deep box under sigma = 250 nm, Lc = 100 nm (L = 2 um, 41 x 41, automatic
        // M = 2). With the stitched relaxation band the rows below the anchor row keep their
        // shape (with one relaxation row per ring, as before review C1, this map fell back
        // to the uniform box: graded walls 3.4 x worse than the uniform ones).
        const HeightMap h =
            generate_gaussian_height_map(make_params(2e-6, 50e-9, 250e-9, 100e-9, 1));
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 1e-6);
        CHECK(g.levels_unreduced == 2);
        CHECK(g.levels == 2);
        WarningCounter warnings;
        check_graded_box(h, 1e-6, std::nullopt, 2, rough_rim_aspect_bound(h, 1e-6));
        CHECK(warnings.count() == 0);
    }
    SECTION("an unlucky seed falls back to fewer levels with one warning") {
        // sigma = 250 nm, Lc = 500 nm, L = 4 um, depth 2 um, seed 9: with M = 3 the anchor
        // row of the steep rim shears the 2:1 cells beyond max(4, 2 x the uniform walls), so
        // the safety net reduces M (expected behaviour, documented in rough_surface.hpp and
        // ADR 0006; 1 of 20 seeds in the [.stats] sweep).
        const HeightMap h =
            generate_gaussian_height_map(make_params(4e-6, 50e-9, 250e-9, 500e-9, 9));
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
        CAPTURE(g.wall_aspect, g.uniform_wall_aspect);
        CHECK(g.levels_unreduced == 3);
        CHECK(g.levels == 2);
        CHECK(g.wall_aspect <= std::max(4.0, 2.0 * g.uniform_wall_aspect));
        WarningCounter warnings;
        const TriangleMesh mesh = make_mesh_from_height_map(h, 2e-6);
        CHECK(warnings.count() == 1);
        CHECK(warnings.count("using 2 levels") == 1);
        CHECK(mesh.is_closed());
        CHECK(mesh.num_triangles() == graded_triangle_count(81, 81, g));
    }
}

TEST_CASE("rough surface mesh: closing box fraction at L = 10 um (WP2c)", "[geometry]") {
    // Layout only (no arrays): 201 x 201 grid, depth 2 um, automatic M = 3. Backlog WP2c:
    // closing box <= 10 % of the top-face triangles. Measured (seeds 1-5): sigma = 50 nm,
    // Lc = 500 nm 7.19 %; sigma = 250 nm, Lc = 500 nm 7.93-8.12 %; sigma = 50 nm,
    // Lc = 100 nm 8.43-8.54 %. sigma = 250 nm, Lc = 100 nm needs 13.7-14.0 % (about 2700
    // relaxation nodes keep the vertical wall edges under the rim <= 2 h_g, review C1):
    // above the 10 % target, bounded here at the measured worst + 0.5 % (ADR 0006).
#ifdef NDEBUG
    const std::vector<std::uint64_t> seeds{1, 2, 3, 4, 5};
#else
    const std::vector<std::uint64_t> seeds{1};
#endif
    struct Corner {
        Real sigma;
        Real lc;
        Real max_ratio;
    };
    for (const Corner& c : {Corner{250e-9, 100e-9, 0.145}, Corner{50e-9, 500e-9, 0.10},
                            Corner{50e-9, 100e-9, 0.10}, Corner{250e-9, 500e-9, 0.10}}) {
        for (const std::uint64_t seed : seeds) {
            CAPTURE(c.sigma, c.lc, seed);
            const HeightMap h =
                generate_gaussian_height_map(make_params(10e-6, 50e-9, c.sigma, c.lc, seed));
            const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
            CHECK(g.levels == 3);
            const Index n_top = 2 * 200 * 200;
            const Real ratio = static_cast<Real>(graded_triangle_count(201, 201, g) - n_top) /
                               static_cast<Real>(n_top);
            CAPTURE(ratio, g.relaxation_rows, g.relaxation_vertices);
            CHECK(ratio <= c.max_ratio);
        }
    }
}

namespace {

/// Extents of the wall triangles with a vertex above z_f (z < z_f) and of those below.
struct BandExtent {
    Index count = 0;              ///< wall triangles with a vertex above z_f
    Real horizontal = 0.0;        ///< their largest horizontal extent
    Real vertical = 0.0;          ///< their largest vertical extent
    Real horizontal_below = 0.0;  ///< largest horizontal extent of the other wall triangles
};
BandExtent fine_band_extent(const Vertices& v, const Triangles& f, const HeightMap& h, Real zf) {
    const Real hx = 0.5 * static_cast<Real>(h.z.rows() - 1) * h.dx;
    const Real hy = 0.5 * static_cast<Real>(h.z.cols() - 1) * h.dy;
    const Real tol = 1e-9 * std::min(h.dx, h.dy);
    BandExtent e;
    for (Index t = 0; t < f.rows(); ++t) {
        Vec3 lo = Vec3::Constant(std::numeric_limits<Real>::infinity());
        Vec3 hi = -lo;
        for (Index q = 0; q < 3; ++q) {
            const Vec3 p = v.row(f(t, q)).transpose();
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
        const bool wall = (hi.x() - lo.x() < tol && std::abs(std::abs(lo.x()) - hx) < tol) ||
                          (hi.y() - lo.y() < tol && std::abs(std::abs(lo.y()) - hy) < tol);
        if (!wall)
            continue;
        const Real horizontal = std::max(hi.x() - lo.x(), hi.y() - lo.y());
        if (lo.z() < zf - 1e-3 * h.dx) {
            ++e.count;
            e.horizontal = std::max(e.horizontal, horizontal);
            e.vertical = std::max(e.vertical, hi.z() - lo.z());
        } else {
            e.horizontal_below = std::max(e.horizontal_below, horizontal);
        }
    }
    return e;
}

}  // namespace

TEST_CASE("rough surface mesh: fine band keeps the top-face spacing down to box_fine_depth",
          "[geometry]") {
    const Real depth = 4e-6;
    const Real zf = 3e-6;
    const Real hs = 50e-9;
    SECTION("flat rim: exact rows") {
        // 41 x 41 bump map (L = 2 um, h = 50 nm, rim heights 0), h_c = 400 nm -> M = 3.
        // Anchor row at h_g = 50 nm, 59 fine-band rows of 50 nm down to z = 3 um,
        // transitions 50 / 100 / 200 nm to 3.35 um, two coarse rows of 325 nm.
        const HeightMap h = bump_map(41, 41, hs, hs, 50e-9);
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, depth, 400e-9, zf);
        CHECK(g.levels == 3);
        CHECK(g.relaxation_rows == 0);
        CHECK(g.fine_rows == 59);
        // (Built element by element: vector::insert of an initializer list after a fill
        // insert trips a GCC 16 -Warray-bounds false positive.)
        std::vector<Index> rings(61, 160);
        for (const Index r : {80, 40, 20, 20, 20}) rings.push_back(r);
        CHECK(g.row_ring_sizes == rings);
        check_graded_box(h, depth, 400e-9, 3, 4.0, zf);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, 400e-9, zf);
        const BandExtent e = fine_band_extent(v, f, h, zf);
        CHECK(e.count == 2 * 160 * 60);
        CHECK(e.horizontal <= hs * (1.0 + 1e-9));
        CHECK(e.vertical <= hs * (1.0 + 1e-9));
        CHECK(e.horizontal_below == 400e-9);  // grading happens below z_f
        // Without the fine band the first transition starts 50 nm below the rim.
        const geometry::detail::BoxGrading g0 = geometry::detail::box_grading(h, depth, 400e-9);
        CHECK(g0.fine_rows == 0);
        CHECK(g0.row_ring_sizes[2] == 80);
    }
    SECTION("rough rim") {
        // sigma = 50 nm, Lc = 250 nm, seed 3: relaxation nodes under the rim; every vertical
        // wall edge above z_f is at most 2 h_g (fine-band rows at most h_g), and the wall
        // triangles there are at most 3 h_g tall (rim steps).
        const HeightMap h = generate_gaussian_height_map(make_params(2e-6, hs, 50e-9, 250e-9, 3));
        REQUIRE(h.z.rows() == 41);
        const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, depth, 400e-9, zf);
        CHECK(g.relaxation_rows >= 1);
        CHECK(g.fine_rows >= 59);
        check_graded_box(h, depth, 400e-9, 3, rough_rim_aspect_bound(h, depth, 400e-9, zf), zf);
        const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, 400e-9, zf);
        const BandExtent e = fine_band_extent(v, f, h, zf);
        CHECK(e.count >= 2 * 160 * 60);
        CHECK(e.horizontal <= hs * (1.0 + 1e-9));
        CHECK(e.vertical <= 3.0 * hs);
        CHECK(e.horizontal_below > 300e-9);
        CHECK(vertical_wall_edges(v, f, h, g.anchor_z, zf).above_fine <= 2.0 * hs * (1.0 + 1e-9));
        // make_rough_surface_mesh passes RoughSurfaceParams::box_fine_depth through.
        RoughSurfaceParams p = make_params(2e-6, hs, 50e-9, 250e-9, 3);
        p.box_depth = depth;
        p.box_mesh_size = 400e-9;
        p.box_fine_depth = zf;
        const TriangleMesh mesh = make_rough_surface_mesh(p);
        CHECK(mesh.triangles() == f);
        CHECK(mesh.vertices() == v);
    }
}

TEST_CASE("rough surface mesh: box_fine_depth validation and the uniform fallback", "[geometry]") {
    const HeightMap h = pattern_map(21, 21, 100e-9, 100e-9, 0.25e-9);  // heights within +-5 nm
    const Real depth = 2e-6;
    for (const Real bad : {0.0, -100e-9, std::numeric_limits<Real>::quiet_NaN(),
                           std::numeric_limits<Real>::infinity(), depth, 3e-6}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(make_mesh_from_height_map(h, depth, std::nullopt, bad),
                        std::invalid_argument);
        CHECK_THROWS_AS(geometry::detail::rough_box_arrays(h, depth, std::nullopt, bad),
                        std::invalid_argument);
        CHECK_THROWS_AS(geometry::detail::box_grading(h, depth, std::nullopt, bad),
                        std::invalid_argument);
        RoughSurfaceParams p = make_params(1e-6, 100e-9, 20e-9, 200e-9, 5);
        p.box_depth = depth;
        p.box_fine_depth = bad;
        CHECK_THROWS_AS(make_rough_surface_mesh(p), std::invalid_argument);
    }
    // Automatic M = 1 (250 nm). One level needs (1.5 * 2 - 1) h_g = 200 nm below z_f and
    // room for half a coarse row below the transition row.
    CHECK(geometry::detail::box_grading(h, depth, std::nullopt, 1.75e-6).levels == 1);
    check_graded_box(h, depth, std::nullopt, 1, 4.0, 1.75e-6);
    // A fine band reaching to within 2 h_g of the bottom plate gives the uniform box (by
    // request, so without a warning).
    const geometry::detail::BoxGrading g =
        geometry::detail::box_grading(h, depth, std::nullopt, 1.85e-6);
    CHECK(g.levels == 0);
    CHECK(g.levels_unreduced == 0);
    const auto [vu, fu] = geometry::detail::rough_box_arrays(h, depth, 100e-9);
    const auto [v, f] = geometry::detail::rough_box_arrays(h, depth, std::nullopt, 1.85e-6);
    CHECK(v == vu);
    CHECK(f == fu);
    {
        // Logged once per mesh as SBEM_INFO (requested by the caller), not as a warning.
        WarningCounter warnings;
        WarningCounter infos(spdlog::level::info);
        const TriangleMesh mesh = make_mesh_from_height_map(h, depth, std::nullopt, 1.85e-6);
        CHECK(mesh.triangles() == fu);
        CHECK(warnings.count() == 0);
        CHECK(infos.count("leaves room for 0 of 1 coarsening levels") == 1);
        CHECK(infos.count("(uniform box)") == 1);
    }
    // A fine depth above the anchor row adds no rows.
    const auto [v0, f0] = geometry::detail::rough_box_arrays(h, depth);
    const auto [v1, f1] = geometry::detail::rough_box_arrays(h, depth, std::nullopt, 10e-9);
    CHECK(geometry::detail::box_grading(h, depth, std::nullopt, 10e-9).fine_rows == 0);
    CHECK(v1 == v0);
    CHECK(f1 == f0);
}

TEST_CASE("rough surface mesh: invalid box_mesh_size throws", "[geometry]") {
    const HeightMap h = pattern_map(11, 11, 100e-9, 100e-9);
    for (const Real bad : {0.0, -100e-9, std::numeric_limits<Real>::quiet_NaN(),
                           std::numeric_limits<Real>::infinity()}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(make_mesh_from_height_map(h, 2e-6, bad), std::invalid_argument);
        CHECK_THROWS_AS(geometry::detail::rough_box_arrays(h, 2e-6, bad), std::invalid_argument);
        RoughSurfaceParams p = make_params(1e-6, 100e-9, 20e-9, 200e-9, 5);
        p.box_mesh_size = bad;
        CHECK_THROWS_AS(make_rough_surface_mesh(p), std::invalid_argument);
    }
}

// Manual (hidden): reproduces the box fractions documented in rough_surface.hpp and ADR 0006.
// Run: specklebem_unit_tests "[.stats]"
TEST_CASE("rough surface mesh: box statistics (manual)", "[.stats]") {
    const auto box_percent = [](const HeightMap& h, const geometry::detail::BoxGrading& g) {
        const Index nx = h.z.rows();
        const Index ny = h.z.cols();
        const Index top = 2 * (nx - 1) * (ny - 1);
        return 100.0 * static_cast<Real>(graded_triangle_count(nx, ny, g) - top) /
               static_cast<Real>(top);
    };
    for (const Real L : {4e-6, 10e-6}) {
        for (const auto& [sigma, lc] : std::vector<std::pair<Real, Real>>{
                 {250e-9, 100e-9}, {50e-9, 500e-9}, {250e-9, 500e-9}, {50e-9, 100e-9}}) {
            for (std::uint64_t seed = 1; seed <= 5; ++seed) {
                const HeightMap h =
                    generate_gaussian_height_map(make_params(L, 50e-9, sigma, lc, seed));
                const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
                const auto [v, f] = geometry::detail::rough_box_arrays(h, 2e-6);
                const VerticalEdges e = vertical_wall_edges(v, f, h, g.anchor_z, std::nullopt);
                WARN("L " << L << " sigma " << sigma << " Lc " << lc << " seed " << seed << ": M "
                          << g.levels << " (of " << g.levels_unreduced << "), R "
                          << g.relaxation_rows << ", relaxation nodes " << g.relaxation_vertices
                          << ", box " << box_percent(h, g) << " %, wall aspect " << g.wall_aspect
                          << " (uniform box " << g.uniform_wall_aspect << ", ratio "
                          << g.wall_aspect / g.uniform_wall_aspect
                          << "), longest vertical edge above the anchor " << e.above_anchor);
            }
        }
    }
    // Vertical resolution under the rim (review C1): L = 4 um, depth 4 um, fine band to 3 um.
    for (const auto& [sigma, lc] : std::vector<std::pair<Real, Real>>{
             {250e-9, 100e-9}, {100e-9, 200e-9}, {50e-9, 100e-9}, {50e-9, 500e-9}}) {
        for (std::uint64_t seed = 1; seed <= 3; ++seed) {
            const HeightMap h =
                generate_gaussian_height_map(make_params(4e-6, 50e-9, sigma, lc, seed));
            const geometry::detail::BoxGrading g =
                geometry::detail::box_grading(h, 4e-6, std::nullopt, 3e-6);
            const auto [v, f] = geometry::detail::rough_box_arrays(h, 4e-6, std::nullopt, 3e-6);
            const VerticalEdges e = vertical_wall_edges(v, f, h, g.anchor_z, 3e-6);
            const geometry::detail::BoxGrading g0 = geometry::detail::box_grading(h, 4e-6);
            const auto [v0, f0] = geometry::detail::rough_box_arrays(h, 4e-6);
            const VerticalEdges e0 = vertical_wall_edges(v0, f0, h, g0.anchor_z, std::nullopt);
            const auto [vu, fu] = geometry::detail::rough_box_arrays(h, 4e-6, 50e-9);
            const VerticalEdges eu = vertical_wall_edges(vu, fu, h, {}, 3e-6);
            WARN("C1 sigma " << sigma << " Lc " << lc << " seed " << seed << ": fine band M "
                             << g.levels << " R " << g.relaxation_rows << ", above z_f "
                             << e.above_fine << ", above anchor " << e.above_anchor
                             << "; no fine band M " << g0.levels << " R " << g0.relaxation_rows
                             << ", above anchor " << e0.above_anchor << "; uniform above z_f "
                             << eu.above_fine);
        }
    }
    // Aspect safety net: seed sweep of the rough corners (review W3).
    for (const auto& [sigma, lc] :
         std::vector<std::pair<Real, Real>>{{250e-9, 100e-9}, {250e-9, 500e-9}}) {
        for (std::uint64_t seed = 1; seed <= 20; ++seed) {
            const HeightMap h =
                generate_gaussian_height_map(make_params(4e-6, 50e-9, sigma, lc, seed));
            const geometry::detail::BoxGrading g = geometry::detail::box_grading(h, 2e-6);
            WARN("W3 sigma " << sigma << " Lc " << lc << " seed " << seed << ": M " << g.levels
                             << " (of " << g.levels_unreduced << "), wall aspect " << g.wall_aspect
                             << ", uniform " << g.uniform_wall_aspect << ", ratio "
                             << g.wall_aspect / g.uniform_wall_aspect);
        }
    }
    // Fine band for Si at 500 nm: 3 delta = 3.387 um (material::field_decay_length).
    for (const Real L : {10e-6, 30e-6}) {
        for (const Real depth : {4e-6, 5.65e-6}) {
            const HeightMap h =
                generate_gaussian_height_map(make_params(L, 50e-9, 50e-9, 500e-9, 1));
            const geometry::detail::BoxGrading g =
                geometry::detail::box_grading(h, depth, std::nullopt, 3.387e-6);
            const geometry::detail::BoxGrading gu = geometry::detail::box_grading(h, depth, 50e-9);
            WARN("Si fine band: L " << L << " depth " << depth << ": M " << g.levels
                                    << ", fine rows " << g.fine_rows << ", box "
                                    << box_percent(h, g) << " %, uniform box " << box_percent(h, gu)
                                    << " %");
        }
    }
    RoughSurfaceParams p = make_params(10e-6, 50e-9, 50e-9, 500e-9, 1);
    p.box_depth = 4e-6;
    p.box_fine_depth = 3.387e-6;
    const auto start = std::chrono::steady_clock::now();
    const TriangleMesh mesh = make_rough_surface_mesh(p);
    WARN("Si fine band mesh L = 10 um, depth 4 um: "
         << mesh.num_triangles() << " triangles, "
         << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
         << " s");
}
