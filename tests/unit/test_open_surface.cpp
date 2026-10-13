#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace specklebem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using geometry::HeightMap;
using geometry::make_open_surface_mesh;
using geometry::OpenSurfaceMesh;
using geometry::OpenSurfaceParams;
using geometry::SurfaceZone;

namespace {

constexpr Real kH = 50e-9;      // grid spacing
constexpr Real kL = 2e-6;       // patch size
constexpr Real kWt = 0.5e-6;    // taper
constexpr Real kWg = 0.2e-6;    // gap
constexpr Real kWpml = 0.5e-6;  // PML
constexpr Real kSigma = 30e-9;  // rms height
constexpr Real kLc = 250e-9;    // correlation length

/// Gaussian map exactly covering patch + taper (61 x 61 points, dx = 50 nm), fixed seed.
HeightMap gaussian_map(std::uint64_t seed = 20261013) {
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = kL + 2.0 * kWt;
    rp.mesh_size = kH;
    rp.rms_roughness = kSigma;
    rp.correlation_length = kLc;
    rp.seed = seed;
    return geometry::generate_gaussian_height_map(rp);
}

OpenSurfaceParams default_params() {
    return {kL, kWt, kWg, kWpml};
}

/// Map node coordinate x_i = -(n - 1) d / 2 + i d (HeightMap convention).
Real map_coord(Index i, Index n, Real d) {
    return -static_cast<Real>(n - 1) * d / 2.0 + static_cast<Real>(i) * d;
}

/// Analytic height map xi(x, y) on a centred n x n grid of spacing d.
template <typename F>
HeightMap analytic_map(Index n, Real d, F&& xi) {
    HeightMap h;
    h.dx = d;
    h.dy = d;
    h.z.resize(n, n);
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            h.z(i, j) = xi(map_coord(i, n, d), map_coord(j, n, d));
        }
    }
    return h;
}

Real max_abs_xy(const Vec3& r) {
    return std::max(std::abs(r.x()), std::abs(r.y()));
}

/// Captures log lines of at least the given level while alive.
class LogCapture {
public:
    explicit LogCapture(spdlog::level::level_enum level)
        : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64)),
          saved_level_(spdlog::default_logger()->level()) {
        sink_->set_level(level);
        spdlog::default_logger()->set_level(std::min(saved_level_, level));
        spdlog::default_logger()->sinks().push_back(sink_);
    }
    ~LogCapture() {
        auto& sinks = spdlog::default_logger()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
        spdlog::default_logger()->set_level(saved_level_);
    }
    LogCapture(const LogCapture&) = delete;
    LogCapture& operator=(const LogCapture&) = delete;
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

}  // namespace

TEST_CASE("open surface: zone widths, counts and band areas", "[open_surface]") {
    const HeightMap h = gaussian_map();
    REQUIRE(h.z.rows() == 61);
    const OpenSurfaceMesh s = make_open_surface_mesh(h, default_params());
    CHECK_THAT(s.patch_half, WithinRel(1.0e-6, 1e-12));
    CHECK_THAT(s.taper_end, WithinRel(1.5e-6, 1e-12));
    CHECK_THAT(s.pml_start, WithinRel(1.7e-6, 1e-12));
    CHECK_THAT(s.outer_half, WithinRel(2.2e-6, 1e-12));

    const auto& mesh = s.mesh;
    const Index m = 88;  // cells per axis: (40 + 2 (10 + 4 + 10))
    REQUIRE(mesh.num_vertices() == (m + 1) * (m + 1));
    REQUIRE(mesh.num_triangles() == 2 * m * m);
    REQUIRE(s.zone.size() == static_cast<std::size_t>(mesh.num_triangles()));

    // Cell counts per band: (2 a_k / h)^2 - (2 a_(k-1) / h)^2 cells, two triangles each.
    const std::array<Index, 5> across = {0, 40, 60, 68, 88};
    const std::array<Real, 4> half = {s.patch_half, s.taper_end, s.pml_start, s.outer_half};
    std::array<Index, 4> count{};
    std::array<Real, 4> projected{};
    std::array<Real, 4> area{};
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const auto z = static_cast<std::size_t>(s.zone[static_cast<std::size_t>(t)]);
        ++count[z];
        projected[z] += mesh.area(t) * std::abs(mesh.normal(t).z());
        area[z] += mesh.area(t);
        // Zone by centroid.
        const Real c = max_abs_xy(mesh.centroid(t));
        const std::size_t expected = c <= half[0] ? 0 : c <= half[1] ? 1 : c <= half[2] ? 2 : 3;
        CHECK(z == expected);
    }
    for (std::size_t k = 0; k < 4; ++k) {
        const Index cells = across[k + 1] * across[k + 1] - across[k] * across[k];
        CHECK(count[k] == 2 * cells);
        const Real inner = k == 0 ? 0.0 : 2.0 * half[k - 1];
        const Real band = 4.0 * half[k] * half[k] - inner * inner;
        CHECK_THAT(projected[k], WithinRel(band, 1e-12));
    }
    // Rough patch and taper are larger than their projection; gap and PML are flat.
    CHECK(area[0] > projected[0] * (1.0 + 1e-4));
    CHECK(area[1] > projected[1]);
    CHECK_THAT(area[2], WithinRel(projected[2], 1e-14));
    CHECK_THAT(area[3], WithinRel(projected[3], 1e-14));
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const SurfaceZone z = s.zone[static_cast<std::size_t>(t)];
        if (z == SurfaceZone::Gap || z == SurfaceZone::Pml) {
            const auto tri = mesh.triangles().row(t);
            for (Index c = 0; c < 3; ++c) {
                CHECK(mesh.vertices()(tri(c), 2) == 0.0);
            }
        }
    }
}

TEST_CASE("open surface: heights equal the map on the patch and vanish beyond the taper",
          "[open_surface]") {
    const HeightMap h = gaussian_map();
    const OpenSurfaceMesh s = make_open_surface_mesh(h, default_params());
    const auto& v = s.mesh.vertices();
    const Index n = 89;
    const Index nm = h.z.rows();
    const Index shift = (n - nm) / 2;  // mesh node i <-> map node i - shift
    const Real tol = 1e-9 * kH;
    Index n_patch = 0;
    Index n_taper = 0;
    Index n_flat = 0;
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            const Index row = i * n + j;
            const Vec3 r = v.row(row).transpose();
            const Real a = max_abs_xy(r);
            if (a >= s.taper_end - tol) {
                CHECK(r.z() == 0.0);  // exactly flat at and beyond the taper end
                ++n_flat;
                continue;
            }
            // Same grid as the map.
            CHECK_THAT(r.x(), WithinAbs(map_coord(i - shift, nm, h.dx), 1e-12 * kL));
            CHECK_THAT(r.y(), WithinAbs(map_coord(j - shift, nm, h.dy), 1e-12 * kL));
            const Real xi = h.z(i - shift, j - shift);
            if (a <= s.patch_half + tol) {
                CHECK(r.z() == xi);  // bit-identical on the patch (window 1)
                ++n_patch;
            } else {
                const Real w = geometry::taper_window(r.x(), s.patch_half, kWt) *
                               geometry::taper_window(r.y(), s.patch_half, kWt);
                CHECK_THAT(r.z(), WithinAbs(xi * w, 1e-12 * kSigma));
                CHECK(std::abs(r.z()) <= std::abs(xi));
                ++n_taper;
            }
        }
    }
    CHECK(n_patch == 41 * 41);
    CHECK(n_taper == 59 * 59 - 41 * 41);
    CHECK(n_flat == 89 * 89 - 59 * 59);
    // The window itself.
    CHECK(geometry::taper_window(0.0, 1e-6, 0.5e-6) == 1.0);
    CHECK(geometry::taper_window(-1e-6, 1e-6, 0.5e-6) == 1.0);
    CHECK_THAT(geometry::taper_window(1.25e-6, 1e-6, 0.5e-6), WithinAbs(0.5, 1e-15));
    CHECK(geometry::taper_window(-1.5e-6, 1e-6, 0.5e-6) == 0.0);
    CHECK(geometry::taper_window(3e-6, 1e-6, 0.5e-6) == 0.0);
    CHECK_THROWS_AS(geometry::taper_window(0.0, 1e-6, 0.0), std::invalid_argument);
}

TEST_CASE("open surface: the tapered surface is C1 across both taper ends", "[open_surface]") {
    // Smooth analytic map with non-zero slope at all four taper ends on the line y = 0. The
    // jump between the second-order one-sided slopes (3 f_0 - 4 f_-1 + f_-2) / (2 h) and
    // (-3 f_0 + 4 f_1 - f_2) / (2 h) at a node is O(h^2) for a C1 profile (h^2 |[f''']| / 3;
    // the window's f'' jumps there) and tends to the slope jump for a C0 kink. Halving h must
    // therefore quarter it.
    constexpr Real kA = 40e-9;
    constexpr Real kP = 0.7e-6;
    const auto xi = [](Real x, Real y) {
        return kA * std::cos(2.0 * constants::pi * x / kP + 0.3) *
               (1.0 + 0.2 * std::cos(2.0 * constants::pi * y / kP));
    };
    constexpr Real kPatch = 1e-6;
    constexpr Real kTaper = 0.4e-6;
    const OpenSurfaceParams p{kPatch, kTaper, 0.0, 0.1e-6};
    struct Jumps {
        Real patch_edge;
        Real taper_end;
    };
    const auto jumps = [&](Real d) {
        const auto n_map = static_cast<Index>(std::lround((kPatch + 2.0 * kTaper) / d)) + 1;
        const HeightMap h = analytic_map(n_map, d, xi);
        const OpenSurfaceMesh s = make_open_surface_mesh(h, p);
        const auto& v = s.mesh.vertices();
        const auto n = static_cast<Index>(std::lround(2.0 * s.outer_half / d)) + 1;
        const Index j0 = (n - 1) / 2;  // the line y = 0
        REQUIRE(v(j0, 1) == 0.0);
        const auto z = [&](Index i) { return v(i * n + j0, 2); };
        const auto jump_at = [&](Real x_node) {
            const auto i = static_cast<Index>(std::lround((x_node + s.outer_half) / d));
            REQUIRE_THAT(v(i * n + j0, 0), WithinAbs(x_node, 1e-9 * d));
            const Real left = (3.0 * z(i) - 4.0 * z(i - 1) + z(i - 2)) / (2.0 * d);
            const Real right = (-3.0 * z(i) + 4.0 * z(i + 1) - z(i + 2)) / (2.0 * d);
            return std::abs(right - left);
        };
        // Largest of the two ends of each kind (+-patch edge, +-taper end).
        return Jumps{std::max(jump_at(s.patch_half), jump_at(-s.patch_half)),
                     std::max(jump_at(s.taper_end), jump_at(-s.taper_end))};
    };
    const Jumps coarse = jumps(20e-9);
    const Jumps fine = jumps(10e-9);
    // Small against the slope scale 2 pi A / P = 0.36 (a C0 kink would be O(1) of it) ...
    const Real slope_scale = 2.0 * constants::pi * kA / kP;
    CHECK(coarse.patch_edge < 0.05 * slope_scale);
    CHECK(coarse.taper_end < 0.05 * slope_scale);
    // ... and second order in h.
    CHECK(fine.patch_edge < 0.35 * coarse.patch_edge);
    CHECK(fine.patch_edge > 0.15 * coarse.patch_edge);
    CHECK(fine.taper_end < 0.35 * coarse.taper_end);
    CHECK(fine.taper_end > 0.15 * coarse.taper_end);
}

TEST_CASE("open surface: orientation, single component, boundary on the outer square",
          "[open_surface]") {
    const OpenSurfaceMesh s = make_open_surface_mesh(gaussian_map(), default_params());
    const auto& mesh = s.mesh;
    const auto& v = mesh.vertices();
    const auto& f = mesh.triangles();
    // CCW seen from R1 (z < 0): the raw cross product of the stored order points to -z.
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 a = v.row(f(t, 0)).transpose();
        const Vec3 b = v.row(f(t, 1)).transpose();
        const Vec3 c = v.row(f(t, 2)).transpose();
        const Vec3 cr = (b - a).cross(c - a);
        CHECK(cr.z() < 0.0);
        CHECK(mesh.normal(t).z() < 0.0);
    }
    CHECK(mesh.is_consistently_oriented());
    CHECK_FALSE(mesh.is_closed());
    CHECK(mesh.num_components() == 1);
    const auto q = mesh.quality();
    CHECK(q.num_vertex_components == 1);
    CHECK(q.euler_characteristic == 1);  // a disk
    CHECK(q.num_unreferenced_vertices == 0);

    // Boundary edges: exactly the 4 M edges of the outer square.
    const Index m = 88;
    const auto& be = mesh.boundary_edges();
    REQUIRE(mesh.num_boundary_edges() == 4 * m);
    const Real tol = 1e-9 * kH;
    std::set<Index> boundary_vertices;
    for (Index e = 0; e < be.rows(); ++e) {
        const Vec3 a = v.row(be(e, 0)).transpose();
        const Vec3 b = v.row(be(e, 1)).transpose();
        const bool on_x =
            std::abs(std::abs(a.x()) - s.outer_half) < tol && std::abs(b.x() - a.x()) < tol;
        const bool on_y =
            std::abs(std::abs(a.y()) - s.outer_half) < tol && std::abs(b.y() - a.y()) < tol;
        CHECK((on_x || on_y));
        CHECK_THAT((b - a).norm(), WithinRel(kH, 1e-9));
        boundary_vertices.insert(be(e, 0));
        boundary_vertices.insert(be(e, 1));
    }
    CHECK(static_cast<Index>(boundary_vertices.size()) == 4 * m);
    for (Index i = 0; i < mesh.num_vertices(); ++i) {
        const bool outer = max_abs_xy(v.row(i).transpose()) > s.outer_half - tol;
        CHECK(outer == (boundary_vertices.count(i) == 1));
    }
}

TEST_CASE("open surface: RWG functions only on interior edges", "[open_surface]") {
    const OpenSurfaceMesh s = make_open_surface_mesh(gaussian_map(), default_params());
    const auto& mesh = s.mesh;
    const basis::RwgSpace space(mesh);
    const Index m = 88;
    // Grid edges: 2 M (M + 1) axis-parallel + M^2 diagonals; minus the 4 M boundary edges.
    const Index all_edges = 2 * m * (m + 1) + m * m;
    CHECK(mesh.num_edges() == all_edges - 4 * m);
    CHECK(space.size() == mesh.num_edges());
    // No basis function lives on an outer-square edge.
    const auto& v = mesh.vertices();
    const Real tol = 1e-9 * kH;
    for (Index e = 0; e < mesh.num_edges(); ++e) {
        const Vec3 a = v.row(mesh.edges()(e, 0)).transpose();
        const Vec3 b = v.row(mesh.edges()(e, 1)).transpose();
        // Along one side of the outer square. (The diagonals of the two corner cells join
        // two different sides; they are interior edges.)
        const bool on_x =
            std::abs(std::abs(a.x()) - s.outer_half) < tol && std::abs(b.x() - a.x()) < tol;
        const bool on_y =
            std::abs(std::abs(a.y()) - s.outer_half) < tol && std::abs(b.y() - a.y()) < tol;
        CHECK_FALSE((on_x || on_y));
    }
    // Every basis function has one plus and one minus triangle: sum of supports = 2 N, and the
    // triangles along the outer edge carry fewer than three.
    Index supports = 0;
    Index reduced = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const auto sup = space.support(t);
        supports += sup.count;
        if (sup.count < 3) {
            ++reduced;
            CHECK(max_abs_xy(mesh.centroid(t)) > s.outer_half - kH);
        }
    }
    CHECK(supports == 2 * space.size());
    CHECK(reduced == 4 * m - 2);  // two corner triangles hold two boundary edges each
}

TEST_CASE("open surface: PML stretch matches the zones", "[open_surface]") {
    const OpenSurfaceMesh s = make_open_surface_mesh(gaussian_map(), default_params());
    const geometry::CoordinateStretch st = geometry::make_pml_stretch(s, 4.0);
    REQUIRE(st.profile_x().has_value());
    CHECK(st.profile_x()->start == s.pml_start);
    CHECK_THAT(st.profile_x()->width, WithinRel(kWpml, 1e-12));
    CHECK(st.profile_x()->order == 2.0);
    CHECK(st.profile_y()->sigma_max == 4.0);
    for (Index t = 0; t < s.mesh.num_triangles(); ++t) {
        const bool pml = s.zone[static_cast<std::size_t>(t)] == SurfaceZone::Pml;
        CHECK(st.in_pml(s.mesh.centroid(t)) == pml);
    }
    CHECK_THROWS_AS(geometry::make_pml_stretch(s, -1.0), std::invalid_argument);
}

TEST_CASE("open surface: rounding to the grid, larger maps, logging", "[open_surface]") {
    SECTION("exact widths log nothing") {
        LogCapture log(spdlog::level::info);
        (void)make_open_surface_mesh(gaussian_map(), default_params());
        CHECK(log.count("rounded up") == 0);
    }
    SECTION("widths off the grid are rounded up") {
        LogCapture log(spdlog::level::info);
        const OpenSurfaceParams p{1.93e-6, 0.47e-6, 0.11e-6, 0.42e-6};
        const OpenSurfaceMesh s = make_open_surface_mesh(gaussian_map(), p);
        CHECK(log.count("rounded up") == 1);
        CHECK_THAT(s.patch_half, WithinRel(1.0e-6, 1e-12));                // 40 cells
        CHECK_THAT(s.taper_end - s.patch_half, WithinRel(0.5e-6, 1e-9));   // 10 cells
        CHECK_THAT(s.pml_start - s.taper_end, WithinRel(0.15e-6, 1e-9));   // 3 cells
        CHECK_THAT(s.outer_half - s.pml_start, WithinRel(0.45e-6, 1e-9));  // 9 cells
    }
    SECTION("odd map grid: the patch edge moves to the next half-cell line") {
        // 72 points: cells 71 (odd), lines at (k + 1/2) h; L = 2 um -> 41 cells.
        const HeightMap h =
            analytic_map(72, kH, [](Real x, Real y) { return 1e-9 * (x + y) / kH; });
        const OpenSurfaceMesh s = make_open_surface_mesh(h, {kL, kWt, 0.0, kWpml});
        CHECK_THAT(s.patch_half, WithinRel(1.025e-6, 1e-12));
        CHECK_THAT(s.outer_half, WithinRel(1.025e-6 + 1.0e-6, 1e-12));
        // Patch heights equal the map (grid nodes coincide).
        const auto& v = s.mesh.vertices();
        for (Index i = 0; i < s.mesh.num_vertices(); ++i) {
            const Vec3 r = v.row(i).transpose();
            if (max_abs_xy(r) <= s.patch_half + 1e-9 * kH) {
                CHECK_THAT(r.z(), WithinAbs(1e-9 * (r.x() + r.y()) / kH, 1e-20));
            }
        }
    }
    SECTION("a larger map: heights beyond the taper are ignored") {
        HeightMap h = analytic_map(101, kH, [](Real x, Real) { return 5e-9 * std::sin(1e7 * x); });
        // NaN outside |x|, |y| < a_1 = 1.5 um (map nodes with |x| >= 1.5 um).
        for (Index i = 0; i < 101; ++i) {
            for (Index j = 0; j < 101; ++j) {
                const Real x = map_coord(i, 101, kH);
                const Real y = map_coord(j, 101, kH);
                if (std::max(std::abs(x), std::abs(y)) > 1.5e-6 - 1e-3 * kH) {
                    h.z(i, j) = std::numeric_limits<Real>::quiet_NaN();
                }
            }
        }
        const OpenSurfaceMesh s = make_open_surface_mesh(h, default_params());
        CHECK_THAT(s.outer_half, WithinRel(2.2e-6, 1e-12));
        for (Index i = 0; i < s.mesh.num_vertices(); ++i) {
            CHECK(std::isfinite(s.mesh.vertices()(i, 2)));
        }
    }
}

TEST_CASE("open surface: invalid input throws", "[open_surface]") {
    const HeightMap h = gaussian_map();
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    const auto throws = [&](const HeightMap& map, const OpenSurfaceParams& p) {
        CHECK_THROWS_AS(make_open_surface_mesh(map, p), std::invalid_argument);
    };
    // Non-positive widths.
    throws(h, {0.0, kWt, kWg, kWpml});
    throws(h, {-kL, kWt, kWg, kWpml});
    throws(h, {kL, kWt, kWg, 0.0});
    throws(h, {kL, kWt, kWg, -kWpml});
    throws(h, {kL, kWt, -kH, kWpml});
    throws(h, {kL, 0.0, kWg, kWpml});
    // Taper narrower than the spacing.
    throws(h, {kL, 0.5 * kH, kWg, kWpml});
    CHECK_NOTHROW(make_open_surface_mesh(analytic_map(43, kH, [](Real, Real) { return 0.0; }),
                                         {kL, kH, 0.0, kH}));
    // Map too small for L + 2 W_t (one cell short on each side, and rounding past its edge).
    throws(h, {kL, kWt + kH, kWg, kWpml});
    throws(h, {kL + 0.5 * kH, kWt, kWg, kWpml});
    throws(analytic_map(59, kH, [](Real, Real) { return 0.0; }), default_params());
    // Non-finite values.
    throws(h, {nan, kWt, kWg, kWpml});
    throws(h, {kL, inf, kWg, kWpml});
    throws(h, {kL, kWt, nan, kWpml});
    throws(h, {kL, kWt, kWg, inf});
    HeightMap bad = h;
    bad.z(30, 30) = nan;  // centre of the patch
    throws(bad, default_params());
    bad = h;
    bad.z(1, 30) = inf;  // inside the taper
    throws(bad, default_params());
    bad = h;
    bad.z(0, 30) = nan;  // at the taper end: ignored
    CHECK_NOTHROW(make_open_surface_mesh(bad, default_params()));
    bad = h;
    bad.dx = nan;
    throws(bad, default_params());
    // Anisotropic spacing and mismatched grid parity.
    bad = h;
    bad.dy = 1.1 * h.dx;
    throws(bad, default_params());
    HeightMap rect;
    rect.dx = kH;
    rect.dy = kH;
    rect.z = MatrixXr::Zero(61, 62);
    throws(rect, default_params());
    rect.z = MatrixXr::Zero(61, 63);
    CHECK_NOTHROW(make_open_surface_mesh(rect, default_params()));
    rect.z = MatrixXr::Zero(1, 1);
    throws(rect, default_params());
}
