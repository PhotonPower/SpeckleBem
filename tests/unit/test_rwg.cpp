#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/sphere.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace specklebem;
using basis::RwgSpace;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using geometry::make_icosphere;
using geometry::TriangleMesh;

namespace {

const Vec3 kCenter(0.3e-6, -1.2e-6, 2.5e-6);
constexpr Real kRadius = 1.7e-6;
constexpr Real kTol = 1e-12;

Vec3 vertex(const TriangleMesh& mesh, Index v) {
    return mesh.vertices().row(v).transpose();
}

/// Icosphere with its last triangle removed: three boundary edges.
TriangleMesh open_icosphere(int subdivisions) {
    const TriangleMesh sphere = make_icosphere(kRadius, subdivisions, kCenter);
    const Triangles reduced = sphere.triangles().topRows(sphere.num_triangles() - 1);
    return TriangleMesh(sphere.vertices(), reduced);
}

/// Unit vector in the plane of the triangle (a, b, q), perpendicular to edge a-b and
/// pointing away from the opposite vertex q (outward edge normal).
Vec3 outward_edge_normal(const Vec3& a, const Vec3& b, const Vec3& q) {
    const Vec3 d = (b - a).normalized();
    const Vec3 w = a - q;
    return (w - w.dot(d) * d).normalized();
}

bool contains(const RwgSpace::Support& s, Index n, int sign) {
    for (int k = 0; k < s.count; ++k) {
        if (s.n[k] == n && s.sign[k] == sign)
            return true;
    }
    return false;
}

/// Structural checks shared by the closed and open meshes.
void check_supports(const RwgSpace& space) {
    const TriangleMesh& mesh = space.mesh();
    const Index num_basis = space.size();
    std::vector<int> plus_hits(static_cast<std::size_t>(num_basis), 0);
    std::vector<int> minus_hits(static_cast<std::size_t>(num_basis), 0);
    Index total = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const RwgSpace::Support s = space.support(t);
        REQUIRE(s.count >= 0);
        REQUIRE(s.count <= 3);
        total += s.count;
        for (int k = 0; k < s.count; ++k) {
            const Index n = s.n[k];
            REQUIRE(n >= 0);
            REQUIRE(n < num_basis);
            if (k > 0)
                CHECK(s.n[k - 1] < n);  // ascending
            if (s.sign[k] == 1) {
                CHECK(mesh.edge_triangles()(n, 0) == t);
                ++plus_hits[static_cast<std::size_t>(n)];
            } else {
                REQUIRE(s.sign[k] == -1);
                CHECK(mesh.edge_triangles()(n, 1) == t);
                ++minus_hits[static_cast<std::size_t>(n)];
            }
        }
        for (int k = s.count; k < 3; ++k) {
            CHECK(s.n[k] == -1);
            CHECK(s.sign[k] == 0);
        }
    }
    CHECK(total == 2 * num_basis);
    for (Index n = 0; n < num_basis; ++n) {
        CHECK(plus_hits[static_cast<std::size_t>(n)] == 1);
        CHECK(minus_hits[static_cast<std::size_t>(n)] == 1);
        CHECK(contains(space.support(space.plus_triangle(n)), n, 1));
        CHECK(contains(space.support(space.minus_triangle(n)), n, -1));
    }
}

}  // namespace

TEST_CASE("RWG size equals interior edge count and supports are consistent", "[basis]") {
    const int sub = GENERATE(1, 2, 3);
    const TriangleMesh mesh = make_icosphere(kRadius, sub, kCenter);
    const RwgSpace space(mesh);
    CHECK(&space.mesh() == &mesh);
    CHECK(space.size() == mesh.num_edges());
    CHECK(space.size() == 30 * (Index{1} << (2 * sub)));
    check_supports(space);
    // On a closed mesh every triangle carries exactly three functions.
    for (Index t = 0; t < mesh.num_triangles(); ++t) CHECK(space.support(t).count == 3);
}

TEST_CASE("RWG free vertices are the vertices opposite the edge", "[basis]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 2, kCenter);
    const RwgSpace space(mesh);
    for (Index n = 0; n < space.size(); ++n) {
        const Index a = mesh.edges()(n, 0);
        const Index b = mesh.edges()(n, 1);
        const std::array<std::pair<Index, Index>, 2> sides = {
            std::pair{space.plus_triangle(n), space.plus_free_vertex(n)},
            std::pair{space.minus_triangle(n), space.minus_free_vertex(n)}};
        for (const auto& [t, p] : sides) {
            CHECK(p != a);
            CHECK(p != b);
            const auto row = mesh.triangles().row(t);
            CHECK((row(0) == p || row(1) == p || row(2) == p));
        }
        CHECK(space.plus_triangle(n) == mesh.edge_triangles()(n, 0));
        CHECK(space.minus_triangle(n) == mesh.edge_triangles()(n, 1));
    }
}

TEST_CASE("RWG on an open mesh: boundary edges carry no function", "[basis]") {
    const TriangleMesh mesh = open_icosphere(2);
    REQUIRE(mesh.num_boundary_edges() == 3);
    const RwgSpace space(mesh);
    CHECK(space.size() == mesh.num_edges());
    check_supports(space);

    // No support entry refers to a boundary edge (compare vertex pairs).
    int reduced = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const RwgSpace::Support s = space.support(t);
        if (s.count < 3)
            ++reduced;
        CHECK(s.count >= 2);
        for (int k = 0; k < s.count; ++k) {
            for (Index e = 0; e < mesh.num_boundary_edges(); ++e) {
                const bool same = mesh.edges()(s.n[k], 0) == mesh.boundary_edges()(e, 0) &&
                                  mesh.edges()(s.n[k], 1) == mesh.boundary_edges()(e, 1);
                CHECK_FALSE(same);
            }
        }
    }
    CHECK(reduced == 3);  // the three neighbours of the removed triangle
}

TEST_CASE("RWG on meshes without interior edges is empty", "[basis]") {
    Vertices v(3, 3);
    v << 0, 0, 0, 1, 0, 0, 0, 1, 0;
    Triangles f(1, 3);
    f << 0, 1, 2;
    const TriangleMesh single(v, f);
    const RwgSpace space(single);
    CHECK(space.size() == 0);
    const RwgSpace::Support s = space.support(0);
    CHECK(s.count == 0);
    for (int k = 0; k < 3; ++k) {
        CHECK(s.n[k] == -1);
        CHECK(s.sign[k] == 0);
    }
    CHECK_THROWS_AS(space.value(0, 0, Vec3::Zero()), std::out_of_range);

    const TriangleMesh empty;
    const RwgSpace empty_space(empty);
    CHECK(empty_space.size() == 0);
    CHECK_THROWS_AS(empty_space.support(0), std::out_of_range);
}

TEST_CASE("RWG on two triangles matches hand-computed values", "[basis]") {
    // Unit square split along the diagonal 0-2. T0 = (0, 1, 2) traverses 2 -> 0 (minus),
    // T1 = (0, 2, 3) traverses 0 -> 2 (plus). p+ = v3 = (0, 1, 0), p- = v1 = (1, 0, 0).
    Vertices v(4, 3);
    v << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
    Triangles f(2, 3);
    f << 0, 1, 2, 0, 2, 3;
    const TriangleMesh mesh(v, f);
    const RwgSpace space(mesh);
    REQUIRE(space.size() == 1);
    CHECK(space.plus_triangle(0) == 1);
    CHECK(space.minus_triangle(0) == 0);
    CHECK(space.plus_free_vertex(0) == 3);
    CHECK(space.minus_free_vertex(0) == 1);

    const Real s2 = std::sqrt(2.0);
    CHECK_THAT(space.divergence(0, 1), WithinAbs(2.0 * s2, kTol));
    CHECK_THAT(space.divergence(0, 0), WithinAbs(-2.0 * s2, kTol));

    // l / (2A) = sqrt(2).
    const Vec3 c_plus(1.0 / 3.0, 2.0 / 3.0, 0.0);
    const Vec3 c_minus(2.0 / 3.0, 1.0 / 3.0, 0.0);
    const Vec3 f_plus = space.value(0, 1, c_plus);
    const Vec3 f_minus = space.value(0, 0, c_minus);
    CHECK((f_plus - s2 * Vec3(1.0 / 3.0, -1.0 / 3.0, 0.0)).norm() < kTol);
    CHECK((f_minus - s2 * Vec3(1.0 / 3.0, -1.0 / 3.0, 0.0)).norm() < kTol);
    // Edge midpoint: normal component (1, -1)/sqrt(2) is 1 from both sides.
    const Vec3 mid(0.5, 0.5, 0.0);
    const Vec3 nu = Vec3(1.0, -1.0, 0.0) / s2;
    CHECK_THAT(space.value(0, 1, mid).dot(nu), WithinAbs(1.0, kTol));
    CHECK_THAT(space.value(0, 0, mid).dot(nu), WithinAbs(1.0, kTol));

    const RwgSpace::Support s0 = space.support(0);
    const RwgSpace::Support s1 = space.support(1);
    CHECK(s0.count == 1);
    CHECK(s0.n[0] == 0);
    CHECK(s0.sign[0] == -1);
    CHECK(s1.count == 1);
    CHECK(s1.n[0] == 0);
    CHECK(s1.sign[0] == 1);
}

TEST_CASE("RWG divergence is +-l/A on T+-, zero elsewhere; integrates to +-l", "[basis]") {
    const int sub = GENERATE(1, 2, 3);
    const TriangleMesh mesh = make_icosphere(kRadius, sub, kCenter);
    const RwgSpace space(mesh);
    for (Index n = 0; n < space.size(); ++n) {
        const Real l = mesh.edge_length(n);
        const Index tp = mesh.edge_triangles()(n, 0);
        const Index tm = mesh.edge_triangles()(n, 1);
        // Consistency checks of the cached divergence against the mesh geometry (the
        // implementation uses the same formula); the independent verification is the
        // flux and normal-component tests below.
        CHECK_THAT(space.divergence(n, tp), WithinRel(l / mesh.area(tp), kTol));
        CHECK_THAT(space.divergence(n, tm), WithinRel(-l / mesh.area(tm), kTol));
        // Closed-form integral of the divergence over each triangle (consistency check).
        CHECK_THAT(space.divergence(n, tp) * mesh.area(tp), WithinRel(l, kTol));
        CHECK_THAT(space.divergence(n, tm) * mesh.area(tm), WithinRel(-l, kTol));
    }
    // Zero (and value zero) on every triangle outside the support: sample a few n per t.
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const RwgSpace::Support s = space.support(t);
        for (Index n = 0; n < space.size(); n += 97) {
            if (contains(s, n, 1) || contains(s, n, -1))
                continue;
            CHECK(space.divergence(n, t) == 0.0);
            CHECK(space.value(n, t, mesh.centroid(t)) == Vec3::Zero());
        }
    }
}

TEST_CASE("RWG normal component: 1 on the shared edge, 0 on the other edges", "[basis]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 1, kCenter);
    const RwgSpace space(mesh);
    const std::array<Real, 3> params = {0.25, 0.5, 0.75};
    for (Index n = 0; n < space.size(); ++n) {
        const Vec3 a = vertex(mesh, mesh.edges()(n, 0));
        const Vec3 b = vertex(mesh, mesh.edges()(n, 1));
        const Index tp = space.plus_triangle(n);
        const Index tm = space.minus_triangle(n);
        const Vec3 pp = vertex(mesh, space.plus_free_vertex(n));
        const Vec3 pm = vertex(mesh, space.minus_free_vertex(n));
        // Outward edge normals of T+ and T- at the shared edge. The current crosses from
        // T+ into T-: f . nu+ = +1 on T+, f . nu- = -1 on T- (i.e. +1 along -nu-, the
        // normal pointing from T+ into T-), so the normal component is continuous.
        const Vec3 nu_p = outward_edge_normal(a, b, pp);
        const Vec3 nu_m = outward_edge_normal(a, b, pm);
        CHECK(std::abs(nu_p.dot(mesh.normal(tp))) < kTol);
        CHECK(std::abs(nu_m.dot(mesh.normal(tm))) < kTol);
        for (const Real s : params) {
            const Vec3 r = a + s * (b - a);
            CHECK_THAT(space.value(n, tp, r).dot(nu_p), WithinAbs(1.0, kTol));
            CHECK_THAT(space.value(n, tm, r).dot(-nu_m), WithinAbs(1.0, kTol));
        }
        // Other two edges of each triangle: (p, a) and (p, b), opposite vertex b resp. a.
        const std::array<std::pair<Index, Vec3>, 2> sides = {std::pair{tp, pp}, std::pair{tm, pm}};
        for (const auto& [t, p] : sides) {
            const std::array<std::array<Vec3, 3>, 2> other = {std::array<Vec3, 3>{p, a, b},
                                                              std::array<Vec3, 3>{p, b, a}};
            for (const auto& e : other) {
                const Vec3 nu = outward_edge_normal(e[0], e[1], e[2]);
                for (const Real s : params) {
                    const Vec3 r = e[0] + s * (e[1] - e[0]);
                    CHECK_THAT(space.value(n, t, r).dot(nu), WithinAbs(0.0, kTol));
                }
            }
        }
    }
}

TEST_CASE("RWG flux across the shared edge equals the edge length", "[basis]") {
    const int sub = GENERATE(1, 2, 3);
    const TriangleMesh mesh = make_icosphere(kRadius, sub, kCenter);
    const RwgSpace space(mesh);
    // 2-point Gauss-Legendre on [0, 1]: exact for the linear integrand f . nu.
    const Real g = 0.5 / std::sqrt(3.0);
    const std::array<Real, 2> nodes = {0.5 - g, 0.5 + g};
    for (Index n = 0; n < space.size(); ++n) {
        const Vec3 a = vertex(mesh, mesh.edges()(n, 0));
        const Vec3 b = vertex(mesh, mesh.edges()(n, 1));
        const Real l = mesh.edge_length(n);
        const Index tp = space.plus_triangle(n);
        const Index tm = space.minus_triangle(n);
        const Vec3 nu_p = outward_edge_normal(a, b, vertex(mesh, space.plus_free_vertex(n)));
        const Vec3 nu_m = outward_edge_normal(a, b, vertex(mesh, space.minus_free_vertex(n)));
        Real flux_out_p = 0.0;
        Real flux_out_m = 0.0;
        for (const Real s : nodes) {
            const Vec3 r = a + s * (b - a);
            flux_out_p += 0.5 * l * space.value(n, tp, r).dot(nu_p);
            flux_out_m += 0.5 * l * space.value(n, tm, r).dot(nu_m);
        }
        // Divergence theorem: outward flux of T+- equals the integral of div f over T+-.
        CHECK_THAT(flux_out_p, WithinRel(l, kTol));
        CHECK_THAT(flux_out_m, WithinRel(-l, kTol));
        CHECK_THAT(flux_out_p, WithinRel(space.divergence(n, tp) * mesh.area(tp), kTol));
        CHECK_THAT(flux_out_m, WithinRel(space.divergence(n, tm) * mesh.area(tm), kTol));
    }
}

TEST_CASE("RWG value is affine on each triangle", "[basis]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 2, kCenter);
    const RwgSpace space(mesh);
    for (Index n = 0; n < space.size(); ++n) {
        for (const Index t : {space.plus_triangle(n), space.minus_triangle(n)}) {
            const Vec3 v0 = vertex(mesh, mesh.triangles()(t, 0));
            const Vec3 v1 = vertex(mesh, mesh.triangles()(t, 1));
            const Vec3 v2 = vertex(mesh, mesh.triangles()(t, 2));
            const Vec3 r1 = 0.6 * v0 + 0.3 * v1 + 0.1 * v2;
            const Vec3 r2 = 0.1 * v0 + 0.2 * v1 + 0.7 * v2;
            const Vec3 mid = 0.5 * (r1 + r2);
            const Vec3 expected = 0.5 * (space.value(n, t, r1) + space.value(n, t, r2));
            CHECK((space.value(n, t, mid) - expected).norm() < kTol);
        }
    }
}

TEST_CASE("RWG invalid indices throw std::out_of_range", "[basis]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 1, kCenter);
    const RwgSpace space(mesh);
    const Index nb = space.size();
    const Index nt = mesh.num_triangles();
    const Vec3 r = mesh.centroid(0);
    for (const Index bad : {Index{-1}, nb}) {
        CHECK_THROWS_AS(space.value(bad, 0, r), std::out_of_range);
        CHECK_THROWS_AS(space.divergence(bad, 0), std::out_of_range);
        CHECK_THROWS_AS(space.plus_triangle(bad), std::out_of_range);
        CHECK_THROWS_AS(space.minus_triangle(bad), std::out_of_range);
        CHECK_THROWS_AS(space.plus_free_vertex(bad), std::out_of_range);
        CHECK_THROWS_AS(space.minus_free_vertex(bad), std::out_of_range);
    }
    for (const Index bad : {Index{-1}, nt}) {
        CHECK_THROWS_AS(space.value(0, bad, r), std::out_of_range);
        CHECK_THROWS_AS(space.divergence(0, bad), std::out_of_range);
        CHECK_THROWS_AS(space.support(bad), std::out_of_range);
    }
    CHECK_NOTHROW(space.value(nb - 1, nt - 1, r));
    CHECK_NOTHROW(space.support(nt - 1));
}

// The space stores a mesh reference: construction from temporaries is rejected.
static_assert(std::is_constructible_v<RwgSpace, const TriangleMesh&>);
static_assert(std::is_constructible_v<RwgSpace, TriangleMesh&>);
static_assert(!std::is_constructible_v<RwgSpace, TriangleMesh&&>);
static_assert(!std::is_constructible_v<RwgSpace, const TriangleMesh&&>);

TEST_CASE("RWG after flip_normals: plus / minus swap, divergence and value invert", "[basis]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 2, kCenter);
    TriangleMesh flipped = mesh;
    flipped.flip_normals();
    const RwgSpace space(mesh);
    const RwgSpace space_f(flipped);
    REQUIRE(space_f.size() == space.size());
    REQUIRE(flipped.edges() == mesh.edges());  // same edge (basis) numbering
    for (Index n = 0; n < space.size(); ++n) {
        const Index tp = space.plus_triangle(n);
        const Index tm = space.minus_triangle(n);
        CHECK(space_f.plus_triangle(n) == tm);
        CHECK(space_f.minus_triangle(n) == tp);
        CHECK(space_f.plus_free_vertex(n) == space.minus_free_vertex(n));
        CHECK(space_f.minus_free_vertex(n) == space.plus_free_vertex(n));
        for (const Index t : {tp, tm}) {
            CHECK_THAT(space_f.divergence(n, t), WithinRel(-space.divergence(n, t), kTol));
            const Vec3 r = mesh.centroid(t);
            CHECK((space_f.value(n, t, r) + space.value(n, t, r)).norm() < kTol);
        }
    }
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const RwgSpace::Support s = space.support(t);
        const RwgSpace::Support s_f = space_f.support(t);
        REQUIRE(s_f.count == s.count);
        for (int k = 0; k < s.count; ++k) {
            CHECK(s_f.n[k] == s.n[k]);
            CHECK(s_f.sign[k] == -s.sign[k]);
        }
    }
}

TEST_CASE("RWG on an icosphere with 81920 triangles builds quickly", "[basis]") {
    const TriangleMesh mesh = make_icosphere(1e-6, 6);
    const auto start = std::chrono::steady_clock::now();
    const RwgSpace space(mesh);
    const auto built = std::chrono::steady_clock::now();
    Index total = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) total += space.support(t).count;
    const auto swept = std::chrono::steady_clock::now();
    const Real build_s = std::chrono::duration<Real>(built - start).count();
    const Real sweep_s = std::chrono::duration<Real>(swept - built).count();
    CHECK(space.size() == 122880);
    CHECK(total == 2 * space.size());
    // Typically a few ms; generous bounds so the sanitizer build cannot flake. A search
    // over edges per lookup (O(F N) sweep) would take far longer than 2 s.
    CHECK(build_s < 10.0);
    CHECK(sweep_s < 2.0);
}
