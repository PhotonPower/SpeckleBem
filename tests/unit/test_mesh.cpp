#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/sphere.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ag_sphere_mlfmm_support.hpp"

using namespace specklebem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using geometry::make_icosphere;
using geometry::make_sphere;
using geometry::MeshQuality;
using geometry::TriangleMesh;

namespace {

const Vec3 kCenter(0.3e-6, -1.2e-6, 2.5e-6);
constexpr Real kRadius = 1.7e-6;

/// True if row t of `tris` contains the directed edge a -> b.
bool has_directed_edge(const Triangles& tris, Index t, Index a, Index b) {
    for (int k = 0; k < 3; ++k) {
        if (tris(t, k) == a && tris(t, (k + 1) % 3) == b)
            return true;
    }
    return false;
}

bool plus_minus_convention_holds(const TriangleMesh& mesh) {
    for (Index e = 0; e < mesh.num_edges(); ++e) {
        const Index a = mesh.edges()(e, 0);
        const Index b = mesh.edges()(e, 1);
        if (!(a < b))
            return false;
        if (!has_directed_edge(mesh.triangles(), mesh.edge_triangles()(e, 0), a, b))
            return false;
        if (!has_directed_edge(mesh.triangles(), mesh.edge_triangles()(e, 1), b, a))
            return false;
    }
    return true;
}

bool all_normals_outward(const TriangleMesh& mesh, const Vec3& center) {
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        if (!(mesh.normal(t).dot(mesh.centroid(t) - center) > 0.0))
            return false;
    }
    return true;
}

Real mean_edge_length(const TriangleMesh& mesh) {
    Real sum = 0.0;
    for (Index e = 0; e < mesh.num_edges(); ++e) sum += mesh.edge_length(e);
    return sum / static_cast<Real>(mesh.num_edges());
}

Real total_area(const TriangleMesh& mesh) {
    Real sum = 0.0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) sum += mesh.area(t);
    return sum;
}

/// Value of an integer line "label : value" in the quality report, or -1 if absent.
Index report_value(const std::string& report, const std::string& label) {
    const std::regex re(label + R"(\s*:\s*(-?\d+))");
    std::smatch m;
    if (!std::regex_search(report, m, re))
        return -1;
    return static_cast<Index>(std::stoll(m[1].str()));
}

/// Real numbers on the report line "label : x / y / z ...", in order.
std::vector<Real> report_reals(const std::string& report, const std::string& label) {
    const std::regex line_re(label + R"(\s*:([^\n]*))");
    std::smatch m;
    std::vector<Real> out;
    if (!std::regex_search(report, m, line_re))
        return out;
    const std::string line = m[1].str();
    const std::regex num_re(R"([-+]?\d+(\.\d*)?([eE][-+]?\d+)?)");
    for (auto it = std::sregex_iterator(line.begin(), line.end(), num_re);
         it != std::sregex_iterator(); ++it) {
        out.push_back(std::stod(it->str()));
    }
    return out;
}

/// Disjoint union of two meshes' connectivity (b's indices shifted by a's vertex count).
std::pair<Vertices, Triangles> concatenate(const TriangleMesh& a, const TriangleMesh& b) {
    Vertices v(a.num_vertices() + b.num_vertices(), 3);
    v << a.vertices(), b.vertices();
    Triangles t(a.num_triangles() + b.num_triangles(), 3);
    t << a.triangles(), (b.triangles().array() + a.num_vertices()).matrix();
    return {v, t};
}

Triangles single_row(Index a, Index b, Index c) {
    Triangles t(1, 3);
    t << a, b, c;
    return t;
}

}  // namespace

TEST_CASE("icosphere topology: counts, Euler characteristic, two triangles per edge",
          "[geometry]") {
    const int n = GENERATE(0, 1, 2, 3, 4);
    const TriangleMesh mesh = make_icosphere(kRadius, n, kCenter);
    const Index pow4 = Index{1} << (2 * n);
    CHECK(mesh.num_triangles() == 20 * pow4);
    CHECK(mesh.num_vertices() == 10 * pow4 + 2);
    CHECK(mesh.num_edges() == 30 * pow4);
    CHECK(mesh.num_vertices() - mesh.num_edges() + mesh.num_triangles() == 2);
    CHECK(mesh.num_boundary_edges() == 0);
    CHECK(mesh.is_closed());
    CHECK(mesh.is_consistently_oriented());

    // Every interior edge has two distinct triangles that both contain it, and every
    // triangle of a closed mesh is adjacent to exactly three interior edges.
    std::vector<int> uses(static_cast<std::size_t>(mesh.num_triangles()), 0);
    bool ok = true;
    for (Index e = 0; e < mesh.num_edges(); ++e) {
        const Index tp = mesh.edge_triangles()(e, 0);
        const Index tm = mesh.edge_triangles()(e, 1);
        ok = ok && tp != tm && tp >= 0 && tm >= 0 && tp < mesh.num_triangles() &&
             tm < mesh.num_triangles();
        if (!ok)
            break;
        ++uses[static_cast<std::size_t>(tp)];
        ++uses[static_cast<std::size_t>(tm)];
    }
    CHECK(ok);
    for (const int u : uses) ok = ok && u == 3;
    CHECK(ok);
    CHECK(plus_minus_convention_holds(mesh));
}

TEST_CASE("octahedron-based sphere of the WP22a study: counts, closed, outward, on the sphere",
          "[geometry]") {
    // tests/support/ag_sphere_mlfmm_support.hpp: the paper's mesh of the 4 um Ag sphere is
    // make_octasphere(2 um, 7) with 131 072 triangles and 2N = 393 216 unknowns.
    const int n = GENERATE(0, 1, 3, 5);
    const Real radius = 2e-6;
    const TriangleMesh mesh = ag_sphere_mlfmm::make_octasphere(radius, n);
    const Index pow4 = Index{1} << (2 * n);
    CHECK(mesh.num_triangles() == 8 * pow4);
    CHECK(mesh.num_vertices() == 4 * pow4 + 2);
    CHECK(mesh.num_edges() == 12 * pow4);
    CHECK(mesh.euler_characteristic() == 2);
    CHECK(mesh.is_closed());
    CHECK(mesh.is_consistently_oriented());
    CHECK(plus_minus_convention_holds(mesh));
    Real max_dev = 0.0;
    for (Index v = 0; v < mesh.num_vertices(); ++v)
        max_dev = std::max(max_dev, std::abs(mesh.vertices().row(v).norm() - radius));
    CHECK(max_dev < 1e-12 * radius);
    // Outward normals (docs/06); the volume approaches 4/3 pi r^3 from below.
    const Real ball = 4.0 / 3.0 * constants::pi * radius * radius * radius;
    CHECK(mesh.signed_volume() > 0.0);
    CHECK(mesh.signed_volume() < ball);
    if (n == 5) {
        CHECK(mesh.signed_volume() > 0.99 * ball);
        // Edge spread of the octahedral subdivision (WP22a record: mean / min / max ~ 30.25 /
        // 24.54 / 38.27 nm at r = 2 um, n = 7, i.e. 0.81 / 1.27 x the mean).
        const ag_sphere_mlfmm::EdgeStats e = ag_sphere_mlfmm::edge_stats(mesh);
        CHECK(e.min > 0.75 * e.mean);
        CHECK(e.max < 1.35 * e.mean);
    }
    CHECK_THROWS_AS(ag_sphere_mlfmm::make_octasphere(radius, -1), std::invalid_argument);
    CHECK_THROWS_AS(ag_sphere_mlfmm::make_octasphere(0.0, 1), std::invalid_argument);
}

TEST_CASE("closedness: icosphere closed, open after removing a triangle", "[geometry]") {
    const TriangleMesh sphere = make_icosphere(kRadius, 2, kCenter);
    CHECK(sphere.is_closed());
    const std::string closed_report = sphere.quality_report();
    CHECK(report_value(closed_report, "boundary edges") == 0);
    CHECK(report_value(closed_report, "Euler characteristic") == 2);
    CHECK(closed_report.find("closed                  : yes") != std::string::npos);

    const Index nf = sphere.num_triangles();
    Triangles reduced = sphere.triangles().topRows(nf - 1);
    const TriangleMesh open(sphere.vertices(), reduced);
    CHECK_FALSE(open.is_closed());
    CHECK(open.num_boundary_edges() == 3);
    CHECK(open.num_edges() == sphere.num_edges() - 3);
    CHECK(open.is_consistently_oriented());
    CHECK(plus_minus_convention_holds(open));
    const std::string report = open.quality_report();
    CHECK(report_value(report, "boundary edges") == 3);
    CHECK(report_value(report, "non-manifold edges") == 0);
    CHECK(report_value(report, "interior edges \\(E\\)") == sphere.num_edges() - 3);
    // Disc topology: V - E_all + F = 1.
    CHECK(report_value(report, "Euler characteristic") == 1);
    CHECK(report.find("closed                  : no") != std::string::npos);

    CHECK_FALSE(TriangleMesh().is_closed());
    CHECK_FALSE(TriangleMesh(Vertices(0, 3), Triangles(0, 3)).is_closed());
}

TEST_CASE("icosphere normals point outward; area and volume converge", "[geometry]") {
    const Real exact_area = 4.0 * constants::pi * kRadius * kRadius;
    const Real exact_volume = 4.0 / 3.0 * constants::pi * kRadius * kRadius * kRadius;
    Real prev_area_err = 1e300;
    Real prev_vol_err = 1e300;
    for (int n = 0; n <= 4; ++n) {
        const TriangleMesh mesh = make_icosphere(kRadius, n, kCenter);
        CHECK(all_normals_outward(mesh, kCenter));
        for (Index t = 0; t < mesh.num_triangles(); t += 7) {
            CHECK_THAT(mesh.normal(t).norm(), WithinAbs(1.0, 1e-14));
        }
        const Real vol = mesh.signed_volume();
        CHECK(vol > 0.0);
        // Inscribed polyhedron: both quantities approach the sphere from below.
        const Real area_err = (exact_area - total_area(mesh)) / exact_area;
        const Real vol_err = (exact_volume - vol) / exact_volume;
        CHECK(area_err > 0.0);
        CHECK(vol_err > 0.0);
        CHECK(area_err < prev_area_err);
        CHECK(vol_err < prev_vol_err);
        prev_area_err = area_err;
        prev_vol_err = vol_err;
    }
    // n = 4: edge ~ 0.066 r, relative errors O(h^2) ~ 1e-3.
    CHECK(prev_area_err < 3e-3);
    CHECK(prev_vol_err < 5e-3);
}

TEST_CASE("orientation repair restores a consistent, outward orientation", "[geometry]") {
    const TriangleMesh ref = make_icosphere(kRadius, 2, kCenter);
    const Index nf = ref.num_triangles();

    auto flipped_copy = [&](const std::vector<Index>& which) {
        Triangles tris = ref.triangles();
        for (const Index t : which) std::swap(tris(t, 1), tris(t, 2));
        return tris;
    };

    std::vector<Index> subset;
    SECTION("one triangle") {
        subset = {17};
    }
    SECTION("random subset (fixed seed)") {
        std::mt19937 rng(20261008u);
        std::bernoulli_distribution coin(0.3);
        for (Index t = 0; t < nf; ++t) {
            if (coin(rng))
                subset.push_back(t);
        }
        REQUIRE(subset.size() > 10);
    }
    SECTION("majority of the triangles (random, fixed seed)") {
        std::mt19937 rng(7u);
        std::bernoulli_distribution coin(0.8);
        for (Index t = 0; t < nf; ++t) {
            if (coin(rng))
                subset.push_back(t);
        }
    }
    SECTION("all triangles (inward normals)") {
        for (Index t = 0; t < nf; ++t) subset.push_back(t);
    }

    const TriangleMesh mesh(ref.vertices(), flipped_copy(subset));
    CHECK(mesh.is_consistently_oriented());
    CHECK(mesh.is_closed());
    CHECK(all_normals_outward(mesh, kCenter));
    CHECK(mesh.signed_volume() > 0.0);
    CHECK_THAT(mesh.signed_volume(), WithinRel(ref.signed_volume(), 1e-12));
    // Identical geometry: same vertices, and every triangle back in its original order.
    CHECK(mesh.vertices() == ref.vertices());
    CHECK(mesh.triangles() == ref.triangles());
    CHECK(mesh.edges() == ref.edges());
    CHECK(mesh.edge_triangles() == ref.edge_triangles());
    CHECK(plus_minus_convention_holds(mesh));
}

TEST_CASE("orientation repair on an open mesh", "[geometry]") {
    // Unit square split into two triangles with opposite orientation.
    Vertices v(4, 3);
    v << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
    Triangles t(2, 3);
    t << 0, 1, 2,  //
        0, 3, 2;   // traverses 0 -> 2 like the first triangle: inconsistent
    const TriangleMesh mesh(v, t);
    CHECK(mesh.is_consistently_oriented());
    CHECK_FALSE(mesh.is_closed());
    CHECK(mesh.num_edges() == 1);
    CHECK(mesh.num_boundary_edges() == 4);
    CHECK(plus_minus_convention_holds(mesh));
    CHECK_THAT(mesh.normal(0).dot(mesh.normal(1)), WithinAbs(1.0, 1e-14));
}

TEST_CASE("flip_normals negates normals and volume; twice restores plus/minus", "[geometry]") {
    const TriangleMesh ref = make_icosphere(kRadius, 2, kCenter);
    TriangleMesh mesh = ref;
    mesh.flip_normals();
    bool negated = true;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        negated = negated && (mesh.normal(t) + ref.normal(t)).norm() < 1e-14;
        negated = negated && mesh.area(t) == ref.area(t);
    }
    CHECK(negated);
    CHECK_THAT(mesh.signed_volume(), WithinRel(-ref.signed_volume(), 1e-12));
    CHECK(mesh.edges() == ref.edges());
    CHECK(mesh.edge_triangles().col(0) == ref.edge_triangles().col(1));
    CHECK(mesh.edge_triangles().col(1) == ref.edge_triangles().col(0));
    CHECK(plus_minus_convention_holds(mesh));

    mesh.flip_normals();
    CHECK(mesh.triangles() == ref.triangles());
    CHECK(mesh.edge_triangles() == ref.edge_triangles());
    CHECK(all_normals_outward(mesh, kCenter));
}

TEST_CASE("plus/minus convention: plus triangle traverses a -> b", "[geometry]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 3, kCenter);
    CHECK(plus_minus_convention_holds(mesh));
    // Edges are sorted lexicographically by (a, b).
    bool sorted = true;
    for (Index e = 1; e < mesh.num_edges(); ++e) {
        const auto& E = mesh.edges();
        sorted =
            sorted && (E(e - 1, 0) < E(e, 0) || (E(e - 1, 0) == E(e, 0) && E(e - 1, 1) < E(e, 1)));
    }
    CHECK(sorted);
}

TEST_CASE("bounding box of an icosphere is [c - r, c + r]", "[geometry]") {
    // For n >= 1 the midpoints of three icosahedron edges project exactly onto the
    // coordinate axes, so the box is exact up to rounding.
    const TriangleMesh mesh = make_icosphere(kRadius, 4, kCenter);
    const auto [lo, hi] = mesh.bounding_box();
    for (int d = 0; d < 3; ++d) {
        CHECK_THAT(lo(d), WithinAbs(kCenter(d) - kRadius, 1e-12 * kRadius));
        CHECK_THAT(hi(d), WithinAbs(kCenter(d) + kRadius, 1e-12 * kRadius));
    }
    CHECK_THROWS_AS(TriangleMesh().bounding_box(), std::logic_error);
}

TEST_CASE("quality report of a single equilateral triangle", "[geometry]") {
    Vertices v(3, 3);
    v << 0, 0, 0, 1, 0, 0, 0.5, std::sqrt(3.0) / 2.0, 0;
    const TriangleMesh mesh(v, single_row(0, 1, 2));
    const std::string report = mesh.quality_report();
    CHECK(report_value(report, "vertices \\(V\\)") == 3);
    CHECK(report_value(report, "triangles \\(F\\)") == 1);
    CHECK(report_value(report, "boundary edges") == 3);
    CHECK(report_value(report, "Euler characteristic") == 1);
    CHECK(report.find("max aspect ratio        : 1 ") != std::string::npos);
    CHECK_THAT(mesh.area(0), WithinRel(std::sqrt(3.0) / 4.0, 1e-14));
    CHECK_THAT(mesh.normal(0).z(), WithinAbs(1.0, 1e-15));
    CHECK_FALSE(mesh.is_closed());
}

TEST_CASE("quality report edge statistics and aspect ratio", "[geometry]") {
    SECTION("right isosceles triangle with unit legs") {
        Vertices v(3, 3);
        v << 0, 0, 0, 1, 0, 0, 0, 1, 0;
        const std::string report = TriangleMesh(v, single_row(0, 1, 2)).quality_report();
        const std::vector<Real> len = report_reals(report, "edge length min/max/mean");
        REQUIRE(len.size() == 3);
        CHECK_THAT(len[0], WithinRel(1.0, 1e-5));
        CHECK_THAT(len[1], WithinRel(std::sqrt(2.0), 1e-5));
        CHECK_THAT(len[2], WithinRel((2.0 + std::sqrt(2.0)) / 3.0, 1e-5));
        // R = sqrt(2)/2, r = (2 - sqrt(2))/2  =>  R / (2r) = (1 + sqrt(2)) / 2.
        const std::vector<Real> aspect = report_reals(report, "max aspect ratio");
        REQUIRE(!aspect.empty());
        CHECK_THAT(aspect[0], WithinRel((1.0 + std::sqrt(2.0)) / 2.0, 1e-5));
    }
    SECTION("icosphere n = 2 against values computed from the vertices") {
        const TriangleMesh mesh = make_icosphere(kRadius, 2, kCenter);
        Real lmin = 1e300, lmax = 0.0, lsum = 0.0;
        for (Index e = 0; e < mesh.num_edges(); ++e) {
            const Vec3 d =
                mesh.vertices().row(mesh.edges()(e, 1)) - mesh.vertices().row(mesh.edges()(e, 0));
            lmin = std::min(lmin, d.norm());
            lmax = std::max(lmax, d.norm());
            lsum += d.norm();
        }
        const std::vector<Real> len =
            report_reals(mesh.quality_report(), "edge length min/max/mean");
        REQUIRE(len.size() == 3);
        CHECK_THAT(len[0], WithinRel(lmin, 1e-5));
        CHECK_THAT(len[1], WithinRel(lmax, 1e-5));
        CHECK_THAT(len[2], WithinRel(lsum / static_cast<Real>(mesh.num_edges()), 1e-5));
        CHECK(lmax > 1.05 * lmin);  // non-trivial spread
        const std::vector<Real> aspect = report_reals(mesh.quality_report(), "max aspect ratio");
        REQUIRE(!aspect.empty());
        CHECK(aspect[0] > 1.0);
        CHECK(aspect[0] < 1.5);
    }
}

TEST_CASE("quality() holds the numbers of the quality report", "[geometry]") {
    const TriangleMesh mesh = make_icosphere(kRadius, 2, kCenter);
    const MeshQuality q = mesh.quality();
    const std::string report = mesh.quality_report();
    CHECK(report == to_string(q));
    CHECK(q.num_vertices == 162);
    CHECK(q.num_edges == 480);
    CHECK(q.num_triangles == 320);
    CHECK(q.num_boundary_edges == 0);
    CHECK(q.num_unreferenced_vertices == 0);
    CHECK(q.euler_characteristic == 2);
    CHECK(q.num_components == 1);
    CHECK(q.num_vertex_components == 1);
    CHECK(q.closed);
    CHECK(q.consistently_oriented);
    const std::vector<Real> len = report_reals(report, "edge length min/max/mean");
    REQUIRE(len.size() == 3);
    CHECK_THAT(q.min_edge_length, WithinRel(len[0], 1e-5));
    CHECK_THAT(q.max_edge_length, WithinRel(len[1], 1e-5));
    CHECK_THAT(q.mean_edge_length, WithinRel(len[2], 1e-5));
    CHECK_THAT(q.max_aspect_ratio, WithinRel(report_reals(report, "max aspect ratio")[0], 1e-5));

    const MeshQuality empty = TriangleMesh().quality();
    CHECK(empty.num_triangles == 0);
    CHECK_FALSE(empty.closed);
    CHECK(std::isnan(empty.min_edge_length));
    CHECK(std::isnan(empty.max_aspect_ratio));
    CHECK(to_string(empty).find("max aspect ratio        : n/a") != std::string::npos);
}

TEST_CASE("bounding box of a small fixed point cloud is exact", "[geometry]") {
    Vertices v(4, 3);
    v << 0.25, -1.5, 3.0,  //
        -2.0, 0.5, 1.0,    //
        1.75, 2.25, -0.5,  //
        0.5, -0.75, 4.5;
    Triangles t(2, 3);
    t << 0, 1, 2, 0, 2, 3;
    const auto [lo, hi] = TriangleMesh(v, t).bounding_box();
    CHECK(lo == Vec3(-2.0, -1.5, -0.5));
    CHECK(hi == Vec3(1.75, 2.25, 4.5));
}

TEST_CASE("each closed component is oriented outward independently", "[geometry]") {
    const Vec3 c_small(0.0, 0.0, 0.0);
    const Vec3 c_large(10e-6, 0.0, 0.0);
    const TriangleMesh small = make_icosphere(1e-6, 2, c_small);
    const TriangleMesh large = make_icosphere(3e-6, 2, c_large);
    auto [v, t] = concatenate(small, large);
    const Triangles expected = t;
    const Index f_small = small.num_triangles();

    SECTION("small sphere inward") {
        for (Index i = 0; i < f_small; ++i) std::swap(t(i, 1), t(i, 2));
    }
    SECTION("large sphere inward (a global volume test would flip the small one)") {
        for (Index i = f_small; i < t.rows(); ++i) std::swap(t(i, 1), t(i, 2));
    }

    const TriangleMesh mesh(v, t);
    CHECK(mesh.num_components() == 2);
    CHECK(mesh.is_closed());
    CHECK(mesh.is_consistently_oriented());
    // Only the inward sphere was flipped; the other one is untouched.
    CHECK(mesh.triangles() == expected);
    bool outward = true;
    for (Index i = 0; i < mesh.num_triangles(); ++i) {
        const Vec3& c = i < f_small ? c_small : c_large;
        outward = outward && mesh.normal(i).dot(mesh.centroid(i) - c) > 0.0;
    }
    CHECK(outward);
    CHECK(plus_minus_convention_holds(mesh));
    CHECK_THAT(mesh.signed_volume(),
               WithinRel(small.signed_volume() + large.signed_volume(), 1e-10));
}

TEST_CASE("bow-tie: two spheres sharing one vertex are accepted with a warning", "[geometry]") {
    // For n >= 1 the poles on the x axis are exact vertices; sphere b touches sphere a at
    // (r, 0, 0). Merge b's pole into a's so the two closed surfaces share one vertex.
    constexpr Real r = 1e-6;
    const TriangleMesh a = make_icosphere(r, 1);
    const TriangleMesh b = make_icosphere(r, 1, Vec3(2 * r, 0, 0));
    Index ia = 0, ib = 0;
    a.vertices().col(0).maxCoeff(&ia);
    b.vertices().col(0).minCoeff(&ib);
    const Index na = a.num_vertices();
    Vertices v(na + b.num_vertices() - 1, 3);
    v.topRows(na) = a.vertices();
    for (Index j = 0, row = na; j < b.num_vertices(); ++j) {
        if (j != ib)
            v.row(row++) = b.vertices().row(j);
    }
    Triangles t(a.num_triangles() + b.num_triangles(), 3);
    t.topRows(a.num_triangles()) = a.triangles();
    for (Index i = 0; i < b.num_triangles(); ++i) {
        for (int k = 0; k < 3; ++k) {
            const Index j = b.triangles()(i, k);
            t(a.num_triangles() + i, k) = j == ib ? ia : (j < ib ? na + j : na + j - 1);
        }
    }

    const TriangleMesh mesh(v, t);
    CHECK(mesh.is_closed());
    CHECK(mesh.num_components() == 2);  // edge-connected
    const std::string report = mesh.quality_report();
    CHECK(report_value(report, "Euler characteristic") == 3);
    CHECK(report_value(report, "components \\(by edges\\)") == 2);
    CHECK(report_value(report, "components \\(by vertices\\)") == 1);
    bool outward = true;
    for (Index i = 0; i < mesh.num_triangles(); ++i) {
        const Vec3 c = i < a.num_triangles() ? Vec3::Zero() : Vec3(2 * r, 0, 0);
        outward = outward && mesh.normal(i).dot(mesh.centroid(i) - c) > 0.0;
    }
    CHECK(outward);
    CHECK(mesh.triangles() == t);  // nothing flipped
}

TEST_CASE("invalid mesh input throws", "[geometry]") {
    Vertices v(5, 3);
    v << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1;

    SECTION("out-of-range vertex index") {
        CHECK_THROWS_AS(TriangleMesh(v, single_row(0, 1, 5)), std::invalid_argument);
        CHECK_THROWS_AS(TriangleMesh(v, single_row(-1, 1, 2)), std::invalid_argument);
    }
    SECTION("repeated vertex") {
        CHECK_THROWS_AS(TriangleMesh(v, single_row(0, 1, 1)), std::invalid_argument);
    }
    SECTION("degenerate (collinear) triangle") {
        Vertices w(3, 3);
        w << 0, 0, 0, 1e-6, 1e-6, 1e-6, 3e-6, 3e-6, 3e-6;
        CHECK_THROWS_AS(TriangleMesh(w, single_row(0, 1, 2)), std::invalid_argument);
        Vertices u = v;
        u.row(2) = u.row(1);  // coincident vertices
        CHECK_THROWS_AS(TriangleMesh(u, single_row(0, 1, 2)), std::invalid_argument);
    }
    SECTION("non-finite coordinate") {
        Vertices u = v;
        u(1, 0) = std::nan("");
        CHECK_THROWS_AS(TriangleMesh(u, single_row(0, 1, 2)), std::invalid_argument);
    }
    SECTION("non-manifold edge (three triangles on one edge)") {
        Triangles t(3, 3);
        t << 0, 1, 2, 1, 0, 3, 0, 1, 4;
        try {
            const TriangleMesh mesh(v, t);
            FAIL("expected std::invalid_argument");
        } catch (const std::invalid_argument& e) {
            CHECK(std::string(e.what()).find("1 non-manifold") != std::string::npos);
        }
    }
    SECTION("duplicate triangle (same vertices, any order)") {
        Triangles same(2, 3);
        same << 0, 1, 2, 1, 2, 0;
        CHECK_THROWS_WITH(TriangleMesh(v, same), ContainsSubstring("duplicate"));
        Triangles reversed(2, 3);
        reversed << 0, 1, 2, 2, 1, 0;
        CHECK_THROWS_WITH(TriangleMesh(v, reversed), ContainsSubstring("duplicate"));
        // A duplicate pair plus a third triangle on a shared edge: rejected as non-manifold.
        Triangles pillow(3, 3);
        pillow << 0, 1, 2, 0, 2, 1, 0, 1, 4;
        CHECK_THROWS_AS(TriangleMesh(v, pillow), std::invalid_argument);
    }
    SECTION("non-orientable surface (Moebius strip)") {
        constexpr int n = 8;
        Vertices m(2 * n, 3);
        for (int i = 0; i < n; ++i) {
            const Real u = 2.0 * constants::pi * i / n;
            for (int s = 0; s < 2; ++s) {
                const Real w = s == 0 ? 0.4 : -0.4;
                m.row(2 * i + s) << (1.0 + w * std::cos(u / 2)) * std::cos(u),
                    (1.0 + w * std::cos(u / 2)) * std::sin(u), w * std::sin(u / 2);
            }
        }
        // Quad i joins column i to column i + 1; the last one joins with the twist.
        Triangles t(2 * n, 3);
        for (Index i = 0; i < n; ++i) {
            const Index t0 = 2 * i, b0 = 2 * i + 1;
            const Index t1 = i + 1 < n ? 2 * (i + 1) : 1;
            const Index b1 = i + 1 < n ? 2 * (i + 1) + 1 : 0;
            t.row(2 * i) << t0, b0, b1;
            t.row(2 * i + 1) << t0, b1, t1;
        }
        CHECK_THROWS_AS(TriangleMesh(m, t), std::invalid_argument);
        CHECK_THROWS_WITH(TriangleMesh(m, t), ContainsSubstring("non-orientable"));
    }
    SECTION("make_sphere rejects too-fine targets before refining") {
        // Targets just below the safe bound 1.0515 r / 2^10 (the old, 2x looser bound
        // built a 21 M-triangle sphere before throwing).
        for (const Real factor : {0.99, 0.6}) {
            const auto start = std::chrono::steady_clock::now();
            CHECK_THROWS_AS(make_sphere(kRadius, factor * 1.0515 * kRadius / 1024.0),
                            std::invalid_argument);
            const Real seconds =
                std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
            CHECK(seconds < 0.5);
        }
    }
    SECTION("generator arguments") {
        CHECK_THROWS_AS(make_icosphere(1.0, -1), std::invalid_argument);
        CHECK_THROWS_AS(make_icosphere(1.0, 11), std::invalid_argument);
        CHECK_THROWS_AS(make_icosphere(0.0, 1), std::invalid_argument);
        CHECK_THROWS_AS(make_icosphere(-1.0, 1), std::invalid_argument);
        CHECK_THROWS_AS(make_sphere(1.0, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(make_sphere(-1.0, 0.1), std::invalid_argument);
        CHECK_THROWS_AS(make_sphere(1.0, 1e-6), std::invalid_argument);  // > 10 subdivisions
    }
}

TEST_CASE("make_sphere meets the target edge length with the fewest subdivisions", "[geometry]") {
    const Real target = GENERATE(1.5e-6, 0.9e-6, 0.4e-6, 0.21e-6, 0.1e-6);
    const TriangleMesh mesh = make_sphere(kRadius, target, kCenter);
    const Real mean = mean_edge_length(mesh);
    CHECK(mean <= target);
    CHECK(mean > 0.5 * target);
    CHECK(mesh.is_closed());
    CHECK(all_normals_outward(mesh, kCenter));
    // Smallest admissible subdivision count: one level coarser misses the target.
    int n = 0;
    while ((Index{20} << (2 * n)) < mesh.num_triangles()) ++n;
    REQUIRE((Index{20} << (2 * n)) == mesh.num_triangles());
    if (n > 0) {
        CHECK(mean_edge_length(make_icosphere(kRadius, n - 1, kCenter)) > target);
    }
}

TEST_CASE("icosphere with 81920 triangles (n = 6) builds quickly", "[geometry]") {
    const auto start = std::chrono::steady_clock::now();
    const TriangleMesh mesh = make_icosphere(1e-6, 6);
    const Real seconds =
        std::chrono::duration<Real>(std::chrono::steady_clock::now() - start).count();
    CHECK(mesh.num_triangles() == 81920);  // 20 * 4^6
    CHECK(mesh.num_edges() == 122880);
    CHECK(mesh.is_closed());
    // Typically ~0.3 s in the debug/sanitizer build; generous bound so it cannot flake.
    CHECK(seconds < 10.0);
}
