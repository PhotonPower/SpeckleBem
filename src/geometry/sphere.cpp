/// @file sphere.cpp
/// Icosphere generator: regular icosahedron, recursive midpoint subdivision with a
/// shared-midpoint cache, projection onto the sphere. Triangles are counter-clockwise
/// seen from outside (R1), so normals point out of the sphere (docs/06_conventions.md).
#include "specklebem/geometry/sphere.hpp"

#include "specklebem/core/logging.hpp"

#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace specklebem::geometry {

namespace {

constexpr int kMaxSubdivisions = 10;

using Face = std::array<Index, 3>;

/// Icosphere on the unit sphere centred at the origin.
struct UnitIcosphere {
    std::vector<Vec3> vertices;
    std::vector<Face> faces;
};

/// Regular icosahedron inscribed in the unit sphere, faces counter-clockwise from outside.
UnitIcosphere unit_icosahedron() {
    const Real phi = 0.5 * (1.0 + std::sqrt(5.0));
    UnitIcosphere s;
    s.vertices = {Vec3(-1, phi, 0), Vec3(1, phi, 0), Vec3(-1, -phi, 0), Vec3(1, -phi, 0),
                  Vec3(0, -1, phi), Vec3(0, 1, phi), Vec3(0, -1, -phi), Vec3(0, 1, -phi),
                  Vec3(phi, 0, -1), Vec3(phi, 0, 1), Vec3(-phi, 0, -1), Vec3(-phi, 0, 1)};
    for (Vec3& v : s.vertices) v.normalize();
    s.faces = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
               {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
               {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};
    // Enforce counter-clockwise order seen from outside, independent of the table above.
    for (Face& f : s.faces) {
        const auto v = [&](std::size_t k) -> const Vec3& {
            return s.vertices[static_cast<std::size_t>(f[k])];
        };
        const Vec3 n = (v(1) - v(0)).cross(v(2) - v(0));
        if (n.dot(v(0) + v(1) + v(2)) < 0.0)
            std::swap(f[1], f[2]);
    }
    return s;
}

/// One level of midpoint subdivision (F -> 4F) with midpoints projected to the unit
/// sphere. A midpoint cache keyed by the vertex pair makes neighbouring faces share
/// their new vertices; the orientation of every face is preserved.
void subdivide(UnitIcosphere& s) {
    const std::size_t nf = s.faces.size();
    const std::size_t n_new = 3 * nf / 2;  // closed mesh: E = 3F/2
    const auto key_stride = static_cast<std::uint64_t>(s.vertices.size() + n_new);
    std::unordered_map<std::uint64_t, Index> cache;
    cache.reserve(n_new);
    s.vertices.reserve(s.vertices.size() + n_new);
    const auto midpoint = [&](Index a, Index b) -> Index {
        if (a > b)
            std::swap(a, b);
        const std::uint64_t key =
            static_cast<std::uint64_t>(a) * key_stride + static_cast<std::uint64_t>(b);
        const auto [it, inserted] = cache.try_emplace(key, 0);
        if (inserted) {
            const Vec3 m =
                (s.vertices[static_cast<std::size_t>(a)] + s.vertices[static_cast<std::size_t>(b)])
                    .normalized();
            it->second = static_cast<Index>(s.vertices.size());
            s.vertices.push_back(m);
        }
        return it->second;
    };
    std::vector<Face> refined;
    refined.reserve(4 * nf);
    for (const Face& f : s.faces) {
        const Index ab = midpoint(f[0], f[1]);
        const Index bc = midpoint(f[1], f[2]);
        const Index ca = midpoint(f[2], f[0]);
        refined.push_back({f[0], ab, ca});
        refined.push_back({f[1], bc, ab});
        refined.push_back({f[2], ca, bc});
        refined.push_back({ab, bc, ca});
    }
    s.faces = std::move(refined);
}

/// Mean edge length. The mesh is closed, so every edge appears in exactly two faces and
/// the mean over all face edges equals the mean over unique edges.
Real mean_edge_length(const UnitIcosphere& s) {
    Real sum = 0.0;
    for (const Face& f : s.faces) {
        for (std::size_t k = 0; k < 3; ++k) {
            sum += (s.vertices[static_cast<std::size_t>(f[k])] -
                    s.vertices[static_cast<std::size_t>(f[(k + 1) % 3])])
                       .norm();
        }
    }
    return sum / static_cast<Real>(3 * s.faces.size());
}

/// Scales to `radius`, translates to `center` and builds the TriangleMesh.
TriangleMesh to_mesh(const UnitIcosphere& s, Real radius, const Vec3& center) {
    const auto nv = static_cast<Index>(s.vertices.size());
    const auto nf = static_cast<Index>(s.faces.size());
    Vertices vertices(nv, 3);
    for (Index i = 0; i < nv; ++i) {
        vertices.row(i) = (radius * s.vertices[static_cast<std::size_t>(i)] + center).transpose();
    }
    Triangles triangles(nf, 3);
    for (Index t = 0; t < nf; ++t) {
        const Face& f = s.faces[static_cast<std::size_t>(t)];
        triangles(t, 0) = f[0];
        triangles(t, 1) = f[1];
        triangles(t, 2) = f[2];
    }
    return {std::move(vertices), std::move(triangles)};
}

void check_radius_and_center(Real radius, const Vec3& center) {
    if (!(radius > 0.0) || !std::isfinite(radius)) {
        throw std::invalid_argument("sphere: radius must be positive and finite");
    }
    if (!center.allFinite()) {
        throw std::invalid_argument("sphere: center must be finite");
    }
}

}  // namespace

TriangleMesh make_icosphere(Real radius, int subdivisions, const Vec3& center) {
    check_radius_and_center(radius, center);
    if (subdivisions < 0 || subdivisions > kMaxSubdivisions) {
        throw std::invalid_argument("make_icosphere: subdivisions must lie in [0, " +
                                    std::to_string(kMaxSubdivisions) + "], got " +
                                    std::to_string(subdivisions));
    }
    UnitIcosphere s = unit_icosahedron();
    for (int level = 0; level < subdivisions; ++level) subdivide(s);
    return to_mesh(s, radius, center);
}

TriangleMesh make_sphere(Real radius, Real target_edge_length, const Vec3& center) {
    check_radius_and_center(radius, center);
    if (!(target_edge_length > 0.0) || !std::isfinite(target_edge_length)) {
        throw std::invalid_argument("make_sphere: target_edge_length must be positive and finite");
    }
    const Real target = target_edge_length / radius;  // relative to the unit sphere
    const auto too_fine = [&]() {
        return std::invalid_argument("make_sphere: target edge length " +
                                     std::to_string(target_edge_length) + " m needs more than " +
                                     std::to_string(kMaxSubdivisions) +
                                     " subdivisions for radius " + std::to_string(radius) + " m");
    };
    // Reject hopeless targets before generating anything. The projected mean edge length
    // is ~ 1.149 * 1.0515 r / 2^n for large n (exactly 1.0515 r at n = 0), so
    // 1.0515 r / 2^kMaxSubdivisions is a safe lower bound on what n = kMaxSubdivisions
    // achieves. Targets between this bound and the actual n = 10 mean (a ~15 % window)
    // are rejected only after refinement.
    if (target < 1.0515 / std::ldexp(1.0, kMaxSubdivisions))
        throw too_fine();
    // Refine until the actual mean edge length of the projected mesh meets the target;
    // this yields the smallest admissible number of subdivisions.
    UnitIcosphere s = unit_icosahedron();
    int n = 0;
    while (mean_edge_length(s) > target) {
        if (n == kMaxSubdivisions)
            throw too_fine();
        subdivide(s);
        ++n;
    }
    SBEM_DEBUG("make_sphere: radius {} m, target edge {} m -> {} subdivisions", radius,
               target_edge_length, n);
    return to_mesh(s, radius, center);
}

}  // namespace specklebem::geometry
