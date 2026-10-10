#pragma once
/// @file octasphere.hpp
/// Octahedron-based sphere mesh of the WP22a study (the paper's mesh of the 4 um Ag sphere is
/// make_octasphere(2 um, 7): 131 072 triangles, 2N = 393 216 unknowns) and edge statistics.
/// Header-only and light (geometry only), so that tests/unit/test_mesh.cpp does not pull in
/// Simulation, MLFMM or Mie; tests/support/ag_sphere_mlfmm_support.hpp re-exports both.
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"

#include <Eigen/Geometry>  // cross

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace octasphere {

using namespace specklebem;

/// Octahedron-based sphere: regular octahedron refined `subdivisions` times by midpoint
/// subdivision with the midpoints projected onto the sphere (as geometry::make_icosphere), 8 4^n
/// triangles, 4 4^n + 2 vertices, 12 4^n edges; counter-clockwise seen from outside (normals out
/// of R2 into R1, docs/06).
/// @throws std::invalid_argument for subdivisions outside [0, 9] or radius not finite and > 0.
inline geometry::TriangleMesh make_octasphere(Real radius, int subdivisions) {
    if (subdivisions < 0 || subdivisions > 9)
        throw std::invalid_argument("make_octasphere: subdivisions must lie in [0, 9]");
    if (!(radius > 0.0) || !std::isfinite(radius))
        throw std::invalid_argument("make_octasphere: radius must be finite and > 0");
    using Face = std::array<Index, 3>;
    std::vector<Vec3> v = {Vec3::UnitX(),  -Vec3::UnitX(), Vec3::UnitY(),
                           -Vec3::UnitY(), Vec3::UnitZ(),  -Vec3::UnitZ()};
    std::vector<Face> f = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
                           {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
    for (Face& t : f) {
        const Vec3& a = v[static_cast<std::size_t>(t[0])];
        const Vec3& b = v[static_cast<std::size_t>(t[1])];
        const Vec3& c = v[static_cast<std::size_t>(t[2])];
        if ((b - a).cross(c - a).dot(a + b + c) < 0.0)
            std::swap(t[1], t[2]);
    }
    for (int level = 0; level < subdivisions; ++level) {
        const std::size_t nf = f.size();
        const auto stride = static_cast<std::uint64_t>(v.size() + 3 * nf / 2);
        std::unordered_map<std::uint64_t, Index> cache;
        cache.reserve(3 * nf / 2);
        const auto midpoint = [&](Index a, Index b) -> Index {
            if (a > b)
                std::swap(a, b);
            const std::uint64_t key =
                static_cast<std::uint64_t>(a) * stride + static_cast<std::uint64_t>(b);
            const auto [it, inserted] = cache.try_emplace(key, 0);
            if (inserted) {
                const Vec3 m =
                    (v[static_cast<std::size_t>(a)] + v[static_cast<std::size_t>(b)]).normalized();
                it->second = static_cast<Index>(v.size());
                v.push_back(m);
            }
            return it->second;
        };
        std::vector<Face> refined;
        refined.reserve(4 * nf);
        for (const Face& t : f) {
            const Index ab = midpoint(t[0], t[1]);
            const Index bc = midpoint(t[1], t[2]);
            const Index ca = midpoint(t[2], t[0]);
            refined.push_back({t[0], ab, ca});
            refined.push_back({t[1], bc, ab});
            refined.push_back({t[2], ca, bc});
            refined.push_back({ab, bc, ca});
        }
        f = std::move(refined);
    }
    Vertices vertices(static_cast<Index>(v.size()), 3);
    for (std::size_t i = 0; i < v.size(); ++i)
        vertices.row(static_cast<Index>(i)) = (radius * v[i]).transpose();
    Triangles triangles(static_cast<Index>(f.size()), 3);
    for (std::size_t t = 0; t < f.size(); ++t)
        for (Index k = 0; k < 3; ++k)
            triangles(static_cast<Index>(t), k) = f[t][static_cast<std::size_t>(k)];
    return geometry::TriangleMesh(std::move(vertices), std::move(triangles));
}

/// Edge statistics of a mesh [m].
struct EdgeStats {
    Real mean = 0, min = 0, max = 0;
};

inline EdgeStats edge_stats(const geometry::TriangleMesh& mesh) {
    EdgeStats s;
    s.min = 1e300;
    Real sum = 0.0;
    for (Index e = 0; e < mesh.num_edges(); ++e) {
        const Vec3 a = mesh.vertices().row(mesh.edges()(e, 0)).transpose();
        const Vec3 b = mesh.vertices().row(mesh.edges()(e, 1)).transpose();
        const Real l = (b - a).norm();
        sum += l;
        s.min = std::min(s.min, l);
        s.max = std::max(s.max, l);
    }
    s.mean = sum / static_cast<Real>(mesh.num_edges());
    return s;
}

}  // namespace octasphere
