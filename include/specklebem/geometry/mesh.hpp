#pragma once
/// @file mesh.hpp
/// Flat-triangle surface mesh: the single geometric representation used by the
/// solver. Any surface (sphere, rough plane, imported CAD/STL) is reduced to
/// this type, which keeps the solver geometry-agnostic.

#include "specklebem/core/types.hpp"

#include <string>
#include <utility>

namespace specklebem::geometry {

/// Closed, consistently oriented triangle surface mesh with per-element cache.
/// Normals point from the object (region R2) into the exterior (region R1).
class TriangleMesh {
public:
    TriangleMesh() = default;
    TriangleMesh(Vertices vertices, Triangles triangles);

    [[nodiscard]] Index num_vertices() const { return vertices_.rows(); }
    [[nodiscard]] Index num_triangles() const { return triangles_.rows(); }
    [[nodiscard]] Index num_edges() const { return edges_.rows(); }

    [[nodiscard]] const Vertices& vertices() const { return vertices_; }
    [[nodiscard]] const Triangles& triangles() const { return triangles_; }

    /// Interior edges; each carries one RWG basis function.
    [[nodiscard]] const Edges& edges() const { return edges_; }
    /// For every edge: the two adjacent triangles (plus / minus).
    [[nodiscard]] const Edges& edge_triangles() const { return edge_triangles_; }

    [[nodiscard]] Vec3 centroid(Index t) const;
    [[nodiscard]] Vec3 normal(Index t) const;
    [[nodiscard]] Real area(Index t) const;
    [[nodiscard]] Real edge_length(Index e) const;

    /// Axis-aligned bounding box (root box of the octree).
    [[nodiscard]] std::pair<Vec3, Vec3> bounding_box() const;

    /// Min / max / mean edge length, aspect ratio, non-manifold edge count.
    [[nodiscard]] std::string quality_report() const;

    [[nodiscard]] bool is_closed() const;
    void flip_normals();

private:
    void build_topology();

    Vertices vertices_;
    Triangles triangles_;
    Edges edges_;
    Edges edge_triangles_;
    Vertices centroids_;
    Vertices normals_;
    VectorXr areas_;
};

}  // namespace specklebem::geometry
