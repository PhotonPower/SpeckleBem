#pragma once
/// @file mesh.hpp
/// Flat-triangle surface mesh: the single geometric representation used by the
/// solver. Any surface (sphere, rough plane, imported CAD/STL) is reduced to
/// this type, which keeps the solver geometry-agnostic.
///
/// Conventions (docs/06_conventions.md): triangles are counter-clockwise seen
/// from the exterior region R1, so the unit normal
///   n_t = (v1 - v0) x (v2 - v0) / |(v1 - v0) x (v2 - v0)|
/// points out of the object (R2) into R1.

#include "specklebem/core/types.hpp"

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace specklebem::geometry {

/// Mesh statistics of TriangleMesh::quality(); formatted by to_string() (quality_report()).
/// Edge lengths are taken over all edges, interior and boundary; undefined values (no edges,
/// no triangles) are NaN.
struct MeshQuality {
    Index num_vertices = 0;
    Index num_unreferenced_vertices = 0;
    Index num_edges = 0;  ///< interior edges (E)
    Index num_boundary_edges = 0;
    Index num_nonmanifold_edges = 0;  ///< always 0 for a constructed mesh
    Index num_triangles = 0;
    /// V_used - E_all + F (E_all includes boundary edges).
    Index euler_characteristic = 0;
    Index num_components = 0;         ///< connected through shared edges
    Index num_vertex_components = 0;  ///< connected through shared vertices
    bool closed = false;
    bool consistently_oriented = true;
    Real min_edge_length = std::numeric_limits<Real>::quiet_NaN();
    Real max_edge_length = std::numeric_limits<Real>::quiet_NaN();
    Real mean_edge_length = std::numeric_limits<Real>::quiet_NaN();
    /// Largest circumradius / (2 inradius) (1 for equilateral).
    Real max_aspect_ratio = std::numeric_limits<Real>::quiet_NaN();
};

/// Multi-line text of TriangleMesh::quality_report().
[[nodiscard]] std::string to_string(const MeshQuality& q);

/// Consistently oriented, edge-manifold triangle surface mesh with per-element cache.
/// Normals point from the object (region R2) into the exterior (region R1).
///
/// The constructor validates the connectivity, builds the edge topology in
/// O(F log F), repairs an inconsistent orientation (breadth-first flipping per
/// connected component, with a warning) and makes every *closed* connected component
/// (edge-connected, without boundary edges) point outward: each such component is
/// oriented independently so that its own signed volume is positive. Nested shells
/// (a closed component inside another, e.g. a cavity) are out of scope for the
/// two-region SIE and are not detected; they would also be oriented outward.
///
/// Open meshes are accepted: boundary edges (one adjacent triangle) are counted and
/// listed in boundary_edges(), but are excluded from edges() and carry no RWG function.
/// Closedness is queried with is_closed(); solvers that need a closed surface must
/// check it. After orientation, a closed mesh whose Euler characteristic differs from
/// 2 x (number of components) triggers a warning (genus > 0 or components touching at
/// a non-manifold vertex, e.g. a bow-tie); such meshes are accepted.
class TriangleMesh {
public:
    TriangleMesh() = default;
    /// @throws std::invalid_argument for out-of-range vertex indices, triangles with
    ///         repeated vertices, zero-area triangles, non-finite coordinates,
    ///         duplicate triangles (same three vertices in any order), non-manifold
    ///         edges (more than two adjacent triangles; the count is in the message)
    ///         or a non-orientable surface.
    TriangleMesh(Vertices vertices, Triangles triangles);

    [[nodiscard]] Index num_vertices() const { return vertices_.rows(); }
    [[nodiscard]] Index num_triangles() const { return triangles_.rows(); }
    /// Number of interior edges (= number of RWG functions N).
    [[nodiscard]] Index num_edges() const { return edges_.rows(); }
    /// Number of boundary edges (edges with exactly one adjacent triangle).
    [[nodiscard]] Index num_boundary_edges() const { return boundary_edges_.rows(); }
    /// Number of connected components, where triangles are connected through shared
    /// (interior) edges. Each component is oriented independently.
    [[nodiscard]] Index num_components() const { return num_components_; }

    [[nodiscard]] const Vertices& vertices() const { return vertices_; }
    [[nodiscard]] const Triangles& triangles() const { return triangles_; }

    /// Interior edges (exactly two adjacent triangles); each carries one RWG basis
    /// function. Row e is the vertex pair (a, b) with a < b. Rows are sorted
    /// lexicographically by (a, b).
    [[nodiscard]] const Edges& edges() const { return edges_; }
    /// For every interior edge e = (a, b), a < b: the two adjacent triangles.
    ///  * edge_triangles()(e, 0) is the **plus** triangle T+: the edge appears as the
    ///    directed edge a -> b in its counter-clockwise vertex order (v0 v1 v2 v0).
    ///  * edge_triangles()(e, 1) is the **minus** triangle T-: the edge appears as b -> a.
    /// For a consistently oriented mesh every interior edge has exactly one of each.
    /// flip_normals() exchanges the two columns. The RWG space relies on this.
    [[nodiscard]] const Edges& edge_triangles() const { return edge_triangles_; }
    /// Boundary edges (one adjacent triangle), (a, b) with a < b, sorted.
    [[nodiscard]] const Edges& boundary_edges() const { return boundary_edges_; }

    [[nodiscard]] Vec3 centroid(Index t) const;
    /// Unit normal (v1 - v0) x (v2 - v0) / |...| of triangle t (out of R2 into R1).
    [[nodiscard]] Vec3 normal(Index t) const;
    [[nodiscard]] Real area(Index t) const;
    /// Length of interior edge e.
    [[nodiscard]] Real edge_length(Index e) const;

    /// Axis-aligned bounding box (root box of the octree): exact min / max over all
    /// vertices. @throws std::logic_error for a mesh without vertices.
    [[nodiscard]] std::pair<Vec3, Vec3> bounding_box() const;

    /// Multi-line summary: V, E (interior), F, Euler characteristic
    /// V_used - E_all + F (E_all includes boundary edges, V_used counts referenced
    /// vertices), connected components (edge- and vertex-connected), closedness,
    /// orientation, boundary / non-manifold edge counts, min / max / mean length over
    /// all edges and max aspect ratio (circumradius / (2 inradius), 1 for equilateral).
    /// The non-manifold edge count is always 0 for a constructed mesh, because the
    /// constructor throws on non-manifold edges (the count is in the exception message).
    [[nodiscard]] std::string quality_report() const { return to_string(quality()); }
    /// The statistics of quality_report() as numbers.
    [[nodiscard]] MeshQuality quality() const;

    /// True if there are no boundary and no non-manifold edges and F > 0.
    [[nodiscard]] bool is_closed() const;
    /// True if every interior edge is traversed in opposite directions by its two
    /// triangles. Always true after construction (the constructor repairs it).
    [[nodiscard]] bool is_consistently_oriented() const { return consistently_oriented_; }
    /// Signed enclosed volume V = (1/3) sum_t (c_t . n_t) A_t of the whole mesh, i.e.
    /// the sum over all components (positive for outward normals). Evaluated relative
    /// to the bounding-box centre to limit cancellation; only meaningful for closed
    /// meshes (it is origin-dependent otherwise).
    [[nodiscard]] Real signed_volume() const;

    /// Reverses the orientation of every triangle (swaps two vertex indices), negates
    /// the normals and exchanges plus / minus triangles of every interior edge.
    void flip_normals();

private:
    void validate() const;
    void build_topology();
    void compute_geometry();
    /// Flips triangles so that every interior edge is traversed in opposite
    /// directions; returns the number of flipped triangles.
    Index repair_orientation();
    /// Labels edge-connected components and counts vertex-connected components and
    /// referenced vertices.
    void label_components();
    /// Flips every closed component whose own signed volume is negative.
    void orient_closed_components_outward();
    /// V_used - E_all + F.
    [[nodiscard]] Index euler_characteristic() const;

    Vertices vertices_;
    Triangles triangles_;
    Edges edges_;
    Edges edge_triangles_;
    Edges boundary_edges_;
    std::vector<Index> boundary_edge_triangle_;  ///< triangle adjacent to boundary edge
    std::vector<Index> triangle_component_;      ///< edge-connected component per triangle
    Vertices centroids_;
    Vertices normals_;
    VectorXr areas_;
    Index num_components_ = 0;
    Index num_vertex_components_ = 0;
    Index num_used_vertices_ = 0;
    Index num_nonmanifold_edges_ = 0;
    bool consistently_oriented_ = true;
};

}  // namespace specklebem::geometry
