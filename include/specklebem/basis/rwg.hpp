#pragma once
/// @file rwg.hpp
/// Rao-Wilton-Glisson (RWG) rooftop basis functions on flat triangles
/// (S. M. Rao, D. R. Wilton, A. W. Glisson, IEEE TAP 30(3), 409-418, 1982).
///
///   f_n(r) = (l_n / 2 A_n^+) (r - p_n^+)   for r in T_n^+
///          = (l_n / 2 A_n^-) (p_n^- - r)   for r in T_n^-
///          = 0 otherwise
///
///   div_s f_n = + l_n / A_n^+ on T_n^+,  - l_n / A_n^- on T_n^-.
///
/// One basis function per interior edge; basis index n equals the interior edge index
/// of geometry::TriangleMesh::edges(). T_n^+ / T_n^- are edge_triangles()(n, 0) / (n, 1)
/// (T^+ traverses the edge a -> b, a < b, in its counter-clockwise order), p_n^+- the
/// vertex of T_n^+- opposite the edge (free vertex). The current of f_n flows from T_n^+
/// across edge n into T_n^-; its component along the edge normal pointing from T_n^+ into
/// T_n^- is 1 on the edge (f . nu = +1 with nu outward of T^+, -1 with nu outward of
/// T^-), and its normal component on the other two edges of T_n^+ and T_n^- is zero.
/// Boundary edges of open meshes carry no basis function. Both the electric current J and
/// the magnetic current M are expanded in the same set (Galerkin: test = basis).

#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"

namespace specklebem::basis {

/// RWG function space on a TriangleMesh.
///
/// Holds a non-owning reference to the mesh: the mesh must outlive the RwgSpace and must
/// not be modified (e.g. flip_normals()) while the space is in use, because the
/// constructor caches free vertices, divergences and the triangle -> basis lookup.
/// All per-element queries are O(1) (no searching). Index arguments are range-checked
/// with plain integer comparisons; out-of-range indices throw std::out_of_range.
class RwgSpace {
public:
    /// Builds the free vertices and the triangle -> (basis, sign) table in O(N + F).
    /// A mesh without interior edges yields an empty space (size() == 0).
    /// @throws std::logic_error if the mesh topology violates the plus / minus convention
    ///         (cannot happen for a constructed TriangleMesh).
    explicit RwgSpace(const geometry::TriangleMesh& mesh);
    /// The space stores a reference to the mesh, so it must not be built from a temporary.
    RwgSpace(const geometry::TriangleMesh&&) = delete;

    [[nodiscard]] Index size() const { return mesh_.num_edges(); }  ///< N
    [[nodiscard]] const geometry::TriangleMesh& mesh() const { return mesh_; }

    /// Value of basis function n at point r on triangle t (0 if n not supported on t).
    /// r is assumed to lie on triangle t (not checked; the affine formula is evaluated).
    /// @throws std::out_of_range for n outside [0, size()) or t outside [0, F).
    [[nodiscard]] Vec3 value(Index n, Index t, const Vec3& r) const;
    /// Surface divergence of basis n on triangle t (+-l_n / A_t, 0 if not supported).
    /// @throws std::out_of_range for n outside [0, size()) or t outside [0, F).
    [[nodiscard]] Real divergence(Index n, Index t) const;

    /// Basis functions supported on triangle t (up to 3) and their sign.
    /// Entries [0, count) are in ascending order of n; sign is +1 if t is T_n^+ and -1 if
    /// t is T_n^-. Unused slots hold n = -1 and sign = 0.
    struct Support {
        Index n[3];
        int sign[3];
        int count;
    };
    /// @throws std::out_of_range for t outside [0, F).
    [[nodiscard]] Support support(Index t) const;

    /// Plus triangle T_n^+ of basis n. @throws std::out_of_range for invalid n.
    [[nodiscard]] Index plus_triangle(Index n) const;
    /// Minus triangle T_n^- of basis n. @throws std::out_of_range for invalid n.
    [[nodiscard]] Index minus_triangle(Index n) const;
    /// Vertex index of the free vertex p_n^+ (vertex of T_n^+ opposite edge n).
    /// @throws std::out_of_range for invalid n.
    [[nodiscard]] Index plus_free_vertex(Index n) const;
    /// Vertex index of the free vertex p_n^- (vertex of T_n^- opposite edge n).
    /// @throws std::out_of_range for invalid n.
    [[nodiscard]] Index minus_free_vertex(Index n) const;

private:
    void check_basis(Index n) const;
    void check_triangle(Index t) const;

    const geometry::TriangleMesh& mesh_;
    Eigen::Matrix<Index, Eigen::Dynamic, 2, Eigen::RowMajor> free_vertex_;  ///< p_n^+, p_n^-
    /// Signed divergence per basis and side: (+l_n / A_n^+, -l_n / A_n^-).
    Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor> divergence_;
    /// Triangle -> supported basis indices (ascending, -1 for unused slots).
    Eigen::Matrix<Index, Eigen::Dynamic, 3, Eigen::RowMajor> triangle_basis_;
    /// Triangle -> sign of the corresponding entry of triangle_basis_ (+1, -1, 0 unused).
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> triangle_sign_;
    /// Triangle -> number of supported basis functions (0..3).
    Eigen::Matrix<int, Eigen::Dynamic, 1> triangle_count_;
};

}  // namespace specklebem::basis
