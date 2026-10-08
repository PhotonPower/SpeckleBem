#pragma once
/// @file rwg.hpp
/// Rao-Wilton-Glisson (RWG) rooftop basis functions on flat triangles.
///
///   f_n(r) = (l_n / 2 A_n^+) (r - p_n^+)   for r in T_n^+
///          = (l_n / 2 A_n^-) (p_n^- - r)   for r in T_n^-
///          = 0 otherwise
///
/// One basis function per interior edge. Both the electric current J and the
/// magnetic current M are expanded in the same set (Galerkin: test = basis).

#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"

namespace specklebem::basis {

class RwgSpace {
public:
    explicit RwgSpace(const geometry::TriangleMesh& mesh);

    [[nodiscard]] Index size() const { return mesh_.num_edges(); }  ///< N
    [[nodiscard]] const geometry::TriangleMesh& mesh() const { return mesh_; }

    /// Value of basis function n at point r on triangle t (0 if n not supported on t).
    [[nodiscard]] Vec3 value(Index n, Index t, const Vec3& r) const;
    /// Surface divergence of basis n on triangle t (+-l_n / A_t).
    [[nodiscard]] Real divergence(Index n, Index t) const;

    /// Basis functions supported on triangle t (up to 3) and their sign.
    struct Support {
        Index n[3];
        int sign[3];
        int count;
    };
    [[nodiscard]] Support support(Index t) const;

private:
    const geometry::TriangleMesh& mesh_;
    Eigen::Matrix<Index, Eigen::Dynamic, 2, Eigen::RowMajor> free_vertex_;  ///< p_n^+, p_n^-
};

}  // namespace specklebem::basis
