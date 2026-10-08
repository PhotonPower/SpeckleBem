/// @file rwg.cpp
/// RWG basis function space (Rao, Wilton, Glisson 1982): free vertices, signed
/// divergences and an O(1) triangle -> (basis, sign) lookup built once from the mesh
/// topology. Plus / minus triangles follow TriangleMesh::edge_triangles().
#include "specklebem/basis/rwg.hpp"

#include <stdexcept>
#include <string>

namespace specklebem::basis {

namespace {

/// Vertex of triangle t that is neither a nor b (n is the edge index, for the message).
/// @throws std::logic_error if a or b is not a vertex of t.
Index free_vertex_of(const Triangles& tris, Index t, Index a, Index b, Index n) {
    Index free = -1;
    int on_edge = 0;
    for (int k = 0; k < 3; ++k) {
        const Index v = tris(t, k);
        if (v == a || v == b) {
            ++on_edge;
        } else {
            free = v;
        }
    }
    if (on_edge != 2 || free < 0) {
        throw std::logic_error("RwgSpace: triangle " + std::to_string(t) +
                               " does not contain interior edge " + std::to_string(n));
    }
    return free;
}

}  // namespace

RwgSpace::RwgSpace(const geometry::TriangleMesh& mesh) : mesh_(mesh) {
    const Index num_basis = mesh_.num_edges();
    const Index num_tri = mesh_.num_triangles();
    const Edges& edges = mesh_.edges();
    const Edges& edge_tri = mesh_.edge_triangles();
    const Triangles& tris = mesh_.triangles();

    free_vertex_.resize(num_basis, 2);
    divergence_.resize(num_basis, 2);
    triangle_basis_.setConstant(num_tri, 3, Index{-1});
    triangle_sign_.setZero(num_tri, 3);
    triangle_count_.setZero(num_tri);

    // Basis indices are visited in ascending order, so each triangle's slots end up sorted.
    for (Index n = 0; n < num_basis; ++n) {
        const Index a = edges(n, 0);
        const Index b = edges(n, 1);
        const Index t_plus = edge_tri(n, 0);
        const Index t_minus = edge_tri(n, 1);
        if (t_plus == t_minus) {
            throw std::logic_error("RwgSpace: plus and minus triangle of edge " +
                                   std::to_string(n) + " coincide");
        }
        free_vertex_(n, 0) = free_vertex_of(tris, t_plus, a, b, n);
        free_vertex_(n, 1) = free_vertex_of(tris, t_minus, a, b, n);

        const Real length = mesh_.edge_length(n);
        divergence_(n, 0) = length / mesh_.area(t_plus);
        divergence_(n, 1) = -length / mesh_.area(t_minus);

        for (int side = 0; side < 2; ++side) {
            const Index t = side == 0 ? t_plus : t_minus;
            const int slot = triangle_count_(t);
            if (slot >= 3) {
                throw std::logic_error("RwgSpace: triangle " + std::to_string(t) +
                                       " is adjacent to more than three interior edges");
            }
            triangle_basis_(t, slot) = n;
            triangle_sign_(t, slot) = side == 0 ? 1 : -1;
            triangle_count_(t) = slot + 1;
        }
    }
}

void RwgSpace::check_basis(Index n) const {
    if (n < 0 || n >= size()) {
        throw std::out_of_range("RwgSpace: basis index " + std::to_string(n) + " outside [0, " +
                                std::to_string(size()) + ")");
    }
}

void RwgSpace::check_triangle(Index t) const {
    if (t < 0 || t >= mesh_.num_triangles()) {
        throw std::out_of_range("RwgSpace: triangle index " + std::to_string(t) + " outside [0, " +
                                std::to_string(mesh_.num_triangles()) + ")");
    }
}

Vec3 RwgSpace::value(Index n, Index t, const Vec3& r) const {
    check_basis(n);
    check_triangle(t);
    // Both sides in one form: f_n = (div_s f_n / 2) (r - p), since on T^- the factor
    // -l / (2 A^-) turns (r - p^-) into (p^- - r).
    for (Index side = 0; side < 2; ++side) {
        if (mesh_.edge_triangles()(n, side) == t) {
            const Vec3 p = mesh_.vertices().row(free_vertex_(n, side)).transpose();
            return (0.5 * divergence_(n, side)) * (r - p);
        }
    }
    return Vec3::Zero();
}

Real RwgSpace::divergence(Index n, Index t) const {
    check_basis(n);
    check_triangle(t);
    for (Index side = 0; side < 2; ++side) {
        if (mesh_.edge_triangles()(n, side) == t)
            return divergence_(n, side);
    }
    return 0.0;
}

RwgSpace::Support RwgSpace::support(Index t) const {
    check_triangle(t);
    Support s{};
    for (int k = 0; k < 3; ++k) {
        s.n[k] = triangle_basis_(t, k);
        s.sign[k] = triangle_sign_(t, k);
    }
    s.count = triangle_count_(t);
    return s;
}

Index RwgSpace::plus_triangle(Index n) const {
    check_basis(n);
    return mesh_.edge_triangles()(n, 0);
}

Index RwgSpace::minus_triangle(Index n) const {
    check_basis(n);
    return mesh_.edge_triangles()(n, 1);
}

Index RwgSpace::plus_free_vertex(Index n) const {
    check_basis(n);
    return free_vertex_(n, 0);
}

Index RwgSpace::minus_free_vertex(Index n) const {
    check_basis(n);
    return free_vertex_(n, 1);
}

}  // namespace specklebem::basis
