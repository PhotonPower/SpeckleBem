#pragma once
/// @file singularity.hpp
/// Treatment of the 1/R singularities when source and test triangles coincide,
/// share an edge or a vertex.
///
/// Strategy (ADR 0004): singularity subtraction with analytic evaluation of the
/// static terms (Hänninen, Taskinen & Sarvas 2006, PIER 63), the smooth
/// remainder is integrated numerically. A Duffy / radial-angular transform is
/// kept as a cross-check implementation.
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"

namespace specklebem::kernels {

enum class Proximity { identical, shared_edge, shared_vertex, near, far };

/// Proximity class of the triangle pair (t_test, t_src), O(1):
///  * `identical`     if t_test == t_src;
///  * `shared_edge`   if the triangles share exactly two vertices (by vertex index);
///  * `shared_vertex` if they share exactly one vertex (by index);
///  * otherwise `near` if |c_test - c_src| < near_distance_factor * max(h_test, h_src), with c
///    the centroids and h the longest edge of each triangle, else `far`.
/// Vertices are compared by index only: geometrically coincident but distinct vertices
/// (e.g. a mesh with duplicated vertices) do not count as shared.
/// @throws std::invalid_argument if an index is outside [0, num_triangles) or
///         near_distance_factor is negative or not finite.
[[nodiscard]] Proximity classify(const geometry::TriangleMesh& m, Index t_test, Index t_src,
                                 Real near_distance_factor = 2.0);

/// Analytic static integrals over a flat source triangle T for an observation point r
/// (R = |r - r'|, r' in T, dS' the surface element of T). Evaluated in closed form with the
/// edge-sum formulas of Wilton et al. (1984), Graglia (1993) and Hänninen et al. (2006);
/// derivation, stable branches and thresholds are documented in src/kernels/singularity.cpp.
///
/// Geometry: n = (v1 - v0) x (v2 - v0) / |...| (any vertex orientation is accepted),
/// d = (r - v0) . n the signed height of r above the plane, Omega >= 0 the solid angle that
/// T subtends at r. For |d| <= 1e-12 x (longest edge) r counts as lying in the plane of T
/// (the true d is still used in the potentials, see singularity.cpp).
///
/// Set needed by the mixed-potential operators with singularity subtraction (ADR 0004): the
/// 1/R and R terms of the Taylor expansion of exp(-jkR)/R, each for a constant (scalar) and a
/// linear (vector, (r' - r) factor, as in RWG sources) source, plus the gradient of 1/R. In
/// Hänninen's notation K_q = int_T R^q dS': I_1R = K_-1, I_R = K_1, I_rho_R and I_rhoR are
/// the vector integrals int_T (r' - r) R^q dS' for q = -1, 1, and I_grad = int_T grad' R^-1.
///
/// In-plane observation points (|d| <= 1e-12 h):
///  * I_grad has normal component exactly 0, and its tangential part is the principal value
///    (a disc of radius eps around r removed, eps -> 0), sum_i m_i int_edge_i 1/R dl' with m_i
///    the outward in-plane edge normals. For d -> 0+- the normal component tends to
///    +-2 pi n (r projecting inside T), the tangential part is continuous.
///  * If r lies on the boundary of T (on an edge or at a vertex), I_1R, I_R, I_rho_R and
///    I_rhoR are finite and exact, but the tangential I_grad diverges logarithmically. The
///    returned value is the finite part sum over the edges *not* containing r of
///    m_i int_edge_i 1/R dl': the divergent edge integrals and the cut-off arc are dropped.
///    This convention is additive: summed over coplanar triangles that surround r (two
///    triangles sharing the edge, or a fan around the vertex), it gives the principal value
///    over their union.
/// The results are finite for every observation point (no NaN or inf).
///
/// Accuracy limits (h = longest edge, D = distance of r from T, eps = 2.2e-16):
///  * Far distances: the edge terms grow like (D/h)^2 relative to the result and cancel, so
///    the relative error is about 1e-16 (D/h)^2: ~1e-13 at D = 10 h, ~1e-11 at D = 100 h. The
///    1e-12 acceptance level (docs/05_validation.md) holds out to D ~ 30 h. The formulas are
///    meant for touching and near pairs; far pairs must use quadrature (ADR 0004).
///  * Sliver triangles (area A << h^2): the rounding error is absolute, about a few eps times
///    the natural size of each member, eps h for I_1R, eps h^2 for I_rho_R, eps h^3 for I_R,
///    eps h^4 for I_rhoR and eps for I_grad, while the results scale with A; the relative
///    error therefore grows like eps h^2 / A (near field).
struct StaticIntegrals {
    Real I_1R;     ///< int_T 1/R dS'                                          [m]
    Vec3 I_rho_R;  ///< int_T (r' - r) / R dS'  (= int_T grad' R dS')          [m^2]
    Vec3 I_grad;   ///< int_T grad'(1/R) dS' = int_T (r - r') / R^3 dS'; normal part
                   ///< n sign(d) Omega, principal value in-plane (see above)  [1]
    Real I_R;      ///< int_T R dS'                                            [m^3]
    Vec3 I_rhoR;   ///< int_T (r' - r) R dS'                                   [m^4]
    /// True iff the tangential I_grad is the boundary finite part described above, i.e. r
    /// lies in the plane of T within 1e-12 h of an edge or a vertex of T. This can only happen
    /// for coplanar touching pairs; element assembly may assert that it is never set for
    /// non-coplanar pairs (there r is always off the plane or away from the edges of T).
    bool grad_finite_part = false;
};

/// Evaluates StaticIntegrals for observation point r and source triangle (v0, v1, v2).
/// Allocation-free, O(1) (three edges: three logs, one atan2).
/// @throws std::invalid_argument for non-finite input or a degenerate triangle (area below
///         1e-12 x longest edge^2).
[[nodiscard]] StaticIntegrals static_integrals(const Vec3& r, const Vec3& v0, const Vec3& v1,
                                               const Vec3& v2);

}  // namespace specklebem::kernels
