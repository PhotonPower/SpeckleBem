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

[[nodiscard]] Proximity classify(const geometry::TriangleMesh& m, Index t_test, Index t_src,
                                 Real near_distance_factor = 2.0);

/// Analytic integrals of 1/R and grad(1/R) over a flat triangle (static kernels).
struct StaticIntegrals {
    Real I_1R;     ///< int_T 1/R dS'
    Vec3 I_rho_R;  ///< int_T (r' - r) / R dS'
    Vec3 I_grad;   ///< int_T grad'(1/R) dS'
};
[[nodiscard]] StaticIntegrals static_integrals(const Vec3& r, const Vec3& v0, const Vec3& v1,
                                               const Vec3& v2);

}  // namespace specklebem::kernels
