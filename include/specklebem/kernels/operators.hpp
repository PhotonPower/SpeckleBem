#pragma once
/// @file operators.hpp
/// Galerkin matrix elements of the integro-differential operators L_i and K_i
/// of the tangential formulation (Eqs. 1-2 of the paper):
///
///   L_i X = j w mu_i  int G_i X dS'  -  (1 / j w eps_i) grad int G_i div' X dS'
///   K_i X = int grad' G_i x X dS'     (principal value; the 1/2 n x X term is
///                                     added separately depending on the side)
///
/// Convention: grad' is the gradient with respect to the SOURCE point r',
/// grad' G = -grad_r G = (1 + jkR) G (r - r') / R^2 (kernels::grad_green returns grad_r G).
/// With this K the scattered fields of region i are (docs/03)
///   E_i^s = -L_i J_i + K_i M_i,   H_i^s = -K_i J_i - (1 / eta_i^2) L_i M_i.
///
/// Element routine: for a pair of triangles (t_test, t_src) return the
/// 3x3 block of <f_m, L_i f_n> and <f_m, K_i f_n> for all RWGs on them.
///
/// Galerkin entries (docs/03_theory_sie.md; G = exp(-jkR) / (4 pi R), exp(+jwt)):
///
///   L_mn = j w mu  int int f_m(r) . f_n(r') G dS' dS
///          + (1 / (j w eps)) int int (div f_m)(div' f_n) G dS' dS
///   K_mn = int int f_m(r) . (grad' G x f_n(r')) dS' dS            (principal value)
///        = -int int f_m(r) . (grad_r G x f_n(r')) dS' dS
///
/// The gradient of the scalar potential is moved onto the test function with
/// <f_m, grad phi> = -<div f_m, phi> (surface divergence theorem on the support of f_m;
/// derivation in src/kernels/operators.cpp). Both blocks are symmetric under
/// (m <-> n, t_test <-> t_src) in exact arithmetic. Quadrature by proximity class and
/// singularity subtraction for touching pairs follow ADR 0004.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/types.hpp"

namespace specklebem::kernels {

struct RegionParams {
    Complex k;    ///< wavenumber
    Complex eta;  ///< wave impedance
    Real omega;
    Complex eps;  ///< absolute permittivity
    Complex mu;   ///< absolute permeability
};

/// Quadrature degrees by proximity class (ADR 0004). Accuracy of the defaults (relative to
/// the block norm, measured in the WP7 review against independent hp-graded references):
///  * far (degree 3): 2.6e-5 for L and 1.3e-3 for K at the class boundary (centroid distance
///    2 x longest edge) on a lambda/13 mesh; up to 1e-2 for Ag; decreasing with distance.
///  * near (degree 8): about 2e-7 for L and 1e-5 for K.
///  * touching (degree 10): about 2e-4 for L and about 1e-4 of the 1/2 I jump term for K.
///    The limit is the outer Dunavant rule (the inner potential is only finitely smooth at
///    the shared edge / vertex); the singularity subtraction itself is exact to ~5e-11. A
///    graded outer rule for touching pairs is a planned follow-up (WP7b).
struct OperatorOptions {
    int quad_degree_far = 3;   ///< Dunavant degree for well-separated pairs (any of 1..20)
    int quad_degree_near = 8;  ///< for near pairs (distance < factor * size); positive-interior
    /// Outer rule and smooth remainder of touching pairs; positive-interior (ADR 0004).
    int quad_degree_sing = 10;
    Real near_distance_factor = 2.0;
};

/// Checks the options (ADR 0004): every degree in 1..20, quad_degree_near and
/// quad_degree_sing positive-interior (triangle_rule_is_positive_interior: 1, 2, 4, 5, 6, 8,
/// 9, 10, 12, 13, 14, 17, 19), near_distance_factor finite and >= 0. O(1) (integer checks
/// against a table built once). element_blocks calls it on every invocation; an assembler
/// may call it once up front to fail early.
/// @throws std::invalid_argument if any check fails.
void validate(const OperatorOptions& opt);

/// Computes the local L and K blocks (3x3 each) for one triangle pair.
///
/// Row slot a is the a-th entry of space.support(t_test), column slot b the b-th entry of
/// space.support(t_src) (ascending basis index); unused slots are zero. The sign of each
/// RWG function on its triangle is included (as in space.value / space.divergence). K is
/// the principal value without the jump term (see jump_block). Quadrature by proximity
/// class (classify with opt.near_distance_factor):
///  * far:  outer and inner Dunavant rule of degree quad_degree_far, plain kernel;
///  * near: both of degree quad_degree_near, plain kernel;
///  * identical, shared_edge, shared_vertex: outer rule of degree quad_degree_sing; inner
///    integral by singularity subtraction (1/R, R, grad(1/R) and grad R analytically with
///    static_integrals over t_src, the smooth remainder with the degree-quad_degree_sing
///    rule). Shared-edge and shared-vertex blocks are averaged over both orderings
///    ((B(t1, t2) + B(t2, t1)^T) / 2, twice the cost), identical L blocks are symmetrised,
///    so touching blocks are exactly symmetric: element_blocks(t_src, t_test) is bitwise
///    the transpose. K is exactly zero for coplanar touching pairs (identical triangles and
///    shared-edge / shared-vertex pairs with parallel normals: f_m, grad' G and f_n coplanar).
/// Allocation-free (rules come from the triangle_rule cache, which the first call of a
/// process may build). Not parallelised; safe to call concurrently.
/// @throws std::out_of_range for triangle indices outside the mesh.
/// @throws std::invalid_argument for invalid options (validate); if a touching pair reports a
///         boundary finite part of the static gradient integral (an outer quadrature point
///         on the boundary of the source triangle, which cannot occur for a valid,
///         non-overlapping mesh); or if L or K is not finite (e.g. geometrically intersecting
///         triangles that do not share vertex indices); the message names the pair.
void element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                    const RegionParams& region, const OperatorOptions& opt,
                    Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K);

/// Local Gram block of the K operator's jump term on triangle t:
///
///   I_mn = int_t f_m . (n x f_n) dS,   n = mesh normal of t (out of R2 into R1),
///
/// slots as in element_blocks (support(t), unused slots zero). Real-valued (stored as
/// complex for direct addition to K), antisymmetric (I_mn = -I_nm, zero diagonal) and exact
/// (quadratic integrand, Dunavant degree 2).
///
/// The K operator's jump term is +-1/2 I, with the sign set by the assembler per region
/// (docs/03_theory_sie.md). For reference, with the grad' convention of K above: for an
/// observation point approaching the surface from the side into which n points (R1),
/// K X -> PV K X + 1/2 n x X; from the opposite side (R2), K X -> PV K X - 1/2 n x X. The
/// Galerkin jump contribution is therefore +1/2 I on the R1 side and -1/2 I on the R2 side
/// for the mesh normal n; formulations that write region 2 with its own outward normal (-n)
/// or with J2 = -J apply those signs on top.
/// @throws std::out_of_range for t outside the mesh.
void jump_block(const basis::RwgSpace& space, Index t, Eigen::Matrix<Complex, 3, 3>& I);

}  // namespace specklebem::kernels
