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

/// Quadrature options (ADR 0004, WP7b). Accuracy of the defaults, relative to the block norm
/// (L and K separately), measured against independent references (tests/unit/test_operators.cpp):
///  * touching pairs (identical, shared edge, shared vertex), outer_grading_levels = 4,
///    |k| h <= 0.78: within 3e-9 (L, identical triangles), 1.2e-9 (L, shared edges; the R^3
///    term of the smooth remainder, growing like (|k| h)^4), 6e-10 (K) and 4e-10 (shared
///    vertices) of a relative-coordinate (Sauter-Schwab type) reference; raw swap asymmetry
///    below 5e-10. A 30 degree fold (near-singular geometry): 1.3e-8.
///  * near and far pairs: the degree is chosen per pair for target_accuracy = 1e-6 (see
///    target_accuracy); measured errors are 0.05 to 0.5 of the target for D/h = 1.1 .. 15 and
///    |k| h = 0.1 .. 3 (vacuum, Si, Ag); at the near/far class boundary of the n = 4 Mie mesh
///    6e-8 (n = 1.5) and 4e-7 (Ag).
///  * WP7 defaults for comparison (outer_grading_levels = 0, target_accuracy = 0,
///    quad_degree_near = 8): touching about 2e-4 (L) and 1e-2 of the K block (about 1e-4 of the
///    1/2 I jump term); at the class boundary of the n = 4 Mie mesh (lambda/13) 4.3e-3
///    (n = 1.5) and 2.5e-2 (Ag), mostly K; near degree 8 about 1e-5 (K).
struct OperatorOptions {
    /// Lower bound of the near/far degree selection (see target_accuracy); any of 1..20. With
    /// target_accuracy = 0 the fixed degree of far pairs.
    int quad_degree_far = 3;
    /// Upper bound of the near/far degree selection; positive-interior (ADR 0004), and
    /// >= quad_degree_far. With target_accuracy = 0 the fixed degree of near pairs. Also the
    /// right-hand-side rule of op::assemble_rhs.
    int quad_degree_near = 19;
    /// Touching pairs: Dunavant degree of the smooth remainder of the singularity subtraction
    /// (inner rule; for outer_grading_levels = 0 also the outer rule); positive-interior.
    int quad_degree_sing = 10;
    Real near_distance_factor = 2.0;
    /// Outer (test-triangle) rule of the analytic static part of touching pairs (WP7b):
    /// tensor rules of generalised Gauss-Legendre-log type on Duffy sub-triangles graded
    /// towards the singular set (all three edges for identical triangles, the shared edge,
    /// the shared vertex); level l in 1..6 uses n = 2 l points per direction for identical
    /// triangles, 4 + 2 l for shared edges and 2 + 2 l for shared vertices (default: 384, 288
    /// and 100 points; table in src/kernels/operators.cpp). 0 restores the WP7 scheme (one Dunavant
    /// rule of degree quad_degree_sing for the whole outer integral, blocks averaged over both
    /// orderings).
    int outer_grading_levels = 4;
    /// Graded path: touching blocks of pairs with |k| h > symmetrize_touching_above_kh (h the
    /// larger longest edge) are averaged over both orderings, (B(t1, t2) + B(t2, t1)^T) / 2
    /// (twice the cost for shared edges and vertices, free for identical triangles), which
    /// makes them bitwise symmetric. Below the threshold the raw asymmetry is < 1e-9 of the
    /// block norm (measured <= 4.4e-10 at |k| h = 0.78) and the blocks are not averaged.
    /// Above it the smooth remainder is no longer smooth on the triangle scale and the raw
    /// asymmetry grows (unaveraged, the PMCHWT matrix of the Ag icosphere n = 1 at 500 nm,
    /// |k| h ~ 12, is asymmetric by 1.5e-5). 0 = always average, infinity = never. Must not
    /// be NaN or negative.
    Real symmetrize_touching_above_kh = 1.0;
    /// Near and far pairs: the Dunavant degree d (the same rule on both triangles) is the
    /// smallest one in the ladder quad_degree_far, then every positive-interior degree up to
    /// quad_degree_near, whose estimated relative block error is <= target_accuracy; the
    /// estimate is an empirical upper envelope E_d(kappa), kappa = sqrt((h / D)^2 +
    /// (0.15 |k| h)^2), D the centroid distance and h the larger longest edge (calibration in
    /// src/kernels/operators.cpp). The choice is monotone: non-decreasing in |k| and h,
    /// non-increasing in D. Near-class pairs only use positive-interior degrees. If no degree
    /// of the ladder reaches the target, quad_degree_near is used. 0 disables the selection
    /// (fixed degrees: quad_degree_far for far pairs, quad_degree_near for near pairs, as in
    /// WP7). Must be finite and in [0, 1).
    Real target_accuracy = 1e-6;
};

/// Checks the options (ADR 0004): quad_degree_far in 1..20, quad_degree_near and
/// quad_degree_sing positive-interior (triangle_rule_is_positive_interior: 1, 2, 4, 5, 6, 8,
/// 9, 10, 12, 13, 14, 17, 19), quad_degree_far <= quad_degree_near (bounds of the degree
/// selection), near_distance_factor finite and >= 0, outer_grading_levels in 0..6,
/// target_accuracy finite and in [0, 1), symmetrize_touching_above_kh >= 0 (may be infinite). O(1)
/// (integer checks against a table built once). element_blocks calls it on every invocation; an
/// assembler may call it once up front to fail early.
/// @throws std::invalid_argument if any check fails.
void validate(const OperatorOptions& opt);

/// Computes the local L and K blocks (3x3 each) for one triangle pair.
///
/// Row slot a is the a-th entry of space.support(t_test), column slot b the b-th entry of
/// space.support(t_src) (ascending basis index); unused slots are zero. The sign of each
/// RWG function on its triangle is included (as in space.value / space.divergence). K is
/// the principal value without the jump term (see jump_block). Quadrature by proximity
/// class (classify with opt.near_distance_factor):
///  * near, far: outer and inner Dunavant rule of the degree chosen by the selection rule of
///    OperatorOptions::target_accuracy (the same for both orderings of a pair), plain kernel;
///  * identical, shared_edge, shared_vertex: inner integral by singularity subtraction
///    (1/R, R, grad(1/R), grad R and, on the graded path, R (r' - r) analytically with
///    static_integrals over t_src, the smooth remainder with the degree-quad_degree_sing
///    rule). With outer_grading_levels >= 1 (default) the analytic part is integrated over
///    t_test with the graded rule and the remainder with a Dunavant outer rule; the blocks are
///    averaged over both orderings only for |k| h > symmetrize_touching_above_kh (raw
///    asymmetry below 1e-9 of the block norm otherwise). With outer_grading_levels = 0 (WP7) the
///    whole outer integral uses the degree-quad_degree_sing rule, shared-edge and shared-vertex
///    blocks are averaged over both orderings ((B(t1, t2) + B(t2, t1)^T) / 2, twice the cost) and
///    identical L blocks are symmetrised. Averaged blocks are exactly symmetric:
///    element_blocks(t_src, t_test) is bitwise the transpose. K is exactly zero for coplanar
///    touching pairs (identical triangles and shared-edge / shared-vertex pairs with parallel
///    normals: f_m, grad' G and f_n coplanar).
/// Allocation-free (rules come from the triangle_rule cache and the graded tables, which the
/// first call of a process may build). Not parallelised; safe to call concurrently.
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
