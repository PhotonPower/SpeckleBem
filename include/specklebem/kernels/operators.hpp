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
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/kernels/singularity.hpp"

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
///    below 5e-10. Sharp folds and obtuse angles (fold_adaptive, WP7c): <= 8.3e-8 for dihedral
///    angles 30 to 179 degrees between the triangles of regular, skewed, obtuse (106 degrees at
///    the shared vertex) and doubly obtuse (106 and 120 degrees at the same shared vertex)
///    shared-edge pairs, shared-edge pairs whose far vertex projects within 0.2 |AB| of a shared
///    vertex (folds below 90 degrees) and regular, skewed and obtuse shared-vertex pairs (slow
///    test "fold sweep"; the WP7b rule alone: up to 4.5e-5 for a skewed 30-degree hinge, 4.2e-5
///    for a 45-degree hinge with the far vertex next to B, 8.7e-5 for an obtuse 30-degree shared
///    vertex). Not covered: folds of 90 to ~95 degrees with the far vertex projecting next to a
///    shared vertex keep the WP7b table, up to 1.5e-7 (90 degrees, far vertex 0.27 |AB| from B).
///  * near and far pairs: the degree is chosen per pair for target_accuracy = 1e-5 (default; see
///    target_accuracy); measured errors are 0.05 to 0.5 of the target for D/h = 1.1 .. 15 and
///    |k| h = 0.1 .. 3 (vacuum, Si, Ag); at the near/far class boundary of the n = 4 Mie mesh
///    4.4e-7 (n = 1.5) and 2.5e-6 (Ag) (target 1e-6: 5.8e-8 and 4.4e-7). Against
///    target_accuracy = 1e-6 the dense n = 3 Mie assembly is 20 to 38 % faster and eps_rr
///    changes by <= 1e-8 (absolute).
///  * WP7 defaults for comparison (outer_grading_levels = 0, target_accuracy = 0,
///    quad_degree_near = 8): touching about 2e-4 (L) and 1e-2 of the K block (about 1e-4 of the
///    1/2 I jump term); at the class boundary of the n = 4 Mie mesh (lambda/13) 4.3e-3
///    (n = 1.5) and 2.5e-2 (Ag), mostly K; near degree 8 about 1e-5 (K).
struct OperatorOptions {
    /// Lower bound of the near/far degree selection (see target_accuracy); any of 1..20. With
    /// target_accuracy = 0 the fixed degree of far pairs.
    int quad_degree_far = 3;
    /// Upper bound of the near/far degree selection; positive-interior (ADR 0004), and
    /// >= quad_degree_far. With target_accuracy = 0 the fixed degree of near pairs.
    int quad_degree_near = 19;
    /// Dunavant degree of the right-hand side <f_m, E_inc>, <f_m, H_inc> (op::assemble_rhs,
    /// WP7c; before WP7c the right-hand side used quad_degree_near); positive-interior. Default
    /// 8: relative error against degree 20 for a plane wave and a paraxial Gaussian beam (w0 =
    /// lambda) on an icosphere and a rough-surface box at h = lambda / 10 and lambda / 27: 7.5e-14
    /// (degree 6: 7.1e-11, degree 5: 5.0e-9, degree 4: 5.2e-7); 9.4e-10 against degree 19 even
    /// at h ~ lambda / 3 (tests/unit/test_assembler.cpp).
    int quad_degree_rhs = 8;
    /// Touching pairs: Dunavant degree of the smooth remainder of the singularity subtraction
    /// (inner rule; for outer_grading_levels = 0 also the outer rule); positive-interior. Fixed,
    /// not chosen by target_accuracy: the remainder error grows like (|k| h)^4 (about 1e-9 at
    /// |k| h = 0.78, about 1e-6 at |k| h = 4); a k-aware choice is a follow-up.
    int quad_degree_sing = 10;
    Real near_distance_factor = 2.0;
    /// Outer (test-triangle) rule of the analytic static part of touching pairs (WP7b):
    /// tensor rules of generalised Gauss-Legendre-log type on Duffy sub-triangles graded
    /// towards the singular set (all three edges for identical triangles, the shared edge,
    /// the shared vertex); level l in 1..6 uses n = 2 l points per direction for identical
    /// triangles, 4 + 2 l for shared edges and 2 + 2 l for shared vertices (default: 384, 288
    /// and 100 points; table in src/kernels/operators.cpp). 0 restores the WP7 scheme (one Dunavant
    /// rule of degree quad_degree_sing for the whole outer integral, blocks averaged over both
    /// orderings). Level 4 reaches ~1e-8 of the block norm for shared edges with dihedral
    /// angles >= ~90 degrees and non-obtuse angles at the shared vertices; sharper folds and
    /// obtuse angles use the fold-adaptive pieces (fold_adaptive), whose point numbers also
    /// follow this level (6 + 2 l radial and 2 + 2 l angular points per piece, each at most 16;
    /// level l + 2 or l + 3 on the pieces of doubly obtuse and near-vertex folds).
    int outer_grading_levels = 4;
    /// Fold-adaptive outer rule of shared-edge and shared-vertex pairs (WP7c, ADR 0004; graded
    /// path only). The test triangle is cut into Duffy pieces with apex at a shared vertex such
    /// that every singular direction of the analytic part through the apex (the apex itself,
    /// the shared edge, the source edges from it, which project onto the test triangle for
    /// folds below 90 degrees) lies on a graded side of a piece or at least half a side length
    /// away from it (algorithm in src/kernels/operators.cpp). Pairs that need no cut keep the
    /// WP7b table bitwise: folds >= 90 degrees whose pieces are not split by the apex point
    /// feature (shared edge: |AB| h_C >= |MC|^2 with h_C the height of C over AB and M the
    /// midpoint of AB, roughly the median MC not longer than AB; shared vertex: 4 area >=
    /// |BC|^2, roughly a non-obtuse angle at A) and without obtuse angles at the same shared
    /// vertex in both triangles, e.g. all touching pairs of the Mie icospheres. Right angles
    /// (structured rough-surface grids, the box rim) count as non-obtuse with a relative
    /// tolerance of 1e-6, so they select one rule independently of rounding (WP7d). Shared
    /// vertices with a steep source triangle (fold ~46 to ~134 degrees) that has a vertex less
    /// than 11.5 degrees above the test plane (box rim) use the adaptive piece (WP7d). Accuracy:
    /// see the struct comment. Cost against the WP7b rule (WP7c measurements; WP7d figures in
    /// ADR 0004) (release, Si, time per call, both
    /// orderings; fold sweep, quietest of three runs; single-pair maxima vary by up to x2 under
    /// load): folds below 90 degrees mean x1.9 (shared edges, at most x3.3 for doubly obtuse
    /// pairs) and x1.6 (shared vertices, at most x3.2), folds >= 90 degrees x1.3 (shared edges,
    /// at most x2.5) and x1.2 (shared vertices, at most x1.7).
    /// Deterministic outer points of the analytic part: see touching_rule_info; hard bound 48
    /// pieces of at most 16 x 16 points per ordering. false restores the WP7b rule for every pair.
    bool fold_adaptive = true;
    /// Graded path: touching blocks of pairs with |k| h > symmetrize_touching_above_kh (h the
    /// larger longest edge) are averaged over both orderings, (B(t1, t2) + B(t2, t1)^T) / 2
    /// (twice the cost for shared edges and vertices), which makes them bitwise symmetric.
    /// Identical-triangle L blocks are always symmetrised, (L + L^T) / 2 (free). Below the
    /// threshold the raw asymmetry is < 1e-9 of the block norm (measured <= 4.4e-10 at |k| h =
    /// 0.78) and the blocks are not averaged. Above it the smooth remainder is no longer smooth on
    /// the triangle scale and the raw asymmetry grows (unaveraged, the PMCHWT matrix of the Ag
    /// icosphere n = 1 at 500 nm, |k| h ~ 12, is asymmetric by 1.5e-5). 0 = always average,
    /// infinity = never. Must not be NaN or negative.
    Real symmetrize_touching_above_kh = 1.0;
    /// Near and far pairs: the Dunavant degree d (the same rule on both triangles) is the
    /// smallest one in the ladder quad_degree_far, then every positive-interior degree up to
    /// quad_degree_near, whose estimated relative block error is <= target_accuracy; the
    /// estimate is an empirical upper envelope E_d(kappa), kappa = sqrt((h / D)^2 +
    /// (0.15 |k| h)^2), D the centroid distance and h the larger longest edge (calibration in
    /// src/kernels/operators.cpp). The choice is monotone: non-decreasing in |k| and h,
    /// non-increasing in D (with the decay-aware target of lossy regions: non-decreasing in
    /// |Re k| and h, non-increasing in D; not monotone in Im k, since a larger -Im k raises |k|
    /// in the error model while the target relaxes only for pairs with R_lb > 0). Near-class
    /// pairs only use positive-interior degrees. If no degree
    /// of the ladder reaches the target, quad_degree_near is used. 0 disables the selection
    /// (fixed degrees: quad_degree_far for far pairs, quad_degree_near for near pairs, as in
    /// WP7). Must be finite and in [0, 1). Governs near and far pairs only: touching pairs
    /// (identical, shared edge, shared vertex) use the fixed quad_degree_sing and
    /// outer_grading_levels, whose remainder error grows like (|k| h)^4 (about 1e-9 at |k| h =
    /// 0.78, about 1e-6 at |k| h = 4) independently of this target. Default 1e-5: against 1e-6
    /// the dense Mie n = 3 assembly is 20 to 38 % faster and eps_rr changes by <= 1e-8.
    /// Lossy regions: see decay_aware_target.
    Real target_accuracy = 1e-5;
    /// Decay-aware target for near and far pairs in lossy regions (WP-P2, WP7c part 4; ADR 0004).
    /// With alpha = -Im k > 0 and R_lb = D - rho_test - rho_src > 0 a lower bound of |r - r'| over
    /// the pair (rho the largest centroid-vertex distance of a triangle), |G| and |grad G| are
    /// at most delta = (1 + alpha R_lb) exp(-alpha R_lb) times their undamped bounds 1 / (4 pi R)
    /// and (1 + |Re k| R) / (4 pi R^2). The degree selection then asks for the relative block error
    /// target_accuracy / delta instead of target_accuracy, and uses the lowest degree of the ladder
    /// when (S_d^2 + 1) delta <= target_accuracy (S_d = sum |w| of the rule: any rule meets the
    /// bound). Guarantee per block B (Frobenius norm, L and K separately):
    ///   ||B_d - B|| <= target_accuracy ||U||,
    /// U the entrywise bound A_test A_src max |integrand| with the undamped kernel magnitudes
    /// above: the error allowed to a block equals what the same pair may have in a lossless region
    /// (target_accuracy ||B|| <= target_accuracy ||U||), so the error summed over a matrix row
    /// is bounded as for a lossless medium of the same geometry, while blocks that the
    /// attenuation makes negligible get cheap rules. Lossless regions (Im k = 0) and touching
    /// pairs are unaffected (bitwise). Ag at 500 nm (alpha = 39 / um): pairs with R_lb > ~0.4 um
    /// are negligible at target 1e-5. false: the relative target applies to every block.
    bool decay_aware_target = true;
    /// Near and far pairs: evaluate exp(-jkR) with the branch-free, vectorisable fastmath
    /// functions (kernels/fast_math.hpp, WP-P2) instead of the C library's sin, cos and exp
    /// (blocks agree to ~1e-15 relative; measured on UCRT64 GCC: x1.35 (lossless) / x1.8 (Si)
    /// per degree-19 far pair, x1.5 on the whole rough-box assembly, x2.9 for the isolated
    /// sin + cos loop; benchmarks/results/dense_assembly_profile.md). Used where the
    /// arguments are in range for the whole pair (|Re k| R <= 1e6, Im k <= 0); otherwise, and
    /// with false, the C library (the pre-WP-P2 arithmetic, bitwise). Touching pairs always
    /// use the C library.
    bool fast_plain_kernel = true;
    /// Touching pairs with a stretched (PML) triangle (WP-PML2, ADR 0009; element_blocks with a
    /// CoordinateStretch): Gauss-Legendre points p per plain and radial direction of the
    /// relative-coordinate (Sauter-Schwab type) rule on the analytic stretched kernel, and
    /// ceil(0.6 p) (at least 4) per panel of the directions graded toward the ridge of an
    /// anisotropic stretch; both raised by max(0, ceil(0.75 |k| s_max h) - 4) for large |k s| h
    /// (lossy media in the PML), at most 16 (src/kernels/stretched_operators.cpp, ADR 0004 note
    /// WP-PML2). Kernel evaluations grow like p^4 (shared vertex) and p^3 (shared edge).
    /// Default 10, measured on all touching pairs of a structured grid (h = lambda / 10, order-2
    /// profile sigma_max = 4, vacuum and Ag, against p = 16;
    /// tests/unit/test_stretched_kernels.cpp): relative block error <= 5e-11 (identical), <= 4e-8
    /// (shared edge), <= 3e-8 (shared vertex) at sigma = 1, <= 6e-7 (shared vertex) at sigma
    /// = 2.6-3.4 on one axis only, <= 1e-8 in the corner; mean cost per call 1.7e4-7e4
    /// (identical), 3.5e5-1.6e6 (shared edge), 5e5-6e6 (shared vertex) kernel evaluations (vacuum
    /// to Ag). p = 12 lowers the shared-vertex error to ~2e-7 at ~1.5 times the cost. Must be
    /// in 4..16. Pairs without a stretched triangle do not use it.
    int stretched_touching_points = 10;
};

/// Checks the options (ADR 0004): quad_degree_far in 1..20, quad_degree_near, quad_degree_rhs and
/// quad_degree_sing positive-interior (triangle_rule_is_positive_interior: 1, 2, 4, 5, 6, 8,
/// 9, 10, 12, 13, 14, 17, 19), quad_degree_far <= quad_degree_near (bounds of the degree
/// selection), near_distance_factor finite and >= 0, outer_grading_levels in 0..6,
/// target_accuracy finite and in [0, 1), symmetrize_touching_above_kh >= 0 (may be infinite),
/// stretched_touching_points in 4..16. O(1)
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
///    t_test with the graded rule (fold-adaptive pieces for sharp folds and obtuse angles of
///    shared-edge and shared-vertex pairs, see fold_adaptive) and the remainder with a
///    Dunavant outer rule; identical L
///    blocks are symmetrised, shared-edge and shared-vertex blocks are averaged over both
///    orderings only for |k| h > symmetrize_touching_above_kh (raw asymmetry below 1e-9 of the
///    block norm otherwise). With outer_grading_levels = 0 (WP7) the
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

/// element_blocks with a complex coordinate stretch (PML, ADR 0009 decisions 2 and 3, WP-PML2).
///
/// The mesh stays real; the stretch r~ = S(r) acts through the Piola transform of the RWG
/// functions, f~ dS~ = DS f dS and (div~ f~) dS~ = (div f) dS with DS = diag(s_x, s_y, 1),
/// s = 1 - j sigma (geometry::CoordinateStretch::jacobian_diag):
///   L_mn = j w mu int int (DS f_m) . (DS' f_n) G(R~) + (1 / (j w eps)) int int (div f_m)(div' f_n)
///   G(R~), K_mn = int int (DS f_m) . (grad~' G x DS' f_n),  grad~' G = (1 + j k R~) G (r~ - r~') /
///   R~^2,
/// G(R~) = exp(-j k R~) / (4 pi R~), R~ = sqrt((r~ - r~') . (r~ - r~')) unconjugated on the
/// principal branch (Re R~ >= 0; Im R~ <= 0 for the monotone stretch). Both blocks remain complex
/// symmetric; the jump term jump_block is unchanged for horizontal triangles (the Piola factors
/// cancel: (DS f_m) . (z x DS f_n) / det DS = f_m . (z x f_n)).
///
/// Dispatch: stretch == nullptr, stretch->is_identity(), or a pair of which neither triangle has a
/// vertex in the stretched zone (sigma_max > 0 and |x| > x.start or |y| > y.start) calls
/// element_blocks above: bitwise identical results. Otherwise (a stretched pair):
///  * near, far: double Dunavant rule of the degree selected for target_accuracy with the error
///    model of the plain kernel at kappa = sqrt((h / D_eff)^2 + (0.15 |k| s_max h)^2), s_max =
///    max |s| of the pair and D_eff <= D the distance shortened for the near-singular ridge of an
///    anisotropic stretch (src/kernels/stretched_operators.cpp); no decay-aware relaxation, C
///    library arithmetic; if no degree of the ladder meets the target, quad_degree_near
///    (stretched_pair_info reports it). The model is calibrated on L-type blocks; non-flat pairs
///    (tilted triangle and PML triangle, K != 0) select their degree for target / 20, since their
///    small K block has ~20 times the relative error of L (ADR 0004 note WP-PML2);
///  * identical, shared edge, shared vertex: relative-coordinate (Sauter-Schwab type) quadrature
///    of the analytic kernel with stretched_touching_points Gauss-Legendre points per direction;
///    for coplanar pairs with an anisotropic stretch the parameter domains are split along the
///    two ridge planes and graded toward them (ADR 0004, "Stretched kernels"). Non-coplanar
///    touching pairs (a tilted taper triangle touching the PML start, gap W_g = 0) use the rule
///    without the ridge split: accurate when sigma ~ 0 there, i.e. for profile order m >= 1,
///    where the first PML cell has sigma <= sigma_max (h / W)^m; for m < 1 the open-surface
///    setup (WP-PML3) must require W_g >= h, so that no tilted triangle touches the PML.
/// Stretched blocks are computed for the ordering (min(t_test, t_src), max(...)) and transposed
/// for the other one, so the stretched Galerkin matrix is bitwise complex symmetric. K is exactly
/// zero for pairs of horizontal triangles in one plane (all vectors in-plane). Allocation-free
/// (the rules live on the stack; the first call builds the Gauss-Legendre tables).
/// @throws std::invalid_argument as element_blocks, and if a stretched triangle is not horizontal
///         (normal not +-z within 1e-9: ADR 0009 requires flat PML triangles; the jump term and
///         the Piola factors rely on it).
void element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                    const RegionParams& region, const OperatorOptions& opt,
                    const geometry::CoordinateStretch* stretch, Eigen::Matrix<Complex, 3, 3>& L,
                    Eigen::Matrix<Complex, 3, 3>& K);

/// Rule of a stretched pair (diagnostics and cost studies, WP-PML2).
struct StretchedPairInfo {
    bool stretched = false;  ///< the pair uses the stretched path (false: element_blocks as before)
    Proximity proximity = Proximity::far;
    int degree = 0;  ///< near / far: Dunavant degree of both integrals (0 for touching pairs)
    /// near / far: the error model meets target_accuracy at `degree` (false: degree is the cap
    /// quad_degree_near and the estimate exceeds the target; true for target_accuracy = 0).
    bool target_reached = true;
    /// Relative angular width of the near-singular ridge of the anisotropic stretched kernel
    /// (1: isotropic, no ridge; ~1 / (1 + sigma^2) for a stretch on one axis only): touching
    /// pairs at the frozen metric of the pair (sigma at the mid-point of the two centroids;
    /// non-coplanar touching pairs, which are not ridge-split, the smaller of the in-plane ridge
    /// and the ridge between the stretched plane and the unstretched z axis), near / far pairs the
    /// smallest over its sigma range.
    Real ridge_width = 1.0;
    /// Effective distance D_eff / D of the near / far selection (1: the plain model).
    Real distance_factor = 1.0;
    bool ridge_split = false;  ///< touching: parameter domain split along the ridge planes
    bool coplanar = false;     ///< touching: coplanar pair (K = 0, ridge splitting possible)
    Index points = 0;          ///< kernel evaluations (point pairs) per call
};

/// The rule element_blocks(..., &stretch, ...) uses for the pair (t_test, t_src) in a region with
/// wavenumber k. Pure geometry and k (no RWG space needed); stretched = false and nothing else set
/// for pairs that keep the unstretched path.
/// @throws std::invalid_argument for invalid options or a stretched triangle that is not
///         horizontal; std::out_of_range for indices outside the mesh.
[[nodiscard]] StretchedPairInfo stretched_pair_info(const geometry::TriangleMesh& mesh,
                                                    Index t_test, Index t_src, Complex k,
                                                    const OperatorOptions& opt,
                                                    const geometry::CoordinateStretch& stretch);

/// Outer rule of the analytic part of a touching pair on the graded path (diagnostics, WP7c).
struct TouchingRuleInfo {
    Index pieces = 0;  ///< Duffy pieces (WP7b table: 6 identical, 2 shared edge, 1 shared vertex)
    Index points = 0;  ///< outer points of the analytic part for the ordering (t_test, t_src)
    bool fold_adaptive = false;  ///< fold-adaptive pieces (false: the WP7b table)
};

/// The outer rule element_blocks uses for the analytic part of the touching pair (t_test, t_src)
/// with options opt (one ordering; pairs averaged over both orderings evaluate both). Pure
/// geometry: deterministic point counts for cost studies. Independent of the RWG space: it
/// reports a rule also for triangles without interior edges, for which element_blocks returns
/// zero blocks without evaluating any rule. Hard bound per ordering: 48 pieces of
/// at most 16 x 16 points (12 288 points; WP7b table at level 4: 384 identical, 288 shared edge,
/// 100 shared vertex).
/// @throws std::invalid_argument for invalid options, outer_grading_levels = 0 (WP7 scheme, no
///         graded rule) or a pair that is not identical, shared_edge or shared_vertex.
/// @throws std::out_of_range for indices outside the mesh.
[[nodiscard]] TouchingRuleInfo touching_rule_info(const geometry::TriangleMesh& mesh, Index t_test,
                                                  Index t_src, const OperatorOptions& opt);

/// Dunavant degree element_blocks uses for both integrals of the near or far pair (t_test,
/// t_src) in a region with wavenumber k (the selection of OperatorOptions::target_accuracy;
/// quad_degree_near / quad_degree_far for target_accuracy = 0). Pure geometry and k:
/// diagnostics for cost studies (WP-P2). Symmetric in (t_test, t_src).
/// @throws std::invalid_argument for invalid options or a touching pair (identical,
///         shared_edge, shared_vertex).
/// @throws std::out_of_range for indices outside the mesh.
[[nodiscard]] int plain_rule_degree(const geometry::TriangleMesh& mesh, Index t_test, Index t_src,
                                    Complex k, const OperatorOptions& opt);

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
