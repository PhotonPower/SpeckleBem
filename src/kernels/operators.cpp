/// @file operators.cpp
/// Galerkin element blocks of the L and K operators for one triangle pair (docs/03, ADR 0004).
///
/// Notation: G = exp(-jkR) / (4 pi R), R = |r - r'|, r on the test triangle T (outer
/// integral), r' on the source triangle T' (inner integral). On a triangle the RWG function
/// is f(r) = (D / 2)(r - p) with D = div_s f = +-l / A (sign included) and p its free vertex
/// (basis/rwg.hpp), so every slot is described by (D, p).
///
/// L block (mixed-potential form). With A = int G f_n dS' and phi = int G div' f_n dS',
///   L f_n = j w mu A - (1 / (j w eps)) grad phi.
/// Gradient transfer: on each flat triangle t of the support of f_m the surface divergence
/// theorem gives int_t f_m . grad phi dS = oint_dt phi f_m . nu dl - int_t phi div_s f_m dS
/// (nu the outward in-plane edge normal; only the tangential part of grad phi enters because
/// f_m is tangential). f_m . nu vanishes on the two outer edges of each support triangle and
/// is +1 on the shared edge seen from T^+ and -1 seen from T^- (rwg.hpp), so the edge terms of
/// T^+ and T^- cancel (phi is continuous). This needs no closed surface and holds for folded
/// pairs. Hence <f_m, grad phi> = -<div f_m, phi> and
///   L_mn = j w mu int int f_m . f_n G + (1 / (j w eps)) int int (div f_m)(div' f_n) G.
///
/// K block. Project convention (docs/03): K X(r) = int grad' G(r, r') x X(r') dS' with the
/// gradient taken with respect to the SOURCE point, grad' G = -grad_r G = (1 + jkR) G (r - r')
/// / R^2 (kernels::grad_green returns grad_r G). With it the scattered fields are
/// E_i^s = -L_i J_i + K_i M_i and H_i^s = -K_i J_i - (1/eta_i^2) L_i M_i, and
///   K_mn = int int f_m(r) . (grad' G x f_n(r')) dS' dS = -int int f_m . (grad_r G x f_n).
///
/// Per outer point r the inner moments
///   Phi0 = int G dS',  Phi1 = int G (r' - r) dS',  Psi = int grad' G dS'
/// suffice: int G f_n dS' = (D_n / 2)(Phi1 + (r - p_n) Phi0), and since grad' G is parallel
/// to r - r', grad' G x (r' - r) = 0 pointwise, so
///   int grad' G x f_n dS' = (D_n / 2) Psi x (r - p_n)                      (exact identity)
/// and K_mn = int f_m(r) . [(D_n / 2) Psi x (r - p_n)] dS. The moments are linear in the
/// kernel, so a kernel split into parts may integrate each part with its own outer rule.
///
/// Near and far pairs: Dunavant rules of one degree d for both integrals, plain kernel (the
/// triangles do not touch, R > 0). d is chosen per pair (OperatorOptions::target_accuracy,
/// default 1e-5; touching pairs are not governed by it, see quad_degree_sing):
/// the smallest degree of the ladder quad_degree_far, then the positive-interior degrees up to
/// quad_degree_near (near-class pairs: positive-interior degrees only), whose estimated relative
/// block error E_d <= target_accuracy, with the empirical model
///   log10 E_d = a_d + 0.3 + p_d log10(kappa),   kappa = sqrt((h / D)^2 + (0.15 |k| h)^2),
/// D the centroid distance, h the larger longest edge of the pair, (a_d, p_d) per degree in
/// kDegreeModel below. Calibration: max(|L - L_ref|/|L_ref|, |K - K_ref|/|K_ref|) for 2 x 11 x 7
/// x 3 x 6 pairs (regular and randomly jittered unit icospheres n = 3, D/h bins 1.0 .. 16,
/// |k| h in {0.1, 0.25, 0.5, 1, 1.5, 2, 3}, vacuum / Si / Ag at 500 nm, reference: composite
/// degree-20 rule on 9 or 16 sub-triangles); per degree the line a_d + p_d x is the tightest
/// upper envelope of the points above 1e-13 (least-squares gap among all upper envelopes),
/// beta = 0.15 minimises the cost over the targets 1e-3 .. 1e-10, and the margin 0.3 (a factor
/// 2) covers pairs outside the calibration set. Non-positive-interior degrees 7, 11, 15, 16,
/// 18, 20 (possible only as quad_degree_far) use the model of the next lower calibrated degree.
/// The estimate is increasing in kappa for every d, so the chosen degree is non-decreasing in
/// |k| and h and non-increasing in D. It depends only on symmetric quantities, so both
/// orderings of a pair use the same rule (blocks symmetric up to rounding).
///
/// Touching pairs (identical, shared_edge, shared_vertex): inner integral by singularity
/// subtraction (ADR 0004):
///   exp(-jkR)/R = rem(R) + 1/R - (k^2/2) R,  rem(R) = [exp(-jkR) - 1 + k^2 R^2 / 2] / R,
///   grad' exp(-jkR)/R = grad' rem + grad'(1/R) - (k^2/2) grad' R.
/// rem and grad' rem are integrated with the degree-quad_degree_sing rule on T'; the static
/// terms analytically with static_integrals(r, T') (singularity.hpp):
///   int 1/R = I_1R,  int (r' - r)/R = I_rho_R,  int R = I_R,  int (r' - r) R = I_rhoR,
///   int grad'(1/R) = I_grad                      (grad'(1/R) = (r - r')/R^3),
///   int grad' R = int (r' - r)/R = I_rho_R.
/// With f_n = (D_n / 2)(r' - p_n) = (D_n / 2)[(r' - r) + (r - p_n)] this gives the four
/// subtraction identities
///   int f_n / R          = (D_n / 2)[I_rho_R + (r - p_n) I_1R],
///   int f_n R            = (D_n / 2)[I_rhoR  + (r - p_n) I_R],
///   int grad'(1/R) x f_n = (D_n / 2) I_grad  x (r - p_n),
///   int grad' R    x f_n = (D_n / 2) I_rho_R x (r - p_n),
/// the last two because grad'(1/R) and grad' R are parallel to (r' - r) and
/// (r' - r) x (r' - r) = 0. For identical (coplanar) triangles I_grad is the in-plane
/// principal value (normal part 0), which is the principal value of K; the jump term is
/// separate (jump_block).
///
/// Remainder near R = 0 (x = jkR; k^2 R^2 = -x^2):
///   rem(R)     = jk (-1 + sum_{n>=3} (-1)^n x^(n-1) / n!)                 -> -jk,
///   grad' rem  = [1 - (1 + x) e^(-x) - x^2/2] / R^3 (r' - r)
///              = (jk)^3 sum_{n>=3} (-1)^n (n - 1) x^(n-3) / n! (r' - r)   -> 0,
/// (rem'(R) = [1 - (1 + x) e^(-x) - x^2/2] / R^2 and grad' R = (r' - r)/R; the bracket's
/// series starts at x^3: 1 - (1 + x) e^(-x) = sum_{n>=2} (-1)^n (n-1) x^n / n!).
/// For |x| < 0.5 both are evaluated with these series (Horner, terms through n = 20, relative
/// truncation error < 1e-19); otherwise with the closed forms (cancellation error at
/// |x| = 0.5 about 5e-15 relative). The series branch never divides by R, so it also covers
/// R = 0 exactly: for identical triangles the outer and inner rules may share points (the
/// "outer points never hit inner points" argument holds only for distinct triangles), and
/// there the remainder takes its limit (-jk, gradient 0).
///
/// Graded outer rule (WP7b, outer_grading_levels >= 1). As a function of the outer point r the
/// analytic part is only finitely smooth at the singular set (r log r terms of the potentials
/// and log r of the tangential I_grad at the shared edge / vertex, and at all three edges of an
/// identical triangle), which limits a single Dunavant outer rule to ~2e-4 (L) and ~1e-2 of the
/// K block (WP7). The test triangle is therefore split into Duffy sub-triangles with the apex at
/// a singular vertex and one side on a singular edge:
///  * identical: the six triangles (V_i, M_i, O) and (V_{i+1}, M_i, O), M_i the midpoint of
///    edge i, O the centroid (all three edges singular, symmetric under vertex permutations);
///  * shared edge AB (third vertex C, M the midpoint of AB): (A, M, C) and (B, M, C);
///  * shared vertex A: the whole triangle (A, B, C).
/// Sub-triangle (P0, P1, P2) is parametrised as r = P0 + u [(P1 - P0) + v (P2 - P1)], dS =
/// 2 A_sub u du dv. With these coordinates the distance of r from the edge line P0 P1 is
/// u v |(P2 - P1) x e| (e the unit edge direction) and from P0 it is u |...|, and the edge
/// integrals of static_integrals separate exactly into log u + log v + analytic terms; the
/// integrand is a(u, v) + b(u, v) log u + c(u, v) log v with a, b, c analytic. Both directions
/// therefore use the n-point generalised Gaussian quadrature on [0, 1] that is exact for
/// x^j and x^j log x, j = 0 .. n - 1 (Ma, Rokhlin & Wandzura, SIAM J. Numer. Anal. 33 (1996)
/// 971), except the angular direction v of a shared vertex (analytic: Gauss-Legendre).
/// Nodes and weights (kLogGauss*) were computed for this work package by Newton's method on the
/// 2n moment equations sum_i w_i P_j(2 x_i - 1) {1, log x_i} = int_0^1 P_j(2x - 1) {1, log x}
/// (shifted Legendre basis) in 60-digit arithmetic (mpmath), started from Gauss-Legendre
/// nodes mapped by x = s^2; moment residual below 1e-60; written with 21 significant digits.
/// The table construction re-checks sum w = 1 and sum w log x = -1 to 1e-14.
/// Points per direction (level l = outer_grading_levels): n = 2 l for identical triangles (only
/// L, whose analytic part is of r log r type), n = 4 + 2 l for shared edges (the K integrand of
/// folded pairs is log-singular along the whole edge) and n = 2 + 2 l for shared vertices;
/// total outer points of the analytic part (Dunavant degree 10, the WP7 outer rule, has 25):
///   level              1     2     3     4     5     6
///   identical         24    96   216   384   600   864    (6 n^2, n = 2 l)
///   shared edge       72   128   200   288   392   512    (2 n^2, n = 4 + 2 l)
///   shared vertex     16    36    64   100   144   196    (n^2,   n = 2 + 2 l)
/// The smallest node product u v is about 7e-9 (l = 6), far above the 1e-12 h in-plane /
/// on-edge threshold of static_integrals (below it the divergent edge term would be dropped as
/// a finite part: a deeper geometric grading evaluated with static_integrals is off by ~1e-10).
/// Accuracy of the analytic part at the default level 4 (self-convergence against level 6 and
/// comparison with the relative-coordinate reference of the tests): ~3e-9 (identical, n = 8),
/// ~6e-10 (K of folded shared edges, n = 12), ~1e-11 (shared vertices, n = 10) of the block
/// norm, for shared-edge dihedral angles >= ~90 degrees. Sharper folds are near-singular: the
/// far vertex of one triangle approaches the other triangle away from the shared edge, so the
/// outer integrand varies sharply where the edge grading does not refine; the error depends on
/// the triangle shapes (60 degrees: 1.5e-8 to 3.4e-7 at level 4, 3.4e-9 at level 6; 30
/// degrees: 6.8e-8 to 4.5e-5 at level 4, up to 3.7e-6 at level 6). Geometric Gauss-Legendre grading
/// needs ~30 levels (10^4 points) for 1e-10.
///
/// The remainder of the graded path uses a Dunavant outer rule (it is far smoother: rem has
/// R^2 and R^3 terms, the analytic part carries all of 1/R and R): degree quad_degree_sing for
/// shared edges / vertices; for identical triangles the smallest positive-interior degree
/// >= quad_degree_sing + 3 (13 for the default 10; 17 for 19), so that outer and inner points
/// differ: with the same rule the C^2 kink of the R^3 term at r' = r is sampled at every outer
/// point (L error 3e-8 at |k| h = 0.78 with 10/10, 2e-9 with 12/10, 8e-11 with 13/10; 2e-7,
/// 2e-7 and 3e-9 at |k| h = 2.3). The gradient remainder additionally has its first non-smooth term
///   (jk)^3 3 a_4 x (r' - r) = (k^4 / 8) R (r' - r)
/// subtracted and integrated analytically, int R (r' - r) = I_rhoR:
///   Psi += (k^4 / 8) I_rhoR,  grad' rem -> grad' rem - (k^4 / 8) R (r' - r)
/// (series: the n = 4 coefficient dropped; closed form: minus (k^4 / 8) R); its next non-smooth
/// term is O((kR)^6 / 144). This reduces the outer-rule error of the K remainder from ~5e-8 to
/// ~1e-11 (folded shared edges, |k| h = 0.78). The L remainder keeps its R^3 term (no analytic
/// int R^3 in singularity.hpp): its Dunavant outer error is ~2e-9 for identical triangles and
/// ~1e-9 for shared edges at |k| h = 0.78, growing like (|k| h)^4.
///
/// Symmetry of touching blocks. Graded path: identical L blocks are always replaced by
/// (L + L^T) / 2 (no extra cost). For shared edges and vertices the blocks of the two orderings
/// differ only by the quadrature errors (< 1e-9 of the block norm for |k| h <= 1), so they are
/// averaged over both orderings only if |k| h > symmetrize_touching_above_kh (default 1): for
/// coarse meshes the
/// remainder is no longer smooth on the triangle scale (unaveraged, the PMCHWT matrix of the Ag
/// icosphere n = 1 at 500 nm, |k| h ~ 12, is asymmetric by 1.5e-5) and the averaging keeps the
/// Galerkin matrix exactly symmetric at twice the cost. WP7 path (outer_grading_levels = 0): a
/// single ordering is accurate only to the outer-rule error (about 1e-4 relative for L and 1e-2 for
/// the small K block of a folded shared-edge pair at degree 10), so shared_edge and shared_vertex
/// blocks are the average of both orderings, B = (B_raw(t1, t2) + B_raw(t2, t1)^T) / 2, which costs
/// a second evaluation but makes them exactly (bitwise) symmetric; identical L blocks are replaced
/// by (L + L^T) / 2 (no extra cost).
///
/// K of coplanar touching pairs (identical triangles, and shared-edge / shared-vertex pairs
/// with parallel normals) is set to exactly zero: f_m(r), grad' G (parallel to r - r') and
/// f_n(r') all lie in the common plane, so the triple product vanishes pointwise and the
/// principal value is 0; the computed sum would be rounding noise.
///
/// Input checks: the outer points of a positive-interior rule and of the graded rules lie
/// strictly inside T, so for a valid, non-overlapping mesh they never lie on the boundary of
/// T' (touching pairs share only vertices and edges) and static_integrals cannot report
/// grad_finite_part; a set flag means an invalid mesh (std::invalid_argument). Non-touching
/// pairs whose triangles geometrically intersect (an invalid mesh, e.g. duplicated vertices)
/// can produce R = 0 and non-finite entries; the blocks are checked with allFinite() at the end
/// of every call.
#include "specklebem/kernels/operators.hpp"

#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/kernels/singularity.hpp"

#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace specklebem::kernels {

namespace {

constexpr Real kInv4Pi = 1.0 / (4.0 * constants::pi);
constexpr int kMaxDegree = 20;
/// Largest Dunavant rule (degree 20).
constexpr std::size_t kMaxPoints = 79;
/// |kR| below which the remainder is evaluated by its Taylor series.
constexpr Real kSeriesLimit = 0.5;
/// Highest power index n of the series (terms n = 3 .. kSeriesLast).
constexpr int kSeriesLast = 20;
constexpr Complex kJ{0.0, 1.0};
/// |n_test x n_src| below which a touching pair counts as coplanar (K set to zero).
constexpr Real kCoplanarTol = 1e-12;
/// Highest outer_grading_levels.
constexpr int kMaxGradingLevel = 6;
/// Largest Gauss-Legendre table of the fold-adaptive rule.
constexpr int kMaxLinePoints = 16;

/// a_n = (-1)^n / n! for n = 3 .. kSeriesLast, stored at index n - 3.
constexpr std::array<Real, kSeriesLast - 2> series_coefficients() {
    std::array<Real, kSeriesLast - 2> a{};
    Real fact = 1.0;
    for (int n = 1; n <= kSeriesLast; ++n) {
        fact *= static_cast<Real>(n);
        if (n >= 3) {
            a[static_cast<std::size_t>(n - 3)] = ((n % 2 == 0) ? 1.0 : -1.0) / fact;
        }
    }
    return a;
}
constexpr std::array<Real, kSeriesLast - 2> kSeriesA = series_coefficients();

/// (n - 1) a_n (gradient series), with the n = 4 term optionally dropped (graded path: that
/// term, (k^4 / 8) R (r' - r), is integrated analytically).
constexpr std::array<Real, kSeriesLast - 2> gradient_coefficients(bool drop_n4) {
    std::array<Real, kSeriesLast - 2> b{};
    for (int n = 3; n <= kSeriesLast; ++n) {
        const auto i = static_cast<std::size_t>(n - 3);
        b[i] = (drop_n4 && n == 4) ? 0.0 : static_cast<Real>(n - 1) * kSeriesA[i];
    }
    return b;
}
constexpr std::array<Real, kSeriesLast - 2> kSeriesB = gradient_coefficients(false);
constexpr std::array<Real, kSeriesLast - 2> kSeriesBGraded = gradient_coefficients(true);

/// Positive-interior flags per degree (index 1..20), built once from the rule tables.
const std::array<bool, kMaxDegree + 1>& positive_interior_table() {
    static const std::array<bool, kMaxDegree + 1> table = [] {
        std::array<bool, kMaxDegree + 1> t{};
        for (int d = 1; d <= kMaxDegree; ++d) {
            t[static_cast<std::size_t>(d)] = triangle_rule_is_positive_interior(d);
        }
        return t;
    }();
    return table;
}

bool is_positive_interior(int degree) {
    return positive_interior_table()[static_cast<std::size_t>(degree)];
}

// ---------------------------------------------------------------------------------------------
// Degree selection for near and far pairs (file comment).
// ---------------------------------------------------------------------------------------------

/// Error model log10 E_d = a + kModelMargin + p log10(kappa), index d = 1..20 (index 0 unused).
/// Calibrated degrees: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19; the others copy the next
/// lower calibrated degree.
struct DegreeModel {
    Real a;
    Real p;
};
constexpr std::array<DegreeModel, kMaxDegree + 1> kDegreeModel{{
    {0.0, 0.0},      // (unused)
    {0.460, 2.0},    // 1
    {-0.757, 2.8},   // 2
    {-0.084, 4.1},   // 3
    {-1.486, 4.8},   // 4
    {-1.110, 6.6},   // 5
    {-2.953, 7.2},   // 6
    {-2.953, 7.2},   // 7  (as 6)
    {-3.580, 8.7},   // 8
    {-4.207, 9.8},   // 9
    {-4.821, 12.2},  // 10
    {-4.821, 12.2},  // 11 (as 10)
    {-5.633, 15.0},  // 12
    {-5.783, 16.3},  // 13
    {-6.279, 17.9},  // 14
    {-6.279, 17.9},  // 15 (as 14)
    {-6.279, 17.9},  // 16 (as 14)
    {-7.318, 23.9},  // 17
    {-7.318, 23.9},  // 18 (as 17)
    {-8.502, 28.1},  // 19
    {-8.502, 28.1},  // 20 (as 19)
}};
/// Safety margin of the error model (decades).
constexpr Real kModelMargin = 0.3;
/// Weight of |k| h in kappa.
constexpr Real kModelBeta = 0.15;

/// Degree of the plain double Dunavant rule for a near or far pair (file comment).
int select_degree(Proximity prox, Real distance, Real size, Real abs_k,
                  const OperatorOptions& opt) {
    if (opt.target_accuracy == 0.0) {
        return prox == Proximity::near ? opt.quad_degree_near : opt.quad_degree_far;
    }
    const Real inv = size / distance;  // distance > 0 for a valid mesh (else inf: upper bound)
    const Real kh = kModelBeta * abs_k * size;
    const Real log_kappa = 0.5 * std::log10(inv * inv + kh * kh);
    const Real log_target = std::log10(opt.target_accuracy);
    for (int d = opt.quad_degree_far; d < opt.quad_degree_near; ++d) {
        const bool allowed =
            is_positive_interior(d) || (prox == Proximity::far && d == opt.quad_degree_far);
        if (!allowed) {
            continue;
        }
        const DegreeModel& m = kDegreeModel[static_cast<std::size_t>(d)];
        if (m.a + kModelMargin + m.p * log_kappa <= log_target) {
            return d;
        }
    }
    return opt.quad_degree_near;
}

// ---------------------------------------------------------------------------------------------
// Generalised Gauss-Legendre-log rules on [0, 1] and the graded outer tables (file comment).
// ---------------------------------------------------------------------------------------------

struct LineNode {
    Real x;
    Real w;
};

// clang-format off
constexpr std::array<LineNode, 2> kLogGauss2{{
    {8.82968651376530117596e-2, 2.98499893705524914709e-1},
    {6.75186490909887201036e-1, 7.01500106294475085292e-1},
}};
constexpr std::array<LineNode, 4> kLogGauss4{{
    {1.18025909978449182649e-2, 4.33910287784143911019e-2},
    {1.42825679977483695137e-1, 2.40452097659460675978e-1},
    {4.89201522654574478719e-1, 4.21403452259775931979e-1},
    {8.78679974069183702808e-1, 2.94753421302349000941e-1},
}};
constexpr std::array<LineNode, 6> kLogGauss6{{
    {3.02580213754625870973e-3, 1.13513388172726094405e-2},
    {4.09782541559506150535e-2, 7.52410699549165229174e-2},
    {1.70863295526877294725e-1, 1.88790041615416354610e-1},
    {4.13255708844793247666e-1, 2.85820721827227311987e-1},
    {7.09095146790628543950e-1, 2.84486427891408800045e-1},
    {9.38239590377167091355e-1, 1.54310399893758401001e-1},
}};
constexpr std::array<LineNode, 8> kLogGauss8{{
    {1.09069394192182289416e-3, 4.12430118519834301986e-3},
    {1.54406535463740903566e-2, 2.92703796746872961161e-2},
    {6.94348621007021558961e-2, 8.31140674531700035696e-2},
    {1.87443244255437031474e-1, 1.53721670342287740450e-1},
    {3.73304421343093006877e-1, 2.13497609522260855377e-1},
    {6.00494013699397216008e-1, 2.31870272443575045639e-1},
    {8.16877339734666626460e-1, 1.90536239040367739008e-1},
    {9.62839759269447967977e-1, 9.38654603384529768202e-2},
}};
constexpr std::array<LineNode, 10> kLogGauss10{{
    {4.82961710689629494318e-4, 1.83340007378984497434e-3},
    {6.98862921431576529132e-3, 1.34531223459917893839e-2},
    {3.26113965946776287364e-2, 4.04971943169583332822e-2},
    {9.28257573891659575430e-2, 8.18223696589036061607e-2},
    {1.98327256895403795245e-1, 1.29192342770137539143e-1},
    {3.48880142979353193431e-1, 1.69545319547258747176e-1},
    {5.30440555787956077359e-1, 1.89100216532995609224e-1},
    {7.16764648511655085127e-1, 1.77965753961470550852e-1},
    {8.75234557506233568190e-1, 1.33724770615461519764e-1},
    {9.75245698684392870312e-1, 6.28655101770324600398e-2},
}};
constexpr std::array<LineNode, 12> kLogGauss12{{
    {2.45284264977222214487e-4, 9.33199883067142806310e-4},
    {3.59369802021369158495e-3, 6.97749591514371598553e-3},
    {1.71229259515961437042e-2, 2.16946890219985339772e-2},
    {5.02056123231881938457e-2, 4.59706943955911246927e-2},
    {1.11523585557362564422e-1, 7.75270386958317845884e-2},
    {2.06003002905123975875e-1, 1.11252518183739606826e-1},
    {3.32462697513606540097e-1, 1.40273106008583333229e-1},
    {4.82606700965931942548e-1, 1.57517130086829627554e-1},
    {6.41679470123992858986e-1, 1.57408342223648975972e-1},
    {7.90709087183355406866e-1, 1.37283832717785858586e-1},
    {9.09883828765562592120e-1, 9.81966954741895006788e-2},
    {9.82347937715761951022e-1, 4.49652573935907951039e-2},
}};
constexpr std::array<LineNode, 14> kLogGauss14{{
    {1.37368615004048954645e-4, 5.23345503822354844483e-4},
    {2.02818814037332207546e-3, 3.95878812686564098288e-3},
    {9.78936322050597623346e-3, 1.25554955935815806413e-2},
    {2.92325939902320059622e-2, 2.73813983013757302372e-2},
    {6.65029096807724709007e-2, 4.80077023917983979767e-2},
    {1.26546239664817348591e-1, 7.24860348412805376618e-2},
    {2.11701102527141655464e-1, 9.76176769699333175954e-2},
    {3.20689687524534811308e-1, 1.19471593887773627230e-1},
    {4.48215447838641500095e-1, 1.34061371560476161369e-1},
    {5.85267922747664527196e-1, 1.38061387262852473335e-1},
    {7.20109711024003528890e-1, 1.29434115290680244029e-1},
    {8.39797854076572052886e-1, 1.07855144482654662925e-1},
    {9.31995078245048568646e-1, 7.48594992983696027671e-2},
    {9.86784901324682133464e-1, 3.37264464885356684065e-2},
}};
constexpr std::array<LineNode, 16> kLogGauss16{{
    {8.27730923471450167846e-5, 3.15635101233280747640e-4},
    {1.22837570081736263523e-3, 2.40602715976104285321e-3},
    {5.97987153820187097142e-3, 7.73114869950849988220e-3},
    {1.80736966997729470946e-2, 1.71796853829647399302e-2},
    {4.17672109064258006827e-2, 3.08841408546478166258e-2},
    {8.10386127210058753819e-2, 4.81523211797444418232e-2},
    {1.38778792065167916948e-1, 6.75209108078691783110e-2},
    {2.16096369380672139065e-1, 8.69258704136187106297e-2},
    {3.11842823044589748634e-1, 1.03967218680519119560e-1},
    {4.22435041903797602967e-1, 1.16232215907694096158e-1},
    {5.42011905327487516981e-1, 1.21632414911149285687e-1},
    {6.62915226033054369022e-1, 1.18707666438879215419e-1},
    {7.76440025124993766776e-1, 1.06854337358897308573e-1},
    {8.73761037106673509464e-1, 8.64452992371116633466e-2},
    {9.46917469831238252483e-1, 5.88256095143840291902e-2},
    {9.89739188200742595393e-1, 2.62194983520175712632e-2},
}};
// clang-format on

std::span<const LineNode> log_gauss_rule(int n) {
    switch (n) {
        case 2:
            return kLogGauss2;
        case 4:
            return kLogGauss4;
        case 6:
            return kLogGauss6;
        case 8:
            return kLogGauss8;
        case 10:
            return kLogGauss10;
        case 12:
            return kLogGauss12;
        case 14:
            return kLogGauss14;
        case 16:
            return kLogGauss16;
        default:
            throw std::logic_error("log_gauss_rule: no table for n = " + std::to_string(n));
    }
}

/// Graded outer table of one (class, level): barycentric coordinates with respect to the
/// reordered test triangle (A, B, C) (shared vertices first) and weights as fractions of its
/// area (sum 1).
struct GradedNode {
    Real l0, l1, l2;
    Real w;
};

enum class GradedKind : std::size_t { identical = 0, shared_edge = 1, shared_vertex = 2 };

/// Points per direction of the graded rule (file comment): identical 2 l (L only, r log r type
/// singularity), shared edge 4 + 2 l (log r of the K kernel along the whole edge), shared
/// vertex 2 + 2 l (point singularity).
int graded_points_per_direction(GradedKind kind, int level) {
    switch (kind) {
        case GradedKind::identical:
            return 2 * level;
        case GradedKind::shared_edge:
            return 4 + 2 * level;
        case GradedKind::shared_vertex:
            break;
    }
    return 2 + 2 * level;
}

using Bary = std::array<Real, 3>;

/// Appends the Duffy tensor rule of sub-triangle (p0, p1, p2) (barycentric corners) with
/// relative area `area_fraction`: r = p0 + u [(p1 - p0) + v (p2 - p1)], weight 2 a u w_u w_v.
void append_duffy(const Bary& p0, const Bary& p1, const Bary& p2, Real area_fraction,
                  std::span<const LineNode> ru, std::span<const LineNode> rv,
                  std::vector<GradedNode>& out) {
    for (const LineNode& nu : ru) {
        for (const LineNode& nv : rv) {
            std::array<Real, 3> l{};
            for (std::size_t c = 0; c < 3; ++c) {
                l[c] = p0[c] + nu.x * ((p1[c] - p0[c]) + nv.x * (p2[c] - p1[c]));
            }
            out.push_back({l[0], l[1], l[2], 2.0 * area_fraction * nu.x * nu.w * nv.w});
        }
    }
}

std::vector<GradedNode> build_graded_table(GradedKind kind, int level) {
    const int n = graded_points_per_direction(kind, level);
    const std::span<const LineNode> lg = log_gauss_rule(n);
    std::vector<GradedNode> out;
    const Bary a{1.0, 0.0, 0.0};
    const Bary b{0.0, 1.0, 0.0};
    const Bary c{0.0, 0.0, 1.0};
    const auto mid = [](const Bary& p, const Bary& q) {
        return Bary{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
    };
    switch (kind) {
        case GradedKind::identical: {
            const Bary o{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
            const std::array<Bary, 3> v{a, b, c};
            for (std::size_t i = 0; i < 3; ++i) {
                const Bary m = mid(v[i], v[(i + 1) % 3]);
                append_duffy(v[i], m, o, 1.0 / 6.0, lg, lg, out);
                append_duffy(v[(i + 1) % 3], m, o, 1.0 / 6.0, lg, lg, out);
            }
            break;
        }
        case GradedKind::shared_edge: {
            const Bary m = mid(a, b);
            append_duffy(a, m, c, 0.5, lg, lg, out);
            append_duffy(b, m, c, 0.5, lg, lg, out);
            break;
        }
        case GradedKind::shared_vertex: {
            const LineRule g = gauss_legendre(n);
            std::vector<LineNode> gl(g.nodes.size());
            for (std::size_t i = 0; i < gl.size(); ++i) {
                gl[i] = {0.5 * (g.nodes[i] + 1.0), 0.5 * g.weights[i]};
            }
            append_duffy(a, b, c, 1.0, lg, gl, out);
            break;
        }
    }
    return out;
}

/// All graded tables, built once (thread-safe static initialisation). The log-Gauss tables
/// are re-checked against their first two moments (transcription guard).
struct GradedTables {
    std::array<std::array<std::vector<GradedNode>, kMaxGradingLevel + 1>, 3> nodes;
    /// Gauss-Legendre rules on [0, 1], index n = 1 .. kMaxLinePoints (fold-adaptive rule).
    std::array<std::vector<LineNode>, kMaxLinePoints + 1> gauss;
    GradedTables() {
        for (int n = 1; n <= kMaxLinePoints; ++n) {
            const LineRule g = gauss_legendre(n);
            std::vector<LineNode>& out = gauss[static_cast<std::size_t>(n)];
            out.resize(g.nodes.size());
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = {0.5 * (g.nodes[i] + 1.0), 0.5 * g.weights[i]};
            }
        }
        for (const int n : {2, 4, 6, 8, 10, 12, 14, 16}) {
            // All 2n moments: int_0^1 x^j = 1 / (j + 1), int_0^1 x^j log x = -1 / (j + 1)^2.
            for (int j = 0; j < n; ++j) {
                Real s0 = 0.0;
                Real s1 = 0.0;
                for (const LineNode& q : log_gauss_rule(n)) {
                    const Real xj = std::pow(q.x, j);
                    s0 += q.w * xj;
                    s1 += q.w * xj * std::log(q.x);
                }
                const Real jp = static_cast<Real>(j + 1);
                if (std::abs(s0 - 1.0 / jp) > 1e-14 || std::abs(s1 + 1.0 / (jp * jp)) > 1e-14) {
                    throw std::logic_error("operators.cpp: corrupted log-Gauss table, n = " +
                                           std::to_string(n) + ", moment " + std::to_string(j) +
                                           ": " + std::to_string(s0 - 1.0 / jp) + ", " +
                                           std::to_string(s1 + 1.0 / (jp * jp)));
                }
            }
        }
        for (const GradedKind kind :
             {GradedKind::identical, GradedKind::shared_edge, GradedKind::shared_vertex}) {
            for (int level = 1; level <= kMaxGradingLevel; ++level) {
                nodes[static_cast<std::size_t>(kind)][static_cast<std::size_t>(level)] =
                    build_graded_table(kind, level);
            }
        }
    }
};

const GradedTables& graded_tables() {
    static const GradedTables tables;
    return tables;
}

std::span<const GradedNode> graded_table(GradedKind kind, int level) {
    return graded_tables().nodes[static_cast<std::size_t>(kind)][static_cast<std::size_t>(level)];
}

std::span<const LineNode> gauss01_rule(int n) {
    return graded_tables().gauss[static_cast<std::size_t>(n)];
}

// ---------------------------------------------------------------------------------------------
// Geometry, slots and quadrature points.
// ---------------------------------------------------------------------------------------------

/// Plain 3-vectors for the hot loops: element_blocks is called O(N) to O(N^2) times, and plain
/// scalar code keeps it fast also in the unoptimised sanitizer build (Eigen expressions of
/// fixed size 3 are not inlined at -O0).
struct P3 {
    Real x, y, z;
};
struct C3 {
    Complex x, y, z;
};

P3 to_p3(const Vec3& v) {
    return {v(0), v(1), v(2)};
}

/// RWG functions supported on one triangle: f_a(r) = (div[a] / 2)(r - p[a]).
struct Slots {
    std::size_t count = 0;
    std::array<Real, 3> div{};
    std::array<P3, 3> p{};
};

Vec3 vertex(const geometry::TriangleMesh& mesh, Index v) {
    return mesh.vertices().row(v).transpose();
}

/// Throws std::out_of_range for an invalid t (via support()).
Slots slots_of(const basis::RwgSpace& space, Index t) {
    const basis::RwgSpace::Support s = space.support(t);
    Slots out;
    out.count = static_cast<std::size_t>(s.count);
    for (std::size_t a = 0; a < out.count; ++a) {
        const Index n = s.n[a];
        out.div[a] = space.divergence(n, t);
        out.p[a] = to_p3(vertex(
            space.mesh(), s.sign[a] > 0 ? space.plus_free_vertex(n) : space.minus_free_vertex(n)));
    }
    return out;
}

struct TriangleGeometry {
    Vec3 v0, v1, v2;
    Real area;
};

TriangleGeometry triangle_geometry(const geometry::TriangleMesh& mesh, Index t) {
    const Triangles& tri = mesh.triangles();
    return {vertex(mesh, tri(t, 0)), vertex(mesh, tri(t, 1)), vertex(mesh, tri(t, 2)),
            mesh.area(t)};
}

Real longest_edge(const TriangleGeometry& g) {
    return std::sqrt(std::max(
        {(g.v1 - g.v0).squaredNorm(), (g.v2 - g.v1).squaredNorm(), (g.v0 - g.v2).squaredNorm()}));
}

/// Quadrature points mapped to a triangle; weights include the area. Fixed capacity.
struct Points {
    std::size_t count = 0;
    std::array<P3, kMaxPoints> r;
    std::array<Real, kMaxPoints> w;
};

void map_rule(const TriangleRule& rule, const TriangleGeometry& g, Points& out) {
    out.count = rule.weights.size();
    const P3 a = to_p3(g.v0);
    const P3 b = to_p3(g.v1);
    const P3 c = to_p3(g.v2);
    for (std::size_t q = 0; q < out.count; ++q) {
        const Real* l = rule.barycentric[q].data();
        out.r[q] = {l[0] * a.x + l[1] * b.x + l[2] * c.x, l[0] * a.y + l[1] * b.y + l[2] * c.y,
                    l[0] * a.z + l[1] * b.z + l[2] * c.z};
        out.w[q] = rule.weights[q] * g.area;
    }
}

// ---------------------------------------------------------------------------------------------
// Inner moments.
// ---------------------------------------------------------------------------------------------

/// Inner moments at one outer point r, without the factor 1/(4 pi):
/// phi0 = int e^{-jkR}/R, phi1 = int e^{-jkR}/R (r' - r), psi = int grad'(e^{-jkR}/R).
struct Moments {
    Complex phi0{0.0, 0.0};
    C3 phi1{};
    C3 psi{};
};

/// Plain kernel (non-touching pairs, R > 0), in real arithmetic (the hot loop of every near and
/// far pair; complex products would go through the NaN-checking library multiplication). With
/// k = kr + j ki: e^{-jkR} = e^{ki R} (cos(kr R) - j sin(kr R)), 1 + jkR = (1 - ki R) + j kr R.
Moments plain_moments(const P3& r, const Points& src, Complex k) {
    const Real kr = k.real();
    const Real ki = k.imag();
    const bool lossy = ki != 0.0;
    // Real and imaginary parts: phi0 (re, im), phi1 and psi (x re, x im, y re, ..., z im).
    std::array<Real, 2> s0{};
    std::array<Real, 6> s1{};
    std::array<Real, 6> s2{};
    for (std::size_t q = 0; q < src.count; ++q) {
        const Real dx = src.r[q].x - r.x;  // r' - r
        const Real dy = src.r[q].y - r.y;
        const Real dz = src.r[q].z - r.z;
        const Real R = std::sqrt(dx * dx + dy * dy + dz * dz);
        const Real inv_r = 1.0 / R;
        const Real amp = src.w[q] * inv_r * (lossy ? std::exp(ki * R) : 1.0);
        const Real phase = kr * R;
        const Real gr = amp * std::cos(phase);  // g = w e^{-jkR} / R
        const Real gi = -amp * std::sin(phase);
        // grad' (e^{-jkR}/R) = (1 + jkR) e^{-jkR} / R^3 (r - r') = -(1 + jkR) e/R^3 (r' - r)
        const Real c = 1.0 - ki * R;
        const Real inv_r2 = inv_r * inv_r;
        const Real ggr = -(gr * c - gi * phase) * inv_r2;
        const Real ggi = -(gr * phase + gi * c) * inv_r2;
        s0[0] += gr;
        s0[1] += gi;
        s1[0] += gr * dx;
        s1[1] += gi * dx;
        s1[2] += gr * dy;
        s1[3] += gi * dy;
        s1[4] += gr * dz;
        s1[5] += gi * dz;
        s2[0] += ggr * dx;
        s2[1] += ggi * dx;
        s2[2] += ggr * dy;
        s2[3] += ggi * dy;
        s2[4] += ggr * dz;
        s2[5] += ggi * dz;
    }
    Moments m;
    m.phi0 = {s0[0], s0[1]};
    m.phi1 = {{s1[0], s1[1]}, {s1[2], s1[3]}, {s1[4], s1[5]}};
    m.psi = {{s2[0], s2[1]}, {s2[2], s2[3]}, {s2[4], s2[5]}};
    return m;
}

/// Numerical remainder of the singularity subtraction (touching pairs). `graded`: the
/// (k^4 / 8) R (r' - r) term of the gradient remainder is left out (added analytically by
/// add_static_moments). `need_k` false: psi is not computed (K known to vanish).
Moments remainder_moments(const P3& r, const Points& src, Complex k, bool graded, bool need_k) {
    const Complex jk = kJ * k;
    const Complex jk3 = jk * jk * jk;
    const Complex half_k2 = 0.5 * k * k;
    const Complex k4_8 = 0.125 * (k * k) * (k * k);
    const std::array<Real, kSeriesLast - 2>& b = graded ? kSeriesBGraded : kSeriesB;
    Moments m;
    for (std::size_t q = 0; q < src.count; ++q) {
        const Real dx = src.r[q].x - r.x;  // r' - r
        const Real dy = src.r[q].y - r.y;
        const Real dz = src.r[q].z - r.z;
        const Real R = std::sqrt(dx * dx + dy * dy + dz * dz);
        const Complex x = jk * R;
        Complex rem;  // [e^{-x} - 1 + k^2 R^2 / 2] / R
        Complex gq;   // grad' rem = gq (r' - r)
        if (std::abs(x) < kSeriesLimit) {
            // Horner in x for s1 = sum a_n x^(n-3) and s2 = sum (n - 1) a_n x^(n-3).
            Complex s1{0.0, 0.0};
            for (int n = kSeriesLast; n >= 3; --n) {
                s1 = s1 * x + kSeriesA[static_cast<std::size_t>(n - 3)];
            }
            rem = jk * (x * x * s1 - 1.0);
            if (need_k) {
                Complex s2{0.0, 0.0};
                for (int n = kSeriesLast; n >= 3; --n) {
                    s2 = s2 * x + b[static_cast<std::size_t>(n - 3)];
                }
                gq = jk3 * s2;
            }
        } else {
            const Complex e = std::exp(-x);
            const Complex quad = half_k2 * (R * R);
            rem = (e - 1.0 + quad) / R;
            if (need_k) {
                gq = (1.0 - (1.0 + x) * e + quad) / (R * R * R);
                if (graded) {
                    gq -= k4_8 * R;
                }
            }
        }
        const Complex wr = src.w[q] * rem;
        m.phi0 += wr;
        m.phi1.x += wr * dx;
        m.phi1.y += wr * dy;
        m.phi1.z += wr * dz;
        if (need_k) {
            const Complex wg = src.w[q] * gq;
            m.psi.x += wg * dx;
            m.psi.y += wg * dy;
            m.psi.z += wg * dz;
        }
    }
    return m;
}

/// Adds the analytic static terms at outer point r (graded: plus (k^4 / 8) I_rhoR in psi).
void add_static_moments(const P3& r, const TriangleGeometry& g_src, Complex k, bool graded,
                        Moments& m) {
    const Complex half_k2 = 0.5 * k * k;
    const StaticIntegrals s = static_integrals(Vec3(r.x, r.y, r.z), g_src.v0, g_src.v1, g_src.v2);
    if (s.grad_finite_part) {
        throw std::invalid_argument(
            "element_blocks: outer quadrature point on the boundary of the source triangle "
            "(boundary finite part of the static gradient integral) for a touching pair; this "
            "cannot occur for a valid, non-overlapping mesh");
    }
    m.phi0 += s.I_1R - half_k2 * s.I_R;
    m.phi1.x += s.I_rho_R(0) - half_k2 * s.I_rhoR(0);
    m.phi1.y += s.I_rho_R(1) - half_k2 * s.I_rhoR(1);
    m.phi1.z += s.I_rho_R(2) - half_k2 * s.I_rhoR(2);
    m.psi.x += s.I_grad(0) - half_k2 * s.I_rho_R(0);
    m.psi.y += s.I_grad(1) - half_k2 * s.I_rho_R(1);
    m.psi.z += s.I_grad(2) - half_k2 * s.I_rho_R(2);
    if (graded) {
        const Complex k4_8 = 0.125 * (k * k) * (k * k);
        m.psi.x += k4_8 * s.I_rhoR(0);
        m.psi.y += k4_8 * s.I_rhoR(1);
        m.psi.z += k4_8 * s.I_rhoR(2);
    }
}

// ---------------------------------------------------------------------------------------------
// Outer integration.
// ---------------------------------------------------------------------------------------------

/// Running sums over the outer points (without 1/(4 pi) and the material factors); 3x3
/// blocks stored row-major (index 3 a + b).
struct Accumulator {
    std::array<Complex, 9> vec{};  // int int f_m . f_n e^{-jkR}/R
    std::array<Complex, 9> k{};    // int int f_m . (grad'(e^{-jkR}/R) x f_n)
    Complex scalar{0.0, 0.0};      // int int e^{-jkR}/R
};

/// `need_k` false: the K sums are left at zero (K known to vanish).
void accumulate(const P3& r, Real w, const Moments& mo, const Slots& test, const Slots& src,
                bool need_k, Accumulator& acc) {
    std::array<C3, 3> v{};  // int e^{-jkR}/R f_n dS' = (D_n / 2)(phi1 + (r - p_n) phi0)
    // int grad'(e^{-jkR}/R) x f_n dS' = (D_n / 2) psi x (r - p_n), written out: Eigen's
    // cross() returns conj(a x b) for complex vectors and must not be used here.
    std::array<C3, 3> u{};
    for (std::size_t b = 0; b < src.count; ++b) {
        const Real h = 0.5 * src.div[b];
        const Real qx = r.x - src.p[b].x;
        const Real qy = r.y - src.p[b].y;
        const Real qz = r.z - src.p[b].z;
        v[b] = {h * (mo.phi1.x + mo.phi0 * qx), h * (mo.phi1.y + mo.phi0 * qy),
                h * (mo.phi1.z + mo.phi0 * qz)};
        if (need_k) {
            u[b] = {h * (mo.psi.y * qz - mo.psi.z * qy), h * (mo.psi.z * qx - mo.psi.x * qz),
                    h * (mo.psi.x * qy - mo.psi.y * qx)};
        }
    }
    for (std::size_t a = 0; a < test.count; ++a) {
        const Real s = 0.5 * w * test.div[a];
        const Real fx = s * (r.x - test.p[a].x);
        const Real fy = s * (r.y - test.p[a].y);
        const Real fz = s * (r.z - test.p[a].z);
        for (std::size_t b = 0; b < src.count; ++b) {
            acc.vec[3 * a + b] += fx * v[b].x + fy * v[b].y + fz * v[b].z;
            if (need_k) {
                acc.k[3 * a + b] += fx * u[b].x + fy * u[b].y + fz * u[b].z;
            }
        }
    }
    acc.scalar += w * mo.phi0;
}

/// One triangle of a pair with its RWG slots and geometry.
struct Side {
    const Slots* slots;
    const TriangleGeometry* geom;
};

/// Plain kernel, same Dunavant rule on both triangles (near and far pairs).
Accumulator integrate_plain(const Side& test, const Side& src, const TriangleRule& rule,
                            Complex k) {
    Points outer;
    Points inner;
    map_rule(rule, *test.geom, outer);
    map_rule(rule, *src.geom, inner);
    Accumulator acc;
    for (std::size_t i = 0; i < outer.count; ++i) {
        accumulate(outer.r[i], outer.w[i], plain_moments(outer.r[i], inner, k), *test.slots,
                   *src.slots, true, acc);
    }
    return acc;
}

/// WP7 touching scheme: one Dunavant rule for the outer integral and the remainder.
Accumulator integrate_touching_dunavant(const Side& test, const Side& src, const TriangleRule& rule,
                                        Complex k, bool need_k) {
    Points outer;
    Points inner;
    map_rule(rule, *test.geom, outer);
    map_rule(rule, *src.geom, inner);
    Accumulator acc;
    for (std::size_t i = 0; i < outer.count; ++i) {
        const P3& r = outer.r[i];
        Moments mo = remainder_moments(r, inner, k, false, need_k);
        add_static_moments(r, *src.geom, k, false, mo);
        accumulate(r, outer.w[i], mo, *test.slots, *src.slots, need_k, acc);
    }
    return acc;
}

/// Outer Dunavant degree of the remainder of identical triangles (file comment): the smallest
/// positive-interior degree >= sing + 3, at most 19; 17 for sing = 19 (outer and inner rules
/// must differ).
int identical_remainder_outer_degree(int sing) {
    if (sing >= 17) {
        return sing == 19 ? 17 : 19;
    }
    int d = sing + 3;
    while (!is_positive_interior(d)) {
        ++d;
    }
    return d;
}

/// Vertices of t ordered with those shared with `other` first (cyclic order otherwise kept).
std::array<Vec3, 3> shared_first(const geometry::TriangleMesh& mesh, Index t, Index other) {
    const Triangles& tri = mesh.triangles();
    std::array<Index, 3> order{};
    std::size_t n = 0;
    for (const bool want_shared : {true, false}) {
        for (Index i = 0; i < 3; ++i) {
            const Index v = tri(t, i);
            const bool shared = v == tri(other, 0) || v == tri(other, 1) || v == tri(other, 2);
            if (shared == want_shared) {
                order[n++] = v;
            }
        }
    }
    return {vertex(mesh, order[0]), vertex(mesh, order[1]), vertex(mesh, order[2])};
}

// ---------------------------------------------------------------------------------------------
// Fold-adaptive outer rule of the analytic part (WP7c, file comment).
// ---------------------------------------------------------------------------------------------

/// v grading of a Duffy piece (apex x, opposite side p1 -> p2, r = x + u [(p1 - x) + v (p2 -
/// p1)]): Gauss-Legendre, or log-Gauss towards the side x p1 (v = 0) or x p2 (v = 1).
enum class Grade { none, at_p1, at_p2 };

struct DuffyPiece {
    Vec3 x, p1, p2;
    Grade grade;
};

/// Singular feature through the apex of a piece: the apex itself (point) or a source-triangle
/// edge from the apex (ray along the unit vector d; the shared edge is one of them).
struct Feature {
    Vec3 d;
    bool ray;
};

/// Capacity of a piece list and the split depth per initial piece.
constexpr std::size_t kMaxPieces = 48;
constexpr int kMaxSplitDepth = 8;
/// Folds below 90 degrees: a source vertex projects onto the test plane with barycentric
/// coordinates (with respect to the test triangle, shared vertices first) lambda_C > kFoldTol
/// (shared edge) or lambda_B, lambda_C > kFoldTol (shared vertex: inside the wedge at A).
constexpr Real kFoldTol = 1e-6;
/// Shared edge: the pieces are cut along the projected source edges (rays A S, B S) if
/// lambda_C > kFoldMin and S is at least kNearVertex |AB| away from A and B.
constexpr Real kFoldMin = 0.1;
constexpr Real kNearVertex = 0.2;
/// Smallest distance (in units of the side p1 p2) of the complex zero of a feature from the
/// real interval v in [0, 1]; closer zeros are split off.
constexpr Real kZeroMin = 0.5;
/// A zero within kEndTol of an end point (real and imaginary part) lies on that side: graded.
constexpr Real kEndTol = 1e-3;

struct PieceList {
    std::size_t count = 0;
    std::array<DuffyPiece, kMaxPieces> p;
    std::array<Feature, 4> features{};  ///< features of the current apex
    std::size_t n_features = 0;
    Real min_area = 0.0;  ///< pieces below this area are dropped (degenerate slivers)
    bool changed = false; ///< a piece was split or the partition differs from the WP7b one
};

/// Complex zero v_f + j q (q >= 0) of the squared distance |w|^2 - (w . d)^2 (ray) or |w|^2
/// (point) for w(v) = (p1 - x) + v (p2 - p1), the singularity of the outer integrand along the
/// side. Returns false if the feature has no zero on this side (ray pointing away, or parallel).
bool feature_zero(const Vec3& x, const Vec3& p1, const Vec3& p2, const Feature& f, Real& vf,
                  Real& q) {
    const Vec3 w0 = p1 - x;
    const Vec3 e = p2 - p1;
    Real a = e.squaredNorm();
    Real b = w0.dot(e);
    Real c = w0.squaredNorm();
    if (f.ray) {
        const Real ed = e.dot(f.d);
        const Real wd = w0.dot(f.d);
        a -= ed * ed;
        b -= wd * ed;
        c -= wd * wd;
    }
    if (!(a > 1e-14 * e.squaredNorm())) {
        return false;
    }
    vf = -b / a;
    q = std::sqrt(std::max(a * c - b * b, 0.0)) / a;
    // A ray is singular only on its forward side (the backward extension is not part of the
    // source triangle).
    return !f.ray || (w0 + vf * e).dot(f.d) > 0.0;
}

/// Adds piece (x, p1, p2) for the features of pl (file comment): a zero within kEndTol of an
/// end grades that side (log-Gauss; zeros at both ends: the piece is halved); the zero nearest
/// to [0, 1] at a relative distance below kZeroMin splits the piece (at its real part if that is
/// inside, else at the distance / kZeroMin from the nearer end). `reserve` slots stay free for
/// pieces still to be added by the callers (count + reserve + 1 <= kMaxPieces on entry), so no
/// piece is ever dropped; splitting stops when the list is full or at kMaxSplitDepth.
void add_piece(PieceList& pl, const Vec3& x, const Vec3& p1, const Vec3& p2, int depth,
               std::size_t reserve) {
    if (0.5 * (p1 - x).cross(p2 - x).norm() <= pl.min_area) {
        return;
    }
    bool at0 = false;
    bool at1 = false;
    Real worst = kZeroMin;
    Real split = -1.0;
    for (std::size_t i = 0; i < pl.n_features; ++i) {
        Real vf = 0.0;
        Real q = 0.0;
        if (!feature_zero(x, p1, p2, pl.features[i], vf, q)) {
            continue;
        }
        if (q <= kEndTol && std::abs(vf) <= kEndTol) {
            at0 = true;
            continue;
        }
        if (q <= kEndTol && std::abs(vf - 1.0) <= kEndTol) {
            at1 = true;
            continue;
        }
        const Real dv = vf < 0.0 ? -vf : (vf > 1.0 ? vf - 1.0 : 0.0);
        const Real dist = std::sqrt(dv * dv + q * q);
        if (dist < worst) {
            worst = dist;
            if (vf >= kEndTol && vf <= 1.0 - kEndTol) {
                split = vf;
            } else if (vf < kEndTol) {
                split = std::hypot(vf, q) / kZeroMin;
            } else {
                split = 1.0 - std::hypot(vf - 1.0, q) / kZeroMin;
            }
        }
    }
    const bool can_split = pl.count + reserve + 2 <= kMaxPieces && depth < kMaxSplitDepth;
    if (can_split && (split >= 0.0 || (at0 && at1))) {
        const Real vs = split >= 0.0 ? std::clamp(split, 2.0 * kEndTol, 1.0 - 2.0 * kEndTol) : 0.5;
        const Vec3 s = p1 + vs * (p2 - p1);
        pl.changed = true;
        add_piece(pl, x, p1, s, depth + 1, reserve + 1);
        add_piece(pl, x, s, p2, depth + 1, reserve);
        return;
    }
    pl.p[pl.count++] = {x, p1, p2, at0 ? Grade::at_p1 : (at1 ? Grade::at_p2 : Grade::none)};
}

/// Barycentric coordinates of p (in the plane of a, b, c) with respect to (a, b, c).
std::array<Real, 3> barycentric_of(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& p) {
    const Vec3 e1 = b - a;
    const Vec3 e2 = c - a;
    const Vec3 d = p - a;
    const Real g11 = e1.dot(e1);
    const Real g12 = e1.dot(e2);
    const Real g22 = e2.dot(e2);
    const Real r1 = d.dot(e1);
    const Real r2 = d.dot(e2);
    const Real det = g11 * g22 - g12 * g12;
    const Real lb = (g22 * r1 - g12 * r2) / det;
    const Real lc = (g11 * r2 - g12 * r1) / det;
    return {1.0 - lb - lc, lb, lc};
}

/// Sets the features of apex x: the point itself and the source edges from x to `ends`.
void set_features(PieceList& pl, const Vec3& x, std::initializer_list<Vec3> ends) {
    pl.n_features = 0;
    pl.features[pl.n_features++] = {Vec3::Zero(), false};
    for (const Vec3& e : ends) {
        pl.features[pl.n_features++] = {(e - x).normalized(), true};
    }
}

/// Pieces of the fold-adaptive rule for a touching pair (file comment). abc: test vertices,
/// shared first; far: the source vertices not shared (shared edge: far[0]; shared vertex: both).
/// Returns false if the WP7b table applies unchanged (no split, WP7b partition and grading).
bool fold_pieces(GradedKind kind, const std::array<Vec3, 3>& abc, const std::array<Vec3, 2>& far,
                 PieceList& pl) {
    const Vec3& A = abc[0];
    const Vec3& B = abc[1];
    const Vec3& C = abc[2];
    const Vec3 n = (B - A).cross(C - A).normalized();
    const auto projected = [&](const Vec3& v) {
        return barycentric_of(A, B, C, v - (v - A).dot(n) * n);
    };
    pl.min_area = 1e-9 * 0.5 * (B - A).cross(C - A).norm();
    if (kind == GradedKind::shared_vertex) {
        // Folded (a source vertex projects into the wedge at A): the source edges from A are
        // features; otherwise only the apex (obtuse angles).
        const std::array<Real, 3> l0 = projected(far[0]);
        const std::array<Real, 3> l1 = projected(far[1]);
        const bool folded = (l0[1] > kFoldTol && l0[2] > kFoldTol) ||
                            (l1[1] > kFoldTol && l1[2] > kFoldTol);
        if (folded) {
            set_features(pl, A, {far[0], far[1]});
        } else {
            set_features(pl, A, {});
        }
        add_piece(pl, A, B, C, 0, 0);
        return folded || pl.changed || pl.p[0].grade != Grade::none;
    }
    const Vec3 M = 0.5 * (A + B);
    const std::array<Real, 3> l = projected(far[0]);
    const bool folded = l[2] > kFoldTol;
    const Real la = std::max(l[0], 0.0);
    const Real lb = std::max(l[1], 0.0);
    const Vec3 S = (la * A + lb * B + l[2] * C) / (la + lb + l[2]);
    const Real ab = (B - A).norm();
    if (l[2] <= kFoldMin || std::min((S - A).norm(), (S - B).norm()) < kNearVertex * ab) {
        // WP7b partition; folds >= 90 degrees: features apex and shared edge only.
        if (folded) {
            set_features(pl, A, {B, far[0]});
        } else {
            set_features(pl, A, {B});
        }
        add_piece(pl, A, M, C, 0, 1);
        const std::size_t n_a = pl.count;
        if (folded) {
            set_features(pl, B, {A, far[0]});
        } else {
            set_features(pl, B, {A});
        }
        add_piece(pl, B, M, C, 0, 0);
        return folded || pl.changed || n_a != 1 || pl.count != 2 ||
               pl.p[0].grade != Grade::at_p1 || pl.p[1].grade != Grade::at_p1;
    }
    // Fold below 90 degrees: the source edges A C' and B C' project onto the rays A S and B S
    // (S = P clipped into T along those rays), which become sides of the pieces.
    pl.changed = true;
    set_features(pl, A, {B, far[0]});
    add_piece(pl, A, M, S, 0, 3);
    add_piece(pl, A, S, C, 0, 2);
    set_features(pl, B, {A, far[0]});
    add_piece(pl, B, M, S, 0, 1);
    add_piece(pl, B, S, C, 0, 0);
    return true;
}

/// Points of the fold-adaptive pieces at grading level `level`: radial (log-Gauss) and angular
/// (log-Gauss on graded pieces, else Gauss-Legendre) directions (file comment).
int adaptive_radial_points(int level) {
    return std::min(6 + 2 * level, kMaxLinePoints);
}
int adaptive_angular_points(int level) {
    return 2 + 2 * level;
}

/// Graded touching scheme (file comment). `abc` are the test-triangle vertices reordered with
/// the shared vertices first; `pieces` the fold-adaptive rule of the analytic part, or nullptr
/// for the WP7b table.
Accumulator integrate_touching_graded(const Side& test, const Side& src,
                                      const std::array<Vec3, 3>& abc, GradedKind kind,
                                      const PieceList* pieces, const OperatorOptions& opt,
                                      Complex k, bool need_k) {
    Accumulator acc;
    // Analytic part on the graded rule.
    if (pieces != nullptr) {
        const int n_ang = adaptive_angular_points(opt.outer_grading_levels);
        const std::span<const LineNode> lgu =
            log_gauss_rule(adaptive_radial_points(opt.outer_grading_levels));
        const std::span<const LineNode> lg = log_gauss_rule(n_ang);
        const std::span<const LineNode> gl = gauss01_rule(n_ang);
        for (std::size_t i = 0; i < pieces->count; ++i) {
            const DuffyPiece& pc = pieces->p[i];
            const P3 x = to_p3(pc.x);
            const P3 e1 = to_p3(pc.p1 - pc.x);
            const P3 e2 = to_p3(pc.p2 - pc.p1);
            const Real area2 = (pc.p1 - pc.x).cross(pc.p2 - pc.x).norm();
            const std::span<const LineNode> rv = pc.grade == Grade::none ? gl : lg;
            for (const LineNode& nu : lgu) {
                for (const LineNode& nv : rv) {
                    const Real v = pc.grade == Grade::at_p2 ? 1.0 - nv.x : nv.x;
                    const Real u = nu.x;
                    const P3 r{x.x + u * (e1.x + v * e2.x), x.y + u * (e1.y + v * e2.y),
                               x.z + u * (e1.z + v * e2.z)};
                    Moments mo;
                    add_static_moments(r, *src.geom, k, true, mo);
                    accumulate(r, area2 * u * nu.w * nv.w, mo, *test.slots, *src.slots, need_k,
                               acc);
                }
            }
        }
    } else {
        const P3 a = to_p3(abc[0]);
        const P3 b = to_p3(abc[1]);
        const P3 c = to_p3(abc[2]);
        const Real area = test.geom->area;
        for (const GradedNode& q : graded_table(kind, opt.outer_grading_levels)) {
            const P3 r{q.l0 * a.x + q.l1 * b.x + q.l2 * c.x, q.l0 * a.y + q.l1 * b.y + q.l2 * c.y,
                       q.l0 * a.z + q.l1 * b.z + q.l2 * c.z};
            Moments mo;
            add_static_moments(r, *src.geom, k, true, mo);
            accumulate(r, q.w * area, mo, *test.slots, *src.slots, need_k, acc);
        }
    }
    // Numerical remainder on Dunavant rules.
    const int outer_degree = kind == GradedKind::identical
                                 ? identical_remainder_outer_degree(opt.quad_degree_sing)
                                 : opt.quad_degree_sing;
    Points outer;
    Points inner;
    map_rule(triangle_rule(outer_degree), *test.geom, outer);
    map_rule(triangle_rule(opt.quad_degree_sing), *src.geom, inner);
    for (std::size_t i = 0; i < outer.count; ++i) {
        accumulate(outer.r[i], outer.w[i], remainder_moments(outer.r[i], inner, k, true, need_k),
                   *test.slots, *src.slots, need_k, acc);
    }
    return acc;
}

/// out = (x + y^T) / 2 for the 3x3 parts and the scalar. Element-wise 0.5 (x + y) is
/// commutative, so the averaged block of (t2, t1) is bitwise the transpose.
void average_with_transpose(Accumulator& acc, const Accumulator& rev) {
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            acc.vec[3 * a + b] = 0.5 * (acc.vec[3 * a + b] + rev.vec[3 * b + a]);
            acc.k[3 * a + b] = 0.5 * (acc.k[3 * a + b] + rev.k[3 * b + a]);
        }
    }
    acc.scalar = 0.5 * (acc.scalar + rev.scalar);
}

void symmetrize_vec(Accumulator& acc) {
    const std::array<Complex, 9> raw = acc.vec;
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            acc.vec[3 * a + b] = 0.5 * (raw[3 * a + b] + raw[3 * b + a]);
        }
    }
}

void check_degree(int degree, const char* name, bool positive_interior) {
    if (degree < 1 || degree > kMaxDegree) {
        throw std::invalid_argument(std::string("OperatorOptions: ") + name +
                                    " must be in 1..20, got " + std::to_string(degree));
    }
    if (positive_interior && !is_positive_interior(degree)) {
        throw std::invalid_argument(
            std::string("OperatorOptions: ") + name + " = " + std::to_string(degree) +
            " is not a positive-interior Dunavant rule (ADR 0004; allowed: 1, 2, 4, 5, 6, 8, "
            "9, 10, 12, 13, 14, 17, 19)");
    }
}

}  // namespace

void validate(const OperatorOptions& opt) {
    check_degree(opt.quad_degree_far, "quad_degree_far", false);
    check_degree(opt.quad_degree_near, "quad_degree_near", true);
    check_degree(opt.quad_degree_sing, "quad_degree_sing", true);
    check_degree(opt.quad_degree_rhs, "quad_degree_rhs", true);
    if (opt.quad_degree_far > opt.quad_degree_near) {
        throw std::invalid_argument(
            "OperatorOptions: quad_degree_far (" + std::to_string(opt.quad_degree_far) +
            ") must not exceed quad_degree_near (" + std::to_string(opt.quad_degree_near) +
            "): they are the bounds of the near/far degree selection");
    }
    if (!(std::isfinite(opt.near_distance_factor) && opt.near_distance_factor >= 0.0)) {
        throw std::invalid_argument(
            "OperatorOptions: near_distance_factor must be finite and >= 0");
    }
    if (opt.outer_grading_levels < 0 || opt.outer_grading_levels > kMaxGradingLevel) {
        throw std::invalid_argument("OperatorOptions: outer_grading_levels must be in 0..6, got " +
                                    std::to_string(opt.outer_grading_levels));
    }
    if (!(std::isfinite(opt.target_accuracy) && opt.target_accuracy >= 0.0 &&
          opt.target_accuracy < 1.0)) {
        throw std::invalid_argument(
            "OperatorOptions: target_accuracy must be finite and in [0, 1) (0 = fixed degrees)");
    }
    if (!(opt.symmetrize_touching_above_kh >= 0.0)) {  // also rejects NaN
        throw std::invalid_argument(
            "OperatorOptions: symmetrize_touching_above_kh must be >= 0 (infinity = never)");
    }
}

void element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                    const RegionParams& region, const OperatorOptions& opt,
                    Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K) {
    validate(opt);
    const geometry::TriangleMesh& mesh = space.mesh();
    // support() range-checks both indices (std::out_of_range) before classify is called.
    const Slots test_slots = slots_of(space, t_test);
    const Slots src_slots = slots_of(space, t_src);
    L.setZero();
    K.setZero();
    if (test_slots.count == 0 || src_slots.count == 0) {
        return;
    }
    const Proximity prox = classify(mesh, t_test, t_src, opt.near_distance_factor);
    const bool touching = prox == Proximity::identical || prox == Proximity::shared_edge ||
                          prox == Proximity::shared_vertex;
    const TriangleGeometry g_test = triangle_geometry(mesh, t_test);
    const TriangleGeometry g_src = triangle_geometry(mesh, t_src);
    const Side test{&test_slots, &g_test};
    const Side src{&src_slots, &g_src};

    // K of coplanar touching pairs is exactly zero (file comment); its sums are then skipped.
    const bool k_zero =
        prox == Proximity::identical ||
        (touching && mesh.normal(t_test).cross(mesh.normal(t_src)).norm() <= kCoplanarTol);
    const bool need_k = !k_zero;
    Accumulator acc;
    if (!touching) {
        const Vec3 c_test = (g_test.v0 + g_test.v1 + g_test.v2) / 3.0;
        const Vec3 c_src = (g_src.v0 + g_src.v1 + g_src.v2) / 3.0;
        const int degree = select_degree(prox, (c_test - c_src).norm(),
                                         std::max(longest_edge(g_test), longest_edge(g_src)),
                                         std::abs(region.k), opt);
        acc = integrate_plain(test, src, triangle_rule(degree), region.k);
    } else if (opt.outer_grading_levels == 0) {
        const TriangleRule& rule = triangle_rule(opt.quad_degree_sing);
        acc = integrate_touching_dunavant(test, src, rule, region.k, need_k);
        if (prox == Proximity::identical) {
            symmetrize_vec(acc);
        } else {
            average_with_transpose(acc,
                                   integrate_touching_dunavant(src, test, rule, region.k, need_k));
        }
    } else {
        const GradedKind kind = prox == Proximity::identical     ? GradedKind::identical
                                : prox == Proximity::shared_edge ? GradedKind::shared_edge
                                                                 : GradedKind::shared_vertex;
        // Fold-adaptive pieces of the analytic part (nullptr: the WP7b table).
        const auto touching_pass = [&](const Side& te, const Side& sr, Index tt, Index ts) {
            const std::array<Vec3, 3> abc = shared_first(mesh, tt, ts);
            PieceList pieces;
            bool adaptive = false;
            if (opt.fold_adaptive && kind != GradedKind::identical) {
                const std::array<Vec3, 3> src_abc = shared_first(mesh, ts, tt);
                const std::array<Vec3, 2> far =
                    kind == GradedKind::shared_edge ? std::array<Vec3, 2>{src_abc[2], src_abc[2]}
                                                    : std::array<Vec3, 2>{src_abc[1], src_abc[2]};
                adaptive = fold_pieces(kind, abc, far, pieces);
            }
            return integrate_touching_graded(te, sr, abc, kind, adaptive ? &pieces : nullptr, opt,
                                             region.k, need_k);
        };
        acc = touching_pass(test, src, t_test, t_src);
        const Real kh = std::abs(region.k) * std::max(longest_edge(g_test), longest_edge(g_src));
        if (prox == Proximity::identical) {
            symmetrize_vec(acc);  // free: always (K is zero, the scalar part has one entry)
        } else if (kh > opt.symmetrize_touching_above_kh) {
            average_with_transpose(acc, touching_pass(src, test, t_src, t_test));
        }
    }

    const Complex c_vec = kJ * region.omega * region.mu * kInv4Pi;
    const Complex c_sca = kInv4Pi / (kJ * region.omega * region.eps);
    for (std::size_t a = 0; a < test_slots.count; ++a) {
        const auto ai = static_cast<Eigen::Index>(a);
        for (std::size_t b = 0; b < src_slots.count; ++b) {
            const auto bi = static_cast<Eigen::Index>(b);
            L(ai, bi) = c_vec * acc.vec[3 * a + b] +
                        c_sca * (test_slots.div[a] * src_slots.div[b]) * acc.scalar;
            K(ai, bi) = k_zero ? Complex(0.0, 0.0) : kInv4Pi * acc.k[3 * a + b];
        }
    }
    if (!(L.allFinite() && K.allFinite())) {
        throw std::invalid_argument(
            "element_blocks: non-finite block for triangles " + std::to_string(t_test) + " and " +
            std::to_string(t_src) +
            " (geometrically intersecting or coincident triangles that do not share vertex "
            "indices, or invalid region parameters)");
    }
}

void jump_block(const basis::RwgSpace& space, Index t, Eigen::Matrix<Complex, 3, 3>& I) {
    const Slots s = slots_of(space, t);  // range check
    I.setZero();
    const geometry::TriangleMesh& mesh = space.mesh();
    const Vec3 n = mesh.normal(t);
    const TriangleRule& rule = triangle_rule(2);  // f_m . (n x f_n) is quadratic: exact
    const TriangleGeometry g = triangle_geometry(mesh, t);
    Eigen::Matrix<Real, 3, 3> acc = Eigen::Matrix<Real, 3, 3>::Zero();
    for (std::size_t q = 0; q < rule.weights.size(); ++q) {
        const Vec3& l = rule.barycentric[q];
        const Vec3 r = l(0) * g.v0 + l(1) * g.v1 + l(2) * g.v2;
        const Real w = rule.weights[q] * g.area;
        std::array<Vec3, 3> f;
        for (std::size_t a = 0; a < s.count; ++a) {
            f[a] = (0.5 * s.div[a]) * (r - Vec3(s.p[a].x, s.p[a].y, s.p[a].z));
        }
        for (std::size_t a = 0; a < s.count; ++a) {
            for (std::size_t b = 0; b < s.count; ++b) {
                acc(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) +=
                    w * f[a].dot(n.cross(f[b]));
            }
        }
    }
    I = acc.cast<Complex>();
}

}  // namespace specklebem::kernels
