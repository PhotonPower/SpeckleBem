/// @file stretched_operators.cpp
/// Galerkin element blocks with a complex coordinate stretch (PML; ADR 0009 decisions 2 and 3,
/// WP-PML2). kernels::element_blocks(..., stretch, ...) dispatches here for every pair with at
/// least one stretched triangle; all other pairs keep the unstretched kernels (bitwise).
///
/// Mapping (ADR 0009 decision 2). The mesh stays real; S(r) = (x~(x), y~(y), z) has the diagonal
/// Jacobian DS = diag(s_x, s_y, 1), s = 1 - j sigma (geometry::CoordinateStretch). The RWG
/// functions are carried over by the Piola transform, f~ dS~ = DS f dS and (div~ f~) dS~ =
/// (div f) dS, so the Galerkin integrals keep the real quadrature points, the real RWG functions
/// and their divergences:
///   L_mn = j w mu int int (DS f_m) . (DS' f_n) G(R~) + (1 / (j w eps)) int int (div f_m)(div' f_n)
///   G(R~), K_mn = int int (DS f_m) . (grad~' G x DS' f_n),   grad~' G = (1 + j k R~) G (r~ - r~')
///   / R~^2,
/// G(R~) = exp(-j k R~) / (4 pi R~), R~^2 = (r~ - r~') . (r~ - r~') without conjugation, R~ on the
/// principal branch. For the monotone stretch x~ - x~' = (x - x')(1 - j tau) with the secant tau =
/// (F(x) - F(x')) / (x - x') in [0, sigma_max] (F = Int sigma, odd and non-decreasing), so
/// R~^2 = sum_i dx_i^2 (1 - j tau_i)^2 has Im R~^2 <= 0 and is never on the negative real axis
/// (Im R~^2 = 0 needs every tau_i dx_i = 0, and then R~^2 = R^2 >= 0): Re R~ >= 0, Im R~ <= 0, the
/// kernel exp(-jkR~) decays for k_1 real and for k_2 with Im k_2 <= 0, and R~ = 0 only for r = r'
/// (|R~^2| >= R^2 / sqrt(1 + sigma_max^2): the terms lie in a sector of opening 2 atan sigma_max <
/// pi). The debug build checks Im R~ <= 0 at every point pair. Both regions use the same R~.
///
/// Symmetry. Exchanging (m, r) and (n, r') maps the L integrand onto itself, and K too: with
/// A = DS' f_n, C = DS f_m, B = r~ - r~' and c the symmetric scalar factor of grad~' G,
/// K_nm = c A . (-B x C) = -c B . (C x A) ... = K_mn (cyclic triple products). The blocks are
/// therefore computed for the ordering (min(t, s), max(t, s)) and transposed for the other one: the
/// stretched Galerkin matrix is bitwise complex symmetric.
///
/// Regions (WP-PML2c, element_blocks_both_regions). The dense assembly needs the blocks of both
/// regions of every pair. The pair setup (triangles, classification, slots) is done once; the rule
/// is selected per region as for a single-region call (near / far: the degree of choose_plain at
/// k_i; touching: touching_points at k_i, which differ between the regions only through the
/// k-aware extra). If both regions select the same rule, one pass over the point pairs computes
/// R~, 1 / R~ and the stretch factors once and accumulates the moments of both kernels (Accum<2>);
/// otherwise each region gets its own pass with its own rule. The fastmath choice (see
/// "Evaluation") is also made per region; in the rare shared pass where only one region's k exceeds
/// the fastmath bound, R~ is computed in both variants. Either way every region's moments
/// receive exactly the operations of a pass of its own, so the blocks are bitwise those of
/// stretched_element_blocks for that region. A common rule at the larger of the two selections
/// (the higher degree, the larger point counts) would also cover both regions at equal or better
/// accuracy, but is more expensive where the selections differ: the Ag touching rule with the
/// k-aware extra has 2-6 times the points of the vacuum rule, and the kernel part of a point pair
/// (exp, sin, cos, the moments) costs more than the shared geometry, so evaluating the vacuum
/// kernel on the Ag points costs more than its own pass (observed in WP-PML2c).
///
/// Jump term. For a horizontal triangle (normal +-z; required for stretched triangles, ADR 0009
/// decision 1) with DS = diag(s_x, s_y, 1) and det DS = s_x s_y:
///   (DS f_m) . (z x DS f_n) = s_x s_y (f_my f_nx - f_mx f_ny) = det DS  f_m . (z x f_n),
/// and the Piola-transformed jump integrand f~_m . (z x f~_n) dS~ = (DS f_m) . (z x DS f_n) dS /
/// det DS is f_m . (z x f_n) dS: kernels::jump_block is unchanged (tested).
///
/// Evaluation. Per point pair (x test, y source, both real) the stretch gives Im x~, Im y~ and
/// sigma_x, sigma_y at x and y; r~ - r~' = (x - y) + j (Im x~ - Im x~', Im y~ - Im y~', 0), with
/// x - y taken from the relative coordinates of the rule (touching pairs) or the point coordinates
/// (near / far). With f_a(x) = (D_a / 2)(X - P_a) (X = x - o, P_a = p_a - o, o a vertex of the
/// pair), (DS f_a) . (DS' f_b) = (D_a D_b / 4) sum_i S_i(x) S_i(y) (X_i - P_ai)(Y_i - P_bi) and
/// (DS f_a) . (grad~' G x DS' f_b) = (D_a D_b / 4) sum_ik tau_ik (X_i - P_ai)(Y_k - P_bk) with
/// tau_ik = c S_i(x) eps_ijk (r~ - r~')_j S_k(y), so per pair the rule accumulates the moments
/// sum w u_i {X_i Y_i, X_i, Y_i, 1} (u_i = g S_i(x) S_i(y)) and sum w tau_ik {X_i Y_k, X_i, Y_k, 1}
/// and the 3 x 3 blocks follow at the end. Real arithmetic in the point loop (the library's
/// complex multiplication checks for NaN). The rules hand their point pairs to add_pair, which
/// collects them in batches of kBatch (structure of arrays, WP-PML2c); per batch loop A computes
/// R~ and 1 / R~ (shared by the regions, see "Regions"), loop B per region the kernel factors and
/// loop C sums the moments in point order (the order of the unbatched WP-PML2 loop). Loops A and B
/// are branch-free so that they vectorise (with -fno-math-errno -fno-trapping-math, value-neutral).
/// With OperatorOptions::fast_plain_kernel (default) and |k| |R~| <= kFastSincosMax for the pair,
/// pairs with K = 0 (flat near / far pairs, coplanar touching pairs) evaluate exp(-j k R~) with
/// kernels::fastmath and |R~^2| as sqrt(re^2 + im^2): blocks within 7e-14 of the C-library
/// arithmetic (measured on a 800-triangle open surface, every 7th pair); pairs with K != 0 keep the
/// C library (their small K blocks, |K| eta / |L| down to 1e-6 near coplanarity, magnify the
/// ulp-level differences to 2e-12 of K). The C-library variant is the WP-PML2 arithmetic up to the
/// compiler's pairing of sin and cos into sincos (<= 5e-15 relative per block).
/// The stretch is evaluated from CoordinateStretch::profile_x / profile_y with one power per
/// stretched axis and point (CoordinateStretch::map and jacobian_diag need two; agreement to a few
/// ulp is tested).
///
/// The anisotropic ridge. For a stretch that differs between the axes (the PML strips: sigma_x up
/// to sigma_max, sigma_y = 0) the kernel 1 / R~ is near-singular along two directions of
/// r - r' even for distant points: with the metric frozen, R~^2 = s_x^2 dx^2 + s_y^2 dy^2 =
/// s_y^2 F_+ F_-, F_+- = dy -+ c dx, c = j s_x / s_y = ((sigma_x - sigma_y) + j (1 + sigma_x
/// sigma_y)) / (1 + sigma_y^2). F_+ vanishes for the complex direction dy / dx = c, which lies
/// close to the real directions when |Re c| >> Im c. Over real unit directions d, |F_+(d)|^2 =
/// d^T M d with M = f_r f_r^T + f_i f_i^T (f_r = (-Re c, 1), f_i = (-Im c, 0)); its smallest
/// eigenvector is the ridge direction (F_+(ridge) / F_+(normal) is then purely imaginary, so the
/// complex zero along the normal has zero real part) and eps = sqrt(lambda_min / lambda_max) the
/// relative angular width of the near-singularity (eps = 1 isotropic; 1 / (1 + sigma^2) for
/// sigma_y = 0, e.g. 0.059 for sigma = 4); F_- gives the mirrored ridge. Plain Gauss rules converge
/// slowly across it: relative-coordinate rules of 20 points per direction left 1.6e-3 (identical
/// triangles), 5.7e-4 (shared edges) and 3.6e-5 (shared vertices) of a structured-grid block at
/// sigma = 4, sigma_y = 0 (prototype study of WP-PML2), and non-touching coplanar pairs along the
/// ridge 1e-4 with 100 x 100 points at D = 1.5 h.
///
/// Touching pairs (identical, shared edge, shared vertex; the singular route of ADR 0009 decision
/// 3, option (b)): relative-coordinate quadrature of Sauter-Schwab type directly on the analytic
/// stretched kernel. The singular set (diagonal, shared edge, shared vertex) is a corner of the
/// parameter domain whose Jacobian removes the 1 / R~ (and the 1 / R~^2 of K) singularity: R~ / R
/// is a smooth, non-vanishing function of the direction of r - r' and of r, r' (the secants),
/// so the transformed integrand is smooth. Parametrisations (the reference rules of
/// tests/unit/test_operators.cpp, generalised to integrands that also depend on the absolute
/// position):
///  * identical: y - x = z1 e1 + z2 e2 over the hexagon T^ - T^ in the six cones over its edges,
///    z = rho (P_i + u (P_i+1 - P_i)), dz = rho drho du, x over T^ n (T^ - z) (a triangle) with the
///    degree-kIdenticalXDegree Dunavant rule; rho and u Gauss-Legendre;
///  * shared edge AB: x = A + s e + t a, y = A + (s + z) e + t' b over the polytope of (z, t, t')
///  in
///    cones from the origin (the singular point) over its four outer facets, split by z = 0 and
///    t - t' - z = 0 (where the s range changes form), p = rho [q0 + u (q1 - q0) + u v (q2 - q1)],
///    dp = |det(q0, q1, q2)| rho^2 u; s on [max(0, -z), min(1 - t, 1 - t' - z)] with kEdgeSPoints
///    Gauss-Legendre points (the kernel depends on s through the secants only);
///  * shared vertex A: (x^, y^) = lambda (edge point, triangle point) on the two facets {s + t = 1}
///    x T^ and T^ x {s' + t' = 1} (Jacobian lambda^3), each facet the prism [0, 1] x T^ in (sigma,
///    t0, t1) cut into three tetrahedra with collapsed coordinates p = q0 + a [(q1 - q0) + b ((q2 -
///    q1) + c (q3 - q2))], dp = 6 V a^2 b.
/// Coplanar pairs (all touching pairs of the flat PML) with eps < kRidgeSplitBelow, at the stretch
/// frozen at the mid-point of the two centroids: the direction of r - r' depends linearly on the
/// parameters of each rule (identical: on u; shared edge: on the base point of the cones, since
/// d = -z e + t a - t' b is linear in p; shared vertex: affinely on the facet point), so the set
/// where d is parallel to a ridge direction is a line or plane in parameter space. The cones /
/// tetrahedra are split along it (identical: the hexagon edges at the ridge crossing; shared edge:
/// the facet polygons by the ridge planes through the origin, like the two planes above; shared
/// vertex: the prism tetrahedra by the affine ridge planes), and every direction of a piece that
/// ends on a ridge plane is integrated with geometrically graded Gauss-Legendre panels toward it
/// (ratio 2, down to kFine eps, ceil(0.6 p) points per panel; the vertex order of each piece is
/// chosen to need the fewest graded directions). The secants vary over the pair, so the actual
/// ridge sits within ~(Delta sigma over the pair) eps of the frozen one, inside the graded panels.
/// Non-coplanar pairs (a stretched triangle touching a tilted one, only at the PML start next to
/// the taper with gap 0) use the same rules without the split; the accuracy then depends on sigma
/// there. For a profile of order m >= 1 the first PML cell has sigma <= sigma_max (h / W)^m, so
/// the ridge is weak (eps ~ 1 / (1 + sigma^2) close to 1); for m < 1 sigma is large right at the
/// start and the open-surface setup (WP-PML3) has to keep W_g >= h (no tilted triangle touching
/// the PML). stretched_pair_info reports the frozen-metric eps of these pairs too (ridge_split
/// false).
/// Points (touching_points): p = stretched_touching_points per plain and radial direction (plus a
/// k-aware extra for |k| s_max h > 5.3), max(ceil(0.6 p), ceil(p_eff / 2)) per graded panel.
/// Measured with the default p = 10 (all touching pairs of the central triangle of a structured
/// right-isosceles grid at h = lambda / 10, order-2 profile sigma_max = 4 on W = 2 lambda, vacuum
/// and Ag, against p = 16; tests/unit/test_stretched_kernels.cpp): identical <= 5e-11, shared edge
/// <= 4e-8, shared vertex
/// <= 3e-8 at sigma = 1, <= 6e-7 at sigma = 2.6-3.4 (sigma_y = 0), <= 1e-8 in the corner (sigma_x
/// 3.4, sigma_y 1.5); the isotropic case (eps >= kRidgeSplitBelow) needs no split (p = 8:
/// ~1e-10). Mean point pairs per call (vacuum / Ag, the k-aware extra raises the latter):
/// identical 1.7-3.2e4 / 3.8-7.2e4, shared edge 3.5-4.7e5 / 1.2-1.6e6, shared vertex 0.5-1.1e6 /
/// 3.5-6.1e6 (WP-PML2 rule; with the WP-PML2c graded points the Ag means of the [stretched_cost]
/// surface drop from 3.1e4 / 6.5e5 / 2.0e6 to 2.6e4 / 5.4e5 / 1.4e6).
///
/// Near and far pairs: the double Dunavant rule of operators.cpp with the degree selected by its
/// error model at
///   kappa_s = sqrt((h / D_eff)^2 + (0.15 |k| s_max h)^2),
/// s_max = max |s| over the vertices of the pair (the oscillation of exp(-j k R~) with |k R~| up to
/// |k| s_max R; exact for a uniform stretch, where G(R~) = G_{k s}(R) / s) and D_eff = D min(1,
/// kRidgeDistance delta), delta = min sqrt(Delta^2 + eps^2) over the four corner combinations of
/// the sigma ranges of the pair (eps the ridge width, Delta the angular gap between the in-plane
/// direction cone of the pair and the ridge directions; for pairs that are not flat also the ridge
/// between the stretched plane and the unstretched z axis). The complex zero of R~^2 lies at
/// transverse distance ~ delta R from a real point pair, which shortens the distance seen by the
/// rule. No decay-aware relaxation (the stretched pairs keep the relative target); arithmetic as
/// in "Evaluation". If no degree of the ladder meets the target (anisotropic near pairs along the
/// ridge), quad_degree_near is used; stretched_pair_info reports target_reached = false.
///
/// Translation key (WP-PML5c review, kernels::stretched_touching_key).
/// op::assemble_sparse_stretched evaluates the touching pairs of a PML strip once per class of
/// bitwise equal keys and copies the blocks to the other members, so the key must hold every
/// pair-dependent input of the touching path of stretched_blocks, in the form that path reads it.
/// It is built here from the same structures (PairSetup, Slots, touching_points, fast_region,
/// touching_origin); tri_of, slots_of, touching_rule and stretched_blocks carry back-references.
/// Inside a strip with the margin kTranslationStripMargin (|u| <= start - 1e-12 max(start, h))
/// every point of the rule has sigma_u = 0 and Im u~ = 0 exactly (eval_axis returns zeros for d <=
/// 0), so the u coordinate enters only through differences from the rule origin; centroids and
/// radii (absolute along u) enter only through the fastmath decision, which the key holds as a
/// flag.
#include "specklebem/kernels/fast_math.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/kernels/singularity.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "stretched_detail.hpp"

namespace specklebem::kernels::detail {

namespace {

constexpr Real kInv4Pi = 1.0 / (4.0 * constants::pi);
constexpr Real kPi = constants::pi;
/// Ridge split for eps below this (eps = 1: isotropic).
constexpr Real kRidgeSplitBelow = 0.9;
/// Finest graded panel (relative to the graded direction) = kFine eps.
constexpr Real kFine = 0.5;
/// Finest graded panel is at least this (at most 11 panels per graded end).
constexpr Real kFinestMin = 1.0 / 1024.0;
/// Effective distance factor of the ridge in the near / far degree selection.
constexpr Real kRidgeDistance = 2.0;
/// Non-flat pairs (K != 0): the error model is calibrated on L-type blocks, while the K block of a
/// nearly coplanar pair is ~100 times smaller than L and its relative error ~20 times the L error
/// (WP-PML2 review fixes). The target of these pairs is tightened by this factor (log10 20).
constexpr Real kNonFlatLogMargin = 1.30103;
/// Dunavant degree of the x integral of identical triangles (exact for the polynomial part).
constexpr int kIdenticalXDegree = 6;
/// Gauss-Legendre points of the s integral of shared edges.
constexpr int kEdgeSPoints = 4;
/// Midpoint subdivision levels of the cone bases of shared edges (4^levels pieces per base).
constexpr int kEdgeBaseLevels = 1;
/// Largest Gauss-Legendre rule (stretched_touching_points <= 16).
constexpr int kMaxGl = 16;
/// Normals of stretched triangles: |n_x|, |n_y| combined below this (horizontal).
constexpr Real kHorizontalTol = 1e-9;
/// |n_t x n_s| below which a touching pair counts as coplanar (as operators.cpp).
constexpr Real kCoplanarTol = 1e-12;
/// Relative tolerance of the on-plane classification of parameter-space vertices.
constexpr Real kOnPlaneTol = 1e-9;

// ---------------------------------------------------------------------------------------------
// Small real 3-vectors and complex numbers in real form (hot loops, also fast at -O0).
// ---------------------------------------------------------------------------------------------

struct P3 {
    Real x, y, z;
};
inline P3 operator+(const P3& a, const P3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline P3 operator-(const P3& a, const P3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline P3 operator*(Real s, const P3& a) {
    return {s * a.x, s * a.y, s * a.z};
}
inline Real dot(const P3& a, const P3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline P3 cross(const P3& a, const P3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Real norm(const P3& a) {
    return std::sqrt(dot(a, a));
}
P3 to_p3(const Vec3& v) {
    return {v(0), v(1), v(2)};
}

struct Cx {
    Real re, im;
};
inline Cx operator+(Cx a, Cx b) {
    return {a.re + b.re, a.im + b.im};
}
inline Cx operator*(Cx a, Cx b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
inline Cx operator*(Real s, Cx a) {
    return {s * a.re, s * a.im};
}
inline Cx operator-(Cx a) {
    return {-a.re, -a.im};
}
inline Cx& operator+=(Cx& a, Cx b) {
    a.re += b.re;
    a.im += b.im;
    return a;
}
inline Cx complex_exp(Cx z) {
    const Real m = std::exp(z.re);
    return {m * std::cos(z.im), m * std::sin(z.im)};
}

// ---------------------------------------------------------------------------------------------
// Stretch evaluation (CoordinateStretch profiles, one power per stretched axis).
// ---------------------------------------------------------------------------------------------

struct Axis {
    bool active = false;
    Real start = 0.0, width = 1.0, sigma_max = 0.0, order = 0.0, inv_m1 = 1.0;
    int int_order = -1;  ///< order as a small integer (repeated products), -1: std::pow
};

Axis axis_of(const std::optional<geometry::StretchProfile>& p) {
    Axis a;
    if (!p || !(p->sigma_max > 0.0)) {
        return a;
    }
    a.active = true;
    a.start = p->start;
    a.width = p->width;
    a.sigma_max = p->sigma_max;
    a.order = p->order;
    a.inv_m1 = 1.0 / (p->order + 1.0);
    for (int m = 0; m <= 4; ++m) {
        if (p->order == static_cast<Real>(m)) {
            a.int_order = m;
        }
    }
    return a;
}

inline Real profile_power(const Axis& a, Real r) {
    switch (a.int_order) {
        case 0:
            return 1.0;
        case 1:
            return r;
        case 2:
            return r * r;
        case 3:
            return r * r * r;
        case 4:
            return (r * r) * (r * r);
        default:
            return std::pow(r, a.order);
    }
}

/// Stretch at one point: Im x~, Im y~ and sigma_x, sigma_y (s = 1 - j sigma).
struct StretchValue {
    Real ix = 0.0, iy = 0.0, sx = 0.0, sy = 0.0;
};

/// Im t~ and sigma(|t|) of one axis (coordinate_stretch.hpp: t~ = t - j sgn(t) Int sigma).
inline void eval_axis(const Axis& a, Real t, Real& im, Real& sigma) {
    im = 0.0;
    sigma = 0.0;
    if (!a.active) {
        return;
    }
    const Real d = std::abs(t) - a.start;
    if (!(d > 0.0)) {
        return;
    }
    Real integral = 0.0;
    if (d >= a.width) {
        sigma = a.sigma_max;
        integral = a.sigma_max * (a.width * a.inv_m1 + (d - a.width));
    } else {
        const Real r = d / a.width;
        const Real pw = profile_power(a, r);
        sigma = a.sigma_max * pw;
        integral = a.sigma_max * a.width * (pw * r) * a.inv_m1;
    }
    im = t > 0.0 ? -integral : integral;
}

struct StretchEval {
    Axis x, y;
    explicit StretchEval(const geometry::CoordinateStretch& s)
        : x(axis_of(s.profile_x())), y(axis_of(s.profile_y())) {}
    [[nodiscard]] StretchValue at(const P3& r) const {
        StretchValue v;
        eval_axis(x, r.x, v.ix, v.sx);
        eval_axis(y, r.y, v.iy, v.sy);
        return v;
    }
};

// ---------------------------------------------------------------------------------------------
// Triangles and RWG slots.
// ---------------------------------------------------------------------------------------------

/// Pair geometry of the rules. Translation key (stretched_touching_key): every field the touching
/// rule reads must enter the key in the form the rule reads it (v, id via the shared-index
/// pattern, n, area, longest; centroid and radius only through the fastmath decision and the
/// ridge mid-point, whose across-strip components follow from the absolute coordinates). A new
/// coordinate-derived field read by the touching path must be added to the key.
struct Tri {
    std::array<P3, 3> v;
    std::array<Index, 3> id;
    P3 n;  ///< unit normal
    Real area;
    P3 centroid;
    Real longest;
    Real radius;  ///< largest centroid-vertex distance
};

/// Translation key: see Tri (any new coordinate-derived input must enter stretched_touching_key).
Tri tri_of(const geometry::TriangleMesh& mesh, Index t) {
    Tri g{};
    for (Index i = 0; i < 3; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        g.id[ii] = mesh.triangles()(t, i);
        g.v[ii] = to_p3(mesh.vertices().row(g.id[ii]).transpose());
    }
    g.n = to_p3(mesh.normal(t));
    g.area = mesh.area(t);
    g.centroid = (1.0 / 3.0) * (g.v[0] + g.v[1] + g.v[2]);
    g.longest = std::max({norm(g.v[1] - g.v[0]), norm(g.v[2] - g.v[1]), norm(g.v[0] - g.v[2])});
    g.radius =
        std::max({norm(g.v[0] - g.centroid), norm(g.v[1] - g.centroid), norm(g.v[2] - g.centroid)});
    return g;
}

struct Slots {
    std::size_t count = 0;
    std::array<Real, 3> div{};
    std::array<P3, 3> p{};
    std::array<Index, 3> vid{};  ///< vertex index of the free vertex (p is its position)
};

/// RWG slots of t (operators.cpp: f_a(r) = (div[a] / 2)(r - p[a])). support() range-checks t.
/// Translation key: div and p (the position of a vertex of t) enter stretched_touching_key per
/// local free vertex; a new slot quantity read by the touching path must be added there.
Slots slots_of(const basis::RwgSpace& space, Index t) {
    const basis::RwgSpace::Support s = space.support(t);
    Slots out;
    out.count = static_cast<std::size_t>(s.count);
    for (std::size_t a = 0; a < out.count; ++a) {
        const Index n = s.n[a];
        out.div[a] = space.divergence(n, t);
        const Index v = s.sign[a] > 0 ? space.plus_free_vertex(n) : space.minus_free_vertex(n);
        out.vid[a] = v;
        out.p[a] = to_p3(space.mesh().vertices().row(v).transpose());
    }
    return out;
}

bool stretched_tri(const StretchEval& se, const Tri& g) {
    for (const P3& v : g.v) {
        if ((se.x.active && std::abs(v.x) > se.x.start) ||
            (se.y.active && std::abs(v.y) > se.y.start)) {
            return true;
        }
    }
    return false;
}

void require_horizontal(const StretchEval& se, const Tri& g, Index t) {
    if (stretched_tri(se, g) && std::hypot(g.n.x, g.n.y) > kHorizontalTol) {
        throw std::invalid_argument(
            "element_blocks: triangle " + std::to_string(t) +
            " lies in the stretched zone (PML) but is not horizontal; ADR 0009 requires flat PML "
            "triangles at z = const (the jump term and the Piola factors rely on it)");
    }
}

// ---------------------------------------------------------------------------------------------
// Point-pair accumulation (file comment "Evaluation").
// ---------------------------------------------------------------------------------------------

struct Sums {
    std::array<Cx, 3> lxy{}, lx{}, ly{}, l0{};
    Cx scalar{0.0, 0.0};
    std::array<Cx, 9> kxy{}, kx{}, ky{}, k0{};  ///< index 3 i + k
};

struct KernelParams {
    Real kr, ki;  ///< k = kr + j ki
};

/// Point pairs per kernel loop. A batch fills in visiting order and may span several outer points
/// of a rule (add_pair flushes whenever it is full, finish_points at the end of the pair), so any
/// size is correct; 80 balances the vector loop lengths against the stack footprint.
constexpr std::size_t kBatch = 80;

/// Point pairs collected for the kernel loops (file comment "Evaluation"), structure of arrays:
/// X, Y relative to the rule origin, r~ - r~' = (d0r + j d0i, d1r + j d1i, d2), the stretch
/// sigma_x, sigma_y at x (sxx, syx) and at y (sxy, syy), and the weight.
struct Batch {
    std::size_t n = 0;
    std::array<Real, kBatch> xx{}, xy{}, xz{}, yx{}, yy{}, yz{};
    std::array<Real, kBatch> d0r{}, d0i{}, d1r{}, d1i{}, d2{};
    std::array<Real, kBatch> sxx{}, syx{}, sxy{}, syy{}, w{};
};

/// R~ and 1 / R~ per point pair of a batch (shared by the regions with the same `fast` choice).
/// Deliberately not value-initialised (flush declares one per batch): geometry_values writes the
/// first b.n entries before kernel_values reads them, and nothing reads beyond b.n.
struct GeometryValues {
    std::array<Real, kBatch> rr, ri, ir, ii;
};

/// Weighted kernel w G (times 4 pi) and the gradient factor c per point pair, one region. Not
/// value-initialised either: kernel_values writes gr, gi (and cr, ci if need_k) for the first b.n
/// entries before accumulate reads exactly those.
struct KernelValues {
    std::array<Real, kBatch> gr, gi, cr, ci;
};

/// Loop A: R~ = sqrt((r~ - r~') . (r~ - r~')) on the principal branch and 1 / R~. Written
/// branch-free (selects) so that the fast variant vectorises; principal_sqrt's value for either
/// sign of Re R~^2. kFast: |R~^2| as sqrt(re^2 + im^2) instead of std::hypot (no overflow or
/// underflow for point distances between 1e-150 and 1e150 m).
template <bool kFast>
void geometry_values(const Batch& b, GeometryValues& v) {
    const std::size_t n = b.n;
    for (std::size_t q = 0; q < n; ++q) {
        const Cx d0{b.d0r[q], b.d0i[q]};
        const Cx d1{b.d1r[q], b.d1i[q]};
        const Real d2 = b.d2[q];
        const Cx z = d0 * d0 + d1 * d1 + Cx{d2 * d2, 0.0};
        const Real r = kFast ? std::sqrt(z.re * z.re + z.im * z.im) : std::hypot(z.re, z.im);
        const Real a = std::sqrt(0.5 * (r + std::abs(z.re)));
        const Real h = 0.5 * z.im / a;
        const Real bq = a > 0.0 ? h : 0.0;
        const bool pos = z.re >= 0.0;
        const Real abq = bq < 0.0 ? -bq : bq;
        const Real sa = std::copysign(a, z.im);
        const Real re = pos ? a : abq;
        const Real im = pos ? bq : sa;
        const Real d = re * re + im * im;
        v.rr[q] = re;
        v.ri[q] = im;
        v.ir[q] = re / d;
        v.ii[q] = -im / d;
    }
#ifndef NDEBUG
    for (std::size_t q = 0; q < n; ++q) {
        if (v.ri[q] > 1e-12 * std::hypot(v.rr[q], v.ri[q])) {
            // Internal invariant (file comment "Mapping"): CoordinateStretch only builds monotone
            // stretches, for which Im R~ <= 0 at every point pair.
            throw std::logic_error(
                "stretched element blocks: internal invariant violated, Im R~ > 0 at a point pair "
                "(the principal-branch distance of the monotone stretch must have Im R~ <= 0; "
                "this is a bug in the stretch evaluation or the quadrature points, not a user "
                "error)");
        }
    }
#endif
}

/// Loop B: w exp(-j k R~) / R~ and c = (1 + j k R~) w exp(-j k R~) / R~^3 per point pair for one
/// region. kFast: kernels::fastmath (fast_exp_nonpositive, fast_sincos), else the C library's exp,
/// cos and sin (the arithmetic of WP-PML2, bitwise). Returns false if kFast and an exponent
/// Re(-j k R~) = k_r Im R~ + k_i Re R~ is positive (rounding of Im R~ ~ 0; fast_exp_nonpositive
/// does not cover it): the caller then repeats the batch with the C library.
template <bool kFast>
bool kernel_values(const Batch& b, const GeometryValues& g, const KernelParams& kp, bool need_k,
                   KernelValues& v) {
    const std::size_t n = b.n;
    bool positive = false;
    for (std::size_t q = 0; q < n; ++q) {
        const Real kRr = kp.kr * g.rr[q] - kp.ki * g.ri[q];
        const Real kRi = kp.kr * g.ri[q] + kp.ki * g.rr[q];
        Cx e;
        if constexpr (kFast) {
            positive = positive || kRi > 0.0;
            const Real m = fastmath::fast_exp_nonpositive(kRi);
            Real sn;
            Real cs;
            fastmath::fast_sincos(-kRr, sn, cs);
            e = Cx{m * cs, m * sn};
        } else {
            e = complex_exp(Cx{kRi, -kRr});
        }
        const Cx inv_r{g.ir[q], g.ii[q]};
        const Cx wg = b.w[q] * (e * inv_r);
        v.gr[q] = wg.re;
        v.gi[q] = wg.im;
        if (need_k) {
            const Cx c = wg * (Cx{1.0 - kRi, kRr} * (inv_r * inv_r));
            v.cr[q] = c.re;
            v.ci[q] = c.im;
        }
    }
    return !positive;
}

/// Loop C: the moments of one region, summed in point order (file comment "Evaluation"). If X_z
/// and Y_z vanish for every point pair of the batch (flat pairs in the plane of the rule origin),
/// the z moments lxy[2], lx[2], ly[2] would only accumulate signed zeros and l0[2] the same sum as
/// `scalar`; they are skipped (l0[2] is set from `scalar` at the end), bitwise the same moments.
void accumulate(const Batch& b, const KernelValues& v, bool need_k, bool planar, Sums& s) {
    const std::size_t n = b.n;
    const std::size_t nz = planar ? 2 : 3;
    for (std::size_t q = 0; q < n; ++q) {
        const Cx wg{v.gr[q], v.gi[q]};
        // S_i(x) S_i(y), S = 1 - j sigma.
        const Cx sxx{1.0, -b.sxx[q]};
        const Cx syx{1.0, -b.syx[q]};
        const Cx sxy{1.0, -b.sxy[q]};
        const Cx syy{1.0, -b.syy[q]};
        const std::array<Real, 3> xa = {b.xx[q], b.xy[q], b.xz[q]};
        const std::array<Real, 3> ya = {b.yx[q], b.yy[q], b.yz[q]};
        const std::array<Cx, 3> u = {wg * (sxx * sxy), wg * (syx * syy), wg};
        for (std::size_t i = 0; i < nz; ++i) {
            s.lxy[i] += (xa[i] * ya[i]) * u[i];
            s.lx[i] += xa[i] * u[i];
            s.ly[i] += ya[i] * u[i];
            s.l0[i] += u[i];
        }
        s.scalar += wg;
        if (!need_k) {
            continue;
        }
        // grad~' G = c (r~ - r~').
        const Cx c{v.cr[q], v.ci[q]};
        const Cx d0{b.d0r[q], b.d0i[q]};
        const Cx d1{b.d1r[q], b.d1i[q]};
        const Cx dz{b.d2[q], 0.0};
        std::array<Cx, 9> tau{};
        tau[2] = c * sxx * d1;           // (x, z)
        tau[1] = -(c * sxx * dz * syy);  // (x, y)
        tau[3] = c * syx * dz * sxy;     // (y, x)
        tau[5] = -(c * syx * d0);        // (y, z)
        tau[7] = c * d0 * syy;           // (z, y)
        tau[6] = -(c * d1 * sxy);        // (z, x)
        for (const std::size_t ik : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5},
                                     std::size_t{6}, std::size_t{7}}) {
            const Real xi = xa[ik / 3];
            const Real yk = ya[ik % 3];
            s.kxy[ik] += (xi * yk) * tau[ik];
            s.kx[ik] += xi * tau[ik];
            s.ky[ik] += yk * tau[ik];
            s.k0[ik] += tau[ik];
        }
    }
}

/// Moments of NR regions accumulated over the same point pairs (file comment "Regions"): the
/// geometry of a point pair (R~, 1 / R~, the stretch) is computed once, the kernel and the moments
/// per region. Each region's sums receive exactly the operations of a single-region pass, so its
/// blocks are bitwise those of a pass of its own over the same rule. `fast[r]`: the fastmath
/// geometry and kernel loops for region r (OperatorOptions::fast_plain_kernel and the argument
/// range checked for the pair at k_r), decided per region so that this holds also when only one
/// region's k exceeds the fastmath bound.
template <std::size_t NR>
struct Accum {
    std::array<KernelParams, NR> kp{};
    bool need_k = true;
    std::array<bool, NR> fast{};
    /// Set once a batch had X_z or Y_z != 0: the z moments are accumulated from then on.
    bool non_planar = false;
    std::array<Sums, NR> sums{};
    Batch b;
};

/// Evaluates and sums the collected point pairs.
template <std::size_t NR>
void flush(Accum<NR>& acc) {
    Batch& b = acc.b;
    if (b.n == 0) {
        return;
    }
    if (!acc.non_planar) {
        for (std::size_t q = 0; q < b.n; ++q) {
            if (b.xz[q] != 0.0 || b.yz[q] != 0.0) {
                // From here on the z moments are summed; l0[2] continues from the sum of wg so
                // far (= scalar), lxy[2], lx[2], ly[2] from the zeros they would hold.
                acc.non_planar = true;
                for (Sums& s : acc.sums) {
                    s.l0[2] = s.scalar;
                }
                break;
            }
        }
    }
    // g[0]: C-library geometry (std::hypot), g[1]: fast geometry; each computed only if a region
    // uses it (normally one of them: both regions share the choice unless only one region's k
    // exceeds the fastmath bound).
    std::array<GeometryValues, 2> g;
    bool any_fast = false;
    bool any_slow = false;
    for (const bool f : acc.fast) {
        any_fast = any_fast || f;
        any_slow = any_slow || !f;
    }
    if (any_slow) {
        geometry_values<false>(b, g[0]);
    }
    if (any_fast) {
        geometry_values<true>(b, g[1]);
    }
    KernelValues v;
    for (std::size_t r = 0; r < NR; ++r) {
        const GeometryValues& gr = g[acc.fast[r] ? 1 : 0];
        if (!(acc.fast[r] && kernel_values<true>(b, gr, acc.kp[r], acc.need_k, v))) {
            (void)kernel_values<false>(b, gr, acc.kp[r], acc.need_k, v);
        }
        accumulate(b, v, acc.need_k, !acc.non_planar, acc.sums[r]);
    }
    b.n = 0;
}

/// Ends the accumulation: flushes the last batch and completes the skipped planar z moments.
template <std::size_t NR>
void finish_points(Accum<NR>& acc) {
    flush(acc);
    if (!acc.non_planar) {
        for (Sums& s : acc.sums) {
            s.l0[2] = s.scalar;
        }
    }
}

/// Adds the point pair x = o + X (test), y = o + Y (source), dxy = x - y with weight w (evaluated
/// in batches of kBatch point pairs; finish_points ends the sum).
template <std::size_t NR>
inline void add_pair(Accum<NR>& acc, const P3& X, const P3& Y, const P3& dxy,
                     const StretchValue& sxv, const StretchValue& syv, Real w) {
    Batch& b = acc.b;
    const std::size_t q = b.n;
    b.xx[q] = X.x;
    b.xy[q] = X.y;
    b.xz[q] = X.z;
    b.yx[q] = Y.x;
    b.yy[q] = Y.y;
    b.yz[q] = Y.z;
    b.d0r[q] = dxy.x;
    b.d0i[q] = sxv.ix - syv.ix;
    b.d1r[q] = dxy.y;
    b.d1i[q] = sxv.iy - syv.iy;
    b.d2[q] = dxy.z;
    b.sxx[q] = sxv.sx;
    b.syx[q] = sxv.sy;
    b.sxy[q] = syv.sx;
    b.syy[q] = syv.sy;
    b.w[q] = w;
    if (++b.n == kBatch) {
        flush(acc);
    }
}

/// The 3 x 3 blocks from the moments (file comment "Evaluation").
void finish(const Sums& s, const Slots& ts, const Slots& ss, const P3& o, const RegionParams& reg,
            bool need_k, Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K) {
    const Complex jj(0.0, 1.0);
    const Complex c_vec = jj * reg.omega * reg.mu * kInv4Pi;
    const Complex c_sca = kInv4Pi / (jj * reg.omega * reg.eps);
    const Complex scalar(s.scalar.re, s.scalar.im);
    for (std::size_t a = 0; a < ts.count; ++a) {
        const P3 pa3 = ts.p[a] - o;
        const std::array<Real, 3> pa = {pa3.x, pa3.y, pa3.z};
        for (std::size_t b = 0; b < ss.count; ++b) {
            const P3 pb3 = ss.p[b] - o;
            const std::array<Real, 3> pb = {pb3.x, pb3.y, pb3.z};
            const Real q = 0.25 * ts.div[a] * ss.div[b];
            Cx vec{0.0, 0.0};
            for (std::size_t i = 0; i < 3; ++i) {
                vec += s.lxy[i];
                vec += (-pb[i]) * s.lx[i];
                vec += (-pa[i]) * s.ly[i];
                vec += (pa[i] * pb[i]) * s.l0[i];
            }
            const auto ai = static_cast<Eigen::Index>(a);
            const auto bi = static_cast<Eigen::Index>(b);
            L(ai, bi) =
                c_vec * (q * Complex(vec.re, vec.im)) + c_sca * (ts.div[a] * ss.div[b]) * scalar;
            if (need_k) {
                Cx kk{0.0, 0.0};
                for (std::size_t ik = 0; ik < 9; ++ik) {
                    const std::size_t i = ik / 3;
                    const std::size_t k = ik % 3;
                    kk += s.kxy[ik];
                    kk += (-pb[k]) * s.kx[ik];
                    kk += (-pa[i]) * s.ky[ik];
                    kk += (pa[i] * pb[k]) * s.k0[ik];
                }
                K(ai, bi) = (kInv4Pi * q) * Complex(kk.re, kk.im);
            } else {
                K(ai, bi) = Complex(0.0, 0.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gauss-Legendre rules on [0, 1] and graded line rules.
// ---------------------------------------------------------------------------------------------

struct GlTables {
    std::array<std::array<Real, kMaxGl>, kMaxGl + 1> x{}, w{};
    GlTables() {
        for (int n = 1; n <= kMaxGl; ++n) {
            const LineRule g = gauss_legendre(n);
            for (std::size_t i = 0; i < g.nodes.size(); ++i) {
                x[static_cast<std::size_t>(n)][i] = 0.5 * (g.nodes[i] + 1.0);
                w[static_cast<std::size_t>(n)][i] = 0.5 * g.weights[i];
            }
        }
    }
};

const GlTables& gl_tables() {
    static const GlTables t;
    return t;
}

constexpr std::size_t kMaxLine = 2 * 12 * static_cast<std::size_t>(kMaxGl);

struct Line {
    std::size_t n = 0;
    std::array<Real, kMaxLine> x{}, w{};
};

void add_panel(Line& l, Real a, Real b, int n) {
    const GlTables& g = gl_tables();
    const auto ni = static_cast<std::size_t>(n);
    for (std::size_t i = 0; i < ni; ++i) {
        l.x[l.n] = a + (b - a) * g.x[ni][i];
        l.w[l.n] = (b - a) * g.w[ni][i];
        ++l.n;
    }
}

/// Panels of [a, b] graded toward a (toward_a) or b, ratio 2, finest panel <= fine (b - a).
void add_graded(Line& l, Real a, Real b, bool toward_a, Real fine, int n) {
    std::array<Real, 16> s{};
    std::size_t m = 0;
    Real v = 1.0;
    while (v > fine && m + 1 < s.size()) {
        v *= 0.5;
        s[m++] = v;
    }
    // Relative panels [s[m-1], s[m-2]], ..., [s[0], 1] and [0, s[m-1]].
    const Real len = b - a;
    const auto map = [&](Real t) { return toward_a ? a + len * t : b - len * t; };
    const auto panel = [&](Real t0, Real t1) {
        const Real p = map(t0);
        const Real q = map(t1);
        add_panel(l, std::min(p, q), std::max(p, q), n);
    };
    panel(0.0, s[m - 1]);
    for (std::size_t i = m - 1; i > 0; --i) {
        panel(s[i], s[i - 1]);
    }
    panel(s[0], 1.0);
}

/// Points of the touching rules: n per plain direction, nr per radial direction (the singular
/// corner's distance variable), ng per graded panel, finest graded panel fine eps.
struct RuleParams {
    int n = 10;
    int nr = 10;
    int ng = 6;
    Real fine = kFine;
};

/// Rule on [0, 1]: Gauss-Legendre (n points), or graded toward 0 (lo) and / or 1 (hi) with finest
/// panel `finest` (relative to [0, 1]) and ng points per panel.
void line_rule(Line& out, int n, int ng, bool lo, bool hi, Real finest) {
    out.n = 0;
    const Real f = std::clamp(finest, kFinestMin, 0.5);
    if (lo && hi) {
        add_graded(out, 0.0, 0.5, true, 2.0 * f, ng);
        add_graded(out, 0.5, 1.0, false, 2.0 * f, ng);
    } else if (lo) {
        add_graded(out, 0.0, 1.0, true, f, ng);
    } else if (hi) {
        add_graded(out, 0.0, 1.0, false, f, ng);
    } else {
        add_panel(out, 0.0, 1.0, n);
    }
}

// ---------------------------------------------------------------------------------------------
// The ridge of an anisotropic stretch (file comment).
// ---------------------------------------------------------------------------------------------

struct Ridge {
    bool active = false;
    Real eps = 1.0;
    /// In-plane unit normals of the two ridge directions (xy components).
    std::array<std::array<Real, 2>, 2> normal{};
    /// Angles of the two ridge directions (atan2), for the near / far selection.
    std::array<Real, 2> angle{};
};

/// Ridge of the frozen metric diag(1 - j sigma_x, 1 - j sigma_y) in the xy plane; active if
/// eps < kRidgeSplitBelow.
Ridge ridge_of(Real sigma_x, Real sigma_y) {
    Ridge rg;
    const Real den = 1.0 + sigma_y * sigma_y;
    const Real cr = (sigma_x - sigma_y) / den;
    const Real ci = (1.0 + sigma_x * sigma_y) / den;
    // M = f_r f_r^T + f_i f_i^T for F_+ (f_r = (-cr, 1), f_i = (-ci, 0)); F_- mirrors x.
    const Real m00 = cr * cr + ci * ci;
    const Real m01 = -cr;
    const Real m11 = 1.0;
    const Real half_tr = 0.5 * (m00 + m11);
    const Real disc = std::hypot(0.5 * (m00 - m11), m01);
    const Real lmax = half_tr + disc;
    const Real lmin = (m00 * m11 - m01 * m01) / lmax;  // det / lmax, no cancellation
    rg.eps = std::sqrt(std::max(lmin, 0.0) / lmax);
    if (!(rg.eps < 1.0 - 1e-12)) {
        return rg;  // isotropic: no ridge direction
    }
    rg.active = rg.eps < kRidgeSplitBelow;
    // Eigenvector of lmin: (M - lmin) v = 0, from the row of larger norm.
    Real vx = m01;
    Real vy = lmin - m00;
    if (std::hypot(vx, vy) < std::hypot(lmin - m11, m01)) {
        vx = lmin - m11;
        vy = m01;
    }
    const Real len = std::hypot(vx, vy);
    vx /= len;
    vy /= len;
    // Branch +: direction (vx, vy); branch -: (-vx, vy). Normals rotate by 90 degrees.
    rg.normal[0] = {-vy, vx};
    rg.normal[1] = {-vy, -vx};
    rg.angle[0] = std::atan2(vy, vx);
    rg.angle[1] = std::atan2(vy, -vx);
    return rg;
}

inline Real ridge_form(const Ridge& rg, std::size_t b, const P3& d) {
    return rg.normal[b][0] * d.x + rg.normal[b][1] * d.y;
}

// ---------------------------------------------------------------------------------------------
// Touching rules (file comment). visit(X, Y, x - y, w) with X, Y relative to the rule origin.
// ---------------------------------------------------------------------------------------------

/// Identical triangle (v0 origin, e1 = v1 - v0, e2 = v2 - v0).
template <class Visit>
void rule_identical(const P3& e1, const P3& e2, const Ridge& rg, const RuleParams& rp,
                    Visit&& visit) {
    const Real jac = norm(cross(e1, e2));
    constexpr std::array<std::array<Real, 2>, 6> hex = {
        {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {1.0, -1.0}}};
    const TriangleRule& xr = triangle_rule(kIdenticalXDegree);
    const GlTables& gl = gl_tables();
    const auto ni = static_cast<std::size_t>(rp.nr);
    Line ul;
    for (std::size_t i = 0; i < 6; ++i) {
        const std::array<Real, 2>& p = hex[i];
        const std::array<Real, 2>& q = hex[(i + 1) % 6];
        const P3 dp = p[0] * e1 + p[1] * e2;
        const P3 dq = q[0] * e1 + q[1] * e2;
        std::array<Real, 4> cuts{0.0, 1.0, 0.0, 0.0};
        std::size_t nc = 2;
        if (rg.active) {
            for (std::size_t b = 0; b < 2; ++b) {
                const Real sp = ridge_form(rg, b, dp);
                const Real sq = ridge_form(rg, b, dq);
                if ((sp > 0.0 && sq < 0.0) || (sp < 0.0 && sq > 0.0)) {
                    cuts[nc++] = sp / (sp - sq);
                }
            }
        }
        // Insertion sort (std::sort on the 4-element array trips a GCC 16 -Warray-bounds false
        // positive).
        for (std::size_t i1 = 1; i1 < nc; ++i1) {
            for (std::size_t j1 = i1; j1 > 0 && cuts[j1] < cuts[j1 - 1]; --j1) {
                std::swap(cuts[j1], cuts[j1 - 1]);
            }
        }
        for (std::size_t c = 0; c + 1 < nc; ++c) {
            const Real ua = cuts[c];
            const Real ub = cuts[c + 1];
            if (!(ub - ua > 1e-12)) {
                continue;
            }
            const bool lo = ua > 0.0;
            const bool hi = ub < 1.0;
            line_rule(ul, rp.n, rp.ng, lo, hi, rp.fine * rg.eps / (ub - ua));
            for (std::size_t ir = 0; ir < ni; ++ir) {
                const Real rho = gl.x[ni][ir];
                const Real wr = gl.w[ni][ir];
                for (std::size_t iu = 0; iu < ul.n; ++iu) {
                    const Real u = ua + (ub - ua) * ul.x[iu];
                    const Real wu = (ub - ua) * ul.w[iu];
                    const Real z1 = rho * (p[0] + u * (q[0] - p[0]));
                    const Real z2 = rho * (p[1] + u * (q[1] - p[1]));
                    const Real a = std::max(0.0, -z1);
                    const Real bb = std::max(0.0, -z2);
                    const Real cc = 1.0 - std::max(0.0, z1 + z2);
                    const Real size = cc - a - bb;
                    if (!(size > 0.0)) {
                        continue;
                    }
                    const Real w0 = jac * jac * rho * wr * wu * 0.5 * size * size;
                    const P3 dz = z1 * e1 + z2 * e2;  // y - x
                    for (std::size_t k = 0; k < xr.weights.size(); ++k) {
                        const Vec3& l = xr.barycentric[k];
                        const Real s = l(0) * a + l(1) * (cc - bb) + l(2) * a;
                        const Real t = l(0) * bb + l(1) * bb + l(2) * (cc - a);
                        const P3 X = s * e1 + t * e2;
                        visit(X, X + dz, -1.0 * dz, w0 * xr.weights[k]);
                    }
                }
            }
        }
    }
}

/// Convex polygon in 3D parameter space (fixed capacity).
struct Poly {
    std::size_t n = 0;
    std::array<P3, 12> v{};
};

/// Splits poly by the plane nrm . p = 0 (through the origin); appends the pieces with >= 3
/// vertices to out (capacity checked).
template <std::size_t N>
void split_poly(const Poly& poly, const P3& nrm, std::array<Poly, N>& out, std::size_t& count) {
    Poly pos;
    Poly neg;
    Real scale = 0.0;
    for (std::size_t i = 0; i < poly.n; ++i) {
        scale = std::max(scale, norm(poly.v[i]));
    }
    const Real tol = kOnPlaneTol * norm(nrm) * scale;
    const auto push = [](Poly& p, const P3& v) {
        if (p.n >= p.v.size()) {
            throw std::logic_error("stretched element blocks: polygon capacity exceeded");
        }
        p.v[p.n++] = v;
    };
    for (std::size_t i = 0; i < poly.n; ++i) {
        const P3& a = poly.v[i];
        const P3& b = poly.v[(i + 1) % poly.n];
        Real da = dot(nrm, a);
        Real db = dot(nrm, b);
        if (std::abs(da) <= tol) {
            da = 0.0;
        }
        if (std::abs(db) <= tol) {
            db = 0.0;
        }
        if (da >= 0.0) {
            push(pos, a);
        }
        if (da <= 0.0) {
            push(neg, a);
        }
        if ((da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0)) {
            const P3 c = a + (da / (da - db)) * (b - a);
            push(pos, c);
            push(neg, c);
        }
    }
    for (const Poly* p : {&pos, &neg}) {
        if (p->n >= 3) {
            if (count >= N) {
                throw std::logic_error("stretched element blocks: piece capacity exceeded");
            }
            out[count++] = *p;
        }
    }
}

inline Real det3(const P3& a, const P3& b, const P3& c) {
    return dot(a, cross(b, c));
}

/// Grading flags of the collapsed coordinates of a cone base (u: q0 -> edge q1 q2; v: q1 -> q2)
/// for one rotation of (q0, q1, q2) and the on-plane flags of its vertices per ridge plane.
struct BaseGrading {
    bool u0 = false, u1 = false, v0 = false, v1 = false;
    [[nodiscard]] int cost() const {
        return (1 + static_cast<int>(u0) + static_cast<int>(u1)) *
               (1 + static_cast<int>(v0) + static_cast<int>(v1));
    }
};

BaseGrading base_grading(const std::array<std::array<bool, 3>, 2>& on, std::size_t planes) {
    BaseGrading g;
    for (std::size_t pl = 0; pl < planes; ++pl) {
        const bool o0 = on[pl][0];
        const bool o1 = on[pl][1];
        const bool o2 = on[pl][2];
        if (o1 && o2) {
            g.u1 = true;
        } else if (o0 && o1) {
            g.v0 = true;
        } else if (o0 && o2) {
            g.v1 = true;
        } else if (o0) {
            g.u0 = true;
        } else if (o1) {
            g.u1 = true;
            g.v0 = true;
        } else if (o2) {
            g.u1 = true;
            g.v1 = true;
        }
    }
    return g;
}

/// Shared edge (A origin; e = B - A, a = C - A, b = C' - A).
template <class Visit>
void rule_shared_edge(const P3& e, const P3& a, const P3& b, const Ridge& rg, const RuleParams& rp,
                      Visit&& visit) {
    const Real jac = norm(cross(e, a)) * norm(cross(e, b));
    constexpr std::size_t kCap = 64;  // 4 facets, each split by at most 4 planes
    std::array<Poly, kCap> work{};
    std::array<Poly, kCap> next{};
    std::size_t nw = 0;
    const auto add_facet = [&](std::initializer_list<P3> pts) {
        Poly p;
        for (const P3& q : pts) {
            p.v[p.n++] = q;
        }
        work[nw++] = p;
    };
    add_facet({{0, 1, 0}, {1, 1, 0}, {0, 1, 1}});
    add_facet({{-1, 0, 1}, {0, 0, 1}, {0, 1, 1}});
    add_facet({{1, 0, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}});
    add_facet({{-1, 0, 0}, {0, 1, 0}, {0, 1, 1}, {-1, 0, 1}});
    std::array<P3, 4> planes{};
    std::size_t np = 0;
    planes[np++] = {1.0, 0.0, 0.0};
    planes[np++] = {-1.0, 1.0, -1.0};
    std::array<P3, 2> ridge_planes{};
    std::size_t nr = 0;
    if (rg.active) {
        for (std::size_t br = 0; br < 2; ++br) {
            // l(d) with d = -z e + t a - t' b.
            const P3 rpl{-ridge_form(rg, br, e), ridge_form(rg, br, a), -ridge_form(rg, br, b)};
            if (norm(rpl) > 0.0) {
                ridge_planes[nr++] = rpl;
                planes[np++] = rpl;
            }
        }
    }
    for (std::size_t ip = 0; ip < np; ++ip) {
        std::size_t nn = 0;
        for (std::size_t k = 0; k < nw; ++k) {
            split_poly(work[k], planes[ip], next, nn);
        }
        std::swap(work, next);
        nw = nn;
    }
    const GlTables& gl = gl_tables();
    const auto ni = static_cast<std::size_t>(rp.nr);
    const auto ns = static_cast<std::size_t>(kEdgeSPoints);
    Line ul;
    Line vl;
    const auto base = [&](const std::array<P3, 3>& q) {
        {
            const Real det = std::abs(det3(q[0], q[1], q[2]));
            const Real scale = norm(q[0]) * norm(q[1]) * norm(q[2]);
            if (!(det > 1e-13 * scale)) {
                return;
            }
            // Rotation of the base with the fewest graded directions.
            BaseGrading best;
            std::size_t best_rot = 0;
            int best_cost = 1 << 20;
            for (std::size_t rot = 0; rot < 3; ++rot) {
                std::array<std::array<bool, 3>, 2> on{};
                for (std::size_t pl = 0; pl < nr; ++pl) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        const P3& qq = q[(rot + j) % 3];
                        on[pl][j] = std::abs(dot(ridge_planes[pl], qq)) <=
                                    kOnPlaneTol * norm(ridge_planes[pl]) * norm(qq);
                    }
                }
                const BaseGrading g = base_grading(on, nr);
                if (g.cost() < best_cost) {
                    best_cost = g.cost();
                    best = g;
                    best_rot = rot;
                }
            }
            const P3 q0 = q[best_rot];
            const P3 q1 = q[(best_rot + 1) % 3];
            const P3 q2 = q[(best_rot + 2) % 3];
            line_rule(ul, rp.n, rp.ng, best.u0, best.u1, rp.fine * rg.eps);
            line_rule(vl, rp.n, rp.ng, best.v0, best.v1, rp.fine * rg.eps);
            for (std::size_t ir = 0; ir < ni; ++ir) {
                const Real rho = gl.x[ni][ir];
                const Real wr = gl.w[ni][ir];
                for (std::size_t iu = 0; iu < ul.n; ++iu) {
                    const Real u = ul.x[iu];
                    for (std::size_t iv = 0; iv < vl.n; ++iv) {
                        const Real v = vl.x[iv];
                        const P3 pb = q0 + u * ((q1 - q0) + v * (q2 - q1));
                        const Real z = rho * pb.x;
                        const Real t = rho * pb.y;
                        const Real tp = rho * pb.z;
                        const Real slo = std::max(0.0, -z);
                        const Real shi = std::min(1.0 - t, 1.0 - tp - z);
                        if (!(shi > slo)) {
                            continue;
                        }
                        const Real w0 =
                            jac * det * rho * rho * u * wr * ul.w[iu] * vl.w[iv] * (shi - slo);
                        const P3 dxy = (-z) * e + t * a - tp * b;
                        for (std::size_t is = 0; is < ns; ++is) {
                            const Real s = slo + (shi - slo) * gl.x[ns][is];
                            visit(s * e + t * a, (s + z) * e + tp * b, dxy, w0 * gl.w[ns][is]);
                        }
                    }
                }
            }
        }
    };
    // Each cone base is subdivided into 4^kEdgeBaseLevels congruent triangles (midpoints): on
    // whole facets the rule converges slowly (file comment).
    const int levels = kEdgeBaseLevels;
    std::array<std::array<P3, 3>, 64> sub{};
    for (std::size_t k = 0; k < nw; ++k) {
        const Poly& f = work[k];
        for (std::size_t i = 1; i + 1 < f.n; ++i) {
            std::size_t nsub = 0;
            sub[nsub++] = {f.v[0], f.v[i], f.v[i + 1]};
            for (int l = 0; l < levels; ++l) {
                const std::size_t m = nsub;
                for (std::size_t j = 0; j < m; ++j) {
                    const std::array<P3, 3> q = sub[j];
                    const P3 m01 = 0.5 * (q[0] + q[1]);
                    const P3 m12 = 0.5 * (q[1] + q[2]);
                    const P3 m20 = 0.5 * (q[2] + q[0]);
                    sub[j] = {q[0], m01, m20};
                    sub[nsub++] = {m01, q[1], m12};
                    sub[nsub++] = {m20, m12, q[2]};
                    sub[nsub++] = {m01, m12, m20};
                }
            }
            for (std::size_t j = 0; j < nsub; ++j) {
                base(sub[j]);
            }
        }
    }
}

/// Tetrahedron in 3D parameter space.
using Tet = std::array<P3, 4>;

/// Splits tet by the plane g . p + c0 = 0 into tetrahedra (appended to out).
template <std::size_t N>
void split_tet(const Tet& tet, const P3& g, Real c0, std::array<Tet, N>& out, std::size_t& count) {
    std::array<Real, 4> s{};
    Real scale = 1.0;
    for (const P3& p : tet) {
        scale = std::max(scale, norm(p));
    }
    const Real tol = kOnPlaneTol * std::max(norm(g), std::abs(c0)) * scale;
    std::array<int, 4> cls{};
    std::array<std::size_t, 4> pos{}, neg{}, zer{};
    std::size_t np = 0, nn = 0, nz = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        s[i] = dot(g, tet[i]) + c0;
        cls[i] = std::abs(s[i]) <= tol ? 0 : (s[i] > 0.0 ? 1 : -1);
        if (cls[i] > 0) {
            pos[np++] = i;
        } else if (cls[i] < 0) {
            neg[nn++] = i;
        } else {
            zer[nz++] = i;
        }
    }
    const auto emit = [&](const P3& a, const P3& b, const P3& c, const P3& d) {
        if (std::abs(det3(b - a, c - a, d - a)) <= 1e-14 * scale * scale * scale) {
            return;
        }
        if (count >= N) {
            throw std::logic_error("stretched element blocks: tetrahedron capacity exceeded");
        }
        out[count++] = {a, b, c, d};
    };
    if (np == 0 || nn == 0) {
        emit(tet[0], tet[1], tet[2], tet[3]);
        return;
    }
    const auto cut = [&](std::size_t i, std::size_t j) {
        return tet[i] + (s[i] / (s[i] - s[j])) * (tet[j] - tet[i]);
    };
    // Prism (v0, v1, v2) - (w0, w1, w2) -> three tetrahedra.
    const auto prism = [&](const P3& v0, const P3& v1, const P3& v2, const P3& w0, const P3& w1,
                           const P3& w2) {
        emit(v0, v1, v2, w2);
        emit(v0, v1, w1, w2);
        emit(v0, w0, w1, w2);
    };
    if (nz == 2) {
        const P3 p = cut(pos[0], neg[0]);
        emit(tet[zer[0]], tet[zer[1]], tet[pos[0]], p);
        emit(tet[zer[0]], tet[zer[1]], tet[neg[0]], p);
    } else if (nz == 1) {
        const bool one_pos = np == 1;
        const std::size_t one = one_pos ? pos[0] : neg[0];
        const std::size_t t0 = one_pos ? neg[0] : pos[0];
        const std::size_t t1 = one_pos ? neg[1] : pos[1];
        const P3 p1 = cut(one, t0);
        const P3 p2 = cut(one, t1);
        const P3& z0 = tet[zer[0]];
        emit(tet[one], z0, p1, p2);
        // Pyramid with apex z0 over the quad (t0, t1, p2, p1).
        emit(z0, tet[t0], tet[t1], p2);
        emit(z0, tet[t0], p2, p1);
    } else if (np == 1 || nn == 1) {
        const bool one_pos = np == 1;
        const std::size_t one = one_pos ? pos[0] : neg[0];
        const std::array<std::size_t, 3> three =
            one_pos ? std::array<std::size_t, 3>{neg[0], neg[1], neg[2]}
                    : std::array<std::size_t, 3>{pos[0], pos[1], pos[2]};
        const P3 c0p = cut(one, three[0]);
        const P3 c1p = cut(one, three[1]);
        const P3 c2p = cut(one, three[2]);
        emit(tet[one], c0p, c1p, c2p);
        prism(tet[three[0]], tet[three[1]], tet[three[2]], c0p, c1p, c2p);
    } else {
        // 2 - 2: two prisms.
        for (int side = 0; side < 2; ++side) {
            const std::size_t x0 = side == 0 ? pos[0] : neg[0];
            const std::size_t x1 = side == 0 ? pos[1] : neg[1];
            const std::size_t y0 = side == 0 ? neg[0] : pos[0];
            const std::size_t y1 = side == 0 ? neg[1] : pos[1];
            prism(tet[x0], cut(x0, y0), cut(x0, y1), tet[x1], cut(x1, y0), cut(x1, y1));
        }
    }
}

/// Graded directions of a collapsed tetrahedron p = q0 + a [(q1 - q0) + b ((q2 - q1) + c (q3 -
/// q2))]: index 0 / 1 for the ends 0 / 1 of a, b, c.
struct TetGrading {
    std::array<std::array<bool, 2>, 3> g{};
    [[nodiscard]] int cost() const {
        int c = 1;
        for (const auto& d : g) {
            c *= 1 + static_cast<int>(d[0]) + static_cast<int>(d[1]);
        }
        return c;
    }
};

/// Features (vertex sets on a ridge plane) mapped to graded ends (see TetGrading).
void tet_feature(const std::array<bool, 4>& on, TetGrading& tg) {
    const int n = static_cast<int>(on[0]) + static_cast<int>(on[1]) + static_cast<int>(on[2]) +
                  static_cast<int>(on[3]);
    auto& g = tg.g;
    if (n == 3) {
        if (!on[0]) {
            g[0][1] = true;  // face q1 q2 q3: a -> 1
        } else if (!on[1]) {
            g[1][1] = true;  // face q0 q2 q3: b -> 1
        } else if (!on[3]) {
            g[2][0] = true;  // face q0 q1 q2: c -> 0
        } else {
            g[2][1] = true;  // face q0 q1 q3: c -> 1
        }
    } else if (n == 2) {
        if (on[0] && on[1]) {
            g[1][0] = true;
        } else if (on[0] && on[2]) {
            g[1][1] = true;
            g[2][0] = true;
        } else if (on[0] && on[3]) {
            g[1][1] = true;
            g[2][1] = true;
        } else if (on[1] && on[2]) {
            g[0][1] = true;
            g[2][0] = true;
        } else if (on[1] && on[3]) {
            g[0][1] = true;
            g[2][1] = true;
        } else {
            g[0][1] = true;
            g[1][1] = true;
        }
    } else if (n == 1) {
        if (on[0]) {
            g[0][0] = true;
        } else if (on[1]) {
            g[0][1] = true;
            g[1][0] = true;
        } else if (on[2]) {
            g[0][1] = true;
            g[1][1] = true;
            g[2][0] = true;
        } else {
            g[0][1] = true;
            g[1][1] = true;
            g[2][1] = true;
        }
    }
}

/// Shared vertex (A origin; da1 = a1 - A, da2 = a2 - A, db1, db2 likewise).
template <class Visit>
void rule_shared_vertex(const P3& da1, const P3& da2, const P3& db1, const P3& db2, const Ridge& rg,
                        const RuleParams& rp, Visit&& visit) {
    const Real jac = norm(cross(da1, da2)) * norm(cross(db1, db2));
    const GlTables& gl = gl_tables();
    const auto ni = static_cast<std::size_t>(rp.nr);
    constexpr std::size_t kCap = 128;
    std::array<Tet, kCap> tets{};
    std::array<Tet, kCap> next{};
    Line al;
    Line bl;
    Line cl;
    // The 24 vertex orders of a tetrahedron.
    std::array<std::array<std::size_t, 4>, 24> perms{};
    {
        std::array<std::size_t, 4> p = {0, 1, 2, 3};
        std::size_t i = 0;
        do {
            perms[i++] = p;
        } while (std::next_permutation(p.begin(), p.end()));
    }
    for (const bool edge_on_test : {true, false}) {
        // d / lambda = w1(p) - w2(p), affine in p = (sigma, t0, t1).
        const auto omega1 = [&](const P3& p) {
            return edge_on_test ? p.x * da1 + (1.0 - p.x) * da2 : p.y * da1 + p.z * da2;
        };
        const auto omega2 = [&](const P3& p) {
            return edge_on_test ? p.y * db1 + p.z * db2 : p.x * db1 + (1.0 - p.x) * db2;
        };
        const P3 P0{0, 0, 0}, P1{0, 1, 0}, P2{0, 0, 1}, P3_{1, 0, 0}, P4{1, 1, 0}, P5{1, 0, 1};
        std::size_t nt = 0;
        tets[nt++] = {P0, P1, P2, P5};
        tets[nt++] = {P0, P1, P4, P5};
        tets[nt++] = {P0, P3_, P4, P5};
        std::array<P3, 2> g{};
        std::array<Real, 2> c0{};
        std::size_t nr = 0;
        if (rg.active) {
            for (std::size_t br = 0; br < 2; ++br) {
                const P3 zero{0, 0, 0};
                const Real f0 = ridge_form(rg, br, omega1(zero) - omega2(zero));
                const P3 ex{1, 0, 0}, ey{0, 1, 0}, ez{0, 0, 1};
                const P3 gg{ridge_form(rg, br, omega1(ex) - omega2(ex)) - f0,
                            ridge_form(rg, br, omega1(ey) - omega2(ey)) - f0,
                            ridge_form(rg, br, omega1(ez) - omega2(ez)) - f0};
                if (norm(gg) > 0.0) {
                    g[nr] = gg;
                    c0[nr] = f0;
                    ++nr;
                }
            }
        }
        for (std::size_t pl = 0; pl < nr; ++pl) {
            std::size_t nn = 0;
            for (std::size_t k = 0; k < nt; ++k) {
                split_tet(tets[k], g[pl], c0[pl], next, nn);
            }
            std::swap(tets, next);
            nt = nn;
        }
        for (std::size_t k = 0; k < nt; ++k) {
            const Tet& t0 = tets[k];
            const Real vol6 = std::abs(det3(t0[1] - t0[0], t0[2] - t0[0], t0[3] - t0[0]));
            // On-plane flags per plane and vertex.
            std::array<std::array<bool, 4>, 2> on{};
            for (std::size_t pl = 0; pl < nr; ++pl) {
                const Real tol = kOnPlaneTol * std::max(norm(g[pl]), std::abs(c0[pl])) * 2.0;
                for (std::size_t j = 0; j < 4; ++j) {
                    on[pl][j] = std::abs(dot(g[pl], t0[j]) + c0[pl]) <= tol;
                }
            }
            TetGrading best;
            std::size_t best_p = 0;
            int best_cost = 1 << 20;
            for (std::size_t ip = 0; ip < perms.size(); ++ip) {
                TetGrading tg;
                for (std::size_t pl = 0; pl < nr; ++pl) {
                    std::array<bool, 4> o{};
                    for (std::size_t j = 0; j < 4; ++j) {
                        o[j] = on[pl][perms[ip][j]];
                    }
                    tet_feature(o, tg);
                }
                if (tg.cost() < best_cost) {
                    best_cost = tg.cost();
                    best = tg;
                    best_p = ip;
                }
            }
            const P3 q0 = t0[perms[best_p][0]];
            const P3 q1 = t0[perms[best_p][1]];
            const P3 q2 = t0[perms[best_p][2]];
            const P3 q3 = t0[perms[best_p][3]];
            line_rule(al, rp.n, rp.ng, best.g[0][0], best.g[0][1], rp.fine * rg.eps);
            line_rule(bl, rp.n, rp.ng, best.g[1][0], best.g[1][1], rp.fine * rg.eps);
            line_rule(cl, rp.n, rp.ng, best.g[2][0], best.g[2][1], rp.fine * rg.eps);
            for (std::size_t ia = 0; ia < al.n; ++ia) {
                const Real aa = al.x[ia];
                for (std::size_t ib = 0; ib < bl.n; ++ib) {
                    const Real bb = bl.x[ib];
                    const Real wab = vol6 * aa * aa * bb * al.w[ia] * bl.w[ib];
                    for (std::size_t ic = 0; ic < cl.n; ++ic) {
                        const Real cc = cl.x[ic];
                        const P3 p = q0 + aa * ((q1 - q0) + bb * ((q2 - q1) + cc * (q3 - q2)));
                        const P3 w1 = omega1(p);
                        const P3 w2 = omega2(p);
                        const P3 dw = w1 - w2;
                        const Real wp = jac * wab * cl.w[ic];
                        for (std::size_t il = 0; il < ni; ++il) {
                            const Real lam = gl.x[ni][il];
                            visit(lam * w1, lam * w2, lam * dw,
                                  wp * lam * lam * lam * gl.w[ni][il]);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pair setup.
// ---------------------------------------------------------------------------------------------

/// Vertex indices of t ordered with those shared with `other` first (cyclic order kept).
std::array<std::size_t, 3> shared_first(const Tri& t, const Tri& other, std::size_t& n_shared) {
    std::array<std::size_t, 3> order{};
    std::size_t n = 0;
    n_shared = 0;
    for (const bool want : {true, false}) {
        for (std::size_t i = 0; i < 3; ++i) {
            const bool sh =
                t.id[i] == other.id[0] || t.id[i] == other.id[1] || t.id[i] == other.id[2];
            if (sh == want) {
                order[n++] = i;
                if (sh) {
                    ++n_shared;
                }
            }
        }
    }
    return order;
}

bool coplanar_pair(const Tri& a, const Tri& b) {
    return norm(cross(a.n, b.n)) <= kCoplanarTol;
}

/// All six vertices at one height (horizontal pair in one plane: K = 0 exactly).
bool flat_pair(const Tri& a, const Tri& b) {
    const Real z0 = a.v[0].z;
    const Real tol = 1e-12 * std::max(a.longest, b.longest);
    for (const Tri* t : {&a, &b}) {
        for (const P3& v : t->v) {
            if (std::abs(v.z - z0) > tol) {
                return false;
            }
        }
    }
    return true;
}

/// The touching rule of the pair (t test, s source); calls visit for every point pair and
/// returns the rule origin. Sets ridge information. Translation key (stretched_touching_key):
/// the rule reads the edge vectors from the origin (u components as v.u - A.u), the shared-index
/// pattern (shared_first), the stretch at the mid-point of the centroids (zero along u inside a
/// strip) and n; any new coordinate-derived input must enter the key.
template <class Visit>
P3 touching_rule(Proximity prox, const Tri& tt, const Tri& ts, const StretchEval& se,
                 const RuleParams& n, bool coplanar, Ridge& rg, Visit&& visit) {
    rg = Ridge{};
    if (coplanar) {
        const StretchValue m = se.at(0.5 * (tt.centroid + ts.centroid));
        rg = ridge_of(m.sx, m.sy);
    }
    if (prox == Proximity::identical) {
        rule_identical(tt.v[1] - tt.v[0], tt.v[2] - tt.v[0], rg, n, visit);
        return tt.v[0];
    }
    std::size_t sh_t = 0;
    std::size_t sh_s = 0;
    const std::array<std::size_t, 3> ot = shared_first(tt, ts, sh_t);
    const std::array<std::size_t, 3> os = shared_first(ts, tt, sh_s);
    if (prox == Proximity::shared_edge) {
        const P3 A = tt.v[ot[0]];
        const P3 B = tt.v[ot[1]];
        const P3 C = tt.v[ot[2]];
        const P3 Cp = ts.v[os[2]];
        rule_shared_edge(B - A, C - A, Cp - A, rg, n, visit);
        return A;
    }
    const P3 A = tt.v[ot[0]];
    rule_shared_vertex(tt.v[ot[1]] - A, tt.v[ot[2]] - A, ts.v[os[1]] - A, ts.v[os[2]] - A, rg, n,
                       visit);
    return A;
}

// ---------------------------------------------------------------------------------------------
// Near / far degree selection (file comment).
// ---------------------------------------------------------------------------------------------

struct PlainChoice {
    int degree = 0;
    bool reached = true;
    Real ridge_width = 1.0;
    Real distance_factor = 1.0;
};

/// Angular distance of two directions modulo pi, in [0, pi / 2].
Real angle_mod_pi(Real a, Real b) {
    Real d = std::fmod(std::abs(a - b), kPi);
    return std::min(d, kPi - d);
}

PlainChoice choose_plain(Proximity prox, const Tri& ta, const Tri& tb, Complex k,
                         const OperatorOptions& opt, const StretchEval& se) {
    PlainChoice ch;
    // sigma ranges of the pair (sigma is non-decreasing in |x|; the x range of the pair may
    // contain x = 0).
    Real sx_lo = 1e300, sx_hi = 0.0, sy_lo = 1e300, sy_hi = 0.0;
    Real x_lo = 1e300, x_hi = -1e300, y_lo = 1e300, y_hi = -1e300;
    Real z_lo = 1e300, z_hi = -1e300;
    for (const Tri* t : {&ta, &tb}) {
        for (const P3& v : t->v) {
            const StretchValue sv = se.at(v);
            sx_lo = std::min(sx_lo, sv.sx);
            sx_hi = std::max(sx_hi, sv.sx);
            sy_lo = std::min(sy_lo, sv.sy);
            sy_hi = std::max(sy_hi, sv.sy);
            x_lo = std::min(x_lo, v.x);
            x_hi = std::max(x_hi, v.x);
            y_lo = std::min(y_lo, v.y);
            y_hi = std::max(y_hi, v.y);
            z_lo = std::min(z_lo, v.z);
            z_hi = std::max(z_hi, v.z);
        }
    }
    if (x_lo < 0.0 && x_hi > 0.0) {
        sx_lo = 0.0;
    }
    if (y_lo < 0.0 && y_hi > 0.0) {
        sy_lo = 0.0;
    }
    const P3 dc = ta.centroid - tb.centroid;
    const Real D = norm(dc);
    const Real h = std::max(ta.longest, tb.longest);
    const Real rho = ta.radius + tb.radius;
    // In-plane direction cone of the pair.
    const Real dxy = std::hypot(dc.x, dc.y);
    const Real theta_c = std::atan2(dc.y, dc.x);
    const Real alpha = dxy > rho ? std::asin(rho / dxy) : 0.5 * kPi;
    Real delta = 1.0;
    for (const Real sx : {sx_lo, sx_hi}) {
        for (const Real sy : {sy_lo, sy_hi}) {
            const Ridge rg = ridge_of(sx, sy);
            ch.ridge_width = std::min(ch.ridge_width, rg.eps);
            Real gap = 0.5 * kPi;
            for (const Real th : rg.angle) {
                gap = std::min(gap, angle_mod_pi(theta_c, th));
            }
            gap = std::max(0.0, gap - alpha);
            if (rg.eps < 1.0) {
                delta = std::min(delta, std::hypot(gap, rg.eps));
            }
        }
    }
    // Pairs that are not flat: the ridge between the stretched plane and the unstretched z axis,
    // at the elevation angle atan(Re c) of the vertical plane through the pair.
    if (z_hi - z_lo > 1e-12 * h) {
        const Real r_lb = std::max(D - rho, 1e-300);
        const Real elev = std::atan((z_hi - z_lo) / r_lb);
        const Ridge rz = ridge_of(std::max(sx_hi, sy_hi), 0.0);
        if (rz.eps < 1.0) {
            const Real ridge_elev =
                std::atan2(std::abs(std::sin(rz.angle[0])), std::abs(std::cos(rz.angle[0])));
            const Real gap = std::max(0.0, ridge_elev - elev);
            delta = std::min(delta, std::hypot(gap, rz.eps));
            ch.ridge_width = std::min(ch.ridge_width, rz.eps);
        }
    }
    ch.distance_factor = std::min(1.0, kRidgeDistance * delta);
    if (opt.target_accuracy == 0.0) {
        ch.degree = prox == Proximity::near ? opt.quad_degree_near : opt.quad_degree_far;
        return ch;
    }
    const Real s_max = std::sqrt(1.0 + std::max(sx_hi, sy_hi) * std::max(sx_hi, sy_hi));
    const Real inv = h / (D * ch.distance_factor);
    const Real kh = 0.15 * std::abs(k) * s_max * h;
    const Real log_kappa = 0.5 * std::log10(inv * inv + kh * kh);
    const bool non_flat = z_hi - z_lo > 1e-12 * h;
    const Real log_target = std::log10(opt.target_accuracy) - (non_flat ? kNonFlatLogMargin : 0.0);
    for (int d = opt.quad_degree_far; d < opt.quad_degree_near; ++d) {
        const bool allowed =
            positive_interior_degree(d) || (prox == Proximity::far && d == opt.quad_degree_far);
        if (allowed && plain_rule_log_error(d, log_kappa) <= log_target) {
            ch.degree = d;
            return ch;
        }
    }
    ch.degree = opt.quad_degree_near;
    ch.reached = plain_rule_log_error(opt.quad_degree_near, log_kappa) <= log_target;
    return ch;
}

constexpr std::size_t kMaxRulePoints = 79;

/// Double Dunavant rule on (tt, ts) of the given degree (origin tt.v[0]).
template <std::size_t NR>
void integrate_plain(const Tri& tt, const Tri& ts, int degree, const StretchEval& se,
                     Accum<NR>& acc) {
    const TriangleRule& rule = triangle_rule(degree);
    const std::size_t nq = rule.weights.size();
    std::array<P3, kMaxRulePoints> xo{}, xi{};
    std::array<Real, kMaxRulePoints> wo{}, wi{};
    std::array<StretchValue, kMaxRulePoints> so{}, si{};
    const P3 o = tt.v[0];
    for (std::size_t q = 0; q < nq; ++q) {
        const Vec3& l = rule.barycentric[q];
        xo[q] = l(0) * tt.v[0] + l(1) * tt.v[1] + l(2) * tt.v[2];
        xi[q] = l(0) * ts.v[0] + l(1) * ts.v[1] + l(2) * ts.v[2];
        wo[q] = rule.weights[q] * tt.area;
        wi[q] = rule.weights[q] * ts.area;
        so[q] = se.at(xo[q]);
        si[q] = se.at(xi[q]);
    }
    for (std::size_t i = 0; i < nq; ++i) {
        const P3 X = xo[i] - o;
        for (std::size_t j = 0; j < nq; ++j) {
            add_pair(acc, X, xi[j] - o, xo[i] - xi[j], so[i], si[j], wo[i] * wi[j]);
        }
    }
    finish_points(acc);
}

/// Points of the touching rules of a stretched pair. With p = stretched_touching_points and the
/// k-aware extra = max(0, ceil(0.75 |k| s_max h) - 4) (resolves exp(-j k R~) with |k R~| up to
/// |k| s_max h, h the larger longest edge; no change for |k| s_max h <= 5.3):
/// n = nr = p_eff = min(p + extra, kMaxGl) plain and radial points, ng = min(max(ng0, ceil(p_eff /
/// 2)), kMaxGl) points per graded panel with ng0 = max(4, ceil(0.6 p)) (6 for p = 10), finest
/// graded panel kFine eps. The coarsest graded panel spans half of its direction (a quarter when
/// both ends are graded), so ng points per panel resolve the oscillation of exp(-j k R~) at least
/// as finely as the p_eff plain points do; WP-PML2 raised ng by the full extra (ng0 + extra), which
/// WP-PML2c dropped: Ag self-convergence at p = 10 unchanged on the strips (2.9e-8, 2.0e-7,
/// 2.1e-7), corners 2.6e-10 -> 6.3e-10 and 9.8e-11 -> 9.5e-10 (vacuum, extra 0: unchanged), 16 %
/// fewer kernel evaluations in the dense assembly of the [stretched_cost] case. Measured
/// (WP-PML2, all touching pairs of the central triangle of a structured grid, strip sigma 1 / 2.6
/// / 3.4 and corner 3.4 / 1.5, vacuum and Ag, against p = 16): the error is set by n and nr
/// (n = 6: shared vertex 3e-5; nr = 6: Ag 2e-6); ng = 6 instead of 10 leaves every error within
/// 1e-3 relative of itself and saves 35-50 % of the kernel evaluations (shared vertex at sigma
/// 3.4: 2.3e6 -> 1.1e6 point pairs); ng = 4 or kFine = 1 cost accuracy on identical pairs
/// (4e-11 -> 4e-8).
RuleParams touching_points(const Tri& ta, const Tri& tb, Complex k, const OperatorOptions& opt,
                           const StretchEval& se) {
    Real smax = 0.0;
    for (const Tri* t : {&ta, &tb}) {
        for (const P3& v : t->v) {
            const StretchValue sv = se.at(v);
            smax = std::max({smax, sv.sx, sv.sy});
        }
    }
    const Real kh = std::abs(k) * std::sqrt(1.0 + smax * smax) * std::max(ta.longest, tb.longest);
    const int extra = std::max(0, static_cast<int>(std::ceil(0.75 * kh)) - 4);
    const int p0 = opt.stretched_touching_points;
    const int p = std::min(p0 + extra, kMaxGl);
    const int ng0 = std::max(4, (3 * p0 + 4) / 5);  // ceil(0.6 p0)
    return RuleParams{p, p, std::min(std::max(ng0, (p + 1) / 2), kMaxGl), kFine};
}

/// Validated pair data: geometry of both triangles and the proximity class.
struct PairSetup {
    Tri tt, ts;
    Proximity prox;
    bool touching;
};

PairSetup setup_pair(const geometry::TriangleMesh& mesh, Index t_test, Index t_src,
                     const OperatorOptions& opt, const StretchEval& se) {
    PairSetup ps{tri_of(mesh, t_test), tri_of(mesh, t_src),
                 classify(mesh, t_test, t_src, opt.near_distance_factor), false};
    ps.touching = ps.prox == Proximity::identical || ps.prox == Proximity::shared_edge ||
                  ps.prox == Proximity::shared_vertex;
    require_horizontal(se, ps.tt, t_test);
    require_horizontal(se, ps.ts, t_src);
    return ps;
}

bool same_rule(const RuleParams& a, const RuleParams& b) {
    return a.n == b.n && a.nr == b.nr && a.ng == b.ng && a.fine == b.fine;
}

/// Bound of |R~| over the point pairs of the pair for the fastmath decision (file comment
/// "Evaluation"): |R~| <= R sqrt(1 + sigma_max^2) (|R~^2| <= sum_i dx_i^2 |1 - j tau_i|^2) and
/// R <= D + rho_t + rho_s. From absolute centroids, so not translation-exact: the translation key
/// holds the decision (fast_region), not this value.
Real fast_r_max(const PairSetup& ps, const StretchEval& se) {
    const Real smax =
        std::max(se.x.active ? se.x.sigma_max : 0.0, se.y.active ? se.y.sigma_max : 0.0);
    return (norm(ps.tt.centroid - ps.ts.centroid) + ps.tt.radius + ps.ts.radius) *
           std::sqrt(1.0 + smax * smax);
}

/// The fastmath choice of one region (Accum::fast): opt.fast_plain_kernel, K = 0 and |k R~| <=
/// kFastSincosMax for every point pair.
bool fast_region(const OperatorOptions& opt, Complex k, bool need_k, Real r_max) {
    return !need_k && opt.fast_plain_kernel && std::isfinite(std::abs(k)) &&
           std::abs(k) * r_max <= fastmath::kFastSincosMax;
}

/// Origin of the touching rule (touching_rule's return value): v0 of the test triangle for
/// identical pairs, else its first vertex (local order) shared with the source triangle.
P3 touching_origin(const PairSetup& ps) {
    if (ps.prox == Proximity::identical) {
        return ps.tt.v[0];
    }
    std::size_t sh = 0;
    return ps.tt.v[shared_first(ps.tt, ps.ts, sh)[0]];
}

/// One region of a stretched_blocks call: its parameters and output blocks.
struct RegionJob {
    const RegionParams* region = nullptr;
    Eigen::Matrix<Complex, 3, 3>* L = nullptr;
    Eigen::Matrix<Complex, 3, 3>* K = nullptr;
};

/// The blocks of the pair (t_test, t_src) for nj regions (file comment "Regions"): the rule of
/// each region is selected on its own (near / far degree, touching points); regions with the same
/// rule share one pass over the point pairs, the others get a pass each. Every region's blocks
/// are therefore bitwise those of a call for that region alone.
void stretched_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                      const std::array<RegionJob, 2>& jobs, std::size_t nj,
                      const OperatorOptions& opt, const geometry::CoordinateStretch& stretch) {
    validate(opt);
    // Canonical ordering (file comment "Symmetry"): (min, max), transposed for the other.
    const bool swap = t_test > t_src;
    const Index t1 = swap ? t_src : t_test;
    const Index t2 = swap ? t_test : t_src;
    const Slots s1 = slots_of(space, t1);  // range checks
    const Slots s2 = slots_of(space, t2);
    for (std::size_t j = 0; j < nj; ++j) {
        jobs[j].L->setZero();
        jobs[j].K->setZero();
    }
    if (s1.count == 0 || s2.count == 0) {
        return;
    }
    const geometry::TriangleMesh& mesh = space.mesh();
    const StretchEval se(stretch);
    const PairSetup ps = setup_pair(mesh, t1, t2, opt, se);
    const auto kparams = [&](std::size_t j) {
        return KernelParams{jobs[j].region->k.real(), jobs[j].region->k.imag()};
    };
    // Fastmath kernel loop (file comment "Evaluation", fast_r_max / fast_region). Decided per
    // region (Accum::fast), so a shared pass keeps each region's single-region choice.
    // Translation key (stretched_touching_key): the touching path below reads the pair only through
    // ps (Tri, proximity), s1 / s2 (Slots), the fastmath decision, touching_points and the rule
    // origin; any new pair-dependent input of that path must enter the key.
    const Real r_max = fast_r_max(ps, se);
    const auto set_fast = [&]<std::size_t NR>(const std::array<std::size_t, NR>& idx, bool need_k,
                                              Accum<NR>& acc) {
        for (std::size_t r = 0; r < NR; ++r) {
            acc.fast[r] = fast_region(opt, jobs[idx[r]].region->k, need_k, r_max);
        }
    };
    // Blocks of job j from its moments (canonical ordering, then the requested one).
    const auto output = [&](std::size_t j, const Sums& sums, const P3& origin, bool need_k) {
        Eigen::Matrix<Complex, 3, 3> Lc;
        Eigen::Matrix<Complex, 3, 3> Kc;
        Lc.setZero();
        Kc.setZero();
        finish(sums, s1, s2, origin, *jobs[j].region, need_k, Lc, Kc);
        if (t1 == t2) {
            // Identical triangles: one ordering only; L is symmetric in exact arithmetic (K = 0
            // for the horizontal triangle), symmetrised as in operators.cpp (free).
            const Eigen::Matrix<Complex, 3, 3> raw = Lc;
            Lc = 0.5 * (raw + raw.transpose());
        }
        Eigen::Matrix<Complex, 3, 3>& L = *jobs[j].L;
        Eigen::Matrix<Complex, 3, 3>& K = *jobs[j].K;
        if (swap) {
            L = Lc.transpose();
            K = Kc.transpose();
        } else {
            L = Lc;
            K = Kc;
        }
        if (!(L.allFinite() && K.allFinite())) {
            throw std::invalid_argument(
                "element_blocks: non-finite stretched block for triangles " +
                std::to_string(t_test) + " and " + std::to_string(t_src) +
                " (geometrically intersecting or coincident triangles that do not share vertex "
                "indices, or invalid region parameters)");
        }
    };
    if (!ps.touching) {
        const bool need_k = !flat_pair(ps.tt, ps.ts);
        std::array<int, 2> degree{};
        for (std::size_t j = 0; j < nj; ++j) {
            degree[j] = choose_plain(ps.prox, ps.tt, ps.ts, jobs[j].region->k, opt, se).degree;
        }
        const P3 origin = ps.tt.v[0];
        const auto run = [&]<std::size_t NR>(const std::array<std::size_t, NR>& idx) {
            Accum<NR> acc;
            acc.need_k = need_k;
            set_fast(idx, need_k, acc);
            for (std::size_t r = 0; r < NR; ++r) {
                acc.kp[r] = kparams(idx[r]);
            }
            integrate_plain(ps.tt, ps.ts, degree[idx[0]], se, acc);
            for (std::size_t r = 0; r < NR; ++r) {
                output(idx[r], acc.sums[r], origin, need_k);
            }
        };
        if (nj == 2 && degree[0] == degree[1]) {
            run(std::array<std::size_t, 2>{0, 1});
        } else {
            for (std::size_t j = 0; j < nj; ++j) {
                run(std::array<std::size_t, 1>{j});
            }
        }
        return;
    }
    const bool coplanar = coplanar_pair(ps.tt, ps.ts);
    const bool need_k = !coplanar;
    std::array<RuleParams, 2> rule{};
    for (std::size_t j = 0; j < nj; ++j) {
        rule[j] = touching_points(ps.tt, ps.ts, jobs[j].region->k, opt, se);
    }
    // The visitor receives X, Y relative to the rule origin; the origin is known only after the
    // rule is set up, so it is computed first (identical: v0, else the shared vertex A).
    const P3 o = touching_origin(ps);
    const auto run = [&]<std::size_t NR>(const std::array<std::size_t, NR>& idx) {
        Accum<NR> acc;
        acc.need_k = need_k;
        set_fast(idx, need_k, acc);
        for (std::size_t r = 0; r < NR; ++r) {
            acc.kp[r] = kparams(idx[r]);
        }
        const auto visit = [&](const P3& X, const P3& Y, const P3& dxy, Real w) {
            add_pair(acc, X, Y, dxy, se.at(o + X), se.at(o + Y), w);
        };
        Ridge rg;
        const P3 origin =
            touching_rule(ps.prox, ps.tt, ps.ts, se, rule[idx[0]], coplanar, rg, visit);
        finish_points(acc);
        for (std::size_t r = 0; r < NR; ++r) {
            output(idx[r], acc.sums[r], origin, need_k);
        }
    };
    if (nj == 2 && same_rule(rule[0], rule[1])) {
        run(std::array<std::size_t, 2>{0, 1});
    } else {
        for (std::size_t j = 0; j < nj; ++j) {
            run(std::array<std::size_t, 1>{j});
        }
    }
}

/// Axis u (1 = y first, then 0 = x) along which the stretch is constant over the pair with the
/// margin of kTranslationStripMargin (stretched_touching_key): the axis is inactive, or every
/// vertex has |u| <= start - margin max(start, h). -1 if there is none (corners, strip ends).
int translation_axis(const StretchEval& se, const PairSetup& ps) {
    const Real h = std::max(ps.tt.longest, ps.ts.longest);
    const auto invariant = [&](const Axis& a, int d) {
        if (!a.active) {
            return true;
        }
        const Real limit = a.start - kTranslationStripMargin * std::max(a.start, h);
        for (const Tri* t : {&ps.tt, &ps.ts}) {
            for (const P3& v : t->v) {
                if (!(std::abs(d == 0 ? v.x : v.y) <= limit)) {
                    return false;
                }
            }
        }
        return true;
    };
    if (invariant(se.y, 1)) {
        return 1;
    }
    if (invariant(se.x, 0)) {
        return 0;
    }
    return -1;
}

}  // namespace

bool stretched_touching_key(const basis::RwgSpace& space, Index t_test, Index t_src,
                            const std::array<RegionParams, 2>& regions,
                            const std::array<bool, 2>& active, const OperatorOptions& opt,
                            const geometry::CoordinateStretch& stretch, StretchedTouchingKey& key) {
    validate(opt);
    key.axis = -1;
    key.words.clear();
    key.slot_vertex = {{{-1, -1, -1}, {-1, -1, -1}}};
    // The canonical ordering of stretched_blocks.
    const Index t1 = std::min(t_test, t_src);
    const Index t2 = std::max(t_test, t_src);
    const Slots s1 = slots_of(space, t1);
    const Slots s2 = slots_of(space, t2);
    if (t1 == t2 || s1.count == 0 || s2.count == 0 || !(active[0] || active[1]) ||
        stretch.is_identity()) {
        return false;
    }
    const StretchEval se(stretch);
    const PairSetup ps = setup_pair(space.mesh(), t1, t2, opt, se);
    if (!(stretched_tri(se, ps.tt) || stretched_tri(se, ps.ts)) || !ps.touching ||
        ps.prox == Proximity::identical) {
        return false;
    }
    const int u = translation_axis(se, ps);
    if (u < 0) {
        return false;
    }
    // The inputs of the touching path of stretched_blocks, as it reads them.
    const bool coplanar = coplanar_pair(ps.tt, ps.ts);
    const bool need_k = !coplanar;
    const Real r_max = fast_r_max(ps, se);
    std::vector<std::uint64_t>& w = key.words;
    const auto bits = [](Real v) { return std::bit_cast<std::uint64_t>(v); };
    const auto word = [](auto v) { return static_cast<std::uint64_t>(v); };
    w.push_back(word(u));
    w.push_back(word(static_cast<int>(ps.prox)));
    std::uint64_t pattern = 0;  // shared vertex indices (shared_first; duplicates of coordinates)
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (ps.tt.id[i] == ps.ts.id[j]) {
                pattern |= std::uint64_t{1} << (3 * i + j);
            }
        }
    }
    w.push_back(pattern);
    w.push_back(word(coplanar));
    for (std::size_t r = 0; r < 2; ++r) {
        if (!active[r]) {
            w.push_back(0);
            continue;
        }
        const RuleParams rp = touching_points(ps.tt, ps.ts, regions[r].k, opt, se);
        w.push_back(fast_region(opt, regions[r].k, need_k, r_max) ? std::uint64_t{3}
                                                                  : std::uint64_t{1});
        w.push_back(word(rp.n));
        w.push_back(word(rp.nr));
        w.push_back(word(rp.ng));
        w.push_back(bits(rp.fine));
    }
    const P3 o = touching_origin(ps);
    const std::array<const Tri*, 2> tris = {&ps.tt, &ps.ts};
    const std::array<const Slots*, 2> slots = {&s1, &s2};
    for (std::size_t i = 0; i < 2; ++i) {
        const Tri& g = *tris[i];
        for (const P3& v : g.v) {
            // Along u relative to the rule origin (the rule's edge vectors and finish's p - o; the
            // stretch does not depend on u here), across it absolute (the stretch does).
            w.push_back(bits(u == 0 ? v.x - o.x : v.x));
            w.push_back(bits(u == 1 ? v.y - o.y : v.y));
            w.push_back(bits(v.z));
        }
        w.push_back(bits(g.n.x));
        w.push_back(bits(g.n.y));
        w.push_back(bits(g.n.z));
        w.push_back(bits(g.area));
        w.push_back(bits(g.longest));
        // Per local vertex: the divergence of the slot whose free vertex it is (presence mask).
        std::uint64_t present = 0;
        std::array<std::uint64_t, 3> div{};
        const Slots& sl = *slots[i];
        for (std::size_t a = 0; a < sl.count; ++a) {
            for (std::size_t j = 0; j < 3; ++j) {
                if (g.id[j] == sl.vid[a]) {
                    present |= std::uint64_t{1} << j;
                    div[j] = bits(sl.div[a]);
                    key.slot_vertex[i][a] = static_cast<int>(j);
                    break;
                }
            }
            if (key.slot_vertex[i][a] < 0) {
                throw std::logic_error("stretched_touching_key: free vertex not on its triangle");
            }
        }
        w.push_back(present);
        w.insert(w.end(), div.begin(), div.end());
    }
    if (w.size() > kMaxStretchedTouchingKeyWords) {
        // Keeps op::estimate_sparse_stretched_cache an upper bound when inputs are added.
        throw std::logic_error(
            "stretched_touching_key: more words than kMaxStretchedTouchingKeyWords");
    }
    key.axis = u;
    return true;
}

bool triangle_stretched(const geometry::CoordinateStretch& stretch,
                        const geometry::TriangleMesh& mesh, Index t) {
    if (t < 0 || t >= mesh.num_triangles()) {
        throw std::out_of_range("element_blocks: triangle index " + std::to_string(t) +
                                " out of range");
    }
    if (stretch.is_identity()) {
        return false;
    }
    const StretchEval se(stretch);
    return stretched_tri(se, tri_of(mesh, t));
}

void stretched_element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                              const RegionParams& region, const OperatorOptions& opt,
                              const geometry::CoordinateStretch& stretch,
                              Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K) {
    const RegionJob job{&region, &L, &K};
    stretched_blocks(space, t_test, t_src, {job}, 1, opt, stretch);
}

void stretched_element_blocks_both(const basis::RwgSpace& space, Index t_test, Index t_src,
                                   const std::array<RegionParams, 2>& regions,
                                   const std::array<bool, 2>& active, const OperatorOptions& opt,
                                   const geometry::CoordinateStretch& stretch,
                                   std::array<Eigen::Matrix<Complex, 3, 3>, 2>& L,
                                   std::array<Eigen::Matrix<Complex, 3, 3>, 2>& K) {
    std::array<RegionJob, 2> jobs{};
    std::size_t nj = 0;
    for (std::size_t i = 0; i < 2; ++i) {
        L[i].setZero();
        K[i].setZero();
        if (active[i]) {
            jobs[nj++] = RegionJob{&regions[i], &L[i], &K[i]};
        }
    }
    if (nj == 0) {
        validate(opt);
        (void)slots_of(space, t_test);  // range checks as element_blocks
        (void)slots_of(space, t_src);
        return;
    }
    stretched_blocks(space, t_test, t_src, jobs, nj, opt, stretch);
}

StretchedPairInfo stretched_info(const geometry::TriangleMesh& mesh, Index t_test, Index t_src,
                                 Complex k, const OperatorOptions& opt,
                                 const geometry::CoordinateStretch& stretch) {
    validate(opt);
    StretchedPairInfo info;
    if (!(triangle_stretched(stretch, mesh, t_test) || triangle_stretched(stretch, mesh, t_src))) {
        return info;
    }
    const Index t1 = std::min(t_test, t_src);
    const Index t2 = std::max(t_test, t_src);
    const StretchEval se(stretch);
    const PairSetup ps = setup_pair(mesh, t1, t2, opt, se);
    info.stretched = true;
    info.proximity = ps.prox;
    if (!ps.touching) {
        const PlainChoice ch = choose_plain(ps.prox, ps.tt, ps.ts, k, opt, se);
        info.degree = ch.degree;
        info.target_reached = ch.reached;
        info.ridge_width = ch.ridge_width;
        info.distance_factor = ch.distance_factor;
        const auto nq = static_cast<Index>(triangle_rule(ch.degree).weights.size());
        info.points = nq * nq;
        return info;
    }
    info.coplanar = coplanar_pair(ps.tt, ps.ts);
    Ridge rg;
    Index count = 0;
    (void)touching_rule(ps.prox, ps.tt, ps.ts, se, touching_points(ps.tt, ps.ts, k, opt, se),
                        info.coplanar, rg, [&](const P3&, const P3&, const P3&, Real) { ++count; });
    info.points = count;
    info.ridge_width = rg.eps;
    info.ridge_split = rg.active;
    if (!info.coplanar) {
        // Non-coplanar touching pairs (a tilted taper triangle touching the PML start, gap 0) use
        // the rule without the ridge split; report the ridge width of their frozen metric anyway
        // (the in-plane ridge and the one between the stretched plane and the unstretched z axis,
        // as choose_plain), so the diagnostic shows how anisotropic the kernel is there.
        const StretchValue m = se.at(0.5 * (ps.tt.centroid + ps.ts.centroid));
        info.ridge_width =
            std::min(ridge_of(m.sx, m.sy).eps, ridge_of(std::max(m.sx, m.sy), 0.0).eps);
    }
    return info;
}

}  // namespace specklebem::kernels::detail
