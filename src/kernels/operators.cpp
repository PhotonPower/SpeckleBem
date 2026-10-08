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
/// and K_mn = int f_m(r) . [(D_n / 2) Psi x (r - p_n)] dS.
///
/// Quadrature by proximity class (classify):
///  * far / near: Dunavant rules of degree quad_degree_far / quad_degree_near for both
///    integrals, plain kernel (the triangles do not touch, R > 0).
///  * identical, shared_edge, shared_vertex: outer rule of degree quad_degree_sing; inner
///    integral by singularity subtraction (ADR 0004):
///      exp(-jkR)/R = rem(R) + 1/R - (k^2/2) R,  rem(R) = [exp(-jkR) - 1 + k^2 R^2 / 2] / R,
///      grad' exp(-jkR)/R = grad' rem + grad'(1/R) - (k^2/2) grad' R.
///    rem and grad' rem are integrated with the degree-quad_degree_sing rule on T'; the
///    static terms analytically with static_integrals(r, T') (singularity.hpp):
///      int 1/R = I_1R,  int (r' - r)/R = I_rho_R,  int R = I_R,  int (r' - r) R = I_rhoR,
///      int grad'(1/R) = I_grad                      (grad'(1/R) = (r - r')/R^3),
///      int grad' R = int (r' - r)/R = I_rho_R.
///    With f_n = (D_n / 2)(r' - p_n) = (D_n / 2)[(r' - r) + (r - p_n)] this gives the four
///    subtraction identities
///      int f_n / R          = (D_n / 2)[I_rho_R + (r - p_n) I_1R],
///      int f_n R            = (D_n / 2)[I_rhoR  + (r - p_n) I_R],
///      int grad'(1/R) x f_n = (D_n / 2) I_grad  x (r - p_n),
///      int grad' R    x f_n = (D_n / 2) I_rho_R x (r - p_n),
///    the last two because grad'(1/R) and grad' R are parallel to (r' - r) and
///    (r' - r) x (r' - r) = 0. For identical (coplanar) triangles I_grad is the in-plane
///    principal value (normal part 0), which is the principal value of K; the jump term is
///    separate (jump_block).
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
/// R = 0 exactly: for identical triangles the outer and inner rules are the same and points
/// coincide (the "outer points never hit inner points" argument holds only for distinct
/// triangles), and there the remainder takes its limit (-jk, gradient 0).
///
/// Symmetric quadrature for touching pairs: the outer integral of a touching pair is the hard
/// part (the inner potential, as a function of r, is only finitely smooth at the shared edge
/// or vertex: r log r terms for L, log r for the K kernel of folded pairs), so a single
/// ordering is accurate only to the outer-rule error (about 1e-4 relative for L and 1e-2 for
/// the small K block of a folded shared-edge pair at degree 10, measured against references
/// with composite outer rules during WP7) and so is the asymmetry L(t1, t2) - L(t2, t1)^T. For
/// shared_edge and shared_vertex pairs the block is therefore the average of both orderings,
/// B = (B_raw(t1, t2) + B_raw(t2, t1)^T) / 2, which costs a second evaluation but makes the
/// blocks exactly (bitwise) symmetric, as the Galerkin matrix is in exact arithmetic; an
/// assembler may compute each unordered pair once. For identical triangles the L block is
/// replaced by (L + L^T) / 2 (no extra cost).
///
/// K of coplanar touching pairs (identical triangles, and shared-edge / shared-vertex pairs
/// with parallel normals) is set to exactly zero: f_m(r), grad' G (parallel to r - r') and
/// f_n(r') all lie in the common plane, so the triple product vanishes pointwise and the
/// principal value is 0; the computed sum would be rounding noise.
///
/// Input checks: the outer points of a positive-interior rule lie strictly inside T, so for a
/// valid, non-overlapping mesh they never lie on the boundary of T' (touching pairs share
/// only vertices and edges) and static_integrals cannot report grad_finite_part; a set flag
/// means an invalid mesh (std::invalid_argument). Non-touching pairs whose triangles
/// geometrically intersect (an invalid mesh, e.g. duplicated vertices) can produce R = 0 and
/// non-finite entries; the blocks are checked with allFinite() at the end of every call.
#include "specklebem/kernels/operators.hpp"

#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/kernels/singularity.hpp"

#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

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

/// Inner moments at one outer point r, without the factor 1/(4 pi):
/// phi0 = int e^{-jkR}/R, phi1 = int e^{-jkR}/R (r' - r), psi = int grad'(e^{-jkR}/R).
struct Moments {
    Complex phi0{0.0, 0.0};
    C3 phi1{};
    C3 psi{};
};

/// Plain kernel (non-touching pairs, R > 0).
Moments plain_moments(const P3& r, const Points& src, Complex k) {
    const Complex mjk = -kJ * k;
    Moments m;
    for (std::size_t q = 0; q < src.count; ++q) {
        const Real dx = src.r[q].x - r.x;  // r' - r
        const Real dy = src.r[q].y - r.y;
        const Real dz = src.r[q].z - r.z;
        const Real R = std::sqrt(dx * dx + dy * dy + dz * dz);
        const Complex g = src.w[q] * std::exp(mjk * R) / R;
        // grad' (e^{-jkR}/R) = (1 + jkR) e^{-jkR} / R^3 (r - r') = -(1 + jkR) e/R^3 (r' - r)
        const Complex gg = -g * (1.0 - mjk * R) / (R * R);
        m.phi0 += g;
        m.phi1.x += g * dx;
        m.phi1.y += g * dy;
        m.phi1.z += g * dz;
        m.psi.x += gg * dx;
        m.psi.y += gg * dy;
        m.psi.z += gg * dz;
    }
    return m;
}

/// Singularity subtraction (touching pairs): numerical remainder plus analytic static terms.
Moments singular_moments(const P3& r, const Points& src, const TriangleGeometry& g_src, Complex k) {
    const Complex jk = kJ * k;
    const Complex jk3 = jk * jk * jk;
    const Complex half_k2 = 0.5 * k * k;
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
            Complex s2{0.0, 0.0};
            for (int n = kSeriesLast; n >= 3; --n) {
                const Real a = kSeriesA[static_cast<std::size_t>(n - 3)];
                s1 = s1 * x + a;
                s2 = s2 * x + static_cast<Real>(n - 1) * a;
            }
            rem = jk * (x * x * s1 - 1.0);
            gq = jk3 * s2;
        } else {
            const Complex e = std::exp(-x);
            const Complex quad = half_k2 * (R * R);
            rem = (e - 1.0 + quad) / R;
            gq = (1.0 - (1.0 + x) * e + quad) / (R * R * R);
        }
        const Complex wr = src.w[q] * rem;
        const Complex wg = src.w[q] * gq;
        m.phi0 += wr;
        m.phi1.x += wr * dx;
        m.phi1.y += wr * dy;
        m.phi1.z += wr * dz;
        m.psi.x += wg * dx;
        m.psi.y += wg * dy;
        m.psi.z += wg * dz;
    }
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
    return m;
}

/// Running sums over the outer points (without 1/(4 pi) and the material factors); 3x3
/// blocks stored row-major (index 3 a + b).
struct Accumulator {
    std::array<Complex, 9> vec{};  // int int f_m . f_n e^{-jkR}/R
    std::array<Complex, 9> k{};    // int int f_m . (grad'(e^{-jkR}/R) x f_n)
    Complex scalar{0.0, 0.0};      // int int e^{-jkR}/R
};

void accumulate(const P3& r, Real w, const Moments& mo, const Slots& test, const Slots& src,
                Accumulator& acc) {
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
        u[b] = {h * (mo.psi.y * qz - mo.psi.z * qy), h * (mo.psi.z * qx - mo.psi.x * qz),
                h * (mo.psi.x * qy - mo.psi.y * qx)};
    }
    for (std::size_t a = 0; a < test.count; ++a) {
        const Real s = 0.5 * w * test.div[a];
        const Real fx = s * (r.x - test.p[a].x);
        const Real fy = s * (r.y - test.p[a].y);
        const Real fz = s * (r.z - test.p[a].z);
        for (std::size_t b = 0; b < src.count; ++b) {
            acc.vec[3 * a + b] += fx * v[b].x + fy * v[b].y + fz * v[b].z;
            acc.k[3 * a + b] += fx * u[b].x + fy * u[b].y + fz * u[b].z;
        }
    }
    acc.scalar += w * mo.phi0;
}

/// Outer quadrature over the test triangle with plain (non-touching) or singularity-subtracted
/// (touching) inner integrals over the source triangle; same rule for both.
Accumulator integrate(const Slots& test, const TriangleGeometry& g_test, const Slots& src,
                      const TriangleGeometry& g_src, const TriangleRule& rule, bool touching,
                      Complex k) {
    Points outer;
    Points inner;
    map_rule(rule, g_test, outer);
    map_rule(rule, g_src, inner);
    Accumulator acc;
    for (std::size_t i = 0; i < outer.count; ++i) {
        const P3& r = outer.r[i];
        const Moments mo =
            touching ? singular_moments(r, inner, g_src, k) : plain_moments(r, inner, k);
        accumulate(r, outer.w[i], mo, test, src, acc);
    }
    return acc;
}

void check_degree(int degree, const char* name, bool positive_interior) {
    if (degree < 1 || degree > kMaxDegree) {
        throw std::invalid_argument(std::string("OperatorOptions: ") + name +
                                    " must be in 1..20, got " + std::to_string(degree));
    }
    if (positive_interior && !positive_interior_table()[static_cast<std::size_t>(degree)]) {
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
    if (!(std::isfinite(opt.near_distance_factor) && opt.near_distance_factor >= 0.0)) {
        throw std::invalid_argument(
            "OperatorOptions: near_distance_factor must be finite and >= 0");
    }
}

void element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                    const RegionParams& region, const OperatorOptions& opt,
                    Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K) {
    validate(opt);
    const geometry::TriangleMesh& mesh = space.mesh();
    // support() range-checks both indices (std::out_of_range) before classify is called.
    const Slots test = slots_of(space, t_test);
    const Slots src = slots_of(space, t_src);
    L.setZero();
    K.setZero();
    if (test.count == 0 || src.count == 0) {
        return;
    }
    const Proximity prox = classify(mesh, t_test, t_src, opt.near_distance_factor);
    const bool touching = prox == Proximity::identical || prox == Proximity::shared_edge ||
                          prox == Proximity::shared_vertex;
    const int degree = touching
                           ? opt.quad_degree_sing
                           : (prox == Proximity::near ? opt.quad_degree_near : opt.quad_degree_far);
    const TriangleRule& rule = triangle_rule(degree);
    const TriangleGeometry g_test = triangle_geometry(mesh, t_test);
    const TriangleGeometry g_src = triangle_geometry(mesh, t_src);

    Accumulator acc = integrate(test, g_test, src, g_src, rule, touching, region.k);
    if (prox == Proximity::shared_edge || prox == Proximity::shared_vertex) {
        // Symmetric Galerkin quadrature: average with the transposed block of the swapped
        // pair (outer rule on the other triangle). Element-wise 0.5 (x + y) is commutative,
        // so element_blocks(t_src, t_test) is bitwise the transpose of this block.
        const Accumulator rev = integrate(src, g_src, test, g_test, rule, true, region.k);
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                acc.vec[3 * a + b] = 0.5 * (acc.vec[3 * a + b] + rev.vec[3 * b + a]);
                acc.k[3 * a + b] = 0.5 * (acc.k[3 * a + b] + rev.k[3 * b + a]);
            }
        }
        acc.scalar = 0.5 * (acc.scalar + rev.scalar);
    } else if (prox == Proximity::identical) {
        const std::array<Complex, 9> raw = acc.vec;
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                acc.vec[3 * a + b] = 0.5 * (raw[3 * a + b] + raw[3 * b + a]);
            }
        }
    }

    const Complex c_vec = kJ * region.omega * region.mu * kInv4Pi;
    const Complex c_sca = kInv4Pi / (kJ * region.omega * region.eps);
    // K of coplanar touching pairs is exactly zero (file comment).
    const bool k_zero =
        prox == Proximity::identical ||
        (touching && mesh.normal(t_test).cross(mesh.normal(t_src)).norm() <= kCoplanarTol);
    for (std::size_t a = 0; a < test.count; ++a) {
        const auto ai = static_cast<Eigen::Index>(a);
        for (std::size_t b = 0; b < src.count; ++b) {
            const auto bi = static_cast<Eigen::Index>(b);
            L(ai, bi) =
                c_vec * acc.vec[3 * a + b] + c_sca * (test.div[a] * src.div[b]) * acc.scalar;
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
