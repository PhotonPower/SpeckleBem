/// @file singularity.cpp
/// Analytic static potential integrals over a flat triangle and the proximity classification
/// of triangle pairs (ADR 0004).
///
/// Notation follows Wilton et al. 1984 (IEEE TAP 32, 276, Sec. II, uniform source on a
/// polygon) and Graglia 1993 (IEEE TAP 41, 1448); the higher-order terms are the q = 1, 3
/// members of the recursive edge-sum family of Hanninen, Taskinen & Sarvas 2006 (PIER 63,
/// 243, Sec. 3, K_q = int_T R^q dS' and its vector forms). Equation numbers are not quoted
/// because the formulas below are re-derived here from the surface divergence theorem.
/// n = (v1 - v0) x (v2 - v0) / |...|, d = (r - v0) . n, rho = r - d n. Edge i runs from
/// a = v_i to b = v_{i+1} (counter-clockwise about n), with unit tangent s_i = (b - a)/|b - a|
/// and outward in-plane normal m_i = s_i x n. Per edge
///   l+ = (b - rho) . s_i,  l- = (a - rho) . s_i,  P0 = (a - rho) . m_i,
///   R+ = |r - b|,  R- = |r - a|,  R0^2 = P0^2 + d^2,
///   f_i  = int_edge 1/R dl' = ln((R+ + l+)/(R- + l-))
///   int_edge R dl'   = (R0^2 f_i + l+ R+ - l- R-) / 2
///   int_edge R^3 dl' = (l+ R+^3 - l- R-^3) / 4 + (3/4) R0^2 int_edge R dl'.
/// Omega >= 0 is the solid angle of T seen from r. Instead of Wilton's edge sum
/// sum_i [atan(P0 l+ / (R0^2 + |d| R+)) - atan(P0 l- / (R0^2 + |d| R-))], whose terms cancel
/// for small Omega (relative error ~1e-12 at Omega ~ 1e-4), it is evaluated with the
/// Van Oosterom & Strackee (IEEE TBME 30 (1983) 125) formula with a_i = v_i - r:
///   tan(Omega / 2) = |d| 2A / (|a0||a1||a2| + (a0.a1)|a2| + (a0.a2)|a1| + (a1.a2)|a0|),
/// using the exact numerator |a0 . (a1 x a2)| = |d| 2A and atan2 (Omega in [0, 2 pi]).
/// The surface integrals follow from the surface divergence theorem on the plane of T
/// (int_T grad_s g dS' = sum_i m_i int_edge g dl') and the identities
/// div_s((r' - rho)/R) = 1/R + d^2/R^3 and div_s((r' - rho) R) = 3R - d^2/R:
///   I_1R    = sum_i P0 f_i - |d| Omega                         (Wilton; Hanninen K_-1)
///   I_R     = (sum_i P0 int_edge R dl' + d^2 I_1R) / 3            (Hanninen K_1)
///   I_rho_R = sum_i m_i int_edge R dl' - d n I_1R                 (Wilton, linear source)
///   I_rhoR  = (1/3) sum_i m_i int_edge R^3 dl' - d n I_R          (Hanninen, vector K_1)
///   I_grad  = sum_i m_i f_i + n sign(d) Omega                     (Wilton / Graglia)
///
/// Numerical robustness:
///   * All lengths are scaled by the longest edge h before evaluation (results are rescaled
///     by h, h^2, h^3, h^4, 1), so the thresholds below are relative and nothing under- or
///     overflows for any SI length scale.
///   * In-plane threshold: for |d| <= 1e-12 h, r counts as lying in the plane of T: Omega
///     is not evaluated (the normal part of I_grad is exactly 0 and the |d| Omega term is
///     dropped, < 2 pi 1e-12 h), and r may be classified as on an edge. The true d is kept in
///     all other terms (R0^2, -d n I_1R, -d n I_R, d^2 I_1R).
///   * f_i is evaluated on a cancellation-free branch (Graglia's recommendation, using
///     (R + l)(R - l) = R0^2):
///       l- >= 0 (both endpoints ahead):  f = ln((R+ + l+)/(R- + l-)),
///       l+ <= 0 (both endpoints behind): f = ln((R- - l-)/(R+ - l+)),
///       l- < 0 < l+ (foot on the edge):  f = ln(((R+ + l+)/R0) ((R- - l-)/R0)), R0 > 0.
///   * r on the closed edge segment (in-plane, R0 <= 1e-12 h, l- <= 1e-12 h, l+ >= -1e-12 h;
///     this includes r at a vertex): f_i diverges logarithmically, while P0 f_i -> 0 and
///     R0^2 f_i -> 0, so these products are set to zero. The contribution m_i f_i of that
///     edge to the tangential I_grad is omitted (finite part, see singularity.hpp) and
///     StaticIntegrals::grad_finite_part is set.
///   * Omega is evaluated only off-plane (|d| > 1e-12 h); atan2 is finite for every input.
///     Near an edge Omega is intrinsically ill-conditioned (a lateral shift dP0 of r changes
///     it by ~dP0 / |d|), which no formula can avoid.
#include "specklebem/kernels/singularity.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace specklebem::kernels {

namespace {

/// Relative (to the longest edge) height below which r counts as lying in the plane of T,
/// and distance below which an in-plane r counts as lying on an edge line / at a vertex.
constexpr Real kInPlaneTol = 1e-12;
/// A triangle is degenerate if its area is below this times the squared longest edge.
constexpr Real kDegenerateAreaTol = 1e-12;

/// int_edge 1/R dl' on the cancellation-free branch; requires R0 > 0 when lm < 0 < lp.
Real edge_log(Real lp, Real lm, Real Rp, Real Rm, Real R0) {
    if (lm >= 0.0) {
        return std::log((Rp + lp) / (Rm + lm));
    }
    if (lp <= 0.0) {
        return std::log((Rm - lm) / (Rp - lp));
    }
    return std::log(((Rp + lp) / R0) * ((Rm - lm) / R0));
}

}  // namespace

StaticIntegrals static_integrals(const Vec3& r, const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    if (!(r.allFinite() && v0.allFinite() && v1.allFinite() && v2.allFinite())) {
        throw std::invalid_argument("static_integrals: non-finite observation point or vertex");
    }
    const Vec3 e01 = v1 - v0;
    const Vec3 e02 = v2 - v0;
    const Real l2max = std::max({e01.squaredNorm(), e02.squaredNorm(), (v2 - v1).squaredNorm()});
    const Vec3 cr = e01.cross(e02);
    const Real twice_area = cr.norm();
    if (!(0.5 * twice_area >= kDegenerateAreaTol * l2max) || !(l2max > 0.0)) {
        throw std::invalid_argument(
            "static_integrals: degenerate triangle (area below 1e-12 x longest edge^2)");
    }
    const Real h = std::sqrt(l2max);
    const Real inv_h = 1.0 / h;
    const Vec3 n = cr / twice_area;

    // Vertices relative to r, scaled by the longest edge.
    const std::array<Vec3, 3> p = {(v0 - r) * inv_h, (v1 - r) * inv_h, (v2 - r) * inv_h};
    const std::array<Real, 3> dist = {p[0].norm(), p[1].norm(), p[2].norm()};
    // (r - v) . n, averaged over the vertices. The true d is kept in every term; the
    // in-plane flag only skips Omega (normal part of I_grad = 0) and enables the on-edge test.
    const Real d = -n.dot(p[0] + p[1] + p[2]) / 3.0;
    const bool in_plane = std::abs(d) <= kInPlaneTol;
    const Real abs_d = std::abs(d);
    bool finite_part = false;

    Real sum_p0_f = 0.0;  // sum_i P0 f_i
    Real sum_p0_r = 0.0;  // sum_i P0 int_edge R
    Vec3 grad_t = Vec3::Zero();
    Vec3 rho_r_t = Vec3::Zero();
    Vec3 rho_r3_t = Vec3::Zero();

    for (std::size_t i = 0; i < 3; ++i) {
        const Vec3& a = p[i];
        const Vec3& b = p[(i + 1) % 3];
        const Vec3 ab = b - a;
        const Vec3 s = ab / ab.norm();
        const Vec3 m = s.cross(n);
        const Real lp = b.dot(s);
        const Real lm = a.dot(s);
        const Real p0 = 0.5 * (a + b).dot(m);
        const Real Rp = dist[(i + 1) % 3];
        const Real Rm = dist[i];
        const Real r0sq = p0 * p0 + d * d;

        const bool on_edge =
            in_plane && std::abs(p0) <= kInPlaneTol && lm <= kInPlaneTol && lp >= -kInPlaneTol;
        if (on_edge) {
            // P0 -> 0 and R0 -> 0: only the l R and l R^3 terms survive; f_i diverges and is
            // dropped from the tangential gradient (finite part).
            const Real int_r = 0.5 * (lp * Rp - lm * Rm);
            const Real int_r3 = 0.25 * (lp * Rp * Rp * Rp - lm * Rm * Rm * Rm);
            rho_r_t += m * int_r;
            rho_r3_t += m * int_r3;
            finite_part = true;
            continue;
        }
        const Real f = edge_log(lp, lm, Rp, Rm, std::sqrt(r0sq));
        const Real int_r = 0.5 * (r0sq * f + lp * Rp - lm * Rm);
        const Real int_r3 = 0.25 * (lp * Rp * Rp * Rp - lm * Rm * Rm * Rm) + 0.75 * r0sq * int_r;
        sum_p0_f += p0 * f;
        sum_p0_r += p0 * int_r;
        grad_t += m * f;
        rho_r_t += m * int_r;
        rho_r3_t += m * int_r3;
    }

    // Solid angle, off-plane only (Van Oosterom & Strackee). In-plane it is left at 0: the
    // |d| Omega term is then below 2 pi 1e-12 h and the normal part of I_grad is 0.
    Real omega = 0.0;
    if (!in_plane) {
        const Real den = dist[0] * dist[1] * dist[2] + p[0].dot(p[1]) * dist[2] +
                         p[0].dot(p[2]) * dist[1] + p[1].dot(p[2]) * dist[0];
        omega = 2.0 * std::atan2(abs_d * (twice_area * inv_h * inv_h), den);
    }

    const Real i_1r = sum_p0_f - abs_d * omega;
    const Real i_r = (sum_p0_r + d * d * i_1r) / 3.0;
    const Vec3 i_rho_r = rho_r_t - (d * i_1r) * n;
    const Vec3 i_rhor = rho_r3_t / 3.0 - (d * i_r) * n;
    Vec3 i_grad = grad_t;
    if (!in_plane) {
        i_grad += (d > 0.0 ? omega : -omega) * n;
    }

    const Real h2 = h * h;
    StaticIntegrals out{};
    out.I_1R = i_1r * h;
    out.I_rho_R = i_rho_r * h2;
    out.I_grad = i_grad;
    out.I_R = i_r * (h2 * h);
    out.I_rhoR = i_rhor * (h2 * h2);
    out.grad_finite_part = finite_part;
    return out;
}

Proximity classify(const geometry::TriangleMesh& m, Index t_test, Index t_src,
                   Real near_distance_factor) {
    const Index nt = m.num_triangles();
    if (t_test < 0 || t_test >= nt || t_src < 0 || t_src >= nt) {
        throw std::invalid_argument(
            "classify: triangle index out of range (t_test = " + std::to_string(t_test) +
            ", t_src = " + std::to_string(t_src) + ", " + std::to_string(nt) + " triangles)");
    }
    if (!(std::isfinite(near_distance_factor) && near_distance_factor >= 0.0)) {
        throw std::invalid_argument("classify: near_distance_factor must be finite and >= 0");
    }
    if (t_test == t_src) {
        return Proximity::identical;
    }
    const Triangles& tri = m.triangles();
    const Index* a = tri.data() + 3 * t_test;  // row-major n x 3
    const Index* b = tri.data() + 3 * t_src;
    int shared = 0;
    for (int i = 0; i < 3; ++i) {
        shared += (a[i] == b[0] ? 1 : 0) + (a[i] == b[1] ? 1 : 0) + (a[i] == b[2] ? 1 : 0);
    }
    if (shared >= 2) {
        return Proximity::shared_edge;
    }
    if (shared == 1) {
        return Proximity::shared_vertex;
    }
    // Longest edge of either triangle (plain scalar code: called for every candidate pair).
    const Real* v = m.vertices().data();  // row-major n x 3
    Real size_sq = 0.0;
    for (const Index* t : {a, b}) {
        for (int i = 0; i < 3; ++i) {
            const Real* p = v + 3 * t[i];
            const Real* q = v + 3 * t[(i + 1) % 3];
            const Real dx = q[0] - p[0];
            const Real dy = q[1] - p[1];
            const Real dz = q[2] - p[2];
            size_sq = std::max(size_sq, dx * dx + dy * dy + dz * dz);
        }
    }
    const Real dist_sq = (m.centroid(t_test) - m.centroid(t_src)).squaredNorm();
    const Real limit = near_distance_factor * near_distance_factor * size_sq;
    return dist_sq < limit ? Proximity::near : Proximity::far;
}

}  // namespace specklebem::kernels
