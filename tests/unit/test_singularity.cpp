#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/kernels/singularity.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace specklebem;
using kernels::classify;
using kernels::Proximity;
using kernels::static_integrals;
using kernels::StaticIntegrals;

namespace {

constexpr Real kPi = constants::pi;
constexpr Real kMicron = 1e-6;
/// In-plane points: the implementation adds no normal term, so the normal component of
/// I_grad is zero up to the rounding of the edge normals m_i = s_i x n (and of the test's
/// own n), i.e. ~1e-16 relative.
constexpr Real kInPlaneNormalTol = 1e-14;

struct Tri {
    Vec3 v0, v1, v2;
    [[nodiscard]] Vec3 vertex(int i) const { return i == 0 ? v0 : (i == 1 ? v1 : v2); }
    [[nodiscard]] Vec3 centroid() const { return (v0 + v1 + v2) / 3.0; }
    [[nodiscard]] Vec3 normal() const { return (v1 - v0).cross(v2 - v0).normalized(); }
    [[nodiscard]] Real longest_edge() const {
        return std::max({(v1 - v0).norm(), (v2 - v1).norm(), (v0 - v2).norm()});
    }
    [[nodiscard]] Vec3 point(Real l0, Real l1, Real l2) const {
        return l0 * v0 + l1 * v1 + l2 * v2;
    }
};

StaticIntegrals eval(const Vec3& r, const Tri& t) {
    return static_integrals(r, t.v0, t.v1, t.v2);
}

/// Accumulator for the numerical references (same members as StaticIntegrals).
struct Acc {
    Real i_1r = 0.0;
    Vec3 i_rho_r = Vec3::Zero();
    Vec3 i_grad = Vec3::Zero();
    Real i_r = 0.0;
    Vec3 i_rhor = Vec3::Zero();
};

Real rel_scalar(Real a, Real b) {
    return std::abs(a - b) / std::abs(b);
}
Real rel_vec(const Vec3& a, const Vec3& b) {
    return (a - b).norm() / b.norm();
}

/// Largest relative error over the members (vectors relative to their norm). I_grad is
/// skipped when `with_grad` is false.
Real max_rel_error(const StaticIntegrals& s, const Acc& ref, bool with_grad = true) {
    Real e = std::max({rel_scalar(s.I_1R, ref.i_1r), rel_vec(s.I_rho_R, ref.i_rho_r),
                       rel_scalar(s.I_R, ref.i_r), rel_vec(s.I_rhoR, ref.i_rhor)});
    if (with_grad) {
        e = std::max(e, rel_vec(s.I_grad, ref.i_grad));
    }
    return e;
}

bool all_finite(const StaticIntegrals& s) {
    return std::isfinite(s.I_1R) && std::isfinite(s.I_R) && s.I_rho_R.allFinite() &&
           s.I_grad.allFinite() && s.I_rhoR.allFinite();
}

/// Composite Dunavant quadrature: T split into n_sub^2 congruent sub-triangles, rule of the
/// given degree on each. Only valid for r away from T. The plain degree-20 rule (n_sub = 1)
/// reaches only ~4e-12 relative at one edge length from the centroid, so the 1e-12 checks use
/// n_sub = 4 (16 sub-triangles x 79 points).
Acc dunavant_reference(const Vec3& r, const Tri& t, int degree, int n_sub) {
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const Vec3 a = (t.v1 - t.v0) / n_sub;
    const Vec3 b = (t.v2 - t.v0) / n_sub;
    const Real sub_area = 0.5 * a.cross(b).norm();
    // Plain scalar accumulation (fast also in the unoptimised sanitizer build).
    std::array<Real, 11> sum{};  // 1/R, (x-r)/R [3], (r-x)/R^3 [3], R, (x-r) R [3]
    const auto integrate_sub = [&](const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            std::array<Real, 3> dx{};
            for (Eigen::Index k = 0; k < 3; ++k) {
                dx[static_cast<std::size_t>(k)] = l(0) * p0(k) + l(1) * p1(k) + l(2) * p2(k) - r(k);
            }
            const Real R = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
            const Real w = rule.weights[q] * sub_area;
            const Real w_r = w / R;
            const Real w_r3 = w_r / (R * R);
            sum[0] += w_r;
            sum[7] += w * R;
            for (std::size_t k = 0; k < 3; ++k) {
                sum[1 + k] += w_r * dx[k];
                sum[4 + k] -= w_r3 * dx[k];
                sum[8 + k] += w * R * dx[k];
            }
        }
    };
    for (int i = 0; i < n_sub; ++i) {
        for (int j = 0; i + j < n_sub; ++j) {
            const Vec3 p = t.v0 + i * a + j * b;
            integrate_sub(p, p + a, p + b);
            if (i + j < n_sub - 1) {
                integrate_sub(p + a, p + a + b, p + b);
            }
        }
    }
    Acc acc;
    acc.i_1r = sum[0];
    acc.i_rho_r = Vec3(sum[1], sum[2], sum[3]);
    acc.i_grad = Vec3(sum[4], sum[5], sum[6]);
    acc.i_r = sum[7];
    acc.i_rhor = Vec3(sum[8], sum[9], sum[10]);
    return acc;
}

/// Composite Gauss-Legendre rule on [0, 1]: n_panels equal panels with n points each.
struct UnitRule {
    std::vector<Real> x;
    std::vector<Real> w;
};
UnitRule composite_gauss_legendre(int n_panels, int n) {
    const kernels::LineRule g = kernels::gauss_legendre(n);
    UnitRule r;
    const Real width = 1.0 / n_panels;
    for (int k = 0; k < n_panels; ++k) {
        for (std::size_t i = 0; i < g.nodes.size(); ++i) {
            r.x.push_back(width * (k + 0.5 * (g.nodes[i] + 1.0)));
            r.w.push_back(0.5 * width * g.weights[i]);
        }
    }
    return r;
}

/// Nested-quadrature reference for r in the plane of T (on T or outside): T is the signed sum
/// of the sub-triangles (r, v_i, v_{i+1}); each is mapped with the Duffy transform
/// r' = r + t w(s), w(s) = p + s (q - p) - r, dS' = 2 A_sub t ds dt, so that every integrand
/// below (1/R, (r' - r)/R, R, (r' - r) R) becomes smooth: polynomial in t and analytic in s.
/// Gauss-Legendre tensor rule: 16 panels of n_s points in s, 4 points in t (exact in t).
Acc duffy_reference(const Vec3& r, const Tri& t, int n_s) {
    const UnitRule gs = composite_gauss_legendre(16, n_s);
    const UnitRule gt = composite_gauss_legendre(1, 4);
    const Vec3 n = t.normal();
    const Real tiny = 1e-14 * t.longest_edge() * t.longest_edge();
    Acc acc;
    for (int e = 0; e < 3; ++e) {
        const Vec3 p = t.vertex(e);
        const Vec3 q = t.vertex((e + 1) % 3);
        const Real signed_twice_area = (p - r).cross(q - r).dot(n);
        if (std::abs(signed_twice_area) <= tiny) {
            continue;  // r on the line of this edge: the sub-triangle is empty
        }
        for (std::size_t is = 0; is < gs.x.size(); ++is) {
            const Real s = gs.x[is];
            const Vec3 w = p + s * (q - p) - r;
            const Real wn = w.norm();
            for (std::size_t it = 0; it < gt.x.size(); ++it) {
                const Real tt = gt.x[it];
                const Real jac = gs.w[is] * gt.w[it] * signed_twice_area * tt;
                acc.i_1r += jac / (tt * wn);
                acc.i_rho_r += (jac / wn) * w;
                acc.i_r += jac * tt * wn;
                acc.i_rhor += (jac * tt * tt * wn) * w;
            }
        }
    }
    return acc;
}

/// Duffy reference at two orders; requires convergence to 1e-12 (1000x below the 1e-9
/// acceptance level) and returns the finer one.
Acc converged_duffy(const Vec3& r, const Tri& t) {
    const Acc a = duffy_reference(r, t, 16);
    const Acc b = duffy_reference(r, t, 32);
    StaticIntegrals sa{};
    sa.I_1R = a.i_1r;
    sa.I_rho_R = a.i_rho_r;
    sa.I_R = a.i_r;
    sa.I_rhoR = a.i_rhor;
    CHECK(max_rel_error(sa, b, false) < 1e-12);
    return b;
}

/// Polar reference for the in-plane principal value of int grad'(1/R) over a polygon (r
/// inside or outside, but not on its boundary): with r' = r + rho u(phi),
///   PV int (r - r')/R^3 dS' = - sum_edges int_{sector} u(phi) ln rho_max(phi) dphi,
/// sectors signed (the ln(eps) terms cancel because int u dphi over a closed loop is 0).
/// Composite Gauss-Legendre in phi: 32 panels of n_phi points per edge sector.
Vec3 polar_pv_grad(const Vec3& r, const std::vector<Vec3>& poly, const Vec3& n, int n_phi) {
    const UnitRule g = composite_gauss_legendre(32, n_phi);
    const Vec3 e1 = n.unitOrthogonal();
    const Vec3 e2 = n.cross(e1);
    Vec3 sum = Vec3::Zero();
    for (std::size_t k = 0; k < poly.size(); ++k) {
        const Vec3 wp = poly[k] - r;
        const Vec3 wq = poly[(k + 1) % poly.size()] - r;
        const Real dphi = std::atan2(wp.cross(wq).dot(n), wp.dot(wq));
        if (std::abs(dphi) < 1e-14) {
            continue;  // r on the line of this edge, outside it: empty sector
        }
        const Real phi_p = std::atan2(wp.dot(e2), wp.dot(e1));
        const Vec3 edge = wq - wp;
        const Vec3 nu = edge.cross(n).normalized();  // unit normal of the edge line
        const Real h = wp.dot(nu);
        for (std::size_t i = 0; i < g.x.size(); ++i) {
            const Real phi = phi_p + dphi * g.x[i];
            const Vec3 u = std::cos(phi) * e1 + std::sin(phi) * e2;
            const Real rho = h / u.dot(nu);
            sum -= (dphi * g.w[i] * std::log(rho)) * u;
        }
    }
    return sum;
}

Vec3 converged_polar_pv_grad(const Vec3& r, const std::vector<Vec3>& poly, const Vec3& n) {
    const Vec3 a = polar_pv_grad(r, poly, n, 16);
    const Vec3 b = polar_pv_grad(r, poly, n, 32);
    CHECK((a - b).norm() <= 1e-12 * std::max(1.0, b.norm()));
    return b;
}

/// Independent reference for off-plane r (any height d != 0): T is the signed sum of the
/// sub-triangles (rho, v_i, v_{i+1}) about the projection rho = r - d n, each mapped with the
/// Duffy transform r' = rho + t w(s), w(s) = p + s (q - p) - rho, dS' = 2 A_sub t ds dt. Then
/// R^2 = d^2 + t^2 |w|^2 and the integrands vary on the scale t ~ c = |d| / |w(s)|, so the t
/// integral uses composite Gauss-Legendre on geometrically graded panels with breakpoints
/// 0, c, 2c, 4c, ..., 1 (a single panel if c >= 1); s uses 16 equal panels. n points per
/// panel in both variables. All five members, including the normal part of I_grad. Plain
/// scalar inner loop so that it is fast enough in the unoptimised sanitizer build.
Acc graded_duffy_reference(const Vec3& r, const Tri& t, int n) {
    const UnitRule gs = composite_gauss_legendre(16, n);
    const kernels::LineRule g = kernels::gauss_legendre(n);
    const std::size_t ng = g.nodes.size();
    const Vec3 nn = t.normal();
    const Real d = (r - t.centroid()).dot(nn);
    const Vec3 rho = r - d * nn;
    const Real abs_d = std::abs(d);
    const Real dn0 = d * nn(0);
    const Real dn1 = d * nn(1);
    const Real dn2 = d * nn(2);
    const Real tiny = 1e-14 * t.longest_edge() * t.longest_edge();
    Real s_1r = 0.0;
    Real s_r = 0.0;
    Real s_rho_r[3] = {0.0, 0.0, 0.0};
    Real s_grad[3] = {0.0, 0.0, 0.0};
    Real s_rhor[3] = {0.0, 0.0, 0.0};
    for (int e = 0; e < 3; ++e) {
        const Vec3 p = t.vertex(e);
        const Vec3 q = t.vertex((e + 1) % 3);
        const Real signed_twice_area = (p - rho).cross(q - rho).dot(nn);
        if (std::abs(signed_twice_area) <= tiny) {
            continue;  // rho on the line of this edge: empty sub-triangle
        }
        for (std::size_t is = 0; is < gs.x.size(); ++is) {
            const Vec3 w = p + gs.x[is] * (q - p) - rho;
            const Real w0 = w(0);
            const Real w1 = w(1);
            const Real w2 = w(2);
            const Real c = abs_d / w.norm();
            const Real ws = gs.w[is] * signed_twice_area;
            Real a = 0.0;
            Real b = std::min(c, 1.0);
            while (a < 1.0) {
                const Real half = 0.5 * (b - a);
                for (std::size_t it = 0; it < ng; ++it) {
                    const Real tt = a + half * (g.nodes[it] + 1.0);
                    const Real jac = ws * half * g.weights[it] * tt;
                    const Real x0 = tt * w0 - dn0;  // r' - r = t w - d n
                    const Real x1 = tt * w1 - dn1;
                    const Real x2 = tt * w2 - dn2;
                    const Real R = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
                    const Real j_r = jac / R;
                    const Real j_r3 = j_r / (R * R);
                    const Real j_rr = jac * R;
                    s_1r += j_r;
                    s_r += j_rr;
                    s_rho_r[0] += j_r * x0;
                    s_rho_r[1] += j_r * x1;
                    s_rho_r[2] += j_r * x2;
                    s_grad[0] -= j_r3 * x0;
                    s_grad[1] -= j_r3 * x1;
                    s_grad[2] -= j_r3 * x2;
                    s_rhor[0] += j_rr * x0;
                    s_rhor[1] += j_rr * x1;
                    s_rhor[2] += j_rr * x2;
                }
                a = b;
                b = std::min(2.0 * b, 1.0);
            }
        }
    }
    Acc acc;
    acc.i_1r = s_1r;
    acc.i_rho_r = Vec3(s_rho_r[0], s_rho_r[1], s_rho_r[2]);
    acc.i_grad = Vec3(s_grad[0], s_grad[1], s_grad[2]);
    acc.i_r = s_r;
    acc.i_rhor = Vec3(s_rhor[0], s_rhor[1], s_rhor[2]);
    return acc;
}

/// Graded Duffy reference at n = 10 and n = 16 points per panel; requires the two to agree to
/// 1e-11 (100x below the 1e-9 acceptance level) and returns the finer one.
Acc converged_graded_duffy(const Vec3& r, const Tri& t) {
    const Acc a = graded_duffy_reference(r, t, 10);
    const Acc b = graded_duffy_reference(r, t, 16);
    StaticIntegrals sa{};
    sa.I_1R = a.i_1r;
    sa.I_rho_R = a.i_rho_r;
    sa.I_grad = a.i_grad;
    sa.I_R = a.i_r;
    sa.I_rhoR = a.i_rhor;
    CHECK(max_rel_error(sa, b) < 1e-11);
    return b;
}

/// Max relative error vs the graded Duffy reference at base + d n for d in {1e-6, 1e-3, 0.3} h
/// (sign `side`), checking 1e-9 per point and that no finite part is reported.
Real check_off_plane_column(const Tri& t, const Vec3& base, Real side) {
    const Real h = t.longest_edge();
    Real worst = 0.0;
    for (const Real height : {1e-6, 1e-3, 0.3}) {
        const Vec3 r = base + side * height * h * t.normal();
        const StaticIntegrals s = eval(r, t);
        REQUIRE(all_finite(s));
        CHECK_FALSE(s.grad_finite_part);
        const Real err = max_rel_error(s, converged_graded_duffy(r, t));
        INFO("height " << height << ", side " << side << ": " << err);
        CHECK(err < 1e-9);
        worst = std::max(worst, err);
    }
    return worst;
}

/// Neighbour T' = (v1, v0, apex) sharing the edge v0 v1 of T, hinged at dihedral angle theta
/// (180 deg = coplanar continuation, 90 deg = perpendicular): the apex leaves the edge in the
/// direction cos(theta) (-m) + sin(theta) n, m the outward in-plane normal of the edge. At the
/// points of the Dunavant degree-6 rule (positive-interior, ADR 0004) on T', the integrals
/// over T must match the graded Duffy reference to 1e-9. Returns the max relative error.
Real check_hinged_neighbour(const Tri& t, Real theta_deg) {
    const int degree = 6;
    REQUIRE(kernels::triangle_rule_is_positive_interior(degree));
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const Real h = t.longest_edge();
    const Vec3 n = t.normal();
    const Vec3 m = (t.v1 - t.v0).normalized().cross(n);
    const Vec3 foot = t.v0 + 0.4 * (t.v1 - t.v0);
    const Real theta = theta_deg * kPi / 180.0;
    const Vec3 apex = foot + 0.8 * h * (std::cos(theta) * (-m) + std::sin(theta) * n);
    const Tri nb{t.v1, t.v0, apex};
    Real worst = 0.0;
    for (std::size_t q = 0; q < rule.weights.size(); ++q) {
        const Vec3& l = rule.barycentric[q];
        const Vec3 r = nb.point(l(0), l(1), l(2));
        const StaticIntegrals si = eval(r, t);
        REQUIRE(all_finite(si));
        CHECK_FALSE(si.grad_finite_part);
        const Real err = max_rel_error(si, converged_graded_duffy(r, t));
        INFO("theta " << theta_deg << ", point " << q << ": " << err);
        CHECK(err < 1e-9);
        worst = std::max(worst, err);
    }
    return worst;
}

/// Van Oosterom & Strackee (1983) signed solid angle of T seen from r; equals -sign(d) Omega
/// for the orientation n = (v1 - v0) x (v2 - v0).
Real vos_solid_angle(const Vec3& r, const Tri& t) {
    const Vec3 a = t.v0 - r;
    const Vec3 b = t.v1 - r;
    const Vec3 c = t.v2 - r;
    const Real an = a.norm();
    const Real bn = b.norm();
    const Real cn = c.norm();
    const Real num = a.dot((b - a).cross(c - a));  // = a . (b x c), better conditioned
    const Real den = an * bn * cn + a.dot(b) * cn + a.dot(c) * bn + b.dot(c) * an;
    return 2.0 * std::atan2(num, den);
}

Vec3 random_unit(std::mt19937_64& rng) {
    std::normal_distribution<Real> g(0.0, 1.0);
    Vec3 v(g(rng), g(rng), g(rng));
    return v.normalized();
}

/// Random triangle with edges ~1 um (0.6 .. 1.4 um), minimum angle >= 20 degrees, placed in
/// a few-micron box.
Tri random_triangle(std::mt19937_64& rng) {
    std::uniform_real_distribution<Real> pos(-3.0 * kMicron, 3.0 * kMicron);
    std::uniform_real_distribution<Real> len(0.6 * kMicron, 1.4 * kMicron);
    while (true) {
        const Vec3 v0(pos(rng), pos(rng), pos(rng));
        const Tri t{v0, v0 + len(rng) * random_unit(rng), v0 + len(rng) * random_unit(rng)};
        const std::array<Vec3, 3> e = {t.v1 - t.v0, t.v2 - t.v1, t.v0 - t.v2};
        Real min_angle = kPi;
        for (std::size_t i = 0; i < 3; ++i) {
            const Vec3 x = -e[(i + 2) % 3];
            const Vec3 y = e[i];
            min_angle = std::min(min_angle, std::acos(x.dot(y) / (x.norm() * y.norm())));
        }
        if (min_angle >= 20.0 * kPi / 180.0) {
            return t;
        }
    }
}

/// In-plane unit vector of T, random direction.
Vec3 random_in_plane(std::mt19937_64& rng, const Tri& t) {
    const Vec3 n = t.normal();
    Vec3 u = random_unit(rng);
    u -= u.dot(n) * n;
    return u.normalized();
}

std::vector<Tri> random_triangles(int count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Tri> ts;
    for (int i = 0; i < count; ++i) {
        ts.push_back(random_triangle(rng));
    }
    return ts;
}

/// Fixed generic (scalene, tilted) reference triangle, edges ~1 um.
Tri reference_triangle() {
    return Tri{Vec3(0.1, -0.2, 0.3) * kMicron, Vec3(1.15, 0.05, 0.2) * kMicron,
               Vec3(0.35, 0.9, 0.55) * kMicron};
}

}  // namespace

TEST_CASE("static integrals on separated triangles match composite Dunavant quadrature",
          "[kernels]") {
    std::mt19937_64 rng(20261008);
    Real worst = 0.0;
    Real worst_19 = 0.0;
    Real worst_omega = 0.0;
    int count = 0;
    for (int k = 0; k < 10; ++k) {
        const Tri t = random_triangle(rng);
        const Real L = t.longest_edge();
        const Vec3 c = t.centroid();
        for (int dist = 1; dist <= 5; ++dist) {
            for (const bool in_plane : {false, true}) {
                const Vec3 u = in_plane ? random_in_plane(rng, t) : random_unit(rng);
                const Vec3 r = c + (dist * L) * u;
                const StaticIntegrals s = eval(r, t);
                const Acc ref20 = dunavant_reference(r, t, 20, 4);
                const Acc ref19 = dunavant_reference(r, t, 19, 4);
                worst = std::max(worst, max_rel_error(s, ref20));
                worst_19 = std::max(worst_19, max_rel_error(s, ref19));
                // Normal component of I_grad = sign(d) Omega = -(signed VOS solid angle), also
                // relative to Omega itself (not only to |I_grad|), and vs Dunavant.
                const Real omega_vos = vos_solid_angle(r, t);
                const Real normal = s.I_grad.dot(t.normal());
                if (in_plane) {
                    CHECK(std::abs(normal) <= kInPlaneNormalTol * s.I_grad.norm());
                } else {
                    worst_omega =
                        std::max(worst_omega, std::abs(normal + omega_vos) / std::abs(omega_vos));
                    worst_omega =
                        std::max(worst_omega, std::abs(normal - ref20.i_grad.dot(t.normal())) /
                                                  std::abs(omega_vos));
                }
                ++count;
            }
        }
    }
    INFO("worst vs Dunavant-20: " << worst << ", vs Dunavant-19: " << worst_19
                                  << ", normal vs VOS: " << worst_omega << " over " << count);
    CHECK(worst < 1e-12);
    CHECK(worst_19 < 1e-12);
    CHECK(worst_omega < 1e-12);
}

TEST_CASE("static integrals on the triangle match the Duffy nested-quadrature reference",
          "[kernels]") {
    for (const Tri& t : random_triangles(4, 77)) {
        std::vector<Vec3> points = {t.centroid(), t.point(0.2, 0.3, 0.5), t.point(0.7, 0.2, 0.1)};
        for (int i = 0; i < 3; ++i) {
            const Vec3 a = t.vertex(i);
            const Vec3 b = t.vertex((i + 1) % 3);
            points.push_back(a);
            points.push_back(0.5 * (a + b));
        }
        Real worst = 0.0;
        for (const Vec3& r : points) {
            const StaticIntegrals s = eval(r, t);
            REQUIRE(all_finite(s));
            CHECK(std::abs(s.I_grad.dot(t.normal())) <= kInPlaneNormalTol * s.I_grad.norm());
            worst = std::max(worst, max_rel_error(s, converged_duffy(r, t), false));
        }
        INFO("worst relative error: " << worst);
        CHECK(worst < 1e-9);
    }
}

TEST_CASE("in-plane I_grad is the principal value at interior points", "[kernels]") {
    Real worst = 0.0;
    for (const Tri& t : random_triangles(4, 78)) {
        const std::vector<Vec3> poly = {t.v0, t.v1, t.v2};
        const Vec3 n = t.normal();
        for (const Vec3& r : {t.centroid(), t.point(0.2, 0.3, 0.5), t.point(0.7, 0.2, 0.1),
                              t.point(0.1, 0.1, 0.8)}) {
            const StaticIntegrals s = eval(r, t);
            const Vec3 ref = converged_polar_pv_grad(r, poly, n);
            CHECK(std::abs(s.I_grad.dot(n)) <= kInPlaneNormalTol * s.I_grad.norm());
            worst = std::max(worst, rel_vec(s.I_grad, ref));
        }
    }
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("normal component of I_grad jumps by 4 pi across the triangle", "[kernels]") {
    Real worst_ratio = 0.0;  // max |normal - limit| / delta
    Real worst_vos = 0.0;
    for (const Tri& t : random_triangles(3, 79)) {
        const Vec3 n = t.normal();
        const Real L = t.longest_edge();
        // Projection inside T (centroid) and outside T (beyond edge v0 v1, in-plane distance
        // ~0.3 L from the triangle).
        const Vec3 mid01 = 0.5 * (t.v0 + t.v1);
        const Vec3 outside = mid01 + 0.3 * L * (mid01 - t.v2).normalized();
        for (const bool inside : {true, false}) {
            const Vec3 r0 = inside ? t.centroid() : outside;
            const StaticIntegrals s0 = eval(r0, t);
            CHECK(std::abs(s0.I_grad.dot(n)) <= kInPlaneNormalTol * s0.I_grad.norm());
            for (const Real delta : {1e-3, 1e-6, 1e-9}) {
                for (const Real side : {1.0, -1.0}) {
                    const Vec3 r = r0 + side * delta * L * n;
                    const StaticIntegrals s = eval(r, t);
                    REQUIRE(all_finite(s));
                    const Real normal = s.I_grad.dot(n);
                    const Real limit = inside ? side * 2.0 * kPi : 0.0;
                    INFO("inside " << inside << ", delta " << delta << ", side " << side
                                   << ", normal " << normal);
                    CHECK(std::abs(normal - limit) <= 100.0 * delta);
                    CHECK(std::abs(normal - limit) > 0.0);
                    worst_ratio = std::max(worst_ratio, std::abs(normal - limit) / delta);
                    // Same value as Van Oosterom-Strackee (absolute, Omega <= 2 pi).
                    CHECK(std::abs(normal + vos_solid_angle(r, t)) <= 1e-13);
                    worst_vos = std::max(worst_vos, std::abs(normal + vos_solid_angle(r, t)));
                    // Tangential part and the potentials are continuous across the plane.
                    const Vec3 tan = s.I_grad - normal * n;
                    CHECK((tan - s0.I_grad).norm() <= 20.0 * delta);
                    CHECK(std::abs(s.I_1R - s0.I_1R) <= 20.0 * delta * L);
                    CHECK((s.I_rho_R - s0.I_rho_R).norm() <= 20.0 * delta * L * L);
                    CHECK(std::abs(s.I_R - s0.I_R) <= 20.0 * delta * L * L * L);
                    CHECK((s.I_rhoR - s0.I_rhoR).norm() <= 20.0 * delta * L * L * L * L);
                }
            }
        }
    }
    INFO("max |I_grad.n - limit| / delta: " << worst_ratio
                                            << ", max |I_grad.n - VOS|: " << worst_vos);
    CHECK(worst_vos <= 1e-13);
}

TEST_CASE("static integrals are finite and accurate at degenerate observation points",
          "[kernels]") {
    const Tri t = reference_triangle();
    const Real L = t.longest_edge();
    const Vec3 n = t.normal();
    const std::vector<Vec3> poly = {t.v0, t.v1, t.v2};
    Real worst_on = 0.0;
    Real worst_ext = 0.0;
    Real worst_grad_ext = 0.0;
    for (int i = 0; i < 3; ++i) {
        const Vec3 a = t.vertex(i);
        const Vec3 b = t.vertex((i + 1) % 3);
        const Vec3 dir = (b - a).normalized();
        // On T's boundary: vertex, edge interior.
        for (const Vec3& r : std::vector<Vec3>{a, 0.5 * (a + b), a + 0.3 * (b - a)}) {
            const StaticIntegrals s = eval(r, t);
            REQUIRE(all_finite(s));
            CHECK(std::abs(s.I_grad.dot(n)) <= kInPlaneNormalTol * s.I_grad.norm());
            worst_on = std::max(worst_on, max_rel_error(s, converged_duffy(r, t), false));
        }
        // On the extension of the edge line, both sides, outside T (in-plane).
        for (const Vec3& r :
             std::vector<Vec3>{b + 0.4 * L * dir, a - 0.4 * L * dir, b + 0.05 * L * dir}) {
            const StaticIntegrals s = eval(r, t);
            REQUIRE(all_finite(s));
            CHECK(std::abs(s.I_grad.dot(n)) <= kInPlaneNormalTol * s.I_grad.norm());
            worst_ext = std::max(worst_ext, max_rel_error(s, converged_duffy(r, t), false));
            const Vec3 ref = converged_polar_pv_grad(r, poly, n);
            worst_grad_ext = std::max(worst_grad_ext, rel_vec(s.I_grad, ref));
        }
        // Just above a vertex / an edge midpoint (off-plane, R0 = |d| tiny): finite and
        // continuous with the in-plane value; the normal part of I_grad tends to the interior
        // angle at the vertex / pi on the edge. Near an edge the solid angle is
        // ill-conditioned (dOmega ~ dP0 / d for a lateral perturbation dP0 of r, here the
        // coordinate rounding ~1e-16 L), so the comparison with Van Oosterom-Strackee uses an
        // absolute tolerance 1e-6 at d = 1e-9 L and 1e-9 at d = 1e-6 L.
        const Vec3 c = t.vertex((i + 2) % 3);
        const Real angle_a = std::acos((b - a).normalized().dot((c - a).normalized()));
        for (const bool at_vertex : {true, false}) {
            const Vec3 r0 = at_vertex ? a : Vec3(0.5 * (a + b));
            const Real limit = at_vertex ? angle_a : kPi;
            const StaticIntegrals s0 = eval(r0, t);
            for (const Real delta : {1e-6, 1e-9}) {
                const Vec3 r = r0 + delta * L * n;
                const StaticIntegrals s = eval(r, t);
                REQUIRE(all_finite(s));
                const Real tol = 10.0 * delta;
                CHECK(std::abs(s.I_1R - s0.I_1R) <= tol * std::abs(s0.I_1R));
                CHECK((s.I_rho_R - s0.I_rho_R).norm() <= tol * s0.I_rho_R.norm());
                CHECK(std::abs(s.I_R - s0.I_R) <= tol * std::abs(s0.I_R));
                CHECK((s.I_rhoR - s0.I_rhoR).norm() <= tol * s0.I_rhoR.norm());
                const Real normal = s.I_grad.dot(n);
                CHECK(std::abs(normal - limit) <= 1e-6 + 100.0 * delta);
                CHECK(std::abs(normal + vos_solid_angle(r, t)) <= 1e-15 / delta);
            }
        }
    }
    INFO("on boundary: " << worst_on << ", on extensions: " << worst_ext
                         << ", I_grad on extensions: " << worst_grad_ext);
    CHECK(worst_on < 1e-9);
    CHECK(worst_ext < 1e-9);
    CHECK(worst_grad_ext < 1e-9);

    // Exactly at the centroid.
    {
        const Vec3 r = t.centroid();
        const StaticIntegrals s = eval(r, t);
        REQUIRE(all_finite(s));
        CHECK(max_rel_error(s, converged_duffy(r, t), false) < 1e-9);
        CHECK(rel_vec(s.I_grad, converged_polar_pv_grad(r, poly, n)) < 1e-9);
    }
    // In-plane, far away (separated: composite Dunavant reference).
    {
        const Vec3 u = (t.v1 - t.v2).cross(n).normalized();
        const Vec3 r = t.centroid() + 10.0 * L * u;
        const StaticIntegrals s = eval(r, t);
        REQUIRE(all_finite(s));
        CHECK(std::abs(s.I_grad.dot(n)) <= kInPlaneNormalTol * s.I_grad.norm());
        const Real err = max_rel_error(s, dunavant_reference(r, t, 20, 4));
        INFO("far in-plane: " << err);
        CHECK(err < 1e-12);
    }
}

TEST_CASE("finite-part I_grad on an edge or vertex is additive over coplanar neighbours",
          "[kernels]") {
    // Hexagonal fan of six coplanar triangles around the origin, counter-clockwise about +z.
    const Real L = kMicron;
    std::vector<Vec3> ring;
    for (int k = 0; k < 6; ++k) {
        const Real phi = 2.0 * kPi * k / 6.0 + 0.1 * std::sin(1.0 + k);
        const Real rad = L * (1.0 + 0.15 * std::cos(2.0 + 3.0 * k));
        ring.emplace_back(rad * std::cos(phi), rad * std::sin(phi), 0.0);
    }
    const Vec3 n(0.0, 0.0, 1.0);
    const Vec3 origin = Vec3::Zero();
    // Vertex: the sum over the fan equals the principal value over the hexagon.
    Vec3 sum = Vec3::Zero();
    for (std::size_t k = 0; k < 6; ++k) {
        const StaticIntegrals s = static_integrals(origin, origin, ring[k], ring[(k + 1) % 6]);
        REQUIRE(all_finite(s));
        sum += s.I_grad;
    }
    const Vec3 ref = converged_polar_pv_grad(origin, ring, n);
    INFO("fan: " << rel_vec(sum, ref));
    CHECK(rel_vec(sum, ref) < 1e-9);

    // Edge: point on the shared edge (origin, ring[0]) of the triangles 5 and 0.
    const Vec3 r = 0.37 * ring[0];
    const StaticIntegrals s0 = static_integrals(r, origin, ring[0], ring[1]);
    const StaticIntegrals s5 = static_integrals(r, origin, ring[5], ring[0]);
    REQUIRE(all_finite(s0));
    REQUIRE(all_finite(s5));
    const std::vector<Vec3> quad = {origin, ring[5], ring[0], ring[1]};
    const Vec3 ref_edge = converged_polar_pv_grad(r, quad, n);
    INFO("edge: " << rel_vec(s0.I_grad + s5.I_grad, ref_edge));
    CHECK(rel_vec(s0.I_grad + s5.I_grad, ref_edge) < 1e-9);
}

TEST_CASE("static integrals are invariant under vertex permutations and rigid motions",
          "[kernels]") {
    const std::array<std::array<int, 3>, 6> perms = {
        {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}}};
    const Eigen::AngleAxis<Real> rot(1.234, Vec3(0.3, -0.5, 0.8).normalized());
    const Mat3 Q = rot.toRotationMatrix();
    const Vec3 shift = Vec3(2.1, -0.7, 1.3) * kMicron;
    Real worst_perm = 0.0;
    Real worst_rigid = 0.0;
    for (const Tri& t : random_triangles(3, 80)) {
        const Real L = t.longest_edge();
        const Vec3 n = t.normal();
        const std::vector<Vec3> points = {
            t.centroid(),                                  // in-plane, on T
            t.v1,                                          // vertex
            t.point(0.0, 0.4, 0.6),                        // edge interior
            t.centroid() + 0.05 * L * n,                   // near, off-plane
            t.centroid() - 1e-6 * L * n,                   // very close below
            t.v0 + 0.8 * L * (t.v0 - t.v2).normalized(),   // in-plane outside
            t.centroid() + 2.0 * L * Vec3(0.6, 0.0, 0.8),  // separated
        };
        for (const Vec3& r : points) {
            const StaticIntegrals s = eval(r, t);
            for (const auto& p : perms) {
                const Tri tp{t.vertex(p[0]), t.vertex(p[1]), t.vertex(p[2])};
                worst_perm = std::max(
                    worst_perm,
                    max_rel_error(eval(r, tp), Acc{s.I_1R, s.I_rho_R, s.I_grad, s.I_R, s.I_rhoR}));
            }
            const Tri tr{Q * t.v0 + shift, Q * t.v1 + shift, Q * t.v2 + shift};
            const StaticIntegrals sr = eval(Q * r + shift, tr);
            const Acc expected{s.I_1R, Q * s.I_rho_R, Q * s.I_grad, s.I_R, Q * s.I_rhoR};
            worst_rigid = std::max(worst_rigid, max_rel_error(sr, expected));
        }
    }
    INFO("permutations: " << worst_perm << ", rigid motions: " << worst_rigid);
    CHECK(worst_perm < 1e-12);
    CHECK(worst_rigid < 1e-12);
}

TEST_CASE("static integrals scale with the geometry", "[kernels]") {
    const Tri t = reference_triangle();
    const Real L = t.longest_edge();
    const Vec3 n = t.normal();
    const std::vector<Vec3> points = {t.centroid(), t.v2, t.point(0.5, 0.5, 0.0),
                                      t.centroid() + 0.1 * L * n, t.centroid() + 3.0 * L * n};
    Real worst = 0.0;
    for (const Real scale : {1e-3, 7.3, 1e3}) {
        const Tri ts{scale * t.v0, scale * t.v1, scale * t.v2};
        for (const Vec3& r : points) {
            const StaticIntegrals s = eval(r, t);
            const StaticIntegrals ss = eval(scale * r, ts);
            const Acc expected{scale * s.I_1R, scale * scale * s.I_rho_R, s.I_grad,
                               scale * scale * scale * s.I_R,
                               scale * scale * scale * scale * s.I_rhoR};
            worst = std::max(worst, max_rel_error(ss, expected));
        }
    }
    INFO("worst: " << worst);
    CHECK(worst < 1e-12);
}

TEST_CASE("static_integrals rejects degenerate triangles and non-finite input", "[kernels]") {
    const Vec3 r(0.3, 0.2, 0.1);
    const Vec3 a(0.0, 0.0, 0.0);
    const Vec3 b(1.0, 0.0, 0.0);
    CHECK_THROWS_AS((void)static_integrals(r, a, b, Vec3(2.0, 0.0, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS((void)static_integrals(r, a, b, Vec3(0.5, 1e-13, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS((void)static_integrals(r, a, a, b), std::invalid_argument);
    CHECK_THROWS_AS((void)static_integrals(r, a, a, a), std::invalid_argument);
    CHECK_NOTHROW((void)static_integrals(r, a, b, Vec3(0.5, 1e-11, 0.0)));
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    CHECK_THROWS_AS((void)static_integrals(Vec3(nan, 0.0, 0.0), a, b, Vec3(0.0, 1.0, 0.0)),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        (void)static_integrals(r, a, b, Vec3(0.0, std::numeric_limits<Real>::infinity(), 0.0)),
        std::invalid_argument);
}

TEST_CASE("classify on an icosphere", "[kernels]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1e-6, 2);
    const Index nt = mesh.num_triangles();
    const Triangles& tri = mesh.triangles();
    const Vertices& vert = mesh.vertices();

    const auto idx = [nt](Index i, Index j) { return static_cast<std::size_t>(i * nt + j); };
    std::vector<char> edge_pair(static_cast<std::size_t>(nt * nt), 0);
    for (Index e = 0; e < mesh.num_edges(); ++e) {
        const Index a = mesh.edge_triangles()(e, 0);
        const Index b = mesh.edge_triangles()(e, 1);
        edge_pair[idx(a, b)] = 1;
        edge_pair[idx(b, a)] = 1;
    }
    std::vector<Real> longest(static_cast<std::size_t>(nt), 0.0);
    std::vector<std::array<Real, 3>> centroid(static_cast<std::size_t>(nt));
    for (Index t = 0; t < nt; ++t) {
        for (Eigen::Index k = 0; k < 3; ++k) {
            const Vec3 p = vert.row(tri(t, k)).transpose();
            const Vec3 q = vert.row(tri(t, (k + 1) % 3)).transpose();
            longest[static_cast<std::size_t>(t)] =
                std::max(longest[static_cast<std::size_t>(t)], (q - p).norm());
        }
        const Vec3 c = mesh.centroid(t);
        centroid[static_cast<std::size_t>(t)] = {c(0), c(1), c(2)};
    }

    // Per-pair conditions are counted as mismatches (one CHECK each at the end) to keep the
    // ~10^5 pairs fast.
    std::map<Proximity, Index> counts;
    Index edge_class_count = 0;
    Index moved = 0;
    Index bad_identical = 0;
    Index bad_edge = 0;
    Index bad_vertex = 0;
    Index bad_near_far = 0;
    Index bad_factor = 0;
    for (Index i = 0; i < nt; ++i) {
        for (Index j = 0; j < nt; ++j) {
            const Proximity p = classify(mesh, i, j);
            const Proximity p4 = classify(mesh, i, j, 4.0);
            ++counts[p];
            bad_identical += ((p == Proximity::identical) != (i == j)) ? 1 : 0;
            const Index* ti = tri.data() + 3 * i;
            const Index* tj = tri.data() + 3 * j;
            int shared = 0;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    shared += ti[a] == tj[b] ? 1 : 0;
                }
            }
            if (i != j) {
                const bool is_edge = p == Proximity::shared_edge;
                bad_edge +=
                    (is_edge != (edge_pair[idx(i, j)] == 1) || is_edge != (shared == 2)) ? 1 : 0;
                bad_vertex += ((p == Proximity::shared_vertex) != (shared == 1)) ? 1 : 0;
                if (shared == 0) {
                    const std::array<Real, 3>& ci = centroid[static_cast<std::size_t>(i)];
                    const std::array<Real, 3>& cj = centroid[static_cast<std::size_t>(j)];
                    const Real dist = std::sqrt((ci[0] - cj[0]) * (ci[0] - cj[0]) +
                                                (ci[1] - cj[1]) * (ci[1] - cj[1]) +
                                                (ci[2] - cj[2]) * (ci[2] - cj[2]));
                    const Real size = std::max(longest[static_cast<std::size_t>(i)],
                                               longest[static_cast<std::size_t>(j)]);
                    const Proximity expected = dist < 2.0 * size ? Proximity::near : Proximity::far;
                    bad_near_far += (p != expected) ? 1 : 0;
                }
            }
            edge_class_count += p == Proximity::shared_edge ? 1 : 0;
            // Factor 4 instead of 2: only far -> near moves.
            if (p != p4) {
                bad_factor += (p != Proximity::far || p4 != Proximity::near) ? 1 : 0;
                ++moved;
            }
        }
    }
    CHECK(bad_identical == 0);
    CHECK(bad_edge == 0);
    CHECK(bad_vertex == 0);
    CHECK(bad_near_far == 0);
    CHECK(bad_factor == 0);
    for (const Proximity p : {Proximity::identical, Proximity::shared_edge,
                              Proximity::shared_vertex, Proximity::near, Proximity::far}) {
        CHECK(counts[p] > 0);
    }
    CHECK(counts[Proximity::identical] == nt);
    CHECK(edge_class_count == 2 * mesh.num_edges());
    CHECK(moved > 0);
    // Independent count from the vertex valences: ordered pairs of distinct triangles around
    // a vertex, sum_v deg(v) (deg(v) - 1), count each edge-sharing pair twice (two shared
    // vertices) and each vertex-sharing pair once.
    std::vector<Index> valence(static_cast<std::size_t>(mesh.num_vertices()), 0);
    for (Index i = 0; i < nt; ++i) {
        for (Eigen::Index a = 0; a < 3; ++a) {
            ++valence[static_cast<std::size_t>(tri(i, a))];
        }
    }
    Index pairs_around_vertices = 0;
    for (const Index deg : valence) {
        pairs_around_vertices += deg * (deg - 1);
    }
    const Index brute_vertex = pairs_around_vertices - 2 * (2 * mesh.num_edges());
    CHECK(counts[Proximity::shared_vertex] == brute_vertex);

    CHECK_THROWS_AS((void)classify(mesh, -1, 0), std::invalid_argument);
    CHECK_THROWS_AS((void)classify(mesh, 0, nt), std::invalid_argument);
    CHECK_THROWS_AS((void)classify(mesh, 0, 1, -1.0), std::invalid_argument);
    CHECK_THROWS_AS((void)classify(mesh, 0, 1, std::numeric_limits<Real>::quiet_NaN()),
                    std::invalid_argument);
}

// Off-plane near-touching points (the regime of shared-edge / shared-vertex assembly) at
// heights d = 1e-6, 1e-3, 0.3 h above (reference triangle) and below (random triangle) four
// base points; split into one test case per base point to keep each test short.
TEST_CASE("off-plane points above the centroid match the graded Duffy reference", "[kernels]") {
    const Tri t = reference_triangle();
    const Tri u = random_triangles(1, 81)[0];
    const Real worst = std::max(check_off_plane_column(t, t.centroid(), 1.0),
                                check_off_plane_column(u, u.centroid(), -1.0));
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("off-plane points above an edge midpoint match the graded Duffy reference", "[kernels]") {
    const Tri t = reference_triangle();
    const Tri u = random_triangles(1, 81)[0];
    const Real worst = std::max(check_off_plane_column(t, 0.5 * (t.v0 + t.v1), 1.0),
                                check_off_plane_column(u, 0.5 * (u.v0 + u.v1), -1.0));
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("off-plane points above a vertex match the graded Duffy reference", "[kernels]") {
    const Tri t = reference_triangle();
    const Tri u = random_triangles(1, 81)[0];
    const Real worst =
        std::max(check_off_plane_column(t, t.v1, 1.0), check_off_plane_column(u, u.v1, -1.0));
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("off-plane points above an outside point match the graded Duffy reference", "[kernels]") {
    const auto outside = [](const Tri& x) {
        const Vec3 mid = 0.5 * (x.v0 + x.v1);
        return Vec3(mid + 0.3 * x.longest_edge() * (mid - x.v2).normalized());
    };
    const Tri t = reference_triangle();
    const Tri u = random_triangles(1, 81)[0];
    const Real worst = std::max(check_off_plane_column(t, outside(t), 1.0),
                                check_off_plane_column(u, outside(u), -1.0));
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("static integrals at the quadrature points of a neighbour hinged at 90 degrees",
          "[kernels]") {
    const Real worst = std::max(check_hinged_neighbour(reference_triangle(), 90.0),
                                check_hinged_neighbour(random_triangles(1, 82)[0], 90.0));
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("static integrals at the quadrature points of a neighbour hinged at 150 degrees",
          "[kernels]") {
    const Real worst = std::max(check_hinged_neighbour(reference_triangle(), 150.0),
                                check_hinged_neighbour(random_triangles(1, 82)[0], 150.0));
    INFO("worst relative error: " << worst);
    CHECK(worst < 1e-9);
}

TEST_CASE("static integrals at ten triangle sizes (off-plane) still meet 1e-12", "[kernels]") {
    std::mt19937_64 rng(83);
    Real worst = 0.0;
    for (int k = 0; k < 5; ++k) {
        const Tri t = random_triangle(rng);
        const Real h = t.longest_edge();
        for (int j = 0; j < 4; ++j) {
            Vec3 u = random_unit(rng);
            if (std::abs(u.dot(t.normal())) < 0.2) {
                u = (u + 0.5 * t.normal()).normalized();  // keep it clearly off-plane
            }
            const Vec3 r = t.centroid() + 10.0 * h * u;
            const StaticIntegrals s = eval(r, t);
            worst = std::max(worst, max_rel_error(s, dunavant_reference(r, t, 20, 2)));
        }
    }
    INFO("worst relative error at D = 10 h: " << worst);
    CHECK(worst < 1e-12);
}

TEST_CASE("grad_finite_part is set exactly for in-plane points on the boundary of T", "[kernels]") {
    const Tri t = reference_triangle();
    const Real h = t.longest_edge();
    const Vec3 n = t.normal();
    const Vec3 mid01 = 0.5 * (t.v0 + t.v1);
    const Vec3 dir01 = (t.v1 - t.v0).normalized();
    const Vec3 inward = n.cross(dir01);  // in-plane, into T from edge v0 v1
    // Set: vertices, edge points, and points within 1e-12 h of them (in or off the plane).
    for (const Vec3& r : std::vector<Vec3>{t.v0, t.v1, t.v2, mid01, 0.5 * (t.v1 + t.v2),
                                           t.v2 + 0.25 * (t.v0 - t.v2), mid01 + 1e-13 * h * n,
                                           mid01 + 1e-13 * h * inward, t.v1 - 1e-13 * h * dir01}) {
        const StaticIntegrals s = eval(r, t);
        REQUIRE(all_finite(s));
        CHECK(s.grad_finite_part);
    }
    // Not set: interior, near-boundary but beyond the threshold, on edge-line extensions,
    // off-plane above edges / vertices, separated points.
    for (const Vec3& r : std::vector<Vec3>{
             t.centroid(), mid01 + 1e-9 * h * inward, mid01 - 1e-9 * h * inward,
             mid01 + 1e-9 * h * n, t.v1 + 1e-9 * h * n, t.v1 + 0.3 * h * dir01,
             t.v0 - 0.3 * h * dir01, t.v1 + 1e-9 * h * dir01, t.centroid() + 3.0 * h * n}) {
        const StaticIntegrals s = eval(r, t);
        REQUIRE(all_finite(s));
        CHECK_FALSE(s.grad_finite_part);
    }
}
