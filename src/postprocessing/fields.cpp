/// @file fields.cpp
/// Stratton-Chu near-field and far-field evaluation from RWG surface currents x = [J; M].
///
/// Derivation of the representation (exp(+jwt), docs/06; convention of docs/03)
/// --------------------------------------------------------------------------------
/// Currents J, M radiating in a homogeneous medium (eps, mu, k, eta) give, with
/// G = exp(-jkR) / (4 pi R), A = mu int G J dS', F = eps int G M dS' (Balanis ch. 6),
///   E = -jw A + (1 / jw mu eps) grad(div A) - (1 / eps) curl F,
///   H = -jw F + (1 / jw mu eps) grad(div F) + (1 / mu) curl A.
/// On a closed surface div A = mu int grad G . J dS' = -mu int grad' G . J dS'
/// = mu int G div' J dS' (surface divergence theorem), so with the operators of docs/03
///   L X = jw mu int G X dS' - (1 / jw eps) grad int G div' X dS',
///   K X = int grad' G x X dS'   (grad' = source-point gradient, grad' G = -grad G),
/// the electric-current part of E is -jw A + (1 / jw mu eps) grad(div A) = -L J, and the
/// magnetic-current part is -(1 / eps) curl F = -curl int G M dS' = -int grad G x M dS'
/// = +int grad' G x M dS' = +K M (curl(G M) = grad G x M for M independent of r). Duality
/// (J -> M, E -> H, H -> -E, eps <-> mu, which maps L -> L / eta^2 and keeps K) gives the H
/// parts -K J and -(1 / eta^2) L M. Hence, for the currents (J_i, M_i) of region i radiating
/// in the medium of region i,
///   E_i = -L_i J_i + K_i M_i ,   H_i = -K_i J_i - (1 / eta_i^2) L_i M_i .                 (*)
///
/// Region R1 (normal n out of R2 into R1): Love's equivalence with J = n x H, M = -n x E taken
/// on the R1 side and radiating in medium 1 reproduces E - E_inc (the scattered field) in R1
/// and -E_inc in R2 (extinction theorem; this gives the tangential equations of docs/03,
/// E_inc = L J - K M and H_inc = K J + L M / eta^2). So J_1 = J, M_1 = M and (*) is the
/// scattered field. Region R2: the normal pointing into R2 is -n, and the tangential fields are
/// continuous, so J_2 = (-n) x H = -J, M_2 = -(-n) x E = -M. Radiating in medium 2 they
/// reproduce the total field in R2 (no incident field there) and zero in R1. Therefore
///   R1: E = -L_1 J + K_1 M,        H = -K_1 J - L_1 M / eta_1^2      (scattered field)
///   R2: E = +L_2 J - K_2 M,        H = +K_2 J + L_2 M / eta_2^2      (total field)
/// The unit test with Mie currents projected onto RWG (tests/unit/test_fields.cpp) checks all
/// of these signs: the R1 result must equal the Mie scattered field, the R2 result the Mie
/// internal field.
///
/// Discretisation: J and M are expanded in the RWG functions (rwg.hpp); on every triangle the
/// integrals are evaluated with one Dunavant rule (FieldOptions::quad_degree, default 6).
/// grad int G div' X dS' = int grad G div' X dS' with the piecewise-constant RWG divergence;
/// grad G = -(1 + jkR) G / R^2 (r - r') = -grad' G. Cost O(N_points * N_triangles * N_quad); OpenMP
/// over observation points, each point summed in a fixed order (results independent of threads).
///
/// Far field (R1, exterior currents): for r = r k_hat, r -> infinity,
///   G -> g exp(j k1 k_hat . r'),  grad' G -> +j k1 k_hat G,  g = exp(-j k1 r) / (4 pi r).
/// With Jt = int exp(j k1 k_hat.r') J dS' and int exp(j k1 k_hat.r') div' J dS' = -j k1 k_hat.Jt
/// (integration by parts on the closed surface),
///   L_1 J -> jw mu1 g (Jt - k_hat (k_hat . Jt)) = -j k1 eta1 g k_hat x (k_hat x Jt),
///   K_1 M -> +j k1 g k_hat x Mt,
/// so E_s = -L_1 J + K_1 M -> F exp(-j k1 r) / r with
///   F(k_hat) = (j k1 / 4 pi) k_hat x int exp(j k1 k_hat . r') [eta1 (k_hat x J) + M] dS'.
///
/// Region test: generalised winding number (sum of signed solid angles / 4 pi; Van Oosterom &
/// Strackee, IEEE TBME 30 (1983) 125, signed form tan(Omega/2) = a0.(a1 x a2) / (...), with
/// a_i = v_i - r). For counter-clockwise triangles seen from R1, a0.(a1 x a2) = -2 A d with
/// d = (r - v0) . n, so Omega > 0 when r lies behind the triangle (R2 side). Closed mesh:
/// w = 1 in R2, 0 in R1, threshold 0.5. Open mesh: w in (0, 1/2) on the R2 side, threshold 0;
/// beyond the patch edge, close to the plane of the patch, w -> 0 and the side is undecided,
/// so |w| < 1e-6 throws instead of classifying silently.
#include "specklebem/postprocessing/fields.hpp"

#include "specklebem/basis/rwg.hpp"
#include "specklebem/kernels/quadrature.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace specklebem::post {

namespace {

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;
using RegionVector = Eigen::Matrix<int, Eigen::Dynamic, 1>;

constexpr Complex kJ{0.0, 1.0};
constexpr Real kFourPi = 4.0 * constants::pi;
/// Open meshes: |w| below this leaves the side of the patch undecided (std::invalid_argument).
constexpr Real kOpenMeshMinWinding = 1e-6;

/// a x b for real a and complex b.
inline Vec3c cross_rc(const Vec3& a, const Vec3c& b) {
    return {a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x()};
}

/// Validated RWG space of the solution (problem, space, currents length).
const basis::RwgSpace& checked_space(const SurfaceSolution& s, const std::string& who) {
    if (s.problem == nullptr) {
        throw std::invalid_argument(who + ": SurfaceSolution::problem is null");
    }
    if (s.problem->space == nullptr) {
        throw std::invalid_argument(who + ": Problem::space is null");
    }
    const basis::RwgSpace& space = *s.problem->space;
    if (s.currents.size() != 2 * space.size()) {
        throw std::invalid_argument(who + ": currents have length " +
                                    std::to_string(s.currents.size()) +
                                    ", expected 2N = " + std::to_string(2 * space.size()));
    }
    return space;
}

/// Angular frequency of the solution: Problem::omega if > 0 (WP9; SurfaceSolution::omega and
/// the excitation's must then agree with it), otherwise SurfaceSolution::omega or the
/// excitation's.
Real resolve_omega(const SurfaceSolution& s, const std::string& who) {
    const excitation::Excitation* exc = s.problem->excitation;
    if (!std::isfinite(s.omega) || s.omega < 0.0) {
        throw std::invalid_argument(who + ": SurfaceSolution::omega must be finite and >= 0");
    }
    const Real pw = s.problem->omega;
    if (!std::isfinite(pw) || pw < 0.0) {
        throw std::invalid_argument(who + ": Problem::omega must be finite and >= 0");
    }
    if (pw > 0.0) {
        if ((s.omega > 0.0 && std::abs(s.omega - pw) > 1e-12 * pw) ||
            (exc != nullptr && std::abs(exc->omega() - pw) > 1e-12 * pw)) {
            throw std::invalid_argument(who +
                                        ": Problem::omega differs from SurfaceSolution::omega "
                                        "or the excitation's");
        }
        return pw;
    }
    if (s.omega > 0.0) {
        if (exc != nullptr && std::abs(exc->omega() - s.omega) > 1e-12 * s.omega) {
            throw std::invalid_argument(who +
                                        ": SurfaceSolution::omega differs from the excitation's");
        }
        return s.omega;
    }
    if (exc == nullptr) {
        throw std::invalid_argument(who +
                                    ": unknown frequency (SurfaceSolution::omega is 0 and the "
                                    "problem has no excitation)");
    }
    if (!(exc->omega() > 0.0) || !std::isfinite(exc->omega())) {
        throw std::invalid_argument(who + ": excitation omega must be positive and finite");
    }
    return exc->omega();
}

/// Medium parameters of one region.
struct Medium {
    Complex k, eta, eps, mu;
};

Medium medium_of(const material::Material& m, Real omega) {
    return {m.wavenumber(omega), m.wave_impedance(omega), constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

/// Source currents at one quadrature point, stored as plain reals (the inner loops below are
/// written in scalar arithmetic so that they stay fast in unoptimised sanitizer builds too).
struct Sample {
    Real x, y, z;
    Real w;             ///< Dunavant weight x triangle area
    Real jr[3], ji[3];  ///< J (real, imaginary part)
    Real mr[3], mi[3];  ///< M
    Real djr, dji;      ///< surface divergence of J (constant per triangle)
    Real dmr, dmi;      ///< surface divergence of M
};

std::vector<Sample> sample_currents(const basis::RwgSpace& space, const VectorXc& x, int degree,
                                    const std::string& who) {
    if (degree < 1 || degree > 20 || !kernels::triangle_rule_is_positive_interior(degree)) {
        throw std::invalid_argument(who +
                                    ": FieldOptions::quad_degree must be a positive-interior "
                                    "Dunavant degree, got " +
                                    std::to_string(degree));
    }
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index n_basis = space.size();
    const Index n_tri = mesh.num_triangles();
    const std::size_t nq = rule.weights.size();

    std::vector<Sample> samples(static_cast<std::size_t>(n_tri) * nq);
    const Vertices& V = mesh.vertices();
    const Triangles& T = mesh.triangles();
    for (Index t = 0; t < n_tri; ++t) {
        const Vec3 v0 = V.row(T(t, 0)).transpose();
        const Vec3 v1 = V.row(T(t, 1)).transpose();
        const Vec3 v2 = V.row(T(t, 2)).transpose();
        const Real area = mesh.area(t);
        const basis::RwgSpace::Support sup = space.support(t);
        Complex div_j{0.0, 0.0};
        Complex div_m{0.0, 0.0};
        for (int a = 0; a < sup.count; ++a) {
            const Index n = sup.n[a];
            const Real dv = space.divergence(n, t);
            div_j += x(n) * dv;
            div_m += x(n_basis + n) * dv;
        }
        for (std::size_t q = 0; q < nq; ++q) {
            const Vec3& l = rule.barycentric[q];
            const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
            Vec3c j = Vec3c::Zero();
            Vec3c m = Vec3c::Zero();
            for (int a = 0; a < sup.count; ++a) {
                const Index n = sup.n[a];
                const Vec3 f = space.value(n, t, r);
                j += x(n) * f;
                m += x(n_basis + n) * f;
            }
            Sample& s = samples[static_cast<std::size_t>(t) * nq + q];
            s.x = r.x();
            s.y = r.y();
            s.z = r.z();
            s.w = rule.weights[q] * area;
            for (Index c = 0; c < 3; ++c) {
                const auto cc = static_cast<std::size_t>(c);
                s.jr[cc] = j(c).real();
                s.ji[cc] = j(c).imag();
                s.mr[cc] = m(c).real();
                s.mi[cc] = m(c).imag();
            }
            s.djr = div_j.real();
            s.dji = div_j.imag();
            s.dmr = div_m.real();
            s.dmi = div_m.imag();
        }
    }
    return samples;
}

/// Closest point of triangle (a, b, c) to p (Ericson, Real-Time Collision Detection, 5.1.5).
Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const Real d1 = ab.dot(ap);
    const Real d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0)
        return a;
    const Vec3 bp = p - b;
    const Real d3 = ab.dot(bp);
    const Real d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3)
        return b;
    const Real vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
        return a + (d1 / (d1 - d3)) * ab;
    const Vec3 cp = p - c;
    const Real d5 = ab.dot(cp);
    const Real d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6)
        return c;
    const Real vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
        return a + (d2 / (d2 - d6)) * ac;
    const Real va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
    }
    const Real denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

/// Per-triangle geometry for the region and distance tests (plain reals, see Sample).
struct TriangleGeometry {
    Real v[3][3];  ///< vertices
    Real c[3];     ///< centroid
    Real h;        ///< longest edge
};

/// Signed solid angle of triangle t seen from r = (rx, ry, rz), positive behind the triangle
/// (opposite to its counter-clockwise normal); Van Oosterom & Strackee. 0 for r in the plane
/// of the triangle (also on it).
Real signed_solid_angle(const TriangleGeometry& t, Real rx, Real ry, Real rz) {
    Real a[3][3];
    Real l[3];
    for (int i = 0; i < 3; ++i) {
        a[i][0] = t.v[i][0] - rx;
        a[i][1] = t.v[i][1] - ry;
        a[i][2] = t.v[i][2] - rz;
        l[i] = std::sqrt(a[i][0] * a[i][0] + a[i][1] * a[i][1] + a[i][2] * a[i][2]);
    }
    const auto dot = [&a](int i, int j) {
        return a[i][0] * a[j][0] + a[i][1] * a[j][1] + a[i][2] * a[j][2];
    };
    const Real num = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) +
                     a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) +
                     a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (num == 0.0)
        return 0.0;
    const Real den = l[0] * l[1] * l[2] + dot(0, 1) * l[2] + dot(0, 2) * l[1] + dot(1, 2) * l[0];
    return 2.0 * std::atan2(num, den);
}

std::vector<TriangleGeometry> triangle_geometry(const geometry::TriangleMesh& mesh) {
    const Vertices& V = mesh.vertices();
    const Triangles& T = mesh.triangles();
    std::vector<TriangleGeometry> g(static_cast<std::size_t>(mesh.num_triangles()));
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        TriangleGeometry& tg = g[static_cast<std::size_t>(t)];
        Real h2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 3; ++c) tg.v[i][c] = V(T(t, i), c);
        }
        for (int c = 0; c < 3; ++c) tg.c[c] = (tg.v[0][c] + tg.v[1][c] + tg.v[2][c]) / 3.0;
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3;
            Real e2 = 0.0;
            for (int c = 0; c < 3; ++c) e2 += (tg.v[j][c] - tg.v[i][c]) * (tg.v[j][c] - tg.v[i][c]);
            h2 = std::max(h2, e2);
        }
        tg.h = std::sqrt(h2);
    }
    return g;
}

/// Regions of all points (1 or 2) by the winding-number rule; throws for points that are not
/// finite or too close to the surface (lowest offending index, independent of threads).
RegionVector regions_impl(const geometry::TriangleMesh& mesh, const Vertices& points, Real factor,
                          const std::string& who) {
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::invalid_argument(who + ": min_distance_factor must be positive and finite");
    }
    if (!points.allFinite()) {
        throw std::invalid_argument(who + ": observation points must be finite");
    }
    const std::vector<TriangleGeometry> tg = triangle_geometry(mesh);
    const bool closed = mesh.is_closed();
    const Index np = points.rows();
    RegionVector region(np);
    // 0 = ok, 1 = too close to the surface, 2 = ambiguous (open mesh, |w| < kOpenMeshMinWinding)
    std::vector<char> status(static_cast<std::size_t>(np), 0);
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (Index p = 0; p < np; ++p) {
        const Real rx = points(p, 0);
        const Real ry = points(p, 1);
        const Real rz = points(p, 2);
        Real omega_sum = 0.0;
        bool close = false;
        for (const TriangleGeometry& t : tg) {
            const Real limit = factor * t.h;
            // Every point of the triangle lies within h of its centroid; the exact distance is
            // needed only inside that ball.
            const Real dx = rx - t.c[0];
            const Real dy = ry - t.c[1];
            const Real dz = rz - t.c[2];
            if (!close && dx * dx + dy * dy + dz * dz <= (t.h + limit) * (t.h + limit)) {
                const Vec3 r(rx, ry, rz);
                const Vec3 q = closest_point_on_triangle(r, Vec3(t.v[0][0], t.v[0][1], t.v[0][2]),
                                                         Vec3(t.v[1][0], t.v[1][1], t.v[1][2]),
                                                         Vec3(t.v[2][0], t.v[2][1], t.v[2][2]));
                close = (r - q).norm() < limit;
            }
            omega_sum += signed_solid_angle(t, rx, ry, rz);
        }
        const Real w = omega_sum / kFourPi;
        const bool ambiguous = !closed && std::abs(w) < kOpenMeshMinWinding;
        status[static_cast<std::size_t>(p)] = close ? 1 : (ambiguous ? 2 : 0);
        region(p) = (closed ? w >= 0.5 : w > 0.0) ? 2 : 1;
    }
    for (Index p = 0; p < np; ++p) {
        const char st = status[static_cast<std::size_t>(p)];
        if (st == 1) {
            throw std::invalid_argument(
                who + ": observation point " + std::to_string(p) +
                " lies within min_distance_factor x (local edge length) of the surface; "
                "near-surface evaluation needs singular quadrature (Phase 5)");
        }
        if (st == 2) {
            throw std::invalid_argument(
                who + ": observation point " + std::to_string(p) +
                " has an ambiguous region on the open mesh (|winding number| < 1e-6, e.g. "
                "beyond the patch edge in the plane of the patch)");
        }
    }
    return region;
}

/// Field (*) of one region at r from the sampled currents, with J_i = sign J, M_i = sign M.
/// Accumulates int G X, int grad G div' X and int grad G x X (observation-point gradient) for
/// X = J, M in scalar arithmetic, with G and grad G = f (r - r') from detail::green_factors;
/// K X = int grad' G x X = -int grad G x X (docs/03).
void radiate(const std::vector<Sample>& src, const Vec3& r, const Medium& med, Real omega,
             Real sign, Vec3c& E, Vec3c& H) {
    const Real kr = med.k.real();
    const Real ki = med.k.imag();
    const Real rx = r.x();
    const Real ry = r.y();
    const Real rz = r.z();
    // [0..2] real, [3..5] imaginary part of each accumulated vector.
    Real g_j[6] = {};  // int G J
    Real g_m[6] = {};
    Real p_j[6] = {};  // int grad G div' J
    Real p_m[6] = {};
    Real neg_k_j[6] = {};  // int grad G x J = -K J
    Real neg_k_m[6] = {};  // int grad G x M = -K M
    for (const Sample& s : src) {
        const Real d[3] = {rx - s.x, ry - s.y, rz - s.z};
        const Real R = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const detail::GreenFactors gf = detail::green_factors(kr, ki, R);
        const Real gr = s.w * gf.g_re;
        const Real gi = s.w * gf.g_im;
        const Real fr = s.w * gf.f_re;
        const Real fi = s.w * gf.f_im;
        const Real fdjr = fr * s.djr - fi * s.dji;
        const Real fdji = fr * s.dji + fi * s.djr;
        const Real fdmr = fr * s.dmr - fi * s.dmi;
        const Real fdmi = fr * s.dmi + fi * s.dmr;
        const Real xjr[3] = {d[1] * s.jr[2] - d[2] * s.jr[1], d[2] * s.jr[0] - d[0] * s.jr[2],
                             d[0] * s.jr[1] - d[1] * s.jr[0]};
        const Real xji[3] = {d[1] * s.ji[2] - d[2] * s.ji[1], d[2] * s.ji[0] - d[0] * s.ji[2],
                             d[0] * s.ji[1] - d[1] * s.ji[0]};
        const Real xmr[3] = {d[1] * s.mr[2] - d[2] * s.mr[1], d[2] * s.mr[0] - d[0] * s.mr[2],
                             d[0] * s.mr[1] - d[1] * s.mr[0]};
        const Real xmi[3] = {d[1] * s.mi[2] - d[2] * s.mi[1], d[2] * s.mi[0] - d[0] * s.mi[2],
                             d[0] * s.mi[1] - d[1] * s.mi[0]};
        for (int c = 0; c < 3; ++c) {
            g_j[c] += gr * s.jr[c] - gi * s.ji[c];
            g_j[c + 3] += gr * s.ji[c] + gi * s.jr[c];
            g_m[c] += gr * s.mr[c] - gi * s.mi[c];
            g_m[c + 3] += gr * s.mi[c] + gi * s.mr[c];
            p_j[c] += fdjr * d[c];
            p_j[c + 3] += fdji * d[c];
            p_m[c] += fdmr * d[c];
            p_m[c + 3] += fdmi * d[c];
            neg_k_j[c] += fr * xjr[c] - fi * xji[c];
            neg_k_j[c + 3] += fr * xji[c] + fi * xjr[c];
            neg_k_m[c] += fr * xmr[c] - fi * xmi[c];
            neg_k_m[c + 3] += fr * xmi[c] + fi * xmr[c];
        }
    }
    const auto vec = [](const Real(&v)[6]) {
        return Vec3c(Complex(v[0], v[3]), Complex(v[1], v[4]), Complex(v[2], v[5]));
    };
    const Complex jwmu = kJ * omega * med.mu;
    const Complex jweps = kJ * omega * med.eps;
    const Vec3c L_j = jwmu * vec(g_j) - vec(p_j) / jweps;
    const Vec3c L_m = jwmu * vec(g_m) - vec(p_m) / jweps;
    const Vec3c K_j = -vec(neg_k_j);
    const Vec3c K_m = -vec(neg_k_m);
    // (*): E_i = -L_i J_i + K_i M_i, H_i = -K_i J_i - L_i M_i / eta_i^2 with (J_i, M_i) = sign (J,
    // M).
    E = sign * (-L_j + K_m);
    H = sign * (-K_j - L_m / (med.eta * med.eta));
}

void scattered_impl(const SurfaceSolution& s, const Vertices& points, FieldMatrix& E,
                    FieldMatrix& H, const FieldOptions& opt, RegionVector& region,
                    const std::string& who) {
    const basis::RwgSpace& space = checked_space(s, who);
    const Real omega = resolve_omega(s, who);
    region = regions_impl(space.mesh(), points, opt.min_distance_factor, who);
    const std::vector<Sample> src = sample_currents(space, s.currents, opt.quad_degree, who);
    const Medium media[2] = {medium_of(s.problem->exterior, omega),
                             medium_of(s.problem->object, omega)};
    const Index np = points.rows();
    E.resize(np, 3);
    H.resize(np, 3);
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (Index p = 0; p < np; ++p) {
        const bool exterior = region(p) == 1;
        Vec3c e;
        Vec3c h;
        radiate(src, points.row(p).transpose(), media[exterior ? 0 : 1], omega,
                exterior ? 1.0 : -1.0, e, h);
        E.row(p) = e.transpose();
        H.row(p) = h.transpose();
    }
}

}  // namespace

Real winding_number(const geometry::TriangleMesh& mesh, const Vec3& r) {
    if (!r.allFinite()) {
        throw std::invalid_argument("winding_number: point must be finite");
    }
    Real sum = 0.0;
    for (const TriangleGeometry& t : triangle_geometry(mesh)) {
        sum += signed_solid_angle(t, r.x(), r.y(), r.z());
    }
    return sum / kFourPi;
}

Eigen::Matrix<int, Eigen::Dynamic, 1> observation_regions(const geometry::TriangleMesh& mesh,
                                                          const Vertices& points,
                                                          Real min_distance_factor) {
    return regions_impl(mesh, points, min_distance_factor, "observation_regions");
}

void scattered_field(const SurfaceSolution& s, const Vertices& points, FieldMatrix& E,
                     FieldMatrix& H) {
    scattered_field(s, points, E, H, FieldOptions{});
}

void scattered_field(const SurfaceSolution& s, const Vertices& points, FieldMatrix& E,
                     FieldMatrix& H, const FieldOptions& opt) {
    RegionVector region;
    scattered_impl(s, points, E, H, opt, region, "scattered_field");
}

void total_field(const SurfaceSolution& s, const Vertices& points, FieldMatrix& E, FieldMatrix& H) {
    total_field(s, points, E, H, FieldOptions{});
}

void total_field(const SurfaceSolution& s, const Vertices& points, FieldMatrix& E, FieldMatrix& H,
                 const FieldOptions& opt) {
    if (s.problem == nullptr) {
        throw std::invalid_argument("total_field: SurfaceSolution::problem is null");
    }
    const excitation::Excitation* exc = s.problem->excitation;
    if (exc == nullptr) {
        throw std::invalid_argument("total_field: the problem has no excitation");
    }
    RegionVector region;
    scattered_impl(s, points, E, H, opt, region, "total_field");
    for (Index p = 0; p < points.rows(); ++p) {
        if (region(p) != 1)
            continue;  // the incident field exists only in R1 (docs/03)
        const Vec3 r = points.row(p).transpose();
        E.row(p) += exc->electric_field(r).transpose();
        H.row(p) += exc->magnetic_field(r).transpose();
    }
}

void far_field(const SurfaceSolution& s, const Vertices& directions, FieldMatrix& F) {
    far_field(s, directions, F, FieldOptions{});
}

void far_field(const SurfaceSolution& s, const Vertices& directions, FieldMatrix& F,
               const FieldOptions& opt) {
    const std::string who = "far_field";
    const basis::RwgSpace& space = checked_space(s, who);
    const Real omega = resolve_omega(s, who);
    const Index nd = directions.rows();
    std::vector<Vec3> k_hat(static_cast<std::size_t>(nd));
    for (Index i = 0; i < nd; ++i) {
        const Vec3 d = directions.row(i).transpose();
        const Real len = d.norm();
        if (!std::isfinite(len) || !(len > 0.0)) {
            throw std::invalid_argument(who + ": direction " + std::to_string(i) +
                                        " is zero or not finite");
        }
        k_hat[static_cast<std::size_t>(i)] = d / len;
    }
    const std::vector<Sample> src = sample_currents(space, s.currents, opt.quad_degree, who);
    const Medium med = medium_of(s.problem->exterior, omega);
    const Complex prefactor = kJ * med.k / kFourPi;
    const Real kr = med.k.real();
    const Real ki = med.k.imag();
    F.resize(nd, 3);
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (Index i = 0; i < nd; ++i) {
        const Vec3& kh = k_hat[static_cast<std::size_t>(i)];
        // Jt = int exp(j k k_hat.r') J dS', Mt likewise; j k s = -ki s + j kr s.
        Real jt[6] = {};
        Real mt[6] = {};
        for (const Sample& smp : src) {
            const Real proj = kh.x() * smp.x + kh.y() * smp.y + kh.z() * smp.z;
            const Real amp = std::exp(-ki * proj) * smp.w;
            const Real er = amp * std::cos(kr * proj);
            const Real ei = amp * std::sin(kr * proj);
            for (int c = 0; c < 3; ++c) {
                jt[c] += er * smp.jr[c] - ei * smp.ji[c];
                jt[c + 3] += er * smp.ji[c] + ei * smp.jr[c];
                mt[c] += er * smp.mr[c] - ei * smp.mi[c];
                mt[c + 3] += er * smp.mi[c] + ei * smp.mr[c];
            }
        }
        const Vec3c Jt(Complex(jt[0], jt[3]), Complex(jt[1], jt[4]), Complex(jt[2], jt[5]));
        const Vec3c Mt(Complex(mt[0], mt[3]), Complex(mt[1], mt[4]), Complex(mt[2], mt[5]));
        F.row(i) = (prefactor * cross_rc(kh, med.eta * cross_rc(kh, Jt) + Mt)).transpose();
    }
}

Vertices plane_grid(const Vec3& origin, const Vec3& u, const Vec3& v, int nu, int nv) {
    if (nu < 1 || nv < 1) {
        throw std::invalid_argument("plane_grid: nu and nv must be >= 1");
    }
    const auto nu_i = static_cast<Index>(nu);
    const auto nv_i = static_cast<Index>(nv);
    Vertices pts(nu_i * nv_i, 3);
    for (Index i = 0; i < nu_i; ++i) {
        const Real a = nu_i > 1 ? static_cast<Real>(i) / static_cast<Real>(nu_i - 1) : 0.0;
        for (Index j = 0; j < nv_i; ++j) {
            const Real b = nv_i > 1 ? static_cast<Real>(j) / static_cast<Real>(nv_i - 1) : 0.0;
            pts.row(i * nv_i + j) = (origin + a * u + b * v).transpose();
        }
    }
    return pts;
}

Vertices cylinder_grid(Real radius, Real theta_min, Real theta_max, int n_theta, Real y_min,
                       Real y_max, int n_y) {
    if (n_theta < 1 || n_y < 1) {
        throw std::invalid_argument("cylinder_grid: n_theta and n_y must be >= 1");
    }
    if (!(radius > 0.0) || !std::isfinite(radius)) {
        throw std::invalid_argument("cylinder_grid: radius must be positive and finite");
    }
    const auto nt = static_cast<Index>(n_theta);
    const auto ny = static_cast<Index>(n_y);
    Vertices pts(nt * ny, 3);
    for (Index k = 0; k < nt; ++k) {
        const Real th = nt > 1 ? theta_min + (theta_max - theta_min) * static_cast<Real>(k) /
                                                 static_cast<Real>(nt - 1)
                               : theta_min;
        for (Index l = 0; l < ny; ++l) {
            const Real y =
                ny > 1 ? y_min + (y_max - y_min) * static_cast<Real>(l) / static_cast<Real>(ny - 1)
                       : y_min;
            pts.row(k * ny + l) =
                Vec3(radius * std::sin(th), y, -radius * std::cos(th)).transpose();
        }
    }
    return pts;
}

}  // namespace specklebem::post
