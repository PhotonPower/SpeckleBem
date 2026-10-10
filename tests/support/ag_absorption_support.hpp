#pragma once
/// @file ag_absorption_support.hpp
/// Diagnostics of the absorbed power of the flat box (WP-A1, follow-up of WP22c): the absorbed
/// power from the surface currents, P_abs = 1/2 Re oint (n x M) . J* dS, of the flat Ag patch was
/// ~40 % below the Fresnel absorptance at 45 deg s while the reflectance met the criterion
/// (benchmarks/results/fresnel_flat.md). Shared by benchmarks/ag_absorption.cpp and the unit
/// checks in tests/validation/test_ag_absorption.cpp; no Catch2.
///
/// Conventions (docs/06): exp(+jwt), R1 = vacuum z < 0, R2 = object z > 0, n = -z on the top face
/// (out of R2 into R1), J = n x H, M = -n x E on the R1 side, so n x M = E_tan and
///   P_abs = 1/2 Re oint E . J* dS = 1/2 Re oint (n x M) . J* dS = -1/2 Re oint (E x H*) . n dS,
/// the power flowing into R2.
///
/// Diagnostics:
///  * absorbed_power_parts: P_abs per surface part (top face, walls, bottom plate) with any
///    Dunavant degree (degree 2 is exact for the quadratic RWG integrand; a higher degree is the
///    independent pointwise check).
///  * extinction_power: P_ext = 1/2 Re oint (E_inc . J* + M . H_inc*) dS (the Ei x Hi* flux of
///    the source-free incident beam through the closed surface vanishes), so that
///    P_ext - P_sca (P_sca from the far field of both hemispheres) is a second, far-field based
///    value of P_abs that does not use the local product n x M . J*.
///  * InterfaceFields: the exact total field of the beam on the infinite interface z = 0^- (every
///    plane wave of the angular spectrum reflected with its own Fresnel coefficients r_s, r_p and
///    the local s / p split of fresnel_flat::beam_reflectance), and the surface currents J_ref,
///    M_ref it implies. Its absorbed flux over the plane is (1 - R_beam - away share) P_inc.
///  * top_metrics: on a region of the top face, the simulated, reference and RWG-interpolated
///    absorbed power, the cross terms 1/2 Re int (n x M_sim) . J_ref* and 1/2 Re int (n x M_ref)
///    . J_sim*, and the complex correlation c_X = int X_sim . X_ref* / int |X_ref|^2 (amplitude and
///    phase of the simulated current relative to the analytic one) and the relative L2 error of J
///    and M.
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fresnel_flat_support.hpp"

namespace ag_absorption {

using namespace specklebem;

/// a . conj(b) (no conjugation of a, unlike Eigen's dot).
inline Complex dot_conj(const Vec3c& a, const Vec3c& b) {
    return a(0) * std::conj(b(0)) + a(1) * std::conj(b(1)) + a(2) * std::conj(b(2));
}

/// n x m for a real n (component form).
inline Vec3c cross_real(const Vec3& n, const Vec3c& m) {
    return Vec3c(n.y() * m.z() - n.z() * m.y(), n.z() * m.x() - n.x() * m.z(),
                 n.x() * m.y() - n.y() * m.x());
}

/// Surface part of a triangle of the flat box.
enum class Part { Top = 0, Walls = 1, Bottom = 2 };

/// Top face: centroid on z = 0 and normal -z; bottom plate: normal +z; walls: the rest.
inline Part part_of(const geometry::TriangleMesh& mesh, Index t, Real spacing) {
    const Vec3 n = mesh.normal(t);
    const Vec3 c = mesh.centroid(t);
    if (n.z() < -0.99 && std::abs(c.z()) < 1e-6 * spacing)
        return Part::Top;
    if (n.z() > 0.99)
        return Part::Bottom;
    return Part::Walls;
}

/// Electric and magnetic RWG expansions of the solution at r on triangle t.
inline std::pair<Vec3c, Vec3c> currents_at(const post::SurfaceSolution& s, Index t, const Vec3& r) {
    const basis::RwgSpace& space = *s.problem->space;
    const Index N = space.size();
    const basis::RwgSpace::Support sup = space.support(t);
    Vec3c J = Vec3c::Zero();
    Vec3c M = Vec3c::Zero();
    for (int a = 0; a < sup.count; ++a) {
        const Vec3 f = space.value(sup.n[a], t, r);
        J += s.currents(sup.n[a]) * f.cast<Complex>();
        M += s.currents(N + sup.n[a]) * f.cast<Complex>();
    }
    return {J, M};
}

/// Quadrature point r and weight (area included) of rule point q on triangle t.
inline std::pair<Vec3, Real> rule_point(const geometry::TriangleMesh& mesh, Index t,
                                        const kernels::TriangleRule& rule, std::size_t q) {
    const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
    const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
    const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
    const Vec3& l = rule.barycentric[q];
    return {l(0) * v0 + l(1) * v1 + l(2) * v2, rule.weights[q] * mesh.area(t)};
}

/// P_abs per surface part [W].
struct AbsorbedParts {
    Real top = 0, walls = 0, bottom = 0;
    [[nodiscard]] Real total() const { return top + walls + bottom; }
};

/// 1/2 Re int (n x M) . J* dS per part with the Dunavant rule of `degree` (>= 2 is exact).
inline AbsorbedParts absorbed_power_parts(const post::SurfaceSolution& s, Real spacing,
                                          int degree) {
    const geometry::TriangleMesh& mesh = s.problem->space->mesh();
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    AbsorbedParts p;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 nt = mesh.normal(t);
        Real acc = 0;
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const auto [r, w] = rule_point(mesh, t, rule, q);
            const auto [J, M] = currents_at(s, t, r);
            acc += w * 0.5 * dot_conj(cross_real(nt, M), J).real();
        }
        switch (part_of(mesh, t, spacing)) {
            case Part::Top:
                p.top += acc;
                break;
            case Part::Walls:
                p.walls += acc;
                break;
            case Part::Bottom:
                p.bottom += acc;
                break;
        }
    }
    return p;
}

/// P_ext = 1/2 Re oint (E_inc . J* + M . H_inc*) dS [W] (Dunavant degree `degree`).
inline Real extinction_power(const post::SurfaceSolution& s, const excitation::Excitation& inc,
                             int degree) {
    const geometry::TriangleMesh& mesh = s.problem->space->mesh();
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    Real p = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const auto [r, w] = rule_point(mesh, t, rule, q);
            const auto [J, M] = currents_at(s, t, r);
            const auto [Ei, Hi] = inc.fields(r);
            p += w * 0.5 * (dot_conj(Ei, J) + dot_conj(M, Hi)).real();
        }
    }
    return p;
}

/// Exact total field of the beam on the infinite interface z = 0 (R1 side), see the file
/// comment. Plane waves with k_z <= 0 (outer cone of an oblique beam) never reach the interface
/// and are left out (their power share is BeamReflectance::away_share).
class InterfaceFields {
public:
    InterfaceFields(const excitation::AngularSpectrumBeam& beam, Real n1, Complex eps2) {
        if (beam.params().focus.norm() != 0.0)
            throw std::invalid_argument("InterfaceFields: the beam focus must be the origin");
        const Real omega = beam.omega();
        eta_ = material::vacuum().wave_impedance(omega).real() / n1;
        k_ = n1 * omega / constants::c0;
        const std::vector<Vec3> dirs = beam.plane_wave_directions();
        const std::vector<Vec3> amps = beam.plane_wave_amplitudes();
        for (std::size_t i = 0; i < dirs.size(); ++i) {
            const Vec3& kh = dirs[i];
            const Vec3& a = amps[i];
            if (!(kh.z() > 0.0) || a.norm() == 0.0)
                continue;
            Vec3 s_hat = kh.cross(Vec3::UnitZ());
            const Real s_norm = s_hat.norm();
            s_hat = s_norm > 1e-12 ? Vec3(s_hat / s_norm) : Vec3(Vec3::UnitY());
            const Vec3 kr(kh.x(), kh.y(), -kh.z());  // reflected direction
            const fresnel_flat::FresnelCoefficients c =
                fresnel_flat::fresnel(n1, eps2, std::min(kh.z(), 1.0));
            const Real as = a.dot(s_hat);
            const Real ap = a.dot(s_hat.cross(kh));  // along p_i = s x k_i
            // Incident: E = a, H = k x a / eta. Reflected s: E = r_s a_s s; reflected p: r_p is
            // the ratio of the tangential H (along s), H = r_p a_p s / eta, E = eta H x k_r.
            const Vec3 hi = kh.cross(a) / eta_;
            const Vec3c e_r = (c.r_s * as) * s_hat.cast<Complex>() +
                              (c.r_p * ap) * s_hat.cross(kr).cast<Complex>();
            const Vec3c h_r = (c.r_s * as / eta_) * kr.cross(s_hat).cast<Complex>() +
                              (c.r_p * ap / eta_) * s_hat.cast<Complex>();
            kx_.push_back(k_ * kh.x());
            ky_.push_back(k_ * kh.y());
            e_.push_back(a.cast<Complex>() + e_r);
            h_.push_back(hi.cast<Complex>() + h_r);
        }
    }

    /// Total E and H at (x, y, 0^-).
    [[nodiscard]] std::pair<Vec3c, Vec3c> fields(Real x, Real y) const {
        Vec3c E = Vec3c::Zero();
        Vec3c H = Vec3c::Zero();
        for (std::size_t i = 0; i < kx_.size(); ++i) {
            const Complex ph = std::polar(1.0, -(kx_[i] * x + ky_[i] * y));
            E += ph * e_[i];
            H += ph * h_[i];
        }
        return {E, H};
    }

    /// J = n x H and M = -n x E with n = -z at (x, y, 0^-).
    [[nodiscard]] std::pair<Vec3c, Vec3c> currents(Real x, Real y) const {
        const auto [E, H] = fields(x, y);
        return {Vec3c(H.y(), -H.x(), 0.0), Vec3c(-E.y(), E.x(), 0.0)};
    }

    [[nodiscard]] std::size_t plane_waves() const { return kx_.size(); }

private:
    Real k_ = 0, eta_ = 0;
    std::vector<Real> kx_, ky_;
    std::vector<Vec3c> e_, h_;
};

/// Absorbed flux 1/2 Re (E . J*) of the reference fields integrated over the square |x|, |y| <=
/// X (trapezoid with spacing <= `spacing`, bounds on grid lines) [W].
inline Real reference_absorbed_flux(const InterfaceFields& f, Real X, Real spacing) {
    const auto half_cells = static_cast<Index>(std::ceil(X / spacing));
    const Index n = 2 * half_cells;
    const Real h = 2.0 * X / static_cast<Real>(n);
    Real sum = 0;
    for (Index i = 0; i <= n; ++i) {
        const Real x = -X + h * static_cast<Real>(i);
        const Real wx = (i == 0 || i == n) ? 0.5 : 1.0;
        for (Index j = 0; j <= n; ++j) {
            const Real y = -X + h * static_cast<Real>(j);
            const Real wy = (j == 0 || j == n) ? 0.5 : 1.0;
            const auto [J, M] = f.currents(x, y);
            const Vec3c nxm = cross_real(-Vec3::UnitZ(), M);
            sum += wx * wy * 0.5 * dot_conj(nxm, J).real();
        }
    }
    return sum * h * h;
}

/// RWG interpolant of the reference currents on the top face: coefficient of basis n = mean
/// normal component (1 / l_n) int_edge X . nu_n dl (f_n . nu_n = 1 on edge n), nu_n in the plane
/// of the top triangle of the edge, pointing from T_n^+ into T_n^-. Edges without a top triangle
/// get 0. Returns [J; M] coefficients.
inline VectorXc interpolate_reference(const basis::RwgSpace& space, const InterfaceFields& f,
                                      Real spacing) {
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    VectorXc x = VectorXc::Zero(2 * N);
    const kernels::LineRule gl = kernels::gauss_legendre(4);
    for (Index n = 0; n < N; ++n) {
        const Index tp = space.plus_triangle(n);
        const Index tm = space.minus_triangle(n);
        const bool top_p = part_of(mesh, tp, spacing) == Part::Top;
        const bool top_m = part_of(mesh, tm, spacing) == Part::Top;
        if (!top_p && !top_m)
            continue;
        const Vec3 a = mesh.vertices().row(mesh.edges()(n, 0)).transpose();
        const Vec3 b = mesh.vertices().row(mesh.edges()(n, 1)).transpose();
        const Vec3 d = (b - a).normalized();
        // Free vertex of the top triangle; nu points away from p+ (or towards p-).
        const Index pv = top_p ? space.plus_free_vertex(n) : space.minus_free_vertex(n);
        const Vec3 p = mesh.vertices().row(pv).transpose();
        Vec3 nu = (p - a) - (p - a).dot(d) * d;
        nu.normalize();
        if (top_p)
            nu = -nu;
        Complex cj = 0, cm = 0;
        for (std::size_t q = 0; q < gl.nodes.size(); ++q) {
            const Vec3 r = a + 0.5 * (gl.nodes[q] + 1.0) * (b - a);
            const auto [J, M] = f.currents(r.x(), r.y());
            const Real w = 0.5 * gl.weights[q];
            cj += w * (J(0) * nu.x() + J(1) * nu.y() + J(2) * nu.z());
            cm += w * (M(0) * nu.x() + M(1) * nu.y() + M(2) * nu.z());
        }
        x(n) = cj;
        x(N + n) = cm;
    }
    return x;
}

/// Metrics of one region of the top face (see the file comment). Powers in W.
struct TopMetrics {
    Real area = 0;
    Real p_sim = 0;    ///< 1/2 Re int (n x M_sim) . J_sim*
    Real p_ref = 0;    ///< 1/2 Re int (n x M_ref) . J_ref*
    Real p_int = 0;    ///< RWG interpolant of the reference currents
    Real p_ms_jr = 0;  ///< 1/2 Re int (n x M_sim) . J_ref*
    Real p_mr_js = 0;  ///< 1/2 Re int (n x M_ref) . J_sim*
    Complex c_j, c_m;  ///< int X_sim . X_ref* / int |X_ref|^2
    Real err_j = 0;    ///< ||J_sim - J_ref|| / ||J_ref||
    Real err_m = 0;
    Real err_j_int = 0;  ///< interpolant
    Real err_m_int = 0;
    Real m_over_j = 0;  ///< ||M_ref|| / ||J_ref|| [Ohm]
};

/// Metrics over the top triangles whose centroid satisfies `in_region` (Dunavant `degree`).
inline TopMetrics top_metrics(const post::SurfaceSolution& s, const InterfaceFields& f,
                              const VectorXc& interp, Real spacing,
                              const std::function<bool(const Vec3&)>& in_region, int degree) {
    const geometry::TriangleMesh& mesh = s.problem->space->mesh();
    const kernels::TriangleRule& rule = kernels::triangle_rule(degree);
    const post::SurfaceSolution si{s.problem, interp};
    TopMetrics m;
    Complex sjr = 0, smr = 0;
    Real njr = 0, nmr = 0, dj = 0, dm = 0, dji = 0, dmi = 0;
    const Vec3 n = -Vec3::UnitZ();
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        if (part_of(mesh, t, spacing) != Part::Top || !in_region(mesh.centroid(t)))
            continue;
        m.area += mesh.area(t);
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const auto [r, w] = rule_point(mesh, t, rule, q);
            const auto [Js, Ms] = currents_at(s, t, r);
            const auto [Ji, Mi] = currents_at(si, t, r);
            const auto [Jr, Mr] = f.currents(r.x(), r.y());
            m.p_sim += w * 0.5 * dot_conj(cross_real(n, Ms), Js).real();
            m.p_ref += w * 0.5 * dot_conj(cross_real(n, Mr), Jr).real();
            m.p_int += w * 0.5 * dot_conj(cross_real(n, Mi), Ji).real();
            m.p_ms_jr += w * 0.5 * dot_conj(cross_real(n, Ms), Jr).real();
            m.p_mr_js += w * 0.5 * dot_conj(cross_real(n, Mr), Js).real();
            sjr += w * dot_conj(Js, Jr);
            smr += w * dot_conj(Ms, Mr);
            njr += w * Jr.squaredNorm();
            nmr += w * Mr.squaredNorm();
            dj += w * (Js - Jr).squaredNorm();
            dm += w * (Ms - Mr).squaredNorm();
            dji += w * (Ji - Jr).squaredNorm();
            dmi += w * (Mi - Mr).squaredNorm();
        }
    }
    if (!(njr > 0.0) || !(nmr > 0.0))
        throw std::invalid_argument("top_metrics: empty region or vanishing reference currents");
    m.c_j = sjr / njr;
    m.c_m = smr / nmr;
    m.err_j = std::sqrt(dj / njr);
    m.err_m = std::sqrt(dm / nmr);
    m.err_j_int = std::sqrt(dji / njr);
    m.err_m_int = std::sqrt(dmi / nmr);
    m.m_over_j = std::sqrt(nmr / njr);
    return m;
}

}  // namespace ag_absorption
