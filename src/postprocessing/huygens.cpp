/// @file huygens.cpp
/// Huygens-plane far field of the PML route (ADR 0010, WP-PML6); the equations and conventions
/// are in include/specklebem/postprocessing/huygens.hpp.
///
/// Sign convention (docs/06): the equivalent currents of a surface separating the sources from
/// the observation region are J = n x H, M = -n x E with n pointing into the observation region.
/// For the plane below the surface (PML route, R1 below) the observation half space is z < z_p,
/// so n = -z_hat: J_H = -z_hat x H = (H_y, -H_x, 0), M_H = z_hat x E = (-E_y, E_x, 0). With these
/// currents the radiation integral of post::far_field gives the far field of the scattered wave
/// (checked on dipole fields and against the windowed currents of flat Ag / Si).
#include "specklebem/postprocessing/huygens.hpp"

#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace specklebem::post {

namespace {

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

constexpr Complex kJ{0.0, 1.0};
constexpr Real kFourPi = 4.0 * constants::pi;

/// a x b for complex vectors without conjugation (Eigen's cross() conjugates complex results).
Vec3c cross_cc(const Vec3c& a, const Vec3c& b) {
    return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// Number of non-negative grid indices i with i h < outer (i = 0 .. n - 1).
Index half_count(Real outer, Real h) {
    const Real m = outer / h;
    auto n = static_cast<Index>(std::floor(m)) + 1;
    // Exclude a node exactly at outer (weight 0 there anyway) and guard the rounding of floor.
    while (n > 0 && !(static_cast<Real>(n - 1) * h < outer)) --n;
    while (static_cast<Real>(n) * h < outer) ++n;
    return n;
}

void check_fields(const HuygensPlane& p, const FieldMatrix& E, const FieldMatrix& H,
                  const std::string& who) {
    if (E.rows() != p.points.rows() || H.rows() != p.points.rows()) {
        throw std::invalid_argument(who + ": E and H need one row per plane point (" +
                                    std::to_string(p.points.rows()) + "), got " +
                                    std::to_string(E.rows()) + " / " + std::to_string(H.rows()));
    }
}

}  // namespace

HuygensPlane huygens_plane(Real z, Real normal_z, const CurrentWindow& window, Real spacing) {
    const std::string who = "huygens_plane";
    if (!std::isfinite(z)) {
        throw std::invalid_argument(who + ": z must be finite");
    }
    if (!(normal_z == 1.0 || normal_z == -1.0)) {
        throw std::invalid_argument(who + ": normal_z must be +1 or -1");
    }
    (void)current_window_value(window, 0.0);  // validates the window
    if (!(std::isfinite(spacing) && spacing > 0.0)) {
        throw std::invalid_argument(who + ": spacing must be finite and > 0");
    }
    const Real per_axis = 2.0 * window.outer_half / spacing + 1.0;
    if (!(per_axis * per_axis <= static_cast<Real>(kMaxHuygensPoints))) {
        throw std::invalid_argument(who + ": the plane would have more than " +
                                    std::to_string(kMaxHuygensPoints) +
                                    " points (outer_half / spacing too large)");
    }
    const Index n = half_count(window.outer_half, spacing);
    std::vector<Real> coord;  // i h for i = -(n - 1) .. n - 1 (exactly antisymmetric)
    coord.reserve(static_cast<std::size_t>(2 * n));
    for (Index i = -(n - 1); i <= n - 1; ++i) coord.push_back(static_cast<Real>(i) * spacing);
    std::vector<Real> wx;  // window weights of the kept points
    std::vector<Vec3> px;
    const Real h2 = spacing * spacing;
    for (const Real x : coord) {
        for (const Real y : coord) {
            const Real w = current_window_value(window, std::max(std::abs(x), std::abs(y)));
            if (w == 0.0)
                continue;
            px.emplace_back(x, y, z);
            wx.push_back(h2 * w);
        }
    }
    HuygensPlane p;
    p.z = z;
    p.normal_z = normal_z;
    p.spacing = spacing;
    p.window = window;
    p.points.resize(static_cast<Index>(px.size()), 3);
    p.weights.resize(static_cast<Index>(wx.size()));
    for (std::size_t i = 0; i < px.size(); ++i) {
        p.points.row(static_cast<Index>(i)) = px[i].transpose();
        p.weights(static_cast<Index>(i)) = wx[i];
    }
    return p;
}

HuygensPlane huygens_plane(const SurfaceSolution& s, const FieldOptions& opt) {
    const std::string who = "huygens_plane";
    if (s.problem == nullptr || s.problem->space == nullptr) {
        throw std::invalid_argument(who + ": SurfaceSolution::problem or its space is null");
    }
    const geometry::TriangleMesh& mesh = s.problem->space->mesh();
    if (mesh.is_closed()) {
        throw std::invalid_argument(
            who +
            ": a Huygens plane needs an open surface facing +-z (PML route), not a closed "
            "mesh");
    }
    Real a_nz = 0.0;
    Real a_sum = 0.0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        a_nz += mesh.area(t) * mesh.normal(t).z();
        a_sum += mesh.area(t);
    }
    if (!(std::abs(a_nz) >= 0.5 * a_sum)) {
        throw std::invalid_argument(who +
                                    ": the open surface does not face +-z (|sum A n_z| < 0.5 "
                                    "sum A); the side of R1 is undecided");
    }
    const Real normal_z = a_nz < 0.0 ? -1.0 : 1.0;
    const Real omega = s.problem->omega;
    if (!(std::isfinite(omega) && omega > 0.0)) {
        throw std::invalid_argument(who + ": Problem::omega must be finite and > 0");
    }
    const Complex k1 = s.problem->exterior.wavenumber(omega);
    if (!(k1.real() > 0.0)) {
        throw std::invalid_argument(who + ": the exterior R1 must carry propagating waves");
    }
    const Real lambda1 = 2.0 * constants::pi / k1.real();
    const HuygensOptions ho = opt.huygens.value_or(HuygensOptions{});
    const Real z_h = ho.z_h.value_or(lambda1);
    if (!(std::isfinite(z_h) && z_h > 0.0)) {
        throw std::invalid_argument(who + ": HuygensOptions::z_h must be finite and > 0");
    }
    const Real spacing = ho.spacing.value_or(0.25 * lambda1);
    if (!(std::isfinite(spacing) && spacing > 0.0 && spacing <= 0.5 * lambda1)) {
        throw std::invalid_argument(who +
                                    ": HuygensOptions::spacing must be finite, > 0 and <= "
                                    "lambda_1 / 2 = " +
                                    std::to_string(0.5 * lambda1) + " m");
    }
    if (!ho.window && !opt.window) {
        throw std::invalid_argument(
            who + ": no plane window (HuygensOptions::window or FieldOptions::window)");
    }
    const CurrentWindow window = ho.window ? *ho.window : *opt.window;
    (void)current_window_value(window, 0.0);
    if (op::has_stretch(*s.problem)) {
        const geometry::CoordinateStretch& st = *s.problem->stretch;
        if (!st.profile_x() || !st.profile_y()) {
            throw std::logic_error(who +
                                   ": internal invariant violated, a non-identity stretch "
                                   "without a profile on both axes");
        }
        const Real sx = st.profile_x()->start;
        const Real sy = st.profile_y()->start;
        if (!(window.outer_half <= std::min(sx, sy))) {
            throw std::invalid_argument(
                who + ": the plane window ends at outer_half = " +
                std::to_string(window.outer_half) + " m, beyond the PML stretch start " +
                std::to_string(std::min(sx, sy)) + " m (plane points over the PML)");
        }
    }
    const Vertices& V = mesh.vertices();
    const Real z = normal_z < 0.0 ? V.col(2).minCoeff() - z_h : V.col(2).maxCoeff() + z_h;
    return huygens_plane(z, normal_z, window, spacing);
}

HuygensPlaneField huygens_plane_field(const SurfaceSolution& s, const FieldOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    HuygensPlaneField f;
    f.plane = huygens_plane(s, opt);
    FieldOptions fo;
    fo.quad_degree = opt.quad_degree;
    fo.min_distance_factor = opt.min_distance_factor;
    exterior_scattered_field(s, f.plane.points, f.E, f.H, fo,
                             !opt.huygens || opt.huygens->pml_sources);
    f.k = s.problem->exterior.wavenumber(s.problem->omega);
    f.eta = s.problem->exterior.wave_impedance(s.problem->omega);
    f.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    return f;
}

HuygensPlaneField huygens_reweight(const HuygensPlaneField& f, const CurrentWindow& window) {
    const std::string who = "huygens_reweight";
    (void)current_window_value(window, 0.0);
    if (!(window.outer_half <= f.plane.window.outer_half)) {
        throw std::invalid_argument(
            who + ": the new window ends at " + std::to_string(window.outer_half) +
            " m, beyond the evaluated plane (" + std::to_string(f.plane.window.outer_half) + " m)");
    }
    const HuygensPlane& p = f.plane;
    check_fields(p, f.E, f.H, who);
    const Real h2 = p.spacing * p.spacing;
    std::vector<Index> keep;
    std::vector<Real> w;
    for (Index i = 0; i < p.points.rows(); ++i) {
        const Real t = std::max(std::abs(p.points(i, 0)), std::abs(p.points(i, 1)));
        const Real v = current_window_value(window, t);
        if (v == 0.0)
            continue;
        keep.push_back(i);
        w.push_back(h2 * v);
    }
    HuygensPlaneField out;
    out.plane.z = p.z;
    out.plane.normal_z = p.normal_z;
    out.plane.spacing = p.spacing;
    out.plane.window = window;
    const auto n = static_cast<Index>(keep.size());
    out.plane.points.resize(n, 3);
    out.plane.weights.resize(n);
    out.E.resize(n, 3);
    out.H.resize(n, 3);
    for (Index r = 0; r < n; ++r) {
        const Index i = keep[static_cast<std::size_t>(r)];
        out.plane.points.row(r) = p.points.row(i);
        out.plane.weights(r) = w[static_cast<std::size_t>(r)];
        out.E.row(r) = f.E.row(i);
        out.H.row(r) = f.H.row(i);
    }
    out.k = f.k;
    out.eta = f.eta;
    out.seconds = f.seconds;
    return out;
}

void huygens_radiate(const HuygensPlane& plane, const FieldMatrix& E, const FieldMatrix& H,
                     Complex k, Complex eta, const Vertices& directions, FieldMatrix& F) {
    const std::string who = "huygens_radiate";
    check_fields(plane, E, H, who);
    if (plane.weights.size() != plane.points.rows()) {
        throw std::invalid_argument(who + ": plane weights and points differ in size");
    }
    const Real nz = plane.normal_z;
    if (!(nz == 1.0 || nz == -1.0)) {
        throw std::invalid_argument(who + ": plane.normal_z must be +1 or -1");
    }
    const Index nd = directions.rows();
    std::vector<Vec3> k_hat(static_cast<std::size_t>(nd));
    for (Index i = 0; i < nd; ++i) {
        const Vec3 d = directions.row(i).transpose();
        const Real len = d.norm();
        if (!std::isfinite(len) || !(len > 0.0)) {
            throw std::invalid_argument(who + ": direction " + std::to_string(i) +
                                        " is zero or not finite");
        }
        const Vec3 u = d / len;
        if (nz * u.z() < -1e-12) {
            throw std::invalid_argument(
                who + ": direction " + std::to_string(i) +
                " points to the surface side of the Huygens plane (n_hat . k_hat < 0), where the "
                "plane currents give no far field");
        }
        k_hat[static_cast<std::size_t>(i)] = u;
    }
    // Weighted equivalent currents J_H = n x H, M_H = -n x E with n = nz z_hat:
    // n x V = nz (-V_y, V_x, 0).
    const Index np = plane.points.rows();
    std::vector<Vec3c> jw(static_cast<std::size_t>(np));
    std::vector<Vec3c> mw(static_cast<std::size_t>(np));
    for (Index p = 0; p < np; ++p) {
        const Real w = plane.weights(p);
        jw[static_cast<std::size_t>(p)] = Vec3c(-nz * H(p, 1), nz * H(p, 0), 0.0) * w;
        mw[static_cast<std::size_t>(p)] = Vec3c(nz * E(p, 1), -nz * E(p, 0), 0.0) * w;
    }
    const Complex prefactor = kJ * k / kFourPi;
    const Real kr = k.real();
    const Real ki = k.imag();
    F.resize(nd, 3);
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (Index i = 0; i < nd; ++i) {
        const Vec3& kh = k_hat[static_cast<std::size_t>(i)];
        Vec3c Jt = Vec3c::Zero();
        Vec3c Mt = Vec3c::Zero();
        for (Index p = 0; p < np; ++p) {
            const Real proj = kh.x() * plane.points(p, 0) + kh.y() * plane.points(p, 1) +
                              kh.z() * plane.points(p, 2);
            // exp(j k proj), k = kr + j ki
            const Real amp = std::exp(-ki * proj);
            const Complex e(amp * std::cos(kr * proj), amp * std::sin(kr * proj));
            Jt += e * jw[static_cast<std::size_t>(p)];
            Mt += e * mw[static_cast<std::size_t>(p)];
        }
        const Vec3c khc = kh.cast<Complex>();
        F.row(i) = (prefactor * cross_cc(khc, eta * cross_cc(khc, Jt) + Mt)).transpose();
    }
}

void huygens_radiate(const HuygensPlaneField& f, const Vertices& directions, FieldMatrix& F) {
    huygens_radiate(f.plane, f.E, f.H, f.k, f.eta, directions, F);
}

void huygens_far_field(const SurfaceSolution& s, const Vertices& directions, FieldMatrix& F,
                       const FieldOptions& opt) {
    const HuygensPlaneField f = huygens_plane_field(s, opt);
    huygens_radiate(f, directions, F);
}

}  // namespace specklebem::post
