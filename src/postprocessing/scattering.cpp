/// @file scattering.cpp
/// Scattering observables from the surface currents and from sampled fields: bistatic RCS
/// (4 pi |F|^2 / |E_inc|^2 with the far-field amplitude of fields.cpp), the relative and absolute
/// differential reflection coefficient on a cylinder, co-/cross-polarised intensities, and the
/// absolute far-field observables of WP25 (direction grids, radiated power, DRC per steradian).
#include "specklebem/postprocessing/scattering.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/kernels/quadrature.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>

namespace specklebem::post {

VectorXr bistatic_rcs(const SurfaceSolution& s, const Vec3& plane_normal,
                      const VectorXr& angles_rad) {
    const Real len = plane_normal.norm();
    if (!std::isfinite(len) || !(len > 0.0)) {
        throw std::invalid_argument("bistatic_rcs: plane_normal must be non-zero and finite");
    }
    const Vec3 n = plane_normal / len;
    if (std::abs(n.z()) > 1e-9) {
        throw std::invalid_argument(
            "bistatic_rcs: plane_normal must be perpendicular to z (the plane contains the "
            "forward direction +z)");
    }
    if (!angles_rad.allFinite()) {
        throw std::invalid_argument("bistatic_rcs: angles must be finite");
    }
    if (s.problem == nullptr) {
        throw std::invalid_argument("bistatic_rcs: SurfaceSolution::problem is null");
    }
    Real e_inc = 1.0;  // V/m, when no excitation is attached
    if (s.problem->excitation != nullptr) {
        e_inc = s.problem->excitation->electric_field(Vec3::Zero()).norm();
        if (!(e_inc > 0.0) || !std::isfinite(e_inc)) {
            throw std::invalid_argument(
                "bistatic_rcs: the incident field vanishes (or is not finite) at the origin");
        }
    } else {
        static std::once_flag warned;  // once per process, not per call
        std::call_once(warned, [] {
            SBEM_WARN("bistatic_rcs: problem has no excitation, assuming |E_inc| = 1 V/m");
        });
    }
    // Rodrigues rotation of z about n (n perpendicular to z): k = z cos t + (n x z) sin t.
    const Vec3 z = Vec3::UnitZ();
    const Vec3 nxz = n.cross(z);
    const Index na = angles_rad.size();
    Vertices dirs(na, 3);
    for (Index i = 0; i < na; ++i) {
        const Real t = angles_rad(i);
        dirs.row(i) = (std::cos(t) * z + std::sin(t) * nxz).transpose();
    }
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    far_field(s, dirs, F);
    VectorXr sigma(na);
    const Real scale = 4.0 * constants::pi / (e_inc * e_inc);
    for (Index i = 0; i < na; ++i) {
        sigma(i) = scale * F.row(i).squaredNorm();
    }
    return sigma;
}

VectorXr differential_reflection_coefficient(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E_cyl,
                                             int n_theta, int n_y) {
    if (n_theta < 1 || n_y < 1) {
        throw std::invalid_argument(
            "differential_reflection_coefficient: n_theta and n_y must be >= 1");
    }
    const auto nt = static_cast<Index>(n_theta);
    const auto ny = static_cast<Index>(n_y);
    if (E_cyl.rows() != nt * ny) {
        throw std::invalid_argument("differential_reflection_coefficient: E_cyl has " +
                                    std::to_string(E_cyl.rows()) +
                                    " rows, expected n_theta * n_y = " + std::to_string(nt * ny));
    }
    VectorXr drc(nt);
    for (Index k = 0; k < nt; ++k) {
        Real sum = 0.0;
        for (Index l = 0; l < ny; ++l) {
            sum += E_cyl.row(k * ny + l).squaredNorm();
        }
        drc(k) = sum / static_cast<Real>(ny);
    }
    return drc;
}

namespace {

/// eta_1 = eta_0 sqrt(mu_r / eps_r) [Ohm] of a lossless background (both real and > 0).
Real lossless_impedance(const material::Material& m, const std::string& who) {
    const bool ok = std::isfinite(m.eps_r.real()) && std::isfinite(m.mu_r.real()) &&
                    m.eps_r.imag() == 0.0 && m.mu_r.imag() == 0.0 && m.eps_r.real() > 0.0 &&
                    m.mu_r.real() > 0.0;
    if (!ok) {
        throw std::invalid_argument(
            who +
            ": the background R1 must be lossless (real eps_r > 0 and real mu_r > 0); in a lossy "
            "R1 the far field decays and the radiated power at infinity is not defined");
    }
    return constants::eta0 * std::sqrt(m.mu_r.real() / m.eps_r.real());
}

void require_power(Real incident_power, const std::string& who) {
    if (!(std::isfinite(incident_power) && incident_power > 0.0)) {
        throw std::invalid_argument(who + ": incident_power must be finite and > 0 [W]");
    }
}

/// eta_1 of the solution's exterior. @throws std::invalid_argument for a null problem.
Real exterior_impedance(const SurfaceSolution& s, const std::string& who) {
    if (s.problem == nullptr) {
        throw std::invalid_argument(who + ": SurfaceSolution::problem is null");
    }
    return lossless_impedance(s.problem->exterior, who);
}

void require_directions(const Vertices& directions, const std::string& who) {
    if (directions.rows() == 0) {
        throw std::invalid_argument(who + ": no directions (empty grid)");
    }
}

}  // namespace

VectorXr differential_reflection_coefficient(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E_cyl,
                                             int n_theta, int n_y, Real radius, Real incident_power,
                                             const material::Material& background) {
    const std::string who = "differential_reflection_coefficient";
    if (!(std::isfinite(radius) && radius > 0.0)) {
        throw std::invalid_argument(who + ": radius must be finite and > 0 [m]");
    }
    require_power(incident_power, who);
    const Real eta1 = lossless_impedance(background, who);
    const VectorXr mean = differential_reflection_coefficient(E_cyl, n_theta, n_y);
    return mean * (radius * radius / (2.0 * eta1 * incident_power));
}

SphereGrid hemisphere_grid(Real z_sign, int n_theta, int n_phi) {
    if (!(z_sign == 1.0 || z_sign == -1.0)) {
        throw std::invalid_argument("hemisphere_grid: z_sign must be +1 or -1");
    }
    if (n_theta < 1 || n_phi < 4) {
        throw std::invalid_argument("hemisphere_grid: need n_theta >= 1 and n_phi >= 4");
    }
    // The operations (and their order) are those of the WP22c test helper, so that the recorded
    // reflectances are reproduced bitwise.
    const kernels::LineRule gl = kernels::gauss_legendre(n_theta);
    const Real dp = 2.0 * constants::pi / static_cast<Real>(n_phi);
    SphereGrid g;
    const Index rows = static_cast<Index>(n_theta) * n_phi;
    g.dirs.resize(rows, 3);
    g.weights.resize(rows);
    g.theta.resize(rows);
    for (int i = 0; i < n_theta; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const Real t = 0.25 * constants::pi * (gl.nodes[ui] + 1.0);
        const Real wt = 0.25 * constants::pi * gl.weights[ui] * std::sin(t) * dp;
        for (int j = 0; j < n_phi; ++j) {
            const Real p = static_cast<Real>(j) * dp;
            const Index row = static_cast<Index>(i) * n_phi + j;
            g.dirs(row, 0) = std::sin(t) * std::cos(p);
            g.dirs(row, 1) = std::sin(t) * std::sin(p);
            g.dirs(row, 2) = z_sign * std::cos(t);
            g.weights(row) = wt;
            g.theta(row) = t;
        }
    }
    return g;
}

SphereGrid sphere_grid(int n_theta, int n_phi) {
    if (n_theta < 1 || n_phi < 4) {
        throw std::invalid_argument("sphere_grid: need n_theta >= 1 and n_phi >= 4");
    }
    const kernels::LineRule gl = kernels::gauss_legendre(n_theta);
    const Real dp = 2.0 * constants::pi / static_cast<Real>(n_phi);
    SphereGrid g;
    const Index rows = static_cast<Index>(n_theta) * n_phi;
    g.dirs.resize(rows, 3);
    g.weights.resize(rows);
    g.theta.resize(rows);
    for (int i = 0; i < n_theta; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const Real t = 0.5 * constants::pi * (gl.nodes[ui] + 1.0);
        const Real wt = 0.5 * constants::pi * gl.weights[ui] * std::sin(t) * dp;
        for (int j = 0; j < n_phi; ++j) {
            const Real p = static_cast<Real>(j) * dp;
            const Index row = static_cast<Index>(i) * n_phi + j;
            g.dirs(row, 0) = std::sin(t) * std::cos(p);
            g.dirs(row, 1) = std::sin(t) * std::sin(p);
            g.dirs(row, 2) = std::cos(t);
            g.weights(row) = wt;
            g.theta(row) = t;
        }
    }
    return g;
}

Real radiated_power(const SurfaceSolution& s, const Vertices& directions, const VectorXr& weights,
                    const FieldOptions& opt) {
    const std::string who = "radiated_power";
    require_directions(directions, who);
    if (weights.size() != directions.rows()) {
        throw std::invalid_argument(who + ": " + std::to_string(weights.size()) + " weights for " +
                                    std::to_string(directions.rows()) + " directions");
    }
    if (!weights.allFinite() || (weights.array() < 0.0).any()) {
        throw std::invalid_argument(who + ": weights must be finite and >= 0 [sr]");
    }
    const Real eta1 = exterior_impedance(s, who);
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    far_field(s, directions, F, opt);
    // Same summation as the WP22c test helper (hemisphere_power): row by row, w |F|^2 / (2 eta).
    Real total = 0.0;
    for (Index r = 0; r < F.rows(); ++r) {
        total += weights(r) * F.row(r).squaredNorm() / (2.0 * eta1);
    }
    return total;
}

Real radiated_power(const SurfaceSolution& s, const SphereGrid& grid, const FieldOptions& opt) {
    return radiated_power(s, grid.dirs, grid.weights, opt);
}

VectorXr drc(const SurfaceSolution& s, const Vertices& directions, Real incident_power,
             const FieldOptions& opt) {
    const std::string who = "drc";
    require_power(incident_power, who);
    require_directions(directions, who);
    const Real eta1 = exterior_impedance(s, who);
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    far_field(s, directions, F, opt);
    const Real scale = 1.0 / (2.0 * eta1 * incident_power);
    VectorXr d(F.rows());
    for (Index r = 0; r < F.rows(); ++r) {
        d(r) = F.row(r).squaredNorm() * scale;
    }
    return d;
}

PolarizedIntensity drc_polarized(const SurfaceSolution& s, const Vertices& directions,
                                 Real incident_power, const Vec3& co_pol_direction,
                                 const FieldOptions& opt) {
    const std::string who = "drc_polarized";
    const Real len = co_pol_direction.norm();
    if (!std::isfinite(len) || !(len > 0.0)) {
        throw std::invalid_argument(who + ": co_pol_direction must be non-zero and finite");
    }
    require_power(incident_power, who);
    require_directions(directions, who);
    const Real eta1 = exterior_impedance(s, who);
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    far_field(s, directions, F, opt);
    PolarizedIntensity p = polarized_intensity(F, co_pol_direction);
    const Real scale = 1.0 / (2.0 * eta1 * incident_power);
    p.co *= scale;
    p.cross *= scale;
    return p;
}

PolarizedIntensity polarized_intensity(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                                       const Vec3& co_pol_direction) {
    const Real len = co_pol_direction.norm();
    if (!std::isfinite(len) || !(len > 0.0)) {
        throw std::invalid_argument(
            "polarized_intensity: co_pol_direction must be non-zero and finite");
    }
    const Vec3 e = co_pol_direction / len;
    const Index n = E.rows();
    PolarizedIntensity out;
    out.co.resize(n);
    out.cross.resize(n);
    for (Index i = 0; i < n; ++i) {
        const Complex proj = E(i, 0) * e.x() + E(i, 1) * e.y() + E(i, 2) * e.z();
        const Real total = E.row(i).squaredNorm();
        const Real co = std::norm(proj);
        out.co(i) = co;
        out.cross(i) = std::max(total - co, Real{0});
    }
    return out;
}

}  // namespace specklebem::post
