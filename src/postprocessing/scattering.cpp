/// @file scattering.cpp
/// Scattering observables from the surface currents and from sampled fields: bistatic RCS
/// (4 pi |F|^2 / |E_inc|^2 with the far-field amplitude of fields.cpp), the relative
/// differential reflection coefficient on a cylinder, and co-/cross-polarised intensities.
#include "specklebem/postprocessing/scattering.hpp"

#include "specklebem/core/logging.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
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
