#pragma once
/// @file scattering.hpp
/// Scattering observables.
#include "specklebem/postprocessing/fields.hpp"

namespace specklebem::post {

/// Bistatic radar cross section sigma(theta) in a given scattering plane
/// (used for the Mie validation, Eq. 7 of the paper for the error metric).
///
/// sigma(theta) = 4 pi |F(k_hat(theta))|^2 / |E_inc|^2 (far_field), with k_hat(theta) the
/// rotation of +z about the unit plane normal by theta (right-hand rule): theta = 0 is forward
/// scattering (+z), theta = pi back-scattering. plane_normal = +y gives
/// k_hat = (sin theta, 0, cos theta), i.e. the spherical angles (theta, phi = 0) of docs/06 and
/// reference::MieSolution::bistatic_rcs(theta, 0) for theta in [0, pi] (polar angle 2 pi - theta
/// at phi = pi for theta in (pi, 2 pi)); plane_normal = -x gives k_hat = (0, sin theta, cos theta),
/// i.e. (theta, phi = pi / 2). |E_inc| = |problem->excitation->electric_field(0)| (the incident
/// amplitude at the origin) if an excitation is set, otherwise 1 V/m (logged once per process
/// with SBEM_WARN).
/// @throws std::invalid_argument for a zero / non-finite plane_normal, a plane normal that is not
///         perpendicular to z (|n_hat . z| > 1e-9), a zero incident amplitude, and everything
///         far_field throws for.
VectorXr bistatic_rcs(const SurfaceSolution& s, const Vec3& plane_normal,
                      const VectorXr& angles_rad);

/// Differential reflection coefficient from fields on a cylindrical surface,
/// averaged along the cylinder axis (Fig. 9).
///
/// E_cyl holds the field on cylinder_grid(...) (row = k * n_y + l, theta outer). Returns, per
/// theta_k, the mean over the n_y axial samples of |E|^2 [V^2/m^2]. This is the relative DRC:
/// the normalisation by the incident power (and the r factor) follows in Phase 5.
/// @throws std::invalid_argument for n_theta < 1, n_y < 1 or E_cyl.rows() != n_theta * n_y.
VectorXr differential_reflection_coefficient(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E_cyl,
                                             int n_theta, int n_y);

/// Decomposition into co- and cross-polarised intensities w.r.t. the incident polarisation.
/// With e_co = co_pol_direction / |co_pol_direction|: co = |E . e_co|^2,
/// cross = |E|^2 - co (clamped at 0 against rounding), so co + cross = |E|^2.
/// @throws std::invalid_argument for a zero or non-finite co_pol_direction.
struct PolarizedIntensity {
    VectorXr co, cross;
};
PolarizedIntensity polarized_intensity(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                                       const Vec3& co_pol_direction);

}  // namespace specklebem::post
