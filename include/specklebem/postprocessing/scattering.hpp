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
/// theta_k, the mean over the n_y axial samples of |E|^2 [V^2/m^2]. This is the relative DRC;
/// the absolute one (per steradian, normalised by the incident power) is the overload below
/// (WP25), or drc() from the far field.
/// @throws std::invalid_argument for n_theta < 1, n_y < 1 or E_cyl.rows() != n_theta * n_y.
VectorXr differential_reflection_coefficient(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E_cyl,
                                             int n_theta, int n_y);

/// Absolute DRC [1/sr] from the scattered field on a cylinder of the given radius r (WP25):
///   D_cyl(theta_k) = r^2 / (2 eta_1 P_inc) * mean_l |E(theta_k, y_l)|^2,
/// i.e. the relative DRC above times r^2 / (2 eta_1 P_inc), with eta_1 = eta_0 sqrt(mu_r / eps_r)
/// of the lossless background R1 and P_inc [W] the incident beam power (e.g.
/// Simulation::incident_power()). E_cyl must be the scattered field (outside the incident beam
/// the total field is the same) on cylinder_grid(radius, ...) [V/m].
///
/// It approximates the far-field DRC D(k_hat) = |F|^2 / (2 eta_1 P_inc) of drc() only in the
/// far zone: with E = F exp(-jk r) / r (1 + O(1/r)) the relative deviation of |E|^2 r^2 from
/// |F|^2 falls like 1 / r, with contributions of order 1 / (k r) (near-field terms), L / r
/// (source offset from the origin, L the illuminated extent) and k L^2 / r (Fresnel phase), so
/// D_cyl -> D needs r >> L^2 / lambda (and r >> L, lambda). The axial samples at y_l != 0 lie at
/// the distance sqrt(r^2 + y_l^2) in a direction tilted out of the xz-plane by atan(y_l / r);
/// they are averaged as they are (the Fig. 9 cylinder, |y| << r). For quantitative work use
/// drc() on far-field directions.
/// @throws std::invalid_argument for what the relative overload throws for, a radius or
///         incident_power not finite and > 0, and a lossy or non-positive background (eps_r or
///         mu_r not real and > 0: no far-field power in a lossy R1).
VectorXr differential_reflection_coefficient(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E_cyl,
                                             int n_theta, int n_y, Real radius, Real incident_power,
                                             const material::Material& background);

/// Decomposition into co- and cross-polarised intensities w.r.t. the incident polarisation.
/// With e_co = co_pol_direction / |co_pol_direction|: co = |E . e_co|^2,
/// cross = |E|^2 - co (clamped at 0 against rounding), so co + cross = |E|^2.
/// @throws std::invalid_argument for a zero or non-finite co_pol_direction.
struct PolarizedIntensity {
    VectorXr co, cross;
};
PolarizedIntensity polarized_intensity(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                                       const Vec3& co_pol_direction);

// ---------------------------------------------------------------------------------------------
// Absolute far-field observables (WP25): direction grids, radiated power, DRC per steradian.
//
// Far-field amplitude F of far_field(): E_s ~ F exp(-j k_1 r) / r in R1, so the radiated power
// per solid angle is dP / dOmega = |F|^2 / (2 eta_1) [W/sr], eta_1 the wave impedance of the
// lossless background R1 (Problem::exterior). In a lossy R1 the far field decays and the power
// at infinity is not defined: these functions throw there.

/// Direction grid on the unit sphere with solid-angle quadrature weights.
struct SphereGrid {
    Vertices dirs;  ///< unit directions k_hat (one per row)
    /// Solid-angle weights [sr]; their sum is the Gauss-Legendre rule for int sin(theta), i.e.
    /// 2 pi (hemisphere) or 4 pi (sphere) to rounding from n_theta >= 7 / 8 (relative error
    /// 0.11 / 0.57 at n_theta = 1, 2e-8 / 8e-6 at 4).
    VectorXr weights;
    VectorXr theta;  ///< polar angle of each row from the grid's pole [rad]
};

/// Product grid on a hemisphere around the pole z_sign z_hat (z_sign = -1: the reflection
/// hemisphere k_z < 0 of a rough surface lit from z < 0; +1: the forward hemisphere): n_theta
/// Gauss-Legendre nodes theta_i in (0, pi / 2) (polar angle from the pole) times n_phi equispaced
/// azimuths phi_j = j 2 pi / n_phi (trapezoid), k_hat = (sin theta cos phi, sin theta sin phi,
/// z_sign cos theta), weights w = (pi / 4) w_GL sin(theta) (2 pi / n_phi). Row = i * n_phi + j.
/// Spectrally accurate for smooth integrands (no node at grazing theta = pi / 2, where box-edge
/// diffraction radiates; the WP22c reflectance grid of tests/support/fresnel_flat_support.hpp).
/// The trapezoid rule integrates trigonometric polynomials of degree < n_phi in phi exactly.
/// @throws std::invalid_argument for z_sign not exactly +1 or -1, n_theta < 1 or n_phi < 4.
[[nodiscard]] SphereGrid hemisphere_grid(Real z_sign, int n_theta, int n_phi);

/// Product grid on the full sphere: n_theta Gauss-Legendre nodes theta_i in (0, pi) (polar angle
/// from +z) times n_phi equispaced azimuths (trapezoid), weights (pi / 2) w_GL sin(theta)
/// (2 pi / n_phi). Row = i * n_phi + j. @throws std::invalid_argument for n_theta < 1 or
/// n_phi < 4.
[[nodiscard]] SphereGrid sphere_grid(int n_theta, int n_phi);

/// Power [W] radiated by the solved currents into the solid angle covered by a quadrature rule,
///   P = (1 / 2 eta_1) sum_i w_i |F(k_hat_i)|^2,
/// with F = far_field(s, directions, F, opt) (so FieldOptions::window applies: the PML route
/// needs it, Simulation::far_field_options()) and eta_1 = eta_0 sqrt(mu_r / eps_r) of
/// Problem::exterior. weights [sr] are the solid-angle weights of the directions (rows,
/// normalised by far_field). With a box-route rough surface or a sphere the full sphere gives the
/// total scattered power (cross section C_sca = P / (|E_0|^2 / (2 eta_1)) for a plane wave).
/// @throws std::invalid_argument for a null problem, no directions, weights.size() !=
///         directions.rows(), negative or non-finite weights, a lossy or non-positive exterior
///         (eps_r or mu_r not real and > 0), and everything far_field() throws for.
[[nodiscard]] Real radiated_power(const SurfaceSolution& s, const Vertices& directions,
                                  const VectorXr& weights, const FieldOptions& opt = {});
/// Same over grid.dirs / grid.weights (hemisphere_grid(), sphere_grid()).
[[nodiscard]] Real radiated_power(const SurfaceSolution& s, const SphereGrid& grid,
                                  const FieldOptions& opt = {});

/// Absolute differential reflection (scattering) coefficient per steradian [1/sr]:
///   D(k_hat) = |F(k_hat)|^2 / (2 eta_1 P_inc) = (dP / dOmega) / P_inc,
/// so that the integral of D over the reflection hemisphere is the reflectance R and
/// sum_i w_i D_i = radiated_power(s, grid, opt) / P_inc on any grid. incident_power P_inc [W] is
/// the power of the incident beam (Simulation::incident_power(): AngularSpectrumBeam::power(),
/// GaussianBeam::power()). F and eta_1 as in radiated_power(); directions are normalised.
/// @throws std::invalid_argument for incident_power not finite and > 0, no directions, a null
///         problem, a lossy or non-positive exterior, and everything far_field() throws for.
[[nodiscard]] VectorXr drc(const SurfaceSolution& s, const Vertices& directions,
                           Real incident_power, const FieldOptions& opt = {});

/// Co- and cross-polarised parts of drc() [1/sr] with the PolarizedIntensity semantics: with
/// e_co = co_pol_direction / |co_pol_direction| (a fixed Cartesian vector, e.g. the incident
/// polarisation; not projected onto the plane transverse to k_hat), co = |F . e_co|^2 /
/// (2 eta_1 P_inc) and cross = D - co (clamped at 0 against rounding), so co + cross = D to
/// rounding. One far-field evaluation.
/// @throws std::invalid_argument for a zero or non-finite co_pol_direction and everything drc()
///         throws for.
[[nodiscard]] PolarizedIntensity drc_polarized(const SurfaceSolution& s, const Vertices& directions,
                                               Real incident_power, const Vec3& co_pol_direction,
                                               const FieldOptions& opt = {});

}  // namespace specklebem::post
