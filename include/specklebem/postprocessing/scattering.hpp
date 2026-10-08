#pragma once
/// @file scattering.hpp
/// Scattering observables.
#include "specklebem/postprocessing/fields.hpp"

namespace specklebem::post {

/// Bistatic radar cross section sigma(theta) in a given scattering plane
/// (used for the Mie validation, Eq. 7 of the paper for the error metric).
VectorXr bistatic_rcs(const SurfaceSolution& s, const Vec3& plane_normal,
                      const VectorXr& angles_rad);

/// Differential reflection coefficient from fields on a cylindrical surface,
/// averaged along the cylinder axis (Fig. 9).
VectorXr differential_reflection_coefficient(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E_cyl,
                                             int n_theta, int n_y);

/// Decomposition into co- and cross-polarised intensities w.r.t. the incident polarisation.
struct PolarizedIntensity {
    VectorXr co, cross;
};
PolarizedIntensity polarized_intensity(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                                       const Vec3& co_pol_direction);

}  // namespace specklebem::post
