#pragma once
/// @file fields.hpp
/// Evaluation of scattered / total fields from the solved surface currents.
///
/// Near field: direct Stratton-Chu evaluation (O(N * N_points)), accelerated
/// through the MLFMM tree for many observation points (Phase 5).
/// Far field: radiation integral in direction k_hat (O(N * N_dirs)).
#include "specklebem/core/types.hpp"
#include "specklebem/operator/assembler.hpp"

namespace specklebem::post {

struct SurfaceSolution {
    const op::Problem* problem = nullptr;
    VectorXc currents;  ///< [J; M]
};

/// Scattered E and H in region R1 (z < surface) or R2 at arbitrary points.
void scattered_field(const SurfaceSolution& s, const Vertices& points,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H);

/// Total field = incident + scattered (in R1 only).
void total_field(const SurfaceSolution& s, const Vertices& points,
                 Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                 Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H);

/// Far-field amplitude F(k_hat) such that E_s ~ F exp(-jkr)/r.
void far_field(const SurfaceSolution& s, const Vertices& directions,
               Eigen::Matrix<Complex, Eigen::Dynamic, 3>& F);

/// Regular grids used for the figures of the paper (xz-plane, xy-plane, cylinder).
Vertices plane_grid(const Vec3& origin, const Vec3& u, const Vec3& v, int nu, int nv);
Vertices cylinder_grid(Real radius, Real theta_min, Real theta_max, int n_theta, Real y_min,
                       Real y_max, int n_y);

}  // namespace specklebem::post
