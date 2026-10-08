#pragma once
/// @file fields.hpp
/// Evaluation of scattered / total fields from the solved surface currents.
///
/// Near field: direct Stratton-Chu evaluation (O(N * N_points)), accelerated
/// through the MLFMM tree for many observation points (Phase 5).
/// Far field: radiation integral in direction k_hat (O(N * N_dirs)).
///
/// Representation (docs/03; derivation in src/postprocessing/fields.cpp). With
///   L_i X = jw mu_i int G_i X dS' - (1 / jw eps_i) grad int G_i div' X dS',
///   K_i X = int grad' G_i x X dS'   (source-point gradient, grad' G = -kernels::grad_green),
/// the field in region i radiated by the currents (J_i, M_i) of that region
/// (J_1 = J, M_1 = M; J_2 = -J, M_2 = -M, docs/06) is
///   E_i = -L_i J_i + K_i M_i ,   H_i = -K_i J_i - (1 / eta_i^2) L_i M_i ,
/// i.e. R1: E = -L_1 J + K_1 M, H = -K_1 J - L_1 M / eta_1^2 (scattered field) and
///      R2: E = +L_2 J - K_2 M, H = +K_2 J + L_2 M / eta_2^2 (total field; the incident field
///      exists only in R1).
///
/// Region of an observation point: generalised winding number w(r) = (1 / 4 pi) sum_t
/// Omega_t(r), with Omega_t the solid angle of triangle t seen from r, signed positive when r
/// lies behind t (on the R2 side, opposite to the outward normal). For a closed mesh w = 1 in
/// R2 and w = 0 in R1; a point is in R2 iff w >= 0.5. For an open mesh (rough-surface patch)
/// w lies in (0, 1/2) on the R2 side of the patch and in (-1/2, 0) on the R1 side, so the
/// threshold is w > 0 there. This is meaningful above / below the patch; laterally beyond the
/// patch edge w is small and its sign only says on which side of the (extended) patch the point
/// lies, and in or near the plane of the patch w -> 0. Points with |w| < 1e-6 on an open mesh
/// are therefore rejected (std::invalid_argument) instead of being classified silently.
///
/// Observation points closer to a triangle t than min_distance_factor * h_t (h_t its longest
/// edge, default factor 1e-3) are rejected with std::invalid_argument: the regular quadrature
/// used here is inaccurate there and near-surface evaluation needs singular quadrature
/// (Phase 5).
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/operator/assembler.hpp"

#include <cmath>

namespace specklebem::post {

struct SurfaceSolution {
    const op::Problem* problem = nullptr;
    VectorXc currents;  ///< [J; M]
    /// Angular frequency [rad/s] of the currents. 0 (default) => problem->excitation->omega();
    /// if both are given they must agree (relative 1e-12). Additive member (WP10): Problem
    /// carries no frequency of its own and the excitation may be absent.
    Real omega = 0.0;
};

/// Quadrature options of the field evaluation (additive API, WP10).
struct FieldOptions {
    /// Dunavant degree per source triangle; must be a positive-interior rule
    /// (kernels::triangle_rule_is_positive_interior), e.g. 1, 2, 4, 5, 6, 8, 9, 10.
    int quad_degree = 6;
    /// Points closer than min_distance_factor * (longest edge of the nearest triangle) to the
    /// surface are rejected.
    Real min_distance_factor = 1e-3;
};

namespace detail {

/// Scalar kernel factors of the near-field quadrature (exposed for unit tests): for
/// k = k_re + j k_im (Im k <= 0) and distance R > 0,
///   G = exp(-jkR) / (4 pi R) = e^{k_im R} (cos(k_re R) - j sin(k_re R)) / (4 pi R),
///   f = -(1 + jkR) G / R^2, so that grad_r G = f (r - r') (= kernels::grad_green).
struct GreenFactors {
    Real g_re, g_im;  ///< G
    Real f_re, f_im;  ///< f
};
[[nodiscard]] inline GreenFactors green_factors(Real k_re, Real k_im, Real R) {
    const Real amp = std::exp(k_im * R) / (4.0 * constants::pi * R);
    const Real g_re = amp * std::cos(k_re * R);
    const Real g_im = -amp * std::sin(k_re * R);
    const Real t_re = k_im * R - 1.0;  // -(1 + jkR) = (k_im R - 1) - j k_re R
    const Real t_im = -k_re * R;
    const Real R2 = R * R;
    return {g_re, g_im, (t_re * g_re - t_im * g_im) / R2, (t_re * g_im + t_im * g_re) / R2};
}

}  // namespace detail

/// Generalised winding number of the triangle mesh at r (see the file comment): 1 inside and
/// 0 outside a closed outward-oriented mesh. Not defined on the surface itself (the solid angle
/// of a triangle containing r is taken as 0 there).
[[nodiscard]] Real winding_number(const geometry::TriangleMesh& mesh, const Vec3& r);

/// Region (1 = R1 exterior, 2 = R2 object) of each observation point (rows of `points`), by the
/// winding-number rule above. @throws std::invalid_argument for non-finite points, a
/// non-positive min_distance_factor, a point closer than min_distance_factor * h_t to some
/// triangle t of the mesh, or (open mesh) a point with |w| < 1e-6.
[[nodiscard]] Eigen::Matrix<int, Eigen::Dynamic, 1> observation_regions(
    const geometry::TriangleMesh& mesh, const Vertices& points, Real min_distance_factor = 1e-3);

/// Scattered E and H in region R1 (z < surface) or R2 at arbitrary points.
/// In R2 the "scattered" field is the total field there (no incident field in R2).
/// @throws std::invalid_argument for a null problem or space, currents.size() != 2N, an
///         unknown frequency (omega <= 0 and no excitation) or points too close to the surface.
void scattered_field(const SurfaceSolution& s, const Vertices& points,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H);
void scattered_field(const SurfaceSolution& s, const Vertices& points,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H, const FieldOptions& opt);

/// Total field = incident + scattered (in R1 only).
/// Points in R2 receive the R2 field (scattered_field), points in R1 additionally the incident
/// field of problem->excitation. @throws std::invalid_argument if problem->excitation is null
/// (and everything scattered_field throws for).
void total_field(const SurfaceSolution& s, const Vertices& points,
                 Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                 Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H);
void total_field(const SurfaceSolution& s, const Vertices& points,
                 Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                 Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H, const FieldOptions& opt);

/// Far-field amplitude F(k_hat) such that E_s ~ F exp(-jkr)/r.
/// Radiated by the exterior currents in R1:
///   F(k_hat) = (j k1 / 4 pi) k_hat x int exp(j k1 k_hat . r') [eta1 (k_hat x J) + M] dS'.
/// The rows of `directions` are normalised internally (zero or non-finite rows throw).
void far_field(const SurfaceSolution& s, const Vertices& directions,
               Eigen::Matrix<Complex, Eigen::Dynamic, 3>& F);
void far_field(const SurfaceSolution& s, const Vertices& directions,
               Eigen::Matrix<Complex, Eigen::Dynamic, 3>& F, const FieldOptions& opt);

/// Regular grids used for the figures of the paper (xz-plane, xy-plane, cylinder).
///
/// plane_grid: nu * nv points origin + (i / (nu - 1)) u + (j / (nv - 1)) v, i = 0..nu-1,
/// j = 0..nv-1 (u, v are the full extents; a count of 1 gives the offset 0). Row-major with i
/// outer: row = i * nv + j. @throws std::invalid_argument for nu < 1 or nv < 1.
Vertices plane_grid(const Vec3& origin, const Vec3& u, const Vec3& v, int nu, int nv);
/// cylinder_grid: cylinder of the given radius around the y-axis; theta is the scattering angle
/// theta_s of docs/06, measured from -z in the xz-plane (towards +x), so that the point is
/// (r sin theta, y, -r cos theta). theta_k = theta_min + k (theta_max - theta_min) / (n_theta - 1),
/// y_l = y_min + l (y_max - y_min) / (n_y - 1) (a count of 1 gives theta_min / y_min).
/// Row = k * n_y + l (theta outer, y inner). @throws std::invalid_argument for n_theta < 1,
/// n_y < 1 or a non-positive / non-finite radius.
Vertices cylinder_grid(Real radius, Real theta_min, Real theta_max, int n_theta, Real y_min,
                       Real y_max, int n_y);

}  // namespace specklebem::post
