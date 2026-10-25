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
#include <optional>

namespace specklebem::post {

/// Solved surface currents of a problem. The frequency lives in op::Problem (Problem::omega).
struct SurfaceSolution {
    const op::Problem* problem = nullptr;
    VectorXc currents;  ///< [J; M]
};

/// Smooth window on the surface currents of far_field() (ADR 0009 decision 5, WP-PML3a): the
/// open PML surface radiates only from its real part (patch, taper, gap), and the window takes
/// the currents smoothly to zero across the flat gap so that the truncation at the PML start does
/// not diffract. It acts on t = max(|x|, |y|) of each quadrature point:
///   w(t) = 1                                                     for t <= inner_half,
///   w(t) = 1/2 (1 + cos(pi (t - inner_half) / (outer_half - inner_half)))   in between,
///   w(t) = 0                                                     for t >= outer_half,
/// which is C^1 (value and slope continuous at both ends). Simulation's PML route uses
/// inner_half = OpenSurfaceMesh::taper_end and outer_half = OpenSurfaceMesh::pml_start.
struct CurrentWindow {
    Real inner_half{};  ///< >= 0 [m]: w = 1 up to here
    Real outer_half{};  ///< > inner_half [m]: w = 0 from here on
};

/// w(t) of the window (see CurrentWindow), t = max(|x|, |y|) of a surface point (t < 0 is
/// treated as |t|). @throws std::invalid_argument for a non-finite t, inner_half or outer_half,
/// inner_half < 0 or outer_half <= inner_half.
[[nodiscard]] Real current_window_value(const CurrentWindow& w, Real t);

/// Huygens-plane far field of the PML route (ADR 0010, WP-PML6; post::huygens_far_field in
/// postprocessing/huygens.hpp). The scattered E and H are evaluated on a plane in R1 at the
/// distance z_h from the surface and radiated as the equivalent currents of the plane, windowed
/// with `window` in the plane. The bound surface-plasmon field decays with the distance from the
/// surface, so it is filtered out by z_h instead of being truncated by the current window.
struct HuygensOptions {
    /// Distance [m] of the plane from the extreme mesh vertex on the R1 side (the plane lies at
    /// z = min vertex z - z_h when R1 is below the surface, as on the PML route); empty: one
    /// wavelength of R1, lambda_1 = 2 pi / Re k_1 (vacuum / air on the PML route: the free-space
    /// wavelength). Must be finite and > 0.
    std::optional<Real> z_h;
    /// Grid spacing [m] of the plane (uniform grid anchored at x = y = 0, nodes i h);
    /// empty: lambda_1 / 4. Must be finite, > 0 and <= lambda_1 / 2 (the radiation integrand has
    /// the bandwidth 2 k_1 for propagating plane fields).
    std::optional<Real> spacing;
    /// Window on the plane (same semantics as the current window, t = max(|x|, |y|) of a plane
    /// point; the grid extends to its outer_half); empty: FieldOptions::window (the currents'
    /// window; Simulation's PML route: taper_end to pml_start). With a PML stretch it must end at
    /// or before the stretch start (plane points over the PML would need stretched observation
    /// points).
    std::optional<CurrentWindow> window;
    /// Diagnostic (WP-PML6): false drops the source triangles in the PML from the plane fields
    /// (exterior_scattered_field with pml_sources = false), i.e. truncates the representation
    /// at the stretch start. The default true is the analytic continuation over the unbounded
    /// surface (benchmarks/results/pml_huygens.md compares both).
    bool pml_sources = true;
};

/// Quadrature options of the field evaluation (additive API, WP10).
struct FieldOptions {
    /// Dunavant degree per source triangle; must be a positive-interior rule
    /// (kernels::triangle_rule_is_positive_interior), e.g. 1, 2, 4, 5, 6, 8, 9, 10.
    int quad_degree = 6;
    /// Points closer than min_distance_factor * (longest edge of the nearest triangle) to the
    /// surface are rejected.
    Real min_distance_factor = 1e-3;
    /// far_field() only: multiply the currents at each quadrature point by the window value
    /// (points with w = 0 drop out). Required with a non-identity PML stretch (outer_half <= the
    /// stretch start on x and y); optional otherwise. Empty: no window, every result bitwise as
    /// before WP-PML3a. scattered_field() / total_field() reject a window (a windowed current
    /// has no physical near field).
    std::optional<CurrentWindow> window;
    /// far_field() only (WP-PML6, ADR 0010): radiate from a Huygens plane in R1 instead of the
    /// surface currents (post::huygens_far_field with these options; the plane fields use
    /// quad_degree and min_distance_factor, `window` is the plane window's default). Empty: the
    /// surface currents, every result bitwise as before WP-PML6. The near-field functions reject
    /// it.
    std::optional<HuygensOptions> huygens;
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

/// Principal-branch distance R~ = sqrt((r - r~') . (r - r~')) (unconjugated) from the real
/// observation point r to the stretched source point r~' = S(r') of exterior_scattered_field
/// (exposed for unit tests). For r in the real zone (|x| <= x_start, |y| <= y_start) and a source
/// point of the monotone stretch, Im R~ <= 0 (each axis gives (r_i - r_i') (+-Im x~_i') with the
/// sign of -(r_i - r_i')), so exp(-j k R~) / R~ decays into the PML, as in the stretched kernels
/// (kernels/stretched_operators.cpp). The observation-point rule of exterior_scattered_field is
/// what guarantees it; this check catches a relaxation of that rule.
/// @throws std::logic_error naming `sample`, r and R~ if Im R~ > 1e-12 |R~| (internal invariant
///         violated: a growing kernel, e.g. an observation point beyond the stretch start on the
///         source's side).
[[nodiscard]] Complex stretched_distance(const Vec3& r, const Vec3c& source, Index sample);

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
/// @throws std::invalid_argument for a null problem or space, currents.size() != 2N, a
///         Problem::omega that is not finite and > 0 or differs from the excitation's omega()
///         (relative 1e-12, as op::validate), a FieldOptions::window (far_field() only), a
///         non-identity PML stretch (Problem::stretch: near fields of the PML route would need
///         the analytically continued currents and the stretched Green's function for sources
///         in the PML; out of scope, ADR 0009 decision 5 covers the far field only), or points
///         too close to the surface.
void scattered_field(const SurfaceSolution& s, const Vertices& points,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H);
void scattered_field(const SurfaceSolution& s, const Vertices& points,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H, const FieldOptions& opt);

/// Scattered E and H at points of region R1 only (WP-PML6), also for a problem with a PML
/// stretch: the real source triangles radiate as in scattered_field() (bitwise its R1 rows when
/// there is no stretch), the stretched ones with the stretched Green's function of the real
/// observation point r and the mapped source point r~' = S(r') (geometry::CoordinateStretch,
/// ADR 0009 decision 2): G(R~) = exp(-j k_1 R~) / (4 pi R~), R~^2 = (r - r~') . (r - r~')
/// unconjugated on the principal branch (Re R~ >= 0, Im R~ <= 0), grad G = -(1 + j k_1 R~) G
/// (r - r~') / R~^2, and the Piola-transformed currents DS J, DS M with the plain divergences
/// (f~ dS~ = DS f dS, (div~ f~) dS~ = (div f) dS). This is the analytic continuation of the
/// representation over the infinite open surface (the deformed source contour passes no branch
/// point of R~ while |x| <= x_start and |y| <= y_start of the observation point), so it is the
/// scattered field of the unbounded surface in the real zone of R1. pml_sources = false
/// (diagnostic) drops the stretched source triangles (the representation truncated at the
/// stretch start; the observation-point rule stays).
/// @throws std::invalid_argument as scattered_field() (except for the stretch), for a
///         FieldOptions::window or ::huygens, for a point not in R1 (observation_regions()), and,
///         with a stretch, for a point with |x| > x_start or |y| > y_start.
void exterior_scattered_field(const SurfaceSolution& s, const Vertices& points,
                              Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                              Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H,
                              const FieldOptions& opt = {}, bool pml_sources = true);

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
/// With FieldOptions::window the currents are weighted by current_window_value() at each
/// quadrature point (the radiation integral over the windowed real part of an open surface).
/// A non-identity PML stretch (Problem::stretch, ADR 0009 decision 5) requires a window with
/// outer_half <= the stretch start of both axes, so that no PML current (whose radiation
/// integral grows like exp(+k Im x~)) contributes. Without a window every result is bitwise
/// unchanged. With FieldOptions::huygens the far field is post::huygens_far_field() instead
/// (radiation of the windowed equivalent currents of a plane in R1, ADR 0010).
/// @throws std::invalid_argument additionally for an invalid window
/// (current_window_value()), a stretch without a window or a window reaching past the stretch
/// start, and (Huygens) everything huygens_far_field() throws for.
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
