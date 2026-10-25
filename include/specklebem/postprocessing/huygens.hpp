#pragma once
/// @file huygens.hpp
/// SPP-aware far field of the PML route by a Huygens plane in R1 (ADR 0010, WP-PML6).
///
/// The windowed current far field (post::far_field with a CurrentWindow, ADR 0009 decision 5)
/// truncates the bound surface-plasmon currents that rough Ag launches into the flat gap; the
/// radiation integral of a truncated bound wave does not converge with the window. Instead, the
/// scattered field (E, H) is evaluated on a plane z = z_p in R1, at the distance z_h from the
/// surface, where the bound field has decayed like exp(-kappa z_h) (kappa = sqrt(k_spp^2 - k^2)
/// ~ 0.34 k for Ag at 500 nm: e^-2 at ~0.94 lambda), and the plane radiates (Love's equivalence
/// principle) the equivalent currents
///   J_H = n_hat x H,   M_H = -n_hat x E,
/// with n_hat the unit normal of the plane pointing into the half space of the observation
/// directions (away from the surface; on the PML route R1 is below the surface, n_hat = -z_hat).
/// This is the convention of docs/06 (J = n x H, M = -n x E with n pointing out of the source
/// region R2 into the observation region R1) with the half space beyond the plane as R1: the
/// equivalent currents on the infinite plane reproduce the scattered field beyond it and zero on
/// the surface side. The far field of the windowed plane currents is the radiation integral of
/// post::far_field (same exp(+j w t) convention, E_s ~ F exp(-j k_1 r) / r):
///   F(k_hat) = (j k_1 / 4 pi) k_hat x sum_p w_p exp(j k_1 k_hat . r_p) [eta_1 (k_hat x J_p) +
///   M_p],
/// valid for directions with n_hat . k_hat >= 0 (the half space beyond the plane). The tests
/// (tests/unit/test_huygens.cpp) check the sign convention on the analytic field of electric
/// and magnetic dipoles (the plane radiation reproduces the dipole far field) and against the
/// windowed-current far field on flat Ag and Si (docs/05 row 7f).
///
/// Plane grid: uniform, anchored at x = y = 0 (nodes (i h, j h), all integers i, j with
/// |i h|, |j h| < outer_half), weights w_p = h^2 w(max(|x_p|, |y_p|)) with the raised-cosine
/// window w of CurrentWindow (points with w = 0 dropped): the trapezoid rule of the windowed
/// integrand (zero beyond outer_half, C^1 at both window ends). Anchoring makes the grid of a
/// smaller window a subset of a larger one with the same spacing (huygens_reweight).
///
/// Cost: the plane fields are a direct O(N_points x N_triangles) Stratton-Chu evaluation
/// (post::exterior_scattered_field, OpenMP over the points), the radiation O(N_points x
/// N_directions). The compressed operator of the fast path (MLFMM + PML ring) is not used.
#include "specklebem/postprocessing/fields.hpp"

namespace specklebem::post {

/// Largest number of plane points huygens_plane() builds (the plane fields cost 96 bytes and
/// N_triangles x N_quad kernel evaluations per point).
inline constexpr Index kMaxHuygensPoints = 10'000'000;

/// Quadrature grid of a Huygens plane.
struct HuygensPlane {
    Real z = 0;         ///< height of the plane [m]
    Real normal_z = 0;  ///< n_hat = normal_z z_hat (+1 or -1): towards the observation side
    Real spacing = 0;   ///< grid spacing h [m]
    CurrentWindow window{};
    Vertices points;   ///< nodes with a non-zero window weight, x outer, y inner
    VectorXr weights;  ///< h^2 w(max(|x|, |y|)) [m^2]
};

/// The anchored plane grid (see the file comment) at height z with normal normal_z z_hat.
/// @throws std::invalid_argument for a non-finite z, normal_z not exactly +1 or -1, an invalid
///         window (current_window_value()), a spacing not finite and > 0, or more than
///         kMaxHuygensPoints nodes.
[[nodiscard]] HuygensPlane huygens_plane(Real z, Real normal_z, const CurrentWindow& window,
                                         Real spacing);

/// The plane of a solved open surface for FieldOptions::huygens (empty: default HuygensOptions):
/// R1 side from the orientation of the mesh (normal_z = sign of sum_t A_t n_z,t; R1 lies on the
/// side the normals point to), z = (min vertex z) - z_h for normal_z = -1 or (max vertex z) +
/// z_h for +1, spacing and window as HuygensOptions (defaults lambda_1 = 2 pi / Re k_1, lambda_1
/// / 4 and FieldOptions::window).
/// @throws std::invalid_argument for a null problem or space, a closed mesh, a mesh that does
///         not face +-z (|sum A n_z| < 0.5 sum A), a non-propagating exterior (Re k_1 <= 0),
///         z_h not finite and > 0, a spacing not finite, > 0 and <= lambda_1 / 2, no window
///         (neither HuygensOptions::window nor FieldOptions::window), and with a PML stretch a
///         window ending beyond the stretch start; and everything huygens_plane() throws for.
[[nodiscard]] HuygensPlane huygens_plane(const SurfaceSolution& s, const FieldOptions& opt);

/// Scattered field of a solution on a Huygens plane (reusable for several radiations).
struct HuygensPlaneField {
    HuygensPlane plane;
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> E;  ///< [V/m] per plane point
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> H;  ///< [A/m]
    Complex k{};       ///< wavenumber of the medium on the plane (R1) [1/m]
    Complex eta{};     ///< its wave impedance [Ohm]
    Real seconds = 0;  ///< wall time of the plane-field evaluation [s]
};

/// huygens_plane(s, opt) and the R1 scattered field on it (exterior_scattered_field with
/// opt.quad_degree and opt.min_distance_factor; the PML sources radiate through the stretched
/// Green's function unless HuygensOptions::pml_sources is false). @throws as huygens_plane() and
/// exterior_scattered_field().
[[nodiscard]] HuygensPlaneField huygens_plane_field(const SurfaceSolution& s,
                                                    const FieldOptions& opt);

/// The same plane field with another window of the same grid: the points of f whose weight
/// under `window` is non-zero, with the weights h^2 w(t) of the new window (bitwise the plane
/// huygens_plane(z, normal_z, window, h) and its fields). @throws std::invalid_argument for an
/// invalid window or one that ends beyond f's window (the larger grid was not evaluated).
[[nodiscard]] HuygensPlaneField huygens_reweight(const HuygensPlaneField& f,
                                                 const CurrentWindow& window);

/// Far-field amplitude F [V] of the plane's equivalent currents J_H = n_hat x H,
/// M_H = -n_hat x E (file comment) for E, H sampled on `plane` in a medium (k, eta), per row of
/// `directions` (normalised internally). Each direction is summed over the points in a fixed
/// order (independent of the number of threads).
/// @throws std::invalid_argument for E or H not of plane.points.rows() rows, a non-finite or
///         zero direction, or a direction with n_hat . k_hat < -1e-12 (the surface side of the
///         plane has no far field of the plane currents).
void huygens_radiate(const HuygensPlane& plane, const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                     const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& H, Complex k, Complex eta,
                     const Vertices& directions, Eigen::Matrix<Complex, Eigen::Dynamic, 3>& F);
/// Same for a HuygensPlaneField.
void huygens_radiate(const HuygensPlaneField& f, const Vertices& directions,
                     Eigen::Matrix<Complex, Eigen::Dynamic, 3>& F);

/// Huygens-plane far field of a solution: huygens_radiate(huygens_plane_field(s, opt), ...).
/// post::far_field(s, directions, F, opt) with opt.huygens set calls this.
/// @throws as huygens_plane_field() and huygens_radiate().
void huygens_far_field(const SurfaceSolution& s, const Vertices& directions,
                       Eigen::Matrix<Complex, Eigen::Dynamic, 3>& F, const FieldOptions& opt);

}  // namespace specklebem::post
