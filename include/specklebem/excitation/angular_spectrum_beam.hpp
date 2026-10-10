#pragma once
/// @file angular_spectrum_beam.hpp
/// Rigorous Gaussian beam as a finite angular spectrum of exact plane waves (WP-E1, ADR 0006
/// amendment item 4). Every plane wave solves Maxwell's equations in the background, so the
/// beam does exactly (the paraxial GaussianBeam violates them at O((lambda/(pi w0))^2)).
#include "specklebem/core/types.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/material/material.hpp"

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

namespace specklebem::excitation {

/// Gaussian beam built from propagating plane waves (exp(+jwt) convention).
///
/// Beam frame (as GaussianBeam): central direction k0_hat = R_y(theta_in) z_hat with
/// |theta_in| < pi/2, transverse unit vectors u_hat = R_y(theta_in) x_hat and v_hat = y_hat;
/// e0_hat = u_hat (p) or y_hat (s).
///
/// Definition: with k = k0 n1 of the lossless background R1 and polar angle alpha around k0_hat,
///   k_hat(alpha, phi) = sin(alpha) (cos(phi) u_hat + sin(phi) v_hat) + cos(alpha) k0_hat,
///   E(r) = C sum_i W_i A_i p_i exp(-j k k_hat_i . (r - focus)),
///   H(r) = C sum_i W_i A_i (k_hat_i x p_i) / eta exp(-j k k_hat_i . (r - focus)),
/// which discretises the angular-spectrum integral over the transverse wavevector
/// k_t = k sin(alpha) (u, v components, relative to k0_hat):
///   A(k_t) = exp(-k_t^2 w0^2 / 4)  (Fourier transform of exp(-rho^2/w0^2)),
///   p = e0_hat - (k_hat . e0_hat) k_hat  (projection onto the plane normal to k_hat, not
///       renormalised: the e0_hat component of the spectrum is A (1 - (k_hat . e0_hat)^2),
///       the same construction as the WP-V1 study beam),
///   d^2 k_t = k^2 sin(alpha) cos(alpha) d alpha d phi -> W_i = Gauss-Legendre weight in alpha
///       on [0, alpha_max] x trapezoid weight 2 pi / n_phi in phi, times k^2 sin cos.
/// C normalises E(focus) . e0_hat = 1 V/m (E(focus) is parallel to e0_hat by symmetry).
/// Evanescent components (k_t > k) are omitted: they grow like exp(|k_z| |zeta|) towards the
/// source side (zeta < 0, i.e. z < 0 at normal incidence), so they cannot describe a beam over a
/// region extending to both sides of the focus; only directions with k_hat . k0_hat > 0 are used
/// (alpha_max <= pi/2, Gauss-Legendre nodes are interior). alpha_max truncates the spectrum
/// where A < 1e-3 tolerance (or at k_t = k). For k w0 small the k_t = k cut-off is visible
/// (logged): the beam is still exact, its focal profile just not exactly Gaussian.
/// The cone is around k0_hat, not z_hat: for |theta_in| + alpha_max > pi/2 some (weak, outer)
/// components travel towards -z (logged once at construction, INFO); the beam as a whole still
/// travels towards +z.
///
/// Quadrature: n_alpha Gauss-Legendre nodes x n_phi azimuths (n_phi a multiple of 4, so the
/// grid is symmetric about u and v). "Change" of a grid = max over probe points of
/// |E_grid - E_finer| / max |E_finer|, where the finer grid has n -> 1.5 n + 8 in the direction(s)
/// considered (the WP-V1 study criterion). Automatic mode starts below the phase-variation
/// estimate (n_alpha ~ k R alpha_max / 4, n_phi ~ k R sin(alpha_max) / 2) and raises each order
/// by ~25 % while its own change is >= tolerance / 2; once both are below, the grid is accepted if
/// its change against the grid refined in both directions is < tolerance (grid_change()).
/// Probe points: the focus and two shells (radius R/2 and R, 32 Fibonacci directions each, in the
/// beam frame) of the ball |r - focus| <= R = region_radius. A finite plane-wave sum is not
/// controlled far outside that ball (the trapezoid rule in phi aliases beyond transverse
/// distances ~ n_phi / (k sin(alpha_max))), so R must cover every point where fields are
/// evaluated (surface quadrature points, observation points of total near fields). The check is
/// an estimate on probe points, not a bound. controlled_radius() returns R and
/// controlled_center() the focus, so Simulation rejects meshes that leave the ball; a field
/// evaluated at |r - focus| > 1.2 R logs one warning per beam object (SBEM_WARN).
/// A fixed grid whose grid_change() >= tolerance is accepted with a warning.
///
/// Fields are evaluated as one pass over the stored arrays (branch-free sincos, four
/// independent partial sums); the plane-wave grid is immutable after construction and field
/// evaluation is thread-safe (the only mutable state is the atomic flag of the outside-the-ball
/// warning, which a copy resets).
///
/// Throws std::invalid_argument for wavelength or w0 <= 0, a non-finite focus,
/// |theta_in| >= pi/2, a lossy background, tolerance outside (0, 1e-2], region_radius < 0,
/// only one fixed order, a fixed order < 1 or > kMaxOrder, max_plane_waves < 1, or a fixed grid
/// whose check grid (both orders refined 1.5x) exceeds max_plane_waves; std::runtime_error when
/// the automatic refinement needs an order above kMaxOrder or a check grid above
/// max_plane_waves (or 40 rounds).
class AngularSpectrumBeam final : public Excitation {
public:
    /// Largest polar or azimuthal order (fixed or automatic); keeps 1.5 n + 8 far from INT_MAX.
    static constexpr int kMaxOrder = 1'000'000;

    struct Params {
        Real wavelength = 500e-9;  ///< vacuum wavelength [m]
        Real waist_radius = 3e-6;  ///< w0 of the spectrum exp(-k_t^2 w0^2 / 4) [m]
        Vec3 focus = Vec3::Zero();
        Real incidence_angle = 0.0;  ///< theta_in [rad], rotation about the y-axis
        Polarization polarization = Polarization::P;
        Real tolerance = 1e-10;  ///< grid check (relative change of E on the probes)
        Real region_radius = 0;  ///< R [m] of the controlled ball; 0 selects 4 w0
        int polar_order = 0;     ///< fixed n_alpha (with azimuth_order); 0 = automatic
        int azimuth_order = 0;   ///< fixed n_phi (rounded up to a multiple of 4); 0 = automatic
        Index max_plane_waves = 2'000'000;  ///< cap of the check grids (automatic and fixed)
    };

    AngularSpectrumBeam(Params p, material::Material background = material::vacuum());

    [[nodiscard]] Vec3c electric_field(const Vec3& r) const override;
    [[nodiscard]] Vec3c magnetic_field(const Vec3& r) const override;
    /// E and H at r in one pass (one sincos per plane wave instead of two calls' worth).
    [[nodiscard]] std::pair<Vec3c, Vec3c> fields(const Vec3& r) const override;
    /// region_radius(): the fields are controlled for |r - focus| <= R.
    [[nodiscard]] Real controlled_radius() const override { return region_radius_; }
    /// The focus.
    [[nodiscard]] Vec3 controlled_center() const override { return p_.focus; }

    [[nodiscard]] const Params& params() const { return p_; }
    [[nodiscard]] Index num_plane_waves() const { return grid_.n; }
    [[nodiscard]] int polar_order() const { return grid_.n_alpha; }
    [[nodiscard]] int azimuth_order() const { return grid_.n_phi; }
    /// Relative change of E on the probes against the 1.5x finer grid (automatic and fixed).
    [[nodiscard]] Real grid_change() const { return grid_change_; }
    [[nodiscard]] Real max_polar_angle() const { return alpha_max_; }
    [[nodiscard]] Real region_radius() const { return region_radius_; }
    /// Time-averaged power through any plane normal to k0_hat [W] (Parseval over the spectrum:
    /// (2 pi)^2 / (2 eta) int |E~|^2 cos(alpha) d^2 k_t, same quadrature). Paraxial limit:
    /// pi w0^2 / (4 eta) for 1 V/m.
    [[nodiscard]] Real power() const { return grid_.power; }
    /// Unit propagation directions k_hat_i of the plane waves.
    [[nodiscard]] std::vector<Vec3> plane_wave_directions() const;
    /// Real E amplitudes C W_i A_i p_i [V/m] (phase reference: the focus); H_i = k_hat_i x E_i /
    /// eta.
    [[nodiscard]] std::vector<Vec3> plane_wave_amplitudes() const;

private:
    /// Plane-wave grid in structure-of-arrays form (immutable after construction, so concurrent
    /// field evaluation is thread-safe), padded with zero amplitudes to a multiple of the
    /// accumulation width.
    struct Grid {
        std::vector<Real> kx, ky, kz;  ///< k k_hat_i [1/m]
        std::vector<Real> ex, ey, ez;  ///< E amplitudes [V/m]
        std::vector<Real> hx, hy, hz;  ///< H amplitudes [A/m]
        Index n = 0;                   ///< plane waves without padding
        int n_alpha = 0;
        int n_phi = 0;
        Real power = 0;
    };
    [[nodiscard]] Grid build(int n_alpha, int n_phi) const;
    /// Sums the M amplitude arrays `amp` of `g` with exp(-j k k_hat_i . (r - focus)).
    template <std::size_t M>
    void sum_waves(const Grid& g, const Vec3& r, const Real* const (&amp)[M],
                   Complex (&out)[M]) const;
    [[nodiscard]] Vec3c probe_field(const Grid& g, const Vec3& r) const;

    Params p_;
    Vec3 k0_hat_, u_hat_, e0_hat_;
    Real k_ = 0;
    Real eta_ = 0;
    Real alpha_max_ = 0;
    Real region_radius_ = 0;
    Grid grid_;
    Real grid_change_ = 0;

    /// One-time flag of the outside-the-ball warning. Copying (and moving, which copies) gives
    /// a fresh flag, so the beam stays copyable although std::atomic is not.
    struct WarnOnce {
        mutable std::atomic<bool> fired{false};
        WarnOnce() = default;
        WarnOnce(const WarnOnce& /*other*/) {}
        WarnOnce& operator=(const WarnOnce& /*other*/) { return *this; }
    };
    /// Logs the outside-the-ball warning on the first call (thread-safe).
    void warn_outside(Real distance) const;
    Real warn_radius_ = 0;  ///< 1.2 region_radius_
    WarnOnce outside_warned_;
};

}  // namespace specklebem::excitation
