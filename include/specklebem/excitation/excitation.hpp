#pragma once
/// @file excitation.hpp
/// Incident fields. Every excitation evaluates E_inc and H_inc at arbitrary
/// points of region R1 (needed for the right-hand side and for total-field
/// post-processing).
#include "specklebem/core/types.hpp"
#include "specklebem/material/material.hpp"

#include <limits>
#include <utility>

namespace specklebem::excitation {

/// p: E in the plane of incidence (xz-plane); s: E along y.
enum class Polarization { P, S };

class Excitation {
public:
    virtual ~Excitation() = default;
    [[nodiscard]] virtual Vec3c electric_field(const Vec3& r) const = 0;
    [[nodiscard]] virtual Vec3c magnetic_field(const Vec3& r) const = 0;
    /// E and H at r. The default calls electric_field and magnetic_field (bitwise the same
    /// values); sources that share work between the two (AngularSpectrumBeam) override it.
    [[nodiscard]] virtual std::pair<Vec3c, Vec3c> fields(const Vec3& r) const {
        return {electric_field(r), magnetic_field(r)};
    }
    /// Radius [m] of the ball around controlled_center() in which the fields are controlled
    /// (AngularSpectrumBeam: its region_radius); +infinity for sources that are defined
    /// everywhere (PlaneWave, GaussianBeam). Simulation rejects meshes that leave the ball.
    [[nodiscard]] virtual Real controlled_radius() const {
        return std::numeric_limits<Real>::infinity();
    }
    /// Centre [m] of the controlled ball (AngularSpectrumBeam: its focus); the origin by default
    /// (irrelevant for an infinite controlled_radius()).
    [[nodiscard]] virtual Vec3 controlled_center() const { return Vec3::Zero(); }
    [[nodiscard]] Real omega() const { return omega_; }
    [[nodiscard]] Real wavelength() const;
    [[nodiscard]] const material::Material& background() const { return background_; }

protected:
    Excitation(Real omega, material::Material background)
        : omega_(omega), background_(background) {}
    Real omega_;
    material::Material background_;
};

/// Plane wave E(r) = e0 exp(-j k k_hat . r), H(r) = k_hat x E(r) / eta (exp(+jwt) convention).
/// Constructed from the vacuum wavelength; k and eta are those of the (lossless) background.
/// k_hat is normalised; e0 is complex (circular/elliptic polarisation) and must be transverse,
/// |e0 . k_hat| <= 1e-12 |e0|. Throws std::invalid_argument for a non-positive wavelength, a zero
/// k_hat or e0, a non-transverse e0 or a lossy background.
class PlaneWave final : public Excitation {
public:
    PlaneWave(Real wavelength, const Vec3& k_hat, const Vec3c& e0,
              material::Material background = material::vacuum());
    [[nodiscard]] Vec3c electric_field(const Vec3& r) const override;
    [[nodiscard]] Vec3c magnetic_field(const Vec3& r) const override;

private:
    Vec3 k_hat_;
    Vec3c e0_;
    Complex k_;
    Complex eta_;
};

/// Gaussian beam propagating along k_hat = R_y(theta_in) z_hat (waist at the focus), used to
/// illuminate rough surfaces without touching the patch edges. Paraxial model (qualitative
/// work); the rigorous angular-spectrum beam that satisfies Maxwell's equations exactly is
/// excitation::AngularSpectrumBeam (angular_spectrum_beam.hpp, ADR 0006 amendment item 4).
///
/// Beam frame: k_hat = R_y(theta_in) z_hat (rotation about y; theta_in = 0 gives +z) with
/// |theta_in| < pi/2, i.e. the beam always travels towards +z (docs/06); waist plane through
/// `focus`; axial coordinate zeta = k_hat . (r - focus), transverse distance rho.
/// p: e_hat = R_y(theta_in) x_hat (xz-plane, perpendicular to k_hat); s: e_hat = y_hat.
/// Fundamental paraxial beam (unit amplitude 1 V/m on the axis at the focus):
///   A = (w0/w) exp(-rho^2/w^2) exp(-j[k zeta + k rho^2/(2R) - psi]),
///   w = w0 sqrt(1 + (zeta/z_R)^2), R = zeta (1 + (z_R/zeta)^2), psi = atan(zeta/z_R),
///   z_R = pi w0^2 n / lambda (k = 2 pi n / lambda, real: the background must be lossless).
/// Fields: E = e_hat A + k_hat E_zeta, eta H = (k_hat x e_hat) A + k_hat H_zeta, with the
/// first-order longitudinal components E_zeta = -(j/k) e_hat . grad_t A and
/// H_zeta = -(j/k) (k_hat x e_hat) . grad_t A (Lax et al. 1975). The transverse fields are the
/// textbook paraxial beam with H_t = k_hat x E / eta; the longitudinal terms vanish on the axis
/// and make the curl equations hold to O((lambda/(pi w0))^2) instead of O(lambda/(pi w0)).
/// Every constructed beam logs one warning about the paraxial approximation.
/// Throws std::invalid_argument for w0 <= 0, wavelength <= 0, a non-finite focus,
/// |theta_in| >= pi/2 or a lossy background.
class GaussianBeam final : public Excitation {
public:
    struct Params {
        Real wavelength = 500e-9;
        Real waist_radius = 3e-6;  ///< 1/e^2 intensity radius w0 [m]
        Vec3 focus = Vec3::Zero();
        Real incidence_angle = 0.0;  ///< theta_in [rad], rotation about the y-axis
        Polarization polarization = Polarization::P;
    };
    GaussianBeam(Params p, material::Material background = material::vacuum());
    [[nodiscard]] Vec3c electric_field(const Vec3& r) const override;
    [[nodiscard]] Vec3c magnetic_field(const Vec3& r) const override;
    [[nodiscard]] const Params& params() const { return p_; }

private:
    struct Sample {
        Complex amplitude;  ///< scalar field A(r) including the carrier exp(-j k zeta)
        Complex grad_e;     ///< e_hat . grad_t A
        Complex grad_h;     ///< h_hat . grad_t A
    };
    [[nodiscard]] Sample sample(const Vec3& r) const;

    Params p_;
    // Per-beam constants (fields are evaluated per quadrature point during assembly).
    Vec3 k_hat_;  ///< propagation direction R_y(theta_in) z_hat
    Vec3 e_p_;    ///< p direction R_y(theta_in) x_hat (transverse coordinate u)
    Vec3 e_hat_;  ///< polarisation direction (e_p_ or y_hat)
    Vec3 h_hat_;  ///< k_hat_ x e_hat_
    Real k_ = 0;
    Real z_r_ = 0;
    Complex eta_;
};

}  // namespace specklebem::excitation
