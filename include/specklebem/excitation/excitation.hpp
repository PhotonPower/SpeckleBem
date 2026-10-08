#pragma once
/// @file excitation.hpp
/// Incident fields. Every excitation evaluates E_inc and H_inc at arbitrary
/// points of region R1 (needed for the right-hand side and for total-field
/// post-processing).
#include "specklebem/core/types.hpp"
#include "specklebem/material/material.hpp"

namespace specklebem::excitation {

/// p: E in the plane of incidence (xz-plane); s: E along y.
enum class Polarization { P, S };

class Excitation {
public:
    virtual ~Excitation() = default;
    [[nodiscard]] virtual Vec3c electric_field(const Vec3& r) const = 0;
    [[nodiscard]] virtual Vec3c magnetic_field(const Vec3& r) const = 0;
    [[nodiscard]] Real omega() const { return omega_; }
    [[nodiscard]] Real wavelength() const;
    [[nodiscard]] const material::Material& background() const { return background_; }

protected:
    Excitation(Real omega, material::Material background)
        : omega_(omega), background_(background) {}
    Real omega_;
    material::Material background_;
};

/// Plane wave with unit propagation direction and complex polarization vector.
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
};

/// Gaussian beam propagating along +z (waist at the focus), used to illuminate
/// rough surfaces without touching the patch edges. Phase 2 uses the paraxial
/// model; a rigorous angular-spectrum representation that satisfies Maxwell's
/// equations exactly is scheduled for Phase 5 (docs/01_project_plan.md).
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
    Params p_;
};

}  // namespace specklebem::excitation
