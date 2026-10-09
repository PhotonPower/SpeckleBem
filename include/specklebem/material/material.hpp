#pragma once
/// @file material.hpp
/// Homogeneous isotropic materials with complex permittivity (exp(+jwt) convention).
#include "specklebem/core/types.hpp"

namespace specklebem::material {

struct Material {
    Complex eps_r{1.0, 0.0};  ///< relative permittivity
    Complex mu_r{1.0, 0.0};   ///< relative permeability (1 for optical materials)

    [[nodiscard]] Complex refractive_index() const;          ///< n = sqrt(eps_r mu_r)
    [[nodiscard]] Complex wave_impedance(Real omega) const;  ///< eta = sqrt(mu / eps)
    [[nodiscard]] Complex wavenumber(Real omega) const;      ///< k = omega sqrt(mu eps)
};

inline Material vacuum() {
    return {};
}

/// Reference values used in Fu et al. 2023 at lambda = 500 nm:
///   Ag (Johnson & Christy):  eps_r = -9.794 - j 0.313
///   Si (Aspnes & Studna):    eps_r = 18.478 - j 0.606
Material silver_500nm();
Material silicon_500nm();

/// Field decay length [m] of a plane wave in a homogeneous medium of relative
/// permittivity eps_r (mu_r = 1) at the vacuum wavelength lambda_0 [m]:
///   delta = lambda_0 / (2 pi |Im sqrt(eps_r)|),
/// the depth over which the field amplitude falls by 1/e (the intensity decays twice as
/// fast). At 500 nm: Si (eps_r = 18.478 - 0.606j) delta = 1.129 um, Ag delta = 25.4 nm.
/// Used to size the fine band of the rough-surface box (Simulation passes 3 delta as
/// geometry::RoughSurfaceParams::box_fine_depth; ADR 0006). Only |Im sqrt(eps_r)| enters,
/// so the sign convention of Im(eps_r) does not matter.
/// @returns +infinity for a lossless medium (Im sqrt(eps_r) = 0: the field does not decay).
/// @throws std::invalid_argument for non-finite eps_r or a non-positive / non-finite
///         wavelength.
[[nodiscard]] Real field_decay_length(Complex eps_r, Real wavelength);

/// Tabulated n,k data (e.g. refractiveindex.info) with interpolation.
class DispersiveMaterial {
public:
    DispersiveMaterial(VectorXr wavelengths_m, VectorXc refractive_indices);
    [[nodiscard]] Material at_wavelength(Real lambda_m) const;

private:
    VectorXr lambda_;
    VectorXc n_;
};

}  // namespace specklebem::material
