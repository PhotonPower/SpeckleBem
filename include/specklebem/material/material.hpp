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
