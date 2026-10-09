#include "specklebem/material/material.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace specklebem::material {

Complex Material::refractive_index() const {
    // Principal branch; for passive media with exp(+jwt) this yields Im(n) <= 0.
    Complex n = std::sqrt(eps_r * mu_r);
    if (n.imag() > 0)
        n = -n;
    return n;
}

Complex Material::wave_impedance(Real /*omega*/) const {
    return constants::eta0 * std::sqrt(mu_r / eps_r);
}

Complex Material::wavenumber(Real omega) const {
    return omega / constants::c0 * refractive_index();
}

Material silver_500nm() {
    return {Complex(-9.794, -0.313), Complex(1, 0)};
}
Material silicon_500nm() {
    return {Complex(18.478, -0.606), Complex(1, 0)};
}

namespace {

void require_finite(Complex v, const char* name) {
    if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) {
        throw std::invalid_argument(std::string("field_decay_length: ") + name + " must be finite");
    }
}

/// delta = lambda_0 / (2 pi |Im n|) after validating the wavelength.
Real decay_length_of_index(Complex n, Real wavelength) {
    if (!(wavelength > 0.0) || !std::isfinite(wavelength)) {
        throw std::invalid_argument(
            "field_decay_length: wavelength must be positive and finite, got " +
            std::to_string(wavelength));
    }
    const Real kappa = std::abs(n.imag());
    if (kappa == 0.0)
        return std::numeric_limits<Real>::infinity();
    return wavelength / (2.0 * constants::pi * kappa);
}

}  // namespace

Real field_decay_length(Complex eps_r, Real wavelength) {
    require_finite(eps_r, "eps_r");
    return decay_length_of_index(std::sqrt(eps_r), wavelength);
}

Real field_decay_length(const Material& m, Real wavelength) {
    require_finite(m.eps_r, "eps_r");
    require_finite(m.mu_r, "mu_r");
    return decay_length_of_index(m.refractive_index(), wavelength);
}

DispersiveMaterial::DispersiveMaterial(VectorXr wavelengths_m, VectorXc refractive_indices)
    : lambda_(std::move(wavelengths_m)), n_(std::move(refractive_indices)) {
    if (lambda_.size() != n_.size() || lambda_.size() < 2) {
        throw std::invalid_argument("DispersiveMaterial: need >= 2 matching samples");
    }
}

Material DispersiveMaterial::at_wavelength(Real lambda_m) const {
    if (lambda_m <= lambda_(0) || lambda_m >= lambda_(lambda_.size() - 1)) {
        throw std::out_of_range("DispersiveMaterial: wavelength outside tabulated range");
    }
    Index i = 0;
    while (lambda_(i + 1) < lambda_m) ++i;
    const Real t = (lambda_m - lambda_(i)) / (lambda_(i + 1) - lambda_(i));
    const Complex n = (1 - t) * n_(i) + t * n_(i + 1);
    return {n * n, Complex(1, 0)};
}

}  // namespace specklebem::material
