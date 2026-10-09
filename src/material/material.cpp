#include "specklebem/material/material.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

/// Real number for error messages in general format (e.g. 4e-07; std::to_string would print
/// sub-micrometre wavelengths as 0.000000).
std::string fmt_real(Real x) {
    std::ostringstream os;
    os << std::setprecision(6) << x;
    return os.str();
}

/// delta = lambda_0 / (2 pi |Im n|) after validating the wavelength.
Real decay_length_of_index(Complex n, Real wavelength) {
    if (!(wavelength > 0.0) || !std::isfinite(wavelength)) {
        throw std::invalid_argument(
            "field_decay_length: wavelength must be positive and finite, got " +
            fmt_real(wavelength));
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
        throw std::invalid_argument("DispersiveMaterial: need >= 2 matching samples, got " +
                                    std::to_string(lambda_.size()) + " wavelengths and " +
                                    std::to_string(n_.size()) + " refractive indices");
    }
    for (Index i = 0; i < lambda_.size(); ++i) {
        if (!std::isfinite(lambda_(i)) || !(lambda_(i) > 0.0)) {
            throw std::invalid_argument(
                "DispersiveMaterial: wavelengths must be finite and positive, got " +
                fmt_real(lambda_(i)) + " at index " + std::to_string(i));
        }
        if (i > 0 && !(lambda_(i) > lambda_(i - 1))) {
            throw std::invalid_argument(
                "DispersiveMaterial: wavelengths must be strictly increasing (index " +
                std::to_string(i) + ")");
        }
        if (!std::isfinite(n_(i).real()) || !std::isfinite(n_(i).imag())) {
            throw std::invalid_argument(
                "DispersiveMaterial: refractive indices must be finite (index " +
                std::to_string(i) + ")");
        }
        if (n_(i).imag() > 0.0) {
            throw std::invalid_argument(
                "DispersiveMaterial: refractive index with Im(n) > 0 at index " +
                std::to_string(i) +
                "; the exp(+jwt) convention needs n - jk (Im(n) <= 0 for passive media), so "
                "n + ik optics data (e.g. refractiveindex.info) must be conjugated "
                "(docs/06_conventions.md)");
        }
    }
}

Material DispersiveMaterial::at_wavelength(Real lambda_m) const {
    if (!std::isfinite(lambda_m)) {
        throw std::invalid_argument("DispersiveMaterial: wavelength must be finite");
    }
    const Index last = lambda_.size() - 1;
    if (lambda_m < lambda_(0) || lambda_m > lambda_(last)) {
        throw std::out_of_range("DispersiveMaterial: wavelength " + fmt_real(lambda_m) +
                                " outside the tabulated range [" + fmt_real(lambda_(0)) + ", " +
                                fmt_real(lambda_(last)) + "]");
    }
    Index i = 0;
    while (i + 1 < last && lambda_(i + 1) < lambda_m) ++i;
    const Real t = (lambda_m - lambda_(i)) / (lambda_(i + 1) - lambda_(i));
    const Complex n = (1 - t) * n_(i) + t * n_(i + 1);
    return {n * n, Complex(1, 0)};
}

}  // namespace specklebem::material
