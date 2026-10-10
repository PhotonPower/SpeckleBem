#include "specklebem/excitation/excitation.hpp"

#include "specklebem/core/logging.hpp"

#include <Eigen/Geometry>  // cross products of real vectors

#include <cmath>
#include <stdexcept>
#include <string>

#include "excitation_detail.hpp"

namespace specklebem::excitation {

namespace {

constexpr Complex kJ{0.0, 1.0};

using detail::all_finite;
using detail::omega_from_wavelength;
using detail::require_lossless;

/// Plain cross product a x b. Eigen's MatrixBase::cross conjugates the result for complex
/// scalars, which is not the vector product needed for complex field phasors.
Vec3c cross(const Vec3& a, const Vec3c& b) {
    return {a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x()};
}

}  // namespace

Real Excitation::wavelength() const {
    return 2 * constants::pi * constants::c0 / omega_;
}

// ---------------------------------------------------------------------------------------------
// PlaneWave
// ---------------------------------------------------------------------------------------------

PlaneWave::PlaneWave(Real wavelength, const Vec3& k_hat, const Vec3c& e0,
                     material::Material background)
    : Excitation(omega_from_wavelength(wavelength, "PlaneWave"), background) {
    require_lossless(background_, "PlaneWave");
    const Real k_norm = k_hat.norm();
    if (!(k_norm > 0) || !all_finite(k_hat)) {
        throw std::invalid_argument("PlaneWave: k_hat must be a finite, non-zero vector");
    }
    k_hat_ = k_hat / k_norm;
    const Real e_norm = e0.norm();
    if (!(e_norm > 0) || !std::isfinite(e_norm)) {
        throw std::invalid_argument("PlaneWave: e0 must be a finite, non-zero vector");
    }
    // Unconjugated product: e0 . k_hat = 0 is the transversality condition for complex e0.
    const Complex longitudinal = e0.x() * k_hat_.x() + e0.y() * k_hat_.y() + e0.z() * k_hat_.z();
    if (std::abs(longitudinal) > 1e-12 * e_norm) {
        throw std::invalid_argument("PlaneWave: e0 must be transverse to k_hat (e0 . k_hat = 0)");
    }
    e0_ = e0;
    k_ = background_.wavenumber(omega_);
    eta_ = background_.wave_impedance(omega_);
}

Vec3c PlaneWave::electric_field(const Vec3& r) const {
    return e0_ * std::exp(-kJ * k_ * k_hat_.dot(r));
}

Vec3c PlaneWave::magnetic_field(const Vec3& r) const {
    return cross(k_hat_, electric_field(r)) / eta_;
}

// ---------------------------------------------------------------------------------------------
// GaussianBeam (paraxial, with first-order longitudinal components)
// ---------------------------------------------------------------------------------------------

GaussianBeam::GaussianBeam(Params p, material::Material background)
    : Excitation(omega_from_wavelength(p.wavelength, "GaussianBeam"), background), p_(p) {
    require_lossless(background_, "GaussianBeam");
    if (!(p_.waist_radius > 0) || !std::isfinite(p_.waist_radius)) {
        throw std::invalid_argument("GaussianBeam: waist_radius must be finite and > 0");
    }
    if (!all_finite(p_.focus)) {
        throw std::invalid_argument("GaussianBeam: focus must be finite");
    }
    if (!(std::abs(p_.incidence_angle) < constants::pi / 2)) {
        throw std::invalid_argument(
            "GaussianBeam: |incidence_angle| must be < pi/2 (beam travelling towards +z)");
    }
    // Beam frame with R_y(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]:
    // k_hat = R_y z_hat, e_p = R_y x_hat (p), y_hat (s).
    const Real c = std::cos(p_.incidence_angle);
    const Real s = std::sin(p_.incidence_angle);
    k_hat_ = Vec3(s, 0, c);
    e_p_ = Vec3(c, 0, -s);
    e_hat_ = (p_.polarization == Polarization::P) ? e_p_ : Vec3(Vec3::UnitY());
    h_hat_ = k_hat_.cross(e_hat_);               // real vectors: Eigen's cross is the plain product
    k_ = background_.wavenumber(omega_).real();  // real: the background is lossless
    z_r_ = 0.5 * k_ * p_.waist_radius * p_.waist_radius;  // = pi w0^2 n / lambda
    eta_ = background_.wave_impedance(omega_);

    const Real ratio = p_.wavelength / (constants::pi * p_.waist_radius);
    SBEM_WARN(
        "GaussianBeam: the paraxial model (w0 = {:.4g} m, lambda = {:.4g} m) violates Maxwell's "
        "equations at the (lambda/(pi w0))^2 = {:.3g} level; use AngularSpectrumBeam for "
        "quantitative results",
        p_.waist_radius, p_.wavelength, ratio * ratio);
}

GaussianBeam::Sample GaussianBeam::sample(const Vec3& r) const {
    // A = (j z_R / q) exp(-j k rho^2 / (2 q)) exp(-j k zeta) with q = zeta + j z_R. This equals
    // (w0/w) exp(-rho^2/w^2) exp(-j[k zeta + k rho^2/(2R) - psi]) written with the complex beam
    // parameter: j k / (2 q) = 1/w^2 + j k/(2R) and j z_R/q = (w0/w) e^{j psi}. This form has no
    // division by zeta at the waist.
    const Vec3 d = r - p_.focus;
    const Real zeta = k_hat_.dot(d);
    const Real u = e_p_.dot(d);
    const Real v = d.y();
    const Real rho2 = u * u + v * v;
    const Complex q(zeta, z_r_);
    const Complex alpha = kJ * k_ / (2.0 * q);
    const Complex a = (kJ * z_r_ / q) * std::exp(-alpha * rho2 - kJ * (k_ * zeta));
    // grad_t A = -2 alpha A (u e_p + v y_hat); only its projections on e_hat and h_hat are needed.
    const Complex g = -2.0 * alpha * a;
    const Vec3 t = u * e_p_ + v * Vec3::UnitY();
    return {a, g * e_hat_.dot(t), g * h_hat_.dot(t)};
}

Vec3c GaussianBeam::electric_field(const Vec3& r) const {
    const Sample s = sample(r);
    // E = e_hat A + k_hat E_zeta with E_zeta = -(j/k) e_hat . grad_t A (makes div E = 0 and
    // the curl equations hold to O((lambda/(pi w0))^2)).
    const Complex e_zeta = -kJ / k_ * s.grad_e;
    return e_hat_.cast<Complex>() * s.amplitude + k_hat_.cast<Complex>() * e_zeta;
}

Vec3c GaussianBeam::magnetic_field(const Vec3& r) const {
    const Sample s = sample(r);
    // eta H = h_hat A + k_hat H_zeta with H_zeta = -(j/k) h_hat . grad_t A; the transverse part
    // is k_hat x E / eta.
    const Complex h_zeta = -kJ / k_ * s.grad_h;
    return (h_hat_.cast<Complex>() * s.amplitude + k_hat_.cast<Complex>() * h_zeta) / eta_;
}

}  // namespace specklebem::excitation
