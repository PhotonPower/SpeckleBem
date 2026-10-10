#pragma once
/// @file excitation_detail.hpp
/// Validation helpers shared by the excitation sources (not installed).
#include "specklebem/core/types.hpp"
#include "specklebem/material/material.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace specklebem::excitation::detail {

/// omega = 2 pi c0 / lambda (vacuum wavelength). Validates lambda first because the base
/// class is initialised from the result.
inline Real omega_from_wavelength(Real wavelength, const char* who) {
    if (!(wavelength > 0) || !std::isfinite(wavelength)) {
        throw std::invalid_argument(std::string(who) + ": wavelength must be finite and > 0");
    }
    return 2 * constants::pi * constants::c0 / wavelength;
}

/// The incident fields are defined for a lossless background only (real k, real eta), as in
/// reference::MieSolution.
inline void require_lossless(const material::Material& m, const char* who) {
    if (m.eps_r.imag() != 0 || !(m.eps_r.real() > 0) || m.mu_r.imag() != 0 ||
        !(m.mu_r.real() > 0)) {
        throw std::invalid_argument(std::string(who) +
                                    ": the background medium must be lossless "
                                    "(real eps_r > 0 and real mu_r > 0)");
    }
}

inline bool all_finite(const Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

}  // namespace specklebem::excitation::detail
