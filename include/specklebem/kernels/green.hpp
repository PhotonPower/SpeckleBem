#pragma once
/// @file green.hpp
/// Scalar free-space Green's function and its gradient for region i:
///
///   G_i(r, r') = exp(-j k_i R) / (4 pi R),   R = |r - r'|
///
/// (exp(+jwt) convention => outgoing waves carry exp(-jkR)).
#include "specklebem/core/types.hpp"

namespace specklebem::kernels {

[[nodiscard]] inline Complex green(const Vec3& r, const Vec3& rp, Complex k) {
    const Real R = (r - rp).norm();
    return std::exp(Complex(0, -1) * k * R) / (4 * constants::pi * R);
}

[[nodiscard]] Vec3c grad_green(const Vec3& r, const Vec3& rp, Complex k);

}  // namespace specklebem::kernels
