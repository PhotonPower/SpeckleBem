#include "specklebem/kernels/green.hpp"

namespace specklebem::kernels {

Vec3c grad_green(const Vec3& r, const Vec3& rp, Complex k) {
    // grad_r G = -(1 + jkR) G / R^2 * (r - r')
    const Vec3 d = r - rp;
    const Real R = d.norm();
    const Complex G = green(r, rp, k);
    const Complex factor = -(Complex(1) + Complex(0, 1) * k * R) * G / (R * R);
    return factor * d.cast<Complex>();
}

}  // namespace specklebem::kernels
