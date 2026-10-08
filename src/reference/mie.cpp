/// @file mie.cpp
/// Mie series for a homogeneous sphere, Bohren & Huffman (BH) 1983, chapter 4 and
/// appendix A, with downward recurrences for the minimal solutions (cf. Wiscombe 1980).
///
/// Time convention mapping (the only place where the two conventions meet)
/// -----------------------------------------------------------------------
/// SpeckleBem uses exp(+jwt) (docs/06_conventions.md): outgoing waves exp(-jkr), passive media
/// Im(n) <= 0. BH use exp(-iwt). A physical field Re{E_BH exp(-iwt)} equals
/// Re{conj(E_BH) exp(+jwt)}, so for real geometry inputs our phasor is E = conj(E_BH), provided
/// BH's formulas are evaluated for the same physical material, i.e. with
///     m_BH = conj(n_sphere / n_medium)   (Im(m_BH) >= 0),   x = k1 * radius (real).
/// Implementation:
///   1. Constructor: m_BH = conj(m) is formed once (conjugation point 1); a_n, b_n, c_n, d_n
///      are computed with BH's formulas and stored conjugated (conjugation point 2), so that
///      a_n()/b_n() return coefficients in our convention.
///   2. field_bh(): un-conjugates the stored coefficients and sums BH's vector spherical
///      harmonic series (BH eqs. 4.37, 4.40, 4.45, 4.50) in BH's convention with BH's outgoing
///      Hankel function h_n^(1) = j_n + i y_n and internal argument rho = m_BH k1 r.
///   3. scattered_E/H, internal_E return conj(field_bh(...)) (conjugation point 3).
/// Checks: E_inc = x_hat exp(-j k1 z) is conj(BH's x_hat exp(i k z)); H follows BH's
/// H = (1/(i w mu)) curl E, whose conjugate satisfies curl E = -j w mu H. Cross sections and the
/// RCS depend only on |a_n|, |b_n|, Re(a_n + b_n), |S1|, |S2| and are convention independent.
///
/// Numerics
///   * n_max = ceil(x + 11 x^(1/3) + 1) unless overridden: the near-field criterion of Neves &
///     Pisignano, Opt. Lett. 37 (2012) 2418. The scattered and internal series are truncated
///     like the expansion of the incident wave, so the far-field order of Wiscombe,
///     x + 4 x^(1/3) + 2, leaves ~5e-6 field errors on the surface; the extra coefficients are
///     < 1e-8 and do not change the cross sections.
///   * D_n(m x) by downward recurrence with D = 0 started at
///     n = max(n_max, |m x|) + 15 + ceil(4 |m x|^(1/3)); the 4 |mx|^(1/3) margin is needed for
///     large |m x| with weak absorption (BH's fixed margin of 15 leaves 1e-3 errors at m = 1.5,
///     x = 100).
///   * a_n, b_n from D_n(m x), psi_n(x), xi_n(x) (BH eq. 4.88): psi_n(x) = x j_n(x) by
///     downward (Miller) recurrence, chi_n(x) = -x y_n(x) by upward recurrence (y_n is the
///     dominant solution), xi_n = psi_n - i chi_n. Computing psi_n downwards (rather than
///     upwards as in BH's program) keeps a_n, b_n accurate for any user-supplied n_max.
///   * c_n, d_n from BH eq. 4.52 in Riccati-Bessel form using the Wronskian
///     psi_n xi_n' - xi_n psi_n' = i:
///         c_n = i m / [psi_n(mx) xi_n'(x) - m xi_n(x) psi_n'(mx)]
///         d_n = i m / [m psi_n(mx) xi_n'(x) - xi_n(x) psi_n'(mx)]
///   * j_n(z) at complex z by Miller's downward recurrence started at
///     n = max(n_max, |z|) + 20 + ceil(4 |z|^(1/3)), normalised with whichever of the closed
///     forms j_0, j_1 has the larger modulus (robust near zeros of j_0 for real z).
///   * Overflow (e.g. a very large user n_max, where y_n(x) exceeds the double range) is
///     detected: the constructor throws std::overflow_error if any intermediate or coefficient
///     is not finite, instead of silently producing zero or NaN coefficients.
///   * h_n^(1)(rho) for real rho by upward recurrence.
///   * pi_n, tau_n by the upward recurrences of BH eq. 4.47, regular at theta = 0, pi.
#include "specklebem/reference/mie.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace specklebem::reference {

namespace {

constexpr Real kPi = constants::pi;
/// Relative tolerance on |r| = radius for the domain checks of the field evaluators.
constexpr Real kSurfaceTol = 1e-10;
/// Limit on |Im(m x)|: sin/cos of the internal argument overflow beyond ~700.
constexpr Real kMaxImagArg = 600.0;

Real to_real(Index n) {
    return static_cast<Real>(n);
}

bool is_finite(Complex z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

/// Spherical Bessel functions j_0..j_nmax at complex argument z (Miller's algorithm).
VectorXc spherical_jn(Complex z, Index nmax) {
    VectorXc j = VectorXc::Zero(nmax + 1);
    if (z == Complex(0, 0)) {
        j(0) = 1;
        return j;
    }
    const Real az = std::abs(z);
    const Index nstart = std::max(nmax, static_cast<Index>(std::ceil(az))) + 20 +
                         static_cast<Index>(std::ceil(4 * std::cbrt(az)));
    VectorXc f = VectorXc::Zero(nstart + 2);
    f(nstart) = 1;
    constexpr Real big = 1e150;
    for (Index n = nstart; n >= 1; --n) {
        f(n - 1) = to_real(2 * n + 1) / z * f(n) - f(n + 1);
        if (std::abs(f(n - 1)) > big) {
            // Rescale; values of the high orders that underflow are negligible. Multiply by
            // 1/big instead of dividing: Eigen's vectorised complex division forms
            // a * conj(b) / |b|^2 and overflows for |a| * big > DBL_MAX.
            f.segment(n - 1, nstart + 3 - n) *= Real(1) / big;
        }
    }
    const Complex s = std::sin(z);
    const Complex c = std::cos(z);
    const Complex j0 = s / z;
    const Complex j1 = (s / z - c) / z;
    const Complex scale = (std::abs(j0) >= std::abs(j1)) ? j0 / f(0) : j1 / f(1);
    j = scale * f.head(nmax + 1);
    return j;
}

/// Spherical Hankel functions h_n^(1)(rho) = j_n + i y_n, n = 0..nmax, real rho > 0 (BH
/// convention, outgoing for exp(-iwt)); upward recurrence, stable for the dominant solution.
VectorXc spherical_h1n(Real rho, Index nmax) {
    VectorXc h(std::max<Index>(nmax + 1, 2));
    const Complex i(0, 1);
    const Complex e = std::exp(i * rho);
    h(0) = -i * e / rho;
    h(1) = -e * (rho + i) / (rho * rho);
    for (Index n = 1; n + 1 <= nmax; ++n) {
        h(n + 1) = to_real(2 * n + 1) / rho * h(n) - h(n - 1);
    }
    return h.head(nmax + 1);
}

/// Angular functions pi_n(cos theta), tau_n(cos theta) for n = 0..nmax (BH eq. 4.47).
void angular_functions(Real mu, Index nmax, VectorXr& pi, VectorXr& tau) {
    pi = VectorXr::Zero(nmax + 1);
    tau = VectorXr::Zero(nmax + 1);
    if (nmax >= 1) {
        pi(1) = 1;
        tau(1) = mu;
    }
    for (Index n = 2; n <= nmax; ++n) {
        const Real rn = to_real(n);
        pi(n) = (2 * rn - 1) / (rn - 1) * mu * pi(n - 1) - rn / (rn - 1) * pi(n - 2);
        tau(n) = rn * mu * pi(n) - (rn + 1) * pi(n - 1);
    }
}

/// Radial functions of one vector spherical harmonic family at argument rho:
/// z_n(rho), z_n(rho)/rho and [rho z_n(rho)]'/rho = z_{n-1} - n z_n / rho, n = 0..nmax.
struct Radial {
    VectorXc z, z_over_rho, dz;
};

Radial radial_from(const VectorXc& zn, Complex rho, Index nmax) {
    Radial out{zn, VectorXc::Zero(nmax + 1), VectorXc::Zero(nmax + 1)};
    for (Index n = 1; n <= nmax; ++n) {
        out.z_over_rho(n) = zn(n) / rho;
        out.dz(n) = zn(n - 1) - to_real(n) * out.z_over_rho(n);
    }
    return out;
}

/// Regular radial functions j_n at rho (handles rho = 0 by the analytic limits
/// j_1/rho -> 1/3, [rho j_1]'/rho -> 2/3, zero for n >= 2). Requires nmax >= 1.
Radial regular_radial(Complex rho, Index nmax) {
    if (rho == Complex(0, 0)) {
        const VectorXc e0 = VectorXc::Unit(nmax + 1, 0);
        const VectorXc e1 = VectorXc::Unit(nmax + 1, 1);
        return {e0, e1 * (Real(1) / 3), e1 * (Real(2) / 3)};
    }
    return radial_from(spherical_jn(rho, nmax), rho, nmax);
}

/// i^n.
Complex i_pow(Index n) {
    switch (n % 4) {
        case 0:
            return {1, 0};
        case 1:
            return {0, 1};
        case 2:
            return {-1, 0};
        default:
            return {0, -1};
    }
}

}  // namespace

MieSolution::MieSolution(const MieParams& p) : p_(p) {
    if (!(p.radius > 0)) {
        throw std::invalid_argument("MieSolution: radius must be > 0");
    }
    if (!(p.wavelength > 0)) {
        throw std::invalid_argument("MieSolution: wavelength must be > 0");
    }
    if (p.n_max < 0) {
        throw std::invalid_argument("MieSolution: n_max must be >= 0 (0 = default criterion)");
    }
    if (p.medium.eps_r.imag() != 0 || !(p.medium.eps_r.real() > 0)) {
        throw std::invalid_argument(
            "MieSolution: the background medium must be lossless (real eps_r > 0)");
    }
    if (p.medium.mu_r != Complex(1, 0)) {
        throw std::invalid_argument("MieSolution: the background medium must have mu_r = 1");
    }
    if (p.sphere.mu_r != Complex(1, 0)) {
        throw std::invalid_argument("MieSolution: the sphere must have mu_r = 1");
    }

    const Real n1 = std::sqrt(p.medium.eps_r.real());
    k1_ = 2 * kPi * n1 / p.wavelength;
    eta1_ = constants::eta0 / n1;
    const Real x = k1_ * p.radius;

    // Conjugation point 1: relative index in BH's exp(-iwt) convention, Im(m_bh_) >= 0.
    m_bh_ = std::conj(p.sphere.refractive_index() / n1);
    const Complex m = m_bh_;
    const Complex mx = m * x;
    if (std::abs(mx.imag()) > kMaxImagArg) {
        throw std::invalid_argument("MieSolution: |Im(m x)| > 600 is not supported (overflow)");
    }

    // Near-field truncation criterion (Neves & Pisignano 2012), see the file comment.
    const Index nmax = (p.n_max > 0) ? static_cast<Index>(p.n_max)
                                     : static_cast<Index>(std::ceil(x + 11 * std::cbrt(x) + 1));

    // Logarithmic derivative D_n(mx) = psi_n'(mx)/psi_n(mx), downward recurrence (BH app. A)
    // with a start index margin growing like |mx|^(1/3).
    const Real amx = std::abs(mx);
    const Index nmx = std::max(nmax, static_cast<Index>(std::ceil(amx))) + 15 +
                      static_cast<Index>(std::ceil(4 * std::cbrt(amx)));
    VectorXc D = VectorXc::Zero(nmx + 1);
    for (Index n = nmx; n >= 1; --n) {
        const Complex t = to_real(n) / mx;
        D(n - 1) = t - Real(1) / (D(n) + t);
    }

    // Riccati-Bessel functions of the real size parameter x.
    const VectorXc jx = spherical_jn(Complex(x, 0), nmax);
    const VectorXc hx = spherical_h1n(x, nmax);
    VectorXc psi(nmax + 1);
    VectorXc xi(nmax + 1);
    for (Index n = 0; n <= nmax; ++n) {
        psi(n) = x * jx(n).real();
        xi(n) = Complex(psi(n).real(), x * hx(n).imag());  // psi_n - i chi_n, chi_n = -x y_n
    }
    // Spherical Bessel functions of the internal argument mx (psi_n(mx) = mx j_n(mx)).
    const VectorXc jmx = spherical_jn(mx, nmax);

    a_.resize(nmax);
    b_.resize(nmax);
    c_.resize(nmax);
    d_.resize(nmax);
    const Complex i(0, 1);
    for (Index n = 1; n <= nmax; ++n) {
        const Real rn = to_real(n);
        // BH eq. 4.88 (with D_n).
        const Complex ta = D(n) / m + rn / x;
        const Complex tb = m * D(n) + rn / x;
        const Complex an = (ta * psi(n) - psi(n - 1)) / (ta * xi(n) - xi(n - 1));
        const Complex bn = (tb * psi(n) - psi(n - 1)) / (tb * xi(n) - xi(n - 1));
        // BH eq. 4.52 in Riccati form (Wronskian psi xi' - xi psi' = i).
        const Complex dxi = xi(n - 1) - rn / x * xi(n);
        const Complex psim = mx * jmx(n);
        const Complex dpsim = mx * jmx(n - 1) - rn * jmx(n);
        const Complex cn = i * m / (psim * dxi - m * xi(n) * dpsim);
        const Complex dn = i * m / (m * psim * dxi - xi(n) * dpsim);
        if (!is_finite(D(n)) || !is_finite(xi(n)) || !is_finite(xi(n - 1)) || !is_finite(psim) ||
            !is_finite(dpsim) || !is_finite(an) || !is_finite(bn) || !is_finite(cn) ||
            !is_finite(dn)) {
            throw std::overflow_error(
                "MieSolution: Riccati-Bessel functions or Mie coefficients"
                " overflow at order n = " +
                std::to_string(n) + " (reduce n_max)");
        }
        // Conjugation point 2: store in the exp(+jwt) convention.
        a_(n - 1) = std::conj(an);
        b_(n - 1) = std::conj(bn);
        c_(n - 1) = std::conj(cn);
        d_(n - 1) = std::conj(dn);
    }
}

Vec3c MieSolution::field_bh(const Vec3& r, Field which) const {
    const Real rr = r.norm();
    const bool inside = which == Field::internal_E;
    if (inside && rr > p_.radius * (1 + kSurfaceTol)) {
        throw std::domain_error("MieSolution::internal_E: point outside the sphere");
    }
    if (!inside && rr < p_.radius * (1 - kSurfaceTol)) {
        throw std::domain_error("MieSolution::scattered_E/H: point inside the sphere");
    }

    // Spherical angles; at theta = 0, pi (and r = 0) phi = 0 is used, which is consistent
    // because the Cartesian field built below does not depend on phi there.
    const Real rho_xy = std::hypot(r.x(), r.y());
    const Real ct = (rr > 0) ? r.z() / rr : Real(1);
    const Real st = (rr > 0) ? rho_xy / rr : Real(0);
    const Real cp = (rho_xy > 0) ? r.x() / rho_xy : Real(1);
    const Real sp = (rho_xy > 0) ? r.y() / rho_xy : Real(0);
    const Vec3 er(st * cp, st * sp, ct);
    const Vec3 eth(ct * cp, ct * sp, -st);
    const Vec3 eph(-sp, cp, 0);

    const Index nmax = a_.size();
    VectorXr pin;
    VectorXr taun;
    angular_functions(ct, nmax, pin, taun);

    const Radial rad = inside
                           ? regular_radial(m_bh_ * (k1_ * rr), nmax)
                           : radial_from(spherical_h1n(k1_ * rr, nmax), Complex(k1_ * rr, 0), nmax);

    const Complex i(0, 1);
    Complex fr(0, 0);
    Complex fth(0, 0);
    Complex fph(0, 0);
    for (Index n = 1; n <= nmax; ++n) {
        const Real rn = to_real(n);
        const Complex En = i_pow(n) * ((2 * rn + 1) / (rn * (rn + 1)));
        // BH-convention coefficients (undo the storage conjugation).
        const Complex an = std::conj(a_(n - 1));
        const Complex bn = std::conj(b_(n - 1));
        const Complex cn = std::conj(c_(n - 1));
        const Complex dn = std::conj(d_(n - 1));
        const Real pn = pin(n);
        const Real tn = taun(n);
        const Complex zn = rad.z(n);
        const Complex zr = rad.z_over_rho(n);
        const Complex dz = rad.dz(n);
        const Real nn1st = rn * (rn + 1) * st;
        switch (which) {
            case Field::scattered_E: {
                // E_s = sum E_n (i a_n N_e1n^(3) - b_n M_o1n^(3))  (BH 4.45)
                const Complex A = i * an;
                const Complex B = -bn;
                fr += En * A * nn1st * pn * zr;
                fth += En * (A * tn * dz + B * pn * zn);
                fph += En * (A * pn * dz + B * tn * zn);
                break;
            }
            case Field::internal_E: {
                // E_1 = sum E_n (c_n M_o1n^(1) - i d_n N_e1n^(1))  (BH 4.40)
                const Complex A = -i * dn;
                const Complex B = cn;
                fr += En * A * nn1st * pn * zr;
                fth += En * (A * tn * dz + B * pn * zn);
                fph += En * (A * pn * dz + B * tn * zn);
                break;
            }
            case Field::scattered_H: {
                // H_s = (k/(w mu)) sum E_n (i b_n N_o1n^(3) + a_n M_e1n^(3))  (BH 4.45)
                const Complex C = i * bn;
                const Complex Dm = an;
                fr += En * C * nn1st * pn * zr;
                fth += En * (C * tn * dz - Dm * pn * zn);
                fph += En * (C * pn * dz - Dm * tn * zn);
                break;
            }
        }
    }

    Complex fr_phi;
    Complex fth_phi;
    Complex fph_phi;
    if (which == Field::scattered_H) {
        // N_o1n, M_e1n: r and theta components ~ sin(phi), phi component ~ cos(phi);
        // prefactor k1/(w mu0) = 1/eta1.
        fr_phi = sp * fr / eta1_;
        fth_phi = sp * fth / eta1_;
        fph_phi = cp * fph / eta1_;
    } else {
        // N_e1n, M_o1n: r and theta components ~ cos(phi), phi component ~ -sin(phi).
        fr_phi = cp * fr;
        fth_phi = cp * fth;
        fph_phi = -sp * fph;
    }
    Vec3c out;
    for (Index k = 0; k < 3; ++k) {
        out(k) = fr_phi * er(k) + fth_phi * eth(k) + fph_phi * eph(k);
    }
    return out;
}

Vec3c MieSolution::scattered_E(const Vec3& r) const {
    return field_bh(r, Field::scattered_E).conjugate();  // conjugation point 3
}

Vec3c MieSolution::scattered_H(const Vec3& r) const {
    return field_bh(r, Field::scattered_H).conjugate();  // conjugation point 3
}

Vec3c MieSolution::internal_E(const Vec3& r) const {
    return field_bh(r, Field::internal_E).conjugate();  // conjugation point 3
}

Real MieSolution::bistatic_rcs(Real theta, Real phi) const {
    const Index nmax = a_.size();
    VectorXr pin;
    VectorXr taun;
    angular_functions(std::cos(theta), nmax, pin, taun);
    Complex s1(0, 0);
    Complex s2(0, 0);
    for (Index n = 1; n <= nmax; ++n) {
        const Real rn = to_real(n);
        const Real w = (2 * rn + 1) / (rn * (rn + 1));
        const Complex an = std::conj(a_(n - 1));  // BH convention; |S1|, |S2| are unaffected
        const Complex bn = std::conj(b_(n - 1));
        s1 += w * (an * pin(n) + bn * taun(n));
        s2 += w * (an * taun(n) + bn * pin(n));
    }
    // Far field (BH 4.74): E_s,theta ~ e^{ikr}/(-ikr) cos(phi) S2,
    // E_s,phi ~ -e^{ikr}/(-ikr) sin(phi) S1
    // => sigma = lim 4 pi r^2 |E_s|^2 = (4 pi / k^2) (|S2|^2 cos^2 phi + |S1|^2 sin^2 phi).
    const Real c = std::cos(phi);
    const Real s = std::sin(phi);
    return 4 * kPi / (k1_ * k1_) * (std::norm(s2) * c * c + std::norm(s1) * s * s);
}

Real MieSolution::scattering_cross_section() const {
    Real sum = 0;
    for (Index n = 1; n <= a_.size(); ++n) {
        sum += (2 * to_real(n) + 1) * (std::norm(a_(n - 1)) + std::norm(b_(n - 1)));
    }
    return 2 * kPi / (k1_ * k1_) * sum;
}

Real MieSolution::extinction_cross_section() const {
    Real sum = 0;
    for (Index n = 1; n <= a_.size(); ++n) {
        sum += (2 * to_real(n) + 1) * (a_(n - 1) + b_(n - 1)).real();
    }
    return 2 * kPi / (k1_ * k1_) * sum;
}

}  // namespace specklebem::reference
