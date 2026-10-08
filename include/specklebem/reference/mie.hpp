#pragma once
/// @file mie.hpp
/// Mie series for a homogeneous sphere (Bohren & Huffman conventions adapted
/// to exp(+jwt)). Used as the exact reference for solver validation:
/// near field, far field and bistatic RCS.
///
/// Geometry and illumination (docs/06_conventions.md): sphere of radius `radius` centred at
/// the origin in a lossless background `medium`; incident plane wave travelling in +z,
///   E_inc = x_hat * 1 V/m * exp(-j k1 z),   H_inc = y_hat * (1/eta1) * exp(-j k1 z),
/// with k1 = 2 pi n_medium / wavelength (wavelength in vacuum) and eta1 = eta0 / n_medium.
///
/// Convention mapping: Bohren & Huffman (BH, 1983, ch. 4) use exp(-i w t). All series are
/// evaluated in BH's convention with the relative index m_BH = conj(n_sphere / n_medium)
/// (Im(m_BH) >= 0) and the real size parameter x = k1 * radius; every field returned by this
/// class is the complex conjugate of the corresponding BH field, which is the exp(+jwt)
/// phasor of the same physical field (outgoing waves exp(-j k r), Im(n_sphere) <= 0).
/// Cross sections and RCS are real and convention independent. See src/reference/mie.cpp.
#include "specklebem/core/types.hpp"
#include "specklebem/material/material.hpp"

namespace specklebem::reference {

struct MieParams {
    Real radius;
    Real wavelength;
    material::Material sphere;
    material::Material medium = material::vacuum();
    int n_max = 0;  ///< 0 => default near-field criterion ceil(x + 11 x^(1/3) + 1)
};

class MieSolution {
public:
    /// Computes the coefficients a_n, b_n, c_n, d_n for n = 1..n_max.
    /// Default n_max = ceil(x + 11 x^(1/3) + 1) with x = k1 * radius (near-field criterion of
    /// Neves & Pisignano 2012; 28 for x = 6.28), so that the near fields converge on the sphere
    /// surface; it exceeds Wiscombe's far-field order x + 4 x^(1/3) + 2, and the additional
    /// coefficients are < 1e-8. p.n_max > 0 overrides the default.
    /// Throws std::invalid_argument for radius <= 0, wavelength <= 0, p.n_max < 0, a lossy or
    /// magnetic medium (Im(eps_r) != 0, Re(eps_r) <= 0 or mu_r != 1), a magnetic sphere
    /// (mu_r != 1), or |Im(m x)| > 600 (overflow of the internal Bessel functions).
    /// Throws std::overflow_error if a Bessel function or coefficient is not finite (e.g. a
    /// user n_max far beyond x, where y_n(x) exceeds the double range).
    explicit MieSolution(const MieParams& p);

    /// Scattered field for an x-polarised plane wave travelling in +z (exact series, not the
    /// far-field approximation). Valid for |r| >= radius; throws std::domain_error inside
    /// (points within a relative 1e-10 of the surface are accepted).
    [[nodiscard]] Vec3c scattered_E(const Vec3& r) const;
    [[nodiscard]] Vec3c scattered_H(const Vec3& r) const;
    /// Field inside the sphere. Valid for |r| <= radius; throws std::domain_error outside
    /// (points within a relative 1e-10 of the surface are accepted).
    [[nodiscard]] Vec3c internal_E(const Vec3& r) const;

    /// Bistatic radar cross section sigma = lim 4 pi r^2 |E_s|^2 / |E_inc|^2 in m^2, from the
    /// far-field amplitudes: sigma = (4 pi / k1^2) (|S2|^2 cos^2 phi + |S1|^2 sin^2 phi).
    /// theta is measured from +z (forward scattering theta = 0, backscattering theta = pi),
    /// phi from +x.
    [[nodiscard]] Real bistatic_rcs(Real theta, Real phi) const;
    /// C_sca = (2 pi / k1^2) sum (2n+1) (|a_n|^2 + |b_n|^2) in m^2.
    [[nodiscard]] Real scattering_cross_section() const;
    /// C_ext = (2 pi / k1^2) sum (2n+1) Re(a_n + b_n) in m^2.
    [[nodiscard]] Real extinction_cross_section() const;

    /// Mie coefficients in the exp(+jwt) convention of this library, i.e. the complex
    /// conjugates of the Bohren & Huffman coefficients a_n^BH, b_n^BH evaluated with
    /// m_BH = conj(n_sphere / n_medium). Element i holds order n = i + 1; the vector length is
    /// the n_max in use. Re(a_n), Re(b_n), |a_n|, |b_n| (and hence all cross sections) are the
    /// same in both conventions.
    [[nodiscard]] const VectorXc& a_n() const { return a_; }
    [[nodiscard]] const VectorXc& b_n() const { return b_; }

private:
    /// Sum of the vector spherical harmonic series in BH's convention (see mie.cpp).
    enum class Field { scattered_E, scattered_H, internal_E };
    [[nodiscard]] Vec3c field_bh(const Vec3& r, Field which) const;

    MieParams p_;
    VectorXc a_, b_, c_, d_;  ///< exp(+jwt) convention (conjugated BH coefficients)
    Real k1_ = 0;             ///< wavenumber of the medium (real)
    Real eta1_ = 0;           ///< wave impedance of the medium (real)
    Complex m_bh_{1, 0};      ///< relative refractive index in BH's convention
};

}  // namespace specklebem::reference
