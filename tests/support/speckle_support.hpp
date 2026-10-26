#pragma once
/// @file speckle_support.hpp
/// Synthetic reference speckle with known statistics for the speckle-statistics tests
/// (tests/unit/test_speckle_statistics.cpp; WP-S1):
///  (a) circular complex Gaussian field samples: intensity with the negative-exponential density,
///      contrast 1; optionally plus a coherent background (partially developed speckle);
///  (b) a propagated speckle field on a regular observation grid (Fourier-optics model): a
///      random-phase aperture (square, disc or Gaussian spot) sampled on an M x M lattice, the
///      Fraunhofer field at distance z by one FFT;
///  (c) two patterns with a prescribed intensity correlation coefficient rho.
///
/// Random numbers: std::mt19937_64 (its output sequence is fixed by the standard) mapped to
/// uniform doubles by the top 53 bits and to Gaussians by Box-Muller here, not by
/// std::normal_distribution (implementation-defined: libstdc++ and libc++ draw different
/// samples), so the GCC and Clang presets see the same realisations.
///
/// Model (b): pupil samples A(xi_p) = |P(xi_p)| exp(j phi_p), phi_p independent and uniform on
/// [0, 2 pi), at xi_p = (p - M / 2) d_xi; the Fraunhofer field is
/// U(x_m) = sum_p A(xi_p) exp(-j 2 pi x_m xi_p / (lambda z)) with x_m = (m - M / 2) dx and
/// dx = lambda z / (M d_xi), i.e. one 2D DFT. For independent random phases the field correlation
/// is <U(x) U*(x + d)> ~ sum_p |P(xi_p)|^2 exp(j 2 pi d xi_p / (lambda z)): the discrete Fourier
/// transform of the aperture intensity (van Cittert-Zernike). For the square of n_s x n_s samples
/// this is the Dirichlet kernel sin(pi n_s m / M) / (n_s sin(pi m / M)) per axis at a lag of m
/// pixels, the sampled sinc(D d / (lambda z)) with D = n_s d_xi. Samples spaced by M / n_s pixels
/// are uncorrelated (kernel zeros) and hence independent (jointly Gaussian).
#include "specklebem/core/types.hpp"
#include "specklebem/postprocessing/speckle_statistics.hpp"

#include <unsupported/Eigen/FFT>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace speckle_test {

using namespace specklebem;
using post::ApertureShape;

inline constexpr Real kPi = constants::pi;

/// Portable random source (see the file comment).
class Rng {
public:
    explicit Rng(std::uint64_t seed) : engine_(seed) {}
    /// Uniform on (0, 1) (never 0, so log() is safe).
    Real uniform() { return (static_cast<Real>(engine_() >> 11) + 0.5) * 0x1.0p-53; }
    /// Standard normal by Box-Muller (one of the pair).
    Real normal() {
        const Real r = std::sqrt(-2 * std::log(uniform()));
        return r * std::cos(2 * kPi * uniform());
    }
    /// Circular complex Gaussian with <|U|^2> = power.
    Complex circular_gaussian(Real power = 1) {
        const Real s = std::sqrt(0.5 * power);
        const Real re = normal();
        const Real im = normal();
        return {s * re, s * im};
    }
    Real phase() { return 2 * kPi * uniform(); }

private:
    std::mt19937_64 engine_;
};

/// (a) n independent circular complex Gaussian samples with <|U|^2> = 1.
inline VectorXc circular_gaussian_field(Index n, std::uint64_t seed) {
    Rng rng(seed);
    VectorXc u(n);
    for (Index i = 0; i < n; ++i) u(i) = rng.circular_gaussian();
    return u;
}

/// |U|^2 per sample.
inline VectorXr intensity_of(const VectorXc& u) {
    return u.cwiseAbs2();
}

/// (a) Partially developed speckle: |U + B|^2 with a coherent background of intensity
/// |B|^2 = ratio <|U|^2> (U from circular_gaussian_field). Contrast sqrt(1 + 2 ratio) / (1 +
/// ratio).
inline VectorXr speckle_with_background(Index n, Real ratio, std::uint64_t seed) {
    const VectorXc u = circular_gaussian_field(n, seed);
    const Complex b = std::sqrt(ratio);
    return (u.array() + b).abs2().matrix();
}

/// Analytic contrast of speckle plus a uniform coherent background of relative intensity ratio.
inline Real contrast_with_background(Real ratio) {
    return std::sqrt(1 + 2 * ratio) / (1 + ratio);
}

/// (c) Two intensity patterns with correlation coefficient rho (0 <= rho <= 1):
/// I1 = |U1|^2, I2 = |sqrt(rho) U1 + sqrt(1 - rho) U2|^2 with U1, U2 independent; the fields have
/// correlation sqrt(rho), the intensities rho = |mu|^2.
struct PatternPair {
    VectorXr i1, i2;
};
inline PatternPair correlated_patterns(Index n, Real rho, std::uint64_t seed) {
    const VectorXc u1 = circular_gaussian_field(n, seed);
    const VectorXc u2 = circular_gaussian_field(n, seed + 0x9E3779B97F4A7C15ULL);
    PatternPair p;
    p.i1 = u1.cwiseAbs2();
    p.i2 = (std::sqrt(rho) * u1 + std::sqrt(1 - rho) * u2).cwiseAbs2();
    return p;
}

/// (b) Propagated speckle field on an M x M observation grid.
struct PropagatedSpeckle {
    MatrixXc field;       ///< U(x_i, y_j), column-major (i: x axis = rows)
    Real pixel_size = 0;  ///< dx = lambda z / (M d_xi) [m]
    Real aperture = 0;    ///< D [m] of the speckle_size_expected convention (see make_speckle)
    Real wavelength = 0;
    Real z = 0;
    ApertureShape shape = ApertureShape::Square;
    Index samples_across = 0;  ///< n_s: pupil samples across D

    MatrixXr intensity() const { return field.cwiseAbs2(); }
};

/// 2D DFT (rows, then columns); the inverse is scaled by 1 / (rows cols) (Eigen::FFT default).
inline void fft2(MatrixXc& a, bool inverse) {
    Eigen::FFT<Real> fft;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<Complex> in(static_cast<std::size_t>(a.rows()));
        std::vector<Complex> out(in.size());
        for (Index c = 0; c < a.cols(); ++c) {
            for (Index r = 0; r < a.rows(); ++r) in[static_cast<std::size_t>(r)] = a(r, c);
            if (inverse)
                fft.inv(out.data(), in.data(), a.rows());
            else
                fft.fwd(out.data(), in.data(), a.rows());
            for (Index r = 0; r < a.rows(); ++r) a(r, c) = out[static_cast<std::size_t>(r)];
        }
        a.transposeInPlace();
    }
}

/// Builds the propagated speckle of a random-phase aperture: grid M x M, n_s pupil samples across
/// the aperture size D = `aperture` (d_xi = D / n_s), observation at distance z. Aperture
/// intensities: Square |P|^2 = 1 on n_s x n_s samples (D = n_s d_xi); Circular 1 for
/// |xi| <= D / 2 (the returned D is the second-moment equivalent 4 sqrt(<xi_x^2>) of the pixelated
/// disc, 0.46 % above the nominal one at n_s = 32); Gaussian exp(-2 |xi|^2 / w0^2) with
/// w0 = D / 2 (needs M >= 4 n_s so that the truncation at |xi| = M d_xi / 2 is below e^{-32}).
inline PropagatedSpeckle make_speckle(ApertureShape shape, Real wavelength, Real z, Real aperture,
                                      Index grid, Index samples_across, std::uint64_t seed) {
    if (grid < 2 * samples_across ||
        (shape == ApertureShape::Gaussian && grid < 4 * samples_across)) {
        throw std::invalid_argument("make_speckle: grid too small for the aperture");
    }
    Rng rng(seed);
    const Real dxi = aperture / static_cast<Real>(samples_across);
    const Real half = 0.5 * aperture;
    const Real w0 = half;
    MatrixXc a = MatrixXc::Zero(grid, grid);
    Real weight = 0;
    Real moment = 0;  // sum |P|^2 xi_x^2
    for (Index j = 0; j < grid; ++j) {
        for (Index i = 0; i < grid; ++i) {
            // Sample-centred coordinates: the square holds exactly n_s samples per axis.
            const Real xi = (static_cast<Real>(i - grid / 2) + 0.5) * dxi;
            const Real eta = (static_cast<Real>(j - grid / 2) + 0.5) * dxi;
            Real p2 = 0;
            switch (shape) {
                case ApertureShape::Square:
                    p2 = (std::abs(xi) < half && std::abs(eta) < half) ? 1 : 0;
                    break;
                case ApertureShape::Circular:
                    p2 = (xi * xi + eta * eta <= half * half) ? 1 : 0;
                    break;
                case ApertureShape::Gaussian:
                    p2 = std::exp(-2 * (xi * xi + eta * eta) / (w0 * w0));
                    break;
            }
            const Real phi = rng.phase();  // drawn for every sample: same stream for all shapes
            if (p2 > 0) {
                a(i, j) = std::sqrt(p2) * Complex(std::cos(phi), std::sin(phi));
                weight += p2;
                moment += p2 * xi * xi;
            }
        }
    }
    fft2(a, false);
    PropagatedSpeckle s;
    s.field = a / std::sqrt(weight);  // <|U|^2> = 1
    s.pixel_size = wavelength * z / (static_cast<Real>(grid) * dxi);
    s.aperture = shape == ApertureShape::Circular ? 4 * std::sqrt(moment / weight) : aperture;
    s.wavelength = wavelength;
    s.z = z;
    s.shape = shape;
    s.samples_across = samples_across;
    return s;
}

/// Every `stride`-th pixel per axis of a pattern, flattened (independent samples when the stride
/// is at least the zero spacing of the field correlation).
inline VectorXr decimate(const MatrixXr& intensity, Index stride) {
    std::vector<Real> v;
    for (Index j = 0; j < intensity.cols(); j += stride) {
        for (Index i = 0; i < intensity.rows(); i += stride) v.push_back(intensity(i, j));
    }
    return Eigen::Map<const VectorXr>(v.data(), static_cast<Index>(v.size()));
}

/// Field autocorrelation |<U(r) U*(r + d)>_r| / <|U|^2> directly from the complex field
/// (zero-padded FFTs), with the lag layout and overlap normalisation of
/// post::intensity_autocorrelation: entry (i, j) is the lag (i - rows / 2, j - cols / 2),
/// averaged over the overlapping pixel pairs.
inline MatrixXr field_correlation_of(const MatrixXc& u) {
    const Index rows = u.rows();
    const Index cols = u.cols();
    const Real mean = u.cwiseAbs2().mean();
    MatrixXc a = MatrixXc::Zero(2 * rows, 2 * cols);
    a.topLeftCorner(rows, cols) = u;
    fft2(a, false);
    a = a.cwiseAbs2().cast<Complex>();
    fft2(a, true);
    MatrixXr mu(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        const Index dj = j - cols / 2;
        for (Index i = 0; i < rows; ++i) {
            const Index di = i - rows / 2;
            const Complex c = a(di >= 0 ? di : di + 2 * rows, dj >= 0 ? dj : dj + 2 * cols);
            const auto overlap = static_cast<Real>((rows - std::abs(di)) * (cols - std::abs(dj)));
            mu(i, j) = std::abs(c) / overlap / mean;
        }
    }
    return mu;
}

/// Expected |mu_A| of the square-aperture lattice model at a lag of m pixels along one axis:
/// |sin(pi n_s m / M) / (n_s sin(pi m / M))| (1 at m = 0).
inline Real dirichlet_mu(Index m, Index samples_across, Index grid) {
    if (m % grid == 0)
        return 1;
    const Real a = kPi * static_cast<Real>(m) / static_cast<Real>(grid);
    return std::abs(std::sin(static_cast<Real>(samples_across) * a) /
                    (static_cast<Real>(samples_across) * std::sin(a)));
}

}  // namespace speckle_test
