/// @file speckle_statistics.cpp
/// Speckle statistics (see the header for the definitions and conventions).
///
/// FFT: Eigen's unsupported FFT module (header-only kissfft, already used by the height-map
/// generator), 2D by rows and columns, padded to 5-smooth sizes (fast kissfft radices).
#include "specklebem/postprocessing/speckle_statistics.hpp"

#include <unsupported/Eigen/FFT>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace specklebem::post {

namespace {

using ArrayXr = Eigen::Array<Real, Eigen::Dynamic, 1>;

/// Checks that every sample is finite and non-negative and that the mean is positive; returns
/// the mean.
template <class Derived>
Real checked_mean(const Eigen::DenseBase<Derived>& intensity, const char* what) {
    if (intensity.size() < 2) {
        throw std::invalid_argument(std::string(what) + ": need at least two samples");
    }
    if (!intensity.allFinite()) {
        throw std::invalid_argument(std::string(what) + ": intensities must be finite");
    }
    if (intensity.minCoeff() < 0) {
        throw std::invalid_argument(std::string(what) + ": intensities must be non-negative");
    }
    const Real mean = intensity.mean();
    if (!(mean > 0)) {
        throw std::invalid_argument(std::string(what) + ": mean intensity must be positive");
    }
    return mean;
}

/// Smallest n' >= n with no prime factor other than 2, 3, 5.
Index next_smooth(Index n) {
    for (Index m = std::max<Index>(n, 1);; ++m) {
        Index r = m;
        for (const Index p : {2, 3, 5}) {
            while (r % p == 0) r /= p;
        }
        if (r == 1)
            return m;
    }
}

/// In-place 2D DFT (rows, then columns); the inverse is scaled by 1 / (rows cols).
void fft2(MatrixXc& a, bool inverse) {
    Eigen::FFT<Real> fft;
    const Index rows = a.rows();
    const Index cols = a.cols();
    std::vector<Complex> in(static_cast<std::size_t>(std::max(rows, cols)));
    std::vector<Complex> out(in.size());
    for (Index c = 0; c < cols; ++c) {
        for (Index r = 0; r < rows; ++r) in[static_cast<std::size_t>(r)] = a(r, c);
        if (inverse)
            fft.inv(out.data(), in.data(), rows);
        else
            fft.fwd(out.data(), in.data(), rows);
        for (Index r = 0; r < rows; ++r) a(r, c) = out[static_cast<std::size_t>(r)];
    }
    for (Index r = 0; r < rows; ++r) {
        for (Index c = 0; c < cols; ++c) in[static_cast<std::size_t>(c)] = a(r, c);
        if (inverse)
            fft.inv(out.data(), in.data(), cols);
        else
            fft.fwd(out.data(), in.data(), cols);
        for (Index c = 0; c < cols; ++c) a(r, c) = out[static_cast<std::size_t>(c)];
    }
}

void require_positive_finite(Real v, const char* what) {
    if (!std::isfinite(v) || !(v > 0)) {
        throw std::invalid_argument(std::string(what) + " must be positive and finite");
    }
}

}  // namespace

Real speckle_contrast(const VectorXr& intensity) {
    const Real mean = checked_mean(intensity, "speckle_contrast");
    const Real var = (intensity.array() - mean).square().mean();
    return std::sqrt(var) / mean;
}

Real correlation_coefficient(const VectorXr& I1, const VectorXr& I2) {
    if (I1.size() != I2.size()) {
        throw std::invalid_argument("correlation_coefficient: patterns differ in size");
    }
    if (I1.size() < 2) {
        throw std::invalid_argument("correlation_coefficient: need at least two samples");
    }
    if (!I1.allFinite() || !I2.allFinite()) {
        throw std::invalid_argument("correlation_coefficient: intensities must be finite");
    }
    if (I1.minCoeff() < 0 || I2.minCoeff() < 0) {
        throw std::invalid_argument("correlation_coefficient: intensities must be non-negative");
    }
    const ArrayXr d1 = I1.array() - I1.mean();
    const ArrayXr d2 = I2.array() - I2.mean();
    const Real s11 = d1.square().sum();
    const Real s22 = d2.square().sum();
    if (!(s11 > 0) || !(s22 > 0)) {
        throw std::invalid_argument("correlation_coefficient: a pattern is constant");
    }
    return (d1 * d2).sum() / std::sqrt(s11 * s22);
}

Histogram intensity_pdf(const VectorXr& intensity, int bins) {
    if (bins < 1) {
        throw std::invalid_argument("intensity_pdf: bins must be >= 1");
    }
    const Real mean = checked_mean(intensity, "intensity_pdf");
    const Real x_max = intensity.maxCoeff() / mean;
    const Real width = x_max / bins;
    Histogram h;
    h.bin_centers.resize(bins);
    h.density = VectorXr::Zero(bins);
    for (int b = 0; b < bins; ++b) h.bin_centers(b) = (b + 0.5) * width;
    for (Index i = 0; i < intensity.size(); ++i) {
        const Real x = intensity(i) / mean;
        const auto b = std::min<Index>(static_cast<Index>(x / width), bins - 1);
        h.density(b) += 1;
    }
    h.density /= static_cast<Real>(intensity.size()) * width;
    return h;
}

Real intensity_pdf_rms_error(const Histogram& histogram) {
    const Index b = histogram.bin_centers.size();
    if (b < 1 || histogram.density.size() != b) {
        throw std::invalid_argument("intensity_pdf_rms_error: empty or inconsistent histogram");
    }
    if (!histogram.bin_centers.allFinite() || !histogram.density.allFinite()) {
        throw std::invalid_argument("intensity_pdf_rms_error: histogram values must be finite");
    }
    const ArrayXr ref = (-histogram.bin_centers.array()).exp();
    return std::sqrt((histogram.density.array() - ref).square().mean());
}

Real kolmogorov_survival(Real lambda) {
    if (!std::isfinite(lambda) || lambda < 0) {
        throw std::invalid_argument("kolmogorov_survival: lambda must be finite and >= 0");
    }
    if (lambda < 1e-3)
        return 1;  // the theta form below is 1 - O(exp(-1e5)) here
    constexpr Real pi = constants::pi;
    Real q = 0;
    if (lambda < 1.18) {
        // 1 - sqrt(2 pi) / lambda sum exp(-(2k - 1)^2 pi^2 / (8 lambda^2)).
        const Real a = -pi * pi / (8 * lambda * lambda);
        Real s = 0;
        for (int k = 1; k <= 100; ++k) {
            const Real m = 2 * k - 1;
            const Real t = std::exp(a * m * m);
            s += t;
            if (t < 1e-17 * s)
                break;
        }
        q = 1 - std::sqrt(2 * pi) / lambda * s;
    } else {
        Real s = 0;
        for (int k = 1; k <= 100; ++k) {
            const Real t = std::exp(-2.0 * k * k * lambda * lambda);
            s += (k % 2 == 1) ? t : -t;
            if (t < 1e-17 * std::abs(s))
                break;
        }
        q = 2 * s;
    }
    return std::clamp(q, Real(0), Real(1));
}

KsResult ks_test_exponential(const VectorXr& intensity) {
    constexpr Index kMinSamples = 50;
    if (intensity.size() < kMinSamples) {
        throw std::invalid_argument("ks_test_exponential: need at least 50 samples (got " +
                                    std::to_string(intensity.size()) + ")");
    }
    const Real mean = checked_mean(intensity, "ks_test_exponential");
    std::vector<Real> x(intensity.data(), intensity.data() + intensity.size());
    std::sort(x.begin(), x.end());
    const auto n = static_cast<Index>(x.size());
    const Real nr = static_cast<Real>(n);
    Real d = 0;
    for (Index i = 0; i < n; ++i) {
        const Real f = -std::expm1(-x[static_cast<std::size_t>(i)] / mean);  // 1 - e^{-x}
        d = std::max({d, static_cast<Real>(i + 1) / nr - f, f - static_cast<Real>(i) / nr});
    }
    KsResult r;
    r.statistic = d;
    r.n = n;
    r.p_value = kolmogorov_survival(std::sqrt(nr) * d);
    const Real sn = std::sqrt(nr);
    r.modified_statistic = (d - 0.2 / nr) * (sn + 0.26 + 0.5 / sn);  // Stephens (1974)
    return r;
}

Real stephens_exponential_critical_value(Real alpha) {
    // Stephens (1974) / D'Agostino and Stephens (1986), exponential with estimated mean: upper-tail
    // points of the modified statistic, ordered by decreasing alpha.
    constexpr std::array<Real, 5> kAlpha = {0.15, 0.10, 0.05, 0.025, 0.01};
    constexpr std::array<Real, 5> kCritical = {0.926, 0.990, 1.094, 1.190, 1.308};
    if (!std::isfinite(alpha) || alpha < kAlpha.back() || alpha > kAlpha.front()) {
        throw std::invalid_argument(
            "stephens_exponential_critical_value: alpha must lie in [0.01, 0.15] (table range)");
    }
    for (std::size_t i = 0; i < kAlpha.size(); ++i) {
        if (alpha == kAlpha[i])
            return kCritical[i];  // table points exactly (no log round-off)
    }
    std::size_t i = 0;
    while (alpha < kAlpha[i + 1]) ++i;  // kAlpha[i] > alpha > kAlpha[i + 1]
    const Real t = std::log(kAlpha[i] / alpha) / std::log(kAlpha[i] / kAlpha[i + 1]);
    return kCritical[i] + t * (kCritical[i + 1] - kCritical[i]);
}

bool KsResult::rejects(Real alpha) const {
    return modified_statistic > stephens_exponential_critical_value(alpha);
}

MatrixXr intensity_autocorrelation(const MatrixXr& intensity) {
    const Index rows = intensity.rows();
    const Index cols = intensity.cols();
    if (rows < 2 || cols < 2) {
        throw std::invalid_argument("intensity_autocorrelation: need at least 2 x 2 pixels");
    }
    const Real mean = checked_mean(intensity.reshaped(), "intensity_autocorrelation");
    // Linear autocorrelation via |FFT|^2 of the zero-padded pattern (padding >= 2 n - 1 keeps
    // positive and negative lags apart).
    const Index pr = next_smooth(2 * rows - 1);
    const Index pc = next_smooth(2 * cols - 1);
    MatrixXc a = MatrixXc::Zero(pr, pc);
    a.topLeftCorner(rows, cols) = intensity.cast<Complex>();
    fft2(a, false);
    a = a.cwiseAbs2().cast<Complex>();
    fft2(a, true);
    MatrixXr gamma(rows, cols);
    const Index ci = rows / 2;
    const Index cj = cols / 2;
    const Real mean2 = mean * mean;
    for (Index j = 0; j < cols; ++j) {
        const Index dj = j - cj;
        const Index sj = dj >= 0 ? dj : dj + pc;
        const auto oj = static_cast<Real>(cols - std::abs(dj));
        for (Index i = 0; i < rows; ++i) {
            const Index di = i - ci;
            const Index si = di >= 0 ? di : di + pr;
            const auto oi = static_cast<Real>(rows - std::abs(di));
            gamma(i, j) = a(si, sj).real() / (oi * oj) / mean2;
        }
    }
    return gamma;
}

MatrixXr field_correlation_from_intensity(const MatrixXr& intensity) {
    const MatrixXr gamma = intensity_autocorrelation(intensity);
    return (gamma.array() - 1).max(0).sqrt().matrix();
}

Real speckle_size(const MatrixXr& mu_A, Real pixel_size, int axis) {
    require_positive_finite(pixel_size, "speckle_size: pixel_size");
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("speckle_size: axis must be 0 or 1");
    }
    if (mu_A.rows() < 1 || mu_A.cols() < 1) {
        throw std::invalid_argument("speckle_size: empty correlation matrix");
    }
    const Index ci = mu_A.rows() / 2;
    const Index cj = mu_A.cols() / 2;
    const VectorXr line = axis == 0 ? VectorXr(mu_A.col(cj)) : VectorXr(mu_A.row(ci).transpose());
    const Index c = axis == 0 ? ci : cj;
    const Real peak = line(c) * line(c);
    if (!std::isfinite(peak) || !(peak > 0)) {
        throw std::invalid_argument("speckle_size: the zero-lag value must be positive and finite");
    }
    const Real half = 0.5 * peak;
    // Distance (in pixels) from zero lag to the half-maximum crossing in direction `step`.
    auto crossing = [&](Index step) {
        for (Index k = 1;; ++k) {
            const Index i = c + step * k;
            if (i < 0 || i >= line.size()) {
                throw std::runtime_error(
                    "speckle_size: |mu_A|^2 does not fall below half of its maximum within the "
                    "grid (pattern smaller than about one speckle)");
            }
            const Real y1 = line(i) * line(i);
            if (!std::isfinite(y1)) {
                throw std::invalid_argument("speckle_size: mu_A must be finite");
            }
            if (y1 < half) {
                const Real y0 = line(i - step) * line(i - step);
                const auto d0 = static_cast<Real>(k - 1);
                const auto d1 = static_cast<Real>(k);
                if (y1 > 0) {
                    // log |mu_A|^2 linear in d^2 (exact for a Gaussian peak).
                    const Real t = std::log(y0 / half) / std::log(y0 / y1);
                    return std::sqrt(d0 * d0 + t * (d1 * d1 - d0 * d0));
                }
                return d0 + (y0 - half) / (y0 - y1);  // linear in |mu_A|^2
            }
        }
    };
    return (crossing(1) + crossing(-1)) * pixel_size;
}

Real speckle_size_constant(ApertureShape shape) {
    // Half-maximum points (see the header): sinc^2(u) = 1/2 at u = 0.44294647068945237,
    // [2 J_1(v) / v]^2 = 1/2 at v = 1.6163399483107028.
    switch (shape) {
        case ApertureShape::Square:
            return 2 * 0.44294647068945237;
        case ApertureShape::Circular:
            return 2 * 1.6163399483107028 / constants::pi;
        case ApertureShape::Gaussian:
            return 4 * std::sqrt(std::log(2.0)) / constants::pi;
    }
    throw std::invalid_argument("speckle_size_constant: unknown aperture shape");
}

Real speckle_size_expected(Real wavelength, Real z, Real aperture, ApertureShape shape) {
    require_positive_finite(wavelength, "speckle_size_expected: wavelength");
    require_positive_finite(z, "speckle_size_expected: z");
    require_positive_finite(aperture, "speckle_size_expected: aperture");
    return speckle_size_constant(shape) * wavelength * z / aperture;
}

}  // namespace specklebem::post
