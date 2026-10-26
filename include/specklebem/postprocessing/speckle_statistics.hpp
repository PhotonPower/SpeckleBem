#pragma once
/// @file speckle_statistics.hpp
/// Statistical descriptors of speckle patterns (docs/01_project_plan.md Phase 6; acceptance rows
/// "Speckle PDF", "Contrast, gamma_12 vs sigma" and "Speckle size" of docs/05_validation.md).
///
/// Background (standard first-order and second-order speckle statistics, Goodman, "Speckle
/// Phenomena in Optics", 2007, chapters 3-5): a fully developed, polarised speckle field U is a
/// circular complex Gaussian random variable. Its intensity I = |U|^2 then has the
/// negative-exponential density p(I) = exp(-I / <I>) / <I> (unit mean: p(x) = e^{-x}, x = I / <I>),
/// contrast C = std(I) / <I> = 1, and the intensity autocorrelation obeys the Siegert relation
/// <I(r) I(r + d)> / <I>^2 = 1 + |mu_A(d)|^2, with mu_A the normalised field correlation.
/// For a random-phase aperture whose intensity distribution is |P(xi)|^2, observed in the
/// Fraunhofer zone at distance z, mu_A(d) is the Fourier transform of |P|^2 evaluated at the
/// spatial frequency d / (lambda z), normalised to 1 at d = 0 (van Cittert-Zernike theorem).
///
/// Input conventions: intensities are non-negative, finite and in any unit (only ratios enter).
/// Grids are column-major Eigen matrices: entry (i, j) is the pixel (u_i, v_j). An intensity from
/// post::plane_grid(origin, u, v, nu, nv) (row i * nv + j) is the matrix with nu rows and nv
/// columns (in NumPy: I.reshape(nu, nv)). Statistics of the samples (contrast, KS test) assume
/// independent samples: speckle pixels closer than about one speckle size are correlated, so
/// decimate a finely sampled pattern before ks_test_exponential (see there).
#include "specklebem/core/types.hpp"

namespace specklebem::post {

/// Speckle contrast C = sqrt(<I^2> - <I>^2) / <I> (population moments over all samples).
/// 1 for fully developed polarised speckle; sqrt(1 + 2 r) / (1 + r) for speckle plus a uniform
/// coherent background of intensity r <I_speckle>. Throws std::invalid_argument for fewer than
/// two samples, negative or non-finite values, or <I> = 0.
Real speckle_contrast(const VectorXr& intensity);

/// Correlation coefficient of two speckle patterns (e.g. at two incidence angles, the angular
/// correlation gamma_12): rho = <dI1 dI2> / sqrt(<dI1^2> <dI2^2>), dI = I - <I> (Pearson, over
/// corresponding samples). For two jointly circular Gaussian fields with field correlation mu,
/// rho = |mu|^2. Throws std::invalid_argument for mismatched sizes, fewer than two samples,
/// negative or non-finite values, or a constant pattern.
Real correlation_coefficient(const VectorXr& I1, const VectorXr& I2);

/// Histogram estimate of the probability density of the normalised intensity x = I / <I>:
/// `bins` equal bins on [0, max x], density = count / (n * bin width) (integrates to 1).
/// Compare with the fully developed speckle density e^{-x}. Throws std::invalid_argument for
/// bins < 1 and for the inputs speckle_contrast rejects.
struct Histogram {
    VectorXr bin_centers, density;
};
Histogram intensity_pdf(const VectorXr& intensity, int bins = 64);

/// Root-mean-square difference sqrt(sum_b (density_b - e^{-x_b})^2 / B) between a histogram of
/// intensity_pdf and the negative-exponential density at the B bin centres (a diagnostic, not a
/// test statistic). Throws std::invalid_argument for an empty or inconsistent histogram.
Real intensity_pdf_rms_error(const Histogram& histogram);

/// Result of ks_test_exponential.
struct KsResult {
    Real statistic = 0;  ///< D_n = sup_x |F_n(x) - (1 - e^{-x})|
    Real p_value = 0;    ///< Q_KS(sqrt(n) D_n), asymptotic Kolmogorov distribution (conservative)
    Index n = 0;         ///< number of samples
    /// Stephens-modified statistic D* = (D_n - 0.2 / n)(sqrt(n) + 0.26 + 0.5 / sqrt(n)) for the
    /// exponential with the mean estimated from the sample (level-correct; ks_test_exponential).
    Real modified_statistic = 0;

    /// Level-correct decision at significance level alpha: modified_statistic >
    /// stephens_exponential_critical_value(alpha); true = not negative-exponential at level
    /// alpha. Throws std::invalid_argument for alpha outside [0.01, 0.15].
    [[nodiscard]] bool rejects(Real alpha = 0.05) const;
};
// KsResult stays an aggregate (a non-virtual member function does not change that), so rejects()
// is a member; the WP-S1 field order statistic, p_value, n is kept and modified_statistic appended.

/// One-sample Kolmogorov-Smirnov test of x_i = I_i / <I> against the unit negative-exponential
/// CDF F(x) = 1 - e^{-x} (fully developed polarised speckle), with two decisions.
///
/// D_n = max_i max(i / n - F(x_(i)), F(x_(i)) - (i - 1) / n) over the sorted samples; the mean
/// <I> is estimated from the same sample.
///  * p_value = Q_KS(sqrt(n) D_n), the asymptotic Kolmogorov survival function (see
///    kolmogorov_survival). It assumes a fully specified CDF; estimating the scale makes D_n
///    smaller under the null hypothesis (Lilliefors), so this p-value is strongly conservative:
///    about 0.5 % false rejections at nominal 5 % because <I> is estimated (unit test, 2000 iid
///    exponential samples of n = 200: 0.35 %), and little power (a coherent background of half
///    the mean intensity, C = 0.943, is rejected in 26 % of 400 samples of n = 2000). Use
///    modified_statistic / rejects() for a level-correct 5 % decision.
///  * modified_statistic / rejects(alpha): the Lilliefors-type test for the exponential with
///    estimated mean in Stephens' form, D* = (D_n - 0.2 / n)(sqrt(n) + 0.26 + 0.5 / sqrt(n))
///    against the upper-tail critical values of stephens_exponential_critical_value (one table
///    for all n). Measured in the same unit tests: level 4.6 % at nominal 5 %, power 57 % in the
///    example above. Sources: M. A. Stephens, "EDF statistics for goodness of fit and some
///    comparisons", J. Amer. Statist. Assoc. 69 (1974) 730-737; R. B. D'Agostino and
///    M. A. Stephens (eds.), "Goodness-of-Fit Techniques", Marcel Dekker 1986, chapter 4
///    (exponential distribution, scale estimated). docs/05 ("Speckle PDF") uses rejects(0.05).
/// The samples must be independent: decimate a speckle pattern to a spacing of about two speckle
/// sizes (correlated pixels inflate D_n and give spurious rejections).
/// Throws std::invalid_argument for n < 50 and for the inputs speckle_contrast rejects.
KsResult ks_test_exponential(const VectorXr& intensity);

/// Upper-tail critical value of KsResult::modified_statistic (exponential, mean estimated) at the
/// significance level alpha, from Stephens' table (sources at ks_test_exponential): alpha 0.15 ->
/// 0.926, 0.10 -> 0.990, 0.05 -> 1.094, 0.025 -> 1.190, 0.01 -> 1.308, interpolated linearly in
/// log(alpha) between the table points (exact at them, decreasing in alpha). Throws
/// std::invalid_argument for alpha outside [0.01, 0.15] (or not finite).
Real stephens_exponential_critical_value(Real alpha);

/// Survival function of the Kolmogorov distribution,
/// Q_KS(lambda) = 2 sum_{k >= 1} (-1)^{k-1} exp(-2 k^2 lambda^2) = P(sqrt(n) D_n > lambda), n ->
/// inf. For lambda < 1.18 the equivalent theta-function form 1 - (sqrt(2 pi) / lambda) sum_{k >= 1}
/// exp(-(2k - 1)^2 pi^2 / (8 lambda^2)) is summed instead (the alternating series converges slowly
/// there). Q_KS(0) = 1; result clamped to [0, 1]. Throws std::invalid_argument for a negative or
/// non-finite lambda.
Real kolmogorov_survival(Real lambda);

/// Normalised intensity autocorrelation Gamma_I(d) = <I(r) I(r + d)>_r / <I>^2 on a regular grid,
/// via zero-padded FFTs (linear, not circular, correlation). The average over r runs over the
/// (rows - |d_i|)(cols - |d_j|) overlapping pixel pairs of lag d = (d_i, d_j) (unbiased in the
/// overlap count); <I> is the mean of the whole pattern. The result has the input's size; entry
/// (i, j) is the lag (i - rows / 2, j - cols / 2) in pixels (integer division: zero lag at
/// (rows / 2, cols / 2), the NumPy fftshift layout), so every lag has at least half of the
/// pixels along each axis in its overlap (one quarter of all pixels at a corner lag).
/// Gamma_I(0) = 1 + C^2; Gamma_I -> 1 at lags beyond the speckle size.
/// Throws std::invalid_argument for fewer than 2 x 2 pixels and the inputs speckle_contrast
/// rejects.
MatrixXr intensity_autocorrelation(const MatrixXr& intensity);

/// Field correlation |mu_A(d)| = sqrt(max(Gamma_I(d) - 1, 0)) from the intensity alone, by the
/// Siegert relation Gamma_I = 1 + |mu_A|^2 (valid for fully developed polarised speckle). Same
/// layout as intensity_autocorrelation. |mu_A(0)| = C, which equals 1 only for fully developed
/// speckle (speckle_size measures relative to the zero-lag value, so this does not enter there).
MatrixXr field_correlation_from_intensity(const MatrixXr& intensity);

/// Speckle size: full width at half maximum of |mu_A|^2 along one axis through zero lag, in metres
/// (pixel_size = grid spacing along that axis). mu_A has the layout of
/// field_correlation_from_intensity (zero lag at (rows / 2, cols / 2)); axis 0 measures along the
/// rows index (the u axis of plane_grid), axis 1 along the columns (v). The maximum is the
/// zero-lag value |mu_A(0)|^2; on each side the first sample below half of it is found, and the
/// crossing is interpolated with log |mu_A|^2 linear in d^2 (exact for a Gaussian peak; linear in
/// |mu_A|^2 if the sample beyond the crossing is 0).
/// Convention relating it to lambda z / D: speckle_size_expected. Sampling: on a sinc^2 peak the
/// interpolation error is below 1 % from 4 pixels per FWHM on (1.9 % at 3; unit test); the
/// statistical error of a pattern with N_s speckles is a few 1 / sqrt(N_s) on top (measure it over
/// seeds). Throws std::invalid_argument for pixel_size <= 0,
/// an axis other than 0 / 1, a non-positive or non-finite zero-lag value, and
/// std::runtime_error if |mu_A|^2 does not fall below half within the grid on both sides
/// (pattern smaller than about one speckle).
Real speckle_size(const MatrixXr& mu_A, Real pixel_size, int axis = 0);

/// Intensity distribution |P(xi)|^2 of the illuminated aperture for speckle_size_expected.
enum class ApertureShape {
    Square,    ///< uniform square, side D (measured along the axis of the speckle size)
    Circular,  ///< uniform disc, diameter D
    Gaussian,  ///< Gaussian beam spot |P|^2 ~ exp(-2 r^2 / w0^2), D = 2 w0 (1/e^2 intensity
               ///< diameter)
};

/// Constant c in FWHM(|mu_A|^2) = c lambda z / D (Fraunhofer zone, paraxial observation, random
/// phase aperture). Derivation: |mu_A(d)|^2 is |FT{|P|^2}|^2 at the frequency d / (lambda z).
///  * Square, side D: |mu_A|^2 = sinc^2(D d / (lambda z)), sinc(u) = sin(pi u) / (pi u);
///    sinc^2(u) = 1/2 at u = 0.442946 -> c = 0.885893.
///  * Disc, diameter D: |mu_A|^2 = [2 J_1(v) / v]^2, v = pi D d / (lambda z); [2 J_1(v) / v]^2 =
///  1/2
///    at v = 1.616340 -> c = 2 * 1.616340 / pi = 1.028994.
///  * Gaussian, D = 2 w0: |mu_A|^2 = exp(-(pi w0 d / (lambda z))^2) = 1/2 at
///    d = sqrt(ln 2) lambda z / (pi w0) -> c = 4 sqrt(ln 2) / pi = 1.060041.
Real speckle_size_constant(ApertureShape shape);

/// Predicted speckle size c lambda z / D [m] for the convention of speckle_size (FWHM of
/// |mu_A|^2), c = speckle_size_constant(shape); wavelength in the medium of observation.
/// Valid for z >> D^2 / lambda (Fraunhofer) and small observation angles. Throws
/// std::invalid_argument for non-positive or non-finite arguments.
Real speckle_size_expected(Real wavelength, Real z, Real aperture, ApertureShape shape);

}  // namespace specklebem::post
