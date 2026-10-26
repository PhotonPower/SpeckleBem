/// Unit tests of post::speckle_statistics (WP-S1) on synthetic speckle with known statistics
/// (tests/support/speckle_support.hpp). Each case < 1 s in release (a few s in win-debug).
#include "specklebem/postprocessing/speckle_statistics.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "speckle_support.hpp"

using namespace speckle_test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using post::ApertureShape;

namespace {

/// Brute-force Gamma_I with the layout and normalisation of post::intensity_autocorrelation.
MatrixXr autocorrelation_direct(const MatrixXr& im) {
    const Index rows = im.rows();
    const Index cols = im.cols();
    const Real mean = im.mean();
    MatrixXr g(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            const Index di = i - rows / 2;
            const Index dj = j - cols / 2;
            Real s = 0;
            Index count = 0;
            for (Index q = 0; q < cols; ++q) {
                for (Index p = 0; p < rows; ++p) {
                    if (p + di < 0 || p + di >= rows || q + dj < 0 || q + dj >= cols)
                        continue;
                    s += im(p, q) * im(p + di, q + dj);
                    ++count;
                }
            }
            g(i, j) = s / static_cast<Real>(count) / (mean * mean);
        }
    }
    return g;
}

/// Bisection for f(x) = 0 on [a, b] with f(a) f(b) < 0.
template <class F>
Real bisect(F f, Real a, Real b) {
    for (int it = 0; it < 200; ++it) {
        const Real m = 0.5 * (a + b);
        if ((f(a) < 0) == (f(m) < 0))
            a = m;
        else
            b = m;
    }
    return 0.5 * (a + b);
}

/// J_1 by its power series (libc++ has no std::cyl_bessel_j); exact to round-off for x < 4.
Real bessel_j1(Real x) {
    Real term = 0.5 * x;
    Real sum = term;
    for (int k = 1; k < 40; ++k) {
        term *= -(0.25 * x * x) / (static_cast<Real>(k) * static_cast<Real>(k + 1));
        sum += term;
    }
    return sum;
}

}  // namespace

TEST_CASE("speckle: Kolmogorov survival function, both series forms and quantiles", "[speckle]") {
    CHECK(post::kolmogorov_survival(0.0) == 1.0);
    CHECK(post::kolmogorov_survival(10.0) < 1e-80);
    // Tabulated quantiles of the Kolmogorov distribution.
    CHECK_THAT(post::kolmogorov_survival(1.3581), WithinAbs(0.05, 1e-4));
    CHECK_THAT(post::kolmogorov_survival(1.2238), WithinAbs(0.10, 1e-4));
    CHECK_THAT(post::kolmogorov_survival(1.6276), WithinAbs(0.01, 1e-4));
    CHECK_THAT(post::kolmogorov_survival(0.8276), WithinAbs(0.50, 1e-3));
    // The two forms agree where they switch (1.18) and on both sides of it.
    for (const Real lambda : {0.6, 1.0, 1.18, 1.5}) {
        Real alt = 0;
        for (int k = 1; k <= 200; ++k) {
            alt += ((k % 2 == 1) ? 2 : -2) * std::exp(-2.0 * k * k * lambda * lambda);
        }
        CHECK_THAT(post::kolmogorov_survival(lambda), WithinAbs(alt, 1e-12));
    }
    // Monotone decreasing.
    Real prev = 1;
    for (int i = 1; i <= 300; ++i) {
        const Real q = post::kolmogorov_survival(0.01 * i);
        CHECK(q <= prev);
        prev = q;
    }
    CHECK_THROWS_AS(post::kolmogorov_survival(-0.1), std::invalid_argument);
    CHECK_THROWS_AS(post::kolmogorov_survival(std::nan("")), std::invalid_argument);
}

TEST_CASE("speckle: KS test accepts fully developed speckle over ten seeds", "[speckle]") {
    // Independent circular complex Gaussian samples, n = 2000 per seed.
    std::vector<Real> p;
    int stephens_rejections = 0;
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
        const post::KsResult r =
            post::ks_test_exponential(intensity_of(circular_gaussian_field(2000, 1000 + seed)));
        INFO("seed " << seed << ": D = " << r.statistic << ", p = " << r.p_value
                     << ", D* = " << r.modified_statistic);
        CHECK(r.n == 2000);
        CHECK(r.p_value > 0.05);
        p.push_back(r.p_value);
        stephens_rejections += r.rejects(0.05) ? 1 : 0;
    }
    // The level-correct test rejects about 5 % of true exponential samples.
    CHECK(stephens_rejections <= 2);
    // Spread: the p-values are conservative (mean estimated from the sample, header) but not
    // degenerate.
    const auto [lo, hi] = std::minmax_element(p.begin(), p.end());
    INFO("p range " << *lo << " .. " << *hi);
    CHECK(*lo < 0.9);
    CHECK(std::count_if(p.begin(), p.end(), [](Real v) { return v > 0.9; }) < 8);
}

TEST_CASE("speckle: KS test on propagated speckle, decimated to independent pixels", "[speckle]") {
    // Square aperture: samples spaced by M / n_s pixels sit on the zeros of the field
    // correlation and are independent (speckle_support.hpp).
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const PropagatedSpeckle s =
            make_speckle(ApertureShape::Square, 500e-9, 1e-3, 40e-6, 256, 32, 2000 + seed);
        const VectorXr x = decimate(s.intensity(), 256 / 32);
        const post::KsResult r = post::ks_test_exponential(x);
        INFO("seed " << seed << ": n = " << r.n << ", D = " << r.statistic
                     << ", p = " << r.p_value);
        CHECK(r.n == 32 * 32);
        CHECK(r.p_value > 0.05);
        CHECK_FALSE(r.rejects(0.05));
        // A coherent background of twice the mean speckle intensity (C = 0.745): rejected.
        const MatrixXr ib = (s.field.array() + Complex(std::sqrt(2.0), 0)).abs2().matrix();
        const post::KsResult rb = post::ks_test_exponential(decimate(ib, 256 / 32));
        INFO("with background: D = " << rb.statistic << ", p = " << rb.p_value);
        CHECK(rb.p_value < 1e-3);
        CHECK(rb.rejects(0.05));
    }
}

TEST_CASE("speckle: KS test rejects partially developed and uniform samples", "[speckle]") {
    // Background ratio 1 (C = 0.866) and 2 (C = 0.745); at 0.5 (C = 0.943) the test has little
    // power at n = 2000 (measured p = 0.08: the conservative p-value, see the header).
    for (const Real ratio : {1.0, 2.0}) {
        const post::KsResult r =
            post::ks_test_exponential(speckle_with_background(2000, ratio, 3000));
        INFO("background ratio " << ratio << ": D = " << r.statistic << ", p = " << r.p_value);
        CHECK(r.p_value < 1e-3);
    }
    Rng rng(3001);
    VectorXr u(2000);
    for (Index i = 0; i < u.size(); ++i) u(i) = rng.uniform();
    const post::KsResult r = post::ks_test_exponential(u);
    INFO("uniform: D = " << r.statistic << ", p = " << r.p_value);
    CHECK(r.p_value < 1e-6);
}

TEST_CASE("speckle: Stephens critical values of the exponential KS test", "[speckle]") {
    static_assert(std::is_aggregate_v<post::KsResult>);
    const std::vector<std::pair<Real, Real>> table = {
        {0.15, 0.926}, {0.10, 0.990}, {0.05, 1.094}, {0.025, 1.190}, {0.01, 1.308}};
    for (const auto& [alpha, c] : table) {
        CHECK(post::stephens_exponential_critical_value(alpha) == c);
    }
    // Log-linear between the table points: the geometric mean of two levels gives the midpoint.
    CHECK_THAT(post::stephens_exponential_critical_value(std::sqrt(0.05 * 0.025)),
               WithinAbs(0.5 * (1.094 + 1.190), 1e-12));
    // Strictly decreasing in alpha over the whole range.
    Real prev = post::stephens_exponential_critical_value(0.01);
    for (int i = 1; i <= 140; ++i) {
        const Real alpha = 0.01 + 0.001 * i;
        const Real c = post::stephens_exponential_critical_value(std::min(alpha, 0.15));
        CHECK(c < prev);
        prev = c;
    }
    CHECK_THROWS_AS(post::stephens_exponential_critical_value(0.0099), std::invalid_argument);
    CHECK_THROWS_AS(post::stephens_exponential_critical_value(0.151), std::invalid_argument);
    CHECK_THROWS_AS(post::stephens_exponential_critical_value(std::nan("")), std::invalid_argument);
    // rejects() compares the modified statistic with the critical value.
    post::KsResult r{0.05, 0.5, 400, 1.1};
    CHECK(r.rejects(0.05));
    CHECK_FALSE(r.rejects(0.025));
    CHECK_THROWS_AS(r.rejects(0.2), std::invalid_argument);
    // The modified statistic of ks_test_exponential.
    const post::KsResult k =
        post::ks_test_exponential(intensity_of(circular_gaussian_field(300, 9)));
    const Real sn = std::sqrt(300.0);
    CHECK_THAT(k.modified_statistic,
               WithinRel((k.statistic - 0.2 / 300) * (sn + 0.26 + 0.5 / sn), 1e-14));
}

TEST_CASE("speckle: KS level on iid exponential samples, Stephens vs plain p", "[speckle]") {
    // n = 200 iid unit-exponential samples (|U|^2 of circular Gaussians), 2000 repetitions:
    // binomial standard deviation of a 5 % rate 0.005.
    const int reps = 2000;
    int stephens = 0;
    int plain = 0;
    for (int k = 0; k < reps; ++k) {
        const post::KsResult r = post::ks_test_exponential(
            intensity_of(circular_gaussian_field(200, 20000 + static_cast<std::uint64_t>(k))));
        stephens += r.rejects(0.05) ? 1 : 0;
        plain += r.p_value < 0.05 ? 1 : 0;
    }
    const Real rate_stephens = stephens / static_cast<Real>(reps);
    const Real rate_plain = plain / static_cast<Real>(reps);
    INFO("false-rejection rate at nominal 5 %: Stephens " << rate_stephens << ", plain p "
                                                          << rate_plain);
    CHECK(rate_stephens >= 0.025);
    CHECK(rate_stephens <= 0.08);
    CHECK(rate_plain < 0.02);  // the plain Kolmogorov p is conservative (header)
}

TEST_CASE("speckle: KS power against a coherent background of half the mean", "[speckle]") {
    // Background ratio 0.5 (C = 0.943), n = 2000, 400 repetitions.
    const int reps = 400;
    int stephens = 0;
    int plain = 0;
    for (int k = 0; k < reps; ++k) {
        const post::KsResult r = post::ks_test_exponential(
            speckle_with_background(2000, 0.5, 30000 + static_cast<std::uint64_t>(k)));
        stephens += r.rejects(0.05) ? 1 : 0;
        plain += r.p_value < 0.05 ? 1 : 0;
    }
    const Real power_stephens = stephens / static_cast<Real>(reps);
    const Real power_plain = plain / static_cast<Real>(reps);
    INFO("power at 5 %: Stephens " << power_stephens << ", plain p " << power_plain);
    CHECK(power_stephens >= 0.45);
    CHECK(power_stephens > power_plain);
}

TEST_CASE("speckle: KS test input checks", "[speckle]") {
    const VectorXr x49 = intensity_of(circular_gaussian_field(49, 7));
    CHECK_THROWS_AS(post::ks_test_exponential(x49), std::invalid_argument);
    CHECK_NOTHROW(post::ks_test_exponential(intensity_of(circular_gaussian_field(50, 7))));
    VectorXr bad = intensity_of(circular_gaussian_field(100, 7));
    bad(3) = -1;
    CHECK_THROWS_AS(post::ks_test_exponential(bad), std::invalid_argument);
    bad(3) = std::nan("");
    CHECK_THROWS_AS(post::ks_test_exponential(bad), std::invalid_argument);
    CHECK_THROWS_AS(post::ks_test_exponential(VectorXr::Zero(100)), std::invalid_argument);
}

TEST_CASE("speckle: contrast of fully developed speckle and with a coherent background",
          "[speckle]") {
    const Index n = 100000;
    const Real c = post::speckle_contrast(intensity_of(circular_gaussian_field(n, 4000)));
    INFO("fully developed C = " << c);
    CHECK_THAT(c, WithinAbs(1.0, 0.02));
    for (const Real ratio : {0.25, 1.0, 4.0}) {
        const Real cb = post::speckle_contrast(speckle_with_background(n, ratio, 4001));
        INFO("ratio " << ratio << ": C = " << cb << ", analytic "
                      << contrast_with_background(ratio));
        CHECK_THAT(cb, WithinAbs(contrast_with_background(ratio), 0.02));
    }
    CHECK_THAT(post::speckle_contrast(VectorXr::Constant(10, 3.0)), WithinAbs(0.0, 1e-15));
    CHECK_THROWS_AS(post::speckle_contrast(VectorXr::Constant(1, 1.0)), std::invalid_argument);
    CHECK_THROWS_AS(post::speckle_contrast(VectorXr::Zero(5)), std::invalid_argument);
}

TEST_CASE("speckle: correlation coefficient of mixed patterns", "[speckle]") {
    const Index n = 100000;
    for (const Real rho : {0.0, 0.3, 0.7, 1.0}) {
        const PatternPair p = correlated_patterns(n, rho, 5000);
        const Real r = post::correlation_coefficient(p.i1, p.i2);
        INFO("rho " << rho << ": estimate " << r);
        CHECK_THAT(r, WithinAbs(rho, 0.02));
    }
    // Symmetric, invariant under positive scaling and offsets.
    const PatternPair p = correlated_patterns(1000, 0.5, 5001);
    const Real r = post::correlation_coefficient(p.i1, p.i2);
    CHECK_THAT(post::correlation_coefficient(p.i2, p.i1), WithinAbs(r, 1e-14));
    const VectorXr scaled = (3.0 * p.i2.array() + 2.0).matrix();
    CHECK_THAT(post::correlation_coefficient(p.i1, scaled), WithinAbs(r, 1e-12));
    CHECK_THROWS_AS(post::correlation_coefficient(p.i1, p.i2.head(999)), std::invalid_argument);
    CHECK_THROWS_AS(post::correlation_coefficient(p.i1, VectorXr::Constant(1000, 1.0)),
                    std::invalid_argument);
    VectorXr negative = p.i2;
    negative(5) = -1e-3;
    CHECK_THROWS_AS(post::correlation_coefficient(p.i1, negative), std::invalid_argument);
    CHECK_THROWS_AS(post::correlation_coefficient(negative, p.i1), std::invalid_argument);
}

TEST_CASE("speckle: intensity histogram of fully developed speckle follows e^-x", "[speckle]") {
    const VectorXr x = intensity_of(circular_gaussian_field(100000, 6000));
    const post::Histogram h = post::intensity_pdf(x, 64);
    REQUIRE(h.bin_centers.size() == 64);
    const Real width = h.bin_centers(1) - h.bin_centers(0);
    CHECK_THAT(h.density.sum() * width, WithinAbs(1.0, 1e-12));
    CHECK_THAT(h.bin_centers(0), WithinRel(0.5 * width, 1e-12));
    const Real e = post::intensity_pdf_rms_error(h);
    INFO("rms deviation from e^-x: " << e);
    CHECK(e < 0.02);
    // Partially developed speckle (background = mean) is far from it.
    const Real eb = post::intensity_pdf_rms_error(
        post::intensity_pdf(speckle_with_background(100000, 1.0, 6001), 64));
    INFO("with background: " << eb);
    CHECK(eb > 5 * e);
    CHECK_THROWS_AS(post::intensity_pdf(x, 0), std::invalid_argument);
}

TEST_CASE("speckle: intensity autocorrelation matches the direct sum and its layout", "[speckle]") {
    Rng rng(7000);
    const std::vector<std::pair<Index, Index>> sizes = {{6, 5}, {7, 8}, {2, 2}};
    for (const auto& [rows, cols] : sizes) {
        MatrixXr im(rows, cols);
        for (Index j = 0; j < cols; ++j) {
            for (Index i = 0; i < rows; ++i) im(i, j) = rng.uniform();
        }
        const MatrixXr g = post::intensity_autocorrelation(im);
        const MatrixXr ref = autocorrelation_direct(im);
        REQUIRE(g.rows() == rows);
        REQUIRE(g.cols() == cols);
        CHECK((g - ref).cwiseAbs().maxCoeff() < 1e-12);
        // Zero lag: <I^2> / <I>^2 = 1 + C^2.
        const Real c = post::speckle_contrast(im.reshaped());
        CHECK_THAT(g(rows / 2, cols / 2), WithinAbs(1 + c * c, 1e-12));
    }
    CHECK((post::intensity_autocorrelation(MatrixXr::Constant(9, 4, 2.0)).array() - 1)
              .abs()
              .maxCoeff() < 1e-12);
    CHECK_THROWS_AS(post::intensity_autocorrelation(MatrixXr::Constant(1, 5, 1.0)),
                    std::invalid_argument);
}

TEST_CASE("speckle: Siegert relation and mu_A of the propagated square-aperture field",
          "[speckle]") {
    const Index grid = 256;
    const Index ns = 32;
    const PropagatedSpeckle s =
        make_speckle(ApertureShape::Square, 500e-9, 1e-3, 40e-6, grid, ns, 8000);
    const MatrixXr im = s.intensity();
    const MatrixXr gamma = post::intensity_autocorrelation(im);
    const MatrixXr mu = post::field_correlation_from_intensity(im);
    const MatrixXr mu_field = field_correlation_of(s.field);
    // Lags within two zero spacings (M / n_s = 8 pixels) of the origin.
    const Index c = grid / 2;
    const Index lag = 2 * grid / ns;
    Real siegert = 0;
    Real model = 0;
    Real inversion = 0;
    for (Index dj = -lag; dj <= lag; ++dj) {
        for (Index di = -lag; di <= lag; ++di) {
            const Real g = gamma(c + di, c + dj);
            const Real m2 = mu_field(c + di, c + dj) * mu_field(c + di, c + dj);
            siegert = std::max(siegert, std::abs(g - 1 - m2));
            const Real expected = dirichlet_mu(di, ns, grid) * dirichlet_mu(dj, ns, grid);
            model = std::max(
                model, std::abs(mu(c + di, c + dj) * mu(c + di, c + dj) - expected * expected));
            // field_correlation_from_intensity inverts the Siegert relation.
            inversion = std::max(inversion, std::abs(mu(c + di, c + dj) * mu(c + di, c + dj) -
                                                     std::max(g - 1, 0.0)));
        }
    }
    INFO("max |Gamma_I - 1 - |mu_field|^2| = " << siegert
                                               << ", max ||mu_A|^2 - model| = " << model);
    // Sampling noise: each lag estimates a correlation from ~(M / (M / n_s))^2 = 1024 speckles
    // (~1 / sqrt(1024) = 0.03), and the bound is the maximum over 17 x 17 lags. Measured 0.049
    // and 0.046 (seed 8000).
    CHECK(siegert < 0.1);
    CHECK(model < 0.1);
    CHECK(inversion < 1e-12);
}

TEST_CASE("speckle: speckle-size constants are the half-maximum points", "[speckle]") {
    const Real u = bisect(
        [](Real x) {
            const Real s = std::sin(kPi * x) / (kPi * x);
            return s * s - 0.5;
        },
        0.1, 0.9);
    CHECK_THAT(post::speckle_size_constant(ApertureShape::Square), WithinRel(2 * u, 1e-13));
    const Real v = bisect(
        [](Real x) {
            const Real a = 2 * bessel_j1(x) / x;
            return a * a - 0.5;
        },
        0.5, 3.0);
    CHECK_THAT(post::speckle_size_constant(ApertureShape::Circular), WithinRel(2 * v / kPi, 1e-13));
    CHECK_THAT(post::speckle_size_constant(ApertureShape::Gaussian),
               WithinRel(4 * std::sqrt(std::log(2.0)) / kPi, 1e-15));
    CHECK_THAT(post::speckle_size_expected(500e-9, 1e-3, 10e-6, ApertureShape::Square),
               WithinRel(0.8858929413789047 * 500e-9 * 1e-3 / 10e-6, 1e-13));
    CHECK_THROWS_AS(post::speckle_size_expected(0, 1e-3, 1e-5, ApertureShape::Square),
                    std::invalid_argument);
    CHECK_THROWS_AS(post::speckle_size_expected(5e-7, -1, 1e-5, ApertureShape::Square),
                    std::invalid_argument);
    CHECK_THROWS_AS(post::speckle_size_expected(5e-7, 1e-3, std::nan(""), ApertureShape::Square),
                    std::invalid_argument);
}

TEST_CASE("speckle: speckle_size interpolation on analytic correlations", "[speckle]") {
    // Gaussian |mu_A|^2: the log-linear interpolation is exact.
    const Index n = 41;
    const Real fwhm_px = 5.3;
    MatrixXr g(n, n + 2);
    for (Index j = 0; j < g.cols(); ++j) {
        for (Index i = 0; i < n; ++i) {
            const Real di = static_cast<Real>(i - n / 2);
            const Real dj = static_cast<Real>(j - g.cols() / 2);
            // |mu|^2 = 2^{-4 d^2 / FWHM^2}; axis 1 twice as wide.
            g(i, j) = std::sqrt(std::exp2(
                -4 * (di * di / (fwhm_px * fwhm_px) + dj * dj / (4 * fwhm_px * fwhm_px))));
        }
    }
    CHECK_THAT(post::speckle_size(g, 2e-6, 0), WithinRel(fwhm_px * 2e-6, 1e-12));
    CHECK_THAT(post::speckle_size(g, 2e-6, 1), WithinRel(2 * fwhm_px * 2e-6, 1e-12));
    // sinc^2: interpolation error < 1 % from 4 pixels per FWHM, < 2.5 % at 3 (header). At an even
    // integer number of pixels per FWHM the half maximum falls on a sample (exact by
    // construction), so the bound is checked over a sweep of [4, 8] in steps of 0.1 px.
    auto sinc_error = [](Real px_per_fwhm) {
        const Real a = 0.8858929413789047 / px_per_fwhm;  // sinc argument per pixel
        MatrixXr m(31, 3);
        for (Index i = 0; i < 31; ++i) {
            const Real x = a * static_cast<Real>(i - 15);
            const Real sv = i == 15 ? 1.0 : std::sin(kPi * x) / (kPi * x);
            m.row(i).setConstant(std::abs(sv));
        }
        return post::speckle_size(m, 1.0, 0) / px_per_fwhm - 1;
    };
    const Real e3 = sinc_error(3.0);
    INFO("3 px per FWHM: rel error " << e3);
    CHECK(std::abs(e3) < 0.025);
    CHECK(std::abs(e3) > 0.01);  // measured 1.9 %: the sweep below is not trivially exact
    Real worst = 0;
    for (int k = 0; k <= 40; ++k) {
        const Real px = 4.0 + 0.1 * k;
        const Real e = sinc_error(px);
        INFO(px << " px per FWHM: rel error " << e);
        CHECK(std::abs(e) < 0.01);
        worst = std::max(worst, std::abs(e));
    }
    INFO("max |rel error| over [4, 8] px per FWHM: " << worst);
    CHECK(worst > 1e-3);  // measured 0.76 % (not degenerate)
    CHECK_THROWS_AS(post::speckle_size(g, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(post::speckle_size(g, 1.0, 2), std::invalid_argument);
    CHECK_THROWS_AS(post::speckle_size(MatrixXr::Constant(5, 5, 1.0), 1.0), std::runtime_error);
    CHECK_THROWS_AS(post::speckle_size(MatrixXr::Zero(5, 5), 1.0), std::invalid_argument);
}

namespace {

/// Speckle size of propagated speckle vs speckle_size_expected for one aperture shape at two
/// (lambda, z, D): FWHM 7.1-8.5 pixels (case 1) and 4.7-5.7 pixels (case 2). Seeds 9000 + 1 +
/// shape_index + 3 case_index (the seeds of the former single test case over all shapes).
void check_size_scaling(ApertureShape shape, std::uint64_t shape_index) {
    struct Case {
        Real wavelength, z, aperture;
        Index grid, samples;
    };
    const Case cases[] = {{500e-9, 1e-3, 40e-6, 256, 32}, {633e-9, 2.5e-3, 120e-6, 256, 48}};
    std::uint64_t seed = 9001 + shape_index;
    for (const Case& k : cases) {
        const PropagatedSpeckle s =
            make_speckle(shape, k.wavelength, k.z, k.aperture, k.grid, k.samples, seed);
        seed += 3;
        const MatrixXr mu = post::field_correlation_from_intensity(s.intensity());
        const Real expected = post::speckle_size_expected(s.wavelength, s.z, s.aperture, s.shape);
        for (const int axis : {0, 1}) {
            const Real size = post::speckle_size(mu, s.pixel_size, axis);
            const Real e = size / expected - 1;
            INFO("lambda " << k.wavelength << " z " << k.z << " D " << s.aperture << " shape "
                           << static_cast<int>(shape) << " axis " << axis << ": size " << size
                           << " expected " << expected << " rel " << e);
            CHECK(std::abs(e) < 0.10);
        }
    }
}

}  // namespace

TEST_CASE("speckle: speckle size scales as lambda z / D (square aperture)", "[speckle]") {
    check_size_scaling(ApertureShape::Square, 0);
}

TEST_CASE("speckle: speckle size scales as lambda z / D (disc aperture)", "[speckle]") {
    check_size_scaling(ApertureShape::Circular, 1);
}

TEST_CASE("speckle: speckle size scales as lambda z / D (Gaussian spot)", "[speckle]") {
    check_size_scaling(ApertureShape::Gaussian, 2);
}
