"""Speckle statistics bindings and the ensemble harness (WP-S1) on synthetic speckle.

Mirrors the key checks of tests/unit/test_speckle_statistics.cpp with NumPy-generated input
(fixed seeds, small sizes): circular complex Gaussian samples and the Fraunhofer field of a
random-phase aperture (one FFT; pixel lambda z / (M d_xi), D = n_s d_xi).
"""

import math

import numpy as np
import pytest

import specklebem as sb
from specklebem import speckle


def circular_gaussian(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def propagated_speckle(rng, grid, samples_across, shape="gaussian"):
    """Unit-mean Fraunhofer speckle field of a random-phase aperture (n_s samples across D)."""
    c = np.arange(grid) - grid // 2 + 0.5
    xi, eta = np.meshgrid(c, c, indexing="ij")
    half = samples_across / 2
    if shape == "square":
        p2 = ((np.abs(xi) < half) & (np.abs(eta) < half)).astype(float)
    else:
        p2 = np.exp(-2 * (xi**2 + eta**2) / half**2)  # w0 = D / 2
    a = np.sqrt(p2) * np.exp(2j * np.pi * rng.uniform(size=(grid, grid)))
    return np.fft.fft2(a) / math.sqrt(p2.sum())


WL, Z, D = 500e-9, 1e-3, 40e-6


def geometry(grid, samples_across):
    return WL * Z * samples_across / (grid * D)  # pixel size [m]


def test_ks_test_accepts_exponential_and_rejects_others():
    p_values = []
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        r = sb.ks_test_exponential(np.abs(circular_gaussian(rng, 2000)) ** 2)
        assert r.n == 2000
        assert r.p_value > 0.05
        p_values.append(r.p_value)
    assert min(p_values) < 0.9
    statistic, p, n = r  # unpacks as a tuple
    assert (statistic, p, n) == (r.statistic, r.p_value, r.n)
    assert "KsResult(" in repr(r)
    rng = np.random.default_rng(10)
    assert sb.ks_test_exponential(rng.uniform(size=2000)).p_value < 1e-6
    background = np.abs(circular_gaussian(rng, 2000) + math.sqrt(2.0)) ** 2
    assert sb.ks_test_exponential(background).p_value < 1e-3
    with pytest.raises(ValueError):
        sb.ks_test_exponential(np.ones(49))
    with pytest.raises(ValueError):
        sb.ks_test_exponential(-np.ones(100))
    assert sb.kolmogorov_survival(1.3581) == pytest.approx(0.05, abs=1e-4)


def test_ks_stephens_level_correct_decision():
    # Table points of the Stephens critical values, log-linear in between, range checked.
    for alpha, c in [(0.15, 0.926), (0.10, 0.990), (0.05, 1.094), (0.025, 1.190), (0.01, 1.308)]:
        assert sb.stephens_exponential_critical_value(alpha) == c
    assert sb.stephens_exponential_critical_value(math.sqrt(0.05 * 0.025)) == pytest.approx(
        0.5 * (1.094 + 1.190), abs=1e-12
    )
    with pytest.raises(ValueError):
        sb.stephens_exponential_critical_value(0.2)
    n = 2000
    rng = np.random.default_rng(70)
    r = sb.ks_test_exponential(np.abs(circular_gaussian(rng, n)) ** 2)
    sn = math.sqrt(n)
    assert r.modified_statistic == pytest.approx(
        (r.statistic - 0.2 / n) * (sn + 0.26 + 0.5 / sn), rel=1e-14
    )
    assert "modified_statistic=" in repr(r)
    assert not r.rejects()  # fully developed: not rejected at 5 %
    assert r.rejects(0.15) == (r.modified_statistic > 0.926)
    with pytest.raises(ValueError):
        r.rejects(0.5)
    # A coherent background equal to the mean speckle intensity (C = 0.866): rejected.
    background = np.abs(circular_gaussian(rng, n) + 1.0) ** 2
    assert sb.ks_test_exponential(background).rejects(0.05)


def test_contrast_correlation_and_histogram():
    rng = np.random.default_rng(20)
    u1 = circular_gaussian(rng, (200, 250))  # 2-D input is flattened
    u2 = circular_gaussian(rng, (200, 250))
    i1 = np.abs(u1) ** 2
    assert sb.speckle_contrast(i1) == pytest.approx(1.0, abs=0.02)
    ratio = 1.0
    ib = np.abs(u1 + math.sqrt(ratio)) ** 2
    analytic = math.sqrt(1 + 2 * ratio) / (1 + ratio)
    assert sb.speckle_contrast(ib) == pytest.approx(analytic, abs=0.02)
    rho = 0.6
    i2 = np.abs(math.sqrt(rho) * u1 + math.sqrt(1 - rho) * u2) ** 2
    assert sb.correlation_coefficient(i1, i2) == pytest.approx(rho, abs=0.02)
    with pytest.raises(ValueError):
        sb.correlation_coefficient(i1, i2[:10])
    centers, density = sb.intensity_pdf(i1, bins=32)
    assert centers.shape == density.shape == (32,)
    assert density.sum() * (centers[1] - centers[0]) == pytest.approx(1.0, abs=1e-12)
    assert sb.intensity_pdf_rms_error(centers, density) < 0.02


def test_autocorrelation_layout_and_siegert():
    rng = np.random.default_rng(30)
    grid, ns = 128, 16
    field = propagated_speckle(rng, grid, ns, "square")
    intensity = np.abs(field) ** 2
    gamma = sb.intensity_autocorrelation(intensity)
    mu = sb.field_correlation_from_intensity(intensity)
    assert gamma.shape == mu.shape == (grid, grid)
    c = grid // 2
    assert gamma[c, c] == pytest.approx(1 + sb.speckle_contrast(intensity) ** 2, rel=1e-12)
    assert np.allclose(mu**2, np.maximum(gamma - 1, 0), atol=1e-12)
    # Zero of the square-aperture field correlation at M / n_s = 8 pixels.
    assert mu[c + 8, c] ** 2 < 0.1
    assert mu[c + 2, c] ** 2 > 0.5
    # Non-square input keeps its orientation: lag along axis 0 = rows.
    rect = intensity[:, :64]
    assert sb.intensity_autocorrelation(rect).shape == (grid, 64)
    with pytest.raises(ValueError):
        sb.intensity_autocorrelation(intensity.ravel())


def test_speckle_size_expected_and_conventions():
    assert sb.speckle_size_constant("square") == pytest.approx(0.885893, abs=1e-6)
    assert sb.speckle_size_constant("circular") == pytest.approx(1.028994, abs=1e-6)
    assert sb.speckle_size_constant(sb.ApertureShape.Gaussian) == pytest.approx(
        4 * math.sqrt(math.log(2)) / math.pi, rel=1e-14
    )
    assert sb.speckle_size_expected(WL, Z, D) == pytest.approx(0.885893 * WL * Z / D, rel=1e-6)
    with pytest.raises(ValueError):
        sb.speckle_size_expected(WL, Z, D, "triangle")
    with pytest.raises(ValueError):
        sb.speckle_size_expected(WL, -Z, D)
    # Gaussian |mu_A|^2 sampled exactly: the interpolation is exact.
    lags = np.arange(41) - 20
    fwhm = 5.5
    mu = np.sqrt(np.exp2(-4 * (lags[:, None] ** 2 + lags[None, :] ** 2) / fwhm**2))
    assert sb.speckle_size(mu, 1e-6) == pytest.approx(fwhm * 1e-6, rel=1e-12)
    assert sb.speckle_size(mu, 1e-6, axis=1) == pytest.approx(fwhm * 1e-6, rel=1e-12)
    with pytest.raises(RuntimeError):
        sb.speckle_size(np.ones((5, 5)), 1e-6)


@pytest.mark.parametrize("shape", ["square", "gaussian"])
def test_speckle_size_scales_with_lambda_z_over_d(shape):
    rng = np.random.default_rng(40)
    grid, ns = 256, 32
    intensity = np.abs(propagated_speckle(rng, grid, ns, shape)) ** 2
    mu = sb.field_correlation_from_intensity(intensity)
    pixel = geometry(grid, ns)
    expected = sb.speckle_size_expected(WL, Z, D, shape)
    for axis in (0, 1):
        assert sb.speckle_size(mu, pixel, axis) / expected - 1 == pytest.approx(0, abs=0.1)


def test_ensemble_statistics_on_synthetic_speckle():
    grid, ns = 192, 24
    pixel = geometry(grid, ns)
    patterns = [
        np.abs(propagated_speckle(np.random.default_rng(50 + s), grid, ns)) ** 2 for s in range(4)
    ]
    st = speckle.ensemble_statistics(patterns, pixel, WL, Z, D, shape="gaussian")
    assert st["n_patterns"] == 4
    assert st["contrast"].shape == (4,)
    assert st["mean_contrast"] == pytest.approx(1.0, abs=0.08)
    # Default stride: two expected speckle sizes (FWHM = 1.06 M / n_s = 8.5 pixels).
    assert st["ks_stride"] == (17, 17)
    assert st["ks"].n == 4 * math.ceil(grid / 17) ** 2
    assert st["ks"].p_value > 0.05
    assert st["ks_rejects_5pct"] is False  # fully developed: level-correct test does not reject
    assert st["ks_rejects_5pct"] == st["ks"].rejects(0.05)
    off = st["correlation"][~np.eye(4, dtype=bool)]
    assert np.all(np.abs(off) < 0.1)
    assert np.allclose(np.diag(st["correlation"]), 1)
    assert st["speckle_size"].shape == (4, 2)
    assert abs(st["speckle_size_rel_error"]) < 0.1
    # A coherent background of twice the mean intensity: partially developed, KS rejects.
    rng = np.random.default_rng(60)
    partial = [np.abs(propagated_speckle(rng, grid, ns) + math.sqrt(2.0)) ** 2 for _ in range(4)]
    sp = speckle.ensemble_statistics(partial, pixel, WL, Z, D)
    assert sp["mean_contrast"] < 0.85
    assert sp["ks"].p_value < 1e-3
    assert sp["ks_rejects_5pct"] is True
    # Background equal to the mean intensity (C = 0.866): rejected by the level-correct test.
    rng = np.random.default_rng(61)
    ratio1 = [np.abs(propagated_speckle(rng, grid, ns) + 1.0) ** 2 for _ in range(4)]
    s1 = speckle.ensemble_statistics(ratio1, pixel, WL, Z, D)
    assert s1["ks_rejects_5pct"] is True
    with pytest.raises(ValueError):
        speckle.ensemble_statistics([], pixel, WL, Z, D)
    with pytest.raises(ValueError):
        speckle.ensemble_statistics([patterns[0], patterns[1][:, :100]], pixel, WL, Z, D)


def test_helpers():
    assert speckle.monotone_decreasing([1.0, 0.8, 0.8, 0.3])
    assert not speckle.monotone_decreasing([1.0, 0.8, 0.85])
    assert speckle.monotone_decreasing([1.0, 0.8, 0.85], tolerance=0.06)
    with pytest.raises(ValueError):
        speckle.monotone_decreasing([1.0, 0.5], tolerance=-1)
    assert speckle.independent_stride(10e-6, 2e-6) == 10
    assert speckle.independent_stride(1e-6, 2e-6) == 1
    assert speckle.decimate(np.zeros((10, 7)), (3, 2)).shape == (4, 4)
    size = speckle.observation_plane_size(WL, Z, 1e-6, shape="gaussian", n_speckles=20)
    assert size == pytest.approx(20 * 1.060041 * WL * Z / 1e-6, rel=1e-6)
