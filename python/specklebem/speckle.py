"""Speckle-ensemble statistics for the Phase 6 acceptance checks (WP-S1).

Harness for the rows "Speckle PDF at z = -1 mm" (level-correct exponential KS test does not
reject at 5 %), "Contrast, gamma_12 vs sigma" (monotone decrease) and "Speckle size"
(lambda z / D within 10 %) of docs/05_validation.md. A NumPy harness over the bound C++
statistics of the compiled module (``speckle_contrast``, ``ks_test_exponential``,
``correlation_coefficient``, ``field_correlation_from_intensity``, ``speckle_size``,
``speckle_size_expected``; docs/10_python_api.md section "Bound in WP-S1: speckle statistics").

Patterns are 2-D intensity arrays ``(nu, nv)`` on a regular observation grid, e.g. the
co-polarised intensity of ``Simulation.field`` on ``plane_grid(origin, u, v, nu, nv)`` reshaped
with ``I.reshape(nu, nv)``. SI units (metres).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ._specklebem import (
    KsResult,
    correlation_coefficient,
    field_correlation_from_intensity,
    ks_test_exponential,
    speckle_contrast,
    speckle_size,
    speckle_size_expected,
)

__all__ = [
    "decimate",
    "ensemble_statistics",
    "independent_stride",
    "monotone_decreasing",
    "observation_plane_size",
]


def _pixel_pair(pixel_size: float | Sequence[float]) -> tuple[float, float]:
    if np.ndim(pixel_size) == 0:
        p = float(pixel_size)
        pair = (p, p)
    else:
        if len(pixel_size) != 2:
            raise ValueError("pixel_size: expected a float or a pair (du, dv)")
        pair = (float(pixel_size[0]), float(pixel_size[1]))
    if not all(math.isfinite(p) and p > 0 for p in pair):
        raise ValueError("pixel_size: must be positive and finite")
    return pair


def independent_stride(size: float, pixel_size: float, factor: float = 2.0) -> int:
    """Pixel stride that makes speckle samples practically independent.

    Parameters
    ----------
    size : float
        Speckle size [m] (FWHM of |mu_A|^2, e.g. ``speckle_size_expected``).
    pixel_size : float
        Grid spacing [m].
    factor : float, optional
        Spacing in speckle sizes (default 2: for a Gaussian spot |mu_A|^2 = 2^-16 there).

    Returns
    -------
    int
        ``max(1, ceil(factor * size / pixel_size))``.
    """
    if not (size > 0 and pixel_size > 0 and factor > 0):
        raise ValueError("independent_stride: size, pixel_size and factor must be positive")
    return max(1, math.ceil(factor * size / pixel_size - 1e-9))


def decimate(intensity: np.ndarray, stride: int | Sequence[int]) -> np.ndarray:
    """Every ``stride``-th pixel per axis of a 2-D pattern (``stride`` int or (s_u, s_v))."""
    a = np.asarray(intensity, dtype=float)
    if a.ndim != 2:
        raise ValueError("decimate: expected a 2-D array (nu, nv)")
    su, sv = (stride, stride) if np.ndim(stride) == 0 else tuple(stride)
    if int(su) < 1 or int(sv) < 1:
        raise ValueError("decimate: stride must be >= 1")
    return a[:: int(su), :: int(sv)]


def observation_plane_size(
    wavelength: float,
    z: float,
    aperture: float,
    *,
    shape: str = "gaussian",
    n_speckles: float = 20.0,
) -> float:
    """Side [m] of an observation plane holding ``n_speckles`` speckle sizes per axis.

    ``n_speckles * speckle_size_expected(wavelength, z, aperture, shape)``; paraxial: the
    estimate is only meaningful while the plane half-width stays well below ``z``.
    """
    return n_speckles * speckle_size_expected(wavelength, z, aperture, shape)


def monotone_decreasing(values: Sequence[float], tolerance: float = 0.0) -> bool:
    """True if ``values[i + 1] <= values[i] + tolerance`` for every i (absolute tolerance >= 0).

    Used for the sigma sweep of the contrast and of the angular correlation gamma_12
    (docs/05: monotone decrease over sigma = 50-250 nm, two seeds each): pass the values ordered
    by increasing sigma (e.g. the seed mean per sigma) and a tolerance from the seed scatter.
    """
    if tolerance < 0:
        raise ValueError("monotone_decreasing: tolerance must be >= 0")
    v = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError("monotone_decreasing: values must be finite")
    return bool(np.all(np.diff(v) <= tolerance))


def ensemble_statistics(
    intensities: Sequence[np.ndarray],
    pixel_size: float | Sequence[float],
    wavelength: float,
    z: float,
    aperture: float,
    *,
    shape: str = "gaussian",
    ks_stride: int | Sequence[int] | None = None,
    ks_spacing: float = 2.0,
) -> dict:
    """Speckle statistics of an ensemble of patterns (e.g. one rough surface, several seeds).

    Parameters
    ----------
    intensities : sequence of ndarray, shape (nu, nv)
        Non-negative intensity patterns on the same regular grid (co-polarised intensity for
        fully developed polarised speckle).
    pixel_size : float or (float, float)
        Grid spacing [m] along u (axis 0) and v (axis 1).
    wavelength, z, aperture : float
        Wavelength [m], distance [m] and aperture size D [m] for ``speckle_size_expected``
        (for a Gaussian beam spot D = 2 w0).
    shape : {"gaussian", "square", "circular"}, optional
        Aperture shape of the prediction (default "gaussian").
    ks_stride : int or (int, int), optional
        Decimation stride for the KS test; default ``independent_stride`` of the expected
        speckle size per axis (``ks_spacing`` speckle sizes).
    ks_spacing : float, optional
        Sample spacing in speckle sizes for the default stride (2).

    Returns
    -------
    dict
        ``n_patterns``; ``contrast`` (per pattern, ndarray), ``mean_contrast``,
        ``std_contrast`` (sample std over patterns, 0 for one pattern); ``ks`` (KsResult of
        the pooled decimated normalised intensities I / <I>_pattern), ``ks_rejects_5pct``
        (bool, ``ks.rejects(0.05)``: the level-correct Stephens decision of docs/05; the plain
        ``ks.p_value`` is conservative), ``ks_stride``
        ((s_u, s_v)), ``ks_per_pattern`` (KsResult or None below 50 samples);
        ``correlation`` ((n, n) correlation coefficients, diagonal 1); ``speckle_size``
        ((n, 2) FWHM of |mu_A|^2 per pattern along u and v [m]), ``speckle_size_mean``,
        ``speckle_size_expected``, ``speckle_size_rel_error`` (mean / expected - 1).

    Raises
    ------
    ValueError
        For an empty ensemble, patterns of different shapes or non-2-D patterns, too few
        pooled KS samples (< 50), or invalid intensities (from the C++ functions).
    RuntimeError
        If a pattern is smaller than about one speckle (``speckle_size``).
    """
    patterns = [np.asarray(p, dtype=float) for p in intensities]
    if not patterns:
        raise ValueError("ensemble_statistics: empty ensemble")
    shape0 = patterns[0].shape
    if len(shape0) != 2 or any(p.shape != shape0 for p in patterns):
        raise ValueError("ensemble_statistics: patterns must be 2-D arrays of one shape")
    du, dv = _pixel_pair(pixel_size)
    expected = speckle_size_expected(wavelength, z, aperture, shape)
    if ks_stride is None:
        stride = (
            independent_stride(expected, du, ks_spacing),
            independent_stride(expected, dv, ks_spacing),
        )
    else:
        stride = (ks_stride, ks_stride) if np.ndim(ks_stride) == 0 else tuple(ks_stride)
        stride = (int(stride[0]), int(stride[1]))

    contrast = np.array([speckle_contrast(p) for p in patterns])
    pooled = []
    ks_each: list[KsResult | None] = []
    for p in patterns:
        d = decimate(p, stride).ravel()
        mean = d.mean()
        if not mean > 0:
            raise ValueError("ensemble_statistics: a decimated pattern has zero mean")
        pooled.append(d / mean)
        ks_each.append(ks_test_exponential(d) if d.size >= 50 else None)
    ks = ks_test_exponential(np.concatenate(pooled))

    n = len(patterns)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = correlation_coefficient(patterns[i], patterns[j])

    sizes = np.empty((n, 2))
    for i, p in enumerate(patterns):
        mu = field_correlation_from_intensity(p)
        sizes[i, 0] = speckle_size(mu, du, 0)
        sizes[i, 1] = speckle_size(mu, dv, 1)
    size_mean = float(sizes.mean())

    return {
        "n_patterns": n,
        "contrast": contrast,
        "mean_contrast": float(contrast.mean()),
        "std_contrast": float(contrast.std(ddof=1)) if n > 1 else 0.0,
        "ks": ks,
        "ks_rejects_5pct": bool(ks.rejects(0.05)),
        "ks_stride": stride,
        "ks_per_pattern": ks_each,
        "correlation": corr,
        "speckle_size": sizes,
        "speckle_size_mean": size_mean,
        "speckle_size_expected": expected,
        "speckle_size_rel_error": size_mean / expected - 1.0,
    }
