"""Windowed plane-wave references of the WP-V3 face spectra (results/ag_box_mechanism.md).

NumPy re-implementation of ag_box_mechanism::radial_spectrum (tests/support/
ag_box_mechanism_support.hpp) with the Hann taper, sampled at the triangle centroids of a
uniform 50 nm two-triangle grid on a face of Lu x Lv (u along the face, v down the wall), for
single plane waves exp(-j k (cos(phi) u + sin(phi) v)) of |k| = 1.055 (Re k_spp of Ag at
500 nm), 1.0, 0.95 and 0.7 k0, phi = 0 ... 90 deg in 15 deg steps. Prints the range over phi of
the fraction above k0, within Re k_spp +- 0.05 k0, in 0.3-0.95 k0 and the centroid in
0.8-1.4 k0 (the columns of the record), per window (4 x 4 um faces, 4 x 2 um walls at D = 2 um).

Usage: python benchmarks/ag_box_mechanism_window.py
"""

from __future__ import annotations

import numpy as np

LAMBDA = 500e-9
K0 = 2.0 * np.pi / LAMBDA
H = 50e-9
EPS_AG = -9.794 - 0.313j
K_SPP = float(np.sqrt(EPS_AG / (1.0 + EPS_AG)).real)


def centroids(lu: float, lv: float) -> tuple[np.ndarray, np.ndarray]:
    """Centroids of the two triangles of every H x H cell of an lu x lv face."""
    nu, nv = int(round(lu / H)), int(round(lv / H))
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    u0 = -0.5 * lu + i * H
    v0 = j * H
    u = np.concatenate([(u0 + H / 3).ravel(), (u0 + 2 * H / 3).ravel()])
    v = np.concatenate([(v0 + H / 3).ravel(), (v0 + 2 * H / 3).ravel()])
    return u, v


def spectrum(u: np.ndarray, v: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    """(above k0, at k_spp +- 0.05, 0.3-0.95 k0, centroid 0.8-1.4 k0) of samples f."""
    w = (np.sin(np.pi * (u - u.min()) / (u.max() - u.min())) ** 2) * (
        np.sin(np.pi * (v - v.min()) / (v.max() - v.min())) ** 2
    )
    half = int(np.ceil(2.5 * 32 - 1e-9))
    ks = np.arange(-half, half + 1) * K0 / 32
    eu = np.exp(1j * np.outer(ks, u))
    ev = np.exp(1j * np.outer(ks, v))
    x = eu @ ((w * f * H * H / 2)[:, None] * ev.T)
    p = np.abs(x) ** 2
    kr = np.hypot(*np.meshgrid(ks, ks, indexing="ij")) / K0
    width = 0.025
    nb = int(np.ceil(np.sqrt(2) * half / 32 / width))
    b = np.minimum(nb - 1, (kr / width).astype(int))
    centre = (np.arange(nb) + 0.5) * width
    pb = np.bincount(b.ravel(), p.ravel(), nb) / p.sum()
    above = p[kr > 1.0].sum() / p.sum()
    near = pb[(centre >= K_SPP - 0.05) & (centre <= K_SPP + 0.05)].sum()
    prop = pb[(centre >= 0.3) & (centre <= 0.95)].sum()
    band = (centre >= 0.8) & (centre <= 1.4)
    cent = (pb[band] * centre[band]).sum() / pb[band].sum()
    return float(above), float(near), float(prop), float(cent)


def main() -> int:
    for lu, lv, name in [(4e-6, 4e-6, "4 x 4 um"), (4e-6, 2e-6, "4 x 2 um (wall, D = 2 um)")]:
        u, v = centroids(lu, lv)
        for k in (K_SPP, 1.0, 0.95, 0.7):
            rows = []
            for phi in np.deg2rad(np.arange(0, 91, 15)):
                ku, kv = k * K0 * np.cos(phi), k * K0 * np.sin(phi)
                rows.append(spectrum(u, v, np.exp(-1j * (ku * u + kv * v))))
            r = np.array(rows)
            print(
                f"window | {name} | k/k0 {k:.4f} | >k0 {100 * r[:, 0].min():.1f}-"
                f"{100 * r[:, 0].max():.1f} % | at k_spp+-0.05 {100 * r[:, 1].min():.1f}-"
                f"{100 * r[:, 1].max():.1f} % | 0.3-0.95 {100 * r[:, 2].min():.1f}-"
                f"{100 * r[:, 2].max():.1f} % | centroid {r[:, 3].min():.3f}-{r[:, 3].max():.3f}"
                f" | >k0 per phi 0..90 deg: {' '.join(f'{100 * x:.1f}' for x in r[:, 0])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
