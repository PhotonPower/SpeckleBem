"""Phase 6 example: speckle from a rough Si surface, statistics with specklebem.speckle.

The runnable part solves a SMALL patch (Si, L = 2.5 um, sigma = 50 nm, Lc = 0.5 um,
h = 125 nm, uniform closing box, MLFMM; about 1.5-3 min on a 12-core desktop) and evaluates
the speckle harness of WP-S1 on the co-polarised intensity at z = -1 mm: contrast, the KS test
against the negative-exponential PDF and the speckle size against lambda z / D.

The acceptance gates of docs/05 (level-correct exponential KS test does not reject at 5 %,
contrast / gamma_12 monotone in sigma, speckle size within 10 % of lambda z / D) are NOT
asserted here and are not expected to hold on this case:
  * too few speckles: the illuminated spot is D = 2 w0 = 0.83 um (w0 = L / 6 for Si), so a
    speckle is c lambda z / D = 1.06 * 0.5 um * 1 mm / 0.83 um = 0.64 mm wide, and 20 speckles
    per axis need a 12.7 mm plane at z = 1 mm: observation angles up to 81 deg, far outside the
    paraxial model behind lambda z / D (the speckles stretch off axis);
  * at sigma = 50 nm the speckle is partially developed: the coherent (specular) fraction is
    exp(-(2 k sigma)^2) = 0.21 and the reflected beam dominates the plane centre (contrast >> 1);
  * one surface realisation, 121 independent samples (KS stride 10 on the 101 x 101 grid),
    h = lambda / 4 (not converged).
WP-S2 runs the sweep on large surfaces (sigma 50-250 nm, two seeds each) with the same calls.

Target sketch (the full Phase 6 case, 30 x 30 um, MLFMM on a large-memory machine; kept for
reference):

    import specklebem as sb
    L, sigma, Lc, h, wl = 30e-6, 100e-9, 1e-6, 50e-9, 500e-9
    si, vac = sb.silicon_500nm(), sb.vacuum()
    # Closing box from the materials (ADR 0006, amendments 2026-10-10 and 2026-10-12b): depth,
    # uniform box (box cells = top-face cells, the default) and the mandatory fine band
    # 3 delta + 3 sigma for Si.
    box = sb.rough_surface_box_params(si, vac, wl, sigma, h)
    surf = sb.RoughSurface(L=L, sigma=sigma, Lc=Lc, mesh_size=h, seed=42, **box.kwargs())
    waist = L * box.beam_waist_fraction   # 5 um: w0 <= L / 6 for Si (ADR 0006 2026-10-12b)
    sb.check_beam_waist(L, waist, max_fraction=box.beam_waist_fraction)  # ValueError if wider
    beam = sb.GaussianBeam(wavelength=wl, waist=waist, polarization="p")
    sim  = sb.Simulation(surf.mesh, beam, object=si, formulation="ICTF", compression="mlfmm")
    sim.solve(tol=1e-3)
    E = sim.field(sb.plane_grid(origin=[0, 0, -1e-3], u="x", v="y", size=300e-6, n=301))
    I = sb.intensity(E)
    print("contrast", sb.speckle_contrast(I))
    mu_A = sb.field_correlation_from_intensity(I)

GaussianBeam is the paraxial beam: it is fine for qualitative work, but it is 2.8 % off in
reflectance at w0 = 0.67 um (ADR 0006 amendment, decision 4). Quantitative rough-surface
results use the rigorous angular-spectrum Gaussian beam (sb.AngularSpectrumBeam, WP-E1; its
controlled ball must cover the whole box) in its place; this small example keeps the paraxial
beam because w0 = 0.42 um < lambda is below the rigorous beam's range (w0 >~ lambda).
"""

import math
import time

import numpy as np

import specklebem as sb
from specklebem import speckle

L, sigma, Lc, h, wl = 2.5e-6, 50e-9, 0.5e-6, 125e-9, 500e-9
z = 1e-3  # observation plane z = -1 mm
n = 101  # observation grid n x n

t0 = time.perf_counter()
si, vac = sb.silicon_500nm(), sb.vacuum()
box = sb.rough_surface_box_params(si, vac, wl, sigma, h)
surf = sb.RoughSurface(L=L, sigma=sigma, Lc=Lc, mesh_size=h, seed=1, **box.kwargs())
waist = L * box.beam_waist_fraction  # L / 6 for Si
sb.check_beam_waist(L, waist, max_fraction=box.beam_waist_fraction)
beam = sb.GaussianBeam(wavelength=wl, waist=waist, polarization="p")  # E along x at focus
sim = sb.Simulation(surf.mesh, beam, object=si, formulation="ICTF", compression="mlfmm")
print(f"2N = {sim.num_unknowns}, box depth {box.box_depth * 1e6:.2f} um, w0 = {waist * 1e6:.3f} um")
sim.assemble()
res = sim.solve(tol=1e-3)
print(f"GMRES: {res.iterations} iterations, converged {res.converged}")

# Observation plane sized for 20 speckles per axis from the paraxial prediction (Gaussian spot,
# D = 2 w0): side = 20 * 1.06 lambda z / D.
D = 2 * waist
side = speckle.observation_plane_size(wl, z, D, shape="gaussian", n_speckles=20)
expected = sb.speckle_size_expected(wl, z, D, "gaussian")
print(
    f"speckle size lambda z / D convention: {expected * 1e3:.3f} mm; plane {side * 1e3:.1f} mm"
    f" (max angle {math.degrees(math.atan(side / 2 / z)):.0f} deg)"
)
pts = sb.plane_grid([-side / 2, -side / 2, -z], [side, 0, 0], [0, side, 0], n, n)
E, _ = sim.field(pts, kind="scattered")
co, _ = sb.polarized_intensity(E, co_pol=[1, 0, 0])  # polarised speckle: co-polarised part
I = co.reshape(n, n)  # row i * n + j -> I[i, j], axis 0 = x

st = speckle.ensemble_statistics([I], side / (n - 1), wl, z, D, shape="gaussian")
print(f"contrast {st['mean_contrast']:.3f} (fully developed: 1)")
print(
    f"KS vs exp(-x): D = {st['ks'].statistic:.3f}, p = {st['ks'].p_value:.3g},"
    f" n = {st['ks'].n} (stride {st['ks_stride']});"
    f" Stephens D* = {st['ks'].modified_statistic:.3f}, rejects at 5 %: {st['ks_rejects_5pct']}"
)
sizes = st["speckle_size"][0]
print(
    f"speckle size (FWHM |mu_A|^2) x {sizes[0] * 1e3:.3f} mm, y {sizes[1] * 1e3:.3f} mm;"
    f" expected {st['speckle_size_expected'] * 1e3:.3f} mm"
    f" (rel {st['speckle_size_rel_error']:+.2f})"
)
print(f"coherent fraction exp(-(2 k sigma)^2) = {math.exp(-((4 * math.pi * sigma / wl) ** 2)):.2f}")
print(f"mean |E|^2 on the plane {np.mean(sb.intensity(E)):.3g} V^2/m^2")
print(f"total {time.perf_counter() - t0:.0f} s")
