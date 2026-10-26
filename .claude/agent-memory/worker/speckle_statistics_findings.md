---
name: speckle-statistics-findings
description: WP-S1 speckle harness - KS p-values conservative (estimated mean), Stephens level/power, power vs background, decimation, size constants, sampling noise, small-example cost limits (Si box walls dominate)
metadata:
  type: project
---

WP-S1 (2026-10-25), post::speckle_statistics + python/specklebem/speckle.py:
- speckle_statistics.cpp did NOT exist before WP-S1 (only the header and commented CMake lines);
  nothing was bound in Python although docs/10 listed sb.speckle_contrast etc. Check before
  assuming a brief's "exists".
- KS vs exp with the mean estimated: Kolmogorov p is conservative. n = 2000 iid, 10 seeds:
  p 0.51-0.98 (3 of 10 > 0.9). Power: background ratio 0.5 (C 0.94) NOT rejected (p 0.08), ratio 1
  (C 0.87) p 5e-7, ratio 2 p 7e-23. Speckle pixels must be decimated (>= ~2 FWHM, or the M/n_s
  Dirichlet zeros for a square pupil) or correlated samples give spurious rejections.
- Review fix (2026-10-26): the docs/05 gate is now KsResult::rejects(0.05), Stephens-modified
  D* = (D - 0.2/n)(sqrt n + 0.26 + 0.5/sqrt n) vs 1.094. Measured (fixed seeds): level n = 200,
  2000 reps: Stephens 4.6 %, plain p < 0.05 0.35 %; power ratio 0.5 n = 2000, 400 reps: Stephens
  57 %, plain 26 %. A member function keeps a struct an aggregate (std::is_aggregate_v holds).
- Constants FWHM(|mu_A|^2) = c lambda z / D: square 0.885893, disc 1.028994, Gaussian spot
  (D = 2 w0) 1.060041. mu_A is the FT of the pupil INTENSITY (van Cittert-Zernike), not the pupil
  autocorrelation (that is the intensity PSD) - a brief said otherwise.
- Size estimate on synthetic 256^2 fields with ~1000-2300 speckles: <= 2.7 % error (statistical);
  interpolation log|mu|^2 linear in d^2: exact for Gaussians, sinc^2 1.9 % at 3 px/FWHM, max
  0.76 % over [4, 8] px (peak near 4.7 px; exact at even integer px: half maximum on a sample).
  Siegert / model max deviation over 17x17 lags ~0.05 (noise ~1/sqrt(N_speckles)): bound 0.1.
- std::normal_distribution differs libstdc++ vs libc++: use mt19937_64 + own Box-Muller in tests
  that must match across the GCC (win-release) and Clang (win-debug) presets.
- Debug (ASan, -O0) kissfft is slow: 384^2 grids (768 pad) took 25 s; keep test grids at 256.
- Example cost (12-core desktop, idle): Si box depth is 5.65 um regardless of L, so walls
  dominate: L 3 um h 50 nm = 2N 205 920 (infeasible for 2 min); h 125 nm L 3 = 33 408 (122 s),
  L 2.5 = 26 880 (93 s, MLFMM, 38 its at tol 1e-3). sigma 100 nm at h 125 nm: MLFMM R2 d0 = 3
  not reachable (r_max/a 0.56) -> assemble() RuntimeError. At sigma 50 nm, D = 1 um the specular
  beam dominates (contrast ~12-13, coherent fraction 0.21); 20 speckles need a +-81 deg plane.

**Why:** WP-S2 runs the real sweep with this harness; these numbers size the ensembles and plane.
**How to apply:** pool seeds for the KS test, decimate, decide with rejects(0.05) (plain p is
conservative); size the plane with
observation_plane_size and keep it paraxial (needs D >~ 20 lambda for +-30 deg). See
[[si-large-area-findings]], [[mlfmm-local-leaf-findings]].
