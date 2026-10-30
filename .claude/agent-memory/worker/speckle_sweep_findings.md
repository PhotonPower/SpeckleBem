---
name: speckle-sweep-findings
description: WP-S2 Phase 6 speckle sweep (Si box route, L = 8 um, w0 = L/6) - costs, speckle-count limit 2 D / lambda, few-scatterer statistics ((w0/Lc)^2 = 1.8), analysis pipeline biases, final DoD numbers, gamma12 options
metadata:
  type: project
---

WP-S2 (2026-10-25/26; record benchmarks/results/speckle_sweep.md, module specklebem.speckle_sweep,
runs in <main checkout>/results/local_runs/wp_s2 with a frozen pkg/ copy):

- L = 10 um not attempted (WP22b3 Si L = 10 um peak 74.8 GB > 60 GB) -> L = 8 um, 2N = 741 120.
  Per case 27-50 min wall (100-180 it, 10-13 s/it), peak 46-56 GB; sweep 5.8 h.
- r_max at Lc = 1 um jumps the global leaf rule to 250 nm for sigma >= 200; leaf_radius_quantile
  0.99 keeps 125 nm for all.
- Speckles per axis <= 2 D / lambda (direction cosines): 10.7 at L = 8 um; inside the 5 % halo
  windows only 3.9-6.7 -> per-sigma KS at 2 speckle sizes has n = 8-32 (< 50): the docs/05 per-sigma
  gate is not testable; pooling over sigma (n = 90) is a fragile, non-independent variant.
- Physics trap: (w0/Lc)^2 = 1.8 correlation areas in the spot -> few-glint, over-dispersed speckle
  even with negligible coherent part: C_env 1.04-1.29, 1-speckle KS rejects; Kirchhoff phase-screen
  surrogate (C_env 1.18-1.32, gate rejection 64 %) agrees qualitatively. Size the spot w0 >> Lc.
- Final (after review fixes): row 42 NOT met as defined (docs/05 pools per sigma: n = 8-32 < 50).
  The sigma-pooled variant (n = 90) passes by 0.7 % (1.086 vs 1.094) only at the default analysis
  parameters: 12 of 20 one-at-a-time neighbours reject (`speckle_sweep_stats.py --sensitivity`),
  and without sigma = 100 nm (1.298). Pooling over sigma reuses the same scaled maps: correlated
  null (`calibration --map-correlation`) level 5.7-6.0 % vs 4.7 % (identical maps: 15.7 %).
  Lesson: a gate passing by < 1 % must get a sensitivity scan before it is called "met"; never let
  a ctest assert such a pass. Row-43 contrast 10.72 -> 2.15 monotone (largely the halo width:
  pure speckle gives 6.7 -> 2.2), row-44 size +2.8/+4.4/+11.4/+0.4 % (method scatter 8-15 %).
- Uniform-box size model: T = 4 n^2 + 8 n m (n = L/h, m = depth/h = 113 for Si 500 nm), 2N = 3T;
  exact at L = 8 and 25 um. Do not scale 2N with L^2 (walls scale with L).
- (w0/Lc)^2 convention: spot area pi w0^2 (1/e^2 radius); the effective area pi w0^2 / 2 halves it.
- Analysis biases (synthetic, real window sizes): self-including envelope C -10..-14 %, per-pattern
  1-speckle smoothing C -24 %, leave-one-out (mirror images + other seed) C -5..+2 %, gate level
  4.7 %; LOO makes independent-seed rho_env = -0.11 (not 0). At n ~ 850 (L = 25 um) LOO from two
  seeds is itself detected (matched-window null rejects 17 % at n ~ 820, 60 reps).
- gamma12 at normal incidence: no sigma trend (second order in dtheta; sigma steps exp(-(2k dsigma)^2)
  = 0.21 const, measured 0.05-0.54). Recommended: 0/30 deg pair, memory-effect aligned, from far
  field; 4 solves prepared (THETA=30, PKG override), not run.

**Why:** WP-S2b ctests and any follow-up speckle runs reuse these sizes, limits and biases.
**How to apply:** size speckle runs by 2 D / lambda AND (w0/Lc)^2 >> 1 first; calibrate any
envelope normalisation on synthetic speckle at the real window size; budget ~30-50 min per Si
L = 8 um case. See [[si-large-area-findings]], [[speckle-statistics-findings]].
