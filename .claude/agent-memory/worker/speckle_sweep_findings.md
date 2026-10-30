---
name: speckle-sweep-findings
description: WP-S2/S2b Phase 6 speckle sweep (Si box route, L = 8 um, w0 = L/6) - lead decision 2026-10-30, server recipe location, costs, speckle-count limit 2 D / lambda, few-scatterer statistics ((w0/Lc)^2 = 1.8), analysis pipeline biases, final DoD numbers, gamma12 options
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

- LEAD DECISION 2026-10-30 (WP-S2b): no more desktop solves; L = 8 um data = Phase 6 record.
  Row 42 "not met with this configuration - physical", open until a 1 TB server run; docs/05 row
  42 stays per sigma pooled over seeds (n >= 50 at 2 speckle sizes, rejects(0.05)), sigma-pooled
  numbers never asserted; gamma12 deferred (0/30 deg pair on the server). DoD test
  tests/python/test_speckle_dod.py (SBEM_SPECKLE_SWEEP_DIR; record tests pin the numbers, gate
  tests apply docs/05 to any other dir). Server commands: record section "Server recipe (to be run
  by hand)" (queue + LC override; series lc05_L8 16 cases, t30_L8 5, L25 10 optional).
- Per-sigma n at 2 speckle sizes is 8 / 8 / 18 / 32 / 32 for sigma 50..250 (the record text once
  said "8, 8, 18, 32 for 100-250": wrong; 8+18+32+32 = 90 = the pooled n).
- WP-S2b smoke found build_simulation broken in the committed driver: wave_impedance is complex
  in the binding, float() raised TypeError (runs used an older frozen pkg). Always smoke-test the
  committed driver, not only the frozen copy.
- Lc = 0.5 um at L = 8 um (mesh only): r_0.99 60-68 nm at sigma 50/100, 72.6-98.5 nm at sigma >=
  150 (r_max to 173 nm) -> 250 nm leaves with q = 0.99 for most of those cases.
- Per-sigma gate at 5 % on k sigma: family-wise false rejection ~ 1 - 0.95^k (synthetic 4-seed
  exact speckle rejected once at 1.098).
- The L = 25 um "17 %" matched-null rejection is the sigma-POOLED sample (n ~ 820), not per sigma;
  the per-sigma level at L = 25 um is uncalibrated (reviewers may quote it as per-sigma).
- Stats-script JSON key is sigma_pooled_ks (was gate_ks until WP-S2b review); the docs/05 gate
  is row42_per_sigma. Plain (JSON) mode prints the per-sigma lines to stderr to keep stdout JSON.

**Why:** WP-S2b ctests and any follow-up speckle runs reuse these sizes, limits and biases.
**How to apply:** size speckle runs by 2 D / lambda AND (w0/Lc)^2 >> 1 first; calibrate any
envelope normalisation on synthetic speckle at the real window size; budget ~30-50 min per Si
L = 8 um case. See [[si-large-area-findings]], [[speckle-statistics-findings]].
