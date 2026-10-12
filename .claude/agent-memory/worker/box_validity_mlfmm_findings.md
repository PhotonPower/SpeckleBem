---
name: box-validity-mlfmm-findings
description: WP-V2 MLFMM box-validity results - Si passes with uniform box at w0 <= L/6 (narrow evidence), Ag fails (cause open: SPPs vs WP-A2 box cavity resonances), noise floors, costs, memory estimates
metadata:
  type: project
---

WP-V2 (2026-10-11/12, benchmarks/box_validity.cpp --compression mlfmm, record
benchmarks/results/box_validity_mlfmm.md; reviewed and reworded 2026-10-12 after WP-A2):
- Noise floor of the docs/05 L2 metric: d0 5 vs 3 = 2e-5 (Ag, Si); Ag tol 1e-4 vs 1e-5 1.8e-4, 1e-3
  1.3e-3; Si tol 1e-4 vs 1e-5 4.7e-4 (!), 1e-5 vs 1e-6 3.7e-5 -> Si needs tol 1e-5. The floor is
  per configuration (tolerance-to-error mapping is case-dependent, WP22c); a pair's noise ~sqrt(2)x.
  Ag tol 1e-4 only for screening; within ~3x of a threshold use tol 1e-5 or a measured floor.
- Si (uniform box): depth x2 0.03 % at w0/L = 1/4 (same at w0 0.75 and 1 um), 0.007 % at 1/6; L x 1.2
  1.28 % at 1/4, 0.19 % at 1/6. Graded 100 nm box at 1/4: 0.085 % from uniform, depth x2 0.14 %
  (fails; 1/6 untested). Evidence: seed 1, sigma 50 nm, Lc 500 nm, normal, x pol, ONE L x 1.2 pair.
- Ag (L = 4 um): depth x2 0.82 % uniform / 0.91 % graded at 1/4, 1.89 % at 1/8 (rim dark!);
  L x 1.2 4.5 / 8.2 %. Im eps = -6 diagnostic: depth 0.093 % (within noise of 0.1 %), 251 vs 1305
  GMRES its. Review lesson: the loss diagnostic does NOT isolate SPPs - it also raises Re Zs/|Zs|
  1.6 % -> 27 % (damps the WP-A2 box cavity resonances) and changes P_refl -25 %; PARTS are
  contaminated by non-radiating cavity components and looked resonance-like (uniform bottom 6x drop
  with depth, graded rises). Cause open; discriminate with strip_spectrum (SPP |k_t| 1.055 k0 vs
  cavity < k0), extinction_residual, fine D sweep, harmonic Ritz lossless vs lossy.
  Rim illumination by the incident beam is falsified as the control for Ag.
- Costs: Ag graded box tol 1e-5 3 746 it (1 h at 2N 7e4); Ag UNIFORM box converges in ~0.5x the
  iterations of the graded box (431 vs 1176 at 1e-3, 2096 vs 3746 at 1e-5) and solves faster despite
  2.2x unknowns. Si uniform ~200-220 it at 1e-5, graded 322-340.
- Peak model (mlfmm_scaling estimate) is ~20-30 % high; Si uniform L = 4.5 depth 11.3 um: 48 GB real
  (58 est), Si d0 = 5 L = 3: 46 GB (59 est). Far field + PARTS on 5e5 unknowns ~25 min.
  The rows file lacks SETUP/ESTIMATE lines and --max-iter: record them when a value is quoted.
- ctest without --preset (plain --test-dir) hangs on Windows (DLL dialog); always use the preset.

**Why:** evidence base for ADR 0006 amendment (uniform box, Si w0 <= L/6; metals: SPP termination
only if SPP-dominated, else resonance-free formulation).
**How to apply:** box/beam rule changes and any rerun of the rough-surface checks; for metals do not
expect w0/L to fix box sensitivity, and do not attribute Ag box effects to SPPs without a
k-spectrum or interior-residual check. See [[box-validity-findings]], [[mlfmm-scaling-findings]],
[[ag-null-mode-findings]].
