---
name: ag-null-mode-findings
description: WP-A2 verdict on the Ag near-null mode - intrinsic non-radiating cavity mode of the truncated box (vacuum standing wave top-bottom), d0=5 unchanged, L/D resonance table, extinction-residual detector, run costs
metadata:
  type: project
---

Measured 2026-10-11/12 (WP-A2, benchmarks/ag_absorption.cpp, record benchmarks/results/ag_absorption.md
section "WP-A2"), Ag 45 deg s, h = 50 nm, w0 = L/5, ICTF + Jacobi, MLFMM d0 = 3, tol 1e-5, D = 2 um:
- Mode = non-radiating damped cavity mode of the box: extinction residual E_inc + E_1 inside the box
  is an E_y vacuum standing wave (kx = +-0.70 k0, kz ~ 0.73 k0, nodes at top face and bottom plate,
  depth-uniform amplitude 2.5 %); no retro far-field lobe.
- Intrinsic (deciding run): d0 = 5 at L = 2.5 um leaves A_local (-52.7 -> -52.6 %) and MODE
  unchanged, |x3 - x5|/|x| = 1.9e-4. d0 = 5 NOT run at L = 4 um (~21 GB).
- E2 (random RHS 3e-5, modal gain <= 0.3) is supporting evidence only: a random db overlaps one
  mode by ~1/sqrt(2N) (1/180 at 2N = 32 250), so it rules out spectrally flat errors, not a
  structured error of the same norm. The L = 4 um dx solve did not converge (3500 it, 9.3e-4).
- "Intrinsic" scope (review): homogeneous PMCHWT <-> complementary problem (vacuum inside box, Ag
  outside), uniquely solvable for lossy Ag -> no exact null space, but high-Q resonances. Open:
  coarse lambda/5 box mesh, RHS quadrature, RT0-error-as-excitation (not proven).
- A_local/A_beam - 1: L 2.0 -0.1, 2.25 -1.7, 2.5 -53, 2.75 -0.7, 3.0 -2.1, 3.25 -5.5, 3.5 -1.5,
  4.0 -43 %; L = 3 depth 1.5 +0.1, 1.8 -4.5, 2.0 -2.1, 2.1 -34, 2.5 -0.4 %. R, A_ext unaffected.
- Detector: rms(E_inc + E_1)/rms(E_inc) on y = 0, z = 150 nm: floor 0.17-0.63 %; 0.87-0.90 %
  <-> -4.5..-5.5 % (moderate); 1.6 % <-> -34 %, 2.7 % <-> -43..-53 %. Proposed levels > 1 % strong,
  0.7-1 % moderate; calibrated on flat 45 deg s h = 50 nm only; probe at z = max xi + >= 3h inside R2
  (extinction_residual throws outside R2). Not in the library until a rough-surface calibration.
- Harmonic Ritz (Jacobi ICTF): band of 8-12 |theta| = 0.016-0.1 at every L. Hessenberg of the
  MLFMM operator, non-invariant Krylov space: cannot exclude an isolated eigenvalue nor separate
  intrinsic from MLFMM; sigma_min(H_bar) is only an upper bound.
- Test cost: the extinction-residual validation test needs icosphere subdivision 2 (max residual
  1.4 %); subdivision 1 gives 9 % and fails the 5 % bound.
- Costs: Eigen ComplexEigenSolver + BDCSVD on k = 2757 took 24 min serial; numpy eigvals even
  slower (> 30 min) -> --no-ritz. Random-RHS GMRES needs ~1.5x the iterations (3900 at L = 2.5).
  d0 = 5 at L = 2.5 um: peak RSS 7.9 GB (setup estimate 13 GB) - check SETUP before launching.
- A tool command killed at 2 h leaves the benchmark exe running as an orphan (it finishes and
  writes its log); count it as a running case.

**Why:** decides that A_local is unusable for Ag on truncated boxes and that MLFMM accuracy is not
the lever. **How to apply:** use A_ext for Ag absorption; check the interior extinction residual
before trusting near-surface quantities; don't try to fix it with d0. See [[ag-absorption-findings]].
