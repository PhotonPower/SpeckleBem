---
name: pml-spp-findings
description: WP-PML4a validation 7b (SPP reflection at the PML start, flat Ag) - launch/fit method, M_y cleaner than J_x (SPP impedance check), fit noise at 9-10 samples, |r| vs sigma/W/m/h (defaults 8e-5, floor 1-2e-4), two reflection mechanisms, costs, refit workflow
metadata:
  type: project
---

WP-PML4a (2026-10-14/15), benchmarks/pml_spp.cpp, tests/support/pml_spp_support.hpp, record
benchmarks/results/pml_validation.md "7b SPP absorption" (7b met, defaults confirmed).

- **Method**: groove (d 50 nm, r_g 100 nm) under the rigorous beam, grooved minus flat solution
  (same grid, only z differs -> same RWG numbering); or a localized RHS (beam RHS kept only within
  150 nm of the origin, flat operator: one assembly). Both give the same |r| and phase (8.0e-5 /
  7.9e-5, arg 1.55 / 1.53 at defaults) -> use the local source for scans (half the cost).
- Fit out H1'(k rho) + in H1'(k(2a - rho)) + outgoing exp(-j k0 rho)(k0 rho)^-p, p in {1, 1.5, 2},
  Re k fitted (nuisance: Jx/My of one run differ 1-3 % in k). Without background terms the fit is
  model-limited (groove defaults |r| 4.2e-3 fake, residual 1.2-1.4e-2). Only 9-10 samples per
  half-axis at h = 100 nm in the 0.6..1.5 um window.
- **M_y is the clean observable**: weak PML (sigma 0.25, W lambda) |r| 1.7e-2 / 2.3e-2 with the
  start-referred phase invariant to 1e-3 rad under a start shift of h; |r(M_y)| e^A = 0.05/0.047/0.039
  for sigma 0.25/0.5/1 (A = 2 Re k_spp sigma W/(m+1): outer-edge reflection ~0.045 seen twice
  attenuated). J_x 7-10x larger there, k at the scan bound. A >= 6.6: Jx, My within 3x (W lambda
  sigma 1, A 4.4: 6.8x).
- **SPP impedance check** (review fix): clean SPP needs |out(My)/out(Jx)| = |E_x/H_y| = eta0 |kappa|/k0
  = 127 Ohm (kappa = 0.337 k0; ZOUT lines, pml_spp::out_impedance). Groove 111 Ohm; local source
  18-50 Ohm in every run -> local J_x "out" carries 2.5-7x non-SPP field, its r is not SPP-normalised.
  Base conclusions on M_y; keep r_spp = max(Jx, My) as the conservative acceptance value.
- **Fit noise at the real sampling** (9/10 samples): noise 1e-4 -> max |dr| 1.3e-4 / 3.8e-4 / 9.6e-4
  at r = 0 / 1e-3 / 1e-2 (10x the dense-grid value, grows with r via the k nuisance); noise-free
  ~1e-5. Groove vs local differ by only ~1e-6 in complex r: the 8e-5 is deterministic.
  One fit_spp costs ~0.3 s in win-debug (ASan, -O0) vs ~5 ms release: keep debug unit cases <~ 10 fits.
- **Floor** 1-2e-4 (refit spread over rho_min / bg sets / Im k). All W >= 2 lambda, m >= 2 runs
  (sigma 2..8, W 3 lambda) at the floor; defaults lambda/5 7.9e-5/3.5e-5, lambda/6 2.2e-5/2.5e-5,
  lambda/7 1.1e-5/2.8e-5.
- **Mechanisms**: (i) outer edge 0.045 e^-A dominates only for A <~ 5 (W lambda, sigma <= 1); the
  m = 3 / W lambda excess (3.2e-4 My vs 6e-5 estimate) is unexplained (steeper profile end?);
  (ii) discretised onset, grows with steepness at fixed h: W lambda sigma 3/4/8 -> My 2.6e-4/4.5e-4/
  2.1e-3; W lambda sigma 4 is 1.3e-3 at lambda/5 but 1.1e-4 at lambda/6 (cells across W matter).
  m = 1: 1.75e-3/1.19e-3 (~10x the floor; do not quote ratios to sub-floor values), identical for
  W lambda sigma 2 and W 2 lambda sigma 4 (same slope). W lambda sigma 0.25/0.5 exceed 1e-2 in J_x.
- Costs (shared machine, times vary up to 3x): 2N 9 440 asm 40-340 s (grows with sigma), 14 800
  420-735 s, 21 360 ~1 000 s, 29 120 (lambda/7) 1 022 s + LU 165 s, peak 13.7 GB.
- Refit workflow: --dump CSV then --refit CSV with other fit options (no solve); scratch scripts
  spread.sh/refit.sh pattern (loop over fit variations, min/max per quantity).

**Why:** 7c/7d (WP-PML4) and any later PML-profile question reuse this measurement and its limits.
**How to apply:** use M_y + local source for cheap PML-reflection checks; for W_pml = lambda check
the mesh (lambda/6) before trusting a sigma optimum; see [[pml-fresnel-findings]].
