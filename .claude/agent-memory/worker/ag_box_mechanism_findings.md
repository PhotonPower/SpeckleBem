---
name: ag-box-mechanism-findings
description: WP-V3 verdict - Ag box sensitivity is surface waves (most likely SPPs) on the finite block, not box cavity resonances (rough box); flat box has one resonant depth; d0=5 leaves PMCHWT plateaus unchanged; numbers, pitfalls
metadata:
  type: project
---

Measured 2026-10-13 (WP-V3 + review round, benchmarks/box_validity.cpp --mechanism/--ritz/
--load-currents --diff/--depth-fit, record benchmarks/results/ag_box_mechanism.md), Ag L = 4 um,
uniform box, w0 = 1 um, MLFMM d0 3, ICTF+Jacobi, tol 1e-5:
- Wall/bottom currents SPP-like: 66-80 % spectral power above k0 (4x4 um faces 75-80 %, 4x2 um walls
  at D = 2 um 72 %, flat 66 %), centroid 1.06 k0. Window references (ag_box_mechanism_window.py):
  4x4 um SPP 77-81 % / k0 wave 50-56 %; 4x2 um SPP 66-83 % / k0 wave 51-59 % -> the short walls
  separate SPP from grazing light only marginally. Flat box: wall currents 6.5x weaker.
- Far-field change vs D: 0.4-1.2 % for every step 0.05-0.4 um, no on/off, no odd depth (vs D = 4 all
  0.74-0.82 %). Travelling-wave fit explains 90-99 %; per-band k_w = beta - k0 cos(theta) falls
  1.074 -> 0.958 k0 (0.96-1.07, mean 1.005 +- 0.045); free fit beta = a + b k0 cos: a = 0.93, b = 1.11
  -> leans to k_w ~ k0, not Re k_spp = 1.055.
- FLAT box: D = 2.0 um is resonant (interior residual 0.19 % vs 0.04-0.05 % at 2.1/4.0) and the far
  field follows: 2.1 vs 4.0 = 0.034 %, both vs 2.0 = 0.24 %. Always compare all pairs, not only
  against one reference depth, before calling a change "depth effect".
- L x1.2 change: rim artefact (hypothesis) and real annulus radiation of SPPs not separated; SPPs
  (decay 41.5 um) radiate off roughness far outside the patch, so no local termination at L << 41.5 um
  approximates the semi-infinite surface; an absorbing frame/PML defines a modified problem.
- Depth-difference far-field pattern: corr 0.93 graded vs uniform box (1.24 vs 1.27 %), 0.9999 tol
  1e-4 vs 1e-5; noise patterns <= 0.16.
- Ritz band (|theta| < 0.1, 10 values at k = 1000) identical for D 2-4 um and L 4.8. Lossy runs converge
  (k = 341/364) before such a band forms; at equal k = 250 lossy ~ lossless (0.10 vs 0.087):
  the loss comparison is inconclusive. Ritz at k = 1000 costs ~5 s (Eigen).
- (v) PMCHWT Ag 3.2 um d0 = 5: residual history = d0 3 to 0.15 % (2126 vs 2127); block check of the
  finest far level 5.0e-4 (d0 5, a = 0.8 lambda) vs 5.2e-3 (d0 3, a = 0.4 lambda; 1.2e-3 at 0.8
  lambda): 2-10x, not the nominal two orders.
- Depth averaging: disjoint groups 0.23-0.30 % (~3x); a group with an overlapping depth looked
  better (0.18 %) - keep groups disjoint.
- WP-V2's "6x bottom PARTS drop" was a tol 1e-3 artefact.
- Eigen x.dot(y) conjugates x: never write x.conjugate().dot(y) (gave NaN norms in DIFFFF).
- Driver pitfall: a background bash driver killed at the 2 h limit may keep running its batch loop
  (its subshell survives); a second driver must not re-run the same batch (race). Renaming a running
  .exe on Windows works and allows relinking the benchmark mid-batch.

**Why:** closes the WP-V2 open cause; remedy must absorb SPPs (frame/walls/PML), not fix cavity modes,
and the termination test must also check grazing light (a surface frame does not absorb it).
**How to apply:** do not propose JMCFIE / tighter d0 for Ag far-field box effects; see [[ag-null-mode-findings]],
[[box-validity-mlfmm-findings]], [[formulation-sweep-findings]].
