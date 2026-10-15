---
name: ag-near-field-findings
description: WP24 Ag sphere near field vs Mie - error levels per mesh/GMRES tol and zone (geometric shadow), near-surface profile (d_min = h/2), quadrature degree 6/12/19, test costs, z = 0 rounding trap
metadata:
  type: project
---

Measured 2026-10-14/15 (win-release, 24 threads, shared machine), WP22a case, post::total_field.
Zones since the WP24 review: shadow = geometric shadow |x| < a, z > 0 (e < 1); lit = every other
exterior point incl. x-axis and the forward region beside the sphere (e < 0.1).
- 4 um paper mesh (2N 393 216, h 30.25 nm), acceptance at GMRES tol 1e-4 (722 it): union lit max
  0.0124 (margin ~8x), beside shadow 0.0040, shadow 0.0062; ctest 2113 s (GMRES 1136 s, field
  775 s), peak 26.5 GB. tol 1e-3 (442 it): lit 0.0876 (shells, standing-wave minimum |E| 0.19 at
  d = 232 nm in front of the lit pole; only 1.14x margin), beside shadow 0.0220, shadow 0.0223
  -> GMRES tol dominates, not discretisation.
- 1 um ico n = 5 tol 1e-5: 0.0050 / 0.0049, ctest 229 s, 7.6 GB; tol 1e-3: 0.05. ico n = 3
  (lambda/6.6) dense CI test: lit 0.050, beside shadow 0.028, shadow 0.086 (23 s alone, 140 s
  under ctest -j 8). ico n = 2 fails region test (sagitta 5.6 nm > lambda/100 shell distance).
- Near-surface: degree 6 ok (<= 3 % of e) for d >= 0.47 h; degree 12 == 19 within 1 % for
  |d| >= 0.2 h; discretisation error rises steeply below ~h/2. Shadow in xz-plane not deep for
  these sizes (|E| >= 0.33).
- Cost: ~4.3-4.6 ns per (point x source sample) on 24 threads incl. winding test; 4 um grid 59k
  points 7 min; degree 19 is ~8x degree 6. Field eval (775-801 s) is cheaper than GMRES only at
  tol 1e-4 (1136 s), not at 1e-3 (584 s).
- Metric denominator is the local |E_Mie| (coordinator decision); lit |E_Mie| up to ~2.2 V/m,
  so abs error vs |E_inc| <= ~2.2 e (tol 1e-3: 0.19). Field 1/e depth in Ag lambda/(2 pi|Im n|)
  = 25.4 nm (12.7 nm is the intensity depth).
- Trap: points on the x-axis built as r cos(pi/2) have z ~ 1e-22 > 0 -> classify with
  |z| < 1e-9 a, not z <= 0, or they fall into the shadow zone.
- Trap: grid_points ceil(2 half / spacing) gives 243 / 123 instead of 241 / 121 (240.00000000000003
  in fp); kept for reproducibility of the recorded runs, unit test pins it.
- Provenance check: the d_min = 0 scan entry prints "0 (0)" (0 nm, no excluded points) only
  with code from 4ff95f1 on; older binaries print lambda/100.

**Why:** baseline for later near-field / Phase 6 field-map work and tolerance choices.
**How to apply:** use GMRES tol <= 1e-4 when near-field accuracy matters at 4 um; keep d_min = h/2.
See [[ag-sphere-mlfmm-findings]].
