---
name: assembly-performance
description: Dense assembly cost facts (WP-P2): where time goes on rough boxes, GCC vectorisation flags, fast-math accuracy, timing methodology on the shared machine
metadata:
  type: project
---

Measured 2026-10-09 (WP-P2, `benchmarks/dense_assembly_profile.cpp`):
- Rough boxes: near/far pairs with a 400 nm box cell dominate; the degree selection uses
  kappa = sqrt((h/D)^2 + (0.15 |k| h)^2), so |k| h >~ 7 forces degrees 12-19 regardless of D
  (Si interior |k| h ~ 30 -> cap 19: 59-63 % of CPU before WP-P2, 66 % after; Ag 46 %
  before). Touching pairs < 3 %.
- Decay-aware degree choice is NOT monotone in Im k (larger alpha raises |k| in kappa, delta
  relaxes only for R_lb > 0). Keep R_lb = D - (rho_t + rho_s) (sum first) for bitwise symmetry.
- OpenMP static schedule over index-ordered rows was x2.5 (Si) / x4.5 (Ag) the ideal on boxes;
  box triangles come last in the mesh. Static schedules also suffer badly from other agents'
  load. Use dynamic, largest-first (bitwise identical: one writer per row per colour).
- GCC (UCRT64) does not vectorise loops with std::sqrt (errno) or FP `<` selects
  (-ftrapping-math): per-file `-fno-math-errno -fno-trapping-math` fixes both (value-neutral).
  `-fopt-info-vec-all` on a scratch TU shows why a loop is missed.
- UCRT std::exp is fast (~3 ns); sin+cos ~13-19 ns. Custom fdlibm-style sincos/exp agree to
  <= 1 ulp measured (tests enforce 2 ulp); element blocks within 2.7e-15.
- Rough-box wall pairs are exactly coplanar: K is rounding noise; relative-K checks need an
  absolute floor (e.g. 1e-12 x |f||f| grad-G scale).
- Timing: absolute numbers vary x2-3 with other agents; compare in the same process
  (`--compare` alternates old replica vs new build). Replica of the old static loop is the
  "before" reference when the old binary is gone.

**Why:** cost most of WP-P2's investigation.
**How to apply:** MLFMM near field (Phase 4) and any kernel/assembly performance work. See
[[build-pitfalls-windows]], [[touching-quadrature-findings]].
