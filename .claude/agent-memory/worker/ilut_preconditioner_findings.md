---
name: ilut-preconditioner-findings
description: WP-SC3 own ILUT of near blocks A, D - its vs fill/drop on rough Ag/Si PML, setup cost (sequential A dominates, ~120 M complex MAC/s), memory rule, benchmarks OFF by default, load signs
metadata:
  type: project
---

WP-SC3 (2026-10-31), `solver::IncompleteLuPreconditioner` (own dual-threshold ILUT; Eigen's
IncompleteLUT rejected: silent zero-pivot replacement, O(len^2) pivot search).

Measured (rough Ag PML, tol 1e-7, idle unless noted):
- U (2N 22 816): every measured set (fill 2/3/5 x drop 1e-3/1e-4, plus 1:1e-3, 1:1e-2, 2:1e-2;
  NOT measured: 1:1e-4, 3:1e-2, 5:1e-2) needs 21-34 its (parent 184, nearlu_ad 21). Fill 3 /
  drop 1e-3: 21 its, 287 MB, setup 17 s. Fill 2 / drop 1e-2: 23 its, 111 MB, setup 4.9 s under
  load (D keeps fill 0.19 only). Natural order (fill 3/1e-3): 24 its, A macs 4.22e9 vs AMD 2.04e9
  (AMD wins); its setup ran under load, so compare work (macs), not setup seconds.
- Setup transient: peak during setup ~2-2.5x the stored factors (fill bound reserved + block
  copies 24 B/entry; 24 um fill 3: 7.0 + 2.3 GB vs 4.0 GB stored); budget pre-check sees only the bound.
- 16 um (2N 152 960): fill 3/1e-3 50 its (parent 520, nearlu 44), 3.86 GB, setup 532 s;
  fill 2/1e-3 59 its, 3.31 GB, 387 s. Setup is the A block alone (D ~ 4x cheaper), sequential.
  fill 2/1e-2: 68 its, 1.2 GB, setup 70 s, total 129 s (parent 457 s).
- 24 um (2N 300 160): fill 3/1e-3 111 its (parent 844), 4.0 GB (20 % of op), setup 329 s,
  total 519 s (parent 1426 s); fill 2/1e-2 140 its, 1.9 GB, setup 100 s, total 333 s. Gates
  of ADR 0012 d6 met. Exponent ~0.6-0.67 (parent 0.58); leaf rule gives 372/648/324 near
  entries per A row at U/16/24 um, so the 16 um point is "too good" (16->24 step N^1.2).
- Si U: 14 its for fill 2 and 3 (parent 48, nearlu_ad 14). Ag sphere box route (2N 3840):
  63 its vs Jacobi 210 (default fill 3/1e-4).
- Setup kernel runs at ~120 M complex MAC/s single-threaded (macs in the ROW line); the A-block
  factorisation is the whole setup cost and grows with the near row length (372 -> 648 per row
  U -> 16 um). A faster kernel or a parallel A elimination is the lever if the setup matters.
- Memory rule of ADR 0012 d6 (4 x nnz(A+D) x 16 B): all sets 0.17-0.78 of it at U.

Pitfalls:
- `specklebem_preconditioner_study` is only built with `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`
  (default OFF): a "both presets build" claim does not cover the driver.
- Load from other sessions shows as matvec 0.13-0.30 s at Ag U (idle 0.097) / Si U 1.2-3.8 s
  (idle 1.0) and orth seconds for few iterations; iterations are unaffected.
- Thread-independence tests: regions with a `num_threads(2)` clause ignore omp_set_num_threads,
  so toggling it compares identical runs (WP-SC3 review W2). Genuine 1-thread run: call from
  thread 0 of an outer `parallel num_threads(2)` with omp_set_max_active_levels(1) and probe a
  nested region's team size (expect 1); compare factors byte-wise (factors_a()/factors_d()).
- Review overclaims to avoid: "every set" for a partial grid, "ceiling" for nearlu_ad (it keeps
  B, C box-diagonal, not a bound on ILUT), comparing a loaded with an idle setup time.

**Why:** the next preconditioner WP (default switch, faster setup, SPAI fallback) needs these.
**How to apply:** compare against these numbers; rerun the parent in the same process.
See [[schur-preconditioner-findings]], [[near-lu-ceiling-findings]].
