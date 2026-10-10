---
name: ag-sphere-mlfmm-findings
description: WP22a measured Ag sphere MLFMM vs Mie costs/accuracy (d = 1/2/4 um), paper mesh is octahedral, GMRES tol dominates eps_rr, Eigen 3.4.0 SparseMatrix copy trap
metadata:
  type: project
---

Measured 2026-10-10 (win-release, 24 cores, shared machine), d0 = 3, ICTF + left Jacobi:
- The paper's 393 216 unknowns = octahedron sphere 8*4^7 (mean edge 30.25 nm = lambda/16.5, not
  lambda/27); icosphere n = 7 (lambda/26.5, 983k) needs a ~76 GB near field. d0 = 5 on the 393k
  mesh: near-field estimate 48.9 GB (lambda/2 leaves) -> not feasible on a shared 128 GB box.
- 4 um, 393k, tol 1e-3: eps_rr 0.023 % / 0.016 %, 442 it, assembly 91 s, solve 780 s,
  near 12.2 GB (1297/row), Ag exact part 6.8 GB (860 pairs/basis; budget did not fire), far
  2.8 GB, peak RSS 33.8 GB. Near/exact per row constant at fixed h (linear in N).
- GMRES tol 1e-3 dominates eps_rr for d = 1 um (0.19 % vs 0.078 % at 1e-6, which equals dense
  LU); MLFMM d0 = 3 error invisible in eps_rr.
- Peak RSS after WP22a-f (SparseOperator(Matrix&&) + swap, no copy): 1.80 x near on d = 2 um
  octa n = 6 (8.47 -> 5.54 GB; far incl. exact 2.25 GB is the rest) + Krylov basis.
- Eigen 3.4.0 (FetchContent) SparseMatrix has NO move ctor: `std::move` of a SparseMatrix copies,
  and so does passing it by value; take `Matrix&&` and swap. Test: compare valuePtr() before/after.
- Cheap peak measurement: specklebem_ag_sphere_mlfmm --mesh octa --n 6 --d 2e-6 --max-iter 5
  (~26 s, peak happens in assembly). Cygwin /proc/meminfo on Windows has no MemAvailable.

**Why:** sizing of WP22b/22c and later large Ag runs; avoid hidden sparse copies.
**How to apply:** use as memory/time baseline; swap Eigen sparse matrices instead of moving.
See [[mlfmm-lossy-policy-findings]], [[gmres-observations]].
