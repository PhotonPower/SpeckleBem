---
name: gmres-orthogonalization-findings
description: WP-G1 GMRES CGS2 vs MGS facts - 0.7 re-orth criterion fires almost always, bandwidth limits on the 5900X, BLAS-threading nondeterminism in test operators
metadata:
  type: project
---

Findings from WP-G1 (2026-10-10, branch wp/g1-gmres-cgs2):

- The DGKS/Kahan-Parlett 0.7 criterion fires in nearly every Arnoldi step for clustered spectra:
  MGS re-orthogonalised 45/46 (Ag PMCHWT+Jacobi sphere n=1), 52/53 (Si ICTF), 157/200 (row-scaled
  random), 200/200 (diag in disk |z-1|<0.95). So "selective" CGS2 == always-two-pass in practice;
  CGS2 was made always two passes with the pass-1 update fused with the pass-2 projection.
- Dev machine = Ryzen 9 5900X (12 cores/24 threads, dual-channel DDR4, 2x32 MB L3): parallel
  streaming tops out ~25-30 GB/s vs ~10-20 GB/s for one thread, so parallel CGS2 (3 sweeps over
  V, ~0.35 ms per basis vector per step at n=2e5, 24 threads) gains only ~2-4x over serial MGS
  (whose second read of v_j hits L3). Do not expect core-count speed-ups for bandwidth-bound
  Krylov kernels on this machine; 24 threads still beat 12 (1.6x).
- Thread-count determinism tests must not use DenseOperator or Eigen GEMV in the test operator:
  OpenBLAS threaded zgemv changes its reduction split with omp_set_num_threads. Use explicit
  dot products in the test double.
- Catch2 INFO only prints on failure; through ctest, temporary WARN() is the way to read
  observed values (revert before commit).
- The machine was under load 100-200 (other worker) for hours; benchmark timings then are
  noise-dominated -- see [[assembly-performance]] for timing method. A loaded run showed 7.7x
  CGS2/MGS, a review spot check under full contention 0.65-1.5x (barrier stalls): never quote
  speed-ups measured under contention. Row blocks now use schedule(dynamic, 1) (WP-G1 review).
