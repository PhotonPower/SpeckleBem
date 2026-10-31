---
name: near-lu-ceiling-findings
description: WP-SC1 near-field-inverse ceiling - Eigen SparseLU pitfalls (uninitialised m_info, fillfactor 20 reservation, thread-dependent solve), level-2 far-zero blocks, rough Ag U/L16/L24 numbers, fill growth, running long studies past the 2 h background limit
metadata:
  type: project
---

WP-SC1 (2026-10-30/31, branch wp/sc1-near-lu-ceiling; record benchmarks/results/preconditioner.md
section WP-SC1, driver variants grandparent / nearlu).

- Eigen 3.4 SparseLU pitfalls: m_info and m_factorizationIsOk are uninitialised until factorize
  sets them (memInit failure leaves m_info unset) -> subclass sets them in its ctor; memInit
  reserves fillfactor (default 20) x nnz for BOTH lusup and ucol up front (64 GB of values per
  1e8 input entries) -> subclass sets m_perfv.fillfactor from the expected fill; Eigen keeps
  its permuted input copy m_mat (only rows()/cols() needed by solve) -> swap with an empty
  matrix. SparseLU<..., Index> with int64 StorageIndex compiles warning-free (GCC/Clang -Werror).
- The SparseLU SOLVE is not thread-count independent (supernodal GEMV via EIGEN_USE_BLAS ->
  OpenBLAS zgemv threads): 1 vs 24 threads differ; omp_set_num_threads(1) around
  factorize/solve (restored after) makes it bitwise. n = 1000 random sparse, 3 per row (fill 37)
  shows it in 0.2 s; n = 3000, 8 per row took 1240 s in win-debug (fill 176) - keep tests small.
- Level 2 (two levels above the leaves): ~60 % of the block entries are far pairs -> blocks of
  Z_n (MissingBlockEntries::Zero).
- MLFMM test meshes: icosphere subdivision s has r_max ~0.535 R (s=1), 0.268 R (s=2), 0.134 R
  (s=3); leaf edge with k forced levels = 2R / 2^(k-1); MlfmmFarOperator throws if r_max >= leaf.
- Rough Ag PML (tol 1e-7, idle): parent / grandparent / nearlu = 184 / 99 / 20 (U), 520 / 335 / 44
  (L16), 844 / 518 / - (L24). Si U 48 / 41 / 13. Fill 7.26 (U, 2.96 GB, 67 s) -> 13.93 (L16,
  66.3 GB, 4921 s sequential, peak 86 GB): fill grows with size, so a small-case fill
  under-estimates by ~2x. nearlu exponent ~0.41 (2 sizes). Gate 2 met -> WP-SC2 suggested.
- Idle-machine timings ~2x faster than WP-PC1's loaded ones (L16 assembly 675 s, L24 1238 s).
- Long studies: a run_in_background shell is stopped at 2 h, but TaskStop / the limit does NOT
  kill the native exe it started (stdout redirect keeps logging); watch it via tasklist and a
  log-polling script, stop it with taskkill //PID if needed.

**Why:** ADR 0012 gate 2 (WP-SC2 Schur preconditioner) depends on these numbers.
**How to apply:** see [[preconditioner-findings]]; for WP-SC2 compare against these iterations.
