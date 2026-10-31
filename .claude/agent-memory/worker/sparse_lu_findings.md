---
name: sparse-lu-findings
description: WP-SC1 Eigen SparseLU memory facts (memInit reservation, factors are views into capacity, peak vs budget), no-OpenMP Clang check, measured test deviations, Catch2 comma pitfall
metadata:
  type: project
---

WP-SC1 review round (2026-10-31, branch wp/sc1-review-fixes; solver::SparseLuPreconditioner).

- Eigen SparseLU memInit (3.4.0 fetched and 5.0.1 MSYS2 identical): nzlumax = nzumax ~ fill x
  nnz(A), nzlmax = max(4, fill) x nnz / 4; lusup + ucol 16 B, usub 8 B, lsub 8 B -> about
  fill x nnz x 42 B reserved up front (the review said ~40 B). expand() grows 1.5x (old+new
  coexist), never shrinks. After factorize, m_Lstore / m_Ustore are MAPS onto glu.lusup/ucol/...,
  so the held memory is the capacity, not nnz(L+U) x 24 B (what memory_bytes() reports).
- Measured 16 um rough-Ag near field: factors 66.3 GB, process 84.7 GB (40 GB budget, refused by
  the post-check) / 86.4 GB (75 GB budget): the budget does not bound the peak.
- Eigen nbThreads(): a user's setNbThreads(n > 0) wins over omp_get_max_threads() in both
  versions, so the OpenMP clamp is bypassable.
- Without OpenMP, Clang -Wunused-private-field fires on members used only under
  #ifdef SPECKLEBEM_HAVE_OPENMP; check by compiling the TU -fsyntax-only with the win-debug flags
  minus -fopenmp / -DSPECKLEBEM_HAVE_OPENMP (all presets have OpenMP, so CI-only otherwise).
- Test deviations (test_sparse_lu_preconditioner.cpp): near-field LU vs dense solve 5-8e-14
  (release and debug); ICTF/MCTF invariance of M^-1 Z 2.3e-11 release, 2.1e-11 debug (every 16th
  column, ICTF only). Bounds 1e-10.
- Catch2 test specs split on commas: a test name containing ", " cannot be passed literally;
  use wildcards ("*give the same preconditioned system") and note they may match other files'
  cases (block_preconditioner has the same title). `-s` prints INFO of passing assertions, so
  measured values can be read without a temporary print.

**Why:** ADR 0012 / Phase 8 preconditioner work reuses the sparse LU and its memory model.
**How to apply:** size near-LU budgets with ~1.3x factors + inputs peak; see
[[preconditioner-findings]], [[dense-lu-findings]].
