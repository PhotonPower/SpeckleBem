---
name: preconditioner-findings
description: WP-PC1 near-field block-diagonal preconditioner - iterations vs Jacobi (rough Ag PML U/L16/L24, Si, Ag sphere worse), exponent unchanged, setup/memory, PML default switched to one level up, invariance rounding growth, test/build pitfalls
metadata:
  type: project
---

WP-PC1 (2026-10-25, branch wp/pc1-block-diagonal-preconditioner; record
benchmarks/results/preconditioner.md, driver benchmarks/preconditioner_study.cpp, one case per
process, operator assembled once, variants jacobi/none/leaf/parent on the same system).

- Rough Ag PML ICTF tol 1e-7, 2N 22 816 / 152 960 / 300 160: Jacobi 602 / 1715 / 2679, leaf
  329 / 610 / 1573, one level up 184 / 520 / 844 (3.2-3.3x at every size). Exponents unchanged
  (0.57 / 0.55 / 0.58): constant factor only -> Phase 8 Schur/Calderon for the growth.
- L24 parent: GMRES 2915 s (loaded) vs 6540 s Jacobi (5b); setup 1.3 s, P 1.41 GB, basis
  12.9 -> 4.06 GB; orth 10.5 % of GMRES -> WP-G2 condition (> 20 %) not met.
- Rough Si PML U: none 121, Jacobi 199, leaf 64, parent 48.
- Ag sphere d 1 um (box route, MLFMM, tol 1e-6): blocks WORSE than Jacobi (ico4 316 / 768 leaf /
  523 parent; ico3 210 / 490 / 266, none 203). Box-route default left as before.
- Default switched: PML route + compression "mlfmm" + GMRES + no explicit setting -> BlockDiagonal,
  block_levels_above_leaf default 1 (also Python block_level=1). Review round (wp/pc1-review-fixes):
  the dense PML route keeps the recommendation (not measured); the automatic path warns when the
  largest block > half the unknowns (tiny 2-level trees); box-route support ROW lines gained
  precond= next to jacobi= (readers group on jacobi=, kept). The WP-PML5b cases are reproduced: U R 0.97833919 Jacobi,
  L16 0.97780528, L24 0.97948552 (map_patch = patch for u_L16 / u_L24).
- Invariance across PMCHWT/ICTF/MCTF: M^{-1} Z equal to 6.6e-16, but GMRES residual histories of the
  Ag sphere diverge exponentially from rounding (1e-11 at it 24, 1e-2 at it 40) for Jacobi AND
  block alike: assert the preconditioned matrix, early history, its +-1, solution.
- API pitfall: Eigen 3.4 MatrixXc has an initializer-list constructor, so f(part, {&a, &b}) is
  ambiguous between vector<const Sparse*> and MatrixXc overloads: pass an explicit vector type.
- GCC 16 -Wnull-dereference fires on `Eigen::PartialPivLU<MatrixXc>(expr).matrixLU()` temporaries
  in tests: name the matrix / use reconstructedMatrix().
- Benchmarks are OFF by default: configure with -DSPECKLEBEM_BUILD_BENCHMARKS=ON (cached).
- Tiny PML meshes (python test_pml.py cases, 2N = 352, 2-level octree): block level 1 = the root
  = one dense LU block, GMRES converges in 1 iteration, so such tests do not exercise the block
  path; use block_level=0 (50 its vs Jacobi 140, Huygens far field 2e-9 vs direct at tol 1e-10).
- Never run ctest while a timed study runs: a concurrent debug ctest made the U matvec 9x slower.
  The machine was shared with other WPs all day: matvec 1.2-3.7x the 5b values.

**Why:** ADR 0011 gate; WP-G2 and Phase 8 (Schur/Calderon) depend on these numbers.
**How to apply:** see [[gmres-observations]], [[pml-ring-findings]].
- Review round test facts: the 2N = 352 tiny PML mesh with default MLFMM params gives a 2-level
  tree (asserted in test_block_preconditioner.cpp). Debug (ASan) case times depend almost only on
  ctest -j load (OpenMP regions under oversubscription): invariance case 2.3 s alone, 101 s in a
  loaded full run before computing M^-1 Z blockwise, 3.3 s after; PML entry case 5 s / 31 s.
- Flake seen once in win-release -j 6: "result_writer: .npy directory under a non-ASCII directory
  name" found a foreign attributes.json in its temp dir (stamp collision between parallel test
  processes?); passed on 3 isolated reruns and the next full run.
