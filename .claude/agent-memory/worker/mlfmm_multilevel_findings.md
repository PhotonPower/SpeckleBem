---
name: mlfmm-multilevel-findings
description: WP20a multilevel far operator measurements (far-only/full-matvec errors vs leaf, r_max/a limits of the order search, symmetry check, test-cost lessons under win-debug)
metadata:
  type: project
---

Measured in WP20a (2026-10-09, tests/unit/test_mlfmm.cpp, slow `[mlfmm_far_sweep]`), 4-level
icospheres, subdivision 3 (r_max / a ~ 0.53), vacuum exterior, PMCHWT, d0 = 3:
- far-only error vs dense (near pairs removed): a = lambda/4: 1.7e-3, lambda/2: 4.6e-4 (n=1.5),
  5.4e-4 (Si), 0.75 lambda: 1.6e-4, lambda: 1.6e-4, 1.5 lambda: 4.6e-4. Full-matvec (docs/05
  metric) is 2-8x smaller: lambda/4: 2.2e-4, lambda/2: 1.7e-4. Multilevel adds ~nothing over the
  single-level (WP19b) block errors.
- d0 = 5: search_truncation_order reports "not achievable" at every leaf size up to 1.5 lambda
  when r_max / a ~ 0.5 (best 2e-5 ... 6e-4, gets WORSE with box size, patience stop). The
  enlarged diagonal sqrt(3) a + 2 r_max exceeds the nearest offset 2a. Needs r_max/a <~ 0.3
  (4 levels -> subdivision 4 -> dense 2N = 15360, too big for a test oracle).
- Coarse meshes (r_max / a ~ 1) fail the search even at d0 = 3: tests need fine meshes.
- PMCHWT symmetry u^T S Z_far v = v^T S Z_far u holds to 2e-12 ... 6e-11 (S = diag(I, -I)):
  confirms anterpolation = I^T with parent weights before anterpolating and conjugate shifts.
- 3-level operator vs far_block sums: 2e-10 relative (round-off amplified by the cancellation
  in the direction sum), not 1e-12.
- Cost: win-debug is 20-80x slower than release for the passes and far_block; the dense oracle
  of 2N = 3840 takes 260 s in win-debug -> dense comparisons release-only.

**Why:** WP20b/WP21 choose leaf sizes and test meshes; the search limits matter more than the
interpolation.
**How to apply:** use r_max / a <= 0.3 for d0 = 5 tests; see [[mlfmm-single-level-findings]].
