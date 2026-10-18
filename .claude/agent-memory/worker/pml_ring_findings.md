---
name: pml-ring-findings
description: WP-PML5a fast PML ring (MLFMM Z_rr + exact ring) - design, costs (1.4x dense assembly, 0.55x memory at 7c), d0 = 5 at 7c, chunked thread-independent apply, 5a limit, tests
metadata:
  type: project
---

WP-PML5a (2026-10-18), record benchmarks/results/pml_ring.md (+ pml_ring_rows.txt).

- Design: octree over ALL functions (not a real-only sub-mesh), so Z_rr = P_r Z_mlfmm(unstretched)
  P_r to rounding; far op gets an exclusion mask (zero x/y, no exact pairs), near = real-real split of
  near_pattern (assemble_sparse on the problem with stretch = nullptr). Ring far blocks per row leaf
  (stretched rows x non-near columns), 64 B/pair, real rows applied TRANSPOSED (far sub-blocks complex
  symmetric, no jump terms). Fused direct + transposed apply, per-thread buffers. q < 1 refused.
- 7c U (2N 22816, 45 % stretched, idle 12-core): ring near 380 s (stretched touching pairs in both
  orderings) + far 522 s vs dense 640 s; diagonal 50 s; GMRES ICTF+Jacobi tol 1e-7: 602 its 125 s;
  op 4.6 GB (ring far 3.65) vs dense 8.3; matvec 1.5e-5 (rough mesh r_max 154 nm -> ~lambda leaves),
  dR 4e-8, l2 3.3e-6. validation-large test 1782 s total.
- Flat Ag d0 3: 3-4e-4; d0 5: 1e-8..5e-8 (few far levels). Below 2N ~1e4 fast is no memory win.
- 5a limit: ring far O(N_s N): 12.9 GB at 2N 5e4, 60 GB at 1.4e5 (W_pml 2 lambda, h lambda/4).
- docs/05 says unit tests < 1 s in release: the minute-long dense-vs-fast accuracy cases belong in
  tests/validation even if a brief lists them as unit tests.
- Read WARN values of a test through `ctest --preset ... -R name -V` (Catch2 wraps at 80 columns).
- Review round (2026-10-18): ring apply = one pass over <= 64 fixed row-leaf chunks with
  per-chunk transposed buffers summed in chunk order (thread-count independent; unit test
  1/2/3/max threads bitwise). Same speed as per-thread buffers (U ring 130 ms); a two-pass variant
  (transposed over column leaves) read the 3.65 GB blocks twice: 1.4-1.75x at U. Memory-bound. Rough-patch unit case (h 100 nm, Lc 200 nm,
  taper 200 nm): transposed far |JM|,|MJ| only ~7e-6 of the column scale (L self terms dominate),
  ring x vs (D - D_rr) x 8e-16. Shared-vertex far pairs possible at d0 = 3 auto leaves (r_max/a
  up to 0.6 > 0.5), not only with automatic_leaf_size = false.
- d0 = 5 at the 7c size (U): leaves 1.94 lambda (r_max 154 nm), 4 levels; matvec 1.4e-7 (gate
  1e-5), dR 5.5e-8 (= GMRES tol), l2 1.2e-7, 602 its; op 6.25 GB (ring near 0.71, far 3.43),
  assemble 1345 s (ring near 730 s: 7.4e6 pairs), dense 642 s; whole benchmark run ~38 min.
- available_memory_bytes lives in core/system_memory.hpp now (tests/support forwards to it).
- `ctest -R pml_ring` also matches the 30-min validation-large 7h case: use `-R "^pml_ring: "`.
  A running test exe locks the static-linked exe: build single targets (`--target`) meanwhile.

**Why:** WP-PML5b (ACA) and 5c (assembly cache) build on this.
**How to apply:** see [[stretched-kernel-findings]], [[mlfmm-lossy-policy-findings]].
