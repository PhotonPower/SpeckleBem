---
name: pml-ring-findings
description: WP-PML5a fast PML ring (MLFMM Z_rr + exact ring) - design, costs (1.4x dense assembly, 0.55x memory at 7c), d0 = 5 at 7c, chunked thread-independent apply, 5a limit, tests; WP-PML5c near-ring cost (touching pairs 95 %, bitwise translation classes, kernels-layer key, plan fingerprint)
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

WP-PML5c (2026-10-19, `benchmarks/pml_ring.cpp --assembly-study [--quick] [--dense-compare]`):
- Near-ring time is ~95 % stretched TOUCHING pairs (U: shared vertex 85 %, 0.55 s/pair, 2.8e6
  point pairs; shared edge 11 %); near/far pairs ~3.6 %. So "each unordered pair once" alone only
  reaches the dense assembler (which already does that, WP-PML2c): ratio ~0.9-0.95, gate 1/2 missed.
- Lever that met the gate: translation classes of touching pairs in the PML strips (stretch
  depends on x only where |y| <= y.start). Key must be EXACT bits of every coordinate-derived
  input (u-diffs from the rule origin A = first vertex of the min triangle shared with the other,
  per-triangle edge u-diffs, area, normal, edge lengths, signs by free vertex): then copies are
  bitwise (tested U, flat CI, tiny). A rounding-tolerant key (16 ulp) finds ~2x more copies but
  gives 1.8e-15 of max entry / 2.3e-13 per entry (cancelling entries) -> rejected.
  Hits: U 65 % of touching pairs, flat CI 47 % (mesh grid y-coords don't translate exactly).
- Review round (2026-10-19): key now kernels::stretched_touching_key (built from PairSetup/Slots/
  touching_points/fast_region/touching_origin in stretched_operators.cpp; back-refs there). Extra
  key inputs: fastmath flag per region (r_max uses ABSOLUTE centroids), 3x3 shared-index pattern,
  rule point counts; pairs with a vertex within 1e-12 max(start, h) of the strip boundary keyless.
  Effect: U copies 13312 -> 12544 (classes 4096 same), flat CI 2112 -> 2048 (1600 -> 1280 classes);
  still bitwise. On this structured grid the boundary case never actually diverged (rule points
  are strictly interior; vertices at d = 0 give sigma 0), so the old code also passed the new test.
  Plan now carries an octree fingerprint (refused on another tree) and near_assembly_bytes
  (estimate_sparse_stretched_cache, 0.26 s index pass at U, 339 MB bound vs 329 MB measured).
  U full study rerun: 5c 114.7 s vs dense same pairs 438.1 s (0.26); quick runs vary 2x with load.
- B(s,t) = B(t,s)^T bitwise for stretched pairs (kernel canonical ordering) -> canonical cache is
  bitwise 5a. Diagonal from near parts: real rows bitwise, stretched rows 3e-16 vs
  assemble_diagonal (dense order 2A + rest).
- Machine shared with another worker: absolute times x1.7-2 the 5a record (5a near ring U 640-700 s
  vs 380 s recorded); compare modes within one process only.

**Why:** WP-PML5b (ACA) and 5c (assembly cache) build on this.
**How to apply:** see [[stretched-kernel-findings]], [[mlfmm-lossy-policy-findings]].
