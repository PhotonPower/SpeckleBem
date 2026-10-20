---
name: pml-ring-findings
description: WP-PML5a fast PML ring (MLFMM Z_rr + exact ring) - design, costs, d0 = 5 at 7c, chunked thread-independent apply, 5a limit; WP-PML5b ACA far ring (row clusters, eta 2, kappa stretch bound, 2N 3e5); WP-PML5c near-ring cost (touching 95 %, bitwise translation classes, kernels key, plan fingerprint); 5b+5c integration; 5b review fixes (min ACA rank 4, zero-rows probe, no timing effect)
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

WP-PML5b (2026-10-19, ACA far ring; record benchmarks/results/pml_ring.md "5b"):
- Leaf rows (brief's base design) compress only ~2.6x at U (rank ~13 at 1e-4 vs ~33 stretched rows per
  leaf, ratio bounded by k/m); H-matrix row clusters (dual same-level descent) 4.5x (818 MB vs 3.65 GB),
  ring err 3.1e-5 (<= 1e-4), full-operator diff 8e-6. Default row_clusters = true.
- Plain ACA stopping rule left 1.04 eps on a k = 6 Helmholtz test block -> confirmations = 2 default.
- Tiny unit meshes: all admissible blocks hit the break-even rank cap (ranks > mn/(m+n)); unit tests
  force ACA with eta 20 + explicit max_rank 1000 and use far-only rings (assemble_near = false) for
  speed (the lossy near ring took 400 s under ASan).
- Stretch shortens Re R~ down to kappa R, kappa = 2/sqrt(4 + sigma^2) (0.447 at sigma 4): any
  exp(-alpha d) drop bound with a stretch must use kappa d. R2 (Ag) parts dropped in ~65 % of U blocks.
- ring compare without rebuilding the exact ring's near part: assemble_near = false + ACA ring's near.
- eta (U, row clusters): 0.5/1/2/3 -> far ring 1260/819/571/524 MB, max rank 19/28/37/39; default 2.
  Per-triangle-pair R2 drop (Ag beyond 1.2 um) cut the far assembly 387 -> 240 s.
- Gates through ACA: U d0 3 matvec 2.1e-5, dR 4.8e-8, l2 3.3e-6 (op 1.52 GB vs 4.6 GB 5a); d0 5 2.1e-7;
  2N 59 600 (L 8.5 um) ACA far 1.64 GB vs exact 16.7 GB, ring err 5.1e-5, far build 387 vs 3568 s.
- Si: R2 not damped, ranks to 69, far ring 1480 s at U (Ag 151 s); Z_rr MLFMM far of Si is 6 GB.
  7d Si via fast path 0.2412 % vs dense 0.2411 %.
- pml_rough_ag::estimate built a dense Simulation -> refused 2N > 1e5; fixed (mlfmm config).
- The machine is shared with other WP worktrees (e.g. gate-pml5c2 test runs): near-ring times 380 -> 614-866 s.
- A 6e4 ACA-vs-exact validation-large case takes ~80 min (exact far ring 3568 s; 54 min with the 5c
  near ring, exact far 2808 s on an idle machine).
- Largest runs (fast only): 2N 152 960 (L 16 um) op 13.3 GB, peak 18 GB, 1715 its, d0 3 vs 5 R 1.5e-8;
  2N 300 160 (L 24 um) op 20.1 GB, peak 33.6 GB (full-GMRES basis 12.9 GB), 2679 its, ~3 h total.
  7e via fast path l2_refl 0.1464 % (= dense). A dense 7e partner stalled 10 h on the overloaded machine.
- Multi-hour runs: one detached (nohup) queue script with a progress file; background tool calls stop at 2 h.

5b + 5c integration (2026-10-20, branch wp/pml5b-integrate):
- PmlRingPlan carries the ACA far partition as a public PmlRingFarBlock list (+ a params copy);
  detail::far_partition_from_blocks rebuilds FarRowLeaf rows/extents from the tree (min/max extents
  are order-independent, so bitwise the partition_far_ring result). A plan made with other use_aca /
  eta / coarse_columns / row_clusters is refused; budget and aca_epsilon may differ.
- Peak semantics combined: pml_ring_peak_bytes = near + max(cache, far_exact_bytes) (ACA factors
  unknown before assembly; using far_bytes would warn on every large ACA run); the measured
  PmlRingOperator::peak_bytes() includes them.
- 7c U integrated (machine ~idle, dense assembly 678 s): assemble() 211 s (near 89, ACA far 119,
  diag 0.002 s) = 0.31x dense; the d0 3 test takes 17 min, the d0 5 matvec test 16 min.
  At d0 = 5 the 5c cache (1.11 GB) > near ring (0.67 GB): peak 3.91 GB vs 3.58 GB in 5b.
- Merging: `checkout --conflict=diff3 <files>` shows the base side; most doc/test conflicts were
  pure appends on both sides.

5b review fixes (2026-10-20, branches wp/pml5b-review-fixes(-2)):
- assemble_near = false is refused by MlfmmOperator and Simulation (diagnostic of a standalone
  PmlRingOperator only); unit tests keep using far-only standalone rings.
- W3: no ACA attempt below break-even rank kPmlRingMinAcaRank = 4 (automatic cap): at U 884
  blocks skipped, -2460 generated rows/cols, -1.2 % pair evaluations, far time unchanged (120 vs
  119 s): no speed lever, only memory/robustness. Flat CI has no such block. Stored ring and all
  7h digits identical.
- ACA k > 0 zero-rows exit now probes one unused column (farthest in index from the pivots) before
  accepting; it never fired at U or flat CI (zero residual rows practically never occur there;
  flat JM/MJ are identically zero -> k = 0 zero_block path). Evidence trick: generated cols - rows
  unchanged between builds.
- Post-assembly SBEM_WARN when exact + ACA bytes exceed max_far_bytes (budget checks only the
  pre-assembly far_exact_bytes). Python: pml_ring_* keys on Simulation() -> ValueError, although
  config() of a closed Simulation still reports them (no round-trip use exists).
- Before/after timing without a second worktree: `git archive <rev>` into scratch and build only
  the benchmark target there with the same preset.
