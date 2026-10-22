# Lead notes 2026-10-22: next work packages after WP-PML5

Binding for the coordinator. Written by the project lead (with Claude in chat) after triaging an
external, LLM-generated code review of `main` at fa856af. That review is **not** committed and must
not be used as a source: several of its findings do not match the code (it cites a non-existent
commit `83677fb`, a `src/scattering/far_field.cpp` that is not on the remote, a `BlockJacobi2x2`
preconditioner and a `make_preconditioner` fallback that do not exist — `preconditioner.cpp` only
has `DiagonalPreconditioner` — a `MlfmmOperator::near_matrix()` that does not exist, and shifted line
numbers), and some of its "bit-exact" patches are not bit-exact (see WP-PML5d). This file keeps only
what was checked against the code and is worth the agent time.

Scale reference for the targets below: at constant mesh density, L = 30 µm gives 2N ≈ 4.5–4.7·10⁵
(2N = 300 160 at L = 24 µm), not 2·10⁶. With the measured growth (iterations ≈ N^0.55) that is
≈ 3 400 full-GMRES iterations, a Krylov basis of ≈ 26 GB and a peak of ≈ 60 GB — memory is not the
limit on the 1 TB node; time per solve is. Many solves are planned (realisations, polarisations,
angles), so per-solve cost matters.

## Decisions

1. **WP-PML6: option (A)** (the coordinator's recommendation of 2026-10-20). Amend ADR 0010 accordingly.
2. New work packages, in this order of priority: **WP-PC1** (near-field block-diagonal
   preconditioner), **WP-PML5d** (triangle-pair cache in the ACA far ring). WP-PML5d is small and
   touches other files than WP-PML6, so it may run in parallel with the WP-PML6 continuation.
3. **WP-G2** (single-precision Krylov basis) is conditional on the WP-PC1 measurements (see below).
4. Deferred and rejected items are listed at the end; do not schedule them without a new lead note.

Acceptance criteria are not loosened anywhere. Determinism (bitwise identical results for any thread
count) holds for every new operator, preconditioner and solver path.

## WP-PML6 continuation: option (A)

Continue on `wp/pml6-huygens-far-field-2` (f5bd897, worker worktree agent-a40c1f6891657e9af; check it
still exists before starting a new worker).

- Huygens plane at z = z_H over the whole real part (patch + taper + gap) with window W ≡ 1 there,
  extended over the PML zone and beyond. Field values above the PML zone from the stretched sources
  by the analytically continued evaluation that is already implemented and tested (dipole through
  the PML to 6·10⁻³).
- The raised-cosine flank 1 → 0 lies **entirely outside** pml_start, so no part of the window cuts
  through the field over the real part. Rationale: the bound SPP field is still ≈ e⁻² in amplitude at
  z_H ≈ λ (ADR 0010); truncating it over the real part turns it into edge radiation that depends on
  z_H and on the window. Over the PML the stretch damps it, so the truncation there is harmless.
- Flank width Δ_H from the point-source error model of the handover (≈ 3–4 λ); measure Δ_H = 2, 3, 4 λ
  and report, then fix the default. z_H default stays ≈ λ (ADR 0010).
- Report the cost of the larger plane (points, time) next to the 7f/7g results.
- Gates as in ADR 0010, unchanged: 7f stays met; 7g: 7d < 0.5 % with the Huygens far field,
  z_H ×1.5 and window ×(1 ± 0.3) each < 0.1 %, rough Si 7e within the box's own sensitivity.
- If 7g is still not met with (A): record the result (ADR 0010 result paragraph, docs/05 rows 7f/7g,
  `benchmarks/results/pml_huygens.md`), keep `FarFieldSource::Currents` as default and hand back to
  the lead. No further far-field variants without a lead decision.
- Finish the open items of the continuation (benchmark record, docs/05 rows, ADR paragraph, full
  ctest both presets, report).

## WP-PC1: near-field block-diagonal preconditioner

Goal: fewer GMRES iterations on the rough-Ag PML systems (602 → 1 715 → 2 679 at 2N = 2.3·10⁴ /
1.5·10⁵ / 3.0·10⁵). Fewer iterations cut matvec time linearly and orthogonalisation time and basis
memory roughly quadratically / linearly, so this is the largest lever for L = 24–30 µm. It is the
`BlockDiagonalPreconditioner` already declared as planned in `include/specklebem/solver/preconditioner.hpp`.

- Blocks: one per octree leaf box, over the J and M coefficients of the box's basis functions
  (2n_b × 2n_b). Entries from the near-field sparse matrix of `MlfmmOperator` (`near_operator()`)
  plus, on the PML route, the near part of the PML ring (`PmlRingOperator::near_operator()`). Check
  (test) that the union contains every same-box entry exactly once and that the blocks equal the
  corresponding blocks of the dense matrix on a small case. A dense-route constructor (blocks
  extracted from the dense matrix, same box partition) is needed for those tests.
- Optional block level (leaf or one level up) as a parameter; measure both, since leaf blocks may be
  too small to capture the coupling.
- LU per block (`Eigen::PartialPivLU`), factorised in parallel; `apply` writes block-disjoint output →
  bitwise independent of the thread count. Report setup time and memory.
- Invariance test: as for Jacobi in WP13, iterations with left preconditioning are invariant under the
  formulation's region-independent block-row scaling (PMCHWT / ICTF / MCTF give the same iterates up
  to rounding).
- Configuration: `Simulation` option for the preconditioner (none / diagonal / block-diagonal),
  backward compatible with `diagonal_preconditioner`, Python mirror. **Default unchanged** until the
  measurement below decides.
- Measurement (the gate): the three rough-Ag PML sizes above (iterations, solve time, setup time,
  memory, fitted iteration exponent), one rough-Si case, one Mie sphere case (results unchanged
  within the solver tolerance). Switch the default for the PML route if the iterations drop by
  ≥ 1.5× at the largest size and the total solve time drops. Otherwise report and stop: a local block
  preconditioner may give a constant factor without changing the exponent if the long-range SPP
  coupling causes the growth; the Schur-complement preconditioner (Phase 8) would then be the next
  lever, and the lead decides.
- Needs the ADR note the handover already asks for (preconditioning of the PML systems).

## WP-PML5d: triangle-pair cache in the ACA far ring

`detail::BlockSource` (`src/compression/mlfmm/pml_ring_far.cpp`) evaluates the same triangle pair
(T, S) several times per admissible block: `evaluate()` returns the full 3×3 block, but `row(i)` uses
one test row and `column(j)` one source column of it; rows sharing a test triangle, columns sharing
a source triangle, and row/column overlaps re-evaluate the pair.

- Step 0 (measure first): count distinct triangle pairs per block next to `pair_evaluations` at U,
  2N ≈ 6·10⁴ Ag and one Si case. Implement only if evaluations / distinct pairs ≥ 1.3 overall;
  otherwise record the numbers and close the WP.
- Implementation: a lazily filled per-block memo of the `op::TrianglePairBlocks` results (and of the
  drop decision of `evaluate()`), used by `row()`, `column()` and `full()`.
- **Keep the loop nesting of `row()` and `column()` exactly as it is** (in `column()`: outer loop over
  `test_`, inner loop over the column's source triangles `ct`); only the call to `evaluate()` is
  replaced by the lookup. Swapping the loops (as the external review proposed) changes the summation
  order in `out` and is not bit-exact.
- Memory: |test_| × |src_| × 576 B is unbounded for large admissible blocks on coarse levels. Use a
  per-block budget (e.g. a few MB per thread) with a fallback (no memo, or a memo per test triangle
  only) above it, and include the memo in the existing cache-byte accounting and budget guards.
- Keep the statistics meaningful: count kernel evaluations and memo hits separately; reductions and
  skips per distinct pair.
- Gates: far ring bitwise identical to `main` (existing bit gates of `test_pml_ring` unchanged, no
  reference values touched), determinism across thread counts, assembly timings before/after at the
  three measurement cases (idle machine if possible).

## WP-G2 (conditional): single-precision storage of the Krylov basis

Start only after WP-PC1 is measured, and only if at the largest measured size the orthogonalisation
is still > ≈ 20 % of the solve time or the basis memory becomes a constraint.

- `GmresParams` option for the basis precision (double stays the default). Panels stored as
  `std::complex<float>`; dot products, updates, `w`, the Hessenberg matrix and all accumulations in
  double. CGS2 structure and the fixed row-block reduction order unchanged, so results stay bitwise
  independent of the thread count (they are not bitwise equal to the double path).
- Gates on the three rough-Ag sizes and one Si case: converged with the true relative residual
  ≤ the tolerance; iterations within +2 % of the double path; far field within 1·10⁻⁴ relative of
  the double path; orthogonalisation time and basis memory reported.

## Deferred (no WP now)

- **Damping of the stretched Green's function for real k in the far-ring drop bound / ACA tolerance.**
  Physically sound (Im R̃ ≤ 0 on the stretched coordinates, so |e^{−jkR̃}| < 1 even for vacuum and Si),
  but it changes results, needs its own gates and an ADR 0008 amendment, and a rigorous lower bound on
  −Im R̃ from box extents is not trivial. Revisit if the far-ring assembly is still a significant share
  of the setup after WP-PML5d.

## Rejected (do not implement)

- Removing the `if (sp[j] != 0) continue;` branch in `AcaFarRing::apply`: the writes go with stride 4
  into the interleaved [Jr, Ji, Mr, Mi] buffer, so no clean SIMD results; a bit-exact version needs a
  second chunk buffer (K × 4N). Reconsider only if a profile shows the far-ring apply above ≈ 15 % of
  the matvec.
- Panel-wise traversal in `KrylovBasis::project_rows` / `subtract_rows`: `j / 64` is a shift, the dot
  products dominate; no measurable gain.
- Coordinate snapping of the generated grid to collapse the ≈ 4 000 translation classes to 384: changes
  the mesh (bitwise references against 5a break) for a small remaining gain after 5c.
- A `BlockJacobi2x2` fix: the code it refers to does not exist.
