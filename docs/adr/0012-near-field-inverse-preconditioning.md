# ADR 0012: Phase 8 preconditioning of the PML-route systems — measure the near-field-inverse ceiling before building a Schur-complement preconditioner

- **Status**: accepted 2026-10-30 (user decision: open Phase 8 with the preconditioner item; lead design); WP-SC1 measured 2026-10-31, gate 2 met → WP-SC2 (section "Result"); WP-SC2 stage 1: the Schur split gains nothing (section "Amendment"), **decision 6 → WP-SC3 (incomplete LU of A and D)**
- **Date**: 2026-10-30
- **Phase**: 8

## Context

ADR 0011 (WP-PC1) gave the fast PML route of rough metal surfaces the near-field block-diagonal
preconditioner (one block per octree box one level above the leaves, J and M unknowns of the box in
one block). Measured on rough Ag (`benchmarks/results/preconditioner.md`): 184 / 520 / 844 iterations
at 2N = 2.3·10⁴ / 1.5·10⁵ / 3.0·10⁵ against Jacobi's 602 / 1 715 / 2 679 — a constant factor 3.2–3.3 at
every size, the growth exponent unchanged (Jacobi N^0.57, blocks N^0.58). At 2N = 3·10⁵ the solve still
costs 2 915 s and a 4 GB Krylov basis; the planned many-solve studies (realisations, angles,
polarisations, L = 30 µm with ≈ 1 200 iterations extrapolated) make the iteration count the dominant
cost. The Phase 8 definition of done (docs/01, "Ag rough surface iteration count reduced by ≥ 2× with
H-LU or Schur preconditioner vs diagonal") is numerically already met by the blocks; the open problem
is the remaining growth and the factor that is still available.

Options on the table for the "formulation-level" lever named in ADR 0011 (iii):

- **(a) Near-field Schur-complement preconditioner** (Ergül, Malas & Gürel 2011): partition the
  near-field matrix Z_n into the J/M blocks `[A B; C D]` (rows = the two combined equations, columns =
  `[J; M]`), approximate A⁻¹ (sparse approximate inverse or block LU) and the Schur complement
  S = D − C A⁻¹ B restricted to the near pattern, apply M⁻¹ by block forward / back substitution.
  Reuses the near sparse matrix, the box partition and the block LU of WP-PC1; a fixed linear
  preconditioner (no inner iterations), so the existing GMRES stays.
- **(b) Calderón preconditioning**: Buffa–Christiansen dual basis on the barycentric refinement
  (6× triangles), mixed Gram matrix, operator products (T² ≈ −¼ I + K²); makes the EFIE / PMCHWT
  blocks second kind. It cures the *h*-refinement breakdown (dense-mesh conditioning), not the growth
  with the electrical size at fixed mesh density, which is what we measure (h = λ/4 … λ/10 fixed, L
  grows). Cost: new basis, new assembly on the refined mesh, MLFMM on 6× the elements; 3–5 work
  packages.
- **(c) Sparse approximate inverse / ILU of Z_n** (ADR 0011 (ii)); **(d) approximate H-LU** of a
  low-accuracy H-matrix representation (docs/09; `HMatrixStrategy` is a placeholder).
- **(e) Krylov recycling across the many-solve studies** (same operator, many right-hand sides:
  deflated / recycled GMRES, GCRO-DR). A different lever (per-solve cost amortised over the study, no
  change of the per-solve conditioning); not part of this decision.

Two facts shape the decision. First, (a), (c) and (d) all approximate the inverse of the **near-field
matrix** (or of a low-accuracy whole-system representation whose far part is weak); their achievable
gain is bounded by the exact near-field inverse M⁻¹ = Z_n⁻¹ used as the preconditioner. That ceiling is
measurable today without any new approximation code: a sparse LU of Z_n at 2N = 2.3·10⁴ (seconds) and,
memory permitting, at 2N = 1.5·10⁵. Second, the current blocks already couple J and M **exactly inside
each box** (the box block is `[A_bb B_bb; C_bb D_bb]`); what the Schur variants add is the inter-box
near coupling, whose value is exactly what the ceiling measures. Building (a) first and then finding
that the ceiling is 1.3× would waste a work package. If the growth is driven by the long-range SPP
coupling (ADR 0010 context, ADR 0011 result), no near-field-based preconditioner changes the exponent,
and the honest outcome is to record that and to turn to (e).

## Decision

1. **WP-SC1 measures the ceiling** on the rough-Ag PML route (the `pml_rough_ag::u_case()` sizes of
   WP-PC1; Si U as the control), same driver `benchmarks/preconditioner_study.cpp`, same operator and
   right-hand side for every variant, one process at a time on an otherwise idle machine:
   - `solver::SparseLuPreconditioner`: exact LU of a given sparse matrix (`Eigen::SparseLU` with
     COLAMD ordering, complex double), `apply` = one solve, `memory_bytes()` from the factors, throws on
     a singular matrix, a non-square matrix or a size mismatch; deterministic and bitwise independent of
     the thread count (Eigen's SparseLU is sequential).
   - `make_near_field_lu_preconditioner(op)` in `simulation.hpp`: Z_n = Z_rr's near part + the near
     ring (the two disjoint sparse matrices of the ADR 0011 contract, summed), with a memory guard: the
     factor memory is checked against a caller-given budget (default: half of the available physical
     memory, `core/system_memory.hpp`) — before the numeric factorisation from the estimate
     `expected_fill × nnz(Z_n) × 24 B` with a caller-given fill factor (Eigen's SparseLU exposes the
     fill only after the numeric phase; the first wording "after the symbolic analysis" was wrong),
     and after it against the actual factors (throwing and releasing them).
   - `block_levels_above_leaf = 2` (grandparent blocks) is allowed in `SimulationConfig` and
     `make_block_preconditioner` as the intermediate point between "one level up" and the exact near
     inverse (the same partition code, `mlfmm::level_unknown_blocks`); the coarse-tree warning of ADR
     0011 applies.
   - Driver variants `grandparent` and `nearlu`, reported with the same ROW fields (iterations, setup,
     memory, GMRES time, R), measured at U and L = 16 µm for Ag (`nearlu` at L = 24 µm only if the
     memory estimate is ≤ 40 GB; `grandparent` at 24 µm in any case), plus the Si U control; `parent`
     rerun in the same processes as the reference (and Jacobi at U), so the timings are comparable.
     Record section in `preconditioner.md`, raw lines in `preconditioner_rows.txt`; results unchanged
     within the solver tolerance (R to 10⁻⁷ against the WP-PML5b / WP-PC1 values).
2. **Gate for building the approximation (decides WP-SC2):** at the largest Ag size where `nearlu` was
   measured, `nearlu` needs **≤ 0.5× the iterations of the one-level-up blocks** (≤ 92 at U, ≤ 260 at
   L = 16 µm) *and* the fitted exponent of the `nearlu` iterations over the measured sizes is below
   0.5. Then WP-SC2 implements (a): `solver::SchurComplementPreconditioner` over the J and M blocks of
   the near matrix — Ã⁻¹ = block LU over the J unknowns of each box (level as the blocks), S̃ = D −
   C Ã⁻¹ B restricted to the near pattern of D, block LU of S̃ over the M unknowns, `apply` = y₁ = Ã⁻¹ r₁,
   y₂ = S̃⁻¹ (r₂ − C y₁), x₁ = y₁ − Ã⁻¹ B y₂, x₂ = y₂ — a fixed linear preconditioner on either GMRES
   side; its gate is ≥ 2× fewer iterations than the one-level-up blocks at 2N = 3·10⁵ with a lower total
   time (setup + GMRES) and ≤ 2× the block preconditioner's memory. If the `grandparent` blocks alone
   reach ≥ 2× over the one-level-up blocks within 2× their memory, they become the new default of the
   "mlfmm" PML route instead (cheapest change) and WP-SC2 is not started.
3. **If the ceiling is below 2× over the one-level-up blocks**, options (a), (c) and (d) are closed for
   the rough-metal PML systems with the record as evidence; the Phase 8 preconditioner deliverable is
   reported as met by WP-PC1 (3.2× vs diagonal) with the growth recorded as physical (long-range
   surface-wave coupling), and the lead proposes (e) Krylov recycling for the many-solve studies as the
   next package (its own ADR).
4. **Calderón (b) is deferred**: it is not the lever for the measured growth. It needs its own ADR if a
   dense-mesh study (λ/20 or finer, e.g. for sharp features) shows the h-refinement breakdown.
5. **No default changes** before a gate decides; the new variants are explicit options; the box route
   and the dense PML route are untouched. `SparseLuPreconditioner` is also usable as the exact inverse
   of a dense-route system in tests.

## Consequences

- New code in `solver/` (`SparseLuPreconditioner`: Eigen only, below `mlfmm/` in the layer order as the
  blocks of ADR 0011), the factory in `simulation.hpp` (which sees both layers), the level-2 partition
  in `Simulation` / Python, two driver variants. ≈ 300–500 lines with tests (WP-SC1).
- Memory: an exact sparse LU of a surface-mesh near matrix with ≈ 10³ non-zeros per row has substantial
  fill-in; at 2N = 1.5·10⁵ tens of GB are possible. The memory guard makes the failure explicit; the
  variant is a measurement tool, never a default. Grandparent blocks cost 16 n_g² bytes per block
  (n_g ≈ 1 200–2 400 unknowns → 5–12 GB in total at the two large sizes).
- The study runs on the desktop: U sizes in minutes, L = 16 µm ≈ 20–40 min per variant (the parent
  reference 1 023 s), the L = 24 µm grandparent run ≈ 1 h; one process at a time (peak ≈ 30–50 GB).
- The ADR 0011 invariance test (left preconditioning, block-constant row scaling → the same
  preconditioned system for PMCHWT / ICTF / MCTF) holds for every preconditioner built from the full
  near matrix and is repeated for `nearlu`.
- Revisit after WP-SC1 (gate 2 / outcome 3) and, if WP-SC2 is built, after its measurement.

## Result (WP-SC1, 2026-10-31)

Measured with `benchmarks/preconditioner_study.cpp` on an idle 12-core desktop (128 GB; record section
"WP-SC1" in `benchmarks/results/preconditioner.md`, raw lines `preconditioner_rows.txt`), rough Ag PML, tolerance
10⁻⁷, one process at a time. Iterations (Jacobi / one level up / two levels up / exact near-field LU): 2N = 22 816:
602 / 184 / 99 / **20**; 2N = 152 960: 1 715 / 520 / 335 / **44**; 2N = 300 160: 2 679 / 844 / 518 / not attempted.
Rough Si U: 48 / 41 / 13. R agrees with WP-PML5b / WP-PC1 to ≤ 1·10⁻⁷ for every variant. **Gate 2 is met:** the
exact near-field inverse needs 0.085× the iterations of the one-level-up blocks at 2N = 1.5·10⁵ (limit 0.5×)
and its fitted exponent over the two measured sizes is 0.41 (< 0.5; Jacobi 0.57, blocks 0.58, two levels up
0.64). The two-levels-up blocks fail their rule (1.86 / 1.55 / 1.63× fewer iterations at 3.99× the memory, gain
shrinking with size; ≈ 60 % of their entries lie outside the near pattern and are zero). The exact LU is a bound,
not a preconditioner: at 2N = 1.5·10⁵ the factors take 66 GB (fill 13.9, up from 7.3 at U) and 82 min of
sequential factorisation (process peak 86 GB); the 40 GB cap of decision 1 refused the first attempt after the
factorisation, and the lead reran it once at 75 GB to obtain the exponent — the cap was a guess, the machine
has 128 GB. **Consequence: WP-SC2 builds the Schur-complement approximation of decision 2;** its gain has to come
from the inter-box near coupling (the box blocks already couple J and M exactly inside a box), with a memory of
a few GB instead of 66. No default changed; `block_levels_above_leaf = 2` and the near-field LU remain explicit
measurement tools. The 24 µm ceiling would need a server (65–200 GB depending on how the fill extrapolates,
> 2 h sequential). Two caveats for WP-SC2: the 0.41 exponent rests on two points, and the near pattern itself
grows with the leaf rule from U to 16 µm (744 → 1 296 entries per row), which favours the 16 µm point; and the
memory gate of decision 2 (≤ 2× the block preconditioner) counts the preconditioner's **own** storage (the
near matrix Z_n is already part of the operator and is not counted).

## Amendment (WP-SC2 stage 1, 2026-10-31)

The Schur-complement preconditioner of decision 2 was built (`solver::SchurComplementPreconditioner`, block LU of
A over the J boxes, S̃ = D − C Ã⁻¹ B on the pattern of D, block LU of S̃ over the M boxes; exact inter-box J–M
coupling through B and C) and measured at U: **184 iterations on rough Ag — exactly the one-level-up blocks'
184** (leaf level 327 vs 329; Ag sphere 261 vs 266; half the blocks' memory). The early gate (≤ 92) failed, so
the long runs were not made. Two diagnostic variants of the exact near-field LU locate the gain: a LU that keeps
the inter-box coupling only in B and C (A and D block-diagonal) needs 184 iterations; a LU that keeps it only in
A and D (B and C block-diagonal) needs **21**, against 20 for the full near-field LU. On a (nearly) planar
surface the near entries of the K operator between coplanar pairs vanish (f_m · ((r − r′) × f_n) = 0): the
inter-box J–M coupling is ≈ 0 and the Schur split on [J; M] is the wrong lever. **The whole gain of the
near-field inverse sits in the inter-box coupling inside A and D (the L operator).** Decision 2's algorithm is
therefore closed for rough surfaces; the implementation stays as the explicit option `PreconditionerKind::Schur`
(correct, tested, not measured further).

6. **WP-SC3 — incomplete LU of A and D separately** (user decision 2026-10-31): M⁻¹ = diag(Ã⁻¹, D̃⁻¹) with Ã, D̃
   the incomplete LU factorisations (ILUT: fill limit per row and drop tolerance as parameters) of the N × N near
   blocks A = Z_n[J, J] and D = Z_n[M, M]; `Eigen::IncompleteLUT<Complex>` if it is sound for complex indefinite
   matrices, otherwise an own ILUT (Saad) in `solver/` — Eigen only, no new dependency. Setup sequential and
   deterministic per block (A and D in parallel), apply = two triangular solves per block, bitwise independent of
   the thread count. Explicit option only. **Stage 1 (U):** ≤ 92 iterations on rough Ag with factor memory
   ≤ 4 × nnz(A + D) × 16 B; the fill / drop parameters are swept at U (fill 2, 3, 5 × the row nnz; drop 10⁻³,
   10⁻⁴), the setup time recorded. **Stage 2 (16 / 24 µm):** ≥ 2× fewer iterations than the one-level-up blocks at
   2N = 300 160 (844 → ≤ 422), lower total time (setup + GMRES), preconditioner memory ≤ 25 % of the operator
   (≤ 5 GB at 24 µm; the 2× rule of decision 2 would be below one copy of A and D at fill 1 and is replaced here).
   If both hold the "mlfmm" PML route switches its default; otherwise the result is recorded and the lead
   decides.
7. **Fallback** if ILUT breaks down or gains < 2× at U (indefinite complex blocks are a known ILU risk): a block
   sparse approximate inverse of A and D on the leaf-level near pattern (one least-squares problem per column
   block), its own decision after the WP-SC3 stage-1 numbers.
8. The latent gap found in WP-SC2 — `Eigen::PartialPivLU::rcond()` can return a finite value for an exactly zero
   pivot — is closed in WP-SC3 for `BlockDiagonalPreconditioner` and the Schur blocks (check the U diagonal).
