# Near-field block-diagonal preconditioner (WP-PC1)

ADR 0011 decisions 1–3 (the gate of the default). Library code `include/specklebem/solver/preconditioner.hpp`
(`solver::BlockDiagonalPreconditioner`, `extract_diagonal_blocks`), `mlfmm::level_unknown_blocks`
(`compression/mlfmm/octree.hpp`), `make_block_preconditioner` and `SimulationConfig::preconditioner` /
`block_levels_above_leaf` (`simulation.hpp`); tests `tests/unit/test_block_preconditioner.cpp`,
`tests/validation/test_block_preconditioner_mie.cpp`; driver `benchmarks/preconditioner_study.cpp`; raw lines
`preconditioner_rows.txt`. Measured 2026-10-25 on a 12-core desktop with 128 GB RAM, win-release, one study process
at a time — **but the machine ran the studies of two other work packages throughout** (their processes were visible
during every run), so wall times are loaded and vary between runs of the same process: the matvec of the U operator
took 0.18 s in the Jacobi run and 1.1–1.6 s in the block runs of the same process; WP-PML5b measured 0.81 s / 1.88 s
at L = 16 / 24 µm, this study 1.1–3.0 s / 2.2–2.4 s. Iteration counts, memory and results are load-independent.

## Summary

- **Gate met (ADR 0011 decision 3).** At the largest rough-Ag size (2N = 300 160, L = 24 µm) the blocks one level above
  the leaves need **844** iterations against Jacobi's 2 679 (**3.17×** fewer), the leaf blocks 1 573 (1.70×); the solve
  (setup + GMRES) took 2 916 s / 5 438 s against 6 540 s for Jacobi in WP-PML5b — on a loaded machine whose matvec was
  1.2× slower than in the 5b run. Both levels pass the rule (≥ 1.5× and a shorter total); the level one up is the
  faster one at every size. **The default of the PML route with compression "mlfmm" is now the block-diagonal
  preconditioner one level above the leaves** (GMRES; `block_levels_above_leaf` default 1); the box route and the dense
  PML route (not measured; review decision) keep the recommendation (Jacobi for metals, none for dielectrics).
- **A constant factor, not a new exponent.** Least-squares exponent of the iterations over the three Ag sizes:
  Jacobi N^0.57, leaf N^0.55, one level up N^0.58. The local blocks remove a factor ≈ 3.2 but not the growth, as
  ADR 0011 anticipated for long-range (SPP) coupling: the next lever is the formulation-level (Schur / Calderón)
  preconditioner of Phase 8.
- **Rough Si (PML, 2N = 22 816):** none 121, Jacobi 199, leaf 64, one level up **48** iterations (2.5× fewer than the
  recommended "none"), GMRES 397 / 647 / 200 / 79 s. The PML-route default therefore covers Si too.
- **Ag sphere (box route, MLFMM): the blocks are worse than Jacobi** — 2N = 15 360: Jacobi 316, leaf 768, one level up
  523; 2N = 3 840: Jacobi 210, leaf 490, one level up 266, none 203. The gain is specific to the rough PML systems;
  the box-route default is unchanged.
- **Results unchanged within the solver tolerance:** R of the rough Ag cases equal to 5·10⁻⁸ (U, tolerance 10⁻⁷) and to
  8 digits at L = 16 / 24 µm against the WP-PML5b Jacobi values; windowed far field L2 ≤ 2.3·10⁻⁷; Si R within
  4·10⁻⁷ (Jacobi vs none: 3.4·10⁻⁷); Ag sphere RCS within 4.8·10⁻⁶ of Jacobi at tolerance 10⁻⁶, ε_rr equal to
  5·10⁻⁷ absolute (gated in `tests/validation/test_block_preconditioner_mie.cpp`). The validation-large 7h case
  (fast path vs dense LU at U) passes with the new default: 184 iterations, dR −4.0·10⁻⁹, L2 3.33·10⁻⁶ (Jacobi in
  WP-PML5b: 602 iterations, dR +4.8·10⁻⁸, L2 3.33·10⁻⁶; 1 901 s test run).
- **Cost:** setup ≤ 3.1 s (extraction + LU) at every size; memory 0.36 GB (leaf) / 1.41 GB (one level up) at
  2N = 3·10⁵ against a 20.1 GB operator; the full-GMRES basis drops from 12.9 GB (Jacobi) to 4.06 GB, the process peak
  from 33.6 GB to 28.6 GB.
- **WP-G2 (ADR 0011 decision 4) is not triggered:** at the largest size the orthogonalisation is 10.5 % of the GMRES
  time (307 of 2 915 s; Jacobi in 5b: ≈ 25 %) and the basis 4.1 GB.

## Setup

| Item | Value |
|---|---|
| Rough Ag PML | `pml_rough_ag::u_case()` (rough Ag, σ = 50 nm, Lc = 500 nm, seed 1, taper 0.5, gap 0.5, W_pml = 1 µm, σ_max 4, m 2, h = λ/4, w₀ = λ, normal incidence); L = 3.75 µm (U) and `--patch L --map-patch L` with L = 16 / 24 µm (the WP-PML5b `u_L16` / `u_L24` cases: 2N and R reproduced) |
| Rough Si PML | the same U case with `--material si` (the 7d control of WP-PML5b) |
| Operator | `Simulation` PML constructor, compression "mlfmm" (d₀ = 3, automatic leaf rule, ACA far ring), assembled once per process; ICTF (automatic) |
| GMRES | full (no restart), CGS2, left preconditioning, tolerance 10⁻⁷ (PML) / 10⁻⁶ (sphere), the same operator and right-hand side for every variant |
| Variants | `jacobi` (MlfmmOperator::diagonal()), `none`, `leaf` (blocks of the octree leaves), `parent` (one level up); blocks over the J and M unknowns of each box, entries from Z_rr's near part + the near ring (every same-box entry exactly once, checked during the extraction) |
| Ag sphere | `ag_sphere_mlfmm_support.hpp`: d = 1 µm, vacuum, 500 nm, icosphere n = 4 (2N = 15 360; n = 3 for the validation test), MLFMM d₀ = 3 |
| Baseline | Jacobi on the three Ag sizes: WP-PML5b (`pml_ring_rows.txt`, FAST lines; not rerun, as the brief asked, except at U where it costs 2 min); the U Jacobi run of this study reproduces 602 iterations and R = 0.97833919 |

## Rough Ag, PML route

| 2N | L | leaf | variant | blocks (unknowns min / mean / max) | P memory | setup | iterations | ratio to Jacobi | GMRES (orth) | matvec ¹ | Krylov basis | R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 816 | 3.75 µm | 0.97 λ | Jacobi | — | 0.4 MB | 0 | 602 | 1 | 114 s (13) | 0.18 s | 0.22 GB | 0.97833919 |
| | | | leaf | 256 (66 / 89 / 96) | 33 MB | 0.04 s | 329 | 1.83 | 392 s (64) ² | 1.61 s | 0.12 GB | 0.97833915 |
| | | | one level up | 64 (322 / 357 / 384) | 131 MB | 0.54 s | **184** | **3.27** | 261 s (41) ² | 1.12 s | 0.07 GB | 0.97833914 |
| 152 960 | 16 µm | 1.25 λ | Jacobi (5b) | — | — | — | 1 715 | 1 | 1 829 s | 0.81 s | 4.20 GB | 0.97780527 |
| | | | leaf | 1 024 (130 / 149 / 170) | 368 MB | 0.41 s | 610 | 2.81 | 1 678 s (220) | 2.97 s | 1.50 GB | 0.97780528 |
| | | | one level up | 256 (560 / 598 / 640) | 1 465 MB | 2.59 s | **520** | **3.30** | 1 023 s (97) | 1.14 s | 1.28 GB | 0.97780528 |
| 300 160 | 24 µm | 0.875 λ | Jacobi (5b) | — | — | — | 2 679 | 1 | 6 540 s (≈ 25 %) | 1.88 s | 12.9 GB | 0.97948552 |
| | | | leaf | 4 096 (54 / 73 / 96) | 359 MB | 3.14 s | 1 573 | 1.70 | 5 434 s (843) | 2.42 s | 7.56 GB | 0.97948552 |
| | | | one level up | 1 024 (266 / 293 / 322) | 1 413 MB | 1.27 s | **844** | **3.17** | 2 915 s (307) | 2.19 s | 4.06 GB | 0.97948552 |

¹ One operator application measured right after the solve (load indicator; the WP-PML5b values on its rows).
² A sanitizer test run of this work package overlapped these two solves (matvec 6–9× slower than the Jacobi run of the
same process); later runs were kept free of own concurrent work.

Fitted exponents (least squares of ln its over ln 2N, three points): Jacobi 0.57, leaf 0.55, one level up 0.58. The
leaf ratio varies with the leaf size the leaf rule picks (0.97 / 1.25 / 0.875 λ: 89 / 149 / 73 unknowns per block); one
level up the ratio is 3.2–3.3 at every size. Load-free estimate of the largest solve with the blocks one level up:
844 × 1.88 s (the 5b matvec) + 307 s orthogonalisation + 844 × the block apply (≈ 0.05 s) ≈ 1.9·10³ s, i.e. ≈ 3.4× faster
than Jacobi; measured (loaded) 2.24×.

Process peak at 2N = 300 160: 28.6 GB (both block runs in one process; 33.6 GB with Jacobi in 5b), of which the operator
is 20.1 GB, the larger preconditioner 1.4 GB and the basis of the leaf run 7.6 GB.

## Rough Si, PML route (U, 2N = 22 816)

| variant | iterations | GMRES (orth) | matvec ¹ | setup | P memory | R | dR vs none |
|---|---|---|---|---|---|---|---|
| none (the recommendation) | 121 | 397 s (21) | 2.55 s | — | — | 0.38713565 | — |
| Jacobi | 199 | 647 s (34) | 3.12 s | 0 | 0.4 MB | 0.38713578 | +3.4·10⁻⁷ |
| leaf | 64 | 200 s (11) | 1.76 s | 0.20 s | 33 MB | 0.38713580 | +4.0·10⁻⁷ |
| one level up | **48** | 79 s (1.5) | 2.84 s | 0.21 s | 131 MB | 0.38713581 | +4.1·10⁻⁷ |

Si's real–real MLFMM far part makes the matvec 10× the Ag one (5.97 GB far operator, WP-PML5b); the 121 iterations
without a preconditioner reproduce WP-PML5b. R differs between the four solves by ≤ 4·10⁻⁷ (true residual 7–12·10⁻⁸;
none and Jacobi differ by the same amount).

## Ag sphere, box route (MLFMM, tolerance 10⁻⁶)

| 2N | variant | blocks | iterations | GMRES ¹ | ε_rr xz / yz | RCS difference to Jacobi |
|---|---|---|---|---|---|---|
| 15 360 (ico 4) | Jacobi | — | 316 | 209 s | 7.849·10⁻⁴ / 5.702·10⁻⁴ | — |
| | leaf | 272 (4 / 56 / 94) | 768 | 237 s | 7.854·10⁻⁴ / 5.707·10⁻⁴ | 4.8·10⁻⁶ |
| | one level up | 56 (42 / 274 / 350) | 523 | 107 s | 7.848·10⁻⁴ / 5.700·10⁻⁴ | 3.9·10⁻⁶ |
| 3 840 (ico 3) | Jacobi / leaf / one level up / none | 56 / 8 | 210 / 490 / 266 / 203 | — | 3.481·10⁻³ / 2.560·10⁻³ (all) | ≤ 2.1·10⁻⁶ |

On the closed sphere the octree boxes cut the surface into small, irregular pieces (4–94 unknowns per leaf at ico 4)
and the block inverses amplify the coupling across the cuts; Jacobi (and for ICTF even no preconditioner) converges
faster. The block preconditioner is therefore not the box-route default. The results agree with the Jacobi solve to the
solver tolerance (RCS to 5·10⁻⁶, ε_rr to 5·10⁻⁷).

## Default decision (rule of ADR 0011 decision 3)

| Criterion at 2N = 300 160 | leaf | one level up |
|---|---|---|
| iterations ≥ 1.5× fewer than Jacobi (2 679) | 1 573: 1.70× — met | 844: 3.17× — met |
| setup + GMRES below Jacobi (6 540 s, 5b) | 3 + 5 434 s — met | 1 + 2 915 s — met |

Switched: `Simulation` on the PML route with compression "mlfmm" (GMRES) and no explicit `preconditioner` /
`diagonal_preconditioner` uses `PreconditionerKind::BlockDiagonal` with `block_levels_above_leaf = 1` (the faster level
at every size; the other level is not tuned further, as the brief asked). The box route, the dense PML route (the review
limited the default to the measured "mlfmm" route; the dense PML route is a measurement item), the direct solver and
every explicit setting are unchanged; `diagonal_preconditioner = true` / `preconditioner = Diagonal` still select Jacobi
(the old path of the "mlfmm" PML references). An explicit `BlockDiagonal` is allowed on every route.

| Route | compression | solver | automatic preconditioner |
|---|---|---|---|
| box (closed mesh) | dense or mlfmm | GMRES | recommendation: Jacobi (metals) / none (dielectrics) |
| PML | dense | GMRES | recommendation: Jacobi (metals) / none (dielectrics) |
| PML | mlfmm | GMRES | **block-diagonal, one level above the leaves** |
| any | dense | direct | none used (the resolved kind is reported as unused) |

On the automatic path a `SBEM_WARN` is logged at assembly when the largest block holds more than half of the unknowns
(a tiny or coarse tree whose block level is the root: the block solve is then a dense LU); explicit settings do not
warn.

## Design notes

- Blocks: one per octree box of the chosen level, over the J and M unknowns of the box's basis functions (J indices in
  the box's permuted order, then the M indices + N). On the leaf level and one level up every same-box pair is a near
  pair of the MLFMM (the leaves coincide or touch), so the near parts hold every block entry; two levels up they do not
  (the extraction then throws, tested).
- Entries: Z_rr's near part plus, on the PML route, the near ring; the two partition the near pattern, and the
  extraction refuses an entry stored in both or in none (tested on the flat and rough tiny PML cases against the dense
  PML matrix: real–real entries bitwise, the others to ≤ 10⁻¹⁴ relative).
- LU per block (Eigen::PartialPivLU), parallel over the blocks; apply with explicit substitutions per block (bitwise
  independent of the thread count, tested 1 vs 24 threads). A block with an estimated reciprocal condition number below
  10⁻¹³ throws with the box id (none occurred: the smallest estimate was 2.7·10⁻⁶).
- Tiny meshes: on the tiny PML meshes of the unit and Python tests (2N = 352, two-level tree) level 1 is the root, so
  the blocks one level up are one dense LU and GMRES converges in one iteration; tests that want the block path there
  use `block_levels_above_leaf = 0`.
- Invariance (WP13 analogue): M⁻¹Z of PMCHWT, ICTF and MCTF agree to 6.6·10⁻¹⁶ on a 2N = 240 Ag sphere; GMRES
  histories agree to 10⁻¹¹ for the first ~24 iterations, after which rounding grows exponentially for Jacobi and the
  block preconditioner alike (0.4 relative at the end of a 10⁻¹⁰ solve) — the test asserts the preconditioned matrix,
  the early history, the iteration count (± 1) and the solution.
