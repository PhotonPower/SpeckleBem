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

## WP-SC1: near-field-inverse ceiling (ADR 0012)

ADR 0012 decision 1: how many iterations the **exact** inverse of the near-field matrix saves as a preconditioner
(the ceiling of every near-field-based preconditioner: block-diagonal, Schur complement, SPAI / ILU of Z_n), and the
intermediate point "blocks two levels above the leaves". Library code `solver::SparseLuPreconditioner`
(`solver/sparse_lu_preconditioner.hpp`), `near_field_matrix` / `make_near_field_lu_preconditioner` and
`block_levels_above_leaf = 2` (`simulation.hpp`); tests `tests/unit/test_sparse_lu_preconditioner.cpp`; driver variants
`grandparent` and `nearlu` of `benchmarks/preconditioner_study.cpp`; raw lines in `preconditioner_rows.txt` (section
WP-SC1). No default was changed.

### Setup

| Item | Value |
|---|---|
| Machine | 12-core desktop, 128 GB RAM, win-release, idle machine (one study process at a time, no builds or tests during the runs; measured 2026-10-30/31) |
| Cases | as WP-PC1: rough Ag PML `u_case()` at L = 3.75 µm (U), `--patch L --map-patch L` with L = 16 / 24 µm; rough Si U (`--material si`) as the control; compression "mlfmm", d₀ = 3, ICTF, full GMRES (CGS2), left preconditioning, tolerance 10⁻⁷, one operator and right-hand side per process for all variants |
| `parent` | blocks one level above the leaves (the current PML-route default; rerun in the same processes as the reference) |
| `grandparent` | blocks two levels above the leaves, over the J and M unknowns of each box. Two levels up a same-box pair can be a **far** pair of the MLFMM, which no sparse part stores: the blocks are those of the near-field matrix Z_n (far entries zero; `MissingBlockEntries::Zero`). 59.7 % (U) / 60.9 % (16 and 24 µm) of the block entries are such far zeros (`far-zero` in the ROW lines) |
| `nearlu` | `make_near_field_lu_preconditioner`: Z_n = Z_rr's near part + the near ring (disjoint patterns, summed; nnz checked), `Eigen::SparseLU` (COLAMD, partial pivoting, complex double, 64-bit indices), factorisation and solve sequential (OpenMP thread count 1 during both: bitwise thread-count independent). Budget `--near-lu-budget 40` GB, checked before the factorisation against `expected_fill × nnz(Z_n) × 24 B` (`--near-lu-fill`: 1 at U, the U fill 7.26 at 16 µm) and after it against the factor memory (nnz(L + U) × 24 B + permutations) |
| Jacobi | rerun at U only (idle-timing reference); 16 / 24 µm: WP-PML5b values from the WP-PC1 table |

### Rough Ag, PML route

| 2N | L | variant | blocks (unknowns min / mean / max) or nnz / fill | P memory | setup | iterations | ratio to parent | ratio to Jacobi | GMRES (orth) | matvec | Krylov basis | R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 816 | 3.75 µm | Jacobi | — | 0.4 MB | 0 | 602 | 3.27 | 1 | 67.8 s (6.3) | 0.098 s | 0.22 GB | 0.97833919 |
| | | parent | 64 (322 / 357 / 384) | 131 MB | 0.13 s | 184 | 1 | 0.306 | 20.1 s (0.5) | 0.098 s | 0.07 GB | 0.97833914 |
| | | grandparent | 16 (1 350 / 1 426 / 1 536), far-zero 59.7 % | 522 MB | 0.70 s | 99 | 0.538 | 0.164 | 12.2 s (0.1) | 0.099 s | 0.04 GB | 0.97833914 |
| | | **nearlu** | nnz(Z_n) 1.697·10⁷, nnz(L + U) 1.232·10⁸, fill 7.26 | 2 958 MB | 67.5 s (LU 66.7 s) | **20** | **0.109** | 0.033 | 4.9 s (0.0) | 0.097 s | 0.01 GB | 0.97833914 |
| 152 960 | 16 µm | Jacobi (5b) | — | — | — | 1 715 | 3.30 | 1 | 1 829 s | 0.81 s | 4.20 GB | 0.97780527 |
| | | parent | 256 (560 / 598 / 640) | 1 465 MB | 1.75 s | 520 | 1 | 0.303 | 448.3 s (32.2) | 0.766 s | 1.28 GB | 0.97780528 |
| | | grandparent | 64 (2 320 / 2 390 / 2 480), far-zero 60.9 % | 5 853 MB | 11.3 s | 335 | 0.644 | 0.195 | 323.7 s (13.6) | 0.760 s | 0.82 GB | 0.97780528 |
| | | nearlu (budget 40 GB) | nnz(Z_n) 1.983·10⁸, nnz(L + U) 2.762·10⁹, fill 13.93 | 66 292 MB > 40 000 MB | 4 977 s | **refused** (post-check) | — | — | — | — | — | — |
| | | **nearlu** (budget 75 GB, extra run ²) | nnz(Z_n) 1.983·10⁸, nnz(L + U) 2.762·10⁹, fill 13.93 | 66 292 MB | 4 931 s (LU 4 921 s) | **44** | **0.085** | 0.026 | 149.5 s (0.2) | 0.755 s | 0.11 GB | 0.97780528 |
| 300 160 | 24 µm | Jacobi (5b) | — | — | — | 2 679 | 3.17 | 1 | 6 540 s (≈ 25 %) | 1.88 s | 12.9 GB | 0.97948552 |
| | | parent | 1 024 (266 / 293 / 322) | 1 413 MB | 1.46 s | 844 | 1 | 0.315 | 1 435.7 s (169.3) | 1.525 s | 4.06 GB | 0.97948552 |
| | | grandparent | 256 (1 120 / 1 173 / 1 232), far-zero 60.9 % | 5 637 MB | 8.0 s | 518 | 0.614 | 0.193 | 915.6 s (64.6) | 1.512 s | 2.49 GB | 0.97948552 |
| | | nearlu | not run: the 16 µm fill 13.93 gives an estimate of 65 GB > 40 GB, and the 16 µm factorisation took 83 min > 1 h (the brief's condition for a 24 µm attempt: estimate ≤ 40 GB and the 16 µm factorisation under 1 h) | | | | | | | | | |

² Beyond the work package's plan: the 40 GB run was refused by the post-check after the full factorisation (as planned), which
left the gate without a second `nearlu` size (lead decision, 2026-10-31: the 40 GB cap of ADR 0012 was a guess; the
machine has 128 GB and the exponent is the purpose of the package), so the same process was rerun with
`--near-lu-budget 75 --near-lu-fill 13.93` (pre-check 66.3 GB passed; the process peaked at 86.4 GB). Same operator
options, same idle machine; parent / grandparent of this size come from the first 16 µm process.

Assembly: 162 s (U), 675 s (16 µm), 1 238 s (24 µm); process peaks 5.9 GB (U, with the 3 GB LU), 22.1 GB (16 µm
without the LU; 84.7 GB working set during the refused factorisation, 86.4 GB in the 75 GB run), 29.2 GB (24 µm).

The 16 µm factorisation (sequential Eigen SparseLU, 1 296 entries per row of Z_n) took 4 977 s including the sum and
the conversions; its factors need **66.3 GB**, i.e. a fill factor 13.93 against 7.26 at U (the fill grows with the
size, as expected for the supernodal LU of a surface near field with COLAMD ordering: the separators of a 2D-like
pattern get wider). The pre-check with the U fill (34.6 GB) passed, the post-check refused, as designed.

### Rough Si, PML route (U, 2N = 22 816; control)

| variant | blocks or nnz / fill | P memory | setup | iterations | ratio to parent | GMRES | matvec | R | dR vs parent |
|---|---|---|---|---|---|---|---|---|---|
| none (WP-PC1) | — | — | — | 121 | 2.52 | 396.8 s ¹ | 2.55 s ¹ | 0.38713565 | −4.1·10⁻⁷ |
| parent | 64 (322 / 357 / 384) | 131 MB | 0.12 s | 48 | 1 | 47.9 s | 0.952 s | 0.38713581 | 0 |
| grandparent | 16 (1 350 / 1 426 / 1 536), far-zero 59.7 % | 522 MB | 0.69 s | 41 | 0.854 | 41.4 s | 0.972 s | 0.38713580 | −7.6·10⁻⁹ |
| **nearlu** | nnz 1.697·10⁷, fill 7.24 | 2 948 MB | 67.4 s | **13** | 0.271 | 15.3 s | 1.001 s | 0.38713579 | −5.5·10⁻⁸ |

¹ WP-PC1, loaded machine.

### Results unchanged (R against WP-PML5b / WP-PC1)

| case | values of this study | reference | largest difference |
|---|---|---|---|
| Ag U | Jacobi 0.97833919, parent / grandparent / nearlu 0.97833914 | WP-PML5b / PC1 Jacobi 0.97833919, PC1 parent 0.97833914 | 0 against the same variant; 5.5·10⁻⁸ (nearlu vs Jacobi; dR −5.48·10⁻⁸, far-field L2 1.2·10⁻⁷) |
| Ag 16 µm | parent / grandparent 0.97780528 , nearlu (75 GB run) 0.97780528 | 5b 0.97780527, PC1 0.97780528 | ≤ 10⁻⁸ (dR grandparent vs parent +1.1·10⁻⁹) |
| Ag 24 µm | parent / grandparent 0.97948552 | 5b / PC1 0.97948552 | < 10⁻⁸ (dR +1.1·10⁻⁹) |
| Si U | parent 0.38713581, grandparent 0.38713580, nearlu 0.38713579 | PC1 parent 0.38713581 | 2·10⁻⁸ (5.5·10⁻⁸ dR; none vs parent in PC1: dR −4.1·10⁻⁷) |

All differences are below 10⁻⁷ (GMRES tolerance 10⁻⁷, true residuals 5–9·10⁻⁸).

### Growth

Least-squares exponents of ln its over ln 2N (three Ag sizes): Jacobi 0.57, parent 0.58, **grandparent 0.64**.
`nearlu` over the two sizes where it was measured (U and 16 µm): 20 → 44 iterations, exponent
ln(44 / 20) / ln(152 960 / 22 816) = **0.41** (parent over the same two sizes: 0.55, grandparent 0.64, Jacobi 0.55).
The exact near-field inverse's advantage over the one-level-up blocks grows with the size (9.2× at U, 11.8× at
16 µm), while the grandparent's stays at 1.6–1.9× (1.86× / 1.55× / 1.63×). Two points only: the `nearlu` exponent is an
estimate, not a fit.

### ADR 0012 gate 2

| Criterion (ADR 0012 decision 2) | Measured | Met? |
|---|---|---|
| `nearlu` ≤ 0.5× the iterations of the one-level-up blocks at the largest Ag size where it was measured | 16 µm (extra 75 GB run): 44 vs 520 = **0.085×** (limit 260); U: 20 vs 184 = 0.109× (limit 92) | **yes** (within the planned 40 GB budget only U was measured; met there too) |
| fitted exponent of the `nearlu` iterations < 0.5 | **0.41** (U and 16 µm; two points) | **yes** (needs the extra run; not evaluable from U alone) |
| `grandparent` ≥ 2× fewer iterations than parent within 2× its memory | 1.86× / 1.55× / 1.63× at 4.0× / 4.0× / 4.0× the memory (522 / 131 MB, 5 853 / 1 465 MB, 5 637 / 1 413 MB) | **no** (neither condition) |

**The numbers point to ADR 0012 outcome "gate met → WP-SC2"** (the Schur-complement approximation of Z_n⁻¹): the
ceiling is ≈ 12× over the current default at 16 µm and grows with the size. The grandparent blocks do not qualify as
the new default (gain < 2×, 4× memory, steeper growth), and outcome 3 (closing options (a), (c), (d)) is not
indicated. No default was changed; the decision is the lead's.

### Assessment

- **The ceiling is not a usable preconditioner itself.** At 16 µm the exact LU costs 4 931 s of sequential setup and
  66 GB (setup + GMRES 5 081 s against 450 s for the parent blocks, 66 GB of factors against their 1.5 GB), and at
  24 µm the estimate from the 16 µm fill (65 GB) exceeds the budget. The value of `nearlu` is the bound: WP-SC2's
  own gate (≥ 2× fewer iterations than the parent blocks at 2N = 3·10⁵ with a lower total time and ≤ 2× their memory)
  asks for a small fraction of the measured 12×.
- **Where the gain sits (inferred, not profiled).** Coarser box blocks capture little (two levels up: 1.6× for 4× the
  memory, and 61 % of their entries are far zeros), while the global near inverse captures 9–12×: the useful part is
  the near coupling across box boundaries, which every box partition cuts, not larger local blocks. A Schur / global
  sparse approximation of Z_n keeps that coupling; a block-Jacobi refinement does not.
- **Growth.** With the exact near inverse the iteration growth flattens (≈ N^0.41 from two sizes against N^0.55–0.58
  for Jacobi and the blocks), so the near field does carry part of the growth; the remaining growth is consistent with
  the long-range (SPP) coupling outside the near field that ADR 0011 / 0012 anticipated.
- **Si control.** nearlu 13, grandparent 41, parent 48 iterations (3.7× / 1.17×): the same ordering, smaller
  factors, as Si's local blocks already work well.
- **Costs and memory.** The fill grows with the size (7.26 at U, 13.93 at 16 µm), so the pre-check with a small-case
  fill under-estimates (34.6 GB estimated, 66.3 GB actual); the post-check made the failure explicit, as designed.
  One `nearlu` apply at 16 µm ≈ 2.6 s (sequential solve over 66 GB of factors; GMRES 149.5 s for 44 iterations
  minus the matvecs), comparable to 3 matvecs.
- **Idle timings.** This machine was idle (WP-PC1's was loaded): Jacobi at U 67.8 s (WP-PC1 114 s), parent at 16 µm
  448 s (1 023 s), parent at 24 µm 1 436 s (2 915 s); the 24 µm assembly took 1 238 s (3 721 s).

## WP-SC2: near-field Schur-complement preconditioner (ADR 0012 decision 2)

ADR 0012 decision 2: the Schur-complement approximation of the near-field inverse that keeps the whole near pattern
in the J–M coupling. Library code `solver::SchurComplementPreconditioner` (`solver/schur_preconditioner.hpp`),
`make_schur_preconditioner` and `PreconditionerKind::Schur` (`simulation.hpp`; explicit only, compression "mlfmm"),
Python `preconditioner="schur"`; tests `tests/unit/test_schur_preconditioner.cpp`, the Schur case of
`tests/validation/test_block_preconditioner_mie.cpp`, `tests/python/test_simulation.py::test_schur_preconditioner`;
driver variant `schur` (box level `--block-level`, default 1) plus two diagnostics `nearlu_bc` / `nearlu_ad` and the
`--near-norms` line of `benchmarks/preconditioner_study.cpp`; raw lines in `preconditioner_rows.txt` (section
WP-SC2). **Stage 1 failed the early gate; stage 2 (16 / 24 µm) was not run, as the brief prescribes.** No default was
changed.

### Setup

| Item | Value |
|---|---|
| Machine | 12-core desktop, 128 GB RAM, win-release, idle machine (one study process at a time, no builds or tests during the runs; measured 2026-10-31) |
| Cases | rough Ag PML `u_case()` at L = 3.75 µm (U, 2N = 22 816) and the rough Si U control (`--material si`); compression "mlfmm", d₀ = 3, ICTF, full GMRES (CGS2), left preconditioning, tolerance 10⁻⁷, one operator and right-hand side per process for all variants (as WP-PC1 / WP-SC1) |
| `schur` | Z_n = Z_rr's near part + the near ring (read in place from the operator: B and C are not copied). Ã = block LU (PartialPivLU) of A = Z_n[J, J] over the J unknowns of each box; S̃ = D − C Ã⁻¹ B on the pattern of D, formed **only on its M diagonal blocks** (the block LU of S̃ never uses the rest), block LU over the M unknowns of each box; apply y₁ = Ã⁻¹ r₁, y₂ = S̃⁻¹ (r₂ − C y₁), x₁ = y₁ − Ã⁻¹ B y₂, x₂ = y₂. Boxes one level above the leaves (as `parent`) and, in a second process, the leaves (as `leaf`) |
| `nearlu_bc` (diagnostic) | exact sparse LU of Z_n **without the inter-box entries of A and D** (A, D box-diagonal on the `--block-level` boxes, B and C full): the practical ceiling of the implemented Schur variant, whose Ã and S̃_b are built from the box blocks of A and D — it bounds only variants of that kind (an exact or incomplete factorisation of S̃ on the full pattern of D keeps D's inter-box entries and is not bounded by it); iteration counts are not strictly monotone in the approximation quality, so "ceiling" is empirical |
| `nearlu_ad` (diagnostic) | exact sparse LU of Z_n **without the inter-box entries of B and C** (A and D full, B and C box-diagonal) |
| `--near-norms` | Frobenius norms of A, B, C, D of Z_n, split into same-box and inter-box entries of the chosen level |
| References | `parent` / `leaf` rerun in the same processes; `nearlu` (the exact near LU, WP-SC1) rerun in the Ag level-1 process; Jacobi from the WP-SC1 record (602 at U, idle) |

### Rough Ag, PML route (U, 2N = 22 816)

| boxes | variant | blocks (unknowns min / mean / max) or nnz / fill | P memory | setup (split) | iterations | ratio to parent / leaf | ratio to Jacobi | ratio to nearlu | GMRES (orth) | R | dR vs parent / leaf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| one level up | parent | 64 (322 / 357 / 384) | 130.8 MB | 0.11 s | 184 | 1 | 0.306 | 9.2 | 19.9 s (0.5) | 0.97833914 | 0 |
| | **schur** | J 64 (161 / 178 / 192), M 64 (161 / 178 / 192); nnz(B) = nnz(C) = 4.24·10⁶; 484 coupling pairs (16.4 GFlop) | **66.1 MB** | 0.30 s (A LU 0.02, S̃ formation 0.25, S̃ LU 0.02) | **184** | **1.00** | 0.306 | 9.2 | 21.3 s (0.5) | 0.97833913 | −1.0·10⁻⁸ |
| | nearlu | nnz(Z_n) 1.697·10⁷, fill 7.26 | 2 958 MB | 66.8 s | 20 | 0.109 | 0.033 | 1 | 4.9 s | 0.97833914 | −3.1·10⁻⁹ |
| | nearlu_bc (diag.) | nnz 1.256·10⁷, fill 7.18 | 2 165 MB | 38.5 s | **184** | 1.00 | 0.306 | 9.2 | 36.8 s | 0.97833913 | −9.7·10⁻⁹ |
| | nearlu_ad (diag.) | nnz 1.256·10⁷, fill 7.12 | 2 146 MB | 37.7 s | **21** | 0.114 | 0.035 | 1.05 | 4.3 s | 0.97833914 | −2.0·10⁻⁹ |
| leaves | leaf | 256 (66 / 89 / 96) | 33.2 MB | 0.08 s | 329 | 1 | 0.547 | 16.5 | 35.2 s (1.8) | 0.97833915 | 0 |
| | **schur** (level 0) | J 256 (33 / 45 / 48), M 256 (33 / 45 / 48); 2 116 coupling pairs (3.1 GFlop) | 17.3 MB | 0.13 s (0.01 / 0.11 / 0.01) | **327** | **0.994** | 0.543 | 16.4 | 37.8 s (1.7) | 0.97833915 | +1.1·10⁻⁹ |
| | nearlu_bc (diag.) | nnz 9.51·10⁶, fill 7.17 | 1 636 MB | 23.4 s | 327 | 0.994 | 0.543 | 16.4 | 58.2 s | 0.97833915 | +9.8·10⁻¹⁰ |
| | nearlu_ad (diag.) | nnz 9.51·10⁶, fill 8.21 | 1 875 MB | 28.7 s | 21 | 0.064 | 0.035 | 1.05 | 4.1 s | 0.97833914 | −6.9·10⁻⁹ |

Matvec 0.096–0.101 s in all runs (idle; WP-SC1 0.098 s). Assembly 161.7 s / 161.0 s; process peaks 5.9 GB (with the
exact LU) and 4.6 GB.

### Rough Si, PML route (U, 2N = 22 816; control)

| variant | P memory | setup | iterations | ratio to parent | GMRES | R | dR vs parent |
|---|---|---|---|---|---|---|---|
| parent | 130.8 MB | 0.12 s | 48 | 1 | 49.3 s | 0.38713581 | 0 |
| **schur** | 66.1 MB | 0.34 s (A LU 0.04, S̃ formation 0.26, S̃ LU 0.03) | **48** | **1.00** | 49.3 s | 0.38713581 | +1.5·10⁻⁹ |
| nearlu_bc (diag.) | 2 029 MB | 36.1 s | 48 | 1.00 | 52.4 s | 0.38713581 | +1.4·10⁻⁹ |
| nearlu_ad (diag.) | 2 061 MB | 35.0 s | 14 | 0.29 | 16.2 s | 0.38713579 | −5.1·10⁻⁸ |

(WP-SC1: nearlu 13 iterations on Si U.) Assembly 619.8 s, matvec 0.98–1.01 s, peak 10.9 GB.

### Near-field parts (`--near-norms`, Frobenius norms of Z_n, ICTF rows)

| case, boxes | A same-box / inter-box | B same / inter | C same / inter | D same / inter | ‖B‖‖C‖ / (‖A‖‖D‖) |
|---|---|---|---|---|---|
| Ag U, one level up | 1.40·10⁻¹² / 5.05·10⁻¹³ | 1.14·10⁻¹⁶ / 1.08·10⁻¹⁶ | 4.48·10⁻¹² / 4.27·10⁻¹² | 1.07·10⁻¹² / 1.60·10⁻¹³ | 6.1·10⁻⁴ |
| Ag U, leaves | 1.32·10⁻¹² / 6.91·10⁻¹³ | 8.08·10⁻¹⁷ / 1.35·10⁻¹⁶ | 3.19·10⁻¹² / 5.31·10⁻¹² | 1.06·10⁻¹² / 2.25·10⁻¹³ | 6.1·10⁻⁴ |
| Si U, one level up | 1.31·10⁻¹² / 4.36·10⁻¹³ | 9.90·10⁻¹⁷ / 9.46·10⁻¹⁷ | 5.33·10⁻¹² / 5.10·10⁻¹² | 1.83·10⁻¹² / 3.12·10⁻¹³ | 3.9·10⁻⁴ |

The last column is invariant under the formulations' row scaling (A, B by α; C, D by β) and under a J / M column
scaling, so it measures the J–M coupling independently of the units: the ratio indicates a weak J–M coupling (an indicator, not a
measurement of ‖C Ã⁻¹ B‖ / ‖D‖, which is bounded by this ratio times ‖A‖ ‖Ã⁻¹‖ and was not accumulated). Each norm counts both stored entries of a part (Z_rr's near part and the
ring); the entry counts per part are equal (2.04·10⁶ same-box, 2.20·10⁶ inter-box at one level up).

### Results unchanged (R against WP-PML5b / WP-PC1 / WP-SC1)

Ag U: schur 0.97833913 (level 1) / 0.97833915 (leaves) against 0.97833914 / 0.97833915 for parent / leaf in the same
processes (|dR| ≤ 1.1·10⁻⁸, far-field L2 ≤ 1.2·10⁻⁷) and 0.97833914 in WP-SC1; Si U: 0.38713581 (dR +1.5·10⁻⁹,
WP-SC1 parent 0.38713581). The diagnostics agree to 5.1·10⁻⁸. All within 10⁻⁷ (true residuals 2.9–7.9·10⁻⁸). The
validation test (Ag sphere, box route, d = 1 µm, 2N = 3 840, tolerance 10⁻⁶) passes with `Schur` one level up: 261
iterations (Jacobi 210, parent 266), RCS within 4.2·10⁻⁷ of Jacobi, ε_rr 3.481·10⁻³ / 2.560·10⁻³ unchanged.
Setup transient (not in the memory row, which counts the stored preconditioner only): a column copy of B (≈ nnz(B) × 24 B ≈ 102 MB at U) plus the S̃ blocks, released after the factorisation.

### Growth

Not evaluable: stage 2 (16 / 24 µm) was not run (early gate). At U the iteration count equals the blocks', at both box
levels, so there is no gain whose growth could be measured.

### ADR 0012 decision 2 gate (WP-SC2)

| Criterion | Measured | Met? |
|---|---|---|
| Early gate of the brief: `schur` ≤ 92 iterations at Ag U (≥ 2× over parent's 184) | **184** (1.00×); leaf boxes 327 vs 329 (0.994×) | **no** |
| ≥ 2× fewer iterations than parent at 2N = 300 160 (≤ 422) | not run (stage 2 skipped); the U result and the diagnostic ceiling (below) leave no room for it | **no** (not evaluable directly; indicated) |
| lower total time (setup + GMRES) than parent | U: 0.30 + 21.3 = 21.6 s vs 0.11 + 19.9 = 20.0 s (1.08×; the apply costs two J-block and one M-block solve plus the B and C products) | **no** at U |
| preconditioner memory ≤ 2× parent's own storage | 66.1 vs 130.8 MB (0.51×; Z_n not counted, B and C read from the operator) | yes |

**The gate is not met.** The memory criterion is met with a factor four to spare, but the preconditioner does not save
a single iteration over the box blocks on Ag or on Si, at either box level.

### Assessment

- **Why the Schur complement gains nothing (measured, not inferred).** The diagnostic `nearlu_bc` — the exact inverse
  of Z_n with A and D restricted to the boxes and B, C full — needs exactly the parent / leaf iterations (184 / 327 on Ag,
  48 on Si). It is the practical ceiling of every variant whose Ã and S̃ are formed from the box blocks of A and D (exact B and
  C) — the implemented one included; an S̃ factorised on the full pattern of D is a different family (WP-SC3's
  lever is the inter-box coupling inside A and D). The complementary diagnostic
  `nearlu_ad` — A and D full, B and C restricted to the boxes — recovers the whole near-field-inverse gain (21 vs
  `nearlu`'s 20 on Ag U, 14 vs 13 on Si U). **The 9–12× of WP-SC1 lives in the inter-box near coupling of A and D**
  (the T-operator blocks J–J and M–M), not in the inter-box J–M coupling that ADR 0012 decision 2 keeps.
- **Physical reading.** The J–M blocks are the K-operator (MFIE-type) terms. On a nearly planar surface (σ = 50 nm,
  h = λ/4) the K kernel of coplanar element pairs vanishes (f_m · ((r − r′) × f_n) = 0 for f_m, f_n and r − r′ in one
  plane), so the near J–M coupling is weak: the scale-free measure ‖B‖‖C‖ / (‖A‖‖D‖) is 6·10⁻⁴ (Ag) / 4·10⁻⁴ (Si), i.e.
  the norm ratio ‖B‖ ‖C‖ / (‖A‖ ‖D‖) = 6·10⁻⁴ indicates a J–M coupling three orders of magnitude below the J–J / M–M coupling; up to a J / M scaling Z_n is close to a block
  triangular matrix, whose spectrum is that of A and D alone. The inter-box parts of A and D are 36 % and 15 % of their same-box parts (Frobenius, one
  level up): that is the coupling every box partition cuts.
- **Cost.** Setup is cheap (0.30 s at U, 16 GFlop of formation; S̃ is formed only on its 64 M blocks, 2.0·10⁶ entries,
  instead of the whole pattern of D) and memory half of parent's (two LUs of half-size blocks). The extrapolated 24 µm
  cost was ≈ 0.7 GB and minutes of setup; it was not run because the outcome at 24 µm is not expected to differ (inferred from U, not measured): the U ceiling
  `nearlu_bc` already equals parent, and the box structure is the same at every size.
- **What could still work (for the lead).** A near-field preconditioner for these systems has to approximate the
  inverse of the **inter-box** A and D couplings. `nearlu_ad` suggests that treating A and D separately and dropping B,
  C's inter-box part costs nothing in iterations; candidates are a sparse approximate factorisation (ILU / SPAI) of A
  and of D separately (each N × N with a quarter of the entries of Z_n), or overlapping box blocks (additive Schwarz
  over a box and its near neighbours) — options (c) / (d) of ADR 0012 restricted to the T blocks. The exact LUs of A
  and D would cost about what `nearlu` costs (the fill grows with the size, WP-SC1), so an approximate factorisation
  with a memory cap would be needed. Not measured here.
- **Recommendation of this package.** Keep the block-diagonal default; keep `Schur` as an explicit option (it is
  correct, tested and cheap, but brings nothing on the rough PML systems). Option (e) of ADR 0012 (Krylov recycling)
  remains the independent lever for the many-solve studies.

## WP-SC3: incomplete LU of the near blocks A and D (ADR 0012 decision 6)

ADR 0012 decision 6 (after the WP-SC2 amendment): M⁻¹ = diag(Ã⁻¹, D̃⁻¹) with Ã, D̃ incomplete LU factorisations of the
N × N near-field blocks A = Z_n[J, J] and D = Z_n[M, M]; B and C are not used. Library code
`solver::IncompleteLuPreconditioner` (`solver/incomplete_lu_preconditioner.hpp`), `make_incomplete_lu_preconditioner`,
`PreconditionerKind::IncompleteLu` and `SimulationConfig::ilut` (`simulation.hpp`; explicit only, compression "mlfmm"),
Python `preconditioner="ilut"` with the `ilut` dict; the zero-pivot guard of decision 8 in
`BlockDiagonalPreconditioner` and the Schur blocks; tests `tests/unit/test_incomplete_lu_preconditioner.cpp`, the ILUT
case of `tests/validation/test_block_preconditioner_mie.cpp`, `tests/python/test_simulation.py::test_ilut_preconditioner`;
driver variant `ilut:F:T[:amd|nat]` of `benchmarks/preconditioner_study.cpp` (built with
`-DSPECKLEBEM_BUILD_BENCHMARKS=ON`); raw lines in `preconditioner_rows.txt` (section WP-SC3). **Stage 1 and stage 2
gates are met.** No default was changed.

### Setup

| Item | Value |
|---|---|
| Machine | 12-core desktop, 128 GB RAM, win-release, idle machine (one study process at a time, no builds or tests during the runs; measured 2026-10-31). Two processes saw foreign load (the second U process and the Si U process: matvec 0.13–0.30 s instead of 0.097 s at Ag U, 1.2–3.8 s instead of 1.0 s at Si U, assembly 222 s / 830 s instead of 161 s / 620 s); their times are marked, their iterations, memory and R are load-independent |
| Cases | as WP-SC1 / WP-SC2: rough Ag PML `u_case()` at L = 3.75 µm (U, 2N = 22 816), `--patch L --map-patch L` with L = 16 / 24 µm (2N = 152 960 / 300 160); rough Si U (`--material si`) as the control; compression "mlfmm", d₀ = 3, ICTF, full GMRES (CGS2), left preconditioning, tolerance 10⁻⁷, one operator and right-hand side per process for all variants |
| Which ILUT | **own implementation** (dual-threshold ILUT, Saad 1994; row-wise IKJ elimination, no numerical pivoting). `Eigen::IncompleteLUT` was checked and rejected: it replaces an exactly zero pivot by √droptol × the row norm without telling (a silent fallback), never checks tiny pivots, drops L entries by an absolute dimensionless threshold, and finds the next pivot column by a linear search (O(len²) per row) |
| Ordering | AMD on the pattern of B + Bᵀ (`Eigen::AMDOrdering`, sequential, pattern-only), applied symmetrically per block; `nat` = the RWG order |
| Drop rule | in row i an entry w_j of the work row is dropped when \|w_j\| ≤ τ ‖b_i‖₂ (the 2-norm of the block's row i); for L entries the magnitude before the division by the pivot, \|l_ij\| \|u_jj\|. Scale-free: a row scaling of the block by a constant (PMCHWT → ICTF / MCTF) leaves the drop decisions unchanged |
| Fill limit `F` | row i of L + U keeps at most max(1, round(F × nnz_i)) entries (nnz_i = the stored entries of the block's row i, diagonal always kept; L and U entries compete by the magnitude above, ties by the smaller column): nnz(L + U) ≤ F × nnz(block) + N |
| Pivot guard | \|u_ii\| ≤ 10⁻¹⁰ ‖b_i‖₂ (also exactly zero or non-finite) and a zero row throw `std::runtime_error` naming the block and the row; a pivot is **never replaced**. A non-finite stability estimate ‖(LU)⁻¹ e‖_∞ (Chow & Saad) also throws |
| Determinism, memory | A and D are factorised in parallel (two OpenMP threads), each sequentially; `apply` = forward / back substitution per block (two threads), explicit loops: bitwise thread-count independent (tested). Memory from the capacities of the factor arrays (24 B per off-diagonal entry + 40 B per row), reserved at the bound of the fill limit and shrunk at the end; budget pre-check against the bound (default: half the available memory) and post-check |
| Variants | `parent` (blocks one level above the leaves, the PML-route default) rerun in the same process as the reference; Jacobi, `nearlu` (exact near LU) and `nearlu_ad` (exact LU with A, D full and B, C box-diagonal) from the WP-SC1 / WP-SC2 records |
| Memory rules | stage 1: factor memory ≤ 4 × nnz(A + D) × 16 B (543 MB at U, 6 346 MB at 16 µm, 6 235 MB at 24 µm; column `rule` = memory / this bound); stage 2: ≤ 25 % of the operator (op 1 525 / 13 334 / 20 070 MB) |

### Rough Ag, PML route, U (2N = 22 816): parameter sweep

nnz(A) = nnz(D) = 4 242 192 (372 per row). `ilut:F:T` = fill F, drop T, AMD.

| variant | nnz(L + U) A / D (fill) | P memory (rule) | setup (A / D; AMD) | iterations | ratio to parent | ratio to Jacobi | ratio to nearlu_ad | GMRES | total (setup + GMRES) | R |
|---|---|---|---|---|---|---|---|---|---|---|
| parent | 64 blocks (322 / 357 / 384) | 130.8 MB | 0.10 s | 184 | 1 | 0.306 | 8.8 | 19.9 s | 20.0 s | 0.97833914 |
| ilut:1:1e-3 ¹ | 3.89·10⁶ / 3.66·10⁶ (0.92 / 0.86) | 181.7 MB (0.33) | 10.38 s (10.37 / 5.42; 0.30) | 34 | 0.185 | 0.056 | 1.62 | 4.7 s ¹ | 15.1 s ¹ | 0.97833914 |
| ilut:1:1e-2 ¹ | 3.08·10⁶ / 0.81·10⁶ (0.73 / 0.19) | 93.7 MB (0.17) | 10.32 s ¹ | 33 | 0.179 | 0.055 | 1.57 | 49.1 s ¹ | — ¹ | 0.97833914 |
| ilut:2:1e-2 ¹ | 3.80·10⁶ / 0.81·10⁶ (0.90 / 0.19) | 111.1 MB (0.20) | 4.92 s (4.90 / 0.70; 0.26) | 23 | 0.125 | 0.038 | 1.10 | 11.8 s ¹ | ≈ 8 s ² | 0.97833914 |
| ilut:2:1e-3 | 6.02·10⁶ / 4.48·10⁶ (1.42 / 1.06) | 252.4 MB (0.46) | 13.29 s (13.27 / 5.23; 0.22) | 24 | 0.130 | 0.040 | 1.14 | 3.0 s | 16.3 s | 0.97833914 |
| ilut:2:1e-4 | 6.42·10⁶ / 6.21·10⁶ (1.51 / 1.46) | 303.3 MB (0.56) | 15.56 s (15.55 / 11.06) | 24 | 0.130 | 0.040 | 1.14 | 3.0 s | 18.6 s | 0.97833914 |
| **ilut:3:1e-3** | 7.45·10⁶ / 4.48·10⁶ (1.76 / 1.06) | 286.7 MB (0.53) | 17.22 s (17.21 / 5.24; 0.23) | **21** | **0.114** | 0.035 | 1.00 | 2.6 s | 19.8 s | 0.97833914 |
| ilut:3:1e-4 | 8.04·10⁶ / 7.35·10⁶ (1.89 / 1.73) | 369.6 MB (0.68) | 20.63 s (20.61 / 13.23) | 21 | 0.114 | 0.035 | 1.00 | 2.7 s | 23.3 s | 0.97833914 |
| ilut:5:1e-3 | 8.89·10⁶ / 4.48·10⁶ (2.10 / 1.06) | 321.4 MB (0.59) | 20.42 s (20.40 / 5.23) | 21 | 0.114 | 0.035 | 1.00 | 2.7 s | 23.1 s | 0.97833914 |
| ilut:5:1e-4 | 10.02·10⁶ / 7.59·10⁶ (2.36 / 1.79) | 423.0 MB (0.78) | 26.81 s (26.79 / 13.57) | 21 | 0.114 | 0.035 | 1.00 | 2.8 s | 29.6 s | 0.97833914 |
| ilut:3:1e-3:nat ¹ | 11.06·10⁶ / 3.55·10⁶ (2.61 / 0.84) | 350.9 MB (0.65) | 67.72 s ¹ (67.65 / 4.65) | 24 | 0.130 | 0.040 | 1.14 | 15.4 s ¹ | — ¹ | 0.97833914 |

¹ Second U process, foreign load (assembly 222 s instead of 161 s; matvec 0.13–0.30 s instead of 0.097 s; `ilut:1:1e-2`
also orth 28.5 s for 33 iterations): times not comparable, iterations / memory / R are. ² The 4.92 s setup measured
in the loaded process ¹ plus 23 iterations at the idle cost of the neighbouring rows (≈ 0.13 s per iteration); the
setup part is load-affected, so the total is indicative only. Smallest pivot ratios |u_ii| / ‖b_i‖₂ 0.10 (A) / 0.22 (D) in
every AMD set, condest 79–118 (A) / 33–35 (D): no breakdown, no pivot near the guard. Jacobi 602, `nearlu` 20,
`nearlu_ad` 21 iterations (WP-SC1 / WP-SC2).

**Every measured set passes the stage-1 gate** (≤ 92 iterations and memory ≤ 1 × the rule): 21–34 iterations at 0.17–0.78 of
the memory rule. Fill 3 or more with drop ≤ 10⁻³ matches `nearlu_ad`'s 21 (the exact LU with A and D full and B, C
box-diagonal; a reference, not a bound on the ILUT) at 13 % of its memory (287 MB vs 2 146 MB). The drop tolerance
mostly thins D (fill 0.19 at 10⁻², 1.06 at 10⁻³) at almost no iteration cost; the fill limit decides A. AMD beats the
natural order at fill 3 / drop 10⁻³: 21 instead of 24 iterations with less fill in A (1.76 instead of 2.61) and half
the elimination work (multiply-adds of the A block, `macs` in the ROW lines: 2.04·10⁹ with AMD vs 4.22·10⁹ natural);
the setup times are not comparable (load ¹).

### Rough Si, PML route (U, 2N = 22 816; control)

| variant | nnz(L + U) A / D (fill) | P memory | setup | iterations | ratio to parent | GMRES ¹ | R | dR vs parent |
|---|---|---|---|---|---|---|---|---|
| parent | 64 blocks | 130.8 MB | 0.14 s | 48 | 1 | 64.5 s | 0.38713581 | 0 |
| **ilut:3:1e-3** | 6.43·10⁶ / 3.19·10⁶ (1.51 / 0.75) | 231.2 MB | 22.98 s ¹ | **14** | 0.292 | 66.3 s | 0.38713579 | −4.5·10⁻⁸ |
| ilut:2:1e-3 | 5.79·10⁶ / 3.19·10⁶ (1.36 / 0.75) | 215.9 MB | 50.76 s ¹ | 14 | 0.292 | 84.7 s | 0.38713579 | −4.4·10⁻⁸ |

¹ Foreign load throughout this process (assembly 830 s instead of 620 s, matvec 1.17–3.83 s instead of 1.0 s): times
not comparable. `nearlu_ad` 14, `nearlu` 13 (WP-SC1 / WP-SC2): the ILUT matches `nearlu_ad`'s 14.

### Rough Ag, PML route, L = 16 µm (2N = 152 960)

nnz(A) = nnz(D) = 4.96·10⁷ (648 per row; the leaf rule gives 1.25 λ leaves at this size). Assembly 689 s, process peak
22.5 GB, matvec 0.740–0.755 s (idle; WP-SC1 0.755–0.766 s).

| variant | nnz(L + U) A / D (fill) | P memory (rule; % of op) | setup (A / D; AMD) | iterations | ratio to parent | ratio to Jacobi | ratio to nearlu | GMRES (orth) | total | R |
|---|---|---|---|---|---|---|---|---|---|---|
| parent | 256 blocks (560 / 598 / 640) | 1 465 MB (—; 11.0 %) | 1.98 s | 520 | 1 | 0.303 | 11.8 | 454.6 s (31.8) | 456.6 s | 0.97780528 |
| **ilut:3:1e-3** | 1.014·10⁸ / 5.91·10⁷ (2.05 / 1.19) | 3 856 MB (0.61; 28.9 %) | 532.2 s (532.0 / 122.5; 4.2) | **50** | **0.096** | 0.029 | 1.14 | 47.7 s (0.3) | 579.9 s | 0.97780528 |
| ilut:2:1e-3 | 7.87·10⁷ / 5.91·10⁷ (1.59 / 1.19) | 3 309 MB (0.52; 24.8 %) | 386.5 s (386.3 / 121.2; 3.0) | 59 | 0.113 | 0.034 | 1.34 | 54.7 s (0.5) | 441.2 s | 0.97780528 |
| ilut:2:1e-2 ³ | 4.35·10⁷ / 6.26·10⁶ (0.88 / 0.13) | 1 196 MB (0.19; 9.0 %) | 70.2 s (70.1 / 6.2; 3.0) | 68 | 0.131 | 0.040 | 1.55 | 58.3 s (0.6) | **128.5 s** | 0.97780528 |

³ Extra process after the planned runs (the cheapest U set), same options (assembly 675 s, matvec 0.744 s, peak
21.8 GB).

### Rough Ag, PML route, L = 24 µm (2N = 300 160)

nnz(A) = nnz(D) = 4.87·10⁷ (324 per row; 0.875 λ leaves). Assembly 1 227 s (1 241 s in the extra process), process
peak 29.3 GB, matvec 1.448–1.470 s (idle; WP-SC1 1.525 s).

| variant | nnz(L + U) A / D (fill) | P memory (rule; % of op) | setup (A / D; AMD) | iterations | ratio to parent | ratio to Jacobi | GMRES (orth) | Krylov basis | total | R |
|---|---|---|---|---|---|---|---|---|---|---|
| parent | 1 024 blocks (266 / 293 / 322) | 1 413 MB (—; 7.0 %) | 1.36 s | 844 | 1 | 0.315 | 1 424.8 s (168.7) | 4.06 GB | 1 426.2 s | 0.97948552 |
| **ilut:3:1e-3** | 1.030·10⁸ / 6.46·10⁷ (2.11 / 1.33) | 4 026 MB (0.65; 20.1 %) | 328.8 s (328.5 / 76.0; 3.9) | **111** | **0.132** | 0.041 | 190.6 s (3.5) | 0.54 GB | 519.4 s | 0.97948552 |
| ilut:2:1e-3 | 7.95·10⁷ / 6.45·10⁷ (1.63 / 1.32) | 3 460 MB (0.55; 17.2 %) | 229.8 s (229.6 / 75.0; 2.8) | 137 | 0.162 | 0.051 | 231.1 s (5.1) | 0.66 GB | 460.9 s | 0.97948552 |
| ilut:2:1e-2 ³ | 6.77·10⁷ / 1.20·10⁷ (1.39 / 0.25) | 1 918 MB (0.31; 9.6 %) | 99.8 s (99.6 / 7.3; 2.9) | 140 | 0.166 | 0.052 | 233.2 s (5.5) | 0.68 GB | 333.0 s | 0.97948552 |

Smallest pivot ratios 0.084–0.094 (A) / 0.22 (D) at 16 and 24 µm, condest 93–178 (A) / 32–36 (D): stable at every size.
The setup is the A block alone (the D block, factorised in parallel, takes 23–40 % of A's time at drop 10⁻³ and
7–14 % at 10⁻²); the AMD ordering is ≈ 1 % of it. The elimination runs at ≈ 1.2·10⁸ complex multiply-adds per second
(`macs` in the ROW lines: 6.3·10¹⁰ for A at 16 µm in 532 s), single-threaded.

Setup transient (not in the P memory column, which holds only the stored factors): during the setup each block
reserves its fill bound up front (the ROW lines' `bound`: 7 018.6 MB for A and D together at 24 µm, fill 3; 7 141.3 MB
at 16 µm), the final `shrink_to_fit` briefly holds the old and the new buffer of an array, and each block is copied
at 24 B per entry (2 × 4.87·10⁷ × 24 B ≈ 2.3 GB at 24 µm, ≈ 2.4 GB at 16 µm); before the reservation the AMD work
(≈ 3 × 16 B per entry of B + Bᵀ) coexists with the copies. The peak during the setup is therefore ≈ 2–2.5× the stored
memory ((7.0 + 2.3) GB against 4.03 GB at 24 µm, (7.1 + 2.4) GB against 3.86 GB at 16 µm). The budget pre-check
covers only the bound, not the copies.

### Results unchanged (R against WP-PML5b / WP-PC1 / WP-SC1)

Ag U: every ILUT set 0.97833914, as parent here and in WP-SC1 (|dR| ≤ 3.7·10⁻⁹, far-field L2 ≤ 1.3·10⁻⁷). Ag 16 µm:
0.97780528 for all variants (dR ≤ 2.0·10⁻⁹; WP-SC1 0.97780528). Ag 24 µm: 0.97948552 for all variants (dR ≤
3.9·10⁻¹⁰; WP-SC1 0.97948552). Si U: 0.38713579 against parent 0.38713581 (dR −4.5·10⁻⁸; `nearlu` in WP-SC1
0.38713579). All within 10⁻⁷ (true residuals 3.1–7.1·10⁻⁸). The validation case (Ag sphere, box route, d = 1 µm,
2N = 3 840, tolerance 10⁻⁶) passes with the default set (fill 3, drop 10⁻⁴): **63 iterations** against Jacobi's 210
(parent 266, Schur 261), RCS within 4.7·10⁻⁷ of Jacobi, ε_rr 3.481·10⁻³ / 2.560·10⁻³ unchanged; setup 8.2 s, 88 MB —
the first of the measured near-field preconditioners (leaf, parent, Schur) that beats Jacobi on the box route.

### Growth

Least-squares exponents of ln its over ln 2N (three Ag sizes): `ilut:3:1e-3` **0.60** (21 / 50 / 111), `ilut:2:1e-3`
0.63 (24 / 59 / 137), `ilut:2:1e-2` 0.67 (23 / 68 / 140), parent 0.58, Jacobi 0.57; WP-SC1 `nearlu` 0.41 (U and
16 µm only). Over U → 16 µm alone the fill-3 ILUT grows like N^0.46 (parent 0.55, nearlu 0.41), over 16 → 24 µm like
N^1.18 (parent 0.72). The near pattern is not
the same operator at the three sizes: the MLFMM leaf rule gives 0.97 / 1.25 / 0.875 λ leaves, i.e. 372 / 648 / 324
near entries per row of A, so the 16 µm point carries a wider near field than the 24 µm point and the fitted exponent
mixes the preconditioner's growth with the leaf rule (the same caveat as WP-SC1's `nearlu`). The advantage over parent
is 8.8× / 10.4× / 7.6× (fill 3) — a large constant factor, not a smaller exponent.

### ADR 0012 decision 6 gates (WP-SC3)

| Criterion | Measured | Met? |
|---|---|---|
| Stage 1: ≤ 92 iterations at Ag U with factor memory ≤ 4 × nnz(A + D) × 16 B (543 MB) | every measured set: 21–34 iterations at 94–423 MB (0.17–0.78 of the rule); best `ilut:3:1e-3` 21 iterations, 287 MB (0.53) | **yes** |
| Stage 2: ≥ 2× fewer iterations than parent at 2N = 300 160 (≤ 422) | `ilut:3:1e-3` **111** (7.6×), `ilut:2:1e-3` 137 (6.2×), `ilut:2:1e-2` 140 (6.0×) vs 844 | **yes** |
| Stage 2: lower total time (setup + GMRES) at 2N = 300 160 | 519.4 s / 460.9 s / **333.0 s** vs 1 426.2 s (0.36× / 0.32× / 0.23×) | **yes** |
| Stage 2: preconditioner memory ≤ 25 % of the operator (≤ 5 GB at 24 µm; op 20 070 MB) | 4 026 MB (20.1 %) / 3 460 MB (17.2 %) / 1 918 MB (9.6 %) | **yes** |

At 16 µm (not a gate size) the fill-3 set is slower in total than parent (579.9 s vs 456.6 s: the sequential A
factorisation of the wide 1.25 λ near pattern, 532 s, outweighs the 407 s saved in GMRES) at 28.9 % of the operator,
`ilut:2:1e-3` is about equal (441.2 s, 24.8 %), and `ilut:2:1e-2` is 3.6× faster (128.5 s, 9.0 %). The gates are
defined at 2N = 300 160, where all three sets pass everything.

### Assessment

- **The ILUT recovers the A / D near-field-inverse gain where it was measured.** At U it matches `nearlu_ad`'s
  iterations (Ag 21 = 21, Si 14 = 14); at 16 µm it needs 50 against `nearlu`'s 44 (`nearlu_ad` not measured there),
  with 10 % (U) to 6 % (16 µm) of the exact near LU's memory and 26 % (U) to 11 % (16 µm) of its setup. The WP-SC2
  diagnosis (the gain lives in the inter-box T-operator coupling inside A and D) is confirmed constructively: dropping
  B and C and approximating A and D separately costs no iterations at U against `nearlu_ad`.
- **Cost structure.** GMRES time falls 7.5× at 24 µm (1 425 → 191 s) and the Krylov basis 7.5× (4.06 → 0.54 GB); the
  setup (329 s for fill 3) becomes the larger part. It is the sequential elimination of the A block; D is 3–14× cheaper
  because most of its entries fall below the drop threshold (fill 0.13–0.25 at drop 10⁻²). At fill 2, the drop tolerance
  10⁻² instead of 10⁻³ cuts the setup 2.3× (24 µm) / 5.5× (16 µm) and the memory 1.8× / 2.8× for 1.02× / 1.15× the
  iterations (U: 0.96×):
  `ilut:2:1e-2` gives the best total at both large sizes (128.5 s vs 456.6 s at 16 µm, 333.0 s vs 1 426.2 s at 24 µm:
  3.6× / 4.3× faster than parent) with 1.2 / 1.9 GB (9.0 % / 9.6 % of the operator).
- **Growth is not changed.** The exponent stays ≈ 0.6–0.7 (constant factor 6–10× over parent); the remaining growth
  is consistent with the long-range coupling outside the near field (ADR 0011 / 0012), and the 16 → 24 µm step is
  distorted by the leaf rule's narrower near pattern at 24 µm.
- **Robustness.** No breakdown in 18 ILUT setups (36 block factorisations) of indefinite complex blocks (smallest pivot ratio 0.084 against the
  10⁻¹⁰ guard, condest ≤ 178); the guard never fired, the fallback of decision 7 (block SPAI) is not needed.
- **Recommendation (no default changed; the lead decides).** The stage-1 and stage-2 gates of ADR 0012 decision 6 are
  met, which by the ADR switches the "mlfmm" PML-route default. Recommended set for that default: **drop 10⁻², fill 2**
  (fastest total at 16 and 24 µm, < 10 % of the operator, at most 1.36× the iterations of fill 3 / drop 10⁻³ at every
  size: 23 / 68 / 140 vs 21 / 50 / 111); drop 10⁻³, fill 3 when the iteration count matters more than the setup (many
  right-hand sides per operator). The current library default (fill 3, drop 10⁻⁴) is safe but not the cheapest and
  should follow the choice. Caveats: (1) the setup is sequential per block and grows with the near row length (the
  fill-3 set lost to parent at 16 µm); a faster elimination kernel (the inner update runs at ≈ 1.2·10⁸ complex
  multiply-adds per second) or a parallel A factorisation (level scheduling) would remove that risk; (2) the box route
  (Ag sphere: 63 vs Jacobi 210) is a second candidate, measured only at the validation size.
