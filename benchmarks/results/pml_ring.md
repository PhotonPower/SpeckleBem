# Fast path of the PML route: MLFMM on the real functions + PML ring (WP-PML5a exact, WP-PML5b ACA far blocks, WP-PML5c ring assembly)

ADR 0008 amendment 2026-10-18 (decisions 1–3a, 5, 6; 3b in section "5b", 4 in section "5c", the combined code in section "5b + 5c integrated" at the end), docs/05 "PML route", row 7h. Library code
`include/specklebem/compression/mlfmm/pml_ring.hpp` (`mlfmm::PmlRingOperator`, `classify_stretched`,
`split_near_pattern`, `estimate_pml_ring`) and `MlfmmOperator` with a stretch; case code
`tests/support/pml_ring_support.hpp`; tests `tests/unit/test_pml_ring.cpp`, `tests/validation/test_pml_ring.cpp`,
`tests/validation_large/test_pml_ring_large.cpp`; driver `benchmarks/pml_ring.cpp`; raw lines `pml_ring_rows.txt`.
Measured 2026-10-18 on a 12-core desktop with 128 GB RAM (OpenBLAS LU), win-release build, one heavy run at a
time. Units: MB and GB here and in the ROW lines (`pml_ring::row_line`, `Simulation::report()`) are 10⁶ / 10⁹
bytes; the library's log lines (`PmlRingOperator`, `MlfmmOperator`, budget messages) print MiB (2²⁰ bytes),
so their values are 4.6 % smaller (e.g. far ring 3 652 MB = 3 483 MiB).

## Summary

- **7h met at 2N = 2.3·10⁴ (rough Ag, the 7c U case)** and on flat Ag at 2N = 2 816 / 4 592: matvec against the
  dense PML matrix 1.5·10⁻⁵ (U, d₀ = 3; flat 3.2–3.9·10⁻⁴) and 1.4·10⁻⁷ (U, d₀ = 5, review round) / 1.2·10⁻⁸ /
  4.9·10⁻⁸ (flat, d₀ = 5); GMRES solve against the dense LU: R within 4·10⁻⁸ (d₀ = 3) / 5.5·10⁻⁸ (d₀ = 5)
  relative, windowed far field L2 3.3·10⁻⁶ / 1.2·10⁻⁷ (U) — the criteria are 10⁻³ /
  10⁻⁵, 10⁻⁴ and 10⁻³. The ring is exact; the difference comes from the real–real MLFMM far part (and GMRES at
  tolerance 10⁻⁷).
- **Memory:** 4.6 GB operator, 4.9 GB peak for U against 8.3 / 8.5 GB dense (0.55×); the exact far ring
  (3.65 GB, 64 bytes per stored pair, N_s × N_far) is 79 % of the fast operator. The model is O(N_s · N): with
  W_pml = 2λ the ring is 32 % of the dense matrix at L = 7.5 µm (2N = 5·10⁴) and 20 % at L = 15 µm, but 60 GB
  there; 5a ends at 2N ≈ 1.2·10⁵ within 50 GB. The scalable form is WP-PML5b (ACA far blocks).
- **Time:** ring + MLFMM setup 905 s against 640 s dense assembly (1.41×), plus 50 s diagonal; GMRES (ICTF +
  Jacobi, tolerance 10⁻⁷) 602 iterations in 125 s against the 56 s LU. The near ring (380 s) evaluates each
  stretched touching pair in both orderings (`op::assemble_sparse_stretched`, row-wise); the far ring (522 s)
  evaluates every pair of a stretched test triangle with every far source triangle (stretched–stretched pairs
  twice). WP-PML5c (canonical-pair cache, ADR 0008 amendment decision 4) targets ≤ dense / 2.
- **5b (WP-PML5b, section "5b: ACA far ring"):** ACA far ring (H-matrix row clusters, η = 2, tolerance
  10^-(d₀+1)): 7h kept (U matvec 2.08·10⁻⁵ / 2.15·10⁻⁷ at d₀ = 3 / 5), far ring 571 MB instead of 3.65 GB at U,
  operator 1.52 GB (0.18× dense); runs up to 2N = 3·10⁵ in 33.6 GB peak.
- **5c (WP-PML5c, section "5c: ring assembly"):** near ring 0.41× the dense assembly of the same pairs at U
  (gate ½ met; 0.26× in the review-round rerun with the complete key; 0.14–0.24× the 5a near ring) by evaluating each unordered stretched pair once (canonical-pair
  cache) and the touching pairs of the PML strips once per bit-exact translation class; the matrix is bitwise the
  5a one; the Jacobi diagonal is read from the near parts (50 s → 0.007 s); the 7h rerun reproduces every 5a digit.
  With the exact far ring of 5a the far ring was then ≈ 87 % of the setup (compressed by 5b; combined picture
  below).
- **Combined (5b + 5c, the current code; section "5b + 5c integrated"):** near ring of 5c, far ring of 5b's ACA.
  At U (d₀ = 3) `Simulation::assemble()` takes 211 s against 678 s for the dense assembly of the same process
  (0.31×; near ring 89 s, ACA far ring 119 s, MLFMM 2.7 s, diagonal 0.002 s), the operator 1.52 GB, peak 1.83 GB,
  matvec 0.10 s against 0.27 s dense, and every printed 5b accuracy digit is reproduced (matvec 2.081·10⁻⁵, dR
  +4.76·10⁻⁸, l2 3.33·10⁻⁶, 602 iterations); d₀ = 5 matvec 2.147·10⁻⁷, operator build 292 s.
- **5d (WP-PML5d, section "5d: pair memo"):** the ACA entry generation evaluated each far triangle pair 1.5×
  (U, 2N = 59 600) / 1.9× (Si) on average; a per-block triangle-pair memo (16 MiB per block by default) cuts the
  kernel evaluations by 31 / 30 / 40 % and the far-ring build to ≈ 0.6× (Ag, in-process A/B on a loaded machine;
  Si: CPU 0.61×, wall 0.94×, bound by its longest row cluster), with
  the far ring bitwise unchanged (product hashes equal to `main`) and +0.05 GB process peak.

## Setup

| Item | Value |
|---|---|
| U (validation-large) | `pml_rough_ag::u_case()`: rough Ag, σ = 50 nm, Lc = 500 nm, seed 1, L = 3.75 µm, taper 0.5, gap 0.5, W_pml = 1 µm = 2λ, σ_max 4, m 2, h = λ/4, w₀ = λ, normal incidence, dark band 1e-4 (ratio 3.3·10⁻⁶); 2N = 22 816, N_s = 5 152 (45.2 %) |
| flat CI (validation) | `pml_ring::flat_ag_ci_case()`: flat Ag, L = 1 µm, taper 0.25, gap 0.5 µm = λ (the 7a window width), W_pml = λ, σ_max 2 (the W_pml = λ configuration of WP-PML4b), h = λ/4, w₀ = λ, dark band relaxed to 0.05 (fast vs dense on the same discrete problem, not against Fresnel); 2N = 4 592, N_s = 1 136 (49.5 %) |
| flat matvec case (validation) | flat Ag, L = 1 µm, taper 0.25, gap 0.25, W_pml = 0.375 µm, σ_max 1, h = λ/4, kernel target accuracy 10⁻⁷ (dense real–real far entries as an oracle at d₀ = 5); 2N = 2 816, N_s = 672 |
| Fast path | `Simulation` PML constructor with compression "mlfmm" (default MLFMM parameters: automatic leaf rule, d₀ = 3 unless stated, leaf quantile 1), GMRES full (no restart), tolerance 10⁻⁷, the recommended Jacobi preconditioner (`op::assemble_diagonal`, also with the stretch); ICTF (automatic for Ag) |
| Dense path | `op::DenseStrategy`, LU in place (LAPACK zgetrf), residual on 64 sampled rows |
| Matvec | max over three random vectors (seeds 101, 202, 303) of \|Z_fast x − Z_dense x\| / \|Z_dense x\| |
| Far field | windowed currents (`Simulation::far_field_options()`, taper_end → pml_start), reflection hemisphere `post::hemisphere_grid(−1, n_θ, n_φ)` (U 90 × 180, CI 60 × 120), metric `pml_rough_ag::compare` (l2 = docs/05 box-check L2, solid-angle weights; dR = R_fast / R_dense − 1) |

## Accuracy (fast vs dense)

| Case | 2N | d₀ | levels / leaf | matvec | dR | l2 (l2 θ ≥ 30°) | GMRES its |
|---|---|---|---|---|---|---|---|
| U, rough Ag (7c size) | 22 816 | 3 | 5 / 0.97 λ | 1.51·10⁻⁵ | +4.1·10⁻⁸ | 3.3·10⁻⁶ (3.9·10⁻⁶) | 602 |
| U, rough Ag (7c size), review round | 22 816 | 5 | 4 / 1.94 λ | 1.40·10⁻⁷ | +5.5·10⁻⁸ | 1.2·10⁻⁷ (1.8·10⁻⁷) | 602 |
| flat CI | 4 592 | 3 | 4 / 0.875 λ | 3.16·10⁻⁴ | +7.4·10⁻⁸ | 1.9·10⁻⁶ (2.8·10⁻⁵) | 207 |
| flat CI | 4 592 | 5 | 3 / 1.75 λ | 4.9·10⁻⁸ | +1.3·10⁻⁹ | 8.2·10⁻⁸ (2.7·10⁻⁶) | 202 |
| flat matvec case | 2 816 | 3 | 4 / 0.69 λ | 3.92·10⁻⁴ | — | — | — |
| flat matvec case | 2 816 | 5 | 3 / 1.38 λ | 1.21·10⁻⁸ | — | — | — |

The U error at d₀ = 3 is small because the rough mesh (r_max = 154 nm) pushes the leaf rule to λ-sized leaves; the
flat meshes run on λ/2–λ/1.4 leaves and show the usual d₀ = 3 level. The d₀ = 5 entries pass by a wide margin
because the leaf rule (λ/2, r/a ≤ 0.3) leaves only two or three far levels on these small surfaces. At the 7c size
(review round, `specklebem_pml_ring --case u --d0 5`) the rough mesh's r_max = 154 nm sets 1.94 λ leaves (4
levels, far expansion on levels 2 and 3 only): matvec 1.40·10⁻⁷, two orders below the 10⁻⁵ gate, with the same
602 GMRES iterations; the dR of 5.5·10⁻⁸ is the GMRES tolerance (10⁻⁷), not the operator. Cost: fast operator
6.25 GB (rr near 0.77, rr far 1.34, ring near 0.71, ring far 3.43 GB) = 0.75× dense, peak 6.54 GB;
`Simulation::assemble()` 1 345 s (ring near 730 s — 7.4·10⁶ pairs, 3.6× the d₀ = 3 ones — far 534 s) against
642 s dense assembly; matvec 0.270 s against 0.283 s. Gated by the second case of
`tests/validation_large/test_pml_ring_large.cpp` (matvec only, no solve).

Review-round rerun of both validation-large 7h cases with the final code (W2 bookkeeping, chunked ring apply;
`ctest --preset win-release -L validation-large -R 7h`, passed, 1 865 s + 1 625 s): d₀ = 3 bitwise the record's
accuracy (matvec 1.508·10⁻⁵, dR +4.07·10⁻⁸, l2 3.33·10⁻⁶, 602 its; matvec 0.203 s against 0.264 s dense),
d₀ = 5 matvec 1.40·10⁻⁷ (operator build 956 s, ring near 440 s — the 730 s of the benchmark run above was
measured on a loaded machine).

## Memory and time (U, 2N = 22 816)

| Part | Fast | Dense |
|---|---|---|
| Z_rr near (real–real near pairs, 2.2·10⁶) | 212 MB | — |
| Z_rr far (MLFMM, stretched functions excluded) | 545 MB | — |
| ring near (2.0·10⁶ pairs, 96 bytes) | 195 MB | — |
| ring far (5.7·10⁷ stored pairs, 64 bytes; 8.9·10⁷ represented) | 3 652 MB | — |
| operator total / process peak | 4 605 MB / 4.88 GB | 8 329 MB / 8.46 GB |
| assembly | operator 905 s (ring near 380 s, far 522 s, MLFMM 3 s) + Jacobi diagonal and rhs 50 s = 955 s `Simulation::assemble()` (the ROW line's "fast assembly") | 640 s |
| solve | GMRES 602 its, 125 s | LU 56 s |
| one matvec | 0.198 s | 0.259 s |

At the flat CI size the fast operator is 0.91× (d₀ = 3) / 1.62× (d₀ = 5) of the dense matrix: below 2N ≈ 10⁴ the
fast path is no memory win (the near parts of λ-sized leaves are a large share of N²).

## Ring apply (review W4)

The first apply summed the transposed products in per-thread buffers (result depends on the thread count). Three
thread-count-independent variants were timed in one process each (same operator, temporary benchmark switch, 24
OpenMP threads, mean of 100 (flat CI) / 20 (U) applies after one warm-up, two rounds; all three agree to 1e-16):

| Case | ring far | per-thread buffers (70ae8a8) | two-pass (direct over row leaves, transposed over column leaves) | **chunked one pass (kept)** | Z_rr | full Z (chunked) | dense |
|---|---|---|---|---|---|---|---|
| flat CI, 2N = 4 592, d₀ = 3 | 150 MB | 5.9 / 6.4 ms | 7.2 / 8.3 ms | 6.1 / 6.2 ms | 10.2 ms | 18–20 ms | 12.2 ms |
| U, 2N = 22 816, d₀ = 3 | 3 652 MB | 127 / 131 ms | 178 / 230 ms | 130 ms ¹ | 68 ms | 202 ms | 399 ms ² |

¹ Second round; the first (170 ms ring, 444 ms Z) overlapped a compile. ² The dense matvec of this process
(the 7h ROW lines give 0.26–0.30 s). The two-pass variant reads every stored block twice and costs 1.2–1.75× the
ring apply; the chunked one pass (at most 64 fixed chunks of consecutive row leaves balanced by stored entries,
one transposed buffer of 4N reals per chunk, summed in chunk order) matches the per-thread buffers. At the flat
CI size the whole fast operator is slower than the dense matvec with every variant (the near parts dominate,
see above); at U it is 0.5× the dense one.

## Memory model of 5a and the projection for 5b

Fast memory ≈ 64 N_s N_far (far ring) + 96 × ring near pairs + 96 × real–real near pairs + MLFMM far tables, with
N_far = N minus the 27-leaf neighbourhood. Pre-assembly estimates (`--estimate-only`, flat surface with the zones
of U, d₀ = 3, h = λ/4, W_pml = 2λ; rows file):

| L | 2N | N_s / N | ring near | ring far (5a, exact) | real–real near | dense |
|---|---|---|---|---|---|---|
| 3.75 µm (U, rough) | 2.3·10⁴ | 45 % | 0.20 GB | 3.65 GB | 0.21 GB | 8.3 GB |
| 7.5 µm | 5.0·10⁴ | 32 % | 0.17 GB | 12.9 GB | 0.35 GB | 40.7 GB |
| 15 µm | 1.4·10⁵ | 20 % | 0.78 GB | 60.5 GB | 3.1 GB | 305 GB |
| 30 µm | 4.4·10⁵ | 11 % | 1.17 GB | 358 GB | 9.0 GB | 3 137 GB |

5a fits 50 GB up to 2N ≈ 1.2·10⁵ (L ≈ 13 µm); the 6·10⁴ point of the ADR scaling gate is within reach, the
≥ 1.5·10⁵ point is not. What 5b needs: the far ring blocks (A_s × far columns per box pair) compressed by ACA to
a ratio of ≥ 10 at L = 15 µm and ≥ 40 at L = 30 µm (then everything else — near parts and MLFMM — dominates,
≈ 10–15 GB at L = 30 µm); the ranks are to be logged per box pair (ADR 0008 amendment decision 3b). The assembly
time of the exact far ring (O(N_s N) stretched pairs) is the second limit: 522 s at U, ≈ 30× that at L = 15 µm,
so 5b's ACA (O(rank × (m + n)) entries per block) is also the time fix for the far ring, and 5c the one for the
near ring.

## Design notes

- **Octree variant.** The octree is built over all functions as without a stretch (real coordinates, the same
  leaf rule); the far operator excludes the stretched functions (zero sources and rows, no exact-part pairs with
  them) and runs on the Problem without the stretch; the near part holds the real–real pairs. So Z_rr x equals the
  unstretched MLFMM of the same mesh applied to the masked x, restricted to the real rows (unit test, < 1e-13), and
  the ring uses the same leaves. The leaf patterns of the stretched functions are still stored (part of the 545 MB).
- **Far ring storage.** Per leaf A with stretched functions: one block of the four sub-blocks over A_s × (all
  functions of the leaves not near A); far pairs share no triangle, so each sub-block is complex symmetric there
  and the real rows × stretched columns are applied transposed from the same blocks. One pass per apply over
  at most 64 fixed chunks of consecutive row leaves (balanced by stored entries): direct products into the
  leaves' own rows, transposed products into a buffer per chunk, summed in chunk order (review W4); every entry
  of y has a fixed summation order, so the result is bitwise independent of the thread count. The first version
  used per-thread buffers summed in thread order (deterministic only for a fixed thread count); timing of the
  variants under "Ring apply (review W4)".
- **Budget.** `PmlRingParams::max_far_bytes` (via `MlfmmParams::pml_ring`), automatic = the dense matrix; checked
  before any allocation (unit test: 1 byte → `std::runtime_error` naming `max_far_bytes`).

## 5b: ACA far ring (WP-PML5b)

ADR 0008 amendment 2026-10-18 decision 3b. Library code `include/specklebem/compression/aca/aca.hpp`
(`aca::compress`, kernel-independent partially pivoted ACA) and the far part of `PmlRingOperator`
(`src/compression/mlfmm/pml_ring_far.{hpp,cpp}`: partition, `BlockSource`, `AcaFarRing`); tests
`tests/unit/test_aca.cpp`, the ACA cases of `tests/unit/test_pml_ring.cpp`, the CI cases of
`tests/validation/test_pml_ring.cpp` (now through ACA) and the 2N ≈ 6·10⁴ case of
`tests/validation_large/test_pml_ring_large.cpp`; driver modes `--ring-compare [--no-exact]`, `--eta`,
`--exact-far`, `--compare-far` of `benchmarks/pml_ring.cpp`, `--pml-solver fast` of
`benchmarks/pml_si_crosscheck.cpp`. Measured 2026-10-19/20 on the same 12-core desktop with 128 GB (builds and
tests of other worktrees shared the machine during most runs: the times are ±30 % or worse; memory and accuracy are not
affected). Defaults: ACA tolerance 10^-(d₀+1), η = 2, H-matrix row clusters, break-even rank cap.

### Summary

- **7h kept through the ACA path** at the 7c size (rough Ag U, 2N = 22 816): d₀ = 3 matvec 2.08·10⁻⁵, dR
  +4.8·10⁻⁸, L2 3.3·10⁻⁶, 602 GMRES iterations (5a: 1.51·10⁻⁵, 4.1·10⁻⁸, 3.3·10⁻⁶, 602); d₀ = 5 matvec
  2.15·10⁻⁷ (5a 1.40·10⁻⁷); flat CI 2N = 4 592 matvec 3.16·10⁻⁴, dR 7.0·10⁻⁸, L2 1.95·10⁻⁶; 2N = 2 816
  3.9·10⁻⁴ / 4.0·10⁻⁸ (d₀ = 3 / 5). Criteria (10⁻³ / 10⁻⁵, 10⁻⁴, 10⁻³) not loosened.
- **Memory:** far ring 571 MB instead of 3 652 MB at U (6.4×), the whole fast operator 1.52 GB (0.18× dense,
  5a 0.55×), peak 1.83 GB; at 2N = 59 600 (L = 8.5 µm) 1.64 GB instead of 16.7 GB (10.2×), operator
  3.68 GB against 57 GB dense; at 2N = 152 960 (L = 16 µm, ACA only) far ring 3.9 GB
  (exact would be 70.8 GB, 18×), operator 13.3 GB, process peak 18.0 GB with the full-GMRES basis of 1 715
  iterations; the largest run, 2N = 300 160 (L = 24 µm), fits 50 GB with margin: far ring 8.3 GB
  (exact 198 GB, 24×), operator 20.1 GB, peak 33.6 GB (2 679 iterations of full GMRES: basis ≈ 12.9 GB); the
  projected 50 GB limit is 2N ≈ 4·10⁵.
- **ACA vs the exact far ring** (same Problem, same octree): ring 3.1·10⁻⁵ (U, η = 1) / 5.1·10⁻⁵ (2N = 59 600)
  relative to the ring product (criterion 10⁻⁴), 0.8–1.1·10⁻⁵ relative to the whole operator.
- **Time:** far ring 151 s at U (5a 522 s), 387 s at 2N = 59 600 (exact 3 568 s, 9.2×); the ring apply
  0.078 s against 0.626 s exact there. The near ring (614–866 s, measured before the WP-PML5c cache) then
  dominated the setup (WP-PML5c, developed in parallel, addressed it; section "5b + 5c integrated").
- **7d Si control and 7e through the fast path:** the rough-Si L ×1.2 change 0.2412 % through the fast path against
  0.2411 % dense (WP-PML4b: 0.241 %; criterion < 0.5 %), each Si case 3.6·10⁻⁶ / 9.0·10⁻⁶ (L2) from its dense
  solve; 7e (rough Si, PML route vs the uniform box) l2_refl 0.1464 %, ε_rr 0.0256 / 0.0412 %, dR −1.63·10⁻⁵ through
  the fast path against 0.1464 %, 0.0257 / 0.0413 %, −1.62·10⁻⁵ dense (the stored WP-PML4c values: the dense partner of this session was stopped, see below), R 0.3856893 on both.

### Design choices (deviations from the brief noted)

- **Row clusters.** The brief's base design (rows = one leaf's stretched functions, columns grouped into the
  coarsest admissible boxes) compresses the far ring of U only 2.6× (1 399 MB, η = 1): the ranks at 10⁻⁴ (median
  13, λ-sized leaves) are a large fraction of the ~33 stretched rows per leaf, and the ratio k/m bounds the gain
  whatever the column clusters. The default is therefore the H-matrix partition (dual descent over same-level box
  pairs, row clusters of every level): 819 MB at η = 1, 571 MB at η = 2. `row_clusters = false` keeps the leaf-row
  design selectable. Each block (R, C) holds R_s × C (stretched rows of R, all functions of C), so the transposed
  use of 5a is unchanged; the direct products go through the chunk buffers too, since a stretched row belongs to
  row clusters of several levels (apply still thread-count independent, unit test 1/2/3/max threads, also for the
  assembly).
- **η.** U, d₀ = 3, row clusters (driver `--ring-compare --no-exact --eta`):

  | η | far ring | ACA blocks / exact / rank-cap | ranks median / max | far assembly |
  |---|---|---|---|---|
  | 0.5 | 1 260 MB | 10 072 / 6 005 / 3 467 | 10 / 19 | 367 s ¹ |
  | 1 | 819 MB | 4 576 / 2 556 / 1 668 | 13 / 28 | 288 s |
  | 2 (default) | 571 MB | 1 853 / 1 092 / 1 035 | 14 / 37 | 240 s |
  | 3 | 524 MB | 2 250 / 40 / 1 450 | 15 / 39 | 261 s |

  ¹ before the per-triangle-pair drop (below). η = 2 takes 70 % of η = 1 and is within 9 % of η = 3 with fewer
  poorly separated blocks; at 2N = 59 600 η = 3 gives 1 566 MB against 1 636 MB (−4 %, max rank 41 both). The admissibility only sets the ranks (the ACA tolerance controls the error).
- **Four sub-blocks.** Each of JJ, JM, MJ, MM is compressed separately (their scales differ by η₀² ≈ 1.4·10⁵); one
  cache of generated function rows/columns serves all four (one triangle-pair evaluation gives all four), and each
  ACA starts from the previous pivots, so the generated rows are ≈ the JJ rank (U: 39 k rows for 1 853 blocks). On
  flat meshes K vanishes between far pairs: JM/MJ are found zero from the cached rows and one probe column (rank 0).
- **Stopping rule.** ‖u_k‖‖v_k‖ ≤ ε‖S_k‖_F must hold in two consecutive steps (`aca::CompressParams::
  confirmations = 2`): the plain rule left 1.04 ε on a Helmholtz test block (kR = 6, η ≈ 0.9). Zero pivots: next
  unused row (cached rows first); after 3 consecutive zero rows the iteration ends (k > 0) or probes columns (k = 0,
  zero block). Rank cap: the break-even rank floor(mn/(m+n)), so no block exceeds its exact bytes; a sub-block
  that reaches it makes the block exact (U: 1 035 such blocks, 130 MB with the 1 092 non-admissible ones).
- **Magnitude bound with the stretch.** The §6 bound exp(−α d)/d does not hold as such for the stretched kernel:
  the complex distance can be shorter than the real one, Re R̃ ≥ κR with κ = 2/√(4 + σ_max²) = 0.447 at
  σ_max = 4 (secant slopes of the stretch in [0, σ_max]; the minimum over random pairs is checked in a unit test).
  The ring uses δ = F (1 + αR*) e^{−ακR*}, F = (1 + σ_max²)²/κ³ (Jacobian and Piola factors), applied per block
  (region parts and whole blocks) and, inside a kept block, per triangle pair (vertex extents; R2 of Ag beyond
  1.2 µm at d₀ = 3). At U the R2 part is dropped in 1 016 of 3 980 blocks and in 4.9 M of 24 M triangle pairs;
  vacuum (R1) is never dropped, so no block of the PML route is dropped as a whole. The per-pair drop cut the far
  assembly from 387 to 240 s (η = 2). For Si (weak loss) nothing is dropped.
- **Budget.** `max_far_bytes` bounds the blocks known to be exact before assembly (non-admissible leaf pairs,
  `PmlRingEstimate::far_exact_bytes`); the ACA blocks are bounded by the break-even cap (worst case = the exact
  estimate `far_bytes`, which stays the upper bound reported before assembly).

### Accuracy and cost per size (d₀ = 3 unless stated, η = 2)

| Case | 2N | ring far (5a exact) | fast op / peak | dense | matvec vs dense | dR / L2 | its | far ring / near ring assembly |
|---|---|---|---|---|---|---|---|---|
| flat CI | 4 592 | 49.6 MB (150 MB) | 207 MB / 0.26 GB | 337 MB | 3.16·10⁻⁴ | 7.0·10⁻⁸ / 1.95·10⁻⁶ | 207 | 3.6 s / 98 s |
| flat 2N = 2 816, d₀ = 3 / 5 | 2 816 | 29 / 21 MB | 105 / 210 MB | 127 MB | 3.9·10⁻⁴ / 4.0·10⁻⁸ | — | — | 3.7 s / 32–41 s |
| U rough Ag | 22 816 | 571 MB (3 652 MB) | 1 525 MB / 1.83 GB | 8 329 MB | 2.08·10⁻⁵ | 4.8·10⁻⁸ / 3.3·10⁻⁶ | 602 | 151 s / 614 s |
| U rough Ag, d₀ = 5 | 22 816 | 659 MB (3 430 MB) | 3 485 MB / 3.58 GB | 8 329 MB | 2.15·10⁻⁷ | — | — | 234 s / 890 s |
| U rough Ag, 5b + 5c integrated ⁴ | 22 816 | 571 MB (3 652 MB) | 1 525 MB / 1.83 GB | 8 329 MB | 2.08·10⁻⁵ | 4.8·10⁻⁸ / 3.3·10⁻⁶ | 602 | 119 s / 89 s |
| U rough Ag, d₀ = 5, 5b + 5c integrated ⁴ | 22 816 | 659 MB (3 430 MB) | 3 485 MB / 3.91 GB | 8 329 MB | 2.15·10⁻⁷ | — | — | 164 s / 123 s |
| U rough Si (7d control) | 22 816 | 823 MB | 7 206 MB ² / 7.53 GB | 8 329 MB | 5.2·10⁻⁵ | 3.8·10⁻⁷ / 3.6·10⁻⁶ | 121 | 1 480 s ³ / 827 s |
| L ×1.2 rough Si (7d control) | 27 472 | 993 MB | 9 471 MB ² / 9.86 GB | 12 075 MB | 4.2·10⁻⁵ | 2.7·10⁻⁷ / 9.0·10⁻⁶ | 128 | 1 807 s ³ / 831 s |
| 7e rough Si (L 4.5 µm, w₀ L/6) | 27 472 | — | 9 460 MB / 9.84 GB | 12 075 MB (peak 24.3 GB with LU) | — | R as dense to 7 digits | 127 | assembly 1 998 s (dense 4 362 s, WP-PML4c) |
| L = 8.5 µm rough Ag | 59 600 | 1 636 MB (16 660 MB) | 3 683 MB / 3.86 GB | 56 835 MB | ACA vs exact ring 5.1·10⁻⁵ (ring), 1.1·10⁻⁵ (operator) | — | — | 387 s (exact 3 568 s) / 866 s |
| L = 8.5 µm rough Ag, 5b + 5c integrated ⁴ | 59 600 | 1 636 MB (16 660 MB) | 3 683 MB / 3.86 GB | 56 835 MB | ACA vs exact ring 5.1·10⁻⁵ (ring), 1.1·10⁻⁵ (operator) | — | — | 333 s (exact 2 808 s) / 105 s |
| L = 16 µm rough Ag | 152 960 | 3 897 MB (70 779 MB) | 13 334 MB / 17.95 GB | 374 GB | — (dense impossible) | d₀ 3 vs 5: dR 1.5·10⁻⁸ / L2 1.5·10⁻⁶ | 1 715 | 661 s / 1 166 s |
| L = 16 µm, d₀ = 5 | 152 960 | 6 205 MB | 16 982 MB / 21.6 GB | 374 GB | — | (reference of the row above) | 1 715 | 1 131 s / 1 270 s |
| L = 24 µm rough Ag | 300 160 | 8 332 MB (198 496 MB) | 20 069 MB / 33.6 GB | 1 442 GB | — | — | 2 679 | 1 478 s / 1 667 s |
⁴ Section "5b + 5c integrated" (the near ring of 5c; the near ring times of the other rows are the 5a/5b near ring).
² 5.97 GB of it is the real–real MLFMM far part of Si (unchanged by this WP). ³ Si's interior kernel is not
damped (no drops) and its ranks are higher (median 16, max 69, mean MM 25): 33 M triangle-pair evaluations.

Ranks (histograms over all ACA sub-blocks, bins [0], [1], [2–3], [4–7], [8–15], [16–31], [32–63], [64–127]): U
[1 884 0 0 524 2 292 2 614 98]; U d₀ = 5 [304 0 0 80 8 744 1 104] (median 31, max 60); 2N = 59 600 [3 214 0 1 782
1 014 8 890 5 560 252] (median 12, max 41); Si U [1 268 0 0 20 1 339 1 640 997 4]. The zero ranks are the JM/MJ
blocks of flat stretched pairs (K = 0) and of blocks with exactly vanishing rows.

### Scaling

The 2N ≈ 6·10⁴ point is the validation-large case "ACA far ring vs the exact far ring at 2N ~ 6e4" (passed,
4 834 s; exact far ring 16.7 GB built in the same process with `assemble_near = false`, its near part taken from the
ACA operator): ring difference 5.1·10⁻⁵ relative to the ring product, 1.1·10⁻⁵ relative to the operator product
(criterion 10⁻⁴, the unit criterion; the 7h matvec budget is 10⁻³). The largest sizes run without a dense or exact
reference (`--fast-only`): L = 16 µm (2N = 152 960, 19 % stretched, 6 levels, leaf 1.25 λ) at d₀ = 3 and d₀ = 5,
GMRES (ICTF + Jacobi, tolerance 10⁻⁷, full) 1 715 iterations both; R = 0.97780527 / 0.97780529 (difference
1.5·10⁻⁸ relative), windowed far field d₀ 3 vs 5 L2 1.5·10⁻⁶ — the d₀ = 5 run is the reference where dense is
impossible. The largest run, L = 24 µm (2N = 300 160, 13.8 % stretched, 7 levels, leaf 0.875 λ), d₀ = 3 only:
R = 0.97948552, 2 679 iterations (true residual 1.0·10⁻⁷), `Simulation::assemble()` 3 745 s (ring near 1 667 s,
ring far 1 478 s), GMRES 6 540 s (1.9 s per matvec plus the full-GMRES orthogonalisation), peak 33.6 GB of which
≈ 12.9 GB is the Krylov basis. With the operator growing ≈ linearly (13.3 → 20.1 GB for 2N ×2) and the basis
growing with N × iterations, 50 GB is reached near 2N ≈ 4·10⁵ (restarted GMRES would push it further). A d₀ = 5
partner was not run at this size (the L = 16 µm pair shows d₀ 3 vs 5 at 1.5·10⁻⁸ in R).

**Finding: GMRES iterations grow with the size (open issue, outside 5b).** ICTF + Jacobi on the rough Ag series,
full GMRES to 10⁻⁷: 602 iterations at 2N = 22 816, 1 715 at 152 960 and 2 679 at 300 160, i.e. ≈ N^0.55–0.58
(the exponent from 2.3·10⁴ to 1.5·10⁵ / 3·10⁵; ≈ N^0.5 within the scatter of three points). At 2N = 300 160 the
Krylov basis (2 679 vectors of 2N complex) is ≈ 12.9 GB of the 33.6 GB peak, and GMRES took 6 540 s against
≈ 5 000 s of matrix-vector products (2 679 × 1.9 s): the orthogonalisation, quadratic in the iteration count, is
already ≈ 25 % of the solve, and the basis, not the operator, will set the memory limit beyond 2N ≈ 4·10⁵. Remedies
(not part of WP-PML5b): restarted GMRES (bounded basis, more iterations) or a better preconditioner than Jacobi,
e.g. one built from the near-field parts the fast operator already stores (real–real near + near ring).

Memory per size (fast operator; the exact 5a far ring in parentheses):

| 2N | ring far ACA | ring near | Z_rr near / far | operator | peak | ring far / exact |
|---|---|---|---|---|---|---|
| 22 816 | 0.57 GB | 0.20 GB | 0.21 / 0.55 GB | 1.52 GB | 1.83 GB | 0.16 |
| 59 600 | 1.64 GB | 0.22 GB | 0.51 / 1.32 GB | 3.68 GB | 3.86 GB | 0.098 |
| 152 960 | 3.90 GB | 0.95 GB | 3.81 / 4.67 GB | 13.3 GB | 18.0 GB | 0.055 |
| 300 160 | 8.33 GB | 0.67 GB | 4.01 / 7.05 GB | 20.1 GB | 33.6 GB | 0.042 |
The ACA far ring grows about like N (0.57 → 1.64 → 3.90 → 8.33 GB for 2N ×2.6 ×2.6 ×2.0): the median rank stays 12–18 at d₀ = 3
(27 at d₀ = 5), the max rank 37–56 (78), so the fallbacks of the amendment (one-level FMM with complex translators,
Huygens separation) are not needed. At 2N ≥ 1.5·10⁵ the Z_rr near part (leaf 1.25 λ there) and the Krylov basis of
full GMRES are as large as the ring.

### 7d (rough Si control) and 7e through the fast path

Both run once through the drivers (record, not ctest: each dense partner takes 30–75 min on the shared machine).
**7d Si control** (`specklebem_pml_ring --case u --material si --tag si_u` and `--patch 4.5e-6 --tag si_l12`,
fast + dense in one process each, then `--compare-far`): fast vs dense per case L2 3.6·10⁻⁶ (U) / 9.0·10⁻⁶
(L ×1.2), dR −3.8·10⁻⁷ / −2.7·10⁻⁷, matvec 5.2·10⁻⁵ / 4.2·10⁻⁵; the 7d metric (L ×1.2 against U, box-check L2)
0.24122 % fast against 0.24107 % dense (difference 1.5·10⁻⁶, criterion "within 10⁻³"; WP-PML4b recorded 0.241 %).
**7e** (`specklebem_pml_si_crosscheck --route box` and `--route pml --pml-solver fast --pml-tol 1e-7`): the box run
reproduces WP-PML4c identically to 7 digits in R (0.3856955) and in the iteration count (92); the fast PML route gives l2_refl 0.1464 %, ε_rr
0.0256 / 0.0412 %, dR −1.625·10⁻⁵ (WP-PML4c dense LU: 0.1464 %, 0.0257 / 0.0413 %, −1.617·10⁻⁵; tolerances 0.229 %,
0.058 %, 4.1·10⁻⁵), R 0.3856893 on both routes, 127 GMRES iterations; the dense partner of this session was stopped
after 10 h of assembly on the overloaded machine, so the dense values are the stored WP-PML4c ones (same driver, same
case, `pml_validation_rows.txt`). Cost: fast operator 9.46 GB (dense LU peak 24.3 GB), assembly 1 998 s against
4 362 s dense in WP-PML4c.

## 5c: ring assembly (WP-PML5c)

ADR 0008 amendment decision 4. Measured 2026-10-19 on the same 12-core desktop with 128 GB (win-release, 24
OpenMP threads), but **not idle**: another work package ran its own studies on the machine throughout, so absolute
times are 1.3–2× those of the 5a sections above (the 5a near ring of U took 640–800 s here against 380 s in the 5a
record). Only times measured in one process, one after the other, are compared. Driver:
`specklebem_pml_ring --case u|flat_ci --assembly-study [--quick] [--dense-compare]` (ASSEMBLY, PROFILE, CLASSES
lines in `pml_ring_rows.txt`).

**What changed.**

1. *Canonical-pair cache* (`op::assemble_sparse_stretched`, `StretchedPairEvaluation::canonical_cache`): the
   stretched element blocks are computed for the ordering (min, max) and transposed for the other, so the combined
   blocks satisfy B(s, t) = B(t, s)ᵀ bitwise. Every needed unordered stretched triangle pair is evaluated once into a
   cache slot (576 bytes) in a parallel pass over the pairs (touching pairs first, `schedule(dynamic, 4)`), and the
   unchanged scatter (test triangles by colour, sources ascending) takes (t, s) from the cache, transposed for t > s:
   the same contributions in the same order, so the matrix is **bitwise** the 5a one. Each pair is written by one
   thread into its own slot, so the result does not depend on the thread count (unit test 1 / 3 / max threads).
2. *Translation classes* (`translation_classes`, the default): the touching pairs dominate (table below), and those
   in a PML strip repeat along it — where all six vertices of a pair satisfy |y| ≤ y.start the stretch depends on x
   only, so the blocks are invariant under a translation along y (x alike; corners are not). The touching pairs are
   grouped by a key that holds, **bit for bit**, every input the stretched touching rule reads from the pair. Since
   the review round the key is built in the kernels layer next to the rule (`kernels::stretched_touching_key`, from
   the same pair setup, slots, point counts and rule origin; back-references in `tri_of`, `slots_of`,
   `touching_rule` and `stretched_blocks`): coordinates relative to the rule origin along the translated axis, the
   absolute ones across it, per triangle normal, area and longest edge, the RWG divergence per free vertex, the
   3 × 3 shared-vertex-index pattern, per region the rule's point counts and the **fastmath decision** (decided from
   absolute centroids, not translation-exact: a threshold flip changes a coplanar block by up to 7·10⁻¹⁴), and pairs
   with a vertex within 10⁻¹² max(start, h) of their strip boundary get no key (a profile of order m < 1 gives a
   visible σ one ulp past the start). One evaluation per class, the members are exact copies with the slots mapped
   by free vertex. Pairs whose inputs differ in the last bit (the mesh grid
   coordinates do not translate exactly) form classes of their own, so the copies are bitwise their own evaluation:
   the matrix stays **bitwise** the 5a one (measured on both cases below, unit test on the tiny meshes). A
   rounding-tolerant key (16 ulp; tried first, CLASSES lines) would copy ≈ 2× more pairs but changes entries by
   1.8·10⁻¹⁵ of the largest entry and up to 2.3·10⁻¹³ relative on single entries with cancellation (5a itself
   differs from the dense matrix by 8.3·10⁻¹⁴ per entry there): rejected in favour of bitwise results.
3. *Jacobi diagonal from the near parts* (`MlfmmOperator::diagonal()`, used by `Simulation`): every pair (m, m) is
   near, so Z(r, r) is stored in Z_rr's near field (real rows; every row without a stretch) or the near ring
   (stretched rows). Real rows bitwise `op::assemble_diagonal`, stretched rows 3.4·10⁻¹⁶ relative (the dense PML
   order 2A + rest); unit test 1e-14 and bitwise without a stretch. The unstretched MLFMM path uses the same code
   (bitwise the old diagonal, unit test).
4. *Shared pattern computations* (`mlfmm::plan_pml_ring`, `PmlRingPlan`): classification, split near pattern and
   estimate once per operator; `MlfmmOperator` uses the plan for its budget check and the real–real near part and
   hands it to the new `PmlRingOperator(problem, tree, plan, params)` constructor (the old constructor plans
   itself; plan consistency checked: classification, size and, since the review round, an octree fingerprint —
   levels, leaves, leaf size and a hash of the permutation and leaf structure — so that a plan of another tree is
   refused). The plan also estimates the transient near-ring assembly cache (`PmlRingEstimate::near_assembly_bytes`,
   an index pass over the ring pattern, 0.26 s at U; 339 MB bound against 329 MB measured), which the memory
   warning now counts: peak = near ring + max(cache, far ring).

**Near ring of U (2N = 22 816, d₀ = 3), one process** (5a = `per_ordering`, the WP-PML5a path):

| Run | 5a | 5c canonical only | **5c (default)** | dense assembler, same pairs | 5c / dense | 5c / 5a |
|---|---|---|---|---|---|---|
| final code, full study | 801.9 s | 390.1 s | **168.6 s** | 407.7 s | **0.41** | 0.21 |
| final code, `--quick` | 641.9 s | — | **93.7 s** | — | — | 0.15 |
| review round (complete key), full study | 801.6 s | 307.7 s | **114.7 s** | 438.1 s | **0.26** | 0.14 |
| review round, `--quick` | 850.8 s | — | **200.6 s** | — | — | 0.24 |
| validation-large 7h run (below) | — | — | **125.0 s** | (whole dense assembly 1 242 s) | — | — |

Evaluations: 5a 1 062 440 stretched triangle-pair blocks; canonical 532 948 (identical 3 456, touching 23 936);
5c 519 636 (13 312 of the 20 480 non-identical touching pairs copied from 4 096 classes; 3 072 corner pairs
evaluated; review round: 12 544 copies, 520 404 evaluations, see below). The dense reference evaluates exactly these pairs with the DenseStrategy rules (each unordered stretched
pair once by the balanced parity rule, every unstretched ordering, colour by colour with schedule(dynamic, 1);
blocks evaluated, not scattered): 592 282 triangle pairs. **Gate (near ring ≤ ½ of the dense assembly of the same
pairs at the 7c size): met, 0.41** in the full study (0.15–0.21 of 5a in the other runs). The canonical cache alone
reaches the dense assembler (0.96: it evaluates the same pairs once each); the factor beyond it comes from the
translation classes. Cache memory 311 MB during the near-ring assembly (1.6× the 195 MB near ring it fills).

Flat CI (2N = 4 592), one process: 5a 94.2 s, canonical 48.0 s, **5c 29.9 s**, dense same pairs 51.3 s (0.58; the
small strips copy only 2 112 of 4 480 touching pairs, 768 corner pairs). Bitwise against 5a: yes (both evaluations).

**Review round (key completeness, 2026-10-19).** The review found inputs of the touching rule outside the 5c key
(the fastmath decision, the shared-index pattern, rule points one ulp past a strip boundary with m < 1) and the key
mirrored kernel internals in the operator layer; the key now comes from `kernels::stretched_touching_key` (point 2
above). Effect on the classes, both cases bitwise against 5a again (all runs below, loaded machine: another work
package's `specklebem_pml_ring` ran throughout):

| Case | touching (non-identical) | copies before → after | classes before → after | evaluated before → after | cache measured / estimate |
|---|---|---|---|---|---|
| U | 23 936 (20 480) | 13 312 → **12 544** | 4 096 → 4 096 | 519 636 → 520 404 (+768) | 329.4 / 339.0 MB |
| flat CI | 5 248 (4 480) | 2 112 → **2 048** | 1 600 → 1 280 | 91 264 → 91 328 (+64) | 58.0 / 59.9 MB |

The dropped copies are the strip-end pairs with a vertex on |u| = start (now keyless; the fastmath flag and the
shared-index pattern are functions of coplanarity and of the grid's local orientation on these meshes and split
no class). They add 7 % to the evaluated touching pairs at U (0.15 % of all evaluations). Timings, one process each:
U full study 5a 801.6 s, canonical 307.7 s, **5c 114.7 s, dense same pairs 438.1 s: 5c / dense 0.26** (gate ½
met; 0.41 in the 5c full study above, the load differs between the runs), 5c / 5a 0.14; U `--quick` 5a 850.8 s,
5c 200.6 s (0.24, the 5c pass overlapped heavier load); flat CI `--quick` 5a 57.9 s, 5c 16.0 s (0.28). The cache
figure now counts every transient array (blocks 576 B, index, order, representatives, slot maps, keys; 311.3 MB
before counted blocks and index only), and `plan_pml_ring` bounds it before the assembly (0.26–0.28 s at U for the
index pass).

**Where the near-ring time goes** (U, thread-seconds of the canonical pairs, timed per pair, R2 rule sizes):

| Class | pairs | share | point pairs per call | ms per pair |
|---|---|---|---|---|
| shared vertex | 15 328 | 85.2 % | 2.77·10⁶ | 534 |
| shared edge | 5 152 | 10.9 % | 9.9·10⁵ | 203 |
| identical | 3 456 | 0.3 % | 3.8·10⁴ | 8.7 |
| near / far, target reached | 150 100 | 0.6 % | 3 479 | 0.40 |
| near / far at the degree cap | 358 912 | 3.0 % | 5 329 | 0.79 |

(flat CI: shared vertex 80 %, shared edge 15 %.) After 5c the remainder is the corner touching pairs (3 072, not
translation invariant; the four corners are mirror images, but a reflection flips the sign of K and needs its own
mapping), the strip pairs whose coordinates do not translate exactly (≈ 4 000 classes instead of 384 with a tolerant
key), and the near/far pairs (3.6 %). Further levers, not done: mirror classes for the corners; coordinate snapping
of the generated grid (would make the copies exact everywhere); fewer graded panels for shared-vertex pairs
(WP-PML2c note; changes results).

**Operator setup and `Simulation::assemble()` (7h validation-large case, U, d₀ = 3)**: rerun once with the final
code (`ctest --preset win-release -L validation-large -R "7h: pml_ring fast path vs dense"`, passed, 2 732 s on the
loaded machine). Results identical to the 5a rows in every printed digit (matvec 1.508·10⁻⁵, R fast 0.97833919,
dR +4.069·10⁻⁸, l2 3.332·10⁻⁶, 602 iterations, true residual 8.41·10⁻⁸; the operator is bitwise the 5a one, the
Jacobi diagonal differs by ≤ 3.4·10⁻¹⁶ relative):

| Part | 5a (record, idle) | 5c (this run, loaded) |
|---|---|---|
| ring near | 380.4 s (0.59 × dense assembly 639.8 s) | **125.0 s (0.10 × dense assembly 1 242 s of the same run)** |
| ring far (the exact 5a ring in this run; ACA since WP-PML5b) | 522.3 s | 904.1 s |
| MLFMM (real–real near + far) | ≈ 3 s | ≈ 5 s |
| operator total | ≈ 905 s | 1 034.6 s |
| right-hand side | ≈ 1 s | 0.5 s |
| Jacobi diagonal | ≈ 50 s (`op::assemble_diagonal`; 72–179 s on this loaded machine) | **0.007 s** |
| `Simulation::assemble()` | 955.4 s (1.49 × dense) | 1 035.1 s (0.83 × dense of the same run) |

The far ring (O(N_s · N) stretched pairs, every pair of a stretched test triangle with every far source triangle) was
then ≈ 87 % of the setup; WP-PML5b (ACA, section "5b") compresses it (section "5b + 5c integrated"). The plan sharing saves only fractions of a second here (each
pattern computation took < 0.01 s at U).

## 5b + 5c integrated

WP-PML5b (ACA far ring) and WP-PML5c (near-ring assembly) were developed in parallel from 5a and merged on
branch `wp/pml5b-integrate` (2026-10-20). The combined `PmlRingOperator` assembles the near ring with 5c's
canonical-pair cache and translation classes (the only near path) and the far ring with 5b's ACA (default) or the
exact 5a blocks (`use_aca = false`); `PmlRingPlan` (5c) now also carries 5b's far partition (`far_blocks`, from the
same pass as the estimate) and the parameters it was made with, so the classification, split near pattern,
estimate and far partition are computed once per operator; both constructors take `accuracy_digits` (d₀). The
memory warning counts the peak known before assembly, near ring + max(near-ring cache, far blocks stored exactly)
(`pml_ring_peak_bytes`; without ACA the 5c form near + max(cache, far ring)); `PmlRingOperator::peak_bytes()` is the
measured peak with the ACA factors. The Jacobi diagonal is `MlfmmOperator::diagonal()` (5c).

Validation-large 7h cases at the 7c size, rerun once with the integrated code (`ctest --preset win-release -L
validation-large -R "7h: pml_ring fast path vs dense"`, passed, 1 024 s; `-R "7h: pml_ring matvec at d0 = 5"`,
passed, 958 s; run one after the other on a machine without other heavy jobs: the dense assembly took 678 s against
640 s in the idle 5a record). Every printed 5b accuracy digit is reproduced (the near ring is bitwise the 5a/5b one,
the far ring the same ACA ring); raw lines in `pml_ring_rows.txt` ("5b + 5c integrated").

**U, d₀ = 3 (2N = 22 816), operator setup and `Simulation::assemble()`:**

| Part | 5a (record, idle) | 5c rerun (exact far ring, loaded) | 5b (ACA, loaded) | **5b + 5c integrated** |
|---|---|---|---|---|
| ring near | 380.4 s | 125.0 s | 614.2 s | **89.1 s** |
| ring far | 522.3 s (exact) | 904.1 s (exact) | 151.3 s (ACA) | **119.0 s (ACA)** |
| MLFMM (real–real near + far) | ≈ 3 s | ≈ 5 s | ≈ 133 s with rhs and diagonal ¹ | **2.7 s (1.82 + 0.84)** |
| Jacobi diagonal | ≈ 50 s (`op::assemble_diagonal`) | 0.007 s (near parts) | (in ¹) | **0.002 s (near parts)** |
| right-hand side | ≈ 1 s | 0.5 s | (in ¹) | **0.17 s** |
| `Simulation::assemble()` | 955.4 s | 1 035.1 s | 898.8 s | **211.0 s** |
| dense assembly (same process) | 639.8 s | 1 242.0 s | 1 096.7 s | **678.0 s** |
| assemble() / dense | 1.49 | 0.83 | 0.82 | **0.31** |
| fast operator / process peak | 4 605 MB / 4.88 GB | 4 605 MB / 4.89 GB | 1 525 MB / 1.83 GB | **1 525 MB / 1.83 GB** |
| matvec fast / dense | 0.198 / 0.259 s | 0.450 / 0.281 s | 0.340 / 0.296 s | **0.101 / 0.266 s** |
| matvec vs dense | 1.508·10⁻⁵ | 1.508·10⁻⁵ | 2.081·10⁻⁵ | **2.081·10⁻⁵** |
| dR / l2 | +4.07·10⁻⁸ / 3.33·10⁻⁶ | +4.07·10⁻⁸ / 3.33·10⁻⁶ | +4.76·10⁻⁸ / 3.33·10⁻⁶ | **+4.76·10⁻⁸ / 3.33·10⁻⁶** |
| GMRES | 602 its, 125.0 s | 602 its, 331.8 s | 602 its, 191.3 s | **602 its, 64.8 s** |

¹ The 5b ROW line gives only the total and the two ring parts; the remainder (898.8 − 614.2 − 151.3 s) is the
MLFMM setup, the right-hand side and `op::assemble_diagonal` on the loaded machine.

**U, d₀ = 5 (matvec case):** operator build 291.9 s (ring near 123.3 s, ACA far ring 163.6 s; 5b 1 131.9 s, 5a
956.2 s), fast operator 3 485 MB (ring near 707, far 659 MB), matvec 2.147·10⁻⁷ against dense (gate 10⁻⁵; 5b
2.147·10⁻⁷), matvec 0.183 s against 0.259 s dense, process peak **3.91 GB** (5b 3.58 GB): at d₀ = 5 the λ-scale
leaves (1.94 λ) give 7.4·10⁶ ring near pairs and a transient canonical-pair cache of 1.11 GB (estimate 1.12 GB,
`near_assembly_bytes`), larger than the near ring it fills (0.67 GB) — the 5c cache, not the far ring, sets the peak
there. At d₀ = 3 the cache (329 MB) stays below the far ring and the peak is unchanged.

The near ring is now 42 % (d₀ = 3) / 42 % (d₀ = 5) of the operator setup and the ACA far ring 56 %; together with the
MLFMM part the whole `Simulation::assemble()` is 0.31× the dense assembly at the 7c size.

**2N = 59 600 (L = 8.5 µm, the 6·10⁴ validation-large case "ACA far ring vs the exact far ring", passed, 3 258 s
against 4 834 s in 5b):** ring difference 5.056·10⁻⁵ relative to the ring product and 1.068·10⁻⁵ relative to the
operator product (5b: identical digits; criterion 10⁻⁴); fast operator 3 683 MB (ring near 215, ACA far 1 636 MB;
exact far ring 16 660 MB), peak 3.86 GB; operator build **445.8 s** (ring near 105.0 s, ACA far 333.3 s) against
1 260.7 s in 5b (ring near 866.2 s, far 386.7 s, loaded machine); the exact far ring of the comparison took 2 808 s
(5b run 3 568 s); ring apply 0.071 s against 0.521 s exact.

**Review fixes (branch `wp/pml5b-review-fixes-2`, 2026-10-20):** after the W3 change (no ACA attempt for admissible
blocks with a break-even rank below `kPmlRingMinAcaRank` = 4 under the automatic cap; the row / column cache of a
failed attempt freed before the exact block) and the ACA zero-rows probe column, the U d₀ = 3 case was rerun once
(`-R "7h: pml_ring fast path vs dense"`, passed, 998 s, machine idle). Every printed digit and the stored ring
(blocks, ranks, bytes) are unchanged (matvec 2.081·10⁻⁵, dR +4.762·10⁻⁸, l2 3.328·10⁻⁶); the 884 skipped blocks
save 2 460 generated rows and columns and 223 320 triangle-pair evaluations (1.2 %), within the timing noise: ACA far
ring 120.2 s (integrated 119.0 s), ring near 85.6 s, `Simulation::assemble()` 208.6 s against 657.9 s dense. On the
flat CI case (`specklebem_pml_ring --case flat_ci --ring-compare`, before / after alternating, two rounds each) no
block is skipped and the RING lines are identical (ring err 7.020·10⁻⁶, generated rows and columns 8 280, far ring
1.8–1.9 s). The probe column never fired in either case (generated columns unchanged up to the skipped blocks).
Raw lines in `pml_ring_rows.txt` ("5b + 5c integrated, review fixes").

## 5d: pair memo (WP-PML5d)

Lead note 2026-10-22, "WP-PML5d"; branch `wp/pml5d-ring-pair-memo` (2026-10-25). `detail::BlockSource`
(`src/compression/mlfmm/pml_ring_far.cpp`) generates the ACA rows and columns of a far block from
triangle-pair blocks: `row(i)` evaluates the pairs of the row's two test triangles with every source triangle of
the block, `column(j)` those of every test triangle with the column's two source triangles, `full()` (exact and
rank-cap blocks) every pair once. Rows sharing a test triangle, columns sharing a source triangle, row / column
overlaps and the exact generation after a failed attempt meet the same pairs again. Measured on the 12-core
desktop with 128 GB, win-release build; the machine was shared all day with other work packages' runs
(preconditioner and far-field benchmarks), so absolute times are 1.5–3× the idle ones of the 5b + 5c record and
vary by up to 2× between runs; evaluation counts and hashes are deterministic.

### 5d: pair memo study (step 0)

Distinct triangle pairs per block next to `pair_evaluations` (`PmlRingFarBlock::pair_distinct`,
`PmlRingFarStats::pair_distinct`, `describe()`, the RING line; per-level PAIRS lines of `specklebem_pml_ring
--ring-compare`), counted exactly from the rows / columns / `full()` passes already made (a pair was met before iff
its test triangle had a generated row, its source triangle a generated column, or `full()` ran). Cases
(`--ring-compare --no-exact`, d₀ = 3, η = 2): U (rough Ag, 2N = 22 816), the 2N = 59 600 Ag case (L = 8.5 µm,
`--case u --patch 8.5e-6 --map-patch 8.5e-6`) and the 7d-Si control geometry (`--case u --material si`, rough Si
on the U mesh; the driver has no flat Si 7a case). Evaluations / distinct pairs (the gate: implement if ≥ 1.3
overall):

| block level, kind | U (Ag) | 2N = 59 600 (Ag) | U Si (7d control) |
|---|---|---|---|
| level 2 ACA | 7.34 M / 5.44 M = 1.35 | 20.19 M / 16.32 M = 1.24 | 14.14 M / 8.63 M = 1.64 |
| level 3 ACA | 6.81 M / 4.28 M = 1.59 | 15.79 M / 11.41 M = 1.38 | 11.64 M / 5.32 M = 2.19 |
| level 3 rank-cap fallback | 0.29 M / 0.15 M = 1.99 | — | 1.20 M / 0.60 M = 2.01 |
| level 4 ACA (leaf level of U) | 2.80 M / 1.26 M = 2.22 | 11.23 M / 6.12 M = 1.83 | 2.31 M / 0.91 M = 2.55 |
| level 5 ACA (leaf level of 59 600) | — | 4.27 M / 1.81 M = 2.36 | — |
| leaf rank-cap fallback | 0.43 M / 0.32 M = 1.36 | 5.94 M / 1.88 M = 3.16 | 2.20 M / 0.81 M = 2.73 |
| leaf exact (non-admissible) | 1.24 M / 1.24 M = 1.00 | 1.71 M / 1.71 M = 1.00 | 1.24 M / 1.24 M = 1.00 |
| **admissible** | 17.68 M / 11.45 M = **1.54** | 57.42 M / 37.54 M = **1.53** | 31.48 M / 16.26 M = **1.94** |
| **all** | 18.92 M / 12.69 M = **1.49** | 59.13 M / 39.25 M = **1.51** | 32.72 M / 17.50 M = **1.87** |

The gate (≥ 1.3) is met in all three cases, so the memo was implemented. The repetition is largest on the finest
far level (nearly every row and column of a small block is generated, and a rank-cap fallback generates the block a
second time), smaller on the coarse levels (few generated rows of many, which share a test triangle less often),
and larger for Si (higher ranks: more generated rows and columns per block).

### Design

- **Memo strips.** `BlockSource` keeps the evaluated `op::TrianglePairBlocks` (576 bytes; 584 with the state byte:
  not evaluated / evaluated / no region left after the per-pair bound) in lazily allocated strips: `row()`
  allocates the strip of each of its test triangles (one entry per source triangle of the block), `column()` the
  strips of its source triangles (one entry per test triangle), as long as the strips of the block fit into
  `PmlRingParams::pair_memo_bytes`; a strip that does not fit is not allocated (its pairs are evaluated and not
  kept; `memo_limited`). A pair is looked up in its test triangle's strip, then in its source triangle's strip, and
  evaluated (and stored in the first strip that exists) only if neither holds it. `full()` reads the memo and
  allocates nothing (it meets each pair once). This holds only what was actually generated (a dense |T| × |S| memo
  of a level-2 block at U would need ≈ 1 GB); the "memo per test triangle only" fallback of the lead note is the
  special case of a budget that admits the test strips only.
- **Bitwise.** The loop nesting of `row()` and `column()` is unchanged (in `column()` the outer loop over `test_`,
  the inner one over the column's source triangles); only the call of `evaluate()` became the lookup `pair()`, so
  the summation order of every entry is that of `main`. `PairBlockEvaluator::evaluate` overwrites its whole output
  (`pair_blocks` zero-fills it), so a memoised pair has the bits of a fresh evaluation.
- **Budget.** Per block (one block per assembling thread at a time), default 16 MiB: at U it holds every block of
  the two finest far levels completely (largest memo 9.9 MB) and 124 of the 128 level-2 blocks in part; 64 MiB
  would hold all of U (largest 46 MB) but at 2N = 59 600 a level-2 block needs up to 131 MB. Study (`--pair-memo
  BYTES`, kernel evaluations):

| memo budget | U | 2N = 59 600 |
|---|---|---|
| 0 (= `main`) | 18.92 M | 59.13 M |
| 4 MiB | 14.33 M (514 blocks at the budget) | — |
| **16 MiB (default)** | **12.99 M** (124) | **41.41 M** (406) |
| 64 MiB | 12.69 M (0) | 39.46 M (86) |
| 1 GiB | 12.69 M (0, largest 46 MB) | 39.25 M (0, largest 131 MB) |
| distinct pairs (floor) | 12.69 M | 39.25 M |

  16 MiB gets 94 % (U) / 89 % (59 600) of the possible saving with at most 12 × 16 MiB = 201 MB of transient memo on
  the 12 threads.
- **Memory accounting.** `BlockSource::cache_bytes()` includes the memo; `release_cache()` still frees the row /
  column caches before a fallback's exact block is allocated and keeps the memo (≤ the budget) for `full()`;
  `PmlRingEstimate::far_memo_bytes` = budget × min(threads, row clusters) enters `pml_ring_peak_bytes` (near +
  max(near cache, far exact + memo)), `PmlRingFarStats::memo_peak_bytes` (threads × largest block memo) enters
  `PmlRingOperator::peak_bytes()`. Measured process peaks: U 1.61 → 1.65–1.67 GB, 2N = 59 600 3.86–3.87 →
  3.93–3.94 GB, Si 7.48 → 7.53–7.55 GB.
- **Statistics.** `pair_evaluations` now counts kernel evaluations, `pair_memo_hits` the blocks taken from the memo
  (their sum is the 5b count), `pair_distinct` the distinct evaluated pairs; `pair_reductions` / `pair_skips` count
  each distinct pair once (5b counted every visit), and the INFO summary / `describe()` say "reduced R of
  (pair_distinct + pair_skips)". `memo_budget`, `memo_bytes_max`, `memo_limited_blocks`, `memo_peak_bytes` per
  ring; `memo_bytes`, `memo_limited` per block. Driver: `--pair-memo BYTES`, PAIRS lines, RINGHASH (FNV-1a of the
  ring products), `--memo-ab` (below).

### Before / after

**Bitwise.** The RINGHASH of the ring products (three fixed vectors) is the same for `main` (0b2df8a with only the
hash print added to the driver) and the branch in every run: U `0c7823704fa32159`, 2N = 59 600
`8b8d67eb1d996525`, Si `a5d3b4ef354f918e`; also for every memo budget of the study (0 … 1 GiB). Unit tests:
the existing bitwise gates of `test_pml_ring` (exact blocks vs the 5a ring, thread-count independence, plan
sharing, etc.) unchanged and green with the memo on by default; new: the applied far ring (ring e_k for every k,
i.e. every stored ACA and exact entry) and the block metadata with budgets 0 / 30 kB / unlimited bitwise equal on
the tiny flat and rough cases and a lossy object and exterior (skips), with forced ACA and the default parameters
(fallbacks), plus assembly with 1 / 2 / 3 / all threads with the default and a small budget.

**Separate processes** (`specklebem_pml_ring --ring-compare --no-exact`, one process per run, order main, branch,
branch, main; far ring assembly time, kernel evaluations):

| case | main r1 / r2 | branch r1 / r2 | evaluations main → branch (memo hits) |
|---|---|---|---|
| U (Ag) | 230.4 / 444.7 s | 170.2 / 232.0 s | 18.92 M → 12.99 M (5.93 M), −31 % |
| 2N = 59 600 (Ag) | 850.0 / 506.3 s | 920.6 / 643.4 s | 59.13 M → 41.41 M (17.72 M), −30 % |
| U Si (7d control) | 2 375.8 / 1 797.9 s | 1 059.4 / 1 568.0 s | 32.72 M → 19.53 M (13.20 M), −40 % |

The load of the other runs swung the times by up to 2× between consecutive processes (U main r2, 59 600 branch
r1), so these rounds bound the effect only loosely. **In one process** (`--memo-ab --reps R`: the far ring alone,
standalone with `assemble_near = false`, on the case's Problem and octree, memo off and on alternating; far time /
process CPU time of the build):

| case | memo off | memo on (16 MiB) | on / off |
|---|---|---|---|
| U, 3 reps | 276.8 / 249.0 / 282.5 s; CPU 2 631 / 2 539 / 2 573 s | 134.1 / 132.3 / 186.7 s; CPU 1 831 / 1 774 / 1 883 s | wall 0.56, CPU 0.71 |
| 2N = 59 600, 2 reps | 730.0 / 655.1 s; CPU 7 511 / 7 797 s | 478.5 / 363.1 s; CPU 5 165 / 5 130 s | wall 0.61, CPU 0.67 |
| U Si, 1 rep | 1 694.5 s; CPU 15 990 s | 1 600.7 s; CPU 9 698 s | wall 0.94, CPU 0.61 |
| flat CI (2N = 4 592), 2 reps | 7.2 / 6.9 s; CPU 36.0 / 36.1 s | 4.3 / 4.2 s; CPU 22.2 / 22.6 s | wall 0.60, CPU 0.62 |

The CPU time falls with the kernel evaluations (0.69 / 0.70 / 0.60 / 0.58 of them at U / 59 600 / Si / flat CI);
the wall time falls somewhat more on this loaded machine for Ag, but hardly for Si: there the far assembly is
bounded by its longest row cluster (OpenMP over row clusters, each cluster's blocks serially; CPU / 12 threads =
1 332 s without and 808 s with the memo against 1 694 / 1 601 s wall), presumably a coarse cluster whose blocks are
larger than the 16 MiB memo (open point: balance the far assembly within the coarse row clusters, or a larger
budget for them). With the 5b + 5c idle far-ring times (119 s at U, 333 s at 2N =
59 600) the expected idle far ring is ≈ 80 s / ≈ 230 s, i.e. `Simulation::assemble()` at U ≈ 0.25× the dense
assembly instead of 0.31×. Raw lines in `pml_ring_rows.txt` ("WP-PML5d").
