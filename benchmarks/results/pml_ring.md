# Fast path of the PML route, baseline: MLFMM on the real functions + exact PML ring (WP-PML5a)

ADR 0008 amendment 2026-10-18 (decisions 1–3a, 5, 6), docs/05 "PML route", row 7h. Library code
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
