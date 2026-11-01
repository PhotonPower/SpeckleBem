# Phase 7 profile of the production PML route (WP-PF1)

Where the per-surface and the per-solve time of the "mlfmm" PML route goes, as the basis of the Phase 7 decision
(more threads, a CUDA backend per ADR 0005, or algorithmic changes). Library code: stage statistics only, no
behaviour change — `MlfmmOperator::setup_stats` / `apply_stats`, `MlfmmFarOperator::setup_stats` / `apply_stats`,
`PmlRingOperator::apply_stats` (with `reset_apply_stats`, `set_stage_timing`; mutex-guarded sums, the clock is read
between the stages only, results bitwise equal with timing on or off; `tests/unit/test_mlfmm_stage_stats.cpp`).
Driver `benchmarks/profile_study.cpp` (`specklebem_profile_study`, built with `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`);
raw lines `profile_rows.txt`. Work package WP-PF1 (formerly WP-P1 in the branch names). This record ranks; it
recommends nothing (the lead decides).

**Measured on a loaded machine (user decision 2026-11-01: measure now, under the load of a foreign compute job).**
Absolute seconds and the thread-scaling table are **preliminary (loaded machine)**; the stage **shares** and the
Amdahl ranking are the result. Where an idle reference exists (WP-SC4 / WP-SC5 / WP-KR1 rows) the ratio loaded / idle
is given as a load indicator.

## Setup

| item | value |
|---|---|
| machine | 12-core desktop (24 logical threads, SMT), 128 GB RAM, win-release (MSYS2 UCRT64 GCC 16, OpenBLAS). `omp_get_max_threads()` = 24 by default: the earlier records (WP-SC3…SC5, WP-KR1) ran at 24 threads, not 12 |
| load | a foreign single-process compute job (not ours) ran throughout at 40–100 % of one logical thread (≈ 2–4 % of the machine; per-run values below); between runs the machine was otherwise quiet (total CPU 5–9 %). One run (16 µm, first attempt) overlapped a foreign build and a SpeckleBem build + ctest of the main checkout (not ours, not of this worktree): it was rerun; its numbers are kept in `profile_rows.txt` (marked disturbed) and quoted only as an illustration of the disturbance under "Discarded runs" (assembly 1038 s vs 739 s, orthogonalisation 27.3 s vs 0.63 s); no table of this record uses them. No build or test of this worktree ran during any measurement (both presets were built and tested before) |
| cases | rough PML route of the preconditioner study, `pml_rough_ag::u_case()`: σ = 50 nm, Lc = 500 nm, seed 1, h = λ/4, W_t = W_g = 0.5 µm, W_pml = 2λ, beam w0 = λ = 500 nm at normal incidence (0° p). **Ag U** (L = 3.75 µm, 2N = 22 816), **Ag 16 µm** (`--patch 16e-6 --map-patch 16e-6`, 2N = 152 960), **Si U** (`--material si`, 2N = 22 816) |
| operator | compression "mlfmm", d₀ = 3, ICTF; the route's default preconditioner (ILUT of the near blocks A and D, fill 2, drop 10⁻², WP-SC4; A and D factorised and applied in parallel, each sequentially) |
| solver | full GMRES (CGS2, no restart), left preconditioning, tolerance 10⁻⁷ (`pml_route::fast_config`, as the WP-SC4 / WP-SC5 rows) |
| stages | **assembly** = `Simulation::assemble()` (MLFMM operator: octree, PML plan, far part, real-real near part, PML ring = near ring (stretched near pairs, cache pass + scatter) + ACA far ring; right-hand side; ILUT setup); **solve** = `solver::gmres` with timing adapters around the operator and the preconditioner (matvec split by the library stats into near / far passes / ring near / ring ACA); **standalone** matvecs (10 at U, 5 at 16 µm) and ILUT applies (20 / 10) of a fixed random vector; **post** = windowed far field on the reflection hemisphere (Gauss-Legendre 90 × 180 = 16 200 directions, the records' R), the two 1° cuts (718 directions), and the scattered field (`post::exterior_scattered_field`, degree 6) on a 161 × 161 plane at z = −1 µm over ±0.9 pml_start (the WP-S2 sampling); peak working set |
| thread scaling | `OMP_NUM_THREADS` = 1 / 6 / 12 / 24, `--no-solve` (assembly, standalone matvecs and ILUT applies; no GMRES, no post) except at 24 threads (full runs) |
| commands | `specklebem_profile_study --tag ag_u_t24 --matvecs 10 --ilut-applies 20`; `--tag si_u_t24 --material si --matvecs 10 --ilut-applies 20`; `--tag ag_L16_t24b --patch 16e-6 --map-patch 16e-6 --matvecs 5 --ilut-applies 10`; scaling: `--tag <case>_t<n> [--material si] --no-solve --matvecs 10 --ilut-applies 20` (Ag U at 1 thread: `--matvecs 3 --ilut-applies 10`), all with `--summary`, one study process at a time |
| repeats | three full runs of the same driver (be01630, identical library code, without `--no-solve`) queued by the first WP-PF1 session ran earlier the same night under the same foreign job (Ag U at 24 and 12 threads, Si U at 24); they are kept in `profile_rows.txt` as repeats |

## Load conditions per run

Foreign job: its CPU share in % of one logical thread (5 samples before / after each run, `typeperf`); total: the
machine total in the same samples (before the run = foreign job + idle machine). "Own builds": builds / tests of this
worktree (none ran during any measurement). Utilisation: mean machine CPU during the assembly window (30 s samples)
minus the foreign job, in busy threads.

| run | threads | start–end | foreign job before / after | total before / after | other load | assembly utilisation |
|---|---|---|---|---|---|---|
| ag_u_t24 | 24 | 01:02–01:06 | 98 / 97 % | 7.0 / 7.7 % | none | 23.0 of 24 |
| si_u_t24 | 24 | 01:06–01:20 | 79 / 1 % (job ended at ≈ 01:19:40, the run's post-processing) | 7.3 / 16.2 % (the foreign build had just started) | none during the assembly and GMRES (the foreign build began at ≈ 01:20, at the run's end) | 16.3 of 24 (**ACA far-ring tail**: the last ≈ 3.5 min of the 13 min assembly ran on ≈ 5 threads) |
| ag_L16_t24 (first attempt, **disturbed**) | 24 | 01:20–01:42 | — | 10.9 / 100 % | a foreign build throughout; SpeckleBem build (01:30) and ctest (01:31–01:51) of the main checkout | — (machine saturated by others) |
| ag_u_t12 | 12 | 02:06–02:11 | 100 / 40 % | 9.1 / 6.3 % | none | 12.3 of 12 |
| ag_u_t6 | 6 | 02:11–02:19 | 86 / 47 % | 8.3 / 5.1 % | a foreign build for ≈ 1 min (02:13–02:14; the foreign job paused 02:12–02:13) | 6.4 of 6 |
| si_u_t6 | 6 | 02:19–02:45 | 64 / 99 % | 6.4 / 8.6 % | one foreign-build sample (02:33) | 6.2 of 6 |
| ag_L16_t24b (rerun) | 24 | 02:45–03:00 | 100 / 100 % | 8.7 / 8.7 % | none | 20.5 of 24 (the last ≈ 1.5 min at ≈ 5 threads: ACA tail, ILUT setup on 2 threads, rhs) |
| si_u_t12 | 12 | 03:00–03:15 | 100 / 100 % | 8.4 / 5.3 % | none | 11.6 of 12 |
| ag_u_t1 | 1 | 03:15–04:00 | 100 / 69 % | 7.8 / 6.0 % | none | — (one thread) |

A queued runner of the first WP-PF1 session started one more study (Si U, 12 threads) at 01:51:14, 13 s before this
session's ag_u_t12 attempt; that attempt was stopped at once (no rows) and repeated at 02:06 after the foreign study
had disappeared (at ≈ 02:04, not stopped by this session; it wrote no rows). From then on every run waited for the
absence of other SpeckleBem processes and builds.

Discarded runs (none enters a table of this record): (1) the first Ag U run of this work package, 35 % slower than
idle — assembly 218.7 s against the idle 161.9 s (WP-SC4) — under a multi-threaded foreign job that a check by
process names did not detect (not in `profile_rows.txt`); (2) the disturbed 16 µm attempt ag_L16_t24 (rows kept in
`profile_rows.txt`, marked disturbed): assembly 1038.4 s vs 739.4 s of the rerun, orthogonalisation 27.3 s vs 0.63 s,
GMRES wall 121.7 s vs 59.5 s, with iterations and R identical to the rerun; (3) the stopped ag_u_t12 attempt above
(no rows).

Load indicators (loaded / idle; idle rows of WP-SC4 / WP-SC5 at 24 threads):

| case | assembly without ILUT (Z + rhs) | matvec (standalone / in GMRES) | GMRES wall | ILUT setup |
|---|---|---|---|---|
| Ag U | 175.5 s vs 161.9 s = **1.08** | 0.1057 / 0.1097 s vs 0.100 s = **1.06 / 1.10** | 2.97 s vs 2.7–2.9 s = 1.02–1.10 | 2.38 s vs 1.41 s = 1.69 |
| Ag 16 µm (rerun) | 720.1 s vs 675.2 s = **1.07** | 0.758 / 0.747 s vs 0.743–0.757 s = **1.00–1.02** | 59.5 s vs 58.2 s = 1.02 | 19.3 s vs 20.3 s (WP-SC5, an upper bound) = 0.95 |
| Si U | 778.9 s vs 622.9 s = **1.25** (vs 695–728 s of the WP-KR1 "idle" runs: 1.07–1.12) | 1.1215 / 1.1194 s vs 0.994 s = **1.13** | 16.95 s vs 14.7 s = 1.15 | 0.84 s (no idle WP-SC5 value; WP-SC4 1.83 s before the faster kernel) |

Repeats (first session's runner, same night, same foreign job): Ag U at 24 threads assembly 176.2 s (this record
177.9 s, −1.0 %), matvec 0.1086 s; Ag U at 12 threads 274.7 s (279.2 s, −1.6 %); Si U at 24 threads 752.7 s
(779.7 s, −3.5 %), matvec 1.0055 s (1.1215 s, −10 %). At stage level the shares agree: the assembly stages within
1.3 percentage points (ring 97.0 / 96.9 %, Si ring 92.1 / 92.9 %; the largest difference is the Si ACA far ring,
79.4 / 80.7 %), the matvec parts near / far / ring in GMRES within 0.4 percentage points (standalone within 2.5, the
repeats averaged 3 matvecs against 10). The per-pass far shares do not agree that well (up to ≈ 4 percentage
points in GMRES and ≈ 12 standalone at Ag U; section "Limits of this measurement").

## Results at 24 threads

### Summary

| case | 2N | assembly | GMRES (its) | per iteration | matvec | ILUT apply | far field (hemisphere) | plane field (25 921 points) | peak working set | R |
|---|---|---|---|---|---|---|---|---|---|---|
| Ag U | 22 816 | 177.9 s | 2.97 s (23) | 0.129 s | 0.106 s | 0.0116 s | 1.41 s | 12.55 s | 2.14 GB | 0.97833914 |
| Ag 16 µm | 152 960 | 739.4 s | 59.5 s (68) | 0.875 s | 0.758 s | 0.095 s | 13.92 s | 62.20 s | 20.90 GB | 0.97780528 |
| Si U | 22 816 | 779.7 s | 16.95 s (14) | 1.211 s | 1.12 s | 0.0088 s | 1.67 s | 13.12 s | 7.65 GB | 0.38713580 |

Operator memory 1 525 MB / 13 334 MB / 7 206 MB (Si: far part 5 974 MB, the expanded object region), ILUT 111 /
1 196 / 69 MB; the peak working set is reached in the assembly (Si: +0.5 GB in GMRES / post). Peak at the other thread
counts: Ag U 2.13–2.23 GB, Si U 7.42–7.48 GB at the end (7.05–7.09 GB after the assembly): thread-count independent
within the allocator noise.

### Assembly (per surface)

| stage | Ag U | share | Ag 16 µm | share | Si U | share |
|---|---|---|---|---|---|---|
| **PML ring, ACA far ring** (`ring far`) | 83.17 s | **46.8 %** | 550.04 s | **74.4 %** | 629.02 s | **80.7 %** |
| **PML ring, near ring** (stretched near pairs: cache pass / scatter) | 89.25 s (89.07 / 0.12) | **50.2 %** | 135.55 s (134.83 / 0.48) | **18.3 %** | 95.02 s (94.46 / 0.51) | **12.2 %** |
| far part (MlfmmFarOperator: plan / translators / patterns) | 0.92 s (0.66 / 0.03 / 0.21) | 0.5 % | 5.86 s (3.57 / 0.31 / 1.97) | 0.8 % | 45.90 s (R2 36.47 / 1.37 / 7.06, R1 0.99) | 5.9 % |
| near part, real-real pairs | 1.85 s | 1.0 % | 20.42 s | 2.8 % | 8.63 s | 1.1 % |
| PML plan (classification, split pattern, ring estimate) | 0.14 s | 0.1 % | 0.76 s | 0.1 % | 0.14 s | 0.0 % |
| octree | 0.00 s | 0.0 % | 0.01 s | 0.0 % | 0.00 s | 0.0 % |
| right-hand side | 0.20 s | 0.1 % | 7.51 s | 1.0 % | 0.17 s | 0.0 % |
| ILUT setup (A ‖ D; A: extract / order / eliminate) | 2.38 s (A 2.30: 0.09 / 0.56 / 1.65; D 0.43) | 1.3 % | 19.27 s (A 19.14: 1.08 / 3.64 / 14.42; D 5.46) | 2.6 % | 0.84 s (A 0.82: 0.11 / 0.26 / 0.45; D 0.59) | 0.1 % |
| **total** (`Simulation::assemble`) | **177.90 s** | | **739.41 s** | | **779.72 s** | |

ACA far ring: 1 853 / 4 340 / 1 317 ACA blocks and 1 092 / 1 048 / 1 092 exact blocks; 13.0 M / 115.6 M / 19.5 M
triangle-pair kernel evaluations (+ 5.9 M / 22.8 M / 13.2 M served by the pair memo), i.e. 154 / 114 / 773 µs of
thread time per evaluated pair (wall time × 24, including the imbalance tail) (Si: the object-region kernel at |k₂| h ≈ 6.8 needs high quadrature degrees). The
near ring at U (89 s for 0.53 M stretched pairs at 24 threads) costs as much as the whole ACA far ring.

### Solve (per solve)

| stage | Ag U | share of GMRES | Ag 16 µm | share | Si U | share |
|---|---|---|---|---|---|---|
| matvec (all calls; calls = its + 1) | 2.632 s (24 × 0.1097) | **88.6 %** | 51.560 s (69 × 0.7473) | **86.7 %** | 16.792 s (15 × 1.1194) | **99.1 %** |
|   far passes (all regions) | 0.0723 s per call | 58.4 % | 0.4327 s | 50.2 % | 1.0695 s | 94.6 % |
|   PML ring, ACA far ring | 0.0249 s | 20.1 % | 0.1767 s | 20.5 % | 0.0367 s | 3.3 % |
|   PML ring, near ring | 0.0061 s | 4.9 % | 0.0280 s | 3.2 % | 0.0064 s | 0.6 % |
|   near part, real-real | 0.0064 s | 5.1 % | 0.1092 s | 12.7 % | 0.0066 s | 0.6 % |
| ILUT apply | 0.303 s (24 × 0.0126) | **10.2 %** | 7.292 s (69 × 0.1057) | **12.3 %** | 0.132 s (15 × 0.0088) | **0.8 %** |
| orthogonalisation (CGS2) | 0.036 s (1.55 ms per it) | 1.2 % | 0.625 s (9.19 ms per it) | 1.1 % | 0.027 s (1.96 ms per it) | 0.2 % |
| rest | 0.001 s | 0.0 % | 0.007 s | 0.0 % | 0.001 s | 0.0 % |
| **GMRES wall** | **2.971 s** (0.129 s per it) | | **59.485 s** (0.875 s per it) | | **16.952 s** (1.211 s per it) | |

Far passes inside the matvec (share of the matvec in GMRES; Ag: only region R1 expands, R2's far part is exact and
negligible; Si: R2 = 93 % of the far time; the Ag U column varies by up to ≈ 4 percentage points per pass between
runs, see "Limits of this measurement"):

| pass | Ag U | Ag 16 µm | Si U (R1 + R2) |
|---|---|---|---|
| aggregation (leaf) | 13.7 % | 16.7 % | 15.1 % |
| upward (interpolation + shift) | 11.5 % | 8.4 % | 14.6 % |
| translation | 17.9 % | 9.4 % | 36.7 % |
| downward (shift + anterpolation) | 8.6 % | 7.5 % | 14.8 % |
| reception (leaf + elevated) | 13.6 % | 15.7 % | 14.2 % |
| exact far parts | 0.4 % | 0.0 % | 0.0 % |

ILUT apply (standalone, A and D in parallel): 0.0116 / 0.0953 / 0.0088 s; a sequential replica of the triangular
solves (bitwise equal to the library's apply) gives A forward + backward 0.0113 / 0.0989 / 0.0074 s and D 0.0028 /
0.0255 / 0.0016 s, so the apply time is the A block's (the parallel D hides behind it); 3.8 M / 43.5 M / 2.3 M
nonzeros of L + U in A, i.e. 3.0 / 2.3 / 3.2 ns per nonzero: memory-streaming bound. The replica's A time against the
parallel library apply is a comparison of two implementations (not a stage-sum check): −3 % (U, 0.0113 vs
0.0116 s), +4 % (16 µm, 0.0989 vs 0.0953 s) and −16 % (Si U, 0.0074 vs 0.0088 s).

### Post-processing (per excitation)

| item | Ag U | Ag 16 µm | Si U |
|---|---|---|---|
| far field, hemisphere 16 200 directions (window) | 1.41 s (0.087 ms per direction) = 0.47 × GMRES | 13.92 s (0.859 ms) = 0.23 × GMRES | 1.67 s (0.103 ms) = 0.10 × GMRES |
| two 1° cuts, 718 directions | 0.09 s | 0.72 s | 0.09 s |
| plane field 161 × 161 = 25 921 points (degree 6) | 12.55 s (0.484 ms per point) = **4.2 × GMRES** | 62.20 s (2.40 ms) = **1.05 × GMRES** | 13.12 s (0.506 ms) = 0.77 × GMRES |

The plane field evaluates every triangle for every point (no compression): its cost grows with N × points.

## Consistency checks

- **Stage sums vs totals (acceptance ≤ 10 %).** Every PROFILE line also prints its unattributed rest, computed as the
  measured total minus the separately timed stages: assembly — `Simulation::assemble` total vs operator stages +
  rhs + ILUT (both from `Simulation::report()`): rest 0.00 s of 177.9 / 739.4 / 779.7 s; operator constructor: Z other
  0.00 s; far constructor: far other 0.00 s; ring: ring other 0.00 s. Matvec — the driver's timing adapter around
  `apply()` (an independent clock) vs near + far + ring + other of the library stats: 100.0 %. Since the library's
  "other" is its own total minus near, far and ring, this line compares the driver's clock with the library's total
  only; the informative numbers are "other (additions + rest)", the additions `y += …` and the temporaries (≤ 0.1 %),
  and "far other", far total minus the passes (≤ 0.003 s per call, ≤ 0.4 %). GMRES — matvec + ILUT apply +
  orthogonalisation vs the wall time: rest 0.001 / 0.007 / 0.001 s (≤ 0.05 %). All inside 10 %. (The ILUT replica
  against the library apply is a comparison of two implementations, not a sum check: section "Solve", −3 % / +4 % /
  −16 %.)
- **Results unchanged (iterations and R equal to the WP-SC4 / WP-SC5 rows).** Ag U 23 iterations, true residual
  6.28·10⁻⁸, R 0.97833914 (WP-SC4 / SC5: 23, 6.28·10⁻⁸, 0.97833914); Ag 16 µm 68, 5.57·10⁻⁸, 0.97780528 (68, 5.57·10⁻⁸,
  0.97780528); Si U 14, 4.58·10⁻⁸, 0.38713580 (14, 4.58·10⁻⁸, 0.38713580). Identical in every full run (also the
  disturbed 16 µm attempt and the three repeats). Operator and ILUT memory, nnz(L + U), ACA block counts and pair
  evaluations are identical at every thread count. The stage timing is bitwise neutral by construction and test
  (`tests/unit/test_mlfmm_stage_stats.cpp`: matvec with timing on and off bitwise equal, also with a PML ring); the
  driver's ILUT replica is bitwise equal to the library apply in every run (the driver now throws on a mismatch
  instead of printing the split).

## Thread scaling (preliminary, loaded machine)

Seconds, and in brackets the speed-up against the 1-thread run (Ag U) or the 6-thread run (Si U) with the parallel
efficiency (speed-up / thread ratio). 24 threads = 12 cores with SMT, so 12 → 24 is the SMT gain and an
efficiency of 50 % at 24 threads means "no gain from SMT". Assembly and matvecs from the `--no-solve` runs (1 / 6 /
12 threads) and the full runs (24 threads); 3 (1 thread) / 10 standalone matvecs and 10 / 20 ILUT applies per run.
**Preliminary (loaded machine, one run per point).**

| stage | 1 thread | 6 threads | 12 threads | 24 threads |
|---|---|---|---|---|
| Ag U assembly (`Simulation::assemble`) | 2706.3 s | 478.6 s¹ (5.65×, 94 %) | 279.2 s (9.69×, 81 %) | 177.9 s (15.21×, 63 %) |
|   PML ring, near ring | 1296.5 s | 255.5 s (5.07×, 85 %) | 142.2 s (9.12×, 76 %) | 89.2 s (14.53×, 61 %) |
|   PML ring, ACA far ring | 1374.3 s | 214.6 s (6.41×, 107 %) | 132.2 s (10.40×, 87 %) | 83.2 s (16.52×, 69 %) |
|   near part, real-real | 29.18 s | 5.51 s (5.30×, 88 %) | 2.40 s (12.16×, 101 %) | 1.85 s (15.77×, 66 %) |
|   far part setup | 2.77 s | 0.98 s (2.83×, 47 %) | 0.84 s (3.30×, 27 %) | 0.92 s (3.01×, 13 %) |
| Ag U matvec (standalone, per call) | 0.5311 s | 0.1420 s (3.74×, 62 %) | 0.1004 s (5.29×, 44 %) | 0.1057 s (5.02×, 21 %) |
|   far passes | 0.3943 s | 0.0962 s (4.10×, 68 %) | 0.0636 s (6.20×, 52 %) | 0.0672 s (5.87×, 24 %) |
|   PML ring (near + ACA) | 0.1205 s | 0.0390 s (3.09×, 51 %) | 0.0301 s (4.00×, 33 %) | 0.0317 s (3.80×, 16 %) |
|   near part, real-real | 0.0163 s | 0.0067 s (2.43×, 41 %) | 0.0066 s (2.47×, 21 %) | 0.0066 s (2.47×, 10 %) |
| Ag U ILUT apply (A ‖ D, per call) | 0.0110 s | 0.0155 s (0.71×, 12 %) | 0.0095 s (1.16×, 10 %) | 0.0116 s (0.95×, 4 %) |
| Si U assembly | — | 1526.5 s | 847.0 s (1.80×, 90 %) | 779.7 s (1.96×, 49 %) |
|   PML ring, near ring | — | 266.2 s | 141.4 s (1.88×, 94 %) | 95.0 s (2.80×, 70 %) |
|   PML ring, ACA far ring | — | 1171.3 s | 646.9 s (1.81×, 91 %) | 629.0 s (1.86×, 47 %) |
|   far part setup (R2 order searches 38.5 / 31.4 / 36.5 s) | — | 60.2 s | 44.0 s (1.37×, 68 %) | 45.9 s (1.31×, 33 %) |
|   near part, real-real | — | 27.37 s | 13.70 s (2.00×, 100 %) | 8.63 s (3.17×, 79 %) |
| Si U matvec (standalone, per call) | — | 1.1423 s | 0.9429 s (1.21×, 61 %) | 1.1215 s (1.02×, 25 %) |
|   far passes | — | 1.0949 s | 0.8988 s (1.22×, 61 %) | 1.0730 s (1.02×, 26 %) |
| Si U ILUT apply (per call) | — | 0.0073 s | 0.0067 s (1.09×, 54 %) | 0.0088 s (0.83×, 21 %) |

¹ Ag U at 6 threads: a foreign build overlapped ≈ 1 min of the assembly (02:13–02:14, load table); the Ag U
6-thread values carry that disturbance.

- **Assembly scales** (compute-bound kernel evaluations): Ag U 15.2× on 24 threads (9.7× on 12, 81 %; SMT adds
  1.57×); the near ring 14.5×, the ACA far ring 16.5× (above-linear at 6 threads: 6.4×¹; a hypothesis, not
  measured: cache effects of the smaller per-thread working set; the 1-thread baseline also ran 45 min next to a
  foreign thread at ≈ 100 % that may have shared its physical core, which would inflate every speed-up against it).
  Si U from 6 to 12 threads 1.80× (90 %), from 12 to 24 only 1.09×: the ACA far ring gains 1.03× (the tail, see the
  utilisation column), the near ring 1.49×; the far setup's order searches (31–39 s) do not scale at all.
- **The matvec saturates at 6–12 threads** (memory bandwidth): Ag U 3.7× at 6, 5.3× at 12 and 5.0× at 24 threads;
  Si U 1.21× from 6 to 12 and 0.84× from 12 to 24 (SMT slows it by 19 %). The real-real near product is flat
  beyond 6 threads (0.0066 s).
- **The ILUT apply does not scale** (two sequential triangular solves, A and D in parallel): 0.007–0.016 s at every
  thread count; the spread is run-to-run noise of a bandwidth-bound kernel under load.
- Repeats: Ag U at 12 threads 274.7 s (this table 279.2 s); not measured: Si U at 1 thread (≈ 3 h), Ag 16 µm at
  1 / 6 / 12 threads (dropped for the budget; the first 16 µm attempt had to be repeated).

## Phase 7 candidates, ranked by the attainable end-to-end gain (Amdahl)

Gain of the whole when one stage (share p) becomes S times faster: 1 / ((1 − p) + p / S); "∞" = the stage removed
(1 / (1 − p)), the upper bound of any speed-up of that stage. Shares from the 24-thread runs above.

### Per surface (assembly)

| rank | stage | Ag U p | ×2 / ×5 / ∞ | Ag 16 µm p | ×2 / ×5 / ∞ | Si U p | ×2 / ×5 / ∞ |
|---|---|---|---|---|---|---|---|
| 1 | PML ring, ACA far ring assembly | 46.8 % | 1.31 / 1.60 / 1.88 | 74.4 % | 1.59 / 2.47 / 3.90 | 80.7 % | 1.68 / 2.82 / 5.17 |
| 2 | PML ring, near ring assembly (stretched near pairs) | 50.2 % | 1.33 / 1.67 / 2.01 | 18.3 % | 1.10 / 1.17 / 1.22 | 12.2 % | 1.06 / 1.11 / 1.14 |
| — | both ring parts together | 96.9 % | 1.94 / 4.45 / 32 | 92.7 % | 1.86 / 3.87 / 13.7 | 92.9 % | 1.87 / 3.89 / 14.0 |
| 3 | far part setup (order searches, patterns) | 0.5 % | 1.00 / 1.00 / 1.01 | 0.8 % | 1.00 / 1.01 / 1.01 | 5.9 % | 1.03 / 1.05 / 1.06 |
| 4 | ILUT setup | 1.3 % | ≤ 1.01 | 2.6 % | 1.01 / 1.02 / 1.03 | 0.1 % | 1.00 |
| 5 | near part (real-real) | 1.0 % | ≤ 1.01 | 2.8 % | 1.01 / 1.02 / 1.03 | 1.1 % | ≤ 1.01 |
| 6 | right-hand side | 0.1 % | 1.00 | 1.0 % | ≤ 1.01 | 0.0 % | 1.00 |

Everything outside the PML ring is ≈ 7 % of the assembly (7.3 % at 16 µm, 7.1 % Si U, 3.1 % Ag U); any gain beyond
1.1× per surface has to come from the ring.

### Per solve (GMRES, one excitation on an assembled operator)

| rank | stage | Ag U p | ×2 / ×5 / ∞ | Ag 16 µm p | ×2 / ×5 / ∞ | Si U p | ×2 / ×5 / ∞ |
|---|---|---|---|---|---|---|---|
| — | whole matvec | 88.6 % | 1.80 / 3.43 / 8.8 | 86.7 % | 1.76 / 3.26 / 7.5 | 99.1 % | 1.98 / 4.82 / 106 |
| 1 | far passes | 58.4 % | 1.41 / 1.88 / 2.40 | 50.2 % | 1.33 / 1.67 / 2.01 | 94.6 % | 1.90 / 4.11 / 18.5 |
|   | of which translation | 15.9 % (± ≈ 4)² | 1.09 / 1.15 / 1.19 | 8.1 % | 1.04 / 1.07 / 1.09 | 36.4 % | 1.22 / 1.41 / 1.57 |
|   | of which aggregation + reception | 24.2 % (± ≈ 4)² | 1.14 / 1.24 / 1.32 | 28.1 % | 1.16 / 1.29 / 1.39 | 29.0 % | 1.17 / 1.30 / 1.41 |
|   | of which upward + downward | 17.8 % (± ≈ 4)² | 1.10 / 1.17 / 1.22 | 13.8 % | 1.07 / 1.12 / 1.16 | 29.1 % | 1.17 / 1.30 / 1.41 |
| 2 | PML ring apply, ACA blocks | 20.1 % | 1.11 / 1.19 / 1.25 | 20.5 % | 1.11 / 1.20 / 1.26 | 3.3 % | 1.02 / 1.03 / 1.03 |
| 3 | ILUT apply | 10.2 % | 1.05 / 1.09 / 1.11 | 12.3 % | 1.07 / 1.11 / 1.14 | 0.8 % | 1.00 / 1.01 / 1.01 |
| 4 | near part apply (real-real) | 5.1 % | 1.03 / 1.04 / 1.05 | 12.7 % | 1.07 / 1.11 / 1.14 | 0.6 % | ≤ 1.01 |
| 5 | PML ring apply, near ring | 4.9 % | 1.02 / 1.04 / 1.05 | 3.2 % | 1.02 / 1.03 / 1.03 | 0.6 % | ≤ 1.01 |
| 6 | orthogonalisation | 1.2 % | 1.01 | 1.1 % | 1.01 | 0.2 % | 1.00 |

² Ag U per-pass shares are good to roughly ± 4 percentage points only: the Ag U repeat at 24 threads splits the far
time differently between the passes (shares of the matvec in GMRES: translation 13.7 % vs 17.9 %, upward 15.3 % vs
11.5 %); the stage-level rows agree within 1.4 percentage points of the GMRES time (repeat vs this run: far passes
57.0 / 58.4 %, ACA ring apply 20.1 / 20.1 %, ILUT apply 11.3 / 10.2 %, near part 5.1 / 5.1 %, near ring 4.9 / 4.9 %).

The iteration count multiplies everything per iteration; the iteration lever itself (preconditioner, recycling) was
measured in WP-SC3…SC5 / WP-KR1 (recycling: ≤ 1.03×).

### Per excitation (operator reuse, WP-KR1: assembly once + k × (rhs + GMRES + far field [+ plane field]))

Shares of the total for k excitations of one surface (24 threads; "far field" = hemisphere only, "+ plane" = also the
161 × 161 plane field):

| case | k | output | total | assembly | of which ACA far ring | GMRES | far field | plane field | rhs |
|---|---|---|---|---|---|---|---|---|---|
| Ag U | 1 | far field | 182 s | 97.5 % | 45.6 % | 1.6 % | 0.8 % | — | 0.1 % |
| Ag U | 10 | far field | 224 s | 79.5 % | 37.2 % | 13.3 % | 6.3 % | — | 0.9 % |
| Ag U | 10 | + plane | 349 s | 50.9 % | 23.8 % | 8.5 % | 4.0 % | 36.0 % | 0.6 % |
| Ag U | 100 | far field | 636 s | 27.9 % | 13.1 % | 46.7 % | 22.2 % | — | 3.1 % |
| Ag U | 100 | + plane | 1 891 s | 9.4 % | 4.4 % | 15.7 % | 7.5 % | 66.4 % | 1.1 % |
| Ag 16 µm | 1 | far field | 813 s | 90.0 % | 67.7 % | 7.3 % | 1.7 % | — | 0.9 % |
| Ag 16 µm | 10 | far field | 1 541 s | 47.5 % | 35.7 % | 38.6 % | 9.0 % | — | 4.9 % |
| Ag 16 µm | 10 | + plane | 2 163 s | 33.8 % | 25.4 % | 27.5 % | 6.4 % | 28.8 % | 3.5 % |
| Ag 16 µm | 100 | far field | 8 823 s | 8.3 % | 6.2 % | 67.4 % | 15.8 % | — | 8.5 % |
| Si U | 1 | far field | 798 s | 97.6 % | 78.8 % | 2.1 % | 0.2 % | — | 0.0 % |
| Si U | 10 | far field | 968 s | 80.6 % | 65.0 % | 17.5 % | 1.7 % | — | 0.2 % |
| Si U | 10 | + plane | 1 099 s | 71.0 % | 57.3 % | 15.4 % | 1.5 % | 11.9 % | 0.2 % |
| Si U | 100 | far field | 2 659 s | 29.3 % | 23.7 % | 63.8 % | 6.3 % | — | 0.6 % |

Break-even (assembly = the k solves + far fields): Ag U k ≈ 39 (far field only) / 10 (with the plane field), Ag 16 µm
k ≈ 9 / 5, Si U k ≈ 42 / 24. Below these the assembly (in practice the ACA far ring and, at U, the near ring)
dominates; above them the solve and the post-processing do.

### Post-processing

| rank | item | gain per excitation (×∞ of that item) |
|---|---|---|
| 1 | plane field 161 × 161 (when requested) | it is 4.2× (Ag U), 1.05× (16 µm), 0.77× (Si U) the GMRES time: removing it shortens an excitation (rhs + GMRES + far field + plane) by 73 % / 43 % / 41 % |
| 2 | far field, hemisphere 16 200 directions | 0.47× / 0.23× / 0.10× GMRES: 31 % / 17 % / 9 % of an excitation without the plane field |

## Interpretation (per stage; no recommendation)

- **ACA far ring assembly** (rank 1 per surface at 16 µm and Si U, 47 % at Ag U; up to 3.9× / 5.2× by Amdahl).
  Compute-bound triangle-pair evaluations of the stretched (complex-coordinate) kernels, 114–773 µs of thread time
  per pair. *Threads*: Ag U 1.62× (6 → 12) and 1.59× (12 → 24, SMT); Si U 1.81× (6 → 12) but **1.03×** (12 → 24): at
  24 threads its last ≈ 3.5 min run on ≈ 5 threads (a tail of long blocks; at 12 threads no tail). With the Ag U SMT
  factor (1.59×) a balanced 24-thread Si ring would take ≈ 410 s instead of 629 s, i.e. ≈ 220 s (≈ 28 %) of the Si
  assembly is imbalance (an estimate from two runs, not a measurement of a fix); the 16 µm run shows a shorter tail
  (≈ 1 min). *CUDA*: the pair evaluations are independent and arithmetic-heavy (GPU-shaped), but each ACA block
  generates its rows and columns one after the other (pivot search), so the parallelism is across blocks (1 317–4 340)
  and within one row / column (hundreds of pairs) — a different structure from the per-thread blocks of today.
  *Algorithmic*: fewer evaluations (the pair memo already serves 16–40 % of them) or cheaper ones (at 12 threads,
  without the tail, a Si pair costs 397 µs of thread time against 122 µs for Ag U, 3.3×: the object-region kernel at
  |k₂| h ≈ 6.8 needs higher quadrature degrees; any change there is an accuracy question).
- **Near ring assembly** (rank 2 per surface; 50 % at Ag U, 12–18 % at 16 µm / Si U; ≤ 2.0×). Stretched near pairs
  with touching / singular quadrature in the cache pass. *Threads*: 1.80× (6 → 12) and 1.59× (12 → 24) at Ag U,
  1.88× / 1.49× at Si U — it scales; its share falls with the size (it grows with the ring, i.e. the perimeter, while
  the ACA far ring grows faster). *CUDA*: branchy (touching classes, Duffy pieces, fold-adaptive splits): harder to
  port than the regular kernels. *Algorithmic*: the translation classes (WP-PML5c) already copy 52 % of the touching
  pairs at U; the remainder is per-pair quadrature.
- **Far passes of the matvec** (rank 1 per solve: 50–58 % of the Ag solve, 95 % of the Si solve; up to 2.0–2.4× for Ag,
  18.5× for Si). Memory-bandwidth-bound: the matvec is fastest at 12 threads (Ag U 0.142 / 0.100 / 0.106 s at 6 / 12 /
  24, Si U 1.142 / 0.943 / 1.12 s); from 12 to 24 threads the matvec time changes by +5 % (Ag U) and +19 % (Si U).
  *CUDA*: this is the stage that a GPU's memory bandwidth (an order of magnitude above a desktop's) addresses; the far
  data are 0.5 GB (Ag U), 4.7 GB (16 µm), 6.0 GB (Si U). *Algorithmic*: less data per pass (the translation is 36 % of the Si matvec,
  aggregation + reception 24–29 % everywhere; ADR 0005 allows single precision only with a passing validation test).
- **ACA ring apply** (rank 2 per solve for Ag, 20 %; ≤ 1.26×), **ILUT apply** (rank 3, 10–12 % for Ag; ≤ 1.14×) and the
  **real-real near apply** (5–13 %): the ILUT apply is two sequential triangular solves in parallel (A ‖ D, ≈ 2–3 ns
  per nonzero, bandwidth-bound; at 1 thread A and D run one after the other); its time stays at 0.0095–0.0155 s
  (Ag U) at 1 to 24 threads.
- **Orthogonalisation** (≈ 1 %): nothing to gain at these iteration counts (68 iterations at 16 µm).
- **Per excitation** (WP-KR1 reuse): below k ≈ 10–40 excitations per surface the assembly dominates (the ACA far ring
  alone is 24–79 % of the total at k = 1–10); above, the solve and the post-processing do. A 161 × 161 plane field is a
  dense O(N × points) sum (no compression): 4.2× the GMRES time at Ag U, ≈ 1× at 16 µm, the largest per-excitation item
  at small sizes; it is regular and compute-bound (Ag U at 12 threads 20.10 s vs 12.27 s at 24 in the repeats,
  12.55 s in this record's own 24-thread run: it gains from SMT), i.e. both GPU-shaped and open to a fast-multipole
  evaluation.
- **What more threads would buy on this machine**: nothing per solve (bandwidth, saturated at 12); per surface the ring
  assembly gains 1.6× from SMT where the blocks are balanced (Ag) and only 1.03× where they are not (Si ACA tail).
  On a machine with more cores the assembly would scale further (Ag U ring: 15.5× on 24 threads of 12 cores, 129 % of the core count), the solve
  only with more memory bandwidth.
- **What a CUDA backend (ADR 0005) would address**: per solve, the far passes (50–95 %) and the ACA ring apply (20 %
  for Ag), together 79 % / 71 % / 98 % of a GMRES iteration at Ag U / 16 µm / Si U (Amdahl: ×5 on both gives 2.7× / 2.3× /
  4.6× per solve, ×∞ 4.7× / 3.4× / 48×, before host–device transfers); per surface, the pair evaluations of the ring (93–97 %), whose ACA structure is the
  obstacle; per excitation, the plane field. The Phase 7 DoD (≥ 3× end-to-end on the 393 k-unknown Ag sphere) is a
  different case (box route, no PML ring) not profiled here.

## Limits of this measurement

- The machine carried a foreign single-thread job throughout (one of 24 logical threads, and part of the shared memory
  bandwidth): absolute times are 0–25 % above the idle records (largest for Si U, whose idle reference itself scatters
  by +11 to +16 % between WP-SC4 and WP-KR1, `recycling.md`); the 16 µm rerun reproduces the idle matvec and GMRES
  times within 2 %.
- Shares are robust at stage level: the two Ag U runs and the two Si U runs at 24 threads agree within 1.3 percentage
  points for the assembly stages (largest: Si ACA far ring 79.4 / 80.7 %) and within 0.4 for the matvec parts near /
  far / ring in GMRES, although their absolute times differ by 1–3.5 % (assembly). The per-pass far shares are not
  robust: Ag U standalone translation 19.0 % vs 8.8 % and downward 7.1 % vs 19.1 % (this run vs the repeat), in GMRES
  translation 17.9 % vs 13.7 % and upward 11.5 % vs 15.3 %; Si U downward 14.8 % vs 13.3 % (shares of the matvec). The
  passes run in the same order in both runs (same code: aggregation, upward, translation, downward, reception); only
  the split of the far time between them moves. The Ag U per-pass rows of the Amdahl table are therefore good to
  roughly ± 4 percentage points.
- The thread scaling compares runs at different times under the same foreign job; with 1 of 24 logical threads
  occupied, the 24-thread runs lose more (relative) than the 1–12-thread runs, which share physical cores with nothing
  at first (Windows schedules the foreign thread on any core). The scaling numbers are therefore lower bounds of the
  idle scaling at 24 threads and upper bounds at 12 and below; "preliminary" applies to them in full.
- Wall-clock stage times inside OpenMP regions include the waiting of the threads at the end of a parallel loop (load
  imbalance); the utilisation column (busy threads from 30 s CPU samples) is the only, coarse, separation of
  imbalance from work. It shows one clear tail: the ACA far ring at 24 threads (Si U ≈ 3.5 of 10.5 min on ≈ 5
  threads; 16 µm ≈ 1 min).
- One case per size, one seed, normal incidence, d₀ = 3, the ILUT default: other operating points (Si at 16 µm, d₀ = 5,
  oblique beams) shift the shares (e.g. the Si far part grows with the area, the ring with the perimeter).
