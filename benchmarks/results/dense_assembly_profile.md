# Dense assembly profile on rough boxes (WP-P2)

Why did the WP15 dense assemblies of the rough-surface boxes (2N ≈ 3·10⁴) take 1050–1540 s on
24 threads, when the icosphere records (2N = 15 360 in 8–11 s) extrapolate to ~40 s? Executable
`benchmarks/dense_assembly_profile.cpp` (`specklebem_dense_assembly_profile`, built with
`-DSPECKLEBEM_BUILD_BENCHMARKS=ON`).

## Setup

| Item | Value |
|---|---|
| Meshes | WP15 rough boxes: σ = 50 nm, Lc = 500 nm, h = 50 nm, seed 1, graded box with 400 nm coarse cells, no fine band; Si depth 5.65 µm, Ag 2 µm. L = 3.2 µm (the WP15 systems: Si 10 400 triangles, 2N = 31 200; Ag 9 760, 2N = 29 280) and L = 1.2 µm (Si 1 950 triangles, 2N = 5 850; Ag 1 710, 2N = 5 130). Icospheres r = 0.5 µm, n = 3 (2N = 3 840) and n = 4 (2N = 15 360) |
| Materials | λ = 500 nm, vacuum exterior; Si ε_r = 18.478 − 0.606j (k₂ = 54.0 − 0.886j µm⁻¹), Ag ε_r = −9.794 − 0.313j (k₂ = 0.63 − 39.3j µm⁻¹), glass n = 1.5 |
| Size classes | longest edge ≤ 2.5 h "fine" (top face, upper walls), ≤ 6 h "medium", > 6 h "coarse" (400 nm cells: longest edge 566 nm, \|k_Si\| h ≈ 30, k₀ h ≈ 7); a pair counts in the class of its larger triangle |
| Profile | `--profile`: every element_blocks call of the assembly (both regions) timed individually, rows distributed dynamically over 24 threads, CPU seconds summed; `--stride 20` samples every 20th test triangle (×20) |
| Machine | Windows 11, 24 threads, MSYS2 UCRT64 GCC 16, `win-release`, 2026-10-09. Other agents' builds and test runs shared the machine throughout: absolute times vary by up to ×2–3 between runs; compare within a table |

## Profile before (code of main e07ee8b), WP15 Si box (L = 3.2 µm, stride 20)

Total 8 510 CPU s for the full matrix (Ag: 3 839 CPU s). Per class (CPU s, share, mean µs per call):

| Class | Region | fine | medium | coarse |
|---|---|---:|---:|---:|
| touching (all) | ext + int | 46 s (0.5 %) | 1.2 s | 5.0 s |
| near | ext | 6.6 s, 19 µs | 3.3 s, 40 µs | 45 s, 145 µs |
| near | int | 19.5 s, 56 µs | 21 s, 250 µs | 80 s, 255 µs |
| far | ext | 216 s (2.5 %), 2.6 µs | 33 s, 8 µs | 1 093 s (12.8 %), 52 µs |
| far | int | 1 145 s (13.5 %), 14 µs | 794 s (9.3 %), 191 µs | **5 001 s (58.8 %), 240 µs** |

Chosen far-pair degrees (L = 1.2 µm meshes, calls of the full matrix; `kernels::plain_rule_degree`):

| Mesh / region | Degrees (calls) |
|---|---|
| Si, exterior | 4–6: 2.29 M; 8–10: 0.12 M; 12–14: 1.09 M; 17–19: 0.08 M (all coarse pairs need 12–17: k₀ h ≈ 7) |
| Si, interior | 8: 1.69 M; 9–14: 0.50 M; 17: 0.05 M; **19 (the cap): 1.36 M = 547 of 872 CPU s** |
| Ag, interior | 6: 1.63 M; 8–12: 0.64 M; 17: 0.04 M; **19: 0.42 M = 114 of 249 CPU s** |
| Ag, interior (after) | 3: 2.36 M; 4–6: 0.34 M; 8–9: 0.02 M; nothing above 9 |

Touching pairs (L = 1.2 µm, both materials): identical 384 points (WP7b table), shared edges 65.8 %
fold-adaptive (mean 3.96 pieces, 574 points, at most 1 980), shared vertices 26 % fold-adaptive
(mean 137 points, at most 420); all touching classes together cost < 3 % of the assembly, so
k-aware `quad_degree_sing` (WP7c part 3) was not pursued.

Schedule (makespan on 24 threads from the measured row costs, L = 1.2 µm):

| Mesh | ideal | static (before) | dynamic, index order | dynamic, largest first (after) |
|---|---:|---:|---:|---:|
| Si box | 36.4 s | 91.4 s (×2.51) | 44.1 s (×1.21) | — |
| Ag box | 10.4 s | 46.7 s (×4.50) | 11.6 s (×1.11) | — |
| Si box, after all changes | 14.5 s | 36.7 s (×2.54) | 15.1 s (×1.04) | 14.7 s (×1.02) |
| Ag box, after all changes | 2.19 s | 10.2 s (×4.66) | 2.39 s (×1.09) | 2.26 s (×1.03) |
| Si icosphere n = 3 | 2.45 s | 3.02 s (×1.23) | 2.57 s (×1.05) | — |

## Root causes

1. **Load imbalance** (all meshes, largest on rough boxes): the assembler distributed the test
   triangles of each colour with an OpenMP *static* schedule in index order. The box triangles
   come last in the mesh and their rows cost 10–30× a top-face row (every pair with a coarse
   triangle needs a high degree), so one or two threads got all coarse rows: ×2.5 (Si) and ×4.5
   (Ag) the ideal time, and on a shared machine every delayed thread holds up a whole colour.
2. **Coarse box cells in the object region**: the near/far degree selection estimates the
   relative block error from κ = sqrt((h/D)² + (0.15 |k| h)²); for the 400 nm cells |k_Si| h ≈ 30
   and |k_Ag| h ≈ 22, no Dunavant degree reaches 1e-5 and the cap 19 (73 × 73 point pairs) is
   used for every pair with a coarse triangle, whatever its distance: 59 % (Si) and 48 % (Ag) of
   the CPU time. For Ag nearly all of these blocks are negligible (decay length 25 nm); for Si
   they are not (decay length 1.13 µm), and their accuracy at degree 19 is below the target
   anyway — the coarse cells do not resolve the Si wavelength (116 nm), which is a mesh-validity
   question (WP-V1), not a quadrature one.
3. **C-library kernel arithmetic**: ~30–40 ns per kernel evaluation, dominated by one `sincos`
   and one `exp` call per point pair (UCRT64), not vectorisable.

## Changes

| Change | Where | Accuracy |
|---|---|---|
| Dynamic schedule, rows of a colour handed out one by one in order of decreasing longest edge | `src/operator/assembler.cpp` (`order_by_cost`, `schedule(dynamic, 1)`) | bitwise identical matrix (each row has one writer per colour; summation order unchanged) |
| Decay-aware near/far target for lossy regions (WP7c part 4): relative target / δ, δ = (1 + αR_lb)e^{−αR_lb}, lowest degree when (S_d² + 1)δ ≤ target | `src/kernels/operators.cpp` (`select_degree`, `log_decay_factor`), `OperatorOptions::decay_aware_target` | ‖B_d − B‖ ≤ target ‖U‖, U the undamped magnitude bound of the block (ADR 0004); measured ≤ 4.0e-3 (Ag) and 7.1e-3 (Si) of the bound; lossless regions bitwise unchanged |
| Branch-free, vectorised sin/cos/exp in the near/far point loop; `-fno-math-errno -fno-trapping-math` for operators.cpp | `include/specklebem/kernels/fast_math.hpp`, `src/kernels/operators.cpp` (`plain_kernel_values`), `src/CMakeLists.txt`, `OperatorOptions::fast_plain_kernel` | ≤ 1 ulp against the C library; blocks within 2.7e-15 relative |

Mie validation after all changes (n = 3 icospheres, dense PMCHWT + LU): ε_rr xz / yz = 0.348 % /
0.256 % (Ag) and 0.554 % / 0.396 % (Si), the values of `mie_sphere_dense.md` to the printed
digits; all validation tests pass.

## After

Sampled profiles of the WP15 systems (stride 20, CPU s of the full matrix, same session):

| System | before | after | factor | largest remaining class |
|---|---:|---:|---:|---|
| Si box, 2N = 31 200 | 8 510 | 5 494 | ×1.55 | far, interior, coarse: 3 258 s (59 %), degree 19 |
| Ag box, 2N = 29 280 | 3 839 | 676 | ×5.7 | far, exterior, coarse: 281 s (42 %), degrees 12–17 |

Real assemblies (`DenseStrategy::build`, 24 threads, shared machine):

| System | WP15 (before, 24 threads) | after | factor |
|---|---:|---:|---:|
| Si box, 2N = 31 200 | 1 513–1 535 s | 242 s | ×6.3 |
| Ag box, 2N = 29 280 | 1 056–1 293 s | 35 s | ×30–37 |
| Ag icosphere n = 4, 2N = 15 360 | 11 s (mie_sphere_dense.md) | 5.6–5.9 s | ×1.9 |
| glass icosphere n = 4, 2N = 15 360 | 7.5 s (n = 1.5, mie_sphere_dense.md) | 5.9–6.1 s | ×1.2–1.3 |

Same-process comparison (`--compare 3`: the pre-WP-P2 pair loop with static schedule, strict
target and C-library kernel, then `DenseStrategy::build` with each change added; median of three
rounds, L = 1.2 µm boxes and n = 3 icospheres; single rounds vary by up to ×2 under load):

| Mesh | before | + dynamic schedule | + decay-aware target | + fast kernel (all) |
|---|---:|---:|---:|---:|
| Si box, 2N = 5 850 | 47.5 s | 28.9 s (×1.6) | 22.9 s | 15.0 s (×3.2) |
| Ag box, 2N = 5 130 | 62.6 s | 10.6 s (×5.9) | 6.95 s | 4.47 s (×14) |
| Ag icosphere n = 3 | 2.46 s | 2.51 s | 1.07 s | 1.05 s (×2.3) |
| glass icosphere n = 3 | 1.38 s | 1.16 s | 1.19 s | 0.93 s (×1.5) |

Single-thread micro benchmark (one far pair, fixed degree 19, 73 × 73 point pairs, loaded
machine): exterior (lossless) 146 → 108 µs, Si interior 212 → 116 µs; isolated loops over 5 329
arguments: sin + cos 18.9 → 6.5 ns, exp 6.5 → 4.5 ns.

## Against the N²-scaled sphere figure (backlog criterion)

The n = 4 icospheres now assemble in 5.6–6.1 s; scaled by (2N)² to the WP15 systems that is
21 s (Ag) and 24 s (Si), or 40–45 s from the older records (8–11 s) the backlog refers to.

- **Ag box: 35 s — within 2× of the scaled sphere figure** (1.7× the current sphere, 0.9× the
  older record).
- **Si box: 242 s — 10× the current, 5–6× the older sphere figure.** Reason: 1 067 of the
  10 400 triangles are 400 nm box cells with |k_Si| h ≈ 30; every pair with one of them uses the
  capped degree 19 in the object region (23.3 M of 107 M far calls, 66 % of the CPU time), and
  the Si field does not decay over the box, so the decay-aware target cannot relax them. Making
  those blocks cheaper without loosening the target would need a mesh that resolves the Si
  wavelength in the box (the ADR 0006 fine band, which WP15 had to leave out) or subdivided
  quadrature for |k| h ≫ 3 (more expensive, but accurate). This belongs to the box-validity
  work package WP-V1. A per-pair cost model of the Si box: fine–fine far pairs cost the same as
  on the sphere (1.9 µs exterior, 9 µs interior per call after WP-P2).

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_dense_assembly_profile
$B --mesh si --L 3.2e-6 --profile --stride 20 --no-decay --libm   # "before" kernel options
$B --mesh si --L 3.2e-6 --profile --stride 20                     # after
$B --mesh ag --L 1.2e-6 --profile                                 # full profile + schedule
$B --mesh ag --L 1.2e-6 --compare 3                               # same-process before/after
$B --mesh ag --L 3.2e-6 --build 1                                 # real assembly (14 GB)
$B --mesh sphere --sphere-n 3 --material si --micro               # single-pair micro benchmark
```
