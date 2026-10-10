# MLFMM scaling on Si and Ag rough surfaces (WP22b1, Phase 4 DoD)

Phase 4 acceptance criterion of `docs/01_project_plan.md`: "Scaling exponent γ (time per solve vs
N) ≤ 1.5 for Si rough surfaces, ≤ 2.0 for Ag, over N = 5·10⁴ … 10⁶". Executable
`benchmarks/mlfmm_scaling.cpp` (`specklebem_mlfmm_scaling`, `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`),
fits `benchmarks/mlfmm_scaling_fit.py`, raw `ROW` lines in
[`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt) (`python benchmarks/mlfmm_scaling_fit.py
benchmarks/results/mlfmm_scaling_rows.txt` reproduces the tables and fits; the `XROW` lines — extra
matvec runs, GMRES(100), automatic box — have the same format and are excluded from the fits).

**Verdict.**

- **Si: γ = 0.73 ≤ 1.5 — met over the measured range 2N = 4.3·10⁴ … 5.2·10⁵** (6 sizes, L = 1 …
  8 µm). Iterations do not grow (154 → 96); the solve time follows the matvec (γ = 1.1–1.3 for the
  far part, which dominates). **The upper half of the DoD range (2N > 5.2·10⁵) was not reachable:
  memory.** The largest case needs 79 GB; 2N ≈ 10⁶ is estimated at 240–310 GB, because the leaves
  are 2–3 × larger than λ/4 (see "Why the leaves are large") and the Si interior leaf patterns grow
  with (k₂ a)².
- **Ag: γ = 1.01 ≤ 2.0 — met over the solved range 2N = 4.3·10⁴ … 2.9·10⁵** (5 sizes, L = 3 …
  9 µm; 1.43 from 2N = 7·10⁴, 1.63 over the upper three sizes). Setup and matvec were measured up
  to **2N = 1.02·10⁶** (L = 17.5 µm, 77 GB): γ = 0.89 and 0.87. The solves at 2N ≥ 4·10⁵ did not
  fit the 2-hour budget per run: GMRES needs 1 000–1 700 iterations, and the full-GMRES modified
  Gram–Schmidt (serial in `solver::gmres`) takes 42–66 % of the solve time and grows as k² N.
  Extrapolated to 10⁶ (iterations γ = 0.24, MGS ∝ k² N): ~5.3 h per solve, γ ≈ 1.15 (from
  4.3·10⁴) … 1.44 (from 7·10⁴) over the DoD range (projection, not a measurement).
- **Automatic box rule (`--box-mesh-size auto`)**: unusable with the MLFMM beyond L ≈ 1.5 µm. Its
  200–400 nm cells have support radii of 230–490 nm, the leaf rule (r_max / a ≤ 0.6) raises the
  leaves to λ/0.7 … λ/0.35 and the near field approaches the dense matrix (Si L = 4 µm: 307 GB near
  field estimate for 2N = 1.8·10⁵). Ag L = 4 µm: 36 % fewer unknowns than the 100 nm box but
  10 × the near field, 9 × the peak memory and 3.8 × the solve time (1 208 vs 1 386 iterations).

## Setup

| Parameter | Value |
|---|---|
| Surface | Gaussian height map (`generate_gaussian_height_map`), σ = 50 nm, Lc = 500 nm, mesh h = 50 nm, seed 1, L × L patch (no crop; every L its own surface) |
| Closing box (ADR 0006) | depth `default_box_depth` (Si 5.65 µm, Ag 2 µm); coarse spacing `box_mesh_size` = **100 nm = λ₁/5** (WP-V1 recommendation 1, M = 1); **Si fine band** `default_box_fine_depth` = 3δ + 3σ = 3.54 µm (WP-V1 recommendation 2); Ag no fine band. Second series `auto` = `min(depth/2, L/8, 10 h)` |
| Media | λ = 500 nm, vacuum R1; Si ε_r = 18.478 − 0.606j, Ag ε_r = −9.794 − 0.313j (exp(+jωt)) |
| Excitation | paraxial `excitation::GaussianBeam`, normal incidence (+z), focus at the origin, E along x, w₀ = L/4 |
| Operator | `Simulation` with compression `"mlfmm"`, d₀ = 3, automatic leaf rule (leaf ≥ max(λ/4, r_max/0.6)) and exact-part budget (2 × near) |
| Solver | `formulation::recommend`: Si ICTF without preconditioner, Ag ICTF + left Jacobi; full GMRES (no restart), x₀ = 0, tolerance 1e-3 on the monitored residual (true residual reported; Ag ≤ 1.2e-3) |
| Timings | setup = `Simulation::assemble` (octree, far setup incl. the Ag exact part, near field, rhs, Jacobi); matvec = median of 5 applies of the `MlfmmOperator` (and of its near and far parts), interleaved after one untimed round; solve = GMRES wall time; s/it = solve / iterations |
| Memory | near field, far operator (leaf patterns, translators, workspace; exact parts separately), Krylov basis (it + 1) 2N 16 B, peak working set after the setup and at the end |
| Machine | Windows 11, 24 cores, 128 GB, MSYS2 UCRT64 GCC 16.1, OpenBLAS, `win-release`, 24 OpenMP threads, 2026-10-10. Shared machine: other agents ran test suites and the WP-V1 dense study concurrently, so the timings carry ±30 % noise (single matvec medians up to 2 ×, e.g. Si L = 7 µm: full apply 21 s vs near + far 8.1 s and 9.8 s per GMRES iteration). Runs marked † overlapped with builds or test runs |

One case per process, sizes in increasing order, each started only when its estimated peak
(`--estimate-only`: 2 near + 1.1 leaf patterns + 2 exact + 1 GB, calibrated on the first runs,
within 5 % of the measured peaks) plus 10 GB fitted the available memory.

## Si (100 nm box, fine band)

| L [µm] | 2N | levels | leaf [nm] | setup [s] | near [s] | far [s] | matvec [s] | near mv [s] | far mv [s] | it | solve [s] | s/it | near [GB] | far [GB] | peak [GB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 960 | 5 | 357 | 34.4 | 22.9 | 11.5 | 0.69 | 0.13 | 0.55 | 154 | 180 | 1.17 | 4.11 | 5.97 | 14.0 |
| 2 | 92 880 | 5 | 360 | 79.4 | 51.4 | 28.0 | 1.64 | 0.27 | 1.91 | 106 | 275 | 2.59 | 7.19 | 13.19 | 27.0 |
| 3 | 148 320 | 5 | 361 | 97.9 | 66.5 | 31.4 | 3.05 | 0.35 | 2.60 | 103 | 325 | 3.15 | 10.54 | 21.38 | 41.2 |
| 4 | 209 760 | 5 | 362 | 264.3 | 163.9 | 100.4 | 5.22 | 0.53 | 4.62 | 104 | 589 | 5.66 | 14.28 | 30.01 | 56.9 |
| 7 | 433 440 | 6 | 219 | 184.5 | 130.2 | 54.3 | 21.0 ‡ | 0.32 | 7.82 | 90 | 882 | 9.80 | 10.60 | 41.88 | 56.6 |
| 8 | 515 520 | 6 | 250 | 204.1 | 114.0 | 90.0 | 10.65 | 0.47 | 10.17 | 96 | 1 044 | 10.88 | 16.16 | 53.60 | 79.1 |

‡ sequential medians (the interleaved timing came later): outlier, near + far = 8.1 s. The first
Si L = 1 µm run (calibration, heavily loaded machine) took 111 s for the same 154 iterations;
the table shows the rerun (180 s).

Exponents (least squares in log-log, all 6 sizes, 2N = 42 960 … 515 520):

| quantity | γ (6 sizes) | γ (L = 1 … 4 µm, fixed leaf ≈ 360 nm) |
|---|---:|---:|
| **time per solve** | **0.73** | 0.68 |
| iterations | −0.18 | −0.25 |
| solve time per iteration | 0.90 | 0.93 |
| matvec (full / near / far) | 1.28 / 0.43 / 1.12 | 1.26 / 0.88 / 1.29 |
| setup (total / near / far) | 0.71 / 0.68 / 0.76 | 1.17 / 1.14 / 1.22 |
| memory (near / far / peak) | 0.47 / 0.85 / 0.64 | 0.78 / 1.02 / 0.88 |

- **Iterations are flat** (ICTF, 90–154; the smallest patch is the slowest), so the solve time is
  the matvec time: time per solve ≈ it × matvec (the GMRES orthogonalisation is ≤ 10 % at ≈ 100
  iterations; Si L = 2 µm is a noisy outlier).
- **The far part dominates the matvec** (85–95 %): the Si interior expansion at leaves of 0.4–0.7 λ₁
  = 3–6 λ₂ needs leaf sampling orders L = 45–60 (2(L+1)² ≈ 4 000–7 400 directions per basis
  function), applied to every basis function in every matvec and stored as leaf patterns (far
  memory 6 → 54 GB, 60–70 % of the peak).
- **Sub-linear setup and memory are an artefact of the leaf size**: the root edge is set by
  max(L, depth) and halved, so the leaf jumps from 357–375 nm (5 levels, L ≤ 6 µm) to 219–250 nm
  (6 levels, L = 7–8 µm), which cuts the near field per unknown 2.6 × and the pattern size per
  unknown ~2 ×. At fixed leaf size (L = 1 … 4 µm) memory grows with γ ≈ 1.0 and setup with γ ≈
  1.2.

## Ag (100 nm box)

| L [µm] | 2N | levels | leaf [nm] | setup [s] | near [s] | far [s] | matvec [s] | near mv [s] | far mv [s] | it | solve [s] | s/it | near [GB] | far [GB] | peak [GB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 43 200 | 4 | 375 | 13.7 | 7.9 | 0.4 | 0.18 | 0.07 | 0.10 | 1 004 | 511 | 0.51 | 2.39 | 0.66 | 5.4 |
| 4 | 69 600 | 5 | 250 | 12.9 | 9.1 | 0.7 | 0.17 | 0.05 | 0.09 | 1 386 | 405 | 0.29 | 1.68 | 0.87 | 4.2 |
| 5 | 103 200 | 5 | 313 | 20.7 | 15.6 | 1.0 | 0.27 | 0.12 | 0.16 | 1 284 | 591 | 0.46 | 4.04 | 1.48 | 9.5 |
| 7 | 186 480 | 6 | 219 | 32.3 | 22.7 | 2.2 | 0.40 | 0.11 | 0.29 | 1 518 | 1 203 | 0.79 | 3.62 | 2.39 | 10.9 |
| 9 | 293 760 | 6 | 281 | 52.9 | 39.3 | 2.3 | 0.67 | 0.27 | 0.38 | 1 697 | 3 326 | 1.96 | 9.55 | 3.86 | 22.7 |
| 11 | 422 400 | 6 | 344 | 87.0 | 67.4 | 2.9 | 1.17 | 0.60 | 0.56 | 0 | — | — | 20.85 | 6.54 | 47.8 |
| 15 | 756 000 | 7 | 234 | 127.8 | 90.8 | 7.2 | 1.58 | 0.51 | 1.03 | 0 | — | — | 17.43 | 9.53 | 43.8 |
| 17.5 | 1 017 450 | 7 | 273 | 184.7 | 136.3 | 8.3 | 2.32 | 0.92 | 1.39 | 0 | — | — | 32.16 | 13.63 | 76.8 |

Exact part (included in "far [s]" and the peak): ≤ 0.83 GB and ≤ 3.3 s at every size. True
residuals 0.89e-3 … 1.17e-3 (left Jacobi: the monitored residual is the preconditioned one).

Rows with it = 0: `--no-solve` (setup and matvec only; a full-GMRES solve would exceed the 2-hour
budget per run). Exponents:

| quantity | γ | range |
|---|---:|---|
| **time per solve** | **1.01** (5 sizes); 1.43 (4 sizes from 7.0·10⁴); 1.63 (upper 3 sizes) | 2N = 4.3·10⁴ … 2.9·10⁵ |
| iterations | 0.24 | 4.3·10⁴ … 2.9·10⁵ |
| it × matvec / orthogonalisation part of the solve | 0.96 / 1.06 (L = 3 µm orthogonalisation 64 %, noisy; 1.8 from 7.0·10⁴) | 4.3·10⁴ … 2.9·10⁵ |
| solve time per iteration | 0.77 | 4.3·10⁴ … 2.9·10⁵ |
| matvec (full / near / far) | 0.87 / 0.90 / 0.89 | 4.3·10⁴ … 1.02·10⁶ (8 sizes) |
| setup (total / near / far) | 0.89 / 0.94 / 0.96 | 4.3·10⁴ … 1.02·10⁶ |
| memory (near / far / peak) | 0.91 / 0.98 / 0.92 | 4.3·10⁴ … 1.02·10⁶ |

Projection to 2N = 1.02·10⁶ (iterations 1 697 (2N/2.9·10⁵)^0.24 ≈ 2 290, matvec 2.32 s measured,
orthogonalisation at the 9.3 GB/s effective bandwidth of the L = 9 µm run): 5 300 s of matvecs +
13 700 s of orthogonalisation ≈ 5.3 h, i.e. γ ≈ 1.15 (from 4.3·10⁴) … 1.44 (from 7.0·10⁴) over
the DoD range. The local slope rises because the orthogonalisation grows as k² N (asymptotically
γ → 1 + 2 · 0.24 ≈ 1.5).

- **Solve time = iterations × matvec + orthogonalisation.** Measured split (orthogonalisation =
  solve − it × matvec): L = 4 µm 169 s of 405 s (42 %), 5 µm 248 / 591 s (42 %), 7 µm 594 / 1 203 s
  (49 %), 9 µm 2 185 / 3 326 s (66 %). The it × matvec part grows with γ ≈ 1.1 (matvec 0.95,
  iterations 0.1–0.2), the modified Gram–Schmidt part (k²/2 · 2N complex dot products and axpys,
  serial Eigen loops in `solver::gmres`, ~10–20 GB/s) with γ ≈ 1.8 = 2 γ_it + 1. Over the solved
  range the measured γ stays below 2.0; at 10⁶ the orthogonalisation would be ~70 % of a ~5 h solve.
- **Iterations** (ICTF + Jacobi, tol 1e-3): 1 004, 1 386, 1 284, 1 518, 1 697 for L = 3, 4, 5, 7, 9 µm (γ = 0.24) — slowly growing and not monotone
  (each L is a different surface). GMRES(100) was tried at L = 3 µm: 2 844 iterations in 601 s,
  the same time as full GMRES (1 004 iterations, 619 s, both on the loaded machine), 0.07 GB
  instead of 0.69 GB Krylov memory — no gain at this size, and restarts slow Ag down 2.8 ×.
- **The Ag exact part is negligible** (≤ 0.8 GB, ≤ 3 s): with leaves of 219–406 nm the near field
  already covers almost all pairs within the decay reach x*/α = 0.30 µm (Ag WP22a sphere with λ/4
  leaves: 6.8 GB). The interior region R2 is truncated or exact on every far level; R1 expands.
- **Near and far share the matvec about equally**; the near field (1.7–32 GB) is the largest memory
  item; the leaf rule raises the leaf to 219–406 nm here as well, so the near field per unknown
  varies 2.6 × with the root quantisation (L = 4 µm: 24 kB, L = 11 µm: 49 kB).

## Automatic box rule

| Case | coarse spacing (M) | r_max [nm] | leaf [nm] | 2N | near [GB] | far [GB] | peak [GB] | it | solve [s] | 100 nm box at the same L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Si L = 1 µm | L/8 = 125 nm (M = 1) | 116 | 357 | 42 960 | 4.1 | 6.0 | 14.0 | — | — | identical mesh (M = 1 either way) |
| Si L = 1.5 µm † | L/8 = 188 nm (M = 2) | 228 | 714 | 61 104 | 22.5 | 26.3 | 71.0 | 210 | 736 | 67 410 unknowns, 5.2 GB near, 15.4 GB peak estimate |
| Si L = 2 µm | 250 nm (M = 2) | 231 | 720 | 83 760 | 34.9 (est.) | 35.9 (est.) | 110 (est.) | not run | | memory: estimate + 10 GB > available |
| Si L = 4 µm | 400 nm (M = 3) | 447 | 1 450 | 182 880 | 307 (est.) | | 768 (est.) | not run | | 14.3 GB near |
| Ag L = 2 µm † | 250 nm (M = 2) | 246 | 528 | 14 640 | 1.45 | 0.41 | 3.3 | 2 495 | 662 | |
| Ag L = 4 µm | 400 nm (M = 3) | 485 | 1 000 | 44 400 | 17.2 | 3.3 | 37.8 | 1 208 | 1 547 | 69 600 unknowns, 1.7 GB near, 4.2 GB peak, 1 386 it, 405 s |

The automatic rule makes the box cheap in unknowns (Ag L = 4 µm: 44 400 vs 69 600) but not in
cost: the single largest RWG support decides the leaf edge for the whole octree. (Ag L = 2 µm has
no 100 nm counterpart; its 2 495 iterations are not comparable. The Ag L = 4 µm auto run has a
true residual of 1.8e-3 at the monitored 1e-3.)

## Why the leaves are large

The ADR 0008 leaf rule uses the **global** largest support radius r_max: leaf ≥ r_max / 0.6. On the
top face (h = 50 nm) r_max ≈ 60 nm and λ/4 = 125 nm leaves would be allowed; the 100 nm box cells
(2:1 transition cells, relaxation band) have r_max = 116–141 nm, so the floor becomes 195–235 nm,
and the power-of-two subdivision of the root (edge ≈ max(L, depth)) rounds it up to 219–406 nm.
Consequences per unknown compared with λ/4 leaves: near field 1.8–3.3 × (surface boxes), Si
interior leaf patterns 3–10 × ((k₂ a)²), both linear in N. This, not the multilevel algorithm,
limits the Si range to 2N ≈ 5·10⁵ on 128 GB.

## What would extend the range (proposals, not applied)

1. **Local leaf rule** (largest lever, both materials): apply r_max / a ≤ 0.6 per box, e.g. by
   letting the few coarse box cells live on a coarser level (non-uniform depth, ADR 0008 change),
   or by splitting coarse box triangles so that box supports stay ≤ the top-face supports (r_max
   ≈ 75 nm, box_mesh_size ≈ 60 nm, ~+40 % unknowns at L = 10 µm). With λ/4 leaves the Si 10⁶ case
   drops from ~240–310 GB to an estimated 40–60 GB.
2. **Si leaf patterns**: compute the leaf radiation patterns on the fly per matvec or store them in
   single precision (accuracy to be checked against d₀ = 3), halving or removing the dominant
   memory item.
3. **GMRES orthogonalisation** (Ag): classical Gram–Schmidt with reorthogonalisation (CGS2) as
   BLAS-2 products over the Krylov basis (threaded zgemv), instead of the serial modified
   Gram–Schmidt loop, would cut the 42–66 % orthogonalisation share several-fold. A better
   preconditioner than Jacobi (near-field ILU / SAI; docs/01 risk table: H-LU, Phase 8) would cut
   the 1 000–1 700 Ag iterations, which drive both the time and the Krylov memory (8 GB at 2.9·10⁵,
   ~35 GB projected at 10⁶).
4. **Box rule**: keep `box_mesh_size` ≤ λ₁/5 explicit (WP-V1); the automatic rule must not be used
   with the MLFMM until (1) exists.

## Pending

- Si 2N = 5.2·10⁵ … 10⁶: not runnable on this machine (estimated peak 116 GB at 2N = 6.1·10⁵,
  240–310 GB at 8.6·10⁵ … 9.8·10⁵). Plan: rerun after proposal 1 (`--L 12e-6 … 13e-6`).
- Ag solves at 2N = 4.2·10⁵, 7.6·10⁵, 1.02·10⁶ (L = 11, 15, 17.5 µm): setup and matvec measured
  (table), solve projected at ~1.7 h, ~3.5 h and ~5 h with the current GMRES — beyond the 2-hour
  budget per run of this session. Plan: one overnight run each
  (`--material ag --L 11e-6 --max-iter 6000`, ...), or after proposal 3.

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_mlfmm_scaling
$B --material si --L 8e-6 --estimate-only           # 2N, octree, memory estimates (seconds)
$B --material si --L 4e-6 > si_L4.log               # Si sweep: L = 1 2 3 4 7 8 um
$B --material ag --L 9e-6 --max-iter 6000 > ag_L9.log   # Ag sweep: L = 3 4 5 7 9 um
$B --material ag --L 17.5e-6 --no-solve > ag_L17p5.log  # setup + matvec only: L = 11 15 17.5 um
$B --material ag --L 4e-6 --box-mesh-size auto       # automatic box rule
python benchmarks/mlfmm_scaling_fit.py si_L*.log     # tables and exponents
```
