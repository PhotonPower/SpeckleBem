# Dense LU: LAPACK and Eigen backends (WP-LU1)

Purpose: one reusable dense LU (`solver::DenseLu`) for the reference solutions of small and
medium problems (sphere validations, the PML route of ADR 0009 with 2N ≈ 10⁴), factorised once
and solved for several right-hand sides (s and p, several incidences). This record gives the
timings of the two backends, the reason the rcond estimate is not LAPACK `zgecon`, and the
reason for the default backend. Benchmark: `benchmarks/dense_lu.cpp`
(`specklebem_dense_lu`); unit tests: `tests/unit/test_direct.cpp` (`[solver]`).

## API summary

| Item | Behaviour |
|------|-----------|
| `DenseLu(Z, backend = Auto)` | factorises a square complex `MatrixXc` or `op::DenseOperator`; `Z` is referenced (refinement), not copied, and must outlive the object (temporaries are rejected at compile time) |
| Backends | `LuBackend::Lapack`: LAPACK `zgetrf`/`zgetrs` (32-bit LAPACK integers, multithreaded through the BLAS); `LuBackend::Eigen`: `Eigen::PartialPivLU` (every build) |
| Auto rule | `Lapack` when `lapack_available()` (CMake found BLAS/LAPACK and a probe linking exactly `zgetrf_`/`zgetrs_` succeeded), else `Eigen`; an explicit `Lapack` request in a build without it throws (no silent fallback). The backend used is in `DenseLu::backend()`, `DirectSolveInfo::backend`, `Simulation::report` and logged once per process |
| Equilibration | columns scaled by powers of two to unit max-norm while copying into the LU storage (`Z D`, exact); the singularity test is invariant to column scaling (e.g. the η² ratio of the PMCHWT J and M blocks); `x = D (Z D)⁻¹ b` |
| rcond | `1 / (‖Z D‖₁ ‖(Z D)⁻¹‖₁)` with the Hager/Higham estimate of the inverse norm (the algorithm of LAPACK `zlacn2`, at most 5 iterations plus the alternating-sign vector) driven by the backend's own LU solves: 4–11 solves (1 for n = 1) with one right-hand side. It is a lower bound of the inverse norm, so the reported rcond is an upper bound of the exact one (within a factor 3 in the tests) |
| Refinement | every solve runs at most two fixed-precision refinement steps against the original `Z`, each kept per right-hand side only if it reduces ‖b − Z x‖; `DirectSolveInfo::residual` is the final relative residual (the largest over the columns for a block solve) |
| Solves | `solve(VectorXc)` and `solve(MatrixXc)` (one `zgetrs` call / blocked triangular solves for all columns); the factorisation is reused for any number of calls |
| Memory | one n × n complex copy (16 n² bytes: 2.09 GB at n = 11 440) besides `Z`, n pivots, O(n) work vectors, O(n m) for m right-hand sides |
| Errors | `std::invalid_argument`: non-square or non-finite `Z`, non-finite or wrongly sized right-hand side, `Lapack` requested but unavailable, n > INT_MAX on the LAPACK path (`detail::checked_lapack_int`); `std::runtime_error`: singular (exactly zero or non-finite pivot, or equilibrated rcond < machine epsilon); `std::logic_error`: LAPACK `info < 0` (a bug); rcond < 10⁻¹⁰ logs one warning per factorisation and sets `ill_conditioned()` |
| `solve_direct(Z, b, info)` | thin wrapper `DenseLu(Z).solve(b, info)` with the solve time logged (signature unchanged) |

## Setup and load

Hardware: Windows 11, 12-core / 24-thread desktop CPU, 128 GB; win-release preset (MSYS2 UCRT64
GCC 16.1, `-O3`), OpenBLAS 0.3.33 built with OpenMP (libgomp, the same thread pool as the
library: `OMP_NUM_THREADS` sets both), Eigen 3.4.0 with `EIGEN_USE_BLAS` (Eigen's GEMM and TRSM
calls go to OpenBLAS). 24 threads unless stated otherwise.

**Load.** Measured 2026-10-15. Other workers ran a validation-large test, debug ctest suites and
a benchmark on the same machine throughout; before and between the runs the total CPU load
without this benchmark was 50–100 %. No idle-machine measurement was possible, so every time
below is the **minimum over 2–4 runs** at the load of the moment and is an upper bound for an
idle machine; the spread is given where it matters. Backends and matrices were alternated
within one process wherever a comparison is drawn (last two tables). The raw outputs (benchmark
logs, outputs of the uncommitted investigation programs, CPU-load logs) are not in the
repository; a number that appears in no table names the raw log it was taken from.

## Random matrices (complex normal entries, fixed seeds, 8 right-hand sides)

`factor` = `DenseLu` constructor (equilibrated copy, LU, rcond); `bare LU` = `zgetrf` or Eigen
`PartialPivLU` on a plain copy (`--raw 1`); `solve 1` / `solve 8` = `DenseLu::solve` with one /
eight right-hand sides including refinement (three LU solves and three products with `Z`).

| n | backend | factor [s] | bare LU [s] | solve 1 [s] | solve 8 [s] | rcond | residual (1 / 8 RHS) | ‖x_L − x_E‖/‖x_L‖ |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 2 000 | LAPACK | 1.68 | 0.78 | 0.032 | 1.26 | 2.944·10⁻⁶ | 4.5·10⁻¹⁴ / 7.0·10⁻¹⁴ | — |
| 2 000 | Eigen | 7.07 | 5.83 | 0.047 | 1.75 | 2.944·10⁻⁶ | 4.4·10⁻¹⁴ / 7.3·10⁻¹⁴ | 3.4·10⁻¹⁴ |
| 5 000 | LAPACK | 1.77 | 1.30 | 0.095 | 1.62 | 4.418·10⁻⁷ | 8.9·10⁻¹⁴ / 1.6·10⁻¹³ | — |
| 5 000 | Eigen | 7.87 | 7.14 | 0.117 | 1.59 | 4.418·10⁻⁷ | 8.8·10⁻¹⁴ / 1.5·10⁻¹³ | 6.6·10⁻¹⁴ |
| 11 440 | LAPACK | 19.4 | 15.3 | 0.95 | 4.38 | 3.883·10⁻⁷ | 6.3·10⁻¹⁴ / 1.0·10⁻¹³ | — |
| 11 440 | Eigen | 60.7 | 18.9 | 0.62 | 2.93 | 3.883·10⁻⁷ | 6.3·10⁻¹⁴ / 1.0·10⁻¹³ | 1.3·10⁻¹³ |

Each column is its own minimum, so `factor` and `bare LU` of a row can come from different runs
(at n = 11 440 Eigen's best factorisation never came close to its best bare LU). Spread at
n = 11 440 in the benchmark (6 runs per backend): LAPACK factor 19–75 s (bare `zgetrf` 15–22 s),
Eigen factor 61–212 s (bare LU 19–89 s). The rcond estimates of the two backends agree to all
printed digits (the same estimator on the same pivot sequence); the residuals and the backend
difference are at rounding level.

## Real matrix: flat Si, PML route, 2N = 11 440

The Si 0° case of `benchmarks/results/pml_validation.md` (ICTF, dense, 3 872 triangles), dumped
from the Simulation's dense operator and read with `--matrix`. Its J–M off-diagonal blocks are
99.9 % exact zeros (50 % of `Z`); partial pivoting makes 915 row interchanges (random: 11 432).

| backend | factor [s] | bare LU [s] | solve 1 [s] | solve 8 [s] | rcond | residual (1 / 8 RHS) | ‖x_L − x_E‖/‖x_L‖ |
|---|---:|---:|---:|---:|---:|---|---:|
| LAPACK | 16.9 | 13.4 | 0.57 | 5.78 | 1.203·10⁻³ | 5.3·10⁻¹⁶ / 4.6·10⁻¹⁶ | — |
| Eigen | 35.7 | 29.7 | 0.78 | 3.64 | 1.203·10⁻³ | 5.4·10⁻¹⁶ / 4.6·10⁻¹⁶ | 5.8·10⁻¹⁶ |

Spread: LAPACK factor 17–92 s, Eigen 36–180 s (4 runs each). For comparison, before the change of
the rcond estimator (2026-10-14, `zgecon`): `DenseLu` factor 149 s (LAPACK) and 122 s (Eigen)
(`exp2.txt`). In the full pipeline (`PML 7a: flat Si, 0 deg`, validation-large, through `DenseLu`,
passed at −0.022 %) the LU took 49.5 s on 2026-10-15 under load (`lu1_pml7a_si0.log`) and 118.4 s
on 2026-10-14 with an earlier state of `DenseLu` that still used `zgecon` (the Si 0° run of
`benchmarks/pml_fresnel.cpp`, `pml_si0_lapack.txt`).

`factor − bare LU` is 3–4 s on the LAPACK path at n = 11 440 (random and PML alike), i.e.
20–25 % of `zgetrf`: the three single-threaded passes over `Z` (column scales, scaled copy,
1-norm) and the estimator (1.5 s, 5 solves here, below). The passes were timed separately only on
2026-10-14 (3.9–4.3 s together in two runs, `exp2.txt`), the more heavily loaded day; their sum
with the estimator (5.4–5.8 s) therefore exceeds the gap measured on 2026-10-15, where the passes
took about 1.4–2.5 s by difference (not timed).

## Why the rcond estimate is not `zgecon`

Same matrix, LAPACK factors, same day (separate program, not committed; `lu1_inv_a.txt`,
`lu1_hager_pml.txt`):

| Step | 24 threads | 1 thread |
|------|-----------:|---------:|
| `zgetrf` | 16.1 s | — |
| `zgecon` (rcond 1.2031·10⁻³) | 5.81 s | 3.00 s |
| `zgetrs`, 1 RHS, `N` / `C` | 0.42 / 0.50 s | 0.59 / 0.84 s |
| Hager/Higham on `zgetrs` as in `DenseLu` (5 solves, rcond 1.2031·10⁻³) | 1.51 s | — |

On 2026-10-14, at a higher load, `zgecon` took 63 s against 36–42 s for `zgetrf` (`exp2.txt`).
`zgecon` runs the same `zlacn2` iteration but does its triangular solves with `zlatrs`,
reference-LAPACK level-2 code with overflow scaling that calls BLAS-1/2 routines; that it was
*slower* on 24 threads than on 1 thread (one measurement) suggests that its many short threaded BLAS
calls cost more in thread synchronisation than they gain, which would also make it the step most
exposed to a loaded machine (not profiled further). Driving the same estimator with the backend's
optimised solves costs 1.5 s here, gives the same rcond, works identically for both backends (the
tests compare them to 10⁻⁸) and removes `zgecon_` from the link probe.

## LAPACK vs Eigen: what the timing difference is and is not

Bare LU at n = 5 000 against the thread count (random matrix, one series, `OMP_NUM_THREADS`):

| threads | `zgetrf` [s] | Eigen [s] | Eigen / `zgetrf` |
|---:|---:|---:|---:|
| 1 | 9.52 | 9.98 | 1.05 |
| 12 | 1.42 | 2.32 | 1.6 |
| 24 | 1.51 | 7.14 | 4.7 |

Content test, 24 threads, all four 11 440 × 11 440 matrices alternated in one process (first
pass; a second pass hit a load spike, 15–84 s / 70–238 s):

| matrix | `zgetrf` [s] | Eigen [s] |
|---|---:|---:|
| random | 15.1 | 50.2 |
| random with the PML matrix's zero pattern | 14.5 | 49.8 |
| PML matrix | 14.0 | 45.8 |
| PML matrix, zeros replaced by 10⁻³ · random | 14.6 | 46.4 |

The same with one thread on the leading 4 000 × 4 000 block (no zeros there; random vs PML):
`zgetrf` 4.0–5.3 s, Eigen 4.3–6.2 s for every matrix.

What the measurements support:

1. **Same arithmetic kernel.** With `EIGEN_USE_BLAS` Eigen's blocked LU sends its trailing updates
   to OpenBLAS `zgemm`/`ztrsm`; single-threaded both backends take the same time (ratio 1.05 at
   n = 5 000).
2. **The difference is parallel scheduling, and it grows with the load.** OpenBLAS replaces
   LAPACK's `zgetrf` by its own natively threaded LU; Eigen 3.4 runs n / 256 sequential block
   steps (block size 256 for n ≥ 2 048), each a panel factorisation in Eigen's own code followed
   by fork–join BLAS calls (`ztrsm`, `zgemm`) for the trailing matrix. From 12 to 24
   threads (two hardware threads per core, other jobs running) `zgetrf` stays flat
   (1.42 → 1.51 s at n = 5 000) while Eigen becomes 3× slower (2.32 → 7.14 s). The mechanism is
   inferred, not profiled: every fork–join step waits for its slowest thread, and with the
   machine oversubscribed some thread is always descheduled. It is consistent with the spreads
   above: over all bare factorisations of 11 440 × 11 440 matrices on this day (benchmark and
   investigation runs, random and PML), `zgetrf` took 13–88 s (median 16 s, 23 runs) and Eigen
   19–238 s (median 70 s, 21 runs) (the bare-LU times in `lu1_random_a.txt`, `lu1_random_b.txt`,
   `lu1_random11440_c.txt`, `lu1_matrix_a.txt`, `lu1_matrix_b.txt`, `lu1_cmp24.txt`,
   `lu1_inv_a.txt` and, `zgetrf` only, `lu1_hager_pml.txt`).
3. **Matrix content does not matter.** The PML matrix, a random matrix and their cross variants
   factor in the same time on either backend; the LU factors contain no subnormal numbers and
   flush-to-zero/denormals-are-zero changed nothing. The earlier observation of this WP —
   "`zgetrf` slower than Eigen on random matrices (29.6 vs 22.2 s) but ≈ 2× faster on the PML
   matrix (36–42 vs 82 s)" — compared runs made at different times on the shared machine, and
   the random-matrix LAPACK time then still included `zgecon`; it was load, not the matrix.

What remains unexplained (not profiled): the absolute size of Eigen's slowdown under load (3–10×)
per phase (panel vs update); why the 8-RHS solve costs 4–10 single solves instead of ≈ 1–2 (the
tall-skinny `zgetrs`/`zgemm` calls are far from the memory bound of reading the factors and `Z`
three times); and the idle-machine ratio between the backends at 24 threads, which could not be
measured.

## Default

**`Auto` = LAPACK when available.** In every comparison made at the same time `zgetrf` was as fast
as Eigen (one thread) or 1.6–4.7× faster (12–24 threads), it degrades far less when other jobs
share the machine, and both backends give the same solution (difference ≤ 1.3·10⁻¹³) and the same
rcond. Eigen remains the backend of builds without LAPACK (CI runners) and can be requested
explicitly (`LuBackend::Eigen`). With the Eigen backend on a shared machine,
`OMP_NUM_THREADS` = number of physical cores was the better setting here (n = 5 000: 2.3 s at 12
threads vs 7.1 s at 24; one series).

**The expected 10–40× was not reproduced.** The work package expected LAPACK to be 10–40× faster
at 2N = 11 440, from WP-PML3b (351 s Eigen vs 8–65 s LAPACK). Measured at the same time, the
`DenseLu` factorisation is 3.1× (random) and 2.1× (PML) faster with LAPACK (tables above; bare LU
2.2–3.4× in the same-process runs without a load spike). The 351 s Eigen figure was taken during
a concurrent debug build and ctest run on the same machine (source: the worker's project memory
of the WP-PML3a smoke run that produced it and that WP-PML3b quoted; its raw log was not kept).
With `EIGEN_USE_BLAS` Eigen already runs its trailing updates in the OpenBLAS kernels, so no
kernel-level gap is left to gain: the remaining difference is the parallel schedule, 2–3× at
2N = 11 440 under load and none on one thread.

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
export PATH=/c/msys64/ucrt64/bin:$PATH                     # MSYS2 runtime DLLs first (Windows)
B=build/win-release/benchmarks/specklebem_dense_lu
$B --n 2000,5000,11440 --nrhs 8 --repeat 2 --raw 1         # random matrices (4.2 GB at n = 11440, 6.3 GB with --raw)
$B --matrix Z.bin --nrhs 8 --repeat 2 --raw 1              # a dumped operator
for t in 1 12 24; do OMP_NUM_THREADS=$t $B --n 5000 --repeat 2 --raw 1; done
```

`--matrix` reads an int64 n followed by the n × n `complex<double>` entries in column-major order
(16 n² bytes, native little-endian), i.e. `Z.rows()` and `Z.data()` of the `MatrixXc` of a
`DenseOperator`; the matrix of this record is the Si 0° case of `benchmarks/pml_fresnel.cpp`
(2.09 GB, not in the repository). Run on an otherwise idle machine where possible and alternate
the backends; on a shared machine report the minimum over repetitions together with the load.
