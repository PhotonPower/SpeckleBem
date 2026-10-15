---
name: dense-lu-findings
description: WP-LU1 DenseLu timing facts - zgetrf vs Eigen PartialPivLU on the Windows OpenBLAS build, load sensitivity, zgecon cost, PML matrix structure, how to measure
metadata:
  type: project
---

WP-LU1 (2026-10-14/15), record benchmarks/results/dense_lu.md, benchmark benchmarks/dense_lu.cpp
(`--matrix FILE.bin` = int64 n + column-major complex<double>; `--raw 1` times bare zgetrf / Eigen).

- Build facts: win-release links MSYS2 OpenBLAS 0.3.33 built with OpenMP (libgomp, same pool as
  our code: OMP_NUM_THREADS controls both) and Eigen 3.4.0 (fetched) with EIGEN_USE_BLAS, so
  Eigen's PartialPivLU already runs its trailing updates in OpenBLAS zgemm/ztrsm.
- Single thread (n = 5000): zgetrf 20.5 s = Eigen 20.8 s -> same kernel; every backend
  difference is in the parallel schedule (Eigen: sequential panel + fork/join zgemm per block;
  OpenBLAS getrf_parallel: its own threaded recursive LU).
- Content does NOT matter: on the Si PML matrix (2N = 11440) vs random, alternated in one
  process, zgetrf 14-15 s and Eigen 46-50 s for all four variants (random, random with the PML
  zero pattern, PML, PML zeros filled). FTZ/DAZ no effect; no subnormals in the LU factors.
  Earlier "Eigen faster on random, 2x slower on PML" was load at different times.
- The PML matrix: J/M off-diagonal blocks 99.9 % exact zeros (50 % of Z), 915 row
  interchanges of 11440 (random: 11432).
- Under load from other OpenMP jobs (machine at ~100 %), Eigen's LU degrades 3-10x, zgetrf
  much less (n = 11440: zgetrf 14-31 s, Eigen 19-240 s). zgecon: 63 s one day, 5.8 s (24 thr) /
  3.0 s (1 thr) the next, vs zgetrs 1 RHS 0.4-0.5 s; DenseLu's Hager loop took 5 solves, 1.5 s.
- Measuring: compare backends/matrices in ONE process, alternating, and repeat; log the total
  CPU load (German Windows counter "\Prozessor(_Total)\Prozessorzeit (%)" for typeperf).
- Review (WP-LU1 fix round): every number in the record must trace to a raw log; the record
  names the scratchpad log basenames (exp2.txt, lu1_*.txt, pml_si0_lapack.txt). The backlog's
  "10-40x" came from the 351 s Eigen LU of the WP-PML3a smoke (concurrent debug build); measured
  same-time ratio is 2-3x (factor 3.1x random / 2.1x PML). A "570 s zgecon" had no raw source.
- Mark mechanisms "(inferred, not profiled)" in records; only timings count as measured.
- `set(BLA_SIZEOF_INTEGER 4)` before find_package(BLAS/LAPACK) works with MSYS2 OpenBLAS (both
  win presets, --fresh configure still finds libopenblas.dll.a and the probe passes).

**Why:** the first record drew a content conclusion from runs at different load levels.
**How to apply:** before explaining a timing difference on the shared machine, alternate the
cases in one process; see [[gmres-orthogonalization-findings]] for the BLAS-thread notes.
