# GMRES orthogonalisation: CGS2 vs MGS (WP-G1)

Background: the MLFMM scaling study (WP22b1) found the serial modified Gram–Schmidt (MGS) loop of
`solver::gmres` at 42–66 % of the Ag solve time (1 000–1 700 full-GMRES iterations, 2N = 4·10⁴ …
3·10⁵), while the MLFMM matvec is OpenMP-parallel. WP-G1 adds classical Gram–Schmidt with
re-orthogonalisation (CGS2) as the default (`GmresParams::orthogonalization`); MGS stays
selectable.

## Design

- **Always two passes.** h₁ = Vᴴw, w −= Vh₁, h₂ = Vᴴw, w −= Vh₂, Hessenberg column h = h₁ + h₂.
  The selective variant (second pass only when ‖w‖ drops below 0.7 of its norm, the criterion MGS
  keeps) saves nothing for the spectra met here: the criterion fired in 45 of 46 MGS steps (dense
  Ag sphere, PMCHWT + left Jacobi, `test_gmres.cpp`), 52 of 53 (Si sphere, ICTF), 157 of 200
  (row-scaled random matrix) and 1 000 of 1 000 (benchmark operator below). With an unconditional
  second pass, the update of pass 1 and the projection of pass 2 are fused per row sub-block
  (the sub-block's part of V, ≈ 1 MiB, is reused from cache), so a step streams V three times
  instead of four.
- **Storage.** The Krylov basis lives in contiguous column-major panels of 64 columns, the last
  one sized to the remainder of m + 1 (m = restart, or max_iter for full GMRES), so a filled basis
  takes exactly 16 (m + 1) n bytes (GMRES(100): 64 + 37 columns); it grows by whole panels without
  copying (a single growing matrix would need a reallocation copy with 2× peak memory at 3–16 GB
  bases).
- **Products.** Vᴴw and Vh are hand-written OpenMP loops over fixed 4 096-row blocks, handed to
  the threads one block at a time (`schedule(dynamic, 1)`, so a descheduled thread does not hold
  a fixed share of the rows while the others wait at the barrier); per block, Eigen dot products /
  four-column AXPY sweeps on contiguous column segments. Each block's partial sums of Vᴴw depend
  on its rows only, not on the thread that computes them, and are added in block order. Results
  are therefore bitwise reproducible and independent of the thread count and the scheduling (unit
  test with 1 and the default number of threads). The benchmark runs below used the earlier
  `schedule(static)`. BLAS GEMV (Eigen with `EIGEN_USE_BLAS` dispatches to OpenBLAS in the
  win builds) was not used: threaded OpenBLAS GEMV splits the Vᴴw reduction by thread count, so its
  results depend on `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` (observed in the determinism test:
  a test operator using one GEMV changed bitwise with the thread count), and calling it inside the
  row-block loop would nest two thread pools. Both reach the same memory bandwidth (microbenchmark
  below).
- **Numerics.** Orthogonality ‖I − VᴴV‖_F after 200 iterations on a row-scaled (scales
  10⁻³ … 10³) dense random system: CGS2 1.2·10⁻¹⁴, MGS (with its selective second pass)
  1.5·10⁻¹⁴; residual histories agree to 3·10⁻¹⁵ relative, ‖A(x_CGS2 − x_MGS)‖/‖b‖ = 3·10⁻¹³.
  n = 12 411 (four row blocks), left/right preconditioning, full and GMRES(3): identical
  iteration counts (43/43, 62–63) and ‖x_CGS2 − x_MGS‖/‖x_MGS‖ ≤ 3·10⁻¹⁶.

## Benchmark

`specklebem_gmres_orthogonalization` (configure with `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`):
operator y = d ⊙ x + U(Wᴴx), d uniform in the disk |z − 1| < 0.95, U, W random n × 2, so a
matvec costs ≈ 1 ms at n = 2·10⁵ (0.6–1.0 ms in the runs below, 0.97–1.46 ms in the review
spot check) and the orthogonalisation dominates; GMRES runs a fixed number of
iterations (tol 10⁻³⁰⁰, no restart); "ms/it k" is the mean wall time per iteration over the ten
iterations ending at k (callback timestamps). Machine: AMD Ryzen 9 5900X (12 cores / 24 threads,
dual-channel DDR4, 2 × 32 MB L3), Windows 11, win-release (UCRT64 GCC, OpenBLAS), 24 OpenMP
threads by default.

### Loaded machine (load average 150–200 from another worker's MLFMM runs)

n = 2·10⁵, 400 iterations (basis 1.28 GB), 2026-10-10:

| variant | threads | total s | orth s | share | reorth | final res | ms/it k=10 | ms/it k=100 | ms/it k=250 | ms/it k=400 |
|---|---|---|---|---|---|---|---|---|---|---|
| MGS  | 1 (serial) | 509.8 | 503.8 | 98.8 % | 400 | 1.46e-11 | 38 | 940 | 630 | 1 155 |
| CGS2 | 24 | 67.7 | 65.6 | 96.9 % | 400 | 1.46e-11 | 34 | 116 | 207 | 243 |
| CGS2 | 12 | 77.2 | 75.6 | 97.9 % | 400 | 1.46e-11 | 54 | 134 | 205 | 308 |

Orthogonalisation speed-up 7.7× (24 threads) and 6.7× (12 threads) in this run, but not
reproducible: a review spot check under full contention gave only 0.65–1.5× (the parallel loops
stall at their barriers when single threads are descheduled). Speed-ups measured on a loaded
machine are noise in either direction (MGS per-iteration times are not even monotone in k) and
are not used in the conclusions.
A 1 000-iteration run was started but stopped: under this load MGS alone took 2 448 s
(99.6 % orthogonalisation, 1 000 of 1 000 steps re-orthogonalised) and serial CGS2 more than an
hour.

### n = 2·10⁵, 1 000 iterations (basis 3.2 GB), runs started at low load

The machine was shared with another worker's MLFMM runs for the whole session; both runs were
started when the load average had dropped below 8, but it rose again during the runs (run A: to
~120 during the CGS2 12-thread row; run B: from 7.5 to ~400 during MGS). Rows marked * ran
mostly under that load.

| run | variant | threads | total s | orth s | share | reorth | ms/it k=10 | k=100 | k=250 | k=500 | k=750 | k=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | CGS2 | 24 | 181.2 | 179.5 | 99.1 % | 1000 | 3.1 | 34 | 83 | 213 | 272 | 360 |
| A | CGS2* | 12 | 295.0 | 293.4 | 99.4 % | 1000 | 17 | 69 | 150 | 240 | 414 | 393 |
| A | CGS2* | 6 | 339.1 | 337.3 | 99.5 % | 1000 | 12 | 96 | 108 | 248 | 427 | 627 |
| A | MGS* | 1 (serial) | 852.6 | 850.2 | 99.7 % | 1000 | 14 | 139 | 225 | 391 | 2 795 | 1 084 |
| B | MGS (* from k ≈ 300) | 1 (serial) | 808.3 | 805.2 | 99.6 % | 1000 | 6.2 | 155 | 271 | 747 | 3 348 | 760 |
| B | CGS2* | 24 | 222.2 | 220.1 | 99.1 % | 1000 | 6.4 | 35 | 88 | 175 | 272 | 344 |

Matvec 0.6–1.0 ms; final Arnoldi residual 1.11·10⁻²⁵ for every row (identical to three digits).
Run B, `--blas-reference`: one Eigen/OpenBLAS GEMV pair (Vᴴw, w −= Vh) on a contiguous basis
under load ~400 took 0.20 / 0.22 / 0.37 / 0.35 / 0.49 s at k = 100 / 250 / 500 / 750 / 1 000, so
a full CGS2 step (three sweeps, 0.34 s at k = 1 000) costs less than one BLAS pass pair under the
same load.

Reading: CGS2 with 24 threads costs ≈ 0.35 ms per basis vector and step at n = 2·10⁵
(≈ 1.7·10⁻⁹ s per complex entry, ≈ 28 GB/s for the three sweeps), nearly the same at low and at high machine load. The
low-load MGS points (k ≤ 250) are 4.4× (k = 100) and 3.1× (k = 250) slower per iteration; the
whole-solve orthogonalisation time drops from 805–850 s (MGS, partly loaded) to 180–220 s
(CGS2, 24 threads): 3.7–4.7×. Fewer threads are slower (12: 1.6×, 6: 1.9× the 24-thread time,
both partly loaded), so the kernel does profit from SMT/all cores despite the bandwidth bound.

### Row-block schedule: static vs dynamic (review W3, 2026-10-11)

Same benchmark harness and flags compiled twice (the library loop with `schedule(static)` and
with `schedule(dynamic, 1)`), CGS2, 24 threads, n = 2·10⁵, 250 iterations, run alternately
twice on the shared machine (moderately loaded by other workers: 104–119 ms per iteration at
k = 250 against 83 ms in run A above):

| schedule | orth s (run 1 / run 2) | ms/it k=100 | ms/it k=250 |
|---|---|---|---|
| static | 18.5 / 14.5 | 77 / 56 | 119 / 108 |
| dynamic, 1 | 13.9 / 13.0 | 40 / 45 | 107 / 104 |

Dynamic is not slower and was 5–30 % faster here (within the load noise); the final residuals
are identical (4.87·10⁻⁸). Its expected benefit is under contention, where a descheduled thread
no longer holds a fixed share of the rows; the 49 blocks of n = 2·10⁵ give enough slack for 24
threads.

### Microbenchmark (one orthogonalisation pass, n = 2·10⁵, k = 1 000, machine load ≈ 5)

Throwaway kernel timings (best of 2–3, before the gmres integration) at nearly idle load, one
pass over a random 3.2 GB basis:

| kernel | threads | s per pass | effective GB/s |
|---|---|---|---|
| MGS (dot + AXPY per column) | 1 | 0.28–0.53 | 12–23 (2 reads, the second from L3) |
| BLAS GEMV pair (OpenBLAS zgemv, Vᴴw and w −= Vh) | 24 | 0.21–0.31 | 21–31 |
| row-block OpenMP pair (4 096–8 192 rows) | 24 | 0.20–0.29 | 22–31 |
| row-block OpenMP pair | 1 | 1.6–2.3 | 3–4 |

The orthogonalisation is memory-bandwidth bound. With dual-channel DDR4 the parallel kernels
reach ~25–30 GB/s against ~12–23 GB/s effective for the serial MGS (whose second read of each v_j
hits the 32 MB L3). An MGS step with its (almost always triggered) second pass costs ≈ 2 MGS
passes; a CGS2 step 1.5 parallel pairs (three sweeps). Expected idle speed-up on this desktop:
≈ 2× (1.6–2.5×); on servers with more memory channels the gap grows with the bandwidth ratio.

## Conclusions

- CGS2 is the default; orthogonality and solutions match MGS to rounding, iteration counts are
  equal in all tests (differences of ±1–2 near the tolerance are possible by design).
- On this machine the gain is bandwidth-limited. Expected on an idle machine: ≈ 2–4× for the
  orthogonalisation (low-load per-iteration times at k ≤ 250: 3.1–4.4×; kernel microbenchmark at
  k = 1 000: 1.6–3×); measured 3.7–4.7× for the whole 1 000-iteration orthogonalisation in
  partly loaded runs. Under heavy contention the gain is unreliable (0.65–1.5× in a review spot
  check). For the WP22b1 Ag solves (MGS 42–66 % of the solve time) the idle estimate means
  roughly a 1.3–2× shorter solve; the real solve gain is still to be confirmed with exclusive
  machine time (WP22b3). The remaining cost is three streams of V per step; a single-synchronisation variant (delayed re-orthogonalisation, DCGS2, Bielich et al.
  2022) would need two and is the next step if the orthogonalisation still dominates.
- Restarting is not affected; GMRES(m) uses the same kernels on a basis of m + 1 columns.
