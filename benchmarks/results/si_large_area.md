# Large-area Si: proxy on 128 GB and the 30 × 30 µm² server run (WP22d)

Phase 4 target of `docs/01_project_plan.md`: "Si 30×30 µm², σ = 100 nm, λ/10 mesh (≈ 2.2 M
unknowns) solves on a 256 GB node". Lead decision 2026-10-11: on the 128 GB development machine
this is shown by a **proxy** (the largest feasible Si patch with the same settings, measured, and
extrapolated); the full case is run by the project lead on a 1 TB Linux server with the run kit
[`scripts/server_run_kit/`](../../scripts/server_run_kit/README.md), together with a server Ag
series (L = 4 / 7 / 11 / 17.5 µm) whose own γ_solve closes the Ag part of the scaling DoD by
measurement on one machine (coordinator decision of the WP22d review).

**Summary**

- **Server (measured 2026-10-12, runs 1–3 of the kit): Si L = 30 µm, σ = 100 nm, 2N = 5 947 200,
  solved** on a 2 × 8-core AMD EPYC 9015 server with ≈ 1.08 TB (32 OpenMP threads): 95
  iterations to the true residual 9.9·10⁻⁴, setup 79 min, solve 5.56 h, wall 8.0 h, peak 697 GB
  (inside the extrapolated 600–720 GB; (peak − Krylov) / model 0.84). L = 20 µm: 95 iterations,
  peak 250.4 GB. Per unit of work the server is slower than the 12-core desktop for Si: 2.5 × in
  the far matvec, 1.9–2.5 × in the power summary, 2–5 × in the setup (cause untested). The
  extrapolation held for the memory and the iterations; method B overestimated the growth of the
  times from 20 to 30 µm by ≈ 35 %. Details: "Server results" below.
- **Server Ag series (measured 2026-10-13, run 4 of the kit): all four sizes converged**, Ag
  L = 4 / 7 / 11 / 17.5 µm, σ = 50 nm, graded 100 nm box, 2N = 6.96·10⁴ … 1.02·10⁶: 1 394 /
  1 518 / 2 272 / 2 920 iterations (4a–4c identical to the desktop), solve 656 s … 6.71 h, peak
  94.3 GB at 10⁶. **γ_solve = 1.37 ± 0.08** (4 sizes, one machine) over 2N = 6.96·10⁴ … 1.02·10⁶:
  the Ag criterion γ ≤ 2.0 is met by measurement over that range; the bottom of the DoD window,
  5·10⁴ … 6.96·10⁴, holds no converged Ag size on either machine (assessment, item 8). Server /
  desktop solve time 1.45–1.88 at equal L (L = 4–11 µm, measured on both; Si ≈ 2).
- **Proxy: Si L = 10.5 µm, σ = 100 nm, 2N = 1 098 720 unknowns, solved** (uniform 50 nm closing
  box with the fine band, rigorous beam w₀ = L/4, ICTF, MLFMM d₀ = 3, full GMRES tol 1e-3):
  90 iterations (true residual 9.6·10⁻⁴), setup 225 s, solve 1 464 s (24 min), peak working set
  89.5 GB (near field 16.3 GB, far operator 71.0 GB, Krylov basis 1.6 GB). Reflectance R = 0.3649,
  absorbed fraction A = 0.6341 (tol 1e-3, not a quantitative reflectance), 1 − R − A = 1.0·10⁻³.
- "Largest feasible" is meant by the model criterion: the largest L for which the
  `--estimate-only` setup peak plus the Krylov basis stays below 110 GB (103.5 + 1.6 GB;
  L = 11 µm: 120.7 GB model). With the proxy's measured ratio, L = 11 µm would probably have
  measured ≈ 104 GB (0.85 × 120.7 GB + 1.8 GB Krylov) and run within the ~116 GB available; it
  was not attempted because the criterion, the one the run kit applies on the server, refuses
  it. The uniform box is the most
  memory-efficient configuration at these sizes: the local leaf rule (q = 0.8) elevates nothing
  on it, and the graded box with q = 0.8 needs more (L = 10 µm model: 105.6 vs 96.5 GB).
- **σ = 100 nm costs more than the σ = 50 nm scaling series** at equal 2N (L = 4 / 6 µm, same
  mesh): iterations +23 / +33 %, solve +48 / +49 %, peak memory +9 / +5 % (steeper top-face
  triangles raise r_max from 64–65 to 74–77 nm, 81 nm at the proxy, and with it the Si leaf
  sampling order).
- **L = 30 µm, σ = 100 nm (extrapolated before the server run)**: 2N = 5.95·10⁶ (uniform box;
  the box adds 175 % to the top face), octree leaf 234 nm, `--estimate-only` model 823 GB; extrapolated
  peak **600–720 GB**, setup 27–53 min, solve **2.2–5.6 h** on the development machine (12 cores
  / 24 threads), 81–117 iterations; total wall time with the timing matvecs and the power
  summary 3.2–7.1 h (run kit, section 4). It fits the 1 TB server, **not a 256 GB node**, for
  three reasons:
  1. the "≈ 2.2 M unknowns" of docs/01 count the top face only: the open 600 × 600 grid has
     2N = 2 157 600 ≈ 2.16·10⁶;
  2. the closing box adds far more than planned. ADR 0006 did include it, but expected "a few
     percent" (WP2b measured 3.4 % at L = 30 µm, depth 2 µm, 400 nm coarse cells). It is
     +85–175 % now because of three later decisions: the ADR 0006 amendment of 2026-10-10 (box
     cells ≤ λ₁/5 = 100 nm, fine band 3δ + 3σ mandatory for Si), the 5.65 µm Si depth
     (10 intensity decay lengths) and the uniform-box recommendation for Si MLFMM (ADR 0008
     decision 2026-10-11). Uniform box: +175 %, 5.95·10⁶ unknowns; graded λ₁/5 box with q = 0.8:
     +85 %, 4.0·10⁶ unknowns, model 627 GB, extrapolated peak 460–550 GB, but about 1.4 × the
     iterations (extrapolated solve 2.9–6.8 h);
  3. independent of the box, the memory per unknown at the 30 µm octree (234 nm leaves) is
     about 138 kB in the model (823 GB / 5.95·10⁶), so the top face alone would model at
     ≈ 300 GB, an expected peak of ≈ 215–260 GB (0.72–0.86 × the model + Krylov): already at the
     limit of a 256 GB node.
- **Octree quantisation dominates the large sizes.** The root edge is the patch edge L, so the
  leaf is L / 2^k, the smallest value ≥ max(λ/4, r_max / 0.6): 156 nm at L = 20 µm, 195 nm at
  25 µm, 234 nm at 30 µm, 125 nm at 32 µm (σ = 100 nm needs the local leaf rule q = 0.8 for the
  last: 2 988 steep functions are elevated). Near field and Si patterns grow with the leaf, so
  **L = 32 µm (2N = 6.65·10⁶, q = 0.8) has a smaller model (546 GB) than L = 30 µm (823 GB)**.
  Proposal (not applied, ADR 0008 change): pad the root to leaf floor × 2^k so that every L gets
  the smallest admissible leaf; L = 30 µm would then need about the L = 32 µm memory.

## Setup

| Parameter | Value |
|---|---|
| Surface | Gaussian height map (`generate_gaussian_height_map`), **σ = 100 nm** (`--sigma 100e-9`; the WP22b3 scaling series used 50 nm), Lc = 500 nm, mesh h = 50 nm = λ/10, seed 1, L × L patch |
| Closing box (ADR 0006) | **uniform box** `--box-mesh-size 50e-9` (box cells = top-face cells, M = 0), depth 5.65 µm (`default_box_depth`, 10 intensity decay lengths), fine band 3δ + 3σ = 3.69 µm (irrelevant for the uniform box, recorded in the label); the recommended Si MLFMM configuration (ADR 0008 decision 2026-10-11) |
| Media | λ = 500 nm, vacuum R1, Si ε_r = 18.478 − 0.606j (exp(+jωt)) |
| Excitation | `AngularSpectrumBeam` (rigorous, ADR 0006 amendment 4), normal incidence (+z), focus at the origin, E along x, **w₀ = L/4** (= L cos θ/4 at θ = 0, amendment 2026-10-11; `check_beam_waist` enforced), controlled ball 1.01 × the largest vertex distance |
| Operator | `Simulation`, compression `"mlfmm"`, d₀ = 3, automatic leaf rule (leaf ≥ max(λ/4, r_max/0.6)), `leaf_radius_quantile` 1 (the default), exact-part budget automatic (no exact part for Si) |
| Solver | ICTF without preconditioner (`formulation::recommend`), full GMRES with CGS2 (default), x₀ = 0, tolerance 1e-3 on the monitored residual (= the true residual without preconditioner). Note: ADR 0006 (amendment 2026-10-11) states that Si is accurate at 1e-4 (quantitative Ag reflectance needs ≤ 1e-5); 1e-3 is the scaling setting, so R and A in this record are consistency checks, not quantitative reflectances |
| Power summary | `--far-field 1`: R = P_refl / P_inc, P_refl = (1/2η₁)∫|F|² dΩ over the reflection hemisphere (Gauss–Legendre in θ × trapezoid in φ, 1° spacing, and the 2° grid as a check), P_inc = `AngularSpectrumBeam::power()`; A = P_abs / P_inc with P_abs = ½ Re∮(n̂×M)·J* dS from the currents (valid for Si here: the balance closes; for Ag at oblique incidence use A_ext, WP-A1) |
| Timings | setup = `Simulation::assemble` (octree, far operator, near field, rhs); matvec = median of 5 interleaved applies (full, near, far); solve = GMRES wall time; per-iteration timestamps from `--history` ([`si_large_area_histories.txt`](si_large_area_histories.txt)) |
| Memory | near field, far operator (patterns, translators, one apply workspace), Krylov (it + 1) 2N 16 B, peak working set after the setup and at the end (`GetProcessMemoryInfo`) |
| Machine | AMD Ryzen 9 5900X (12 cores / 24 threads, 2-channel DDR4), 128 GB, Windows 11, MSYS2 UCRT64 GCC 16.1, OpenBLAS, `win-release`, 24 OpenMP threads; branch `wp/22d-si-large-area` (main a65381a + the benchmark options of 18a927e), 2026-10-11 |
| Protocol | machine lock (shared with WP-A2), one case per process, started after the 1-min load average had been < 5 at two checks 30 s apart with no other benchmark or compiler running (`load0` = the value at the start) |

## Proxy and σ = 100 nm points (measured)

| L [µm] | 2N | levels | leaf [nm] | r_max [nm] | setup [s] (near / far) | matvec [s] (near / far) | it | solve [s] | s/it | near [GB] | far [GB] | Krylov [GB] | peak after setup [GB] | peak [GB] | model [GB] | R (2° grid) | A | 1 − R − A | post [s] | load0 |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 4 | 293 760 | 6 | 186 | 74.4 | 83.9 (51.0 / 29.8) | 3.33 (0.18 / 3.03) | 87 | 300 | 3.44 | 5.78 | 18.77 | 0.41 | 21.6 | 25.2 | 30.3 | 0.36827 (0.36827) | 0.63001 | 1.7·10⁻³ | 64 | 1.50 |
| 6 | 498 240 | 6 | 188 | 77.1 | 117.1 (75.9 / 38.1) | 5.51 (0.29 / 5.23) | 85 | 511 | 6.01 | 9.80 | 32.23 | 0.69 | 37.0 | 43.0 | 51.1 | 0.37183 (0.37183) | 0.62743 | 7.4·10⁻⁴ | 146 | 3.87 |
| **10.5** | **1 098 720** | 7 | 164 | 81.4 | **225.4** (139.0 / 81.0) | **15.54** (0.47 / 15.12) | **90** | **1 464** | 16.27 | 16.32 | 70.96 | 1.60 | 72.2 | **89.5** | 103.5 | 0.36490 (0.36486) | 0.63411 | 1.0·10⁻³ | 324 | 1.74 |

Peak = peak working set at the end (near + far + Krylov; the 14.6 GB apply workspace of the far
operator is allocated at the first apply, so the peak after the setup is lower). Model = the
`--estimate-only` setup peak (1.5 near + 1.1 far + 1 GB): the measured peak minus the Krylov
basis is 0.82–0.85 × the model (WP22b3 series 0.72–0.86). Near and far memory equal the
estimates within 0.01 %. Octree setup ≤ 0.09 s, rhs 5 s (L = 10.5 µm). The far matvec is 91 /
95 / 97 % of the matvec (L = 4 / 6 / 10.5 µm; WP22b3 series 93–98 %). Solve / (it × matvec) =
1.03–1.09 (proxy 1.05); the CGS2 orthogonalisation of the proxy is ≈ 8.5 s by the WP22b3
bandwidth model (24 · 2N · k² bytes at 25 GB/s; too small to resolve in the history at 90
iterations).

**Comparison with the σ = 50 nm uniform-box series (WP22b3, same mesh sizes and 2N):**

| L [µm] | 2N | σ [nm] | r_max [nm] | it | setup [s] | matvec [s] | solve [s] | peak [GB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 293 760 | 50 / 100 | 64.1 / 74.4 | 71 / 87 | 58.9 / 83.9 | 2.88 / 3.33 | 203 / 300 | 23.2 / 25.2 |
| 6 | 498 240 | 50 / 100 | 65.2 / 77.1 | 64 / 85 | 100.4 / 117.1 | 5.28 / 5.51 | 342 / 511 | 40.9 / 43.0 |
| 10 / 10.5 | 1 022 400 / 1 098 720 | 50 / 100 | 67.2 / 81.4 | 68 / 90 | 198 / 225 | 13.72 / 15.54 | 954 / 1 464 | 74.8 / 89.5 |

- Rougher surface, same grid: the steepest top-face triangles are longer (r_max +16–21 %), the
  Si interior leaf sampling order follows r_max (L = 10.5 µm: order 35, patterns 43.5 GB for R2),
  and the near-field quadrature costs more (near setup +32 % at L = 4 µm). The iteration count
  is flat in L for both σ (σ = 100 nm: 87 / 85 / 90), 23–33 % above σ = 50 nm at equal L (and
  90 vs 68 at L = 10.5 vs 10 µm).
- Reflectance: flat Si at normal incidence has |r|² = 0.388; the rough patches reflect
  0.365–0.372 in total (specular + diffuse) and the balance R + A closes to ≤ 1.7·10⁻³ (the
  remainder: forward scattering around the box and the 1e-3 solver tolerance). The 1° and 2°
  grids agree to 1·10⁻⁴.

## Extrapolation to L = 20 / 25 / 30 / 32 µm

Script [`benchmarks/si_large_area_extrapolation.py`](../si_large_area_extrapolation.py) (reads
the ROW lines of [`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt) and the estimates
[`si_large_area_estimates.txt`](si_large_area_estimates.txt); its output is the table below).
Two methods, anchored at the proxy (L = 10.5 µm, σ = 100 nm):

- **A, power law in 2N**, Q = Q_proxy (2N / 2N_proxy)^γ, with γ bounds = the union of γ ± 2 se
  of the σ = 50 nm uniform-box series (setup 0.83 ± 0.04, matvec 1.17 ± 0.02, solve
  1.09 ± 0.04, peak 0.95 ± 0.03; ± 1 se) and of the three σ = 100 nm rows (0.76 ± 0.05,
  1.18 ± 0.10, 1.21 ± 0.09, 0.96 ± 0.02): setup 0.65–0.90, matvec 0.98–1.37, solve
  1.00–1.39, peak 0.89–1.01. The σ = 100 nm γ rest on 3 points (1 degree of freedom of the fit),
  so their standard errors are themselves uncertain; they widen the bounds of the setup, the
  matvec and the solve, not of the peak. Blind to the leaf size, which the series covers only
  from 125 to 188 nm: it underestimates sizes whose octree quantisation gives larger leaves (L = 25–32 µm
  with q = 1), and it applies to the uniform box only.
- **B, component model**: the target's estimated near-field and far-operator memory (exact to
  < 1 % in every measured row) times the measured cost per GB (far matvec: proxy 0.213 s/GB to
  1.4 × that for one more octree level, as between 6 and 7 levels in the series; near matvec,
  near and far setup: range over all rows), iterations 0.9–1.3 × the proxy's 90 (graded box:
  times the graded / uniform iteration ratio 1.39–1.46 measured at L = 4 / 8 µm in WP22b3),
  solve = it × matvec × 0.99–1.09 (solve / (it × matvec) of the rows: CGS2 and the rest of the
  iteration), peak = (0.72–0.86) × model + Krylov.

**Stated ranges** (summary, run kit): method B for the peak memory and the setup (method A
cannot see the growth of the near field and the Si patterns with the leaf; at L ≥ 25 µm with
q = 1 its peak falls below 0.72 × the model, the floor of every measured row), the union of A
and B for the matvec and the solve; B alone for the graded box. **All values are extrapolations
for the development machine (12 cores / 24 threads), not measurements.**

| target (σ = 100 nm) | 2N | leaf [nm] | model [GB] | peak A [GB] | peak B [GB] | setup A / B | matvec A / B [s] | it (B) | solve A | solve B |
|---|---:|---:|---:|---:|---:|---|---|---:|---|---|
| L = 20 µm, uniform 50 nm, q = 1 | 3 004 800 | 156 | 285 | 219–246 | 209–250 | 7–9 min / 8–15 min | 42–62 / 45–62 | 81–117 | 1.1–1.6 h | 1.0–2.2 h |
| L = 20 µm, graded 100 nm, q = 0.8 | 2 056 800 | 156 | 298 | — | 219–261 | — / 8–16 min | — / 46–64 | 113–171 | — | 1.4–3.3 h |
| L = 25 µm, uniform 50 nm, q = 1 | 4 356 000 | 195 | 499 | 305–357 | 366–437 | 9–13 min / 15–30 min | 60–103 / 73–102 | 81–117 | 1.6–2.8 h | 1.6–3.6 h |
| L = 25 µm, graded 100 nm, q = 0.8 | 2 964 000 | 195 | 396 | — | 291–348 | — / 10–21 min | — / 62–86 | 113–171 | — | 1.9–4.5 h |
| L = 30 µm, uniform 50 nm, q = 1 | 5 947 200 | 234 | 823 | 402–489 | 602–718 | 11–17 min / 27–53 min | 82–158 / 113–157 | 81–117 | 2.2–4.3 h | 2.5–5.6 h |
| L = 30 µm, graded 100 nm, q = 0.8 | 3 999 600 | 234 | 627 | — | 460–549 | — / 18–36 min | — / 94–131 | 113–171 | — | 2.9–6.8 h |
| L = 32 µm, uniform 50 nm, q = 1 | 6 650 880 | 250 | 1 009 | 444–547 | 737–879 | 12–19 min / 34 min–1.1 h | 91–185 / 137–190 | 81–117 | 2.5–5.0 h | 3.1–6.7 h |
| L = 32 µm, uniform 50 nm, q = 0.8 | 6 650 880 | 125 | 546 | 444–547 | 403–482 | 12–19 min / 13–26 min | 91–185 / 91–127 | 81–117 | 2.5–5.0 h | 2.0–4.5 h |
| L = 32 µm, graded 100 nm, q = 0.8 | 4 458 240 | 125 | 552 | — | 406–486 | — / 13–27 min | — / 91–127 | 113–171 | — | 2.8–6.6 h |

Stated ranges (extrapolated): **L = 20 µm** uniform: peak 210–250 GB, setup 8–15 min, solve
1.0–2.2 h. **L = 30 µm** uniform: peak 600–720 GB, setup 27–53 min, solve 2.2–5.6 h; graded
q = 0.8: peak 460–550 GB, setup 18–36 min, solve 2.9–6.8 h. **L = 32 µm** uniform q = 0.8: peak
400–480 GB, setup 13–26 min, solve 2.0–5.0 h. The total wall time of a server run adds the
timing matvecs before GMRES (2 (matvecs + 1) full-matvec equivalents: 12 with the default
`--matvecs 5`, 4 with the `--matvecs 1` of the large runs) and the `--far-field 1` power summary
(directions × triangles: 324 s at the proxy, ≈ 30 min at L = 30 µm); the run kit's run list
(README section 4) gives setup, timing matvecs, solve, post and the total per run (L = 30 µm
uniform: 3.2–7.1 h on the desktop).

For the server, the times scale with its throughput relative to the 12-core desktop. The Si
solve is far-matvec bound (91–97 % of the matvec in these rows), and the far matvec streams only
~5 GB/s of patterns at the proxy (71 GB in 15.1 s, against the 23–27 GB/s that CGS2 reaches on
this machine), so it is compute-bound and should scale with the core count until the memory
bandwidth limits it. On a two-socket server with interleaved memory a factor 3–8 shorter is
plausible but unmeasured; the run list starts with a smoke test and L = 20 µm to calibrate it.
(Measured since on the lead's 2 × 8-core server: 1.9–2.5 × longer, not shorter; see "Server
results: assessment", item 4.)
The Ag solve is different: its CGS2 orthogonalisation (30–44 % of the WP22b3 Ag solves) is
bandwidth-bound and scales with the memory bandwidth, so the Si factor does not apply to it, and a
single server Ag point could not be fitted together with the desktop series. The run kit
therefore runs a **server Ag series** (L = 4 / 7 / 11 / 17.5 µm, the WP22b3 options with
`--max-iter 4000`), which gives the `machine=server` rows their own γ_solve over
2N = 7.0·10⁴ … 1.02·10⁶ and closes the Ag scaling DoD by measurement on one machine.

## `--estimate-only` for L = 20 / 25 / 30 / 32 µm

Model = setup peak model (1.5 near + 1.1 far + 1 GB; the measured peak minus Krylov was
0.72–0.86 × the model). Uniform: `--box-mesh-size 50e-9`, q = 1; graded: the λ₁/5 = 100 nm box
(M = 1, Si with the fine band), q = 0.8 (the local leaf rule). All with the rigorous beam,
d₀ = 3. Box share = box triangles / top-face triangles (2 (L/h)²). Lines in
[`si_large_area_estimates.txt`](si_large_area_estimates.txt) (the first three rows are the
smoke test, the proxy and the largest case, 4d, of the server Ag series).

| L [µm] | σ [nm] | box | q | 2N | box share | levels | leaf [nm] | near [GB] | patterns [GB] (elevated) | far tables [GB] | model [GB] | elevated functions |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2 | 100 | uniform 50 nm | 1 | 127 680 | 1 230 % | 6 | 183 | 2.7 | 6.7 | 1.4 | 14 | 0 |
| 10.5 | 100 | uniform 50 nm | 1 | 1 098 720 | 315 % | 7 | 164 | 16.3 | 54.6 | 16.4 | 104 | 0 |
| 17.5 (Ag) | 50 | graded 100 nm | 1 | 1 017 450 | 38 % | 7 | 273 | 32.2 | 11.8 | 1.9 | 64 | 0 |
| 20 | 100 | uniform 50 nm | 1 | 3 004 800 | 213 % | 8 | 156 | 39.4 | 152.4 | 51.9 | 285 | 0 |
| 20 | 100 | graded 100 nm | 0.8 | 2 056 800 | 114 % | 8 | 156 | 43.4 | 156.9 (61.7) | 53.6 | 298 | 110 800 |
| 25 | 100 | uniform 50 nm | 1 | 4 356 000 | 190 % | 8 | 195 | 88.5 | 270.0 | 62.1 | 499 | 0 |
| 25 | 100 | graded 100 nm | 0.8 | 2 964 000 | 98 % | 8 | 195 | 55.0 | 215.1 (0.9) | 68.9 | 396 | 1 195 |
| 30 | 100 | uniform 50 nm | 1 | 5 947 200 | 175 % | 8 | 234 | 175.1 | 434.1 | 74.3 | 823 | 0 |
| 30 | 100 | graded 100 nm | 0.8 | 3 999 600 | 85 % | 8 | 234 | 104.0 | 344.3 | 82.8 | 627 | 0 |
| 32 | 100 | uniform 50 nm | 1 | 6 650 880 | 171 % | 8 | 250 | 221.8 | 531.6 | 82.3 | 1 009 | 0 |
| 32 | 100 | uniform 50 nm | 0.8 | 6 650 880 | 171 % | 9 | 125 | 56.3 | 274.3 (1.1) | 144.8 | 546 | 2 988 |
| 32 | 100 | graded 100 nm | 0.8 | 4 458 240 | 81 % | 9 | 125 | 59.1 | 270.9 (109.0) | 149.1 | 552 | 259 807 |
| 20 | 50 | uniform 50 nm | 1 | 3 004 800 | 213 % | 8 | 156 | 39.2 | 132.8 | 45.6 | 256 | 0 |
| 20 | 50 | graded 100 nm | 0.8 | 2 008 800 | 109 % | 8 | 156 | 41.9 | 152.5 (62.2) | 50.0 | 287 | 114 000 |
| 25 | 50 | uniform 50 nm | 1 | 4 356 000 | 190 % | 8 | 195 | 89.8 | 228.6 | 53.0 | 445 | 0 |
| 25 | 50 | graded 100 nm | 0.8 | 2 904 000 | 94 % | 8 | 195 | 52.4 | 210.1 (0.1) | 64.6 | 382 | 94 |
| 30 | 50 | uniform 50 nm | 1 | 5 947 200 | 175 % | 8 | 234 | 175.4 | 378.5 | 65.5 | 753 | 0 |
| 30 | 50 | graded 100 nm | 0.8 | 3 927 600 | 82 % | 8 | 234 | 100.4 | 325.9 | 76.7 | 594 | 0 |
| 32 | 50 | uniform 50 nm | 1 | 6 650 880 | 171 % | 9 | 125 | 54.9 | 266.0 | 131.0 | 520 | 0 |
| 32 | 50 | graded 100 nm | 0.8 | 4 381 440 | 78 % | 9 | 125 | 57.8 | 265.1 (106.4) | 137.9 | 531 | 261 761 |

- The uniform box has 1.46–1.49 × the unknowns of the graded box (box share 171–213 % vs
  81–114 % of the top face at σ = 100 nm). Model memory of the graded box relative to the
  uniform one: +5 % at L = 20 µm, −21 % at 25 µm, −24 % at 30 µm (627 vs 823 GB), +1 % at 32 µm
  with q = 0.8 on both. The graded box needed 96–104 instead of 69–71 iterations at equal L at
  σ = 50 nm (WP22b3, L = 4 / 8 µm), so it is slower per solve (the B column applies that ratio).
- q = 0.8 on the uniform box changes nothing at L ≤ 30 µm: the L = 10.5 / 20 / 25 / 30 µm
  estimates with q = 0.8 equal the q = 1 lines (0 elevated functions; checked 2026-10-11). It
  rescues L = 32 µm at σ = 100 nm (r_max 90 nm > 0.6 × 125 nm would force 250 nm leaves with
  q = 1: model 1 009 GB); at σ = 50 nm the global rule already gives 125 nm there.

## Server results (measured 2026-10-12: Si runs 1–3; 2026-10-13: Ag series 4a–4d)

Runs of `scripts/server_run_kit/run_large_area.sh`, one per row; the raw `row.txt` lines are in
[`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt) (block "WP22d server"). A `row.txt` line is the
benchmark's `ROW` line (the fields below) plus the annotations `gmres=cgs2 machine=server
threads= places= numa= matvecs= vmhwm_gb= rss_max_gb= load0= wall_s=`. Runs 3, 3b and 5 use
`--matvecs 1` (`matvecs=1`): their `matvec_s`, `near_mv_s` and `far_mv_s` are single applies
after one untimed round, not the median of 5 as in every desktop row and in run 2; compare
matvec times between machines with run 2. A `post_error=1` annotation means the power summary
failed after the solve (no R / A; the solve fields are valid). The raw logs stay off the
repository; only the `row.txt` lines and the hardware and software description below are recorded.

**Server** (from `system.txt` and the lead's note):

| field | value |
|---|---|
| CPU model | AMD EPYC 9015 8-Core Processor (Zen 5, family 26), max 4.1 GHz, AVX-512, L3 128 MiB in 4 instances |
| sockets × cores per socket (threads per core) | 2 × 8 (2): 16 physical cores, 32 hardware threads |
| memory (total; MemAvailable at the start of run 3) | MemTotal 1 055 844 144 kB ≈ 1.08 TB (1 081.2 GB in the script's units); MemAvailable 1 072.4 GB |
| NUMA nodes; policy used (`numa=` annotation: interleave / none) | 2 nodes of ≈ 541 GB each (≈ 515 500 MB in `numactl -H`, which counts MiB); `numactl --interleave=all` (`numa=interleave`) |
| OpenMP threads, `OMP_PLACES`, `OMP_PROC_BIND` | 32, `threads`, `spread`: all hardware threads including the SMT siblings (README section 2 recommends the physical core count, 16 here, with `OMP_PLACES=cores`; that setting was not tried) |
| OS, kernel; transparent huge pages | an EL10 Linux distribution, kernel 6.12 (x86_64); THP `always` |
| compiler (version), build type, flags | GCC 14.3.1 (Red Hat), `release` preset (Release, `-O3 -DNDEBUG`, no `-march`); Eigen and spdlog from the system packages |
| BLAS / LAPACK (`BLAS_LIBRARIES` from `CMakeCache.txt`) | not recorded (system OpenBLAS presumably; the MLFMM solve hardly uses BLAS, README section 1) |
| batch system, cgroup memory limit | none; cgroup v2 user slice without a memory limit |
| commit (`git rev-parse HEAD`), date | 1be8f2b139610a21b50ba38fd3155fb180081767; runs started 2026-10-12 08:17 UTC (run 1), 08:43 UTC (run 2), 13:16 UTC (run 3); 2026-10-13 02:15 UTC (run 4d, its own call), 10:42 UTC (runs 4a, 4b, 4c in one series call, in that order) |
| other jobs on the node during the runs | none reported; `load0` 0.16 / 0.22 / 0.47 (runs 1–3), 0.11 / 26.58 / 26.79 / 23.85 (4a–4d; see assessment item 9); no `--force-memory`; `--numa` and `--threads` not varied |

**Runs, sizes and times** (ROW fields `N2 levels leaf_nm rmax_nm elevated setup_s near_s far_s
matvec_s near_mv_s far_mv_s it conv true_res solve_s`; annotations `load0 wall_s`):

| run | material | L [µm] | σ [nm] | box, q | 2N | levels | leaf [nm] | r_max [nm] | elevated | setup [s] (near / far) | matvec [s] (near / far) | it | conv | true res | solve [s] | s/it | load0 | wall [s] |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 smoke | Si | 2 | 100 | uniform 50 nm, 1 | 127 680 | 6 | 183 | 73.5 | 0 | 111.1 (49.7 / 59.1) | 3.29 (0.056 / 3.23) | 86 | 1 | 9.76·10⁻⁴ | 287 | 3.34 | 0.16 | 509 |
| 2 | Si | 20 | 100 | uniform 50 nm, 1 | 3 004 800 | 8 | 156 | 85.3 | 0 | 1 924.7 (1 001.3 / 912.8) | 111.23 (0.81 / 110.36) | 95 | 1 | 9.72·10⁻⁴ | 10 760 | 113.3 | 0.22 | 15 692 |
| 3 target | Si | 30 | 100 | uniform 50 nm, 1 | 5 947 200 | 8 | 234 | 90.2 | 0 | 4 759.7 (2 648.9 / 2 090.1) | 207.7 (3.60 / 204.4), single apply | 95 | 1 | 9.94·10⁻⁴ | 20 007 | 210.6 | 0.47 | 28 914 |
| 3b (alt.) | Si | 30 | 100 | graded 100 nm, 0.8 | not run (not needed: run 3 fitted) | | | | | | | | | | | | | |
| 4a | Ag | 4 | 50 | graded 100 nm (M = 1, no fine band), 1 | 69 600 | 5 | 250 | 127.9 | 0 | 38.8 (25.7 / 2.8) | 0.317 (0.035 / 0.282) | 1 394 | 1 | 1.09·10⁻³ | 656 | 0.470 | 0.11 | 737 |
| 4b | Ag | 7 | 50 | graded 100 nm (M = 1, no fine band), 1 | 186 480 | 6 | 219 | 129.4 | 0 | 99.5 (64.5 / 8.0) | 0.946 (0.075 / 0.867) | 1 518 | 1 | 9.38·10⁻⁴ | 1 839 | 1.21 | 26.58 | 2 055 |
| 4c | Ag | 11 | 50 | graded 100 nm (M = 1, no fine band), 1 | 422 400 | 6 | 344 | 141.2 | 0 | 259.7 (186.9 / 12.3) | 2.044 (0.428 / 1.617) | 2 272 | 1 | 8.84·10⁻⁴ | 6 902 | 3.04 | 26.79 | 7 424 |
| 4d | Ag | 17.5 | 50 | graded 100 nm (M = 1, no fine band), 1 | 1 017 450 | 7 | 273 | 137.1 | 0 | 565.6 (388.2 / 32.7) | 5.010 (0.660 / 4.346) | 2 920 | 1 | 8.28·10⁻⁴ | 24 163 | 8.28 | 23.85 | 25 362 |
| 5 (opt.) | Si | 32 | 100 | uniform 50 nm, 0.8 | not run so far (optional, after run 4) | | | | | | | | | | | | | |

Wall time = setup + timing matvecs (2 (matvecs + 1) × matvec: 39 / 1 335 / 831 s) + solve + post
within 1 / 20 / 43 s (mesh, rhs and the estimate). The last GMRES iteration of run 3 ended
19 799 s into the solve (from the last `HIST` line of the run log, which is not in the
repository; the only figure in this section that is not a `row.txt` field): 208.4 s per
iteration, 1.003 × the matvec; the remaining 208 s of `solve_s` match one more apply (the true
residual). Solve / (it × matvec) = 1.014–1.018 in all
three runs (desktop rows 0.99–1.09): CGS2 is negligible, the solve is the far matvec (98–99 % of
the matvec).

Runs 4a–4d (`matvecs=5`, medians as on the desktop): wall time = setup + 12 matvecs + solve + post
within 1 / 4 / 8 / 19 s. Their setup also contains the Ag exact part (`exact_s` 0.7 / 3.1 / 0.7 /
8.4 s). Solve / (it × matvec) = 1.48 / 1.28 / 1.49 / 1.65: with 1 394–2 920 iterations the CGS2
orthogonalisation is 22–39 % of the Ag solve (`benchmarks/mlfmm_scaling_projection.py`, section 5,
on [`mlfmm_scaling_ag_server_histories.txt`](mlfmm_scaling_ag_server_histories.txt); desktop
30–44 %), unlike the 95-iteration Si solves.

**Memory and power** (ROW fields `near_gb exact_gb far_gb krylov_gb peak_setup_gb peak_gb
peak_est_gb R R_coarse A post_s`; annotations `vmhwm_gb rss_max_gb`):

| run | near [GB] | exact [GB] | far [GB] | Krylov [GB] | peak after setup [GB] | peak [GB] (getrusage) | VmHWM [GB] | model [GB] | (peak − Krylov) / model | R (R_coarse) | A | 1 − R − A | post [s] | vs. estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1 smoke | 2.69 | 0 | 8.08 | 0.18 | 9.60 | 11.14 | 11.14 | 13.9 | 0.787 | 0.37597 (0.37597) | 0.62071 | 3.3·10⁻³ | 70 | peak inside 10–12 GB; model = README 14 GB |
| 2 | 39.37 | 0 | 204.27 | 4.62 | 194.7 | 250.42 | 250.42 | 284.7 | 0.863 | 0.36325 (0.36176) | 0.63632 | 4.2·10⁻⁴ | 1 653 | peak 0.4 GB above 210–250 GB; model = README 285 GB |
| 3 target | 175.07 | 0 | 508.42 | 9.13 | 613.1 | 697.01 | 697.01 | 822.9 | 0.836 | 0.36711 (0.37500) | 0.63220 | 6.9·10⁻⁴ | 3 273 | peak inside 600–720 GB; model = README 823 GB |
| 3b (alt.) | not run | | | | | | | | | | | | | |
| 4a | 1.68 | 0.047 | 0.87 | 1.55 | 2.50 | 4.21 | 4.21 | 4.48 | 0.593 | 0.97342 (0.97342) | 0.00287 | 2.4·10⁻² | 38 | peak = desktop 4.22 GB (README "≈ 4.2"); model = README 4.5 GB |
| 4b | 3.62 | 0.271 | 2.39 | 4.53 | 5.92 | 10.92 | 10.92 | 9.09 | 0.703 | 0.97782 (0.97783) | 0.02314 | −9.6·10⁻⁴ | 101 | peak = desktop 10.94 GB (README "≈ 10.9"); model = README 9.1 GB |
| 4c | 20.85 | 0.011 | 6.54 | 15.36 | 26.97 | 43.01 | 43.01 | 39.47 | 0.700 | 0.97776 (0.97769) | 0.01770 | 4.5·10⁻³ | 230 | peak = desktop 43.01 GB (README "≈ 43"); model = README 39.5 GB |
| 4d | 32.16 | 0.414 | 13.63 | 47.55 | 44.65 | 94.26 | 94.26 | 64.23 | 0.727 | 0.97702 (0.97861) | 0.01260 | 1.0·10⁻² | 554 | peak inside 87–101 GB; Krylov inside 41–55 GB; model = README 64 GB |
| 5 (opt.) | not run so far | | | | | | | | | | | | | |

The sampled RSS maximum (`rss_max_gb` 11.13 / 250.38 / 696.92 GB) agrees with VmHWM and with
`peak_gb`. Near and far memory equal the estimates (far within 0.005 %, near to the digits of
[`si_large_area_estimates.txt`](si_large_area_estimates.txt)). Run 3's memory check needed
869.3 GB (model 822.9 + Krylov basis for 150 iterations 14.4 + margin 32 GB) and passed against
1 072.4 GB available; the peak used 697.0 GB, 172 GB below the reserve.

Runs 4a–4d: the sampled RSS maximum (`rss_max_gb` 4.21 / 10.91 / 43.00 / 94.26 GB) agrees with
VmHWM and `peak_gb` to 0.01 GB; MemAvailable (sampled every 30 s, the samples are not in the
repository) stayed above 978 GB throughout. Near, exact and far memory, the model and the peak
after the setup equal the desktop WP22b3 rows at equal L (ratio 1.000; peak after setup within
0.2 %): memory is machine independent. The Krylov basis is (it + 1) 2N · 16 B and at 4d the
largest single item (47.6 of 94.3 GB). (peak − Krylov) / model = 0.59–0.73, the same as the
desktop Ag rows at equal L (0.595 / 0.705 / 0.700 at 4–11 µm) and below the band 0.72–0.86 of
item 2, which the desktop WP22b3 Ag rows at L ≤ 11 µm do not reach either (0.58–0.71; the capped
L = 15 / 17.5 µm rows 0.75 / 0.72): for Ag at these sizes the model is more conservative than for
Si, on both machines.

**Server Ag series** (runs 4a–4d, all converged): γ_solve of the `machine=server` Ag rows
(`benchmarks/mlfmm_scaling_fit.py`) over 2N = 6.96·10⁴ … 1.02·10⁶: **1.37 ± 0.08** (4 sizes,
standard error of the slope; DoD: ≤ 2.0; desktop WP22b3 series 1.40 ± 0.14 over 6.96·10⁴ …
4.2·10⁵, `mlfmm_scaling.md`). Assessment: items 8–13 below.

### Server results: assessment

All figures below are computed from the three `row.txt` lines and the desktop rows of
[`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt) (one exception, named where it is used: the
time of run 3's last iteration from the run log); "measured" means a ROW field, "estimated" the
`--estimate-only` model or the extrapolation above, "hypothesis" an untested explanation.

**1. The Phase 4 case is solved on the 1 TB server (measured).** Run 3 solved Si 30 × 30 µm²,
σ = 100 nm, λ/10 mesh (h = 50 nm) with the ADR 0006 closing box (uniform 50 nm, depth 5.65 µm),
2N = 5 947 200, rigorous beam, ICTF, MLFMM d₀ = 3: full GMRES converged to the true residual 9.94·10⁻⁴ in 95
iterations; setup 79 min, solve 5.56 h, power summary 55 min, wall time 8.0 h, peak 697 GB of the
≈ 1.08 TB node. This completes the remaining part of the fourth Phase 4 DoD item of
[`docs/01_project_plan.md`](../../docs/01_project_plan.md): "the full case is run by the project
lead on a 1 TB server with the run kit of WP22d (decision 2026-10-11)"; the other part, the proxy
on the 128 GB machine, was done in WP22d. The item's first sentence, "solves on a 256 GB node", is
not met and cannot be at these settings: the measured peak is 2.7 × 256 GB (the reasons are in
the summary above: the box adds 175 % to the top face, and the 30 µm octree has 234 nm leaves).
Even run 2 (L = 20 µm, 2N = 3.0·10⁶, already more than the "≈ 2.2 M unknowns" of docs/01) peaked
at 250.4 GB, which leaves no room for the operating system on a 256 GB node. Lead decision 2026-10-13:
the DoD item is reworded to "solves on a 1 TB node" and is met by run 3; a memory reduction towards
256 GB is the optional WP22f. The Ag part of the scaling DoD, open when part 1 was written, is
assessed in item 8 (server run 4, WP22e part 2). R and A of these runs are consistency
checks, not quantitative results: tol 1e-3 and w₀ = L/4, wider than the w₀ ≤ L/6 that ADR 0006
(amendment 2026-10-12b) asks for quantitative Si work.

**2. Memory against the prediction (measured vs estimated).** The peaks are 11.1 / 250.4 /
697.0 GB against the expected 10–12 / 210–250 / 600–720 GB of the run list (README section 4):
runs 1 and 3 inside, run 2 0.4 GB (0.2 %) above the upper end. The `--estimate-only` model
(`peak_est_gb` 13.9 / 284.7 / 822.9 GB) equals the README column (14 / 285 / 823 GB): same
estimator, same mesh. (peak − Krylov) / model = 0.787 / 0.863 / 0.836, against the desktop band
0.72–0.86 (WP22b3 and WP22d rows; the three σ = 100 nm desktop rows 0.82–0.85). Runs 1 and 3 lie
inside it, run 2 just above it (0.863), which is why its peak reached the top of the range. The
peak after the setup is 0.68–0.75 × the model. The model stays conservative by 14–16 % at
L = 20 / 30 µm (21 % at 2 µm), defined as 1 − (peak − Krylov) / model = 13.7 / 16.4 % (21.3 %);
the need of the memory check (model + full 150-iteration Krylov basis + 32 GB) was 25 % above
the actual peak at L = 30 µm.

**3. Iterations (measured).** 86 / 95 / 95 at L = 2 / 20 / 30 µm against 87 / 85 / 90 on the
desktop at L = 4 / 6 / 10.5 µm (σ = 100 nm) and the extrapolated 81–117. Over a factor 47 in 2N
(1.3·10⁵ … 5.9·10⁶) the count is flat (fit exponent 0.03); the large patches need 6 % more than the
proxy, and runs 2 and 3 need exactly the same number. The iteration count is machine independent
up to the summation order of the reductions, so this is a property of the formulation and not of
the server. The desktop smoke test of the kit (L = 1 µm) needed 116; the smallest patches iterate
more, not the largest.

**4. Server against desktop speed (measured, both sides where possible).** The server is slower
than the 12-core / 24-thread desktop for Si:

| quantity | server | desktop | server / desktop |
|---|---|---|---:|
| far matvec per GB of far operator, 6 levels | 0.400 s/GB (run 1, L = 2 µm) | 0.162 s/GB (L = 4 and 6 µm, σ = 100 nm) | 2.5 |
| matvec, run 2 (median of 5) | 111.2 s | 42–62 s (extrapolated, A ∪ B) | 1.8–2.6 |
| s/it, run 2 | 113.3 s | 42–68 s (extrapolated: matvec × 0.99–1.09) | 1.7–2.7 |
| setup, run 2 | 32.1 min | 8–15 min (extrapolated, B) | 2.1–4.0 |
| near setup per GB of near field | 18.5 s/GB (run 1) | 7.7–8.8 s/GB (σ = 100 nm rows) | 2.1–2.4 |
| far setup per GB of far operator | 7.3 s/GB (run 1), 4.5 s/GB (run 2) | 1.6 s/GB (L = 4 µm), 1.1 s/GB (proxy) | 4.6, 3.9 |
| near matvec per GB of near field | 0.0206–0.0208 s/GB (runs 1–3) | 0.029–0.032 s/GB | 0.65–0.72 |
| power summary per unknown | 5.48–5.50·10⁻⁴ s (runs 1–3) | 2.16 / 2.92 / 2.95·10⁻⁴ s (L = 4 / 6 / 10.5 µm) | 1.86–2.55 |
| wall time, run 3 | 8.03 h | 3.2–7.1 h (extrapolated) | 1.1–2.5 |

The power-summary range uses all three desktop rows. The desktop post time is not proportional
to 2N at small L (2.16·10⁻⁴ s per unknown at L = 4 µm, 26 % below L = 6 and 10.5 µm, cause
untested; the L = 4 µm run had `load0` 1.50, a quiet machine), while the server's is constant to
0.3 % from 2 to 30 µm. Size against size the ratio is 2.54 at the small end (server 2 µm against
desktop 4 µm) and 1.86 at the large end (server 20 / 30 µm against the proxy).

Every compute-heavy phase (far matvec, near-field quadrature in the setup, the far-operator setup,
the far-field evaluation of the power summary) is 1.9–4.6 × slower on the server; only the sparse
near matvec is 1.4–1.5 × faster per GB. That the near matvec is memory-bandwidth bound is itself a
hypothesis (untested): two EPYC sockets (a platform with up to 12 DDR5 channels per socket; the
populated channels were not recorded) should have several times the bandwidth of the desktop's
2-channel DDR4, yet the near matvec is only 1.4–1.5 × faster per GB. A plausible reason for the
slow phases (hypothesis, untested): the far matvec is floating-point bound (README section 2: it streams only
≈ 5 GB/s of patterns at the proxy on the desktop; on the server 1.9 / 2.5 GB/s in runs 2 / 3),
and the server has 16 cores (Zen 5, up to 4.1 GHz) against the desktop's 12 cores / 24 threads
(Zen 3, up to 4.8 GHz). Core count and clock alone would, however, predict about equal speed,
not a factor 2, so other factors probably contribute; none was tested. SMT alone does not
distinguish the machines: both ran 2 OpenMP threads per physical core (server 32 threads on 16
cores, desktop 24 threads on 12 cores, see the Setup table "Machine"). What does differ: the
thread pinning (server `OMP_PLACES=threads`, `OMP_PROC_BIND=spread`; desktop unpinned, Windows
scheduler), the two sockets with interleaved memory and the cross-socket synchronisation of the
level sweeps (desktop: one socket), the all-core clock under load (not recorded on either), and
the compiler (GCC 14.3 against 16.1; neither build passes `-march`, so the server's AVX-512 is not
used). A cheap way to separate pinning / SMT and NUMA is to repeat run 1 (8 min) with
`--threads 16` and with `--numa none`.

**5. Power summary (measured; the explanation is a hypothesis).** R and R_coarse (half the
far-field nodes: 2° instead of 1°) differ by 0.0 % / 0.41 % / 2.1 % at L = 2 / 20 / 30 µm (proxy
0.01 %). With λ = 500 nm the beam's angular half-width λ / (π w₀) is 18.2° / 1.82° / 1.22° for
w₀ = L/4 = 0.5 / 5 / 7.5 µm (proxy 3.5°). The far field of the illuminated spot has structure on
that angular scale (the specular lobe and the speckle grain of the diffuse field), and the 2° grid
under-samples it once the half-width approaches the spacing. The R_coarse shift is therefore
attributed to the under-sampling of the speckle of the diffuse field (hypothesis, untested): the
diffuse field carries nearly all of R, while the coherent specular part is reduced by
exp(−(4πσ/λ)²) ≈ 1.8·10⁻³ in the Kirchhoff estimate, ≈ 2·10⁻³ of R or ≈ 7·10⁻⁴ absolute
(0.388 × 1.8·10⁻³), negligible against the 7.9·10⁻³ shift at L = 30 µm. The power
balance supports the 1° value: 1 − R − A = 3.3·10⁻³ / 4.2·10⁻⁴ / 6.9·10⁻⁴ with R, but 3.3·10⁻³ /
1.9·10⁻³ / −7.2·10⁻³ with R_coarse. The 1° grid itself has only ≈ 1 node per half-width at
L = 30 µm and is not verified; quantitative R at L ≥ 20 µm needs a finer `--far-field` spacing
(≈ 0.3–0.5°) with a convergence check, at 4–11 × the post-processing time. R = 0.376 / 0.363 /
0.367 against the desktop 0.365–0.372 (run 1 above, run 2 0.002 below, run 3 inside): each L is
its own surface realisation (the noise grid depends on L), and the smoke patch (w₀ = 0.5 µm, one
correlation length) is expected to fluctuate most. The post time scales with 2N on
the server (5.48–5.50·10⁻⁴ s per unknown) and is 1.86–1.87 × the README estimate
"≈ 324 s × 2N / 1.1·10⁶" (38 / 885 / 1 752 s), which is the desktop proxy's figure: the machine
factor of item 4 at the proxy size, not a wrong scaling.

**6. Extrapolation check (measured vs extrapolated).** The memory, the model and the iterations
are machine independent and can be compared directly; the times are desktop extrapolations, so
they are checked by their growth from run 2 to run 3 and by run 3 predicted from run 2 (the
calibration the run list planned):

| quantity | run 2 measured / stated | run 3 measured / stated | growth 20 → 30 µm measured / A / B | run 3 from run 2 × B growth | verdict |
|---|---|---|---|---|---|
| model | 284.7 / 285 GB | 822.9 / 823 GB | 2.89 | — | exact |
| peak | 250.4 / 209–250 GB (B), 219–246 (A) | 697.0 / 602–718 GB (B), 402–489 (A) | 2.78 / 1.84–1.99 / 2.87–2.88 | — | B held (run 2 0.4 GB, 0.2 %, above its upper end); A's upper end is 30 % below the measured peak at L = 30 µm, as the record expected |
| iterations | 95 / 81–117 | 95 / 81–117 | 1.00 | — | held |
| setup | 32.1 min / 8–15 min (B, desktop) | 79.3 min / 27–53 min (B, desktop) | 2.47 / 1.57–1.89 / 3.38–3.53 | 108–113 min | between A and B; B 1.36–1.43 × too high |
| matvec | 111.2 s / 42–62 s (desktop) | 207.7 s / 82–158 s (desktop) | 1.87 / 1.95–2.55 / 2.51–2.53 | 279–282 s | below both; B 1.34–1.36 × too high |
| solve | 2.99 h / 1.0–2.2 h (desktop) | 5.56 h / 2.2–5.6 h (desktop) | 1.86 / 2.00–2.69 / 2.50–2.55 | 7.5–7.6 h | below both; B 1.34–1.37 × too high (A 1.08–1.45 ×) |

The memory extrapolation (method B, as defined in "Extrapolation": peak = (0.72–0.86) × model +
Krylov, with model = the `--estimate-only` setup peak 1.5 near + 1.1 far + 1 GB from the estimated
near and far memory) held at both sizes; the power law A underestimates it once the leaf grows (156 →
234 nm). For the times the component model B assumed a cost per GB that stays constant or rises
with the octree depth, but on the server it fell from L = 20 to 30 µm: far matvec 0.540 →
0.402 s/GB (−26 %), near setup 25.4 → 15.1 s/GB (−40 %), far setup 4.47 → 4.11 s/GB (−8 %). With
larger leaves the memory (patterns, near field) grows faster than the work (hypothesis for why:
the leaf-level pattern and near-field storage grows with the leaf size, the translation work of
the upper levels does not). In raw times run 3 is above the desktop ranges for the setup
(79 min against 27–53 min) and the matvec (208 s against 82–158 s) by the machine factor of item
4, and its solve (5.56 h) falls inside the desktop range 2.2–5.6 h only because the smaller growth
compensates the slower machine. The extrapolation script
[`si_large_area_extrapolation.py`](../si_large_area_extrapolation.py) has no input for measured
points of another machine (it filters the desktop σ = 50 nm series and the `series=wp22d` rows);
its output is unchanged by the server rows (checked) and it was not modified.

**7. Fit script.** `python benchmarks/mlfmm_scaling_fit.py benchmarks/results/mlfmm_scaling_rows.txt`
parses the three rows as their own series (`... sigma_nm 100, waist 4, beam angular_spectrum, d0 3,
tol 0.001, lq 1, machine server`, separate from the `machine dev` σ = 100 nm series) and fits,
over 2N = 1.28·10⁵ … 5.95·10⁶ (3 sizes, 1 degree of freedom; γ ± standard error): setup
0.95 ± 0.06, matvec 1.09 ± 0.03, iterations 0.03 ± 0.00, **solve 1.12 ± 0.04**, solve per
iteration 1.09 ± 0.03, peak memory 1.05 ± 0.08. Caveats: three points spanning 6 to 8 levels and
leaves of 183 / 156 / 234 nm (the octree quantisation of section "Octree quantisation"), only one
point inside the DoD window 5·10⁴ … 10⁶ (the "DoD window" fit is the same three points because the
script applies only the lower bound), and run 3's matvec is a single apply (its solve and s/it are
unaffected). The value is consistent with the Si scaling DoD (γ ≤ 1.5) that the desktop series
already met (WP22b3: γ_solve 1.09 over 5.9·10⁴ … 1.02·10⁶), but it is not a DoD measurement.

Items 8–13 (WP22e part 2) are computed from the four run 4 `row.txt` lines, the desktop WP22b3 Ag
rows, the decimated server histories
[`mlfmm_scaling_ag_server_histories.txt`](mlfmm_scaling_ag_server_histories.txt) (every 50th
iteration plus the last; the per-iteration files and the memory samples stay off the repository)
and the desktop histories [`mlfmm_scaling_ag_histories.txt`](mlfmm_scaling_ag_histories.txt), with
`mlfmm_scaling_fit.py` and `mlfmm_scaling_projection.py` (commands under "How to reproduce").

**8. The Ag scaling DoD (measured; the window stated exactly).** The fit script parses the four
rows as their own series (`material ag, box 100.000000, fine 0, … lq 1, machine server`, separate
from the `machine dev` WP22b3 series) and fits, over 2N = 69 600 … 1 017 450 (4 sizes, all
`conv=1`, 2 degrees of freedom; γ ± standard error): **solve 1.37 ± 0.08**, iterations
0.29 ± 0.06, solve per iteration 1.07 ± 0.03, proxy it × (near + far matvec) 1.31 ± 0.03, matvec
1.02 ± 0.02 (near 1.18 ± 0.21, far 1.00 ± 0.05), setup 1.01 ± 0.04, peak memory 1.20 ± 0.10 (the
peak includes the Krylov basis, 36–50 % of it; the peak after the setup, `peak_setup_gb`, gives
1.14 ± 0.16 by the same least squares, not printed by the script). With 2 degrees of freedom the
95 % interval of γ_solve is 1.37 ± 4.30 × 0.085 = 1.00–1.73, below 2.0. γ = 2.0 from 4a would
allow 140 152 s at 4d, 5.8 × the measured 24 163 s, or ≈ 9 074 iterations with the 4d t(k) fit,
3.1 × the measured 2 920 (projection script, section 4).

The window actually covered is 2N = 6.96·10⁴ … 1.02·10⁶, not 5·10⁴ … 10⁶: the upper end is 1.7 %
above 10⁶, the lower end 1.39 × above 5·10⁴; in log 2N the series spans 1.16 of the window's 1.30
decades (90 %). The gap 5·10⁴ … 6.96·10⁴ holds no Ag row of any series on either machine: the
desktop's next smaller size is L = 3 µm at 2N = 4.32·10⁴, below the window (the desktop "DoD
window" fit also starts at 6.96·10⁴). Combining machines is not admissible for a time exponent:
the server / desktop solve ratio at equal L is 1.45–1.88 and not constant (item 9), so a mixed fit
would contain a machine step of that size; the fit script keeps the series apart for this reason.
An argument, not a measurement, why the gap cannot reverse the verdict: for even the two-point
slope from 5·10⁴ to 1.02·10⁶ to reach 2.0, the server solve at 2N = 5·10⁴ would have to take
≤ 58 s, 11 × less than at 6.96·10⁴ (656 s) for 1.39 × fewer unknowns; the machine-independent
desktop counts below the window (956 iterations at 4.32·10⁴, 1 394 at 6.96·10⁴) show no such drop.
**Statement: the Ag criterion γ ≤ 2.0 is met by measurement on one machine over 2N = 6.96·10⁴ …
1.02·10⁶ (γ_solve = 1.37 ± 0.08); the bottom 0.14 decade of the window, 5·10⁴ … 6.96·10⁴, is not
measured.**

Against the desktop: 1.40 ± 0.14 measured over 6.96·10⁴ … 4.2·10⁵ and 1.40–1.51 projected over
the whole window (`mlfmm_scaling.md`). The server value is at the low end. On the desktop scale
(estimated, not measured): the desktop's own t(k) fit of its capped L = 17.5 µm run
(t = 6.4 + 2.3893 k + 0.900 ns · 2N · k², projection script section 1 on the desktop file) gives
14 793 s at the measured 2 920 iterations, and the desktop rows at the four server sizes with this
point fit 1.44 ± 0.11. The difference to 1.37 is the falling machine ratio (1.88 at 4 µm, 1.45–1.63
at 11–17.5 µm): γ depends on the machine by ≈ 0.07 here, small against the margin to 2.0.

**9. Server against desktop (measured on both at L = 4 / 7 / 11 µm; at 17.5 µm the desktop solve
is the estimate of item 8).** Server / desktop:

| quantity | 4 µm | 7 µm | 11 µm | 17.5 µm |
|---|---:|---:|---:|---:|
| iterations | 1 394 = 1 394 | 1 518 = 1 518 | 2 272 = 2 272 | 2 920 (desktop capped at 1 450) |
| setup (near / far) | 2.99 (2.91 / 3.59) | 3.00 (2.95 / 3.39) | 3.04 (2.93 / 4.08) | 2.95 (2.84 / 3.42) |
| matvec (near / far) | 1.91 (0.70 / 2.37) | 2.31 (0.71 / 2.88) | 1.75 (0.72 / 2.79) | 2.10 (0.70 / 2.92) |
| CGS2 term c / 2N of t(k) [ns] | 1.769 / 0.919 = 1.92 | 1.030 / 0.890 = 1.16 | 1.134 / 1.033 = 1.10 | 1.143 / 0.900 = 1.27 |
| solve = s/it (equal iterations) | 656 / 348 = 1.88 | 1 839 / 993 = 1.85 | 6 902 / 4 748 = 1.45 | 24 163 / 14 793 (est.) = 1.63 |
| near, exact, far memory; model | 1.000 | 1.000 | 1.000 | 1.000 |

The iterations are not just close but equal, and so are the final monitored and true residuals
(`res` and `true_res` agree to all 6 printed digits at 4a–4c). The residual histories agree at
every common 50th iteration in all four printed digits of the desktop file, in all four
runs, 4d included up to the desktop cap (k = 1 450: 4.148·10⁻³ in both files): the
server 4d solve continues the capped desktop run. This fits the GMRES documentation (CGS2 sums its
block partial products in block order, independent of the thread count); whether the MLFMM apply
is bitwise equal on the two machines was not checked.

The Si factor of item 4 reappears in the operator, not in the orthogonalisation: the far matvec is
2.4–2.9 × slower (Si 2.5 ×), the sparse near matvec 0.70–0.72 × (Si 0.65–0.72 ×), the setup
2.95–3.04 × (Si 2–5 ×); the CGS2 term c, which the record attributes to streaming the basis
(24 · 2N / c = 21–23 GB/s on the server at L ≥ 7 µm against 23–27 GB/s on the desktop), is only
1.10–1.27 × slower there (1.92 × at 4 µm, cause untested). The Ag solve ratio therefore lies below
the matvec ratio (1.45–1.88 against 1.75–2.31) because 22–39 % of the server solve is
orthogonalisation. This is the measured decomposition; why CGS2 does not profit from the server's
two sockets and why the far matvec is slower are hypotheses as in item 4 (untested). The README's
expectation that the Si factor would not carry over to the Ag solve holds in part: it carries over
to the matvec share.

Load: `load0` was 0.11 for 4a and 26.58 / 26.79 for 4b / 4c, which started right after the
previous case of the same series call ended (its 32 threads were busy until its last seconds, the
power summary), so a high 1-min load average at their start is expected from the series itself
(inference). For 4d (23.85; its own call, the first Ag run) the cause is not recorded and no other
jobs were reported. No slowdown is visible in any run: the setup ratio of 4d (2.95) equals that of
4a (2.99, started at load 0.11); the t(k) = a + b k + c k² fits have rms 1.1 / 1.1 / 6.8 / 23.1 s,
≤ 0.17 % of the solve, and b within −2.3 … −7.0 % of the matvec (desktop −10 … +0.3 % at L ≥ 4 µm), i.e. no
load episode at the 50-iteration resolution. The in-run 1-min load samples (means 27–31) are
dominated by the run's own 32 threads, as on the desktop.

**10. Predictions against the measurement (README run 4 table; projection script).**

- Iterations: 4a–4c exactly the desktop counts (the README allowed "a few" iterations of
  difference). 4d: 2 920 inside the WP22b3 bracket 2 530–3 360, 15 % above the low (log-linear)
  end and 13 % below the high (power-law) end; the projection script, run on the server history
  truncated at the desktop cap 1 450 (`--back-tests "17.5:1450"`), gives the same 2 534–3 359. The
  scenario of the desktop record "γ_it = 0.80 sustained from 11 µm" (≈ 4 600 iterations at
  17.5 µm) did not occur (item 11).
- Memory: 4a–4c peaks "as measured on the desktop": 4.21 / 10.92 / 43.01 GB against 4.22 / 10.94 /
  43.01 GB. 4d: Krylov basis 47.55 GB inside the projected 41–55 GB, peak 94.26 GB inside 87–101
  GB. Model 4.48 / 9.09 / 39.47 / 64.23 GB = README 4.5 / 9.1 / 39.5 / 64 GB.
- Times: the README times were desktop references, explicitly not server predictions. Measured
  (README desktop reference): setup 39 s / 99 s / 4.3 min / 9.4 min (13 s / 33 s / 85 s / 3 min);
  timing matvecs 4 / 11 / 25 / 60 s (2 / 5 / 14 / 30 s); solve 656 s / 1 839 s / 1.92 h / 6.71 h
  (348 s / 993 s / 4 748 s / 3.3–5.1 h projected); post 38 / 101 / 230 / 554 s (~20 s / ~1 min /
  ~2 min / ~5 min); total 12.3 min / 34.3 min / 2.06 h / 7.05 h (6 min / 18 min / 1.4 h /
  3.4–5.2 h); the whole run 4 9.88 h of wall time (5.2–7.0 h). The 4d desktop-scale estimate of
  item 8 (14 793 s) lies inside the projected 11 900–18 400 s; the bracket times from the server
  history truncated at 1 450 are 20 152–30 254 s around the measured 24 163 s.
- Post: 5.40–5.45·10⁻⁴ s per unknown, constant to 1 % and equal to Si on the server
  (5.48–5.50·10⁻⁴ s); 1.85 × the README's desktop figure 324 s × 2N / 1.1·10⁶ at 4d (Si 1.86–1.87 ×).
- Exponent: projected (desktop) 1.40–1.51 over the window; measured (server) 1.37 ± 0.08 (item 8).

**11. Iteration growth (measured).** 1 394 → 1 518 → 2 272 → 2 920 at 2N = 6.96·10⁴ → 1.86·10⁵ →
4.22·10⁵ → 1.02·10⁶: γ_it = 0.29 ± 0.06, local 0.09 / 0.49 / 0.29 (projection script, section 5).
The steep intervals of the desktop (9 → 11 µm: 0.80) and of the server (7 → 11 µm: 0.49) were not
sustained to 17.5 µm; with one surface realisation per L the local rates scatter (desktop 5 µm
needed fewer iterations than 4 µm). What it implies, as measured here:

- the iterations, not the operator, set the Ag exponent: matvec 1.02, setup 1.01, solve per
  iteration 1.07, solve 1.37. The per-iteration time grows along each solve with the CGS2 term
  (4d: 5.06 s per iteration over the first 50 iterations, 11.57 s per iteration over the last 20), so the solve exceeds
  it × matvec by the orthogonalisation share s_orth = 22–39 %; the desktop record's estimate
  γ_solve ≈ γ_matvec + γ_it (1 + s_orth) = 1.02 + 0.29 × 1.32 ≈ 1.40 (s_orth ≈ 0.32, the mean)
  is consistent with the fitted 1.37;
- full GMRES stores the whole basis: (it + 1) 2N · 16 B grows with 2N^1.29 (fitted to `krylov_gb`)
  and is the largest memory item at 10⁶ unknowns (47.6 of 94.3 GB, near field 32.2, far 13.6);
- at 2 920 iterations a solve of 10⁶ unknowns takes 6.7 h on the server; fewer iterations (a
  stronger preconditioner than Jacobi, already named in `mlfmm_scaling.md`) would shorten time
  and memory together. Whether γ_it stays near 0.3 beyond 10⁶ is not measured.

**12. L = 4 µm: true residual 1.09·10⁻³ above tol 1e-3 (from the solver definitions).** With the
left Jacobi preconditioner M = diag(A), GMRES solves M⁻¹A x = M⁻¹b and stops on the monitored
residual |M⁻¹(b − A x)| / |M⁻¹b| (the Arnoldi estimate; `include/specklebem/solver/gmres.hpp`,
`PreconditionerSide::Left`); `true_res` is |b − A x| / |b|, computed after the solve with one extra
apply (`src/solver/gmres.cpp`). For r = b − A x the two differ by the weighting with the diagonal:
|r| / |b| = |M (M⁻¹r)| / |M (M⁻¹b)| lies between (min |a_ii| / max |a_ii|) and
(max |a_ii| / min |a_ii|) times the monitored value, depending on how the remaining residual is
distributed over rows of different diagonal magnitude (which rows dominate at L = 4 µm was not
examined). Measured true / monitored: 1.094 / 0.940 / 0.885 / 0.828 (4a–4d), identical on the
desktop at 4–11 µm; desktop L = 3 / 5 / 9 µm: 1.165 / 1.012 / 0.888. So 1.09·10⁻³ is the expected
behaviour of the stopping rule, the same on both machines, not a convergence failure (the
`mlfmm_scaling.md` setup already states true residuals ≤ 1.2·10⁻³ for Ag). It does not matter for
the scaling: every size stops on the same criterion, as in all desktop rows. Estimated effect of
a stop on the true residual instead (log-linear slope of the last 300 iterations as in the bracket,
true / monitored ratio assumed constant, from the decimated histories): +49 / −39 / −108 / −239
iterations, solve +4.8 / −3.2 / −6.3 / −11.3 %, γ_solve ≈ 1.30; the recorded 1.37 is the more
conservative value. For R and A it is irrelevant at this tolerance (item 13).

**13. R and A (box route, rough Ag: not quantitative).** R = 0.97342 / 0.97782 / 0.97776 / 0.97702;
R_coarse differs by ≤ 7·10⁻⁵ at 4a–4c and by 1.6·10⁻³ at 4d; A (from the currents) = 0.00287 /
0.02314 / 0.01770 / 0.01260; 1 − R − A = 2.4·10⁻² / −9.6·10⁻⁴ / 4.5·10⁻³ / 1.0·10⁻². They are
recorded, not interpreted: ADR 0006 (amendment 2026-10-12) reports Ag absorption only as A_ext,
never from the currents (a non-radiating current component spoils the local value by up to
−53 %), and the A column is that current value; the amendment 2026-10-12b (decision 3) and WP-V3
(surface waves on the finite block) rule out quantitative rough-Ag results on the closing-box
route, which ADR 0009 (PML) is to replace; and ADR 0006 asks GMRES tol ≤ 1e-5 for quantitative Ag
reflectance, against 1e-3 here. The spread of A across L (factor 8) and the open 1 − R − A are
compatible with the first point but are not analysed here. The desktop WP22b3 Ag rows have no
power summary, so there is no machine comparison of R and A.

Checks when recording (the Si checks are applied in items 1–7, the Ag checks in items 8–13):
iterations and R against the proxy (Si σ = 100 nm: 85–90 iterations, R = 0.365–0.372; R and A at tol 1e-3 are consistency checks, not quantitative reflectances);
(peak − Krylov) / model against 0.72–0.86; peak, setup and solve of runs 2, 3 and 5 against the
stated ranges above (the ratio server / desktop time from run 2 calibrates runs 3 and 5);
matvec scaling from run 2 to 3 against method B (run 3 with `matvecs=1`: a single apply). Ag
series 4a–4d: iterations against the desktop counts 1 394 / 1 518 / 2 272 (4a–4c; machine
independent up to rounding, a different thread count changes the summation order of the
reductions, so a few iterations of difference were expected; measured: identical, item 9, since the CGS2 sums do not depend on the thread count) and against the WP22b3 bracket
2 530–3 360 (4d, capped at 1 450 on the desktop); all four converged (`conv=1`), else note it;
the server solve times against the desktop references 348 / 993 / 4 748 s and 3.3–5.1 h (4d,
projected) only as a machine comparison (the Ag solve is bandwidth-bound, not calibrated by run
2); the series' own γ_solve closes the Ag scaling DoD by measurement on one machine. Their R and
A are informative only (the Ag A at normal incidence from the currents is not checked against
A_ext here).

## How to reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_mlfmm_scaling
# all rows used w0 = L / 4; since WP-B3 the Si default is L / 6, so the waist is explicit
W="--waist-factor 4 --allow-wide-beam"
# proxy and the sigma = 100 nm points (one case per process, quiet machine)
$B --material si --L 10.5e-6 --sigma 100e-9 --box-mesh-size 50e-9 --beam angular-spectrum $W --history --far-field 1
$B --material si --L 4e-6 --sigma 100e-9 --box-mesh-size 50e-9 --beam angular-spectrum $W --history --far-field 1  # also 6e-6
# estimates (seconds each): uniform box and graded box with the local leaf rule
$B --material si --L 30e-6 --sigma 100e-9 --box-mesh-size 50e-9 --beam angular-spectrum $W --estimate-only
$B --material si --L 30e-6 --sigma 100e-9 --leaf-radius-quantile 0.8 --beam angular-spectrum $W --estimate-only
python benchmarks/si_large_area_extrapolation.py
# server (Linux): scripts/server_run_kit/README.md
# server results from the committed data (no runs needed): fits of every series incl. machine=server
python benchmarks/mlfmm_scaling_fit.py benchmarks/results/mlfmm_scaling_rows.txt
# server Ag series: t(k) fits, bracket back-test at the desktop cap, exponent, margin, CGS2 shares
python benchmarks/mlfmm_scaling_projection.py benchmarks/results/mlfmm_scaling_ag_server_histories.txt \
    --back-tests "17.5:1450" --capped ""
python benchmarks/mlfmm_scaling_projection.py   # desktop file: the c / 2N and t(k) of item 9
```
