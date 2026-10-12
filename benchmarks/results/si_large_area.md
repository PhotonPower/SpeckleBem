# Large-area Si: proxy on 128 GB and the 30 × 30 µm² server run (WP22d)

Phase 4 target of `docs/01_project_plan.md`: "Si 30×30 µm², σ = 100 nm, λ/10 mesh (≈ 2.2 M
unknowns) solves on a 256 GB node". Lead decision 2026-10-11: on the 128 GB development machine
this is shown by a **proxy** (the largest feasible Si patch with the same settings, measured, and
extrapolated); the full case is run by the project lead on a 1 TB Linux server with the run kit
[`scripts/server_run_kit/`](../../scripts/server_run_kit/README.md), together with a server Ag
series (L = 4 / 7 / 11 / 17.5 µm) whose own γ_solve closes the Ag part of the scaling DoD by
measurement on one machine (coordinator decision of the WP22d review).

**Summary**

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
- **L = 30 µm, σ = 100 nm (extrapolated, not measured)**: 2N = 5.95·10⁶ (uniform box; the box
  adds 175 % to the top face), octree leaf 234 nm, `--estimate-only` model 823 GB; extrapolated
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

## Server results (template; filled in when the data come back)

Runs of `scripts/server_run_kit/run_large_area.sh`, one per row; the raw `row.txt` lines go to
[`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt) (block "WP22d server"). A `row.txt` line is the
benchmark's `ROW` line (the fields below) plus the annotations `gmres=cgs2 machine=server
threads= places= numa= matvecs= vmhwm_gb= rss_max_gb= load0= wall_s=`. Runs 3, 3b and 5 use
`--matvecs 1` (`matvecs=1`): their `matvec_s`, `near_mv_s` and `far_mv_s` are single applies
after one untimed round, not the median of 5 as in every desktop row and in run 2; compare
matvec times between machines with run 2. A `post_error=1` annotation means the power summary
failed after the solve (no R / A; the solve fields are valid).

**Server** (from `system.txt` and the lead's note):

| field | value |
|---|---|
| CPU model | |
| sockets × cores per socket (threads per core) | |
| memory (total; MemAvailable at the start of run 3) | |
| NUMA nodes; policy used (`numa=` annotation: interleave / none) | |
| OpenMP threads, `OMP_PLACES`, `OMP_PROC_BIND` | |
| OS, kernel; transparent huge pages | |
| compiler (version), build type, flags | |
| BLAS / LAPACK (`BLAS_LIBRARIES` from `CMakeCache.txt`) | |
| batch system, cgroup memory limit | |
| commit (`git rev-parse HEAD`), date | |
| other jobs on the node during the runs | |

**Runs, sizes and times** (ROW fields `N2 levels leaf_nm rmax_nm elevated setup_s near_s far_s
matvec_s near_mv_s far_mv_s it conv true_res solve_s`; annotations `load0 wall_s`):

| run | material | L [µm] | σ [nm] | box, q | 2N | levels | leaf [nm] | r_max [nm] | elevated | setup [s] (near / far) | matvec [s] (near / far) | it | conv | true res | solve [s] | s/it | load0 | wall [s] |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 smoke | Si | 2 | 100 | uniform 50 nm, 1 | | | | | | | | | | | | | | |
| 2 | Si | 20 | 100 | uniform 50 nm, 1 | | | | | | | | | | | | | | |
| 3 target | Si | 30 | 100 | uniform 50 nm, 1 | | | | | | | | | | | | | | |
| 3b (alt.) | Si | 30 | 100 | graded 100 nm, 0.8 | | | | | | | | | | | | | | |
| 4a | Ag | 4 | 50 | graded 100 nm, 1 | | | | | | | | | | | | | | |
| 4b | Ag | 7 | 50 | graded 100 nm, 1 | | | | | | | | | | | | | | |
| 4c | Ag | 11 | 50 | graded 100 nm, 1 | | | | | | | | | | | | | | |
| 4d | Ag | 17.5 | 50 | graded 100 nm, 1 | | | | | | | | | | | | | | |
| 5 (opt.) | Si | 32 | 100 | uniform 50 nm, 0.8 | | | | | | | | | | | | | | |

**Memory and power** (ROW fields `near_gb exact_gb far_gb krylov_gb peak_setup_gb peak_gb
peak_est_gb R R_coarse A post_s`; annotations `vmhwm_gb rss_max_gb`):

| run | near [GB] | exact [GB] | far [GB] | Krylov [GB] | peak after setup [GB] | peak [GB] (getrusage) | VmHWM [GB] | model [GB] | (peak − Krylov) / model | R (R_coarse) | A | 1 − R − A | post [s] | vs. estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1 smoke | | | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | | | |
| 3 target | | | | | | | | | | | | | | |
| 3b (alt.) | | | | | | | | | | | | | | |
| 4a | | | | | | | | | | | | | | |
| 4b | | | | | | | | | | | | | | |
| 4c | | | | | | | | | | | | | | |
| 4d | | | | | | | | | | | | | | |
| 5 (opt.) | | | | | | | | | | | | | | |

**Server Ag series** (runs 4a–4d): γ_solve of the `machine=server` rows
(`benchmarks/mlfmm_scaling_fit.py`) over 2N = 7.0·10⁴ … 1.02·10⁶: ___ ± ___ (DoD: ≤ 2.0; desktop
WP22b3 series: see `mlfmm_scaling.md`).

Checks when recording: iterations and R against the proxy (Si σ = 100 nm: 85–90 iterations,
R = 0.365–0.372; R and A at tol 1e-3 are consistency checks, not quantitative reflectances);
(peak − Krylov) / model against 0.72–0.86; peak, setup and solve of runs 2, 3 and 5 against the
stated ranges above (the ratio server / desktop time from run 2 calibrates runs 3 and 5);
matvec scaling from run 2 to 3 against method B (run 3 with `matvecs=1`: a single apply). Ag
series 4a–4d: iterations against the desktop counts 1 394 / 1 518 / 2 272 (4a–4c; machine
independent up to rounding, a different thread count changes the summation order of the
reductions, so a few iterations of difference are expected) and against the WP22b3 bracket
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
```
