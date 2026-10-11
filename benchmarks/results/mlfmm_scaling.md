# MLFMM scaling on Si and Ag rough surfaces (WP22b1, WP22b3; Phase 4 DoD)

Phase 4 acceptance criterion of `docs/01_project_plan.md`: "Scaling exponent γ (time per solve vs
the number of unknowns 2N) ≤ 1.5 for Si rough surfaces, ≤ 2.0 for Ag, over 2N = 5·10⁴ … 10⁶ (here
the count of unknowns, not N = interior edges of docs/06; WP22b1)". (The criterion first read
"vs N … over N = 5·10⁴ … 10⁶"; the coordinator decided for this record to read N as the number of
unknowns 2N, and docs/01 now says so.) Executable `benchmarks/mlfmm_scaling.cpp`
(`specklebem_mlfmm_scaling`, `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`), fits
`benchmarks/mlfmm_scaling_fit.py`, raw rows in [`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt)
(`python benchmarks/mlfmm_scaling_fit.py benchmarks/results/mlfmm_scaling_rows.txt` reproduces the
fits: one series per material, box, fine band, restart, h, waist factor, beam, d₀, tol and leaf
radius quantile;
repeated runs of a size merged by the per-quantity minimum; iterations and solve times only from
converged solves; slopes with standard errors, over the full range and over the DoD window
2N ≥ 5·10⁴).

**Verdict (WP22b3, CGS2 GMRES, rigorous beam, quiet machine): Si (uniform 50 nm box): DoD met
by measurement over 2N = 5·10⁴ … 10⁶. Ag: γ_solve = 1.40 ± 0.14 measured over 2N = 7·10⁴ …
4.2·10⁵; over the full window projected (1.40–1.51), not measured; the Ag part of the DoD
remains open by measurement. The converged L = 17.5 µm solve (~5 h, ~100 GB) is scheduled for
the 1 TB server run kit (WP22d).**

- **Si (uniform 50 nm box): γ_solve = 1.09 ± 0.04 over 2N = 5.9·10⁴ … 1.02·10⁶** (7 sizes, all
  solved, `--box-mesh-size 50e-9`; largest case L = 10 µm, 68 iterations, 954 s, 74.8 GB
  peak). Iterations flat (64–85, γ = −0.07), matvec γ = 1.17 ± 0.02, setup 0.83, peak memory
  0.95. Below 1.5. The graded 100 nm box with the global leaf rule stays memory-limited (70.9
  GB at 2N = 5.2·10⁵); with the local leaf rule q = 0.8 52.1 GB (section "Si, local leaf rule").
- **Ag: γ_solve = 1.40 ± 0.14 measured over 2N = 7.0·10⁴ … 4.2·10⁵** (5 solved sizes in the
  DoD window; 1.35 ± 0.10 with 2N = 4.3·10⁴), below 2.0. Setup, matvec and memory measured to
  2N = 1.02·10⁶: γ = 0.99 ± 0.02 / 0.99 ± 0.04 / 1.06 ± 0.08. The two largest solves, capped at
  2 100 and 1 450 iterations (6 542 s and 5 411 s), project to 9 480–10 360 s (2N = 7.6·10⁵) and
  11 900–18 400 s (2N = 1.02·10⁶); with these points γ_solve = 1.40 ± 0.07 … 1.51 ± 0.08 over
  the whole window (projected, not measured; it assumes that the iterations do not keep growing
  at the local rate of the last measured interval, see the Ag section). γ = 2.0 at 10⁶ would
  take ≈ 7 800 iterations, 2.3 × the high extrapolation. To turn the projection into a
  measurement: the converged L = 17.5 µm run (≈ 5 h, ≈ 100 GB; WP22d server run kit), or a
  better preconditioner than Jacobi (fewer iterations: shorter and smaller).
- **CGS2 vs the WP22b1 MGS solves**: 1.16–1.21 × faster at L = 4–7 µm, 1.71 × at L = 9 µm;
  orthogonalisation share 30–44 %, not monotone in L (was 42–66 %).
- **Local leaf rule (q = 0.8) measured at Si L = 4 / 8 µm**: peak −34 % / −26 %, setup +32 % /
  +20 %, matvec and iterations unchanged. Measured; default unchanged (coordinator decision
  2026-10-11): `leaf_radius_quantile` stays 1.0 (opt-in); for Si the uniform box is the
  recommended configuration (details in its section).
- The WP22b1 rows (MGS GMRES, paraxial beam, shared machine) are kept below as an older series;
  the fit script separates them by the `beam` field.

## WP22b3 series (CGS2 GMRES, rigorous beam, quiet machine)

Same surfaces, boxes, media, operator and tolerance as WP22b1 (setup table below), with three
changes: **GMRES orthogonalisation CGS2** (WP-G1 default: two parallel classical Gram–Schmidt
passes, instead of the serial MGS loop of WP22b1), **`--beam angular-spectrum`**
(`AngularSpectrumBeam`, rigorous; controlled ball 1.01 × the largest vertex distance) and
**exclusive machine time**: one case per process, each started only after the 1-min load average
had been below 5 for two checks 30 s apart (Ag L = 3 µm: started by hand after a quiet period;
`load0` = the value at the start, 1.7–4.3 for every
row; the in-run `/proc/loadavg` samples, 16–170 with run means 24–84, are dominated by the run's
own 24 threads and are not a foreign-load measure). Check of the quiet state: Ag L = 3 µm setup /
matvec 10.0 s / 0.118 s against 9.3 s / 0.120 s in the quietest WP22b1 run; the GMRES timestamps
of every Ag run (every 50 iterations) follow t(k) = a + b k + c k² (rms deviation 0.1–12 s), with
b within 11 % of the measured matvec for L ≥ 4 µm (−10 … 0 %; L = 3 µm +27 %, where the work per
iteration beyond the matvec weighs most) and c / 2N = 0.89–1.03 ns in all runs (CGS2 streams
24 · 2N · k² bytes in total: 23–27 GB/s, the WP-G1 bandwidth figure), i.e. no load episodes
(per-run a, b, c: section "Ag", reproducible with `benchmarks/mlfmm_scaling_projection.py`).
Rows: block "WP22b3" in
[`mlfmm_scaling_rows.txt`](mlfmm_scaling_rows.txt) (`gmres=cgs2`, `load0=` annotations);
2026-10-11, main 5dab69f, `win-release`, 24 threads.

### Si, uniform box (`--box-mesh-size 50e-9`, fine band)

Box cells = top cells (r_max 64–67 nm, measured in every row), so the leaf rule gives 125–188 nm leaves.

| L [µm] | 2N | levels | leaf [nm] | setup [s] | matvec [s] | near mv [s] | far mv [s] | it | solve [s] | s/it | near [GB] | far [GB] | peak [GB] | model [GB] | load0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 040 | 6 | 178 | 17.1 | 0.475 | 0.036 | 0.456 | 85 | 42 | 0.50 | 1.26 | 3.36 | 4.8 | 6.6 | 4.1 |
| 2 | 127 680 | 6 | 180 | 31.1 | 1.10 | 0.073 | 1.04 | 72 | 83 | 1.15 | 2.41 | 7.44 | 10.2 | 12.8 | 3.4 |
| 3 | 205 920 | 6 | 181 | 46.3 | 2.02 | 0.114 | 1.96 | 73 | 153 | 2.10 | 3.91 | 12.23 | 16.6 | 20.3 | 4.3 |
| 4 | 293 760 | 6 | 181 | 58.9 | 2.88 | 0.158 | 2.73 | 71 | 203 | 2.85 | 5.36 | 17.27 | 23.2 | 28.0 | 3.8 |
| 6 | 498 240 | 6 | 188 | 100.4 | 5.28 | 0.295 | 4.93 | 64 | 342 | 5.35 | 9.90 | 30.18 | 40.9 | 49.1 | 2.9 |
| 8 | 741 120 | 7 | 125 | 115.7 | 9.02 | 0.185 | 8.80 | 69 | 638 | 9.25 | 6.31 | 40.72 | 48.3 | 55.3 | 2.8 |
| 10 | 1 022 400 | 7 | 156 | 198.0 | 13.72 | 0.410 | 13.26 | 68 | 954 | 14.04 | 13.58 | 59.46 | 74.8 | 86.8 | 2.8 |

Peak = peak working set at the end of the run (≈ near + far + Krylov); model = the
`--estimate-only` setup peak (conservative by +15 … +36 %). True residuals 0.96–1.00·10⁻³.

| quantity (7 sizes, 2N = 5.9·10⁴ … 1.02·10⁶ = the DoD window) | γ ± se |
|---|---:|
| **time per solve** | **1.09 ± 0.04** |
| iterations | −0.07 ± 0.02 |
| solve time per iteration | 1.16 ± 0.02 |
| matvec (full / near / far) | 1.17 ± 0.02 / 0.79 ± 0.11 / 1.18 ± 0.02 |
| proxy it × (near + far matvec) | 1.09 ± 0.04 |
| setup (total / near / far) | 0.83 ± 0.04 / 0.92 ± 0.04 / 0.74 ± 0.07 |
| memory (near / far / peak) | 0.78 ± 0.10 / 1.00 ± 0.01 / 0.95 ± 0.03 |

- **Iterations are flat (64–85)**, so the solve follows the matvec. The matvec is 93–98 % far
  part (the Si interior expansion), which grows slightly faster than N (γ 1.18): leaf patterns are
  linear in N, the translations per level grow with the number of boxes and the depth (6 → 7
  levels at L = 8 µm).
- **Local slopes of the solve time**: L = 6 → 8 µm 1.57 (6 → 7 levels), L = 8 → 10 µm 1.25
  (the earlier intervals 0.79–1.29). The step at the extra level is followed by a lower slope
  again, so the global 1.09 hides no rising trend towards 10⁶.
- **The uniform box is cheaper than the graded 100 nm box** despite 40–44 % more unknowns
  (same beam, same session, global rule): L = 4 µm 203 s / 23 GB vs 396 s / 45 GB, L = 8 µm
  638 s / 48 GB (2N = 7.4·10⁵) vs 945 s / 71 GB (2N = 5.2·10⁵). The graded box needs 96–104
  iterations instead of 69–71 (ICTF without preconditioner; the 100 nm cells and the 2:1
  transition rows) and its coarse cells force 250–362 nm leaves.
- WP22b1's Si iterations at the same L (paraxial beam) equal the angular-spectrum ones (L = 4 /
  8 µm: 104 / 96 both), so the beam does not change the conditioning; the iteration gap is the
  box.

### Si, local leaf rule vs global rule (100 nm box, fine band, same session)

| L [µm] | 2N | q | levels | leaf [nm] | elevated | setup [s] (near / far) | matvec [s] | it | solve [s] | near [GB] | far [GB] | peak after setup [GB] | peak [GB] | model [GB] | load0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 209 760 | 0.8 | 6 | 181 | 8 081 (3.9 %) | 145.4 (108.4 / 34.9) | 3.67 | 104 | 400 | 6.40 | 22.47 | 25.2 | 29.4 | 35.3 | 3.4 |
| 4 | 209 760 | 1 | 5 | 362 | 0 | 110.0 (76.3 / 31.6) | 3.69 | 104 | 396 | 14.28 | 30.01 | 42.6 | 44.8 | 55.4 | 2.9 |
| 8 | 515 520 | 0.8 | 7 | 125 | 27 040 (5.2 %) | 212.3 (150.2 / 59.5) | 9.94 | 96 | 979 | 7.47 | 43.51 | 38.4 | 52.1 | 60.1 | 2.8 |
| 8 | 515 520 | 1 | 6 | 250 | 0 | 177.3 (113.1 / 61.5) | 9.72 | 96 | 945 | 16.16 | 53.60 | 63.0 | 70.9 | 84.2 | 4.0 |

(elevated: share of the 2N unknowns, i.e. 7.7 % / 10.5 % of the basis functions.) Same
iterations and true residuals (9.68·10⁻⁴ / 9.86·10⁻⁴ local, 9.68·10⁻⁴ / 9.85·10⁻⁴ global).

- **Memory: −34 % (L = 4 µm) and −26 % (L = 8 µm) peak**, −41 % / −39 % after the setup: near
  field −55 % / −54 %, far operator −25 % / −19 % (smaller Si leaf patterns, but elevated patterns and
  one more level of far tables).
- **Time: setup +32 % / +20 %** (the elevated rows of the near field and the elevated
  patterns), **matvec −1 % / +2 %, solve +1 % / +4 %**: the far part dominates both and is
  nearly unchanged.
- The WP21L estimates (35 / 60 GB setup peak model) hold: the model is conservative by +20 % /
  +15 % against the measured peak.

**Default quantile: q = 0.8 measured; default unchanged (coordinator decision 2026-10-11).**
q = 0.8 cuts the peak memory of the graded Si boxes by 26–34 % for a 20–32 % longer setup and an
unchanged solve, and the same iterations; the d₀ = 3 accuracy against the dense matrix (WP21L,
3.4·10⁻⁴ vs 2.1·10⁻⁴ global) stays inside the 10⁻³ budget. The default `leaf_radius_quantile`
nevertheless stays 1.0 (the local rule remains opt-in), because:

- for Si MLFMM runs the uniform box (`box_mesh_size` = the top-face mesh size, previous section)
  beats both leaf rules on the graded box, so the uniform box is the recommended Si
  configuration;
- the local rule stays the memory option for graded boxes (`--leaf-radius-quantile 0.8`,
  `mlfmm=dict(leaf_radius_quantile=0.8)`, d₀ ≤ 3);
- the q = 0.8 evidence covers two sizes (L = 4 / 8 µm) of one surface family;
- on box-dominated geometries it costs more memory: where the elevated functions are the
  majority (WP21L tall box, 54 % elevated) the near field grows instead (0.99 vs 0.24 GB).

### Ag (100 nm box, no fine band)

| L [µm] | 2N | levels | leaf [nm] | setup [s] | matvec [s] | near mv [s] | far mv [s] | it | solve [s] | s/it | WP22b1 MGS solve [s] (it) | near [GB] | far [GB] | peak [GB] | load0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 43 200 | 4 | 375 | 10.0 | 0.118 | 0.069 | 0.078 | 956 | 179 | 0.19 | 325 (1 004) | 2.39 | 0.66 | 3.8 | 2.5 |
| 4 | 69 600 | 5 | 250 | 13.0 | 0.166 | 0.050 | 0.119 | 1 394 | 348 | 0.25 | 405 (1 386) | 1.68 | 0.87 | 4.2 | 3.9 |
| 5 | 103 200 | 5 | 313 | 21.1 | 0.278 | 0.119 | 0.163 | 1 275 | 507 | 0.40 | 591 (1 284) | 4.04 | 1.48 | 7.7 | 4.0 |
| 7 | 186 480 | 6 | 219 | 33.1 | 0.410 | 0.106 | 0.302 | 1 518 | 993 | 0.65 | 1 203 (1 518) | 3.62 | 2.39 | 10.9 | 1.7 |
| 9 | 293 760 | 6 | 281 | 55.0 | 0.690 | 0.274 | 0.411 | 1 699 | 1 948 | 1.15 | 3 326 (1 697) | 9.55 | 3.86 | 21.7 | 2.3 |
| 11 | 422 400 | 6 | 344 | 85.5 | 1.168 | 0.592 | 0.579 | 2 272 | 4 748 | 2.09 | — | 20.85 | 6.54 | 43.0 | 4.0 |
| 15 | 756 000 | 7 | 234 | 135.8 | 1.765 | 0.542 | 1.217 | cap 2 100 | (6 542) | — | — | 17.43 | 9.53 | 53.6 | 3.0 |
| 17.5 | 1 017 450 | 7 | 273 | 191.7 | 2.384 | 0.947 | 1.486 | cap 1 450 | (5 411) | — | — | 32.16 | 13.63 | 70.2 | 2.0 |

ICTF + left Jacobi; true residuals of the converged solves 0.88–1.17·10⁻³ (the monitored
residual is the preconditioned one). **L = 15 and 17.5 µm: runs capped by `--max-iter`** so
that a run fits the 2-hour limit per background command (a full solve needs ~2.6–5 h, below);
monitored residual at the cap 1.89·10⁻³ (true 1.56·10⁻³) and 4.15·10⁻³ (true 3.27·10⁻³); their
rows (`conv=0`) enter only the setup, matvec and memory fits; peak includes the Krylov basis of
the capped run (25.4 / 23.6 GB). The beam changes the iterations by ≤ 5 % (L = 3 µm: 956 vs
1 004).

| quantity | γ ± se, full range | γ ± se, DoD window 2N ≥ 5·10⁴ |
|---|---:|---:|
| **time per solve, measured** (2N ≤ 4.2·10⁵) | 1.35 ± 0.10 (6 sizes) | **1.40 ± 0.14** (5 sizes) |
| time per solve incl. projected L = 15 / 17.5 µm (low / high bracket) | 1.37 ± 0.06 / 1.46 ± 0.06 | **1.40 ± 0.07 / 1.51 ± 0.08** |
| iterations (2N ≤ 4.2·10⁵) | 0.31 ± 0.06 | 0.27 ± 0.08 |
| solve time per iteration | 1.04 ± 0.07 | 1.13 ± 0.08 |
| proxy it × (near + far matvec) | 1.22 ± 0.09 | 1.28 ± 0.13 |
| matvec (full / near / far), 8 sizes to 1.02·10⁶ | 0.97 ± 0.03 / 0.92 ± 0.12 / 0.94 ± 0.03 | 0.99 ± 0.04 / 1.04 ± 0.14 / 0.95 ± 0.03 |
| setup, to 1.02·10⁶ | 0.96 ± 0.03 | 0.99 ± 0.02 |
| memory (near / far / peak), to 1.02·10⁶ | 0.91 ± 0.13 / 0.98 ± 0.03 / 1.00 ± 0.07 | 1.04 ± 0.15 / 1.01 ± 0.03 / 1.06 ± 0.08 |

- **CGS2 effect** (same surfaces, quiet machine, iterations within 5 %): solve 1.16–1.21 ×
  faster at L = 4–7 µm, 1.71 × at L = 9 µm (where MGS was 66 % of the solve); the
  orthogonalisation share (solve − it × matvec) / solve is now 30–44 %, not monotone in L
  (L = 3 / 4 / 5 / 7 / 9 / 11 µm: 37 / 33 / 30 / 37 / 40 / 44 %), instead of 42–66 %. (The
  L = 3 µm factor 1.8 compares with a WP22b1 run that was itself load-affected in its solve.)
  The orthogonalisation still grows as k² · 2N, so the solve exponent stays above the matvec
  one: with γ_matvec ≈ 1, γ_solve ≈ 1 + γ_it + s_orth γ_it = 1 + 0.27 + 0.4 · 0.27 ≈ 1.38
  (s_orth = orthogonalisation share).
- **Local slope of the last measured interval**: from L = 9 to 11 µm (2N = 2.9·10⁵ → 4.2·10⁵)
  the solve goes 1 948 → 4 748 s, a local γ_solve of 2.45, and the iterations 1 699 → 2 272, a
  local γ_it of 0.80. This is iteration scatter, not a trend we can resolve: there is one surface
  realisation per L, and the counts are non-monotone (L = 5 µm needs fewer iterations than
  L = 4 µm: 1 275 vs 1 394; local γ_it −0.23 there, 0.25–0.29 between 5, 7 and 9 µm). The
  projections below assume that the iteration growth does not continue at that local rate:
  γ_it = 0.80 sustained from L = 11 µm would give ≈ 3 600 iterations at L = 15 µm and ≈ 4 600 at
  L = 17.5 µm, above both high brackets (2 848 / 3 360); the latter would still be below the
  ≈ 7 800 iterations that γ = 2.0 allows (margin bullet below).
- **Projection beyond 4.2·10⁵.** For the capped runs the iterations to 10⁻³ are bracketed by
  two extrapolations of the residual history over the last 300 iterations (log-linear: low;
  power law r ∝ k^p: high), and the time by the run's own fit t(k) = a + b k + c k². The bracket
  was checked on the converged runs truncated at a cap: L = 11 µm capped at 1 400 / 1 600 / 1 800
  iterations gives 2 107–2 480 / 2 178–2 394 / 2 274–2 406 iterations and 4 106–5 169 /
  4 340–4 964 / 4 656–5 053 s around the true 2 272 iterations and 4 748 s (the low value at the
  1 800 cap misses by 2 iterations); L = 9 µm capped at
  1 000 / 1 200 gives 1 512–1 853 / 1 586–1 746 iterations and 1 662–2 216 / 1 777–2 036 s
  around 1 699 and 1 948 s. Results: **L = 15 µm 2 690–2 848
  iterations, 9 480–10 360 s (2.6–2.9 h)**; **L = 17.5 µm (cap at 1 450 only, wider bracket)
  2 530–3 360 iterations, 11 900–18 400 s (3.3–5.1 h)**, Krylov basis 41–55 GB (peak ≈ 87–101
  GB). These are projections, not measurements.
- **Margin to the criterion**: with the measured cost t(k) of the L = 17.5 µm run, γ = 2.0
  over 6.96·10⁴ → 1.02·10⁶ (348 s × 14.6²) would allow 74 500 s, i.e. ≈ 7 800 iterations,
  2.3 × the high extrapolation (and more than the 128 GB machine could hold as a full-GMRES
  basis).
- **Reproducibility.** The GMRES histories behind the projection (k, wall time, monitored
  residual every 50 iterations, all eight WP22b3 Ag runs) are committed in
  [`mlfmm_scaling_ag_histories.txt`](mlfmm_scaling_ag_histories.txt);
  `python benchmarks/mlfmm_scaling_projection.py` re-derives from them the per-run time fits,
  the back-test, the two brackets, the exponents with the projected points, the margin, the
  orthogonalisation shares and the local slopes quoted in this section and in the verdict.
  Not committed: the full run logs and the in-run load samples (only the figures quoted above).
  Time fits t(k) = a + b k + c k² over all logged samples:

  | L [µm] | 2N | a [s] | b [s] | matvec [s] | b / matvec − 1 | c / 2N [ns] | rms [s] |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 3 | 43 200 | −0.4 | 0.1496 | 0.1182 | +26.6 % | 0.920 | 0.1 |
  | 4 | 69 600 | 0.6 | 0.1596 | 0.1664 | −4.1 % | 0.919 | 0.3 |
  | 5 | 103 200 | 0.3 | 0.2792 | 0.2784 | +0.3 % | 0.888 | 0.3 |
  | 7 | 186 480 | 2.5 | 0.3978 | 0.4103 | −3.1 % | 0.890 | 1.2 |
  | 9 | 293 760 | 0.7 | 0.6799 | 0.6899 | −1.5 % | 0.928 | 2.4 |
  | 11 | 422 400 | 30.5 | 1.0725 | 1.1683 | −8.2 % | 1.033 | 11.9 |
  | 15 (cap 2 100) | 756 000 | 42.0 | 1.5829 | 1.7651 | −10.3 % | 0.946 | 11.8 |
  | 17.5 (cap 1 450) | 1 017 450 | 6.4 | 2.3893 | 2.3843 | +0.2 % | 0.900 | 3.5 |

  The L = 15 / 17.5 µm rows are the fits used for the projected times (the truncated fits of
  the back-test are recomputed per cap by the script).
- **No converged Ag solve beyond 2N = 4.2·10⁵ was possible within the 2-hour limit per run**:
  L = 12–14 µm (2N = 5.0–6.6·10⁵) fall on 6-level octrees with 375–440 nm leaves (near field
  30–54 GB, `--estimate-only`), so they are not cheaper per unknown than L = 15 µm and would
  need ≥ 1.7 h at the measured costs; not run.

**Verdict (WP22b1, older GMRES and beam; kept for reference): Phase 4 scaling DoD partially
demonstrated, still open.**

- **Si: γ_solve = 0.73 ± 0.07 over 2N = 4.3·10⁴ … 5.2·10⁵** (6 sizes, single sweep as measured;
  0.80 ± 0.09 over the DoD part 9.3·10⁴ … 5.2·10⁵). With noise-reduced times γ_solve ≈ 0.88:
  0.88 ± 0.07 with the minimum of repeated runs (the clean first L = 1 µm run took 111 s, the
  sweep's rerun 180 s), 0.88 ± 0.06 for the proxy iterations × (near + far) matvec. Below 1.5 over
  this range. **Limited by memory with the 100 nm graded box**: 79 GB at 2N = 5.2·10⁵ (before
  d00317a); see "Memory limit and the uniform box" for the route to 2N ≈ 10⁶.
- **Ag: γ_solve ≈ 1.2 ± 0.2 over 5 solved sizes 2N = 4.3·10⁴ … 2.9·10⁵** (1.20 ± 0.18; 1.43 ±
  0.19 from 7·10⁴, i.e. over the DoD part; the last interval 1.9·10⁵ → 2.9·10⁵ alone gives 2.2
  and is noise-sensitive: one loaded run moves it by ±0.5). Below 2.0 over this range. The
  L = 3 µm point is the quiet merged-tree rerun (325 s, deterministic 1 004 iterations); the
  first sweep's 511 s and 619 s and the fix rerun's 1 145 s (machine loaded by a parallel WP22c
  validation run) are kept as repeats in the rows. Setup and matvec measured to 2N = 1.02·10⁶
  (L = 17.5 µm): γ ≈ 0.9 (0.89 / 0.87 single sweep; 0.96 ± 0.03 / 0.95 ± 0.04 with the repeat
  minima).
- **Not shown over 2N = 5·10⁴ … 10⁶.** Remaining: the Ag solves at 2N = 4.2·10⁵ … 1.02·10⁶
  (L = 11, 15, 17.5 µm; 1.7–5 h each with the current serial GMRES orthogonalisation) after WP-G1
  (fast GMRES orthogonalisation, in progress), and Si at 2N ≥ 6·10⁵ via the uniform box (memory
  estimates below), both with `--beam angular-spectrum`.
- **Automatic box rule without the λ₁/5 cap (`--box-mesh-size uncapped`, the "auto" series of the
  first record)**: unusable with the MLFMM beyond L ≈ 1.5 µm. Its 200–400 nm cells have support
  radii of 230–490 nm, the leaf rule (r_max / a ≤ 0.6) raises the leaves to λ/0.7 … λ/0.35 and
  the near field approaches the dense matrix (Si L = 4 µm: 307 GB near-field estimate for 2N =
  1.8·10⁵). Ag L = 4 µm: 36 % fewer unknowns than the 100 nm box but 10 × the near field, 9 × the
  peak memory and 3.8 × the solve time (1 208 vs 1 386 iterations). Since WP-B1 the default
  automatic rule is capped at λ₁/5 (`--box-mesh-size auto`): at h = 50 nm and L ≥ 1 µm it builds
  the same mesh as the explicit 100 nm (checked by estimate at Si L = 4 µm).

## Setup (WP22b1; WP22b3 differs in GMRES, beam and machine load, see its section)

| Parameter | Value |
|---|---|
| Surface | Gaussian height map (`generate_gaussian_height_map`), σ = 50 nm, Lc = 500 nm, mesh h = 50 nm, seed 1, L × L patch (no crop; every L its own surface) |
| Closing box (ADR 0006) | `rough_surface_box_params` (vacuum, λ = 500 nm): depth `default_box_depth` (Si 5.65 µm, Ag 2 µm), λ₁ = 500 nm; coarse spacing `box_mesh_size` = **100 nm = λ₁/5** explicit (M = 1); **Si fine band** 3δ + 3σ = 3.54 µm; Ag no fine band. Second series `uncapped` = `min(depth/2, L/8, 10 h)` without the λ₁/5 cap (recorded before WP-B1 as `auto`) |
| Media | λ = 500 nm, vacuum R1; Si ε_r = 18.478 − 0.606j, Ag ε_r = −9.794 − 0.313j (exp(+jωt)) |
| Excitation | paraxial `excitation::GaussianBeam` (all WP22b1 runs), normal incidence (+z), focus at the origin, E along x, w₀ = L/4 (`check_beam_waist` enforced; `--allow-wide-beam` for deliberate edge studies). WP22b3: `--beam angular-spectrum` (`AngularSpectrumBeam`, controlled ball 1.01 × the largest vertex distance) |
| Operator | `Simulation` with compression `"mlfmm"`, d₀ = 3, automatic leaf rule (leaf ≥ max(λ/4, r_max/0.6)) and exact-part budget (2 × near) |
| Solver | `formulation::recommend`: Si ICTF without preconditioner, Ag ICTF + left Jacobi; full GMRES (no restart; WP22b1: modified Gram–Schmidt with a selective second pass, WP22b3: CGS2), x₀ = 0, tolerance 1e-3 on the monitored residual (true residual reported; Ag ≤ 1.2e-3) |
| Timings | setup = `Simulation::assemble` (octree, far setup incl. the Ag exact part, near field, rhs, Jacobi); matvec = median of 5 applies of the `MlfmmOperator` (and of its near and far parts), interleaved after one untimed round; solve = GMRES wall time; s/it = solve / iterations (the operator figure per iteration) |
| Memory | near field, far operator (leaf patterns, translators, workspace; exact parts separately), Krylov basis (it + 1) 2N 16 B, peak working set after the setup and at the end. **All peaks in the tables were measured before d00317a** (near-field matrix taken without a copy), which removed about one near field from the setup peak |
| Machine | Windows 11, 24 cores, 128 GB, MSYS2 UCRT64 GCC 16.1, OpenBLAS, `win-release`, 24 OpenMP threads, 2026-10-10. Shared machine: other agents ran test suites and dense studies concurrently, so single timings carry +30 … +250 % load noise (it only ever adds time: Ag L = 3 µm solve 325 / 511 / 619 / 1 145 s for identical iterations; Si L = 7 µm full apply 21 s vs near + far 8.1 s). Repeats and the per-quantity minimum reduce it |

One case per process, sizes in increasing order, each started only when its estimated setup peak
(`--estimate-only`, model below) plus 10 GB fitted the available memory.

## WP22b1: Si (100 nm box, fine band; older series: MGS GMRES, paraxial beam, shared machine)

| L [µm] | 2N | levels | leaf [nm] | setup [s] | near [s] | far [s] | matvec [s] | near mv [s] | far mv [s] | it | solve [s] | s/it | near [GB] | far [GB] | peak [GB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 960 | 5 | 357 | 34.4 | 22.9 | 11.5 | 0.69 | 0.13 | 0.55 | 154 | 180 (111) | 1.17 (0.72) | 4.11 | 5.97 | 14.0 |
| 2 | 92 880 | 5 | 360 | 79.4 | 51.4 | 28.0 | 1.64 | 0.27 | 1.91 | 106 | 275 | 2.59 | 7.19 | 13.19 | 27.0 |
| 3 | 148 320 | 5 | 361 | 97.9 | 66.5 | 31.4 | 3.05 | 0.35 | 2.60 | 103 | 325 | 3.15 | 10.54 | 21.38 | 41.2 |
| 4 | 209 760 | 5 | 362 | 264.3 | 163.9 | 100.4 | 5.22 | 0.53 | 4.62 | 104 | 589 | 5.66 | 14.28 | 30.01 | 56.9 |
| 7 | 433 440 | 6 | 219 | 184.5 | 130.2 | 54.3 | 21.0 ‡ | 0.32 | 7.82 | 90 | 882 | 9.80 | 10.60 | 41.88 | 56.6 |
| 8 | 515 520 | 6 | 250 | 204.1 | 114.0 | 90.0 | 10.65 | 0.47 | 10.17 | 96 | 1 044 | 10.88 | 16.16 | 53.60 | 79.1 |

(111 s): the first L = 1 µm run (same code, same 154 iterations; the table row is the sweep's
rerun). ‡ sequential medians (the interleaved timing came later): load outlier, near + far =
8.1 s; excluded from the matvec fit (`exclude=matvec_s` in the rows; rerunning it with
`--no-solve` needs 53 GB near + far (63 GB model), far above the 8 GB allowed alongside the WP22c
runs).

Exponents γ ± standard error (least squares in log-log):

| quantity | single sweep, 6 sizes | repeat minima, 6 sizes | DoD window 2N ≥ 5·10⁴ (5 sizes, repeat minima) |
|---|---:|---:|---:|
| **time per solve** | **0.73 ± 0.07** | 0.88 ± 0.07 | 0.80 ± 0.09 |
| proxy it × (near + far matvec) | — | 0.88 ± 0.06 | 0.83 ± 0.10 |
| iterations | −0.18 ± 0.05 | −0.18 ± 0.05 | −0.08 ± 0.03 |
| solve time per iteration | 0.90 | 1.06 ± 0.09 | 0.89 ± 0.08 |
| matvec (full / near / far / near + far) | 1.28 (with ‡) | 1.13 ± 0.07 (5 sizes) / 0.43 ± 0.17 / 1.12 ± 0.08 / 1.05 ± 0.08 | 1.08 ± 0.12 / 0.17 ± 0.19 / 0.97 ± 0.07 / 0.91 ± 0.08 |
| setup (total / near / far) | 0.71 / 0.68 / 0.76 | 0.71 ± 0.19 / 0.69 ± 0.19 / 0.76 ± 0.21 | 0.53 ± 0.28 / 0.49 ± 0.27 / 0.58 ± 0.33 |
| memory (near / far / peak) | 0.47 / 0.85 / 0.64 | 0.47 ± 0.12 / 0.85 ± 0.06 / 0.64 ± 0.09 | 0.32 ± 0.17 / 0.76 ± 0.07 / 0.52 ± 0.12 |

- **Iterations fall from 154 to 90–106** and then stay flat. At L = 1 µm the waist w₀ = L/4 =
  250 nm = λ/2 is far from paraxial (divergence λ/(π w₀) ≈ 0.64 rad), so a large part of the beam
  lights the side walls of the box; and the closing box dominates the mesh at small L (box / top
  triangles 16.9 at L = 1 µm, 8.7 at 2 µm, 4.5 at 4 µm, 2.4 at 8 µm), so the smallest patches are
  the worst conditioned. The solve time therefore follows the time per iteration (γ ≈ 0.9–1.1),
  the operator figure; the GMRES orthogonalisation is ≤ 10 % at ≈ 100 iterations.
- **The far part dominates the matvec** (85–95 %): the Si interior expansion at leaves of 0.4–0.7 λ₁
  = 3–6 λ₂ needs leaf sampling orders L = 45–60 (2(L+1)² ≈ 4 000–7 400 directions per basis
  function), applied to every basis function in every matvec and stored as leaf patterns (far
  memory 6 → 54 GB; leaf patterns 40–61 % and the far operator 43–74 % of the pre-d00317a peak).
- **Sub-linear setup and memory are an artefact of the leaf size**: the root edge is set by
  max(L, depth) and halved, so the leaf jumps from 357–375 nm (5 levels, L ≤ 6 µm) to 219–250 nm
  (6 levels, L = 7–8 µm), which cuts the near field per unknown 2.6 × and the pattern size per
  unknown ~2 ×. At fixed leaf size (L = 1 … 4 µm) memory grows with γ ≈ 1.0 and setup with γ ≈
  1.2.

## WP22b1: Ag (100 nm box; older series: MGS GMRES, paraxial beam, shared machine)

| L [µm] | 2N | levels | leaf [nm] | setup [s] | near [s] | far [s] | matvec [s] | near mv [s] | far mv [s] | it | solve [s] | s/it | near [GB] | far [GB] | peak [GB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 43 200 | 4 | 375 | 13.7 | 7.9 | 0.4 | 0.18 | 0.07 | 0.10 | 1 004 | 325 † | 0.32 | 2.39 | 0.66 | 5.4 |
| 4 | 69 600 | 5 | 250 | 12.9 | 9.1 | 0.7 | 0.17 | 0.05 | 0.09 | 1 386 | 405 | 0.29 | 1.68 | 0.87 | 4.2 |
| 5 | 103 200 | 5 | 313 | 20.7 | 15.6 | 1.0 | 0.27 | 0.12 | 0.16 | 1 284 | 591 | 0.46 | 4.04 | 1.48 | 9.5 |
| 7 | 186 480 | 6 | 219 | 32.3 | 22.7 | 2.2 | 0.40 | 0.11 | 0.29 | 1 518 | 1 203 | 0.79 | 3.62 | 2.39 | 10.9 |
| 9 | 293 760 | 6 | 281 | 52.9 | 39.3 | 2.3 | 0.67 | 0.27 | 0.38 | 1 697 | 3 326 | 1.96 | 9.55 | 3.86 | 22.7 |
| 11 | 422 400 | 6 | 344 | 87.0 | 67.4 | 2.9 | 1.17 | 0.60 | 0.56 | — | — | — | 20.85 | 6.54 | 47.8 |
| 15 | 756 000 | 7 | 234 | 127.8 | 90.8 | 7.2 | 1.58 | 0.51 | 1.03 | — | — | — | 17.43 | 9.53 | 43.8 |
| 17.5 | 1 017 450 | 7 | 273 | 184.7 | 136.3 | 8.3 | 2.32 | 0.92 | 1.39 | — | — | — | 32.16 | 13.63 | 76.8 |

† minimum of four runs with identical iterations and residuals: 325 s (reviewer, merged tree,
quiet machine), 511 s (sweep), 619 s (first run), 1 145 s (fix rerun on the merged tree during a
WP22c validation run; near setup 21.5 s instead of 7.2 s); the other columns are the sweep's.
Rows without iterations: `--no-solve` (setup and matvec only; a full-GMRES solve would exceed the
2-hour budget per run). Exact part (included in "far [s]" and the peak): ≤ 0.83 GB and ≤ 3.3 s at
every size. True residuals 0.89e-3 … 1.17e-3 (left Jacobi: the monitored residual is the
preconditioned one).

Exponents γ ± standard error:

| quantity | single sweep | repeat minima | DoD window 2N ≥ 5·10⁴ (repeat minima) |
|---|---:|---:|---:|
| **time per solve** (5 sizes, 4.3·10⁴ … 2.9·10⁵) | 1.01 ± 0.28 (with 511 s) | **1.20 ± 0.18** | 1.43 ± 0.19 (4 sizes) |
| last interval 1.9·10⁵ → 2.9·10⁵ | 2.24 | 2.24 | |
| proxy it × (near + far matvec) | — | 1.12 ± 0.07 | 1.19 ± 0.09 |
| iterations | 0.24 ± 0.06 | 0.24 ± 0.06 | 0.16 ± 0.07 |
| solve time per iteration | 0.77 | 0.96 ± 0.21 | 1.27 ± 0.16 |
| matvec (full / near / far), 8 sizes to 1.02·10⁶ | 0.87 / 0.90 / 0.89 | 0.95 ± 0.04 / 0.91 ± 0.13 / 1.00 ± 0.03 | 0.97 ± 0.05 / 1.04 ± 0.14 / 0.99 ± 0.04 |
| setup (total / near / far), 8 sizes | 0.89 / 0.94 / 0.96 | 0.96 ± 0.03 / 0.95 ± 0.04 / 0.96 ± 0.05 | 0.98 ± 0.03 / 0.99 ± 0.05 / 0.92 ± 0.07 |
| memory (near / far / peak), 8 sizes | 0.91 / 0.98 / 0.92 | 0.91 ± 0.13 / 0.98 ± 0.03 / 0.92 ± 0.11 | 1.04 ± 0.15 / 1.01 ± 0.03 / 1.03 ± 0.12 |

(The repeat minima include the extra `--no-solve` runs at L = 3, 7, 9 µm; the quiet L = 3 µm
setup, 9.3 s instead of 13.7 s, raises the setup and matvec slopes from 0.89 / 0.87 to 0.96 /
0.95, an example of the size of the load noise.)

Projection to 2N = 1.02·10⁶ with the current GMRES (iterations 1 697 (2N/2.9·10⁵)^0.24 ≈ 2 290,
matvec 2.32 s measured, orthogonalisation at the 9.3 GB/s effective bandwidth of the L = 9 µm
run): 5 300 s of matvecs + 13 700 s of orthogonalisation ≈ 5.3 h, i.e. γ ≈ 1.15 (from 4.3·10⁴) …
1.44 (from 7.0·10⁴) over the DoD range (projection, not a measurement). The local slope rises
because the orthogonalisation grows as k² N (asymptotically γ → 1 + 2 · 0.24 ≈ 1.5).

- **Solve time = iterations × matvec + orthogonalisation.** Measured split (orthogonalisation =
  solve − it × matvec): L = 4 µm 169 s of 405 s (42 %), 5 µm 248 / 591 s (42 %), 7 µm 594 / 1 203 s
  (49 %), 9 µm 2 185 / 3 326 s (66 %). The it × matvec part grows with γ ≈ 1.1 (matvec 0.95,
  iterations 0.1–0.2), the modified Gram–Schmidt part (k²/2 · 2N complex dot products and axpys,
  plus the selective second pass when the norm drops below 0.7 of its value, serial Eigen loops in
  `solver::gmres`, ~10–20 GB/s) with γ ≈ 1.8 = 2 γ_it + 1. Over the solved range the measured γ
  stays below 2.0; at 10⁶ the orthogonalisation would be ~70 % of a ~5 h solve.
- **Iterations** (ICTF + Jacobi, tol 1e-3): 1 004, 1 386, 1 284, 1 518, 1 697 for L = 3, 4, 5, 7,
  9 µm (γ = 0.24) — slowly growing and not monotone (each L is a different surface). GMRES(100)
  was tried at L = 3 µm: 2 844 iterations in 601 s, against 619 s for full GMRES (1 004
  iterations) in the same loaded period (the full-GMRES runs took 325–1 145 s depending on the
  load), 0.07 GB instead of 0.69 GB Krylov memory — no gain at this size; the restarts cost 2.8 ×
  the iterations.
- **The Ag exact part is negligible** (≤ 0.8 GB, ≤ 3 s): with leaves of 219–406 nm the near field
  already covers almost all pairs within the decay reach x*/α = 0.30 µm (Ag WP22a sphere with λ/4
  leaves: 6.8 GB). The interior region R2 is truncated or exact on every far level; R1 expands.
- **Near and far share the matvec about equally**; the near field (1.7–32 GB) is the largest memory
  item; the leaf rule raises the leaf to 219–406 nm here as well, so the near field per unknown
  varies 2.6 × with the root quantisation (L = 4 µm: 24 kB, L = 11 µm: 49 kB).

## Automatic box rule without the λ₁/5 cap

| Case | coarse spacing (M) | r_max [nm] | leaf [nm] | 2N | near [GB] | far [GB] | peak [GB] | it | solve [s] | 100 nm box at the same L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Si L = 1 µm | L/8 = 125 nm (M = 1) | 116 | 357 | 42 960 | 4.1 | 6.0 | 14.0 | — | — | identical mesh (M = 1 either way) |
| Si L = 1.5 µm † | L/8 = 188 nm (M = 2) | 228 | 714 | 61 104 | 22.5 | 26.3 | 71.0 | 210 | 736 | 67 410 unknowns, 5.2 GB near |
| Si L = 2 µm | 250 nm (M = 2) | 231 | 720 | 83 760 | 34.9 (est.) | 35.9 (est.) | 110 (est.) | not run | | memory: estimate + 10 GB > available |
| Si L = 4 µm | 400 nm (M = 3) | 447 | 1 450 | 182 880 | 307 (est.) | | 768 (est.) | not run | | 14.3 GB near |
| Ag L = 2 µm † | 250 nm (M = 2) | 246 | 528 | 14 640 | 1.45 | 0.41 | 3.3 | 2 495 | 662 | |
| Ag L = 4 µm | 400 nm (M = 3) | 485 | 1 000 | 44 400 | 17.2 | 3.3 | 37.8 | 1 208 | 1 547 | 69 600 unknowns, 1.7 GB near, 4.2 GB peak, 1 386 it, 405 s |

† overlapped with builds or test runs. The uncapped rule makes the box cheap in unknowns (Ag
L = 4 µm: 44 400 vs 69 600) but not in cost: the single largest RWG support decides the leaf edge
for the whole octree. (Ag L = 2 µm has no 100 nm counterpart; its 2 495 iterations are not
comparable. The Ag L = 4 µm uncapped run has a true residual of 1.8e-3 at the monitored 1e-3.)
The Si L = 2 and 4 µm estimates are those of the first record (old peak model).

## Why the leaves are large

The ADR 0008 leaf rule uses the **global** largest support radius r_max: leaf ≥ r_max / 0.6. On the
top face (h = 50 nm) r_max ≈ 60 nm and λ/4 = 125 nm leaves would be allowed; the 100 nm box cells
(2:1 transition cells, relaxation band) have r_max = 116–141 nm, so the floor becomes 195–235 nm,
and the power-of-two subdivision of the root (edge ≈ max(L, depth)) rounds it up to 219–406 nm.
Consequences per unknown compared with λ/4 leaves: near field 1.8–3.3 × (surface boxes), Si
interior leaf patterns 3–10 × ((k₂ a)²), both linear in N. This, not the multilevel algorithm,
limits the Si range **with 100 nm box cells** to 2N ≈ 5·10⁵ on 128 GB.

## Memory limit and the uniform box

With 100 nm box cells the next Si sizes do not fit: 2N = 6.1·10⁵ needs ~116 GB, 2N ≈ 8.6·10⁵ …
9.8·10⁵ 240–310 GB (first-record estimates, old model). The uniform box (`--box-mesh-size 50e-9`:
box cells = top cells, r_max 64–67 nm, leaves 156–181 nm) has more unknowns but much smaller
leaves; `--estimate-only` on the merged tree (estimates only, not measured):

| case | 2N | leaf [nm] | near [GB] | leaf patterns [GB] | far tables [GB] | setup peak model [GB] | old model [GB] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Si L = 4 µm, uniform | 293 760 | 181 | 5.4 | 14.0 | 3.2 | 28.0 | 27 |
| Si L = 10 µm, uniform | 1 022 400 | 156 | 13.6 | 45.2 | 14.3 | 86.8 | 78 |
| Ag L = 10 µm, uniform | 672 000 | 156 | 9.1 | 4.8 | 1.1 | 28.1 (incl. 3.4 GB exact part estimate) | 31 |

(Old model = 2 near + 1.1 patterns + 2 exact + 1 GB, the reviewer's figures.) **The uniform-box
Si series is the route to 2N ≈ 10⁶ available today**: Si L = 10 µm reaches 1.02·10⁶ unknowns at
~73 GB measured-model memory (near + far), i.e. the guard needs ≈ 97 GB available, a quiet
machine. Its cost per unknown at fixed L is unknown (more unknowns, more iterations possible:
box/top 3.3 at L = 10 µm); the local leaf rule (proposal 1) would avoid that cost by keeping the
graded box with λ/4 leaves. **Measured in WP22b3** (section "Si, uniform box"): L = 10 µm 74.8 GB
peak (model 86.8 GB), 68 iterations, 954 s; fewer iterations than the graded box (64–85 vs
96–104), so the uniform box is also faster per patch.

## Local leaf rule (WP21L, ADR 0008 amendment 2026-10-11)

Proposal 1 below is implemented: `MlfmmParams::leaf_radius_quantile` q takes the leaf from the
radius quantile r_q, and every basis function with r > ratio(d₀) a_leaf lives on its home level
(patterns at that level's sampling, near iff the ancestors on the coarser home level touch).

- **Default q = 1** (the global WP21 rule, bitwise unchanged): the local rule is opt-in
  (`mlfmm=dict(leaf_radius_quantile=0.8)`, `--leaf-radius-quantile 0.8`); the WP22b1 Si boxes
  need q ≈ 0.8 (below). WP22b3 measured q = 0.8 at L = 4 / 8 µm; the default stays 1
  (coordinator decision 2026-10-11, section "Si, local leaf rule vs global rule").
- **d₀ ≤ 3 only**: q < 1 with `accuracy_digits` > 3 raises `std::invalid_argument`
  (`kLocalLeafRuleMaxDigits`), because the λ/2 leaves it permits at d₀ = 5 miss 1e-5 (accuracy
  table below: 5.6e-5 Si / 6.4e-5 Ag). No fallback to the global rule.
- **Elevated ratio ρ_e = ratio(d₀)/2** (`kElevatedSupportRatioFactor`), a deviation from the
  amendment as written: with the leaf ratio above the leaf the level order exceeds k r_min and the
  translators amplify the interpolation error of the children's fields (graded box 1.5 × 2 µm,
  Si, d₀ = 3, q = 0.5: matvec **5.0e-2 with ρ, 1.5e-4 with ρ/2**). Hence an elevated function sits
  at least two levels above the leaf (ρ_e a_{D−1} = ρ a_D).

`--estimate-only` (model now with per-level radii, leaf patterns of the leaf functions and
elevated patterns at their home level), Si, 100 nm box, fine band, d₀ = 3:

| case | q | levels | leaf [nm] | elevated (home level) | near [GB] | patterns [GB] (elevated) | far tables [GB] | setup peak model [GB] |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Si L = 4 µm, 2N = 209 760 | 1 (global, default) | 5 | 362 | 0 | 14.3 | 27.9 | 2.1 | 55.4 |
| | 0.99 | 5 | 362 | 0 | 14.3 | 27.9 | 2.1 | 55.4 |
| | 0.9 / 0.8 / 0.5 | 6 | 181 | 8 081 = 7.7 % (3) | 6.4 | 18.4 (5.4) | 4.1 | 35.3 |
| Si L = 8 µm, 2N = 515 520 | 1 / 0.99 / 0.9 | 6 | 250 | 0 | 16.2 | 46.0 | 7.6 | 84.2 |
| | 0.8 / 0.5 | 7 | 125 | 27 040 = 10.5 % (4) | 7.5 | 30.0 (11.0) | 13.5 | 60.1 |

**q = 0.99 (the first default) changes nothing for these boxes**: the coarse 100 nm cells (r = 112–117 nm)
are 7.7–10.5 % of the basis functions (the fine band holds most box cells), so r_q is a coarse-cell
radius unless q is below ~0.9. With q = 0.8 the setup peak model drops by 36 % (L = 4 µm) and 29 %
(L = 8 µm): near field −55 %, patterns −34 %, far tables ×2 (one more level). Measured (Si L = 1 µm,
2N = 42 960, ICTF, same session, after d00317a):

| q | levels | leaf [nm] | elevated | setup [s] (near / far) | matvec [s] | it | solve [s] | near [GB] | far [GB] | peak [GB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 357 | 0 | 35.8 (21.9 / 13.8) | 0.73 | 154 | 114 | 4.11 | 5.97 | 10.3 |
| 0.8 | 6 | 178 | 1 793 (8.3 %) | 47.8 (29.6 / 18.1) | 0.74 | 154 | 119 | 1.69 | 4.38 | 6.3 |

Same iterations and true residual (9.984e-4), −39 % peak memory, +34 % setup (the elevated rows
of the near field and the level-3 Si patterns), equal matvec. Where the box functions are the
majority (tall test box 1 × 1 × 4 µm, 54 % elevated) the local rule costs more memory than the
global one (near 0.99 vs 0.24 GB), since every elevated function is near to the 27 boxes around a
box four times the leaf edge.

Accuracy against the dense matrix (graded box 1 × 1 × 4 µm, 2N = 13 200, PMCHWT; tests/validation_large):
d₀ = 3 local / global: Si 3.4e-4 / 2.1e-4, Ag 2.5e-4 / 2.0e-4. **d₀ = 5 local missed 1e-5**
(Si 5.6e-5, Ag 6.4e-5; global 4.5e-6 / 5.2e-6): the error is the leaf–leaf vacuum part at the
λ/2 leaves of the WP21 rule a_min(5) (5.3e-5 / 6.1e-5; the part involving elevated functions is
6.8e-7 / 1.0e-6). The global rule passes only because r_max forces λ leaves; at λ/2 the far blocks
carry ~4e-4 (WP19b) and the far part is ~8 % of |Z x| on this box. Therefore the local rule is
rejected for d₀ > 3 (coordinator decision 2026-10-11; the validation-large cases check d₀ = 3 local
and global and d₀ = 5 global, the rejection is a unit test). Open: a_min(5) ≥ λ or a binding
lossless block check, which would allow the local rule at d₀ = 5. The default q is decided:
WP22b3 measured q = 0.8 at L = 4 / 8 µm (section "Si, local leaf rule vs global rule" above);
default unchanged at q = 1 (coordinator decision 2026-10-11: the uniform box is the recommended
Si configuration, the local rule the opt-in memory option for graded boxes).

## Memory model (`--estimate-only`, the run guard)

- **Far memory**: leaf patterns (N × 2(L+1)² directions × 2 components × 16 B per expanding
  region) plus the far tables of every expansion level (translators of the distinct interaction
  offsets, one apply workspace 2 × 4 fields × boxes × directions × 16 B, phase shifts), with the
  orders of `leaf_sampling` / `search_truncation_order` as in `MlfmmFarOperator` (block check not
  modelled). Matches the measured far memory **within 0.1 % in all 17 recorded cases** (6 Si,
  8 Ag, 3 uncapped). The first model counted the leaf patterns only, and the version used for
  the uncapped Si L = 1.5 µm run skipped the expanding Si interior (25.8 GB of patterns at leaf
  order 109): −32 % there.
- **Setup peak** = 1.5 near + 1.1 (patterns + far tables) + 2 exact + 1 GB (after d00317a).
  Measured on the merged tree, the peak after the setup is near + far within 1 % (Ag L = 3 µm
  3.05 GB, Ag L = 4 / 5 µm 2.50 / 5.44 GB, Ag uncapped L = 2 µm 1.87 GB, Si L = 0.6 µm
  6.61 GB; `PROW` lines), so the model is conservative by +46 … +94 % at these small sizes (the
  1 GB and 0.5 near margins) and by an estimated +20 … +40 % at the largest recorded sizes.
- **First model** (2 near + 1.1 patterns + 2 exact + 1 GB, calibrated before d00317a): against
  the pre-d00317a peaks of the tables +2 … +23 % (main series), +3 % / +31 % (Ag uncapped
  L = 4 / 2 µm) and −32 % (Si uncapped L = 1.5 µm); "within 5 %" in the first record was wrong.

## What would extend the range (proposals, not applied)

1. **Local leaf rule** (implemented in WP21L, opt-in, d₀ ≤ 3, section "Local leaf rule" above;
   needs q ≈ 0.8 for these boxes): apply r_max / a ≤ 0.6 per box, e.g. by
   letting the few coarse box cells live on a coarser level (non-uniform depth, ADR 0008 change),
   or by splitting coarse box triangles so that box supports stay ≤ the top-face supports (r_max
   ≈ 75 nm, box_mesh_size ≈ 60 nm, ~+40 % unknowns at L = 10 µm). With λ/4 leaves the Si 10⁶ case
   drops from ~240–310 GB to an estimated 40–60 GB, without the extra unknowns of the uniform box.
2. **Si leaf patterns**: compute the leaf radiation patterns on the fly per matvec or store them in
   single precision (accuracy to be checked against d₀ = 3), halving or removing the dominant
   memory item.
3. **GMRES orthogonalisation** (Ag; WP-G1, done: CGS2 default, measured in WP22b3 at 1.2–1.7 ×
   shorter Ag solves): classical Gram–Schmidt with
   reorthogonalisation (CGS2) as BLAS-2 products over a contiguous Krylov basis (threaded zgemv),
   instead of the serial modified Gram–Schmidt loop with its selective second pass, would cut the
   42–66 % orthogonalisation share several-fold. A better preconditioner than Jacobi (near-field
   ILU / SAI; docs/01 risk table: H-LU, Phase 8) would cut the 1 000–1 700 Ag iterations, which
   drive both the time and the Krylov memory (8 GB at 2.9·10⁵, ~35 GB projected at 10⁶).
4. **Box rule**: keep the coarse spacing ≤ λ₁/5 (WP-B1 default); the uncapped automatic rule must
   not be used with the MLFMM until (1) exists.

## Pending

- Ag converged solves at 2N = 7.6·10⁵ and 1.02·10⁶ (L = 15, 17.5 µm): ≈ 2.6–2.9 h and 3.3–5.1 h
  (projection of the capped WP22b3 runs), ≈ 87–101 GB at 10⁶ with the full-GMRES basis; the
  converged L = 17.5 µm solve is scheduled for the 1 TB server run kit (WP22d, coordinator
  decision 2026-10-11). Until then the Ag part of the DoD is open by measurement. A stronger
  preconditioner (fewer iterations) would also shorten it.
- Default `leaf_radius_quantile`: decided, stays 1.0 (opt-in local rule; coordinator decision
  2026-10-11, WP22b3 section "Si, local leaf rule vs global rule").

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_mlfmm_scaling
# WP22b3 (CGS2 is the GMRES default; one case per process on a quiet machine)
$B --material ag --L 11e-6 --beam angular-spectrum --max-iter 6000      # Ag: L = 3 4 5 7 9 11 um
$B --material ag --L 15e-6 --beam angular-spectrum --max-iter 2100      # capped (17.5 um: 1450)
$B --material si --L 10e-6 --box-mesh-size 50e-9 --beam angular-spectrum   # Si: L = 1 2 3 4 6 8 10
$B --material si --L 8e-6 --leaf-radius-quantile 0.8 --beam angular-spectrum  # local rule (and q = 1)
# WP22b1
$B --material si --L 8e-6 --estimate-only           # 2N, octree, memory estimates (seconds)
$B --material si --L 4e-6 > si_L4.log               # Si sweep: L = 1 2 3 4 7 8 um
$B --material ag --L 9e-6 --max-iter 6000 > ag_L9.log   # Ag sweep: L = 3 4 5 7 9 um
$B --material ag --L 17.5e-6 --no-solve > ag_L17p5.log  # setup + matvec only: L = 11 15 17.5 um
$B --material ag --L 4e-6 --box-mesh-size uncapped   # automatic box rule without the cap
$B --material si --L 10e-6 --box-mesh-size 50e-9 --estimate-only   # uniform box
python benchmarks/mlfmm_scaling_fit.py si_L*.log     # tables and exponents
# from the committed data (no runs needed)
python benchmarks/mlfmm_scaling_fit.py benchmarks/results/mlfmm_scaling_rows.txt   # all fits
python benchmarks/mlfmm_scaling_projection.py   # Ag time fits, back-test, brackets, projected γ
```
