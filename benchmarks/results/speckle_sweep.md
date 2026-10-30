# Phase 6 speckle sweep on rough Si (WP-S2)

Data for the Phase 6 definition of done (`docs/01_project_plan.md`, docs/05 rows 42–44): speckle at
z = −1 mm from Gaussian rough Si, σ = 50 … 250 nm, two seeds each. Driver
`benchmarks/speckle_sweep.py` (one case per process, `specklebem.speckle_sweep.run_case`), queue
`benchmarks/speckle_sweep_queue.sh`, statistics `benchmarks/speckle_sweep_stats.py` (WP-S1
harness, section "Analysis"), calibration on synthetic speckle
`benchmarks/speckle_sweep_calibration.py`; ROW lines in `speckle_sweep_rows.txt`. The DoD ctest
asserting rows 42–44 is WP-S2b.

Hardware: 12-core desktop, 128 GB, win-release (UCRT64 GCC, OpenBLAS), all logical threads,
shared with one other worker for most cases (timings are upper bounds for an idle machine).

**Status (2026-10-26): all ten cases solved (one restarted after a machine reset, "Run log") and
evaluated with the WP-S1 harness.** Verdicts against docs/05 (details in "Phase 6 DoD evaluation"):

- **Row 42 (PDF): not met as defined** (per-σ samples too small: 8–32 decimated samples per σ at
  2 speckle sizes against n ≥ 50; see open decision c). A σ-pooled variant (σ = 100–250 nm, both
  seeds, n = 90) passes by 0.7 % at the default analysis parameters (Stephens-modified statistic
  1.086 < 1.094, plain p = 0.20) and rejects under most neighbouring choices (12 of 20
  one-at-a-time variants, section "Row-42 sensitivity"); its samples are not independent (the
  same two maps at every σ), and it passes only with σ = 100 nm in the pool (without it: 1.298,
  rejected). Together with the 1-speckle rejection (same pool at 1 speckle size, n = 278: 1.73,
  p = 0.006; null level 6.7 %), the per-σ rejections at σ = 150, 200 and 250 nm (harness KS at
  1 speckle size) and the envelope-normalised contrast C_env = 1.04–1.29 (ideal speckle through the
  same pipeline: 0.95–1.01 ± 0.12–0.15 per pattern) **the patterns are measurably not
  negative-exponential** — physically consistent with the few correlation areas in the spot: the
  beam spot covers only (w₀/Lc)² = π w₀²/(π Lc²) = 1.8 surface correlation areas (convention: spot
  area π w₀², the 1/e² intensity radius; with the effective area π w₀²/2 of the intensity
  footprint it is 0.9), so the speckle is not fully developed in the Goodman sense (many
  independent scatterers) although its coherent part is negligible; a Kirchhoff phase-screen
  surrogate of this configuration shows the same (C_env 1.18–1.32, σ-pooled gate rejected in 64 %
  of its realisations).
- **Row 43 contrast: met.** Seed means 10.72, 5.65, 3.67, 2.66, 2.15 for σ = 50 … 250 nm, strictly
  decreasing, each seed on its own too. **γ₁₂: not testable as defined** (normal incidence; open
  decision b).
- **Row 44 speckle size: met for 3 of the 4 fully developed σ** (+2.8, +4.4, +0.4 % at 100, 150,
  250 nm), **σ = 200 nm +11.4 %**; mean over the four +4.8 %. The method's own two-seed scatter on
  ideal speckle is 8–15 % per σ, so the per-σ 10 % criterion is at the statistical limit of two
  seeds. The scaling with λz/D is not tested (one D, one z).
- **Open decisions for the lead**: speckle count (a), the definition of γ₁₂ (b) and the
  classification of σ = 100 nm and pooling over σ for row 42 (c), with options and costs; a
  4-solve oblique-incidence queue for γ₁₂ is prepared, not started.

## Configuration

| Parameter | Value |
|---|---|
| Media | λ = 500 nm, vacuum R1 (z < ξ), Si ε_r = 18.478 − 0.606j (`silicon_500nm`, exp(+jωt)) |
| Surface | Gaussian height map, Lc = 1 µm (1/e radius of exp(−r²/Lc²)), σ ∈ {50, 100, 150, 200, 250} nm, seeds 1 and 2 (the same seed gives the same map scaled by σ), h = 50 nm = λ/10, **L = 8 µm** (161 × 161 grid) |
| Closing box | ADR 0006 defaults via `rough_surface_box_params`: uniform box (box cells = 50 nm), depth 5.65 µm (10 intensity decay lengths), fine band 3δ + 3σ (irrelevant for the uniform box); 247 040 triangles, **2N = 741 120** for every case |
| Excitation | `AngularSpectrumBeam`, normal incidence (+z), p (E along x), focus at the origin, **w₀ = L/6 = 1.333 µm** (`check_beam_waist` with the Si fraction 1/6, ADR 0006 2026-10-12b), controlled ball 1.01 × the largest vertex distance (2 380 plane waves) |
| Operator | ICTF (automatic), compression "mlfmm", d₀ = 3, **`leaf_radius_quantile` = 0.99** (see deviations), 125 nm leaves, 7 levels |
| Solver | GMRES (full, CGS2), preconditioner "auto" = none on the box route for Si, **tol 1e-4** |
| Observation | scattered field E_s from `Simulation.field` (direct evaluation) on the plane z = −1 mm, 4 × 4 mm², 161 × 161 points (spacing 25 µm); far field F on the 1° reflection hemisphere (`hemisphere_grid(-1, 90, 180)`) for R |

Deviations from the lead decision (L ~ 10 µm, ≥ 30 speckles per axis): L = 10 µm was not
attempted because the uniform-box row Si σ = 50 nm, L = 10 µm of WP22b3 peaked at 74.8 GB (budget
60 GB); L = 8 µm peaks at 46–56 GB. `leaf_radius_quantile` = 0.99: at Lc = 1 µm the global leaf rule
r_max/0.6 jumps to 250 nm leaves for σ ≥ 200 nm (r_max 84–98 nm, one steep cell); the 0.99
quantile keeps 125 nm leaves for every case (r_0.99 ≤ 72.6 nm), identical to the global rule for
σ ≤ 100 nm and σ = 150 nm seed 2 (r_max 59.8–73.7 nm). σ = 150 nm seed 1 has r_max = 75.48 nm,
r_max/a = 0.604 > 0.6 for a = 125 nm leaves (8 RWG edges above 75 nm), so the global rule would
already have chosen 250 nm leaves there. The speckle count is discussed in "Open decisions for
the lead".

### Plane size and sampling (derivation)

The illuminated footprint is the beam's intensity exp(−2r²/w₀²), D = 2w₀ = 2.667 µm. In the far
zone (z = 1 mm ≫ D²/λ = 14 µm, and ≫ 2L²/λ = 256 µm for the whole patch) the field correlation
is the Fourier transform of the footprint (van Cittert–Zernike):
|μ_A(Δx)|² = exp(−(π w₀ Δx/(λz))²) on axis, so

- on-axis speckle size λz/D = **187.5 µm**; FWHM of |μ_A|² = c·λz/D with c = 1.060041 (WP-S1
  `speckle_size_expected(λ, z, 2w₀, "gaussian")`) = **198.8 µm**, the row-44 reference;
- grid spacing 25 µm = **7.5 samples per λz/D** (8 per FWHM; ≥ 4 required);
- the plane edge reaches θ = atan(2) = 63.4° along the axes (70.5° in the corners).

The pattern is stationary in the direction cosines u = sin θ (speckle FWHM c·λ/D = 0.199 in u),
not in plane coordinates: along an axis x = z tan θ, so a speckle is (λz/D)/cos³θ wide at the
polar angle θ, and a plane reaching θ_max holds 2 sin θ_max · D/λ speckles per axis, at most
2D/λ = **10.7** for an infinite plane. The 4 mm plane holds **9.5**; the paraxial count
S/(λz/D) = 21.3 overstates it. The largest (u, v) square inside the plane is |u|, |v| ≤ 2/3; the
analysis uses 0.66 (6.6 FWHM speckles per axis). Inside the diffuse halo, where the statistics are
taken, there are fewer still (section "Results": 3.9–6.7 per axis).

The incident beam is not included. The rigorous beam is controlled only within its 8 µm ball, so
`kind="total"` at z = −1 mm is not valid; the paraxial Gaussian beam of the same waist
(`incident_paraxial_intensity.npy`, a note only) is a converging spot of 1/e² radius
λz/(πw₀) = 119 µm with peak |E_inc|² = 1.25·10⁻⁴ (V/m)² for the unit on-axis amplitude, about 6× the
peak of the reflected specular spot at σ = 50 nm; it lies inside the central speckle and is
excluded from every statistic here (scattered field only, the primary quantity).

### Solver tolerance

tol 1e-4 (the brief's default). Records: on the Si uniform box, tol 1e-4 vs 1e-5 changes |F|² over the
reflection hemisphere by 4.7·10⁻⁴ (L2, WP-V2, `box_validity_mlfmm.md`), far below the percent-level
precision of contrast, correlation and speckle size from ~10² speckles; tol 1e-3 (the paper's
value) would leave ~10× more. Cost: 100 iterations at σ = 50 nm (69 at 1e-3, WP22b3).

## Run log

Ten cases, one process each, sequentially (queue). σ = 50 nm seed 1 was the scout (2026-10-25);
the queue ran σ = 250 (seeds 1, 2), 50 (2), 150 (1, 2), 100 (1, 2) and started σ = 200 seed 1 on
2026-10-26 03:18. **A machine reset at 03:47 interrupted σ = 200 seed 1 at GMRES iteration ≈ 100**
(no output written; the partial log is kept as `log_sigma200e-9_seed1_interrupted.txt`). The
coordinator restarted the queue for σ = 200 seeds 1 and 2 at 07:25 with the same frozen package and
arguments (finished cases are skipped by their `row.txt`); both completed (rows below). No other case
was affected. Data (outside git): `<main checkout>/results/local_runs/wp_s2/` (`cases/<tag>/` with
`intensity.npy` (|E_s|², all components), `E.npy`, `far_F.npy`, `currents.npy`, `meta.json`,
`row.txt`; `rows.txt`, `progress.txt`, one log per case).

## Analysis

`benchmarks/speckle_sweep_stats.py` → `specklebem.speckle_sweep.analyse_ensemble` (one call per σ,
both seeds) → `specklebem.speckle.ensemble_statistics` (WP-S1). The WP-S1 functions assume
stationary speckle on a regular grid; the raw plane pattern is not stationary (diffuse halo,
specular spot, 1/r² and the 1/cos³θ widening of the speckles), so the patterns are prepared first:

1. **Co-polarised intensity** I = |E_x|² (`polarized_intensity(E, co_pol=[1, 0, 0])`; docs/05 row 42).
   The plane holds 68–98 % of |E_s|² in E_x, 0.0–3 % in E_y, the rest in E_z (the oblique-propagation
   component of the same polarisation).
2. **Direction cosines**: bilinear resampling onto |u|, |v| ≤ 0.66, du = (λ/D)/8 = 0.0234
   (`plane_to_direction_cosines`), multiplied by (r/|z|)² (radiant intensity ∝ |F|²), where far-zone
   speckle is stationary with FWHM c·λ/D.
3. **Envelope normalisation, leave-one-out**: each pattern is divided by the mean of the four
   mirror images (u → −u, v → −v; the ensemble-mean co-polarised intensity is even in u and v at normal
   incidence with E along x) of the other seed and the three non-trivial mirror images of itself,
   smoothed with a Gaussian of 1/e radius 0.5 speckle sizes (`leave_one_out_envelopes`). The
   pattern itself is excluded, except on the mirror axes u = 0 and v = 0, where one of its own
   mirror images coincides with it (all three at the centre). Envelope choices on synthetic speckle in (u, v) with the sweep's
   window sizes (flat, narrow and wide halo; 200 repetitions of two seeds, KS on 32 samples at 2
   speckle sizes, true envelope as reference: contrast 0.93–0.97, level 4.0–4.5 %): self-including
   ensemble envelope contrast 0.86–0.89, level 6–12 %; each pattern smoothed on its own scale (the
   interim stand-in) over 1 speckle size 0.74–0.78, level 14–26 %, over 2 speckle sizes
   0.88–1.02, level 5.5–9.5 %; **leave-one-out 0.95–1.02, level 6.5–8 %** (scratch study; the
   end-to-end calibration of the chosen pipeline follows).
4. **Window**: the centred square where the smoothed mean envelope (both seeds, all mirror images)
   is ≥ 5 % of its maximum, the same for both seeds (`centred_window`). No specular masking (lead
   rule): the window always contains the specular direction.
5. **WP-S1 harness** on the windows: `ensemble_statistics(windows, pixel = du·|z|, λ, |z|, 2w₀,
   shape="gaussian")`. Its pixel size du·|z| makes its speckle size a plane length on axis, directly
   comparable with `speckle_size_expected` (198.8 µm).

Which number gates which docs/05 row, and how it maps onto the WP-S1 definitions:

| Row | Quantity (this record) | WP-S1 definition used | Gate |
|---|---|---|---|
| 42 PDF | decimated I/⟨I⟩ of the normalised windows, stride = `independent_stride(198.8 µm, du·|z|, 2)` = 17 px (2 speckle sizes), each window divided by its decimated mean, **pooled over both seeds per σ** (docs/05; n = 8–32 here, so not testable), and as a **σ-pooled variant** over both seeds and all fully developed σ (coherent fraction exp(−(2kσ)²) < 1e-2: σ = 100–250 nm; samples not independent, decision c) | `ks_test_exponential` → `KsResult.rejects(0.05)` (Stephens-modified statistic vs 1.094), plain Kolmogorov p alongside | does not reject, n ≥ 50 independent samples |
| 42 (supplementary) | per σ: the harness's own pooled KS (stride for 2 speckle sizes if n ≥ 50, else 1 speckle size, else none); all σ pooled at 1 speckle size | same | reported only (neighbour correlation \|μ_A\|² = 1/16 at 1 speckle size) |
| 43 contrast | `speckle_contrast` of the raw co-polarised plane pattern (whole 4 × 4 mm², no normalisation), seed mean per σ | `speckle_contrast`, `monotone_decreasing` | monotone decrease |
| 43 γ₁₂ | see "Open decisions for the lead"; the seed-to-seed and σ-step `correlation_coefficient` values are reported | `correlation_coefficient` | not testable as defined |
| 44 size | FWHM of \|μ_A\|² (`field_correlation_from_intensity` + `speckle_size`) along u and v of each normalised window, mean over axes and seeds per σ | `speckle_size_rel_error` of the harness vs `speckle_size_expected(λ, z, 2w₀, "gaussian")` | \|rel. error\| < 10 % per fully developed σ |

The envelope-normalised contrast C_env (harness `contrast` on the windows) is the local contrast of
the speckle itself (1 for fully developed polarised speckle, < 1 with a coherent part); it is a
diagnostic, not the row-43 quantity.

### Calibration on synthetic speckle

`benchmarks/speckle_sweep_calibration.py` runs the same analysis (two seeds per entry, the sweep's
plane, waist and sampling) on synthetic plane patterns. Columns: the window, the per-entry KS of the
harness (stride used, false-rejection rate), the KS sample sizes at 2 / 1 speckle sizes, the raw
co-polarised plane contrast (row 43 quantity, seed mean), the envelope-normalised contrast per
pattern, the speckle-size error per entry (mean over seeds and axes) and the seed-to-seed
correlations; mean ± standard deviation over the repetitions. Sample sizes n are the mean and,
in brackets, the range over the repetitions (the window follows the realised envelope).

**Null (exact circular Gaussian speckle, `synthetic_far_zone_speckle`, Gaussian halo of rms s in u;
300 repetitions, seed 20261026; every entry drawn independently):**

| halo rms (u) | window ±u | KS harness: spacing, rejection rate | n (2 / 1 speckle) | C plane (seed mean) | C env | size error | ρ seeds plane | ρ seeds env |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.352 | 1, 0.000 | 9 [8–18] / 31 [18–50] | 6.70 ± 0.69 | 0.945 ± 0.154 | +0.018 ± 0.149 | 0.553 ± 0.117 | −0.102 ± 0.179 |
| 0.28 | 0.479 | 1, 0.046 | 18 [8–18] / 51 [32–72] | 4.29 ± 0.40 | 0.974 ± 0.133 | +0.015 ± 0.116 | 0.511 ± 0.097 | −0.110 ± 0.126 |
| 0.35 | 0.612 | 1, 0.060 | 27 [18–32] / 84 [50–98] | 3.06 ± 0.31 | 0.989 ± 0.116 | +0.013 ± 0.085 | 0.478 ± 0.088 | −0.112 ± 0.099 |
| 0.45 | 0.656 | 1, 0.057 | 32 [18–32] / 98 [72–98] | 2.23 ± 0.22 | 1.011 ± 0.121 | +0.022 ± 0.082 | 0.416 ± 0.087 | −0.117 ± 0.088 |

Pooled over the four entries: **σ-pooled gate (2 speckle sizes, n = 86 [62–100]) rejects in
4.7 %** of the repetitions (level-correct), the supplementary pool at 1 speckle size (n = 263
[212–300]) in 6.7 %. The speckle-size estimate is unbiased to +1–2 % with a two-seed scatter of
8–15 %; the seed-to-seed correlation of independent patterns is −0.11 ± 0.09–0.18 on the
normalised windows (the leave-one-out envelope of one pattern contains the other) and +0.42–0.55
on the raw plane (common envelope). Plane contrast seed means are monotone over the four halo
widths in 99.7 %: the raw plane contrast of pure speckle falls from 6.7 to 2.2 with the halo
width alone, so the row-43 contrast largely tracks the width of the diffuse halo (which grows
with σ), not a change of the speckle statistics.

**Correlated null (`--map-correlation μ`, the sweep's dependence):** the same entries, but each
seed's aperture field (circular Gaussian, `circular_gaussian_field`) is carried from one entry to
the next as μa + √(1 − μ²)b (`correlated_field`), so adjacent entries share their random field as
the sweep's σ steps share the scaled height map (intensity correlation μ²; measured on the
normalised windows: ρ_env = 0.05, 0.34–0.36, 0.49–0.52 for μ² = 0.05, 0.38, 0.54, the range of
the sweep's σ steps 0.05–0.54, mean 0.38). 300 repetitions each, seed 20261026:

| μ (μ²) | σ-pooled gate: n, rejection rate | 1-speckle pool: n, rejection rate |
|---|---|---|
| independent (record above) | 86 [62–100], 0.047 | 263 [212–300], 0.067 |
| 0.224 (0.05) | 86 [62–100], 0.060 | 265 [204–300], 0.043 |
| 0.616 (0.38) | 86 [62–100], 0.060 | 266 [186–300], 0.077 |
| 0.735 (0.54) | 87 [66–100], 0.057 | 265 [186–318], 0.067 |
| 1 (identical fields at every entry) | 86 [66–100], 0.157 | 265 [184–318], 0.200 |

At the sweep's dependence the σ-pooled gate's level is 5.7–6.0 % (binomial standard error
± 1.4 % at 300 repetitions; the reviewer's scratch run of the same construction gave 7.1–7.6 %):
the nominal 5 % roughly holds, but the pooled samples are not independent, and with fully
reused fields the level triples. Per-entry numbers are unchanged within their scatter.

**Kirchhoff surrogate of the sweep (`phase_screen_far_zone_speckle`: footprint exp(−r²/w₀²) times
exp(−2jkσh), h Gaussian with correlation exp(−r²/Lc²), one unit map per seed scaled by σ; scalar,
no Fresnel or obliquity factors; 200 repetitions):**

| σ [nm] | failures | window ±u | KS harness: spacing, rejection rate | n (2 / 1 speckle) | C plane (seed mean) | C env | size error | ρ seeds plane | ρ seeds env |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 2 | 0.473 | 1, 0.542 | 18 [8–32] / 52 [18–98] | 4.83 ± 0.94 | 1.240 ± 0.325 | +0.412 ± 0.263 | 0.434 ± 0.137 | −0.141 ± 0.163 |
| 150 | 0 | 0.631 | 1, 0.690 | 29 [18–32] / 91 [50–98] | 2.82 ± 0.63 | 1.317 ± 0.331 | +0.500 ± 0.304 | 0.346 ± 0.126 | −0.137 ± 0.125 |
| 200 | 2 | 0.655 | 1, 0.571 | 32 [18–32] / 97 [72–98] | 2.04 ± 0.41 | 1.240 ± 0.280 | +0.374 ± 0.295 | 0.282 ± 0.125 | −0.115 ± 0.127 |
| 250 | 0 | 0.656 | 1, 0.420 | 32 / 98 | 1.71 ± 0.30 | 1.180 ± 0.262 | +0.262 ± 0.255 | 0.241 ± 0.127 | −0.104 ± 0.127 |

("failures": ensembles where |μ_A|² stays above one half inside the window, a single glint lobe.)
σ-pooled gate (n = 111 [76–128]) rejects in **64 %**, the 1-speckle pool (n = 337 [220–392]) in
82 %; plane contrast monotone in 96 %; σ-step correlation of the normalised windows +0.49 to +0.55 ± 0.2 (the Gaussian-statistics
value exp(−(2kΔσ)²) = 0.21 assumes many scatterers). With only ≈ 1.8 correlation areas in the spot
the far field is a sum of a few glints: over-dispersed intensities (C > 1) and larger, irregular
"speckles". The surrogate is approximate (Lc = 2λ, scalar) and is a model expectation, not a
reference; the BEM patterns lie between it and the null (section "Results").

Two variations (surrogate, same pipeline): **Lc = 0.5 µm** at the same L (7 correlation areas, 50
repetitions): window ±0.656 for every σ, C env 1.03, 1.03, 1.01, 1.00, size error +4.4, +1.4,
−1.4, −1.0 %, σ-pooled gate rejection 6 % (n = 128 in every repetition), 1-speckle pool 10 %
(n = 392), but the halo then fills the plane and the plane contrast saturates (2.00, 1.47, 1.32,
1.28; monotone in only 56 %). **L = 25 µm** (w₀ = 4.17 µm, 17 correlation areas, plane 641² to
keep 9.6 samples per speckle, `--n-fft 512`; 60 repetitions): C env 1.04–1.07, size error +2.5
to +4.3 %, σ-pooled gate rejection 15 % at n = 811 [684–888], 1-speckle pool 60 % — against a pipeline null at the same waist with
matched windows (exact speckle, halos 0.265, 0.42, 0.5, 0.6 in u; n = 819 [782–854]) of 17 % and
28 % (C env 1.02–1.04, size error 0 to +1.5 %). At large n the leave-one-out envelope from two
seeds is itself detectable: the 2-speckle gate does not separate surrogate and null there (the
1-speckle pool and C env still do), so more seeds or a smoother envelope model are needed. (From
the committed script with the options under "Reproduction"; they replace earlier scratch figures.)

## Results

`benchmarks/speckle_sweep_stats.py <OUT>/cases --markdown` (all ten cases, ≈ 10 s).

### Per case

All cases: 2N = 741 120, ICTF, no preconditioner, converged to 1e-4 (true residual equal to the
GMRES residual within 6·10⁻¹⁰), 125 nm leaves, 7 levels. Wall time = setup + solve + plane + far field.
C total = std/mean of |E_s|² (all components, raw plane); C co-pol = the same for |E_x|² (row 43);
C env = envelope-normalised window (diagnostic); size = FWHM of |μ_A|² along u / v in plane units on
axis (reference 198.8 µm).

| σ [nm] | seed | it | setup [s] | solve [s] | plane [s] | far [s] | wall [min] | peak [GB] | R | C total | C co-pol | C env | size u / v [µm] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 1 | 100 | 134 | 1125 | 241 | 99 | 26.6 | 47.4 | 0.3872 | 11.14 | 11.29 | 0.928 | 209 / 298 |
| 50 | 2 | 106 | 180 | 1083 | 292 | 136 | 28.2 | 45.8 | 0.3873 | 10.00 | 10.15 | 0.784 | 184 / 226 |
| 100 | 1 | 123 | 112 | 1224 | 232 | 93 | 27.7 | 50.7 | 0.3857 | 6.41 | 6.58 | 1.207 | 228 / 240 |
| 100 | 2 | 125 | 111 | 1252 | 245 | 110 | 28.6 | 48.9 | 0.3861 | 4.51 | 4.71 | 0.879 | 159 / 191 |
| 150 | 1 | 134 | 121 | 1662 | 344 | 104 | 37.2 | 52.8 | 0.3830 | 4.04 | 4.20 | 1.381 | 232 / 230 |
| 150 | 2 | 134 | 218 | 1614 | 271 | 105 | 36.8 | 52.7 | 0.3838 | 2.85 | 3.13 | 0.942 | 217 / 152 |
| 200 | 1 | 152 | 121 | 1593 | 232 | 101 | 34.1 | 53.7 | 0.3784 | 2.90 | 3.12 | 1.504 | 222 / 232 |
| 200 | 2 | 148 | 121 | 1577 | 228 | 88 | 33.6 | 53.5 | 0.3786 | 1.93 | 2.20 | 1.084 | 275 / 157 |
| 250 | 1 | 180 | 133 | 2314 | 422 | 136 | 50.1 | 55.7 | 0.3678 | 2.34 | 2.57 | 1.360 | 223 / 225 |
| 250 | 2 | 178 | 155 | 2332 | 274 | 111 | 47.9 | 55.1 | 0.3644 | 1.50 | 1.74 | 0.957 | 195 / 155 |

Total 5.8 h of solver wall time for the sweep (plus the interrupted ≈ 30 min). 10.0–13.1 s per
GMRES iteration; peak 46–56 GB (setup peak 33–40 GB; the solve and evaluation phase adds
13–16 GB: Krylov basis 1.2–2.1 GB at 2N × 16 B per iteration, the rest not attributed).
R = reflected power / incident power from the 1° far field (Fresnel, flat Si, normal incidence:
0.3877).
The co-polarised share of |E_s|² on the plane is 98 % at σ = 50 nm and 68–75 % at 250 nm (E_y ≤ 3 %;
E_z is the oblique-propagation component).

### Per σ (two seeds)

| σ [nm] | exp(−(2kσ)²) | fully dev. | window ±u (speckle sizes per axis) | C co-pol mean (spread) | C env mean | harness KS: spacing, n, D, p, modified, rejects | KS at 2 speckle sizes: n | size mean [µm] (rel. error) | ρ seeds plane / env |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 2.1·10⁻¹ | no | 0.234 (2.5) | 10.72 (1.14) | 0.856 | – (n < 50) | 8 | 229.3 (+0.154) | 0.976 / 0.876 |
| 100 | 1.8·10⁻³ | yes | 0.375 (3.9) | 5.65 (1.88) | 1.043 | – (n < 50) | 8 | 204.4 (+0.028) | 0.854 / 0.433 |
| 150 | 6.7·10⁻⁷ | yes | 0.516 (5.3) | 3.67 (1.07) | 1.162 | 1, 50, 0.155, 0.18, 1.12, **yes** | 18 | 207.6 (+0.044) | 0.625 / 0.161 |
| 200 | 1.1·10⁻¹¹ | yes | 0.656 (6.7) | 2.66 (0.92) | 1.294 | 1, 98, 0.128, 0.08, 1.29, **yes** | 32 | 221.5 (**+0.114**) | 0.465 / −0.007 |
| 250 | 7.2·10⁻¹⁸ | yes | 0.656 (6.7) | 2.15 (0.84) | 1.158 | 1, 98, 0.120, 0.12, 1.21, **yes** | 32 | 199.6 (+0.004) | 0.329 / −0.108 |

**Row 42 as defined** (pooled over the seeds per σ, ≥ 50 samples at 2 speckle sizes): not
testable, n = 8, 8, 18, 32 for σ = 100, 150, 200, 250 nm (column "KS at 2 speckle sizes").

**σ-pooled variant** (σ = 100, 150, 200, 250 nm, both seeds, 2 speckle sizes = stride 17 px):
**n = 90, D = 0.1131, plain Kolmogorov p = 0.200, Stephens-modified 1.086 (critical 1.094): not
rejected, by 0.7 % of the statistic.** The pooled samples come from the same two maps at every σ
(σ-step ρ_env 0.05–0.54, table "Correlations"), so they are not independent; the correlated null
(section "Calibration") puts the level of this pooled test at 5.7–6.0 % instead of 4.7 %.
Supplementary, the same pool at 1 speckle size: n = 278, D = 0.1026, p = 0.006, modified 1.729:
rejected (null level 6.7 %).

### Row-42 sensitivity

`benchmarks/speckle_sweep_stats.py <OUT>/cases --sensitivity --markdown` (`SENSITIVITY_VARIANTS`):
the σ-pooled variant recomputed with one analysis parameter changed at a time against the
defaults (envelope radius 0.5 speckle sizes, window threshold 0.05, 8 (u, v) samples per λ/D,
decimation offset 0, KS spacing 2 speckle sizes, σ = 100–250 nm, u_max = 0.66).

| Variant | σ pooled [nm] | n | D | plain p | modified (crit. 1.094) | rejects |
|---|---|---|---|---|---|---|
| baseline | 100, 150, 200, 250 | 90 | 0.1131 | 0.200 | 1.086 | no |
| envelope radius 0.35 | 100, 150, 200, 250 | 90 | 0.1365 | 0.070 | 1.316 | **yes** |
| envelope radius 0.75 | 100, 150, 200, 250 | 100 | 0.1595 | 0.012 | 1.623 | **yes** |
| envelope radius 1.0 | 100, 150, 200, 250 | 114 | 0.1427 | 0.019 | 1.548 | **yes** |
| threshold 0.03 | 100, 150, 200, 250 | 100 | 0.1507 | 0.021 | 1.533 | **yes** |
| threshold 0.08 | 100, 150, 200, 250 | 90 | 0.1417 | 0.054 | 1.367 | **yes** |
| threshold 0.10 | 100, 150, 200, 250 | 76 | 0.1242 | 0.192 | 1.098 | **yes** |
| threshold 0.15 | 100, 150, 200, 250 | 76 | 0.1055 | 0.366 | 0.929 | no |
| samples per speckle 6 | 100, 150, 200, 250 | 90 | 0.1219 | 0.138 | 1.173 | **yes** |
| samples per speckle 7 | 100, 150, 200, 250 | 90 | 0.1037 | 0.288 | 0.995 | no |
| samples per speckle 10 | 100, 150, 200, 250 | 90 | 0.1082 | 0.243 | 1.038 | no |
| decimation offset 2 px | 100, 150, 200, 250 | 90 | 0.1528 | 0.030 | 1.476 | **yes** |
| decimation offset 4 px | 100, 150, 200, 250 | 90 | 0.1377 | 0.066 | 1.328 | **yes** |
| decimation offset 6 px | 100, 150, 200, 250 | 62 | 0.0558 | 0.990 | 0.431 | no |
| decimation offset 8 px | 100, 150, 200, 250 | 62 | 0.1094 | 0.448 | 0.871 | no |
| decimation offset 10 px | 100, 150, 200, 250 | 62 | 0.0791 | 0.833 | 0.622 | no |
| KS spacing 1.5 | 100, 150, 200, 250 | 150 | 0.1002 | 0.099 | 1.240 | **yes** |
| KS spacing 2.5 | 100, 150, 200, 250 | 62 | 0.1720 | 0.051 | 1.383 | **yes** |
| smallest fully developed σ left out | 150, 200, 250 | 82 | 0.1410 | 0.077 | 1.298 | **yes** |
| u_max 0.60 | 100, 150, 200, 250 | 62 | 0.1024 | 0.534 | 0.813 | no |
| u_max 0.63 | 100, 150, 200, 250 | 90 | 0.0995 | 0.335 | 0.954 | no |

**12 of the 20 variants reject** (every envelope radius and decimation offset of 2–4 px tried, 3 of
4 thresholds, both other KS spacings, 6 samples per speckle, and leaving σ = 100 nm out); 5 of the
8 that do not reject change n (62 or 76 instead of 90). The default pass is one point in a
neighbourhood where the decision flips both ways with n ≈ 60–150 samples: the gate does not
separate these patterns from exponential at this sample size, while the 1-speckle pool, the
per-σ harness KS and C_env do (see the summary at the top).

**Row 43**: contrast seed means 10.72, 5.65, 3.67, 2.66, 2.15 — `monotone_decreasing` true
(also within the largest seed spread 1.88); seed 1 alone 11.29, 6.58, 4.20, 3.12, 2.57, seed 2 alone
10.15, 4.71, 3.13, 2.20, 1.74. The seeds are the same two maps at every σ (scaled), so the two
sequences are paired, and seed 1 is the brighter-speckled one at every σ. The surrogate gives the
same trend at a lower level (4.83, 2.82, 2.04, 1.71 for 100–250 nm). The raw plane contrast
largely tracks the width of the diffuse halo: pure speckle through the same pipeline goes from
6.7 to 2.2 as the halo widens (null calibration), so the monotone decrease measures the halo's
growth with σ more than a change of the speckle statistics.

**Row 44**: rel. errors +2.8, +4.4, +11.4, +0.4 % for the fully developed σ, mean +4.8 %
(σ = 50 nm: +15.4 %, coherent part 21 %, Siegert relation not applicable). Single-axis values
range 152–298 µm; the scatter of the method on ideal speckle is ±8–15 % per two-seed entry.

### Correlations

Seed-to-seed (same σ, different surfaces; `correlation_coefficient`): raw plane 0.976, 0.854,
0.625, 0.465, 0.329 and normalised windows 0.876, 0.433, 0.161, −0.007, −0.108 for σ = 50 … 250 nm —
both monotone decreasing; the window values reach the independent-pattern null (−0.11 ± 0.1–0.18)
at σ ≥ 150 nm. The excess at 50 and 100 nm is the common coherent (specular) part.

σ steps (same seed, scaled map, common window of the two σ):

| σ step [nm] | seed | ρ plane | ρ env | exp(−(2kΔσ)²) |
|---|---|---|---|---|
| 50 → 100 | 1 | 0.506 | 0.523 | 0.206 |
| 50 → 100 | 2 | 0.527 | 0.236 | 0.206 |
| 100 → 150 | 1 | 0.629 | 0.538 | 0.206 |
| 100 → 150 | 2 | 0.577 | 0.051 | 0.206 |
| 150 → 200 | 1 | 0.636 | 0.469 | 0.206 |
| 150 → 200 | 2 | 0.579 | 0.233 | 0.206 |
| 200 → 250 | 1 | 0.624 | 0.505 | 0.206 |
| 200 → 250 | 2 | 0.578 | 0.489 | 0.206 |

Mean of the normalised values 0.38 — above the many-scatterer value 0.21, inside the surrogate's
+0.49–0.55 ± 0.2 (few glints that move with σ but persist). No trend with σ, as expected.

## Phase 6 DoD evaluation

Against docs/01 Phase 6 DoD, docs/05 rows 42–44 and the WP-S2b lead rule (negative-exponential
gate only where exp(−(2kσ)²) < 1e-2, no specular masking):

| DoD item | docs/05 | Verdict | Numbers |
|---|---|---|---|
| Fully developed speckle at z = −1 mm shows a negative-exponential PDF (KS p > 0.05) | 42 | **not met as defined** | per σ (the docs/05 pooling over seeds) n = 8, 8, 18, 32 < 50 at 2 speckle sizes (σ = 100–250 nm; decision c). A σ-pooled variant (n = 90, samples not independent: correlated-null level 5.7–6.0 %) passes by 0.7 % at the default analysis parameters (modified 1.086 < 1.094, plain p 0.20) and rejects under 12 of 20 neighbouring one-at-a-time choices (section "Row-42 sensitivity"), and without σ = 100 nm (1.298). With the 1-speckle rejection (1.73, p = 0.006, n = 278, null level 6.7 %), the per-σ harness rejections at 150/200/250 nm and C env 1.04–1.29 (null 0.95–1.01) the patterns are measurably not negative-exponential — physically consistent with the few correlation areas in the spot ((w₀/Lc)² = 1.8 with the area π w₀²; surrogate C env 1.18–1.32, σ-pooled gate rejection 64 %). |
| Contrast decreases monotonically with σ over 50–250 nm (two seeds each) | 43 | **met** | 10.72 → 5.65 → 3.67 → 2.66 → 2.15 (seed means, co-polarised plane contrast); each seed monotone on its own; the raw plane contrast largely tracks the halo width (pure speckle through the pipeline: 6.7 → 2.2 over the null's halo widths) |
| Angular correlation decreases monotonically with σ | 43 | **not testable with this configuration** | normal incidence only: the incidence-angle decorrelation is second order in Δθ; the σ-step correlation has no σ trend by construction (0.05–0.54, mean 0.38); the seed-to-seed correlation is monotone (0.88 → −0.11) but measures the coherent part, not an angular correlation. Open decision (b). |
| Speckle size from μ_A scales as λz/D within 10 % | 44 | **partly met** | +2.8 / +4.4 / **+11.4** / +0.4 % at σ = 100/150/200/250 nm, mean +4.8 %; statistical scatter of two seeds 8–15 %; only one (λ, z, D): the magnitude is tested, the scaling is not |
| σ = 50 / 100 nm partially developed (lead rule: report contrast < 1 as the expectation) | 42 | reported | C env 0.86 (σ = 50 nm, coherent fraction 0.21) and 1.04 (σ = 100 nm, 1.8·10⁻³); by the exp(−(2kσ)²) < 1e-2 rule σ = 100 nm counts as fully developed and is in the σ-pooled variant, whose pass depends on it (the WP-S2b row text lists it among the partially developed ones — open decision c) |

Not met in the backlog configuration: "≥ 30 speckles per axis" (3.9–6.7 per axis inside the
windows, ≤ 10.7 possible at L = 8 µm; decision a).

## Open decisions for the lead

Prepared by the worker; not decided. Costs per solve of this size on the desktop: 27–50 min wall
(σ = 100: 28, 150: 37, 200: 34, 250: 49 min), peak 46–56 GB, one at a time.

### (a) Speckle count ("≥ 30 speckles per axis")

Facts. Far-zone speckles are λ/D wide in direction cosines, so at most 2D/λ = 10.7 fit per axis
(c·λ/D = 0.199 FWHM: 10.1) whatever the plane; the 4 mm plane's largest (u, v) square holds 6.6,
the halo windows 3.9 (σ = 100 nm), 5.3 (150) and 6.7 (200, 250). Independent KS samples at 2 speckle
sizes: 4–16 per pattern, 8–32 per σ, 90 pooled over the fully developed σ (not independent:
the same two maps at every σ). 30 per axis would need 2u_w·D/(cλ) ≥ 30 with the halo half-width
u_w = 0.375–0.66, i.e. D = 12–21 µm, w₀ = 6–11 µm and L = 36–64 µm at w₀ = L/6. Unknowns from the
uniform-box model (top and bottom face (L/h)² cells, walls L × depth, so the walls scale with L
only): triangles T = 4n² + 8nm with n = L/h, m = depth/h = 113 (h = 50 nm, depth 5.65 µm), 2N = 3T;
it reproduces L = 8 µm (247 040 triangles, 2N = 741 120) and L = 25 µm (2N = 4.36·10⁶,
`si_large_area.md`) exactly and gives **2N ≈ 8.2·10⁶ at L = 36 µm and 2.3·10⁷ at 64 µm**
(peak ≈ 0.5–1.0 TB at 36 µm by scaling the L = 30 / 32 µm extrapolations of `si_large_area.md`
linearly with 2N, ≈ 1.4–2.8 TB at 64 µm: beyond the 1 TB server at the upper end); even the full
hemisphere needs D ≥ 8 µm, L ≥ 24 µm. A second, independent limit appeared here: the spot covers
(w₀/Lc)² = π w₀²/(π Lc²) = 1.8 correlation areas (spot area π w₀², the 1/e² intensity radius;
0.9 with the effective area π w₀²/2), too few for circular Gaussian speckle (row-42 caveats,
surrogate). The same convention gives 7 at Lc = 0.5 µm (option D) and 17 at L = 25 µm (option C).

| Option | What | Extra solves (desktop) | Memory | Effect |
|---|---|---|---|---|
| A | Accept L = 8 µm; replace "≥ 30 per axis" by the docs/05 wording "≥ 50 independent samples" and record row 42 as not met as defined: per σ only 8–32 samples (not met); pooled over σ 90, but not independent (the same scaled maps; σ-step ρ_env 0.05–0.54, correlated-null level 5.7–6.0 % vs 4.7 %) | 0 | — | row 42 as now: the σ-pooled variant passes by 0.7 % at the defaults and rejects under 12 of 20 neighbouring choices and without σ = 100 nm (decision c); low power (null 4.7 %, few-scatterer surrogate 64 %). **WP-S2b must not assert `rejects(0.05) == false` on these data**: it can assert the observed non-exponential behaviour with its explanation (e.g. the 1-speckle pool rejects and the per-σ mean C env > 1.1 for σ ≥ 150 nm, the few-correlation-area expectation of the surrogate) or report the gate unasserted |
| B | More seeds at L = 8 µm (seeds 3–6 for σ = 100–250) | 16 solves ≈ 10 h | ≤ 56 GB | per-σ KS at 2 speckle sizes reaches n ≥ 50 for σ ≥ 150 nm (54–96), σ = 100 nm stays at 24; contrast and size scatter /√3 (size ±5–9 % per σ, so the 10 % row-44 test becomes meaningful); the few-scatterer statistics remain (it measures them better) |
| C | Server run, L = 25 µm (w₀ = 4.17 µm, (w₀/Lc)² = 17), 10 cases | 10 server cases of 2N = 4.36·10⁶ (uniform box). Solve 1.6–3.6 h at σ = 100 nm (desktop extrapolation of `si_large_area.md`, made for Lc = 0.5 µm), scaled with this sweep's iteration growth relative to σ = 100 nm (σ = 50: 0.83, 150: 1.08, 200: 1.21, 250: 1.44, from 123–125 to 178–180 iterations; mean over the ten cases 1.11): 1.3–5.2 h per case, 18–40 h for the ten solves on the desktop, × 1.9–2.5 on the lead's server ≈ 34–100 h ≈ 1.5–4 days, plus setup (15–30 min per case extrapolated) and the far field; an estimate, not a measurement | 366–437 GB at σ = 100 nm, ≈ 10 % more at σ = 250 nm (this sweep: 49.8 → 55.4 GB) ≈ 400–480 GB (1 TB server only) | ≈ 12–21 speckles per axis in the windows (still < 30), far field on a 0.25° grid instead of a 641² plane (the direct plane evaluation would cost ≈ 6 h per case); surrogate (60 repetitions): C env 1.04–1.07, size +2.5–4.3 %, 1-speckle pool rejects in 60 % vs 28 % for exact speckle, but the 2-speckle gate rejects exact speckle in 17 % at n ≈ 820 through the two-seed envelope: needs more seeds or a smoother envelope estimate |
| D | Shorter Lc = 0.5 µm at L = 8 µm ((w₀/Lc)² = 7), 10 cases | 10 solves ≈ 6–7 h (more iterations likely: rms slope √2σ/Lc up to 0.71) | ≤ 56 GB (leaf rule to re-check: r_max grows with the slope) | surrogate: Gaussian statistics (C env 1.00–1.03, gate level 6 %, size within 4.4 %), but the halo fills the plane at every σ and the row-43 plane contrast saturates (2.00, 1.47, 1.32, 1.28; monotone in 56 % of the surrogate realisations): changes the lead's configuration and weakens row 43 |

**Recommendation: A now** with row 42 recorded as not met as defined (the patterns are measurably
not negative-exponential, consistent with 1.8 correlation areas in the spot) and the WP-S2b ctest
asserting that behaviour or reporting the gate unasserted — not asserting the σ-pooled pass;
**plus B if the lead wants per-σ PDF tests or a resolvable 10 % size criterion** (it is the
cheapest way to more independent samples, and it would measure the non-exponential statistics,
not remove them). C is the definitive "fully developed" statement and belongs to the server list;
D is not recommended (row 43).

### (b) γ₁₂ ("angular correlation decreases monotonically with σ")

At normal incidence a change of the incidence angle by Δθ changes the vertical wavevector transfer
q_z only to second order (k σ Δθ²/2), and the memory effect shifts the pattern by Δ(sin θ_in); the
present data contain one incidence angle. Candidate definitions (Kirchhoff, Gaussian heights,
intensity correlation of the diffuse part = exp(−σ² Δq_z²)):

| Option | Definition | Monotone in σ? | Extra solves | Notes |
|---|---|---|---|---|
| b1 | **Incidence-angle pair 0° / 30°, memory-effect aligned**: `correlation_coefficient` of the radiant intensities at equal transverse wavevector transfer (u' = u + sin 30°) over the halo window, from the far field F (1° grid; the plane does not reach u = 0.5 ± 0.66) | yes: Δq_z = k[(1 + cos θ_s) − (cos 30° + cos θ_s')] ≈ 0.27k at the window centre → 0.97, 0.89, 0.77, 0.63, 0.49 for σ = 50 … 250 nm | **4** (σ = 100–250, seed 1, θ = 30°; reuses the 0° solves); 8 with seed 2; +1 for σ = 50 | w₀ = L cos 30°/6 = 1.155 µm (footprint rule); footprint overlap with the 0° runs 0.99; 35–55 min each (oblique iterations unmeasured), ≤ 56 GB; estimate scatter ±0.1–0.2 per σ from 15–40 speckles, so the full-range trend (0.89 → 0.49) is resolvable, adjacent steps are not; needs a far-field resampling function (WP-S2b or a follow-up) |
| b2 | Seed-to-seed correlation (two surfaces, same σ), already computed | yes, measured 0.88, 0.43, 0.16, −0.01, −0.11 (normalised windows) | 0 | measures the coherent part and saturates at the independent-pattern null for σ ≥ 150 nm; an ensemble, not an angular, correlation |
| b3 | σ-step correlation (same map, σ → σ + 50 nm) | no: exp(−(2kΔσ)²) = 0.21 for every step (measured 0.05–0.54) | 0 | — |
| b4 | Spatial (angular-lag) correlation \|μ_A(Δu)\|² at a fixed lag | no: set by D only | 0 | — |
| b5 | Wavelength pair, exp(−(2Δkσ)²) | yes | 5–10 | needs Si data at a second wavelength (materials are tabulated at 500 nm only) and Δλ ≈ 100 nm for a usable range; not an angular correlation |

**Recommendation: b1 with seed 1 (4 solves, ≈ 3 h)**, and docs/05 row 43 amended to define γ₁₂ as
the memory-effect-aligned incidence-angle correlation (0°/30°) with the Kirchhoff curve as the
expectation; b2 as a zero-cost supporting number. The queue is prepared (driver `--theta-deg`, queue
`THETA`/`PKG`; a separate frozen package so that `pkg/` stays untouched), **not started**:

```bash
OUT=<main checkout>/results/local_runs/wp_s2
PKG30="$OUT/pkg_t30"
mkdir -p "$PKG30" && cp -r build/win-release/python/specklebem "$PKG30/" \
    && cp benchmarks/speckle_sweep.py benchmarks/speckle_sweep_queue.sh "$PKG30/"
PKG="$PKG30" THETA=30 nohup sh "$PKG30/speckle_sweep_queue.sh" "$OUT" \
    /c/msys64/ucrt64/bin/python 100e-9:1 150e-9:1 200e-9:1 250e-9:1 > "$OUT/queue_t30.out" 2>&1 &
```

(cases `si_L8um_s<σ>nm_Lc1um_seed1_t30`, waist L cos 30°/6 by default; the 30° case construction
was smoke-tested on a 2 µm patch, no 8 µm oblique solve has been run.)

### (c) Classification of σ = 100 nm and pooling over σ (row 42)

Facts. docs/05 row 42 pools I/⟨I⟩ over the seeds, i.e. per σ; with two seeds at L = 8 µm that
gives n = 8, 8, 18, 32 decimated samples at 2 speckle sizes for σ = 100, 150, 200, 250 nm, below
n ≥ 50 for every σ. The record's gate pooled over σ as well (n = 90). That pool reuses the same
two maps, scaled by σ, at every σ: its samples are not independent (σ-step ρ_env 0.05–0.54,
mean 0.38); on exact speckle with that dependence the pooled test's level is 5.7–6.0 % instead
of 4.7 % (correlated null, section "Calibration"; 15.7 % if the fields were identical at every
entry), so the level roughly holds but "≥ 50 independent samples" is not met literally. σ = 100 nm
is fully developed by the WP-S2b lead rule (coherent fraction exp(−(2kσ)²) = 1.8·10⁻³ < 10⁻²),
but the WP-S2b row text lists it among the partially developed cases, and its seed-to-seed
correlation on the normalised windows (0.43, null −0.11) still shows a common part. The σ-pooled
pass depends on it: without σ = 100 nm the pool (n = 82) rejects (1.298), and 12 of 20
neighbouring analysis choices reject with it (section "Row-42 sensitivity").

| Option | Rule | Row 42 on these data | Cost | Notes |
|---|---|---|---|---|
| c1 | Keep the lead rule (σ = 100 nm fully developed) and amend docs/05 row 42 to allow pooling over the fully developed σ, with the measured dependence stated (level 5.7–6.0 % on the correlated null of this record, 7.1–7.6 % in the review's scratch run) | passes by 0.7 % at the defaults, rejects under 12 of 20 neighbouring choices: not a robust statement either way | 0 | a ctest asserting the pass would be fragile against any change of the analysis; the other diagnostics (1-speckle pool, C env, per-σ harness KS) contradict it |
| c2 | Classify σ = 100 nm as partially developed (the WP-S2b row text; or tighten the rule to exp(−(2kσ)²) < 10⁻³) and pool over σ = 150–250 nm | rejects (n = 82, 1.298) | 0 | consistent with the other diagnostics; still pools over σ (same dependence) |
| c3 | Keep row 42 per σ as defined (pooled over the seeds only) | not testable (n = 8–32 < 50); per σ ≥ 50 needs option (a) B (six seeds: 54–96 for σ ≥ 150 nm, σ = 100 nm stays at 24) or C | as (a) B / C | the classification of σ = 100 nm then only decides whether σ = 100 nm needs n ≥ 50 too |

**Recommendation: c3 for the row as defined** (no σ-pooled gate asserted), with the σ-pooled
numbers of c1 and c2 reported as supplementary diagnostics; σ = 100 nm stays fully developed by the
lead rule (its coherent part is negligible; the residual common part is visible in the seed
correlation, not in the PDF test, which cannot be done per σ here). Under every option the
measured patterns are not negative-exponential at the 1-speckle level and in C env, as expected
for 1.8 correlation areas in the spot.

## Reproduction

```bash
OUT=<main checkout>/results/local_runs/wp_s2
# solves (one case per process; frozen package in $OUT/pkg, finished cases skipped)
sh benchmarks/speckle_sweep_queue.sh "$OUT" /c/msys64/ucrt64/bin/python 50e-9:1 50e-9:2 \
    100e-9:1 100e-9:2 150e-9:1 150e-9:2 200e-9:1 200e-9:2 250e-9:1 250e-9:2
# statistics (seconds) and calibration (minutes)
/c/msys64/ucrt64/bin/python benchmarks/speckle_sweep_stats.py "$OUT/cases" \
    --pythonpath build/win-release/python --markdown --sensitivity --json "$OUT/summary.json"
CAL="/c/msys64/ucrt64/bin/python benchmarks/speckle_sweep_calibration.py --pythonpath build/win-release/python"
$CAL --model gaussian --reps 300
for mu in 0.224 0.616 0.735 1; do $CAL --model gaussian --reps 300 --map-correlation $mu; done
$CAL --model phase-screen --reps 200
$CAL --model phase-screen --lc 0.5e-6 --reps 50
W25="--waist 4.1666666666666667e-6 --plane-n 641 --n-fft 512 --reps 60"
$CAL --model phase-screen $W25
$CAL --model gaussian --halos 0.265,0.42,0.5,0.6 $W25
```
