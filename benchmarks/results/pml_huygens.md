# Huygens-plane far field of the PML route: validations 7f and 7g (WP-PML6), wide-gap confirmation (WP-PML6b)

ADR 0010 decision 2–3, docs/05 "PML route", rows 7f and 7g. Library code
`include/specklebem/postprocessing/huygens.hpp` (`post::huygens_plane`, `huygens_plane_field`, `huygens_reweight`,
`huygens_radiate`, `huygens_far_field`), `post::exterior_scattered_field` (fields.hpp; the PML source triangles
radiate through the stretched Green's function), `FarFieldSource` / `PmlTermination::far_field_source, huygens`
(simulation.hpp). Case code `tests/support/pml_huygens_support.hpp`; tests `tests/unit/test_huygens.cpp`,
`tests/validation/test_pml_huygens.cpp` (7f at CI size, 2N = 7 632), `tests/validation_large/test_pml_huygens_large.cpp`
(7f in the 7a geometry; 7g, reported values with regression guards; 6b, the gates of WP-PML6b), `tests/unit/test_pml_huygens_support.cpp`;
study driver `benchmarks/pml_huygens.cpp` (`--study 7f|7g|si7d|6b`), 7e through
`benchmarks/pml_si_crosscheck.cpp --far-field huygens`; raw lines `pml_huygens_rows.txt`. Measured on a 12-core
desktop with 128 GB RAM, win-release build, one heavy run at a time (7f, 7g, 7e on 2026-10-20, the extended-plane
diagnostic and WP-PML6b on 2026-10-25).

## Summary

- **7f met** (flat Ag and Si, 7a geometry, dense LU): Huygens plane vs windowed currents L2 2.73·10⁻⁴ / 2.83·10⁻⁴
  (≤ 10⁻³), ΔR/R −1.7·10⁻⁷ / −4.2·10⁻⁷ (≤ 10⁻⁴). The plane's own sensitivities are small: z_h ×1.5 2.6 / 2.7·10⁻⁴,
  spacing /1.5 8.5 / 7.4·10⁻⁷, plane quadrature degree 10 vs 6 5·10⁻¹².
- **7g not met** (rough Ag U, fast path): 7d with the Huygens far field 2.78 % (each run with its own default
  plane window) / 1.83 % (the same absolute window) against 0.5 %; the windowed currents give 2.57 % (reproduces
  WP-PML4b). The plane's own sensitivities are not small: z_h ×1.5 1.05 %, plane window ×0.7 / ×1.3 1.75 % /
  0.66 % (limit 0.1 % each). Rough Si 7e with the Huygens far field: l2_refl 0.220 % and ε_rr 0.038 / 0.044 % pass,
  ΔR/R −5.3·10⁻⁴ fails (tolerance 4.1·10⁻⁵).
- **Cause: the truncation of the plane by its window**, not the physics. The plane window runs across the gap
  (raised cosine from taper_end to pml_start, transition Δ = W_g = λ) and the plane cannot reach over the PML. A
  plane of finite width at the distance z_h misses the radiation of the patch at grazing angles (θ 75–90°: −67 %
  of that band's power against the currents at U) and the window end leaves an error of order π²/(2(kΔ)²) per
  end for every field that crosses the window zone without decaying (point-source unit test: 4.7·10⁻² / 9.8·10⁻³ /
  2.3·10⁻³ at Δ = 2 / 4 / 8 λ; at Δ = λ the model gives 0.125 per end, the unit-test geometry measures
  2.8·10⁻¹, section "Diagnostic: extended plane"). Both grow with z_h. On the flat surfaces of 7f the
  reflected beam does not reach the window zone, which is why 7f passes.
- **ADR 0010's "if 7g fails" clause does not apply**: it attributes a failed 7g to a physical L dependence only if
  the Huygens far field itself is converged (small z_h and window sensitivities); here the sensitivities are
  1–2 %, of the size of the 7d change.
- **Default kept**: `FarFieldSource::Currents` stays the far-field source of the PML route; the Huygens plane is a
  documented option (state z_h, spacing and plane window with every result). The tolerances of docs/05 are not
  loosened.
- **Diagnostic "extended plane"** (section at the end; not validated, for the lead decision): the plane cannot
  reach over the PML without code changes (and a theory for stretched observation points); measured instead on
  the U surface with a 3 µm real gap. With a window transition Δ = 2 / 3 / 4 / 6 λ the plane's own z_h ×1.5
  sensitivity falls to 0.49 / 0.28 / 0.18 / 0.10 % (θ ≤ 30°: 0.21 / 0.073 / 0.042 / 0.018 %) and Huygens and
  currents agree at θ ≤ 30° to 0.16 / 0.079 / 0.038 / 0.018 %, but 7d stays at 2.4–2.5 % (θ ≤ 30°: 2.2–2.4 %) for
  the plane and the currents alike: in the cone θ ≤ 30° this is evidence that the L dependence on rough Ag is
  physical (ADR 0010's clause; pending the lead decision), at the hemisphere the plane is still not converged
  (grazing band −44 % at Δ = 6 λ).

- **Validation-large** (`test_pml_huygens_large.cpp`, 7g values reported with 10 % regression guards): the rough-Ag
  case ran once on 2026-10-25 (3 289 s on a machine shared with other studies) and reproduced every documented
  value to the printed digits (7d 2.7763 %, z_h ×1.5 1.0536 %, window ×0.7 / ×1.3 1.7488 / 0.6642 %); its one
  failure was an exact floating-point comparison of the window (2.375 µm up to round-off), fixed with a 10⁻¹²
  margin. **Review round, 2026-10-25, both 7g cases run as ctests** (`ctest --preset win-release -L
  validation-large -R "PML 7g"`, win-release, machine shared with other workers' builds and tests): rough Ag
  **passed** in 3 471 s (602 / 669 / 705 GMRES iterations, true residuals 8.4 / 8.6 / 8.5·10⁻⁸; every value
  identical to the documented one: 7d 2.7763 %, z_h ×1.5 1.0536 %, window ×0.7 / ×1.3 1.7488 / 0.6642 %; R
  Huygens / currents 0.9752538 / 0.9783392 at U); rough Si 7e **passed** in 3 676 s (run alone after the first
  attempt was stopped by the 2 h limit of the shell during the PML assembly; box: 92 iterations, R 0.3856955,
  peak 12.75 GB; PML fast path: 127 iterations, true residual 9.3·10⁻⁸, assembly 2 646 s, solve 338 s, ff check
  3.6·10⁻¹²): Huygens vs box l2_refl 0.2197 %, ε_rr 0.0379 / 0.0443 % (checked, met), ΔR/R −5.332·10⁻⁴ (reported,
  the documented value; tolerance 4.1·10⁻⁵ not met); the windowed currents of the same solve vs the box l2_refl
  0.1464 %, ΔR/R −1.6·10⁻⁵ (checked, met: the WP-PML4c 7e result). The two 7a-size 7f cases, moved to
  validation-large in the same round (`-R "PML 7f"`), passed in 389 / 486 s and reproduced the 7f table (Ag / Si
  L2 2.73·10⁻⁴ / 2.83·10⁻⁴, ΔR/R −1.7·10⁻⁷ / −4.2·10⁻⁷). Raw lines in `pml_huygens_rows.txt`.

## Setup

| Item | Value |
|---|---|
| 7f cases | `pml_huygens::flat_case("ag"/"si")`: the 7a geometry at 0°, patch 1.5 µm, taper 0.5 µm, gap 0.5 µm, W_pml = 2λ, σ_max 4, m 2, w₀ = λ, h = λ/4, 2N = 11 440, dense ICTF + LU (assembly 401 / 415 s, solve 13 s, peak 4.24 GB) |
| 7g cases | the WP-PML4b rough-Ag U case (`pml_rough_ag::u_case()`: σ = 50 nm, Lc = 500 nm, seed 1, L = 3.75 µm, taper 0.5 µm, gap 0.5 µm, W_pml = 2λ, w₀ = λ, h = λ/4, 2N = 22 816), its L ×1.2 partner (`patch_case()`, 2N = 27 472) and a wide-gap partner (`wide_gap_case()`: gap 1.375 µm, pml_start 3.75 µm, 2N = 34 352) on which U's window can be scaled by 0.7 and 1.3 |
| 7g solver | fast path (compression "mlfmm": MLFMM d₀ = 3 on the real functions + PML ring with ACA far blocks, GMRES 10⁻⁷; the 7h configuration): 602 / 669 / 705 iterations, true residual 8.4–8.6·10⁻⁸, assembly 332 / 376 / 342 s, solve 67 / 93 / 124 s, peak 1.8 / 2.4 / 3.0 GB |
| 7e | `pml_si_crosscheck` defaults (rough Si, L = 4.5 µm, w₀ = L/6, h = λ/4): uniform box (MLFMM + GMRES, 92 iterations, R 0.3856955: reproduces WP-PML4c) against the PML route on the fast path (127 iterations, assembly 1 862 s of which 1 629 s the ACA far ring, solve 214 s, peak 9.7 GB) with the Huygens far field |
| Huygens plane | defaults: z_h = λ below the lowest mesh vertex (U: plane at z = −0.639 µm, lowest vertex −0.139 µm), spacing λ/4 (uniform grid anchored at x = y = 0), plane window = the current window (taper_end, pml_start); the plane fields from `exterior_scattered_field` (Dunavant degree 6, the PML source triangles through the stretched Green's function) |
| Far fields | reflection hemisphere `post::hemisphere_grid(−1, 90, 180)` (Gauss–Legendre in θ, solid-angle weights) and the xz / yz cuts; R = P_refl / P_inc; currents: `Simulation::far_field_options()` window |
| Metric | box-check L2 of \|F\|² over the hemisphere (`pml_rough_ag::compare`; l2_off30: θ ≥ 30°), ε_rr on the cuts, ΔR/R = R_T/R_R − 1; the cone views (θ ≤ 30°, θ ≥ 30°, band powers) are computed from the stored far-field files with the same weights |

## 7f: flat Ag and Si (met)

| Comparison | flat Ag L2 | ΔR/R | flat Si L2 | ΔR/R | criterion |
|---|---:|---:|---:|---:|---|
| Huygens (defaults) vs currents | 2.73·10⁻⁴ | −1.7·10⁻⁷ | 2.83·10⁻⁴ | −4.2·10⁻⁷ | ≤ 10⁻³; ≤ 10⁻⁴ |
| z_h ×1.5 vs Huygens | 2.64·10⁻⁴ | −5.9·10⁻⁷ | 2.72·10⁻⁴ | −9.3·10⁻⁷ | — |
| z_h ×1.5 vs currents | 5.28·10⁻⁴ | −7.6·10⁻⁷ | 5.44·10⁻⁴ | −1.3·10⁻⁶ | ≤ 10⁻³ |
| spacing /1.5 (λ/6) vs Huygens | 8.5·10⁻⁷ | −3.5·10⁻⁸ | 7.4·10⁻⁷ | −3.5·10⁻⁸ | — |
| spacing /1.5 vs currents | 2.72·10⁻⁴ | −2.0·10⁻⁷ | 2.82·10⁻⁴ | −4.6·10⁻⁷ | ≤ 10⁻³ |
| plane quadrature degree 10 vs 6 | 5.2·10⁻¹² | −4.8·10⁻¹² | 4.6·10⁻¹² | −5.2·10⁻¹² | — |

R: Ag 0.9816288 (Huygens) / 0.9816290 (currents), Si 0.3880908 / 0.3880910. The residual difference lies off
specular (l2_off30 2.7 / 3.0·10⁻³ on a far field that is concentrated around the specular direction) and grows
with z_h (the window effect below); the grid spacing and the source quadrature are converged.

**CI size (2026-10-25, review round).** The 7a-size cases above run in `validation-large` since the review
(`test_pml_huygens_large.cpp`, `PML 7f: …`); CI's `validation` label checks 7f on `flat_ci_case()`: the same 7a
zones (patch 1.5 µm, taper 0.5 µm, gap 0.5 µm, w₀ = λ, h = λ/4; plane window 1.25–1.75 µm = 2.5–3.5 w₀) with the
thin PML of the 7h CI case (W_pml = λ, σ_max 2) and the 60 × 120 hemisphere, 2N = 7 632, dense LU (assembly
165 / 325 s, solve 57 / 166 s on a machine shared with two other builds; peak 1.9 GB):

| Comparison (CI size) | flat Ag L2 | ΔR/R | flat Si L2 | ΔR/R | criterion |
|---|---:|---:|---:|---:|---|
| Huygens (defaults) vs currents | 2.73·10⁻⁴ | −1.7·10⁻⁷ | 2.84·10⁻⁴ | −4.1·10⁻⁷ | ≤ 10⁻³; ≤ 10⁻⁴ |
| z_h ×1.5 vs currents | 5.28·10⁻⁴ | −7.6·10⁻⁷ | 5.45·10⁻⁴ | −1.3·10⁻⁶ | ≤ 10⁻³ |
| spacing /1.5 vs currents | 2.72·10⁻⁴ | −2.0·10⁻⁷ | 2.84·10⁻⁴ | −4.5·10⁻⁷ | ≤ 10⁻³ |
| z_h ×1.5 vs Huygens | 2.64·10⁻⁴ | −5.9·10⁻⁷ | 2.71·10⁻⁴ | −9.3·10⁻⁷ | — |
| spacing /1.5 vs Huygens | 8.5·10⁻⁷ | −3.5·10⁻⁸ | 7.5·10⁻⁷ | −3.5·10⁻⁸ | — |

The thin PML and the coarser hemisphere change the 7a-size values by < 1 % of themselves (R 0.9816288 /
0.9816290 and 0.3880908 / 0.3880910, as above). The smaller geometry of the 7h CI case (`pml_ring::
flat_ag_ci_case()`: patch 1 µm, taper 0.25 µm, 2N = 4 592; `pml_huygens::flat_ci7h_case()`, tags `flat_ci7h_ag` /
`flat_ci7h_si`) does **not** qualify for 7f: its plane window starts
at 0.75 µm = 1.5 w₀, the reflected beam crosses the window zone (rms gap currents 4.3·10⁻² of the patch level
against 8·10⁻⁴ in the CI case) and the plane's truncation error (section "Why 7g
fails") dominates: Ag / Si L2 8.1·10⁻³ / 7.9·10⁻³, ΔR/R −5.4·10⁻⁴ / −5.5·10⁻⁴, z_h ×1.5 1.8·10⁻² /
1.8·10⁻² (spacing /1.5 vs the default plane 8.7·10⁻⁵, i.e. the plane quadrature is fine); with a 1 µm patch and
a 0.5 µm taper (`flat_ci7h_taper_case()`, tag `flat_ci7h_t05_ag`; window from 1.0 µm = 2 w₀, 2N = 6 016) Ag still gives L2 2.1·10⁻³, ΔR/R −1.6·10⁻⁵ (z_h ×1.5
4.8·10⁻³). This is a direct control of the 7f premise: the plane agrees with the currents only when the
reflected beam has decayed before the plane window. The CI rows and these controls are reproduced by
`benchmarks/pml_huygens.cpp --study 7f --cases ci` and `--cases ci-control` (WP-PML6b; rerun on 2026-10-25: every
value identical, only times and round-off residuals differ).

## 7g: rough Ag and rough Si (not met)

Rough Ag, fast path (L2 metric over the hemisphere, the θ ≤ 30° cone, ΔR/R):

| Comparison | L2 hemisphere | L2 θ ≤ 30° | L2 θ ≥ 30° | ΔR/R | criterion |
|---|---:|---:|---:|---:|---|
| 7d (L ×1.2), Huygens, own default windows | 2.78 % | 2.68 % | 3.90 % | +4.6·10⁻⁴ | < 0.5 % |
| 7d, Huygens, U's absolute window on L ×1.2 | 1.83 % | 1.82 % | 2.00 % | −1.7·10⁻⁴ | — |
| 7d, currents, own default windows (WP-PML4b) | 2.57 % | 1.92 % | 6.83 % | +1.8·10⁻⁴ | — |
| 7d, currents, U's absolute window | 2.49 % | — | 4.16 % | +8.4·10⁻⁶ | — |
| z_h ×1.5 on U | 1.05 % | 0.81 % | 2.72 % | −1.5·10⁻³ | < 0.1 % |
| z_h ×1.5 on L ×1.2 | 0.95 % | — | 2.53 % | −1.2·10⁻³ | — |
| plane window ×0.7 (wide-gap run, vs U's window) | 1.75 % | 1.48 % | 3.89 % | −1.7·10⁻³ | < 0.1 % |
| plane window ×1.3 (wide-gap run) | 0.66 % | 0.43 % | 2.00 % | +5.4·10⁻⁴ | < 0.1 % |
| spacing /1.5 on U / L ×1.2 / wide gap | 4.4 / 2.6 / 0.43·10⁻⁵ | — | — | ≤ 1.1·10⁻⁷ | — |
| gap 0.5 → 1.375 µm at U's window: Huygens / currents | 2.9·10⁻⁵ / 3.5·10⁻⁵ | — | — | −1.1·10⁻⁶ / −5.8·10⁻⁷ | (inert) |
| Huygens vs currents on U | 1.41 % | 0.97 % | 4.05 % | −3.2·10⁻³ | — |
| Huygens vs currents on L ×1.2 | 1.34 % | — | 4.14 % | −2.9·10⁻³ | — |
| wide-gap run (window 2.375–3.75 µm, Δ = 1.375 µm): z_h ×1.5 | 0.32 % | 0.085 % | 1.18 % | −1.1·10⁻³ | — |
| wide-gap run: Huygens vs currents | 0.49 % | 0.09 % | 1.85 % | −2.2·10⁻³ | — |

R on U: 0.97525 (Huygens) vs 0.97834 (currents); L ×1.2: 0.97570 vs 0.97851; wide gap: 0.97557 vs 0.97775. Gap
currents (area-weighted rms over the gap relative to the patch): U \|J\| 0.0999, \|M\| 0.0877; L ×1.2 0.105 / 0.098;
wide gap 0.092 / 0.081 — the SPP currents of WP-PML4b.

Band powers (power of the test far field in the band over that of the reference, −1):

| Comparison | θ 60–75° | θ 75–90° |
|---|---:|---:|
| Huygens vs currents, U | −5.5 % | −67.4 % |
| Huygens vs currents, wide gap | −8.0 % | −61.4 % |
| z_h ×1.5, U | −18.9 % | −25.1 % |
| z_h ×1.5, wide gap | −12.8 % | −21.3 % |
| plane window ×0.7 / ×1.3 (wide gap) | −21.0 % / +11.1 % | −6.7 % / −3.3 % |

Rough Si, 7e repeated with the Huygens far field (PML fast path vs the uniform box; the 7e tolerances):

| Comparison | l2_refl | l2_off30 | ε_rr xz / yz | ΔR/R | result |
|---|---:|---:|---:|---:|---|
| PML Huygens vs box | 0.220 % (≤ 0.229 %) | 1.06 % | 0.038 / 0.044 % (≤ 0.058 %) | −5.3·10⁻⁴ (≤ 4.1·10⁻⁵) | l2, ε pass; R fails |
| PML Huygens vs PML currents (same solve) | 0.167 % | 0.85 % | 0.027 / 0.024 % | −5.2·10⁻⁴ | — |

The windowed currents of the same solve reproduce WP-PML4c (0.146 %, ΔR/R −1.6·10⁻⁵). The ΔR of 5·10⁻⁴ is the
plane's own deficit (Huygens vs currents on the same solution), not the PML.

## Why 7g fails: the plane window

1. **Geometric acceptance.** The plane lies at z_h ≈ λ below the surface and ends at pml_start (the raised cosine
   from taper_end). A source on the patch radiates into the direction θ through the plane around the lateral
   offset z_h tan θ; from the beam centre the window zone (2.375–2.875 µm on U, plane 0.64 µm below the mean
   plane) is reached at θ ≈ 75°, from the patch edge (1.875 µm) at θ ≈ 38–57°. The power at grazing angles is
   therefore lost (θ 75–90°: −67 % at U, −61 % on the wide-gap run, whose window starts at the same 2.375 µm), and it is lost more the farther the plane is from the surface (z_h ×1.5: −25 % in
   that band, −19 % at 60–75°). Rough Si shows the same deficit (7e: ΔR/R −5.2·10⁻⁴ against its own currents).
2. **Window-end error.** A field that crosses the window zone without decaying (radiation of the patch, the
   reflected diffuse field) is truncated by the C¹ raised cosine. For a point source at distance λ from the plane
   the unit test `huygens: the window error of a point source falls like 1 / (k Delta)^2` measures
   4.72·10⁻² / 9.83·10⁻³ / 2.31·10⁻³ at Δ = 2 / 4 / 8 λ against the model π²/(2(kΔ)²) per end (3.1·10⁻² /
   7.8·10⁻³ / 2.0·10⁻³); at Δ = λ (the default plane window across a gap of λ) the point source is off by
   2.8·10⁻¹ (model 0.125 per end). Further values of the same replica: Δ = 3 λ 1.90·10⁻², 6 λ 4.18·10⁻³.
3. **Consequence.** Neither the 7d change nor the sensitivities of the Huygens far field are converged with the
   plane: z_h ×1.5 and the window ×0.7 / ×1.3 move it by 0.7–1.8 %, as much as L ×1.2. The 7g failure is a
   property of the bounded plane, not evidence of a physical L dependence (ADR 0010's physical-conclusion clause
   presupposes small sensitivities). The windowed currents fail 7d with the same 0.5 µm window for the SPP
   truncation (WP-PML4b); the Huygens plane removes that (the bound field reaches the plane only as e^−κz_h)
   but adds its own truncation of the radiating field. Whether 7d converges once the window is wide is the
   question of the extended-plane diagnostic at the end (it does not: ≈ 2.3 % at θ ≤ 30° for both sources).

## Plane-evaluation cost

The plane fields are a direct O(N_points · N_triangles) evaluation (`exterior_scattered_field`, OpenMP over the
points); the radiation is negligible.

| Case | plane points | spacing | time |
|---|---:|---|---:|
| flat Ag / Si (7f) | 729 | λ/4 | 0.3 s |
| flat, λ/6 | 1 681 | λ/6 | 0.7 s |
| flat, degree 10 | 729 | λ/4 | 0.6 s |
| U | 2 025 | λ/4 | 1.5 s |
| U, λ/6 | 4 761 | λ/6 | 3.5 s |
| L ×1.2 | 2 601 | λ/4 | 2.2 s |
| wide gap (window ×1.3 plane) | 3 481 | λ/4 | 3.6 s |
| wide gap, λ/6 | 7 921 | λ/6 | 8.3 s |

## Conclusion and what a fix would need

- The Huygens plane does what ADR 0010 expected for the bound wave (the gap currents are filtered by distance; at
  θ ≤ 30° on the wide-gap run the plane and the currents differ by 0.09 % and z_h ×1.5 moves it by 0.085 %), but
  a plane that ends at pml_start with a window of Δ = λ truncates the radiating field. 7g is not met; the default
  stays `FarFieldSource::Currents`.
- A fix needs a Huygens surface that captures the grazing radiation and a window transition of several
  wavelengths, i.e. a plane that extends well beyond the patch (for θ up to 80° about 6 z_h beyond the
  illuminated area) and a Δ of ≥ 3–4 λ (point-source window error ≤ 2·10⁻²). On the PML route this means
  either (a) a wider real gap (cost ∝ (outer half width)²: the 3 µm gap of the diagnostic below has 2.7× the
  unknowns of U; it converges the cone θ ≤ 30°, not the grazing band) or (b) plane points over the PML with
  stretched observation coordinates (not available, see below) — or (c) restricting the far-field claim to a cone
  (θ ≤ θ_max given by the plane geometry), which ADR 0010 did not foresee and needs a lead decision.
- The diagnostic below also shows that a wide window (Δ ≥ 3 λ) makes the windowed currents and the Huygens plane
  agree at θ ≤ 30°, and that 7d stays at ≈ 2.3 % there: the remaining L ×1.2 change of rough Ag is not removed
  by either far-field source.

## Diagnostic: extended plane (not validated, for the lead decision)

**Status: diagnostic measurement, not a validation; no default or option changed.** Question: does the Huygens
far field converge, and does 7d fall below 0.5 %, when the plane window has a transition Δ of several
wavelengths (inner edge at the taper end as today)?

**The plane cannot reach over the PML without code changes.** What is missing, exactly:

1. Both entry points refuse it: `post::exterior_scattered_field` throws for an observation point with
   |x| > x_start or |y| > y_start of the stretch, and `post::huygens_plane(s, opt)` throws for a plane window with
   outer_half > the stretch start (`HuygensOptions::window` documents the same rule).
2. Removing the guards is not enough. The stretched-source representation equals the physical field only at
   real observation points over the real zone: the deformed source contour passes no branch point of R̃ only
   while |x| ≤ x_start, |y| ≤ y_start of the observation point (fields.hpp). Over the PML the solution only
   carries the field continued to complex coordinates, so a plane point there needs a stretched observation
   point r̃ = S(r) (a complex–complex R̃ between stretched source and observation points, with its own branch
   rule), complex plane weights with the Jacobian of the stretched plane (dS̃ = det DS dS, the Piola-transformed
   plane currents) and a radiation kernel exp(j k k̂·r̃) that grows like exp(k k̂_x |Im x̃|) — convergent only where
   the continued plane field decays faster, i.e. for directions below the local transverse wavenumber of the
   field in the PML. That direction-dependent validity is the open question ADR 0009 decision 5 answered
   negatively for the surface currents; it would have to be derived and tested before an implementation.
3. Beyond the outer PML edge (pml_start + W_pml = pml_start + 2λ) the mesh ends; there is no representation of
   the unbounded surface there at all, so Δ = 3 and 4 λ from the taper end (2.375 + 1.5 / 2 µm against an outer
   edge of 3.875 µm) cannot be reached on the U geometry even in principle.

**Substitute measured instead:** the same U surface with the real gap widened to 3 µm (`wide_gap3_case()`,
pml_start 5.375 µm, 2N = 62 016) and its L ×1.2 partner with a 2.625 µm gap (`wide_gap3_patch_case()`, the same
pml_start), fast path as in 7g (`benchmarks/pml_huygens.cpp --study 7g --cases ugap3,l12gap3`, function
`delta_windows`). The plane lies over the real flat gap, the window runs from the taper end over Δ = 1, 2, 3, 4
and 6 λ; the gap width is inert at a fixed window (3 µm gap vs U at U's window: Huygens 2.9·10⁻⁵, currents
3.6·10⁻⁵, the same as the 1.375 µm gap), so Δ = λ reproduces 7g. Runs: 886 / 904 GMRES iterations (true residual
8.7 / 8.9·10⁻⁸), assembly 420 / 849 s, solve 569 / 698 s, peak 4.5 GB (the debug build ran concurrently for
part of the time, so these times are not benchmarks); plane fields 12.5 s (7 225 points, Δ = 6 λ). Gap currents
(rms over the wider gap) \|J\| 0.079 / 0.086, \|M\| 0.072 / 0.082 of the patch.

Per Δ (L2 metric over the hemisphere / over θ ≤ 30°; ΔR/R). "own windows": each run's window starts at its own
taper end (2.375 / 2.75 µm), "U window": the absolute window (2.375 µm, 2.375 µm + Δ) on both:

| Δ | Huygens vs currents (U-gap3) | z_h ×1.5 (U-gap3) | z_h ×1.5 (L ×1.2-gap3) | 7d Huygens, own windows | 7d Huygens, U window | 7d currents, own windows |
|---|---|---|---|---|---|---|
| 1 λ | 1.41 % / 0.97 %; −3.2·10⁻³ | 1.05 % / 0.81 %; −1.5·10⁻³ | 0.95 % / 0.70 % | 2.78 % / 2.68 % | 1.83 % / 1.82 % | 2.57 % / 1.92 % |
| 2 λ | 0.67 % / 0.16 %; −2.6·10⁻³ | 0.49 % / 0.21 %; −1.2·10⁻³ | 0.47 % / 0.20 % | 2.55 % / 2.39 % | 2.20 % / 2.15 % | 2.40 % / 2.20 % |
| 3 λ | 0.45 % / 0.079 %; −2.1·10⁻³ | 0.28 % / 0.073 %; −1.0·10⁻³ | 0.27 % / 0.073 % | 2.48 % / 2.32 % | 2.31 % / 2.22 % | 2.40 % / 2.23 % |
| 4 λ | 0.35 % / 0.038 %; −1.8·10⁻³ | 0.18 % / 0.042 %; −8.1·10⁻⁴ | 0.18 % / 0.042 % | 2.45 % / 2.30 % | 2.36 % / 2.24 % | 2.40 % / 2.25 % |
| 6 λ (U-gap3) / 5.25 λ (L ×1.2-gap3) | 0.23 % / 0.018 %; −1.2·10⁻³ | 0.10 % / 0.018 %; −5.6·10⁻⁴ | 0.12 % (5.25 λ) | 2.43 % / 2.28 % | 2.39 % / 2.25 % | 2.41 % / 2.26 % |

Further: the grazing band θ 75–90° stays short against the currents (−67 / −64 / −60 / −56 / −44 % of the band
power at Δ = 1 / 2 / 3 / 4 / 6 λ) — the geometric acceptance of a plane 0.64 µm below the surface does not reach
grazing angles at these widths; z_h ×0.5 / ×2 at Δ = 6 λ: 0.045 % / 0.25 %; the PML source triangles dropped from
the plane fields (`pml_sources = false`): 0.058 % at Δ = 6 λ. The plane window ×(1 ± 0.3) was not repeated on the
wide plane (it would move the inner edge into the taper or the patch).

Point-source window-error model at the same Δ (replica of the unit test, plane at λ from the source, spacing λ/8,
both window ends): Δ = 1 / 2 / 3 / 4 / 6 λ: 2.8·10⁻¹ / 4.7·10⁻² / 1.9·10⁻² / 9.8·10⁻³ / 4.2·10⁻³ against
π²/(2(kΔ)²) = 0.125 / 3.1·10⁻² / 1.4·10⁻² / 7.8·10⁻³ / 3.5·10⁻³ (ratio 2.3 / 1.5 / 1.4 / 1.3 / 1.2). The plane's
own z_h sensitivity on rough Ag falls somewhat slower than the model (hemisphere 1.05 → 0.10 % from Δ = 1 to
6 λ, ≈ Δ^−1.3; θ ≤ 30°: 0.81 → 0.018 %, ≈ Δ^−2.1).

Reading (for the lead decision, not validated):

- With Δ ≥ 3–4 λ the Huygens far field converges in the cone θ ≤ 30°: z_h ×1.5 ≤ 0.07 / 0.04 % and Huygens vs
  currents ≤ 0.08 / 0.04 % (both below the 0.1 % sensitivity criterion of 7g there); on the full hemisphere the
  z_h sensitivity reaches 0.10 % only at Δ = 6 λ, and the grazing band stays deficient.
- **7d does not converge to < 0.5 %**: at every Δ ≥ 2 λ the L ×1.2 change is 2.4–2.5 % (hemisphere) and
  2.2–2.4 % (θ ≤ 30°), the same for the Huygens plane, for the windowed currents and for own vs common absolute
  windows, while the plane's own sensitivities at θ ≤ 30° are 30–120× smaller. In that cone the conditions of
  ADR 0010's "if 7g fails" clause are met on the wide plane: this is evidence that the L dependence of the
  rough-Ag far field is physical (the added rough annulus re-scatters the SPPs launched from the illuminated
  centre) and not a window artefact, pending the lead decision (docs/05 footnote ⁴, ADR 0010 "Result"). Rough Si (no SPP) passes 7d at 0.24 % with the 0.5 µm gap.
- The windowed currents with a wide window (Δ ≥ 3 λ) agree with the Huygens plane at θ ≤ 30° to ≤ 0.08 %; the
  windowed-current far field of the default 0.5 µm gap is the configuration that is window-limited, not the
  current far field as such.
- Caveats: one surface, one seed, h = λ/4 (rough-Ag mesh convergence not measured), plane window ×(1 ± 0.3) not
  repeated on the wide plane, hemisphere sensitivities 0.1–0.2 %, the wide gap costs 2.7× the unknowns of U. The
  second seed and the window ×(1 ± 0.3) on the wide plane are WP-PML6b (next section).

## WP-PML6b: wide-gap confirmation

**Question** (ADR 0010, decision 2026-10-25, items 3–4): on the wide-gap plane, are both far-field sources converged
in the cone θ ≤ 30° with respect to their window (×(1 ± 0.3)), for a second roughness seed too, and is the
L ×1.2 change (7d) there ≥ 10× those sensitivities (the physical L dependence)? Acceptance: window sensitivities
≤ 1·10⁻³ at θ ≤ 30° for the windowed currents and the Huygens plane (Δ = 4 λ) on both seeds; 7d at θ ≤ 30°
(currents and Huygens) ≥ 10× the respective sensitivities on both seeds.

**Result: every gate met on both seeds** (GATE lines in `pml_huygens_rows.txt`): window sensitivity at θ ≤ 30°
currents 4.0·10⁻⁴ / 8.7·10⁻⁴, Huygens 3.0·10⁻⁴ / 3.2·10⁻⁴ (seed 1 / seed 2, max over U and L ×1.2 and over
×0.7 / ×1.3); 7d at θ ≤ 30° currents 2.26 % / 1.58 % (ratio 56 / 18), Huygens 2.30 % / 1.61 % (ratio 76 / 50).
Over the hemisphere the currents are not converged with the window (0.17–0.59 %, grazing band θ 75–90° ±11–24 %):
results outside the cone stay qualitative.

**Setup.** `pml_huygens::wide_gap3_case(seed)` (U: L = 3.75 µm, 3 µm gap) and `wide_gap3_patch_case(seed)`
(L ×1.2 = 4.5 µm, 2.625 µm gap), both pml_start 5.375 µm, 2N = 62 016, every other parameter as U of WP-PML4b
(σ = 50 nm, Lc = 500 nm, map generated on 4.5 + 1 µm and cropped, h = λ/4, w₀ = λ, W_pml = 2λ, σ_max 4); seeds
`pml_huygens::kSeeds = {1, 2}`: seed 1 is the surface of WP-PML4b / WP-PML6 (tags `ugap3`, `l12gap3` as in the
diagnostic above), seed 2 an independent surface of the same statistics (tags `ugap3_s2`, `l12gap3_s2`; its far
field differs from seed 1 by 49 % at θ ≤ 30° and 51 % over the hemisphere; its gap currents are 1.6–1.7× those
of seed 1). Fast path as in 7g (MLFMM d₀ = 3 + PML ring, GMRES 10⁻⁷). One solve per case; only the far-field
evaluation changes (`pml_huygens::six_b_far_fields`). Driver: `benchmarks/pml_huygens.cpp --study 6b --seed N
--cases ugap3|l12gap3 --out DIR --in DIR` (one case per process; the partner's far fields are read from DIR,
bitwise); without `--seed` / `--cases` one process runs all four cases.

**Windows** (`pml_huygens::six_b_windows`). The reference of the window study is the window (taper_end,
taper_end + Δ) with Δ = 4 λ for both sources (the converged flank of the diagnostic above); ×0.7 / ×1.3 scale the
flank Δ at a fixed inner edge (2.8 λ / 5.2 λ; on L ×1.2 the ×1.3 window ends at 5.35 µm, inside pml_start) — a
deviation from the 7g definition of "window ×(1 ± 0.3)": the 7g way of scaling both half widths does not fit the
wide plane (×0.7 moves the inner edge into the patch,
1.66 < 1.875 µm; ×1.3 the outer edge beyond pml_start). The currents' default window (taper_end, pml_start;
Δ = gap = 6 λ / 5.25 λ) is reported with its flank ×0.7 (×1.3 would reach into the PML). 7d is evaluated with each
case's own windows (default and Δ = 4 λ) and, as a check, with U's absolute windows on L ×1.2. Cone metric:
`pml_huygens::cone_compare` (the L2 metric of `pml_rough_ag::compare` restricted to θ ≤ 30°); it reproduces the
CONE values of the diagnostic (Huygens vs currents at Δ = 4 λ on `ugap3` 3.84·10⁻⁴, z_h ×1.5 4.18·10⁻⁴).

**Reproduction of WP-PML6.** The seed-1 runs reproduce the diagnostic above to the printed digits (886 / 904
GMRES iterations, true residuals 8.68 / 8.89·10⁻⁸, R currents 0.9769936 / 0.9774399, R Huygens 0.9757742 /
0.9761625; 7d with the currents and the default windows 2.41 % / 2.26 % (hemisphere / θ ≤ 30°), Huygens Δ = 4 λ
2.45 % / 2.30 %).

L2 metric over the hemisphere / over θ ≤ 30° (window rows: perturbed vs the unperturbed window of the same solve;
7d rows: L ×1.2 vs U of the same seed):

| Comparison | seed 1 U | seed 1 L ×1.2 | seed 2 U | seed 2 L ×1.2 | gate |
|---|---|---|---|---|---|
| currents, Δ = 4 λ, flank ×0.7 | 0.28 % / 4.0·10⁻⁴ | 0.27 % / 3.4·10⁻⁴ | 0.59 % / 8.0·10⁻⁴ | 0.58 % / 8.7·10⁻⁴ | cone ≤ 1·10⁻³ |
| currents, Δ = 4 λ, flank ×1.3 | 0.18 % / 1.4·10⁻⁴ | 0.17 % / 1.4·10⁻⁴ | 0.37 % / 3.3·10⁻⁴ | 0.36 % / 2.8·10⁻⁴ | cone ≤ 1·10⁻³ |
| Huygens, Δ = 4 λ, flank ×0.7 | 0.086 % / 2.7·10⁻⁴ | 0.086 % / 3.0·10⁻⁴ | 0.10 % / 2.6·10⁻⁴ | 0.12 % / 3.2·10⁻⁴ | cone ≤ 1·10⁻³ |
| Huygens, Δ = 4 λ, flank ×1.3 | 0.043 % / 1.0·10⁻⁴ | 0.042 % / 1.2·10⁻⁴ | 0.051 % / 1.0·10⁻⁴ | 0.059 % / 1.3·10⁻⁴ | cone ≤ 1·10⁻³ |
| currents, default window (Δ = gap), flank ×0.7 | 0.22 % / 1.6·10⁻⁴ | 0.22 % / 1.7·10⁻⁴ | 0.44 % / 3.4·10⁻⁴ | 0.47 % / 3.9·10⁻⁴ | — |
| currents, Δ = 4 λ vs default window | 0.24 % / 1.9·10⁻⁴ | 0.17 % / 1.4·10⁻⁴ | 0.50 % / 4.1·10⁻⁴ | 0.37 % / 2.6·10⁻⁴ | — |
| Huygens vs currents, Δ = 4 λ | 0.35 % / 3.8·10⁻⁴ | 0.35 % / 3.6·10⁻⁴ | 0.70 % / 5.4·10⁻⁴ | 0.69 % / 6.2·10⁻⁴ | (rule: cone ≤ 1·10⁻³) |
| Huygens vs currents, default windows | 0.23 % / 1.8·10⁻⁴ | 0.27 % / 2.0·10⁻⁴ | 0.48 % / 2.4·10⁻⁴ | 0.54 % / 3.8·10⁻⁴ | — |
| z_h ×1.5, Δ = 4 λ | 0.18 % / 4.2·10⁻⁴ | 0.18 % / 4.2·10⁻⁴ | 0.23 % / 2.6·10⁻⁴ | 0.21 % / 3.0·10⁻⁴ | — |

| 7d (L ×1.2 vs U), hemisphere / θ ≤ 30° | seed 1 | seed 2 | gate (θ ≤ 30°) |
|---|---|---|---|
| currents, default windows | 2.41 % / 2.26 % | 2.06 % / 1.58 % | ≥ 10× the currents' sensitivity: 56× / 18× |
| currents, U's absolute window (2.375, 5.375 µm) | 2.42 % / 2.27 % | 2.04 % / 1.59 % | — |
| currents, Δ = 4 λ, own windows | 2.40 % / 2.25 % | 2.14 % / 1.58 % | ≥ 10×: 56× / 18× |
| currents, Δ = 4 λ, U's window | 2.42 % / 2.27 % | 2.04 % / 1.59 % | — |
| Huygens, Δ = 4 λ, own windows | 2.45 % / 2.30 % | 2.05 % / 1.61 % | ≥ 10× the plane's sensitivity: 76× / 50× |
| Huygens, Δ = 4 λ, U's window | 2.36 % / 2.24 % | 1.93 % / 1.57 % | — |

Runs (fast path, win-release, 12-core desktop with 128 GB, one heavy run at a time, the machine shared with other
workers' builds and tests: the times are not benchmarks):

| Case | seed | 2N | GMRES its | true residual | assembly | solve | peak | R currents / Huygens | gap rms \|J\| / \|M\| |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| U (`ugap3`) | 1 | 62 016 | 886 | 8.7·10⁻⁸ | 1 250 s | 616 s | 4.50 GB | 0.9769936 / 0.9757742 | 0.079 / 0.072 |
| L ×1.2 (`l12gap3`) | 1 | 62 016 | 904 | 8.9·10⁻⁸ | 786 s | 595 s | 4.52 GB | 0.9774399 / 0.9761625 | 0.086 / 0.082 |
| U (`ugap3_s2`) | 2 | 62 016 | 874 | 8.9·10⁻⁸ | 924 s | 597 s | 4.48 GB | 0.9699002 / 0.9676458 | 0.128 / 0.104 |
| L ×1.2 (`l12gap3_s2`) | 2 | 62 016 | 888 | 8.9·10⁻⁸ | 374 s | 322 s | 4.51 GB | 0.9705283 / 0.9680171 | 0.145 / 0.119 |

Wall time per process 13–33 min; setup and far fields (10 / 13 far fields per case, two plane fields of 7 225
points) take 0.6–2 min of it.

**Reading.**

- In the cone θ ≤ 30° both sources are converged with respect to the Δ = 4 λ flank on both surfaces (the currents'
  default window only at ×0.7): the largest
  window sensitivity is 8.7·10⁻⁴ (currents, flank ×0.7, seed 2), the Huygens plane stays ≤ 3.2·10⁻⁴; plane and
  currents agree to ≤ 6.2·10⁻⁴ (Δ = 4 λ) and ≤ 3.8·10⁻⁴ (default windows), and z_h ×1.5 moves the plane by
  ≤ 4.2·10⁻⁴. These are the conditions of the configuration rule (ADR 0010 decision 2026-10-25, item 3).
- L ×1.2 changes the far field in the cone by 2.24–2.30 % (seed 1) and 1.57–1.61 % (seed 2) for every source and
  window choice (own or common absolute windows), 18–76× the window sensitivities: the L dependence belongs to
  the surface (its size and realisation), not to the far-field evaluation, on two surfaces.
- The margin of the currents is smaller on seed 2 (8.7·10⁻⁴ against 1·10⁻³): its gap currents are 1.6–1.7× those
  of seed 1, and the window sensitivity of the currents grows with them (doubled; the WP-PML4b observation that
  the window effect follows the current level in the gap). A surface that launches markedly stronger surface
  waves may need a gap wider than 6 λ to meet 1·10⁻³ with the currents; the Huygens plane (Δ = 4 λ) keeps a 3×
  margin on both seeds.
- Outside the cone the windowed currents are not converged with the window (hemisphere 0.17–0.59 %, grazing band
  θ 75–90° moved by 11–24 %), and the Huygens plane misses grazing power (36–64 % of the band power in θ 75–90°
  against the currents): rough-metal results outside θ ≤ 30° and with narrower gaps are qualitative.

**Validation-large.** `test_pml_huygens_large.cpp`, case `PML 6b: wide-gap rule on rough Ag, second seed …`
(seed 2, both solves in one case because 7d needs both; the gates asserted, not loosened): `ctest --preset
win-release -L validation-large -R "PML 6b"` **passed** on 2026-10-25 in 2 497 s (run alone among my jobs, the
machine shared with other workers), 874 / 888 GMRES iterations, every CONE value, R and the GATE line identical to
the driver runs above (window sensitivity currents 8.717·10⁻⁴, Huygens 3.228·10⁻⁴; 7d 1.578 / 1.580 / 1.609 %).
