# PML route validation (ADR 0009 decision 7; WP-PML3b: 7a)

Acceptance criterion: ADR 0009 decision 7a / `docs/05_validation.md` "PML route" row 7a — a flat Ag
and a flat Si open surface terminated by the complex-coordinate-stretching PML, under the rigorous
angular-spectrum beam at 0° and 45°, s and p, reproduce Fresnel |r|² to **< 1 % relative**. As in
WP22c (`fresnel_flat.md`) the sharp reference is the beam reflectance **R_beam** of the same
rigorous beam on the infinite interface (`fresnel_flat::beam_reflectance`); the plane-wave Fresnel
value |r(θ₀)|² at the central angle is reported alongside.

Executable `benchmarks/pml_fresnel.cpp` (`specklebem_pml_fresnel`,
`-DSPECKLEBEM_BUILD_BENCHMARKS=ON`, one geometry per process, s and p from one LU), shared code
`tests/support/pml_fresnel_support.hpp`, test `tests/validation_large/test_pml_fresnel_large.cpp`
(the eight cases below plus the mesh-converged Si 45° case, label `validation-large`, release only,
memory-guarded). Raw lines: `pml_validation_rows.txt`.

**Verdict: 7a met**, with one caveat for Si at oblique incidence (accepted at h = λ/6, not at λ/4).
- At h = λ/4 with the default library tolerances (dark band 1e-4, quadrature defaults) Ag (0° and 45°) and Si at
  0° are accepted: Ag 45° −0.28 / −0.15 %, the 0° cases ≤ 0.022 %. Si 45° is also within 1 % there (−0.76 % p,
  −0.41 % s), but that value is not mesh-converged and is **reported only, not an acceptance result**.
- Ag (both angles) and Si at 0° converge monotonically with h: λ/6 gives ≤ 0.04 %.
- **Si at 45° is not mesh-converged at λ/4.** λ/5 is off by +5.9 % (p). The λ/4 pass is fortuitous.
- At **λ/6 (0.72 Si wavelengths) Si 45° is +0.18 % (p) / +0.03 % (s)**, and the power balance closes to 0.2 %;
  this is the Si 45° acceptance result. It needs a smaller geometry: the literal 7a geometry (w₀ = 0.75 µm,
  L = 7 µm) at λ/6 has 2N = 104 016, i.e. 173 GB dense (346 GB with the LU copy), which is infeasible densely.
  The run therefore uses w₀ = λ, L = 3 µm with the dark band relaxed to 1e-2 (measured ratio 5.9e-3). The
  dark-band level is an ADR 0009 decision-4 configuration check, not the 7a tolerance (< 1 %, unchanged), and it
  does not move R: the levels 5·10⁻⁵ … 2.6·10⁻² change R by ≤ 0.15 % at λ/4, and **at λ/6 the 5.9·10⁻³
  (L = 3 µm) and 2.6·10⁻² (L = 1.5 µm) geometries agree within 0.05 % (p) / 0.08 % (s)** (+0.1755 vs +0.1272 %,
  +0.0246 vs −0.0572 %). A λ/8 point (L = 1.5 µm) changes the λ/6 value by ≤ 0.16 %.
- Not the cause of any deficit: the quadrature (W1: ≤ 1·10⁻⁷), the gap/window (×1.5: ≤ 1.2·10⁻⁵) and the PML
  (W_pml or σ_max ×1.5: ≤ 2·10⁻⁷).
- **Rules from 7a:**
  - waist w₀ cos θ ≳ λ and a PML start beyond the defocused footprint (45°: w₀ = 0.75 µm, pml_start 4.5 µm);
  - gap = window = 0.5 µm;
  - W_pml = 2λ, σ_max = 4, m = 2;
  - h ≤ λ/4 for Ag and for Si at normal incidence, **h ≤ λ/6 for Si at oblique incidence**.

## Setup

| Parameter | Value |
|---|---|
| Surface | `geometry::make_open_surface_mesh` of a zero height map: patch L × L, taper W_t = 0.5 µm, gap W_g = 0.5 µm, PML W_pml = 1 µm = 2λ₁ (square bands, open outer edge), grid spacing h (cells split along one diagonal) |
| PML | `PmlTermination` defaults: σ_max = 4, order m = 2, dark band tolerance 1e-4 (amplitude at the PML-start square / max on the patch) |
| Media | λ = 500 nm, vacuum R1; Ag ε_r = −9.794 − 0.313j, Si ε_r = 18.478 − 0.606j |
| Beam | `AngularSpectrumBeam`, focus at the origin, incidence about y from z < 0, region radius 1.02 √2 pml_start (the unmasked part) |
| Operator | `Simulation` PML constructor, dense, `formulation::recommend` (Ag ICTF + Jacobi — unused by LU, Si ICTF), kernel defaults (`target_accuracy` 1e-5, `stretched_touching_points` 10) |
| Solver | dense LU of a copy of Z, LAPACK `zgetrf`/`zgetrs` (OpenBLAS); one factorisation for s and p (the second right-hand side from `op::assemble_rhs` of a second `Simulation` with the other beam, which also runs the dark-band and controlled-region checks); residuals ‖b − Zx‖/‖b‖ ≤ 3·10⁻¹⁵ |
| Reflected power | far field of the real part with the default raised-cosine window (inner = taper_end, outer = pml_start, across the gap), Dunavant degree 6, Gauss–Legendre θ (90) × φ (180) over k_z < 0; checks with half the nodes and with degree 8: ≤ 7·10⁻¹² / 6·10⁻⁹ |
| R_sim | P_refl / `power()`; diagnostics: A_real = ½ Re ∫ (n̂ × M)·J* over the triangles inside pml_start, 1 − R − A_real, edge decay max \|J\|, \|M\| on the outermost ring / max |
| Machine | Windows 11, 12 cores / 24 threads, 128 GB, win-release (GCC 16.1, OpenBLAS), 2026-10-13; main cases one at a time on an otherwise idle machine |

### Waist and PML start (the configuration rule)

The PML constructor requires the dark band (ADR 0009 decision 4): max |E_inc| on the PML-start square
≤ 1e-4 of its maximum on the patch. Measured ratios (`--estimate-only`, no assembly; p / s):

| θ | w₀ [µm] | L [µm] (pml_start) | ratio |
|---|---|---|---|
| 0° | 0.5 | 1.5 (1.75) | 1.3e-5 |
| 0° | 0.6 | 1.5 (1.75) / 3.0 (2.5) | 3.8e-4 / 9.0e-8 |
| 0° | 0.75 | 3.0 (2.5) / 4.0 (3.0) | 2.6e-5 / 2.3e-7 |
| 45° | 0.5 | 1.5 … 5.5 (1.75 … 3.75) | 2.6e-2 … 2.2e-3 (never below 2e-3) |
| 45° | 0.6 | 5.5 (3.75) | 5.8e-4 |
| 45° | 0.65 | 8.0 (5.0) | 1.2e-4 / 1.3e-4 |
| 45° | 0.7 | 6.5 (4.25) / 7.0 (4.5) | 1.1e-4 / 7.7e-5 |
| 45° | **0.75** | 6.5 (4.25) / **7.0 (4.5)** / 8.0 (5.0) | 8.9e-5 / **5.2e-5 / 4.0e-5** / 2.3e-5 |
| 45° | 0.8 | 7.0 (4.5) / 8.0 (5.0) | 5.4e-5 / 1.6e-5 |
| 45° | 0.9 | 7.0 (4.5) / 8.0 (5.0) | 1.1e-4 / 2.5e-5 |

The scan's `DARK` lines were printed to the terminal and are not stored raw (`pml_validation_rows.txt` holds
only the ROW lines of solved cases, whose `dark` field repeats the ratios of the solved geometries); the table
is reproducible with `--estimate-only` (see Reproduce; about a second per point, no assembly).

Two effects make the oblique case much harder than the provisional rule pml_start ≳ 3.5 w₀/cos θ
(ADR 0009, WP-PML3a) suggests (measured; the mechanisms are the standard beam formulas, fitted
only by eye):

1. **Grazing spectrum.** The rigorous beam cuts its spectrum at k_t = k about the *beam axis*; at
   θ = 45° that keeps plane waves running along the surface (90° from z) with relative amplitude
   exp(−(k w₀ cos θ)²/4) = 7·10⁻³ at w₀ = λ. They do not decay along z = 0, so at w₀ = 0.5 µm the
   ratio stays ≥ 2·10⁻³ for any patch. The waist must satisfy **w₀ cos θ ≳ λ** (0.75 µm at 45°).
2. **Defocus along the footprint.** At x on the surface the beam is a distance x sin θ from its
   focus; with the Rayleigh range π w₀²/λ (3.5 µm at w₀ = 0.75 µm) the footprint widens, so a wider
   waist can need a *larger* PML start (0.9 µm needs more than 0.75 µm).

The smallest grid points that pass at 45° (ratio ≤ 1e-4) are **w₀ = 0.75 µm, L = 6.5 µm** (pml_start
4.25 µm, 8.9e-5, 2N = 42 000 at λ/4) and **w₀ = 0.7 µm, L = 7 µm** (pml_start 4.5 µm, 7.7e-5, same 2N as
L = 7 µm below); for these two points the scan lists a single ratio each, not split into p and s (where both
are listed, at L = 7 µm for w₀ = 0.75 µm, p is the larger), so no s value is recorded for them. 7a uses
**w₀ = 0.75 µm, L = 7 µm (pml_start 4.5 µm ≈ 4.2 w₀/cos θ)**, ratio 5.2e-5 (p) / 4.0e-5 (s), i.e. ≈ 2×
below the 1e-4 tolerance: chosen with margin, not the cheapest point (+10 % unknowns over L = 6.5 µm).
2N = 46 112 at λ/4: 4× the unknowns of the 0° case (w₀ = λ, L = 1.5 µm, pml_start 1.75 µm = 3.5 w₀,
2N = 11 440). These are `pml_fresnel::seven_a_case()`.

## Results (the eight 7a cases, h = λ/4 = 125 nm)

| Case | w₀, L [µm] | 2N | dark band | R_sim | R_beam | R_sim/R_beam − 1 | \|r(θ₀)\|² | assembly / LU [s] |
|---|---|---|---|---|---|---|---|---|
| Si 0° p | 0.5, 1.5 | 11 440 | 1.3e-5 | 0.3880910 | 0.3881757 | **−0.022 %** | 0.3877182 | 233 / 8 |
| Si 0° s | 0.5, 1.5 | 11 440 | 1.3e-5 | 0.3880910 | 0.3881757 | **−0.022 %** | 0.3877182 | (same LU) |
| Ag 0° p | 0.5, 1.5 | 11 440 | 1.3e-5 | 0.9816290 | 0.9816671 | **−0.0039 %** | 0.9816493 | 237 / 65 |
| Ag 0° s | 0.5, 1.5 | 11 440 | 1.3e-5 | 0.9816290 | 0.9816671 | **−0.0039 %** | 0.9816493 | (same LU) |
| Si 45° p | 0.75, 7 | 46 112 | 5.2e-5 | 0.2553905 | 0.2573421 | **−0.758 %** | 0.2602448 | 2 216 / 444 |
| Si 45° s | 0.75, 7 | 46 112 | 4.0e-5 | 0.5104525 | 0.5125628 | **−0.412 %** | 0.5101420 | (same LU) |
| Ag 45° p | 0.75, 7 | 46 112 | 5.2e-5 | 0.9719370 | 0.9746550 | **−0.279 %** | 0.9747743 | 1 054 / 534 |
| Ag 45° s | 0.75, 7 | 46 112 | 4.0e-5 | 0.9858288 | 0.9873075 | **−0.150 %** | 0.9873066 | (same LU) |

Wall time per geometry (both polarisations): 0° 4–5 min, 45° Si 45 min, Ag 27 min; peak memory
4.3 GB (0°) and 68.2 GB (45°: matrix + LU copy). Far field ≤ 11 s. At 0° s and p coincide to all
printed digits: the mesh is symmetric under the mirror x ↔ y about the plane x = y, which swaps E_x and E_y
(p ↔ s at normal incidence) and maps the diagonal split of every cell (corners (i, j)–(i+1, j+1),
`open_surface.cpp`) onto itself. A 90° rotation would not do this: it maps that diagonal onto the other one.

Diagnostics (all cases): edge decay |J|, |M| at the truncation ≤ 8·10⁻⁹ of their maximum; hemisphere
grid and degree checks ≤ 7·10⁻¹² and 6·10⁻⁹; specular cone (3λ/(πw₀)) ≥ 0.999999 of P_refl;
θ > 80° ≤ 4·10⁻⁸. **A_real is not a usable absorption measure at λ/4**: Si 0° 0.6045 (1 − R − A =
+0.74 %), Si 45° 0.685 / 0.456 (+6.0 / +3.3 %), Ag 0° 0.0138 vs Fresnel 0.0184, Ag 45° *negative*
(−0.048 / −0.015). The local (n̂ × M)·J* product is first-order sensitive to current errors (WP-A1
found the same for Ag in the box); A_ext (ADR 0006 amendment 2026-10-12) remains the absorption
measure, and it is not part of 7a.

## Mesh convergence

R_sim / R_beam − 1 (p / s) and the power balance 1 − R − A_real (p / s) against h. The 45° study uses
a **smaller geometry** (w₀ = λ = 0.5 µm, L = 3 µm, pml_start 2.5 µm; dark band 5.9·10⁻³ / 5.6·10⁻³,
accepted with `--dark-band-tol 1e-2`), because the literal 7a geometry (w₀ = 0.75 µm, L = 7 µm) has
2N = 104 016 at λ/6, 173 GB dense (346 GB with the LU copy): infeasible densely on this machine. The
relaxed dark band is an ADR 0009 decision-4 configuration check, not the 7a tolerance, which is unchanged.
At λ/4 the two 45° geometries differ by ≤ 0.06 % in R (Si −0.70 / −0.48 % vs −0.76 / −0.41 %,
Ag −0.32 / −0.15 % vs −0.28 / −0.15 %), so the dark band level is not what limits R; the same holds at λ/6
(λ/8 finding below: 5.9·10⁻³ vs 2.6·10⁻² within 0.05 % (p) / 0.08 % (s)).

| Case | h | 2N | R_sim/R_beam − 1 | 1 − R − A_real | assembly [s] |
|---|---|---|---|---|---|
| Si 0° p (L = 1.5 µm) | λ/4 | 11 440 | −0.022 % | +0.74 % | 233 |
| Si 0° p (L = 1.6 µm¹) | λ/5 | 18 592 | +0.0086 % | +0.037 % | 415² |
| Si 0° p (L = 1.5 µm) | λ/6 | 25 872 | +0.0008 % | +0.006 % | 710² |
| Ag 0° p = s | λ/4 | 11 440 | −0.0039 % | +0.45 % | 237 |
| Ag 0° p = s | λ/6 | 25 872 | −0.0004 % | +0.14 % | 684² |
| Si 45° p / s (small) | λ/4 | 18 592 | −0.699 / −0.475 % | +6.2 / +3.3 % | 432 |
| Si 45° p / s (small) | λ/5 | 29 120 | **+5.902 / −1.430 %** | +5.3 / +3.3 % | 756² |
| Si 45° p / s (small) | λ/6 | 42 000 | **+0.176 / +0.025 %** | +0.21 / +0.17 % | 1 259² |
| Ag 45° p / s (small) | λ/4 | 18 592 | −0.321 / −0.151 % | +7.9 / +2.8 % | 882² |
| Ag 45° p / s (small) | λ/6 | 42 000 | −0.039 / −0.009 % | +2.4 / +0.8 % | 1 844³ |

¹ λ/5 = 100 nm does not divide L/2 = 0.75 µm; L = 1.6 µm keeps the zone boundaries on grid lines.
² Run beside a debug build / ctest (times inflated, results unaffected). ³ Two identical copies of the
run were accidentally executed concurrently (identical results to all printed digits; time ≈ 2×).

Findings (measured):
- **0° converges monotonically** for both materials (Si −0.022 → +0.009 → +0.001 %; Ag −0.004 →
  −0.0004 %); λ/4 is ample at normal incidence, where the currents carry no tangential phase.
- **Ag at 45° converges** (−0.32 → −0.04 % for p, faster than h⁴ between λ/4 and λ/6), so the λ/4 pass of
  the 7a table is a genuine pass with ~0.3 % discretisation error.
- **Si at 45° is not converged at λ/4 and not monotone**: λ/4 −0.70 %, λ/5 **+5.9 %** (p), λ/6
  +0.18 %. The power balance tracks it: 1 − R − A_real is 5–6 % (p) at λ/4 and λ/5 and 0.2 % at
  λ/6. h = λ/4 = 125 nm is 1.08 Si wavelengths (λ/n_Si = 116 nm), λ/5 0.86, λ/6 0.72: the Si
  interior field under the oblique beam is not resolved above ~λ_Si (hypothesis for the
  mechanism; the outlier at λ/5 shows that the λ/4 pass of the 7a table is **fortuitous**).
  At normal incidence the same under-resolution costs only 0.02 % (no tangential phase).
- **λ/8 confirms the λ/6 level** on an even smaller 45° geometry (w₀ = 0.5 µm, L = 1.5 µm,
  pml_start 1.75 µm, dark band 2.6·10⁻² / 2.0·10⁻² accepted with `--dark-band-tol 1e-1`; R_beam p
  0.2543137, s 0.5157891):

  | h | 2N | Si 45° p / s | 1 − R − A_real p / s | assembly / LU [s] |
  |---|---|---|---|---|
  | λ/4 | 11 440 | −0.752 / −0.557 % | +6.2 / +3.4 % | 282 / 10 |
  | λ/6 | 25 872 | +0.127 / −0.057 % | +0.24 / +0.22 % | 701 / 99 |
  | λ/8 | 46 112 | −0.034 / −0.073 % | +0.070 / +0.074 % | 1 565 / 481 |

  The λ/6 → λ/8 change is 0.16 % (p) / 0.016 % (s), so the Si 45° value at λ/6 is converged to
  ≈ 0.2 %; the three geometries agree at λ/4 within 0.06 % (p) and 0.15 % (s), i.e. the dark band
  from 5·10⁻⁵ to 2.6·10⁻² moves R by less than the discretisation error.
  **At λ/6 the dark band does not matter either:** the 5.9·10⁻³ geometry (L = 3 µm, ROW lines 15/16 =
  file lines 21/22 of the rows file: R_sim 0.2547605 / 0.5159166, +0.1755 / +0.0246 %) and the 2.6·10⁻²
  geometry (L = 1.5 µm, ROW lines 41/42 = file lines 47/48: R_sim 0.2546372 / 0.5154938, +0.1272 /
  −0.0572 %) agree within **0.05 % (p) / 0.08 % (s)** (R_sim ratio − 1: 4.8·10⁻⁴ / 8.2·10⁻⁴), so the
  converged Si 45° value does not depend on the relaxed dark band of `seven_a_si45_fine_case()`.

**Mesh rule from 7a:** h ≤ λ/4 for Ag (and for Si at normal incidence); **h ≤ λ/6 ≈ 0.7 λ/n_Si for
Si at oblique incidence**, checked by the power balance |1 − R − A_real| ≲ 0.5 % (A_real is a
resolution indicator for Si only; for Ag the local product is not resolved at λ/6 either, +2.4 %).
The mesh-converged Si 45° result is **+0.176 % (p) / +0.025 % (s)**, within 7a; it is the case
`seven_a_si45_fine_case()` of the validation-large test (relaxed dark band, see above).

## Sensitivities (45°, small geometry, h = λ/4)

Each a single extra solve of the 45° small-geometry case (w₀ = 0.5 µm, L = 3 µm, dark band 5.9e-3,
2N = 18 592) instead of the 7a geometry (2N = 46 112, 27–45 min per solve); R printed to 7 digits,
so "0" means a relative change below ≈ 2·10⁻⁷.

| Perturbation | Si p | Si s | Ag p | Ag s |
|---|---|---|---|---|
| base R_sim | 0.2525366 | 0.5133394 | 0.9714301 | 0.9858278 |
| (i) quadrature: `target_accuracy` 1e-6 (10× tighter), `stretched_touching_points` 12 | 0 | 0 | −1·10⁻⁷ | +1·10⁻⁷ |
| (ii) gap and window ×1.5 (W_g 0.75 µm, window 2.0 → 2.75 µm) | +1.2·10⁻⁵ | +1.0·10⁻⁵ | +5.1·10⁻⁶ | +6.5·10⁻⁶ |
| (iii a) W_pml ×1.5 (1.5 µm) | 0 | 0 | 0 | 0 |
| (iii b) σ_max ×1.5 (6) | 0 | 0 | 0 | +1·10⁻⁷ |

- (i) The WP-PML2 review item W1 (capped near/far pairs in the deep PML) does not reach R:
  ≤ 1·10⁻⁷ ≪ 1e-3. The tighter quadrature cost 1.25× (Si: 539 vs 432 s) and ≈ 1.0× (Ag: 425 s vs 428 s for
  σ_max 6 at default quadrature) of the assembly time; σ_max 6 costs Si 1.4× (k-aware touching
  order), W_pml 1.5 µm 1.6–1.8× (more PML triangles).
- (ii) The window across a 0.5 µm (= λ) gap is converged to ~1e-5: the Huygens-plane fallback of
  ADR 0009 decision 5 is not needed for the flat surface.
- (iii) Preview of 7c (report only; 7c proper uses the far-field metric of docs/05 on a rough
  surface, WP-PML4): the flat-surface R is blind to the PML parameters at the 2·10⁻⁷ level, i.e. the
  PML reflection of what reaches it (dark band 6e-3 here) does not feed back into the specular
  power. Edge decay at the truncation: 1.6·10⁻⁷ (W 1 µm, σ 4) → 1.9·10⁻⁸ (σ 6) → 1.1·10⁻⁹ (W 1.5 µm).
  For Ag the SPP channel (7b) is the real test of the PML; it is not excited by a flat surface.

## Validation-large test

`tests/validation_large/test_pml_fresnel_large.cpp`, five cases with label `validation-large` and RUN_SERIAL:
- the acceptance cases Si 0°, Ag 0° and Ag 45° (`seven_a_case()`, s and p each), with `REQUIRE`/`CHECK`
  |R_sim/R_beam − 1| < 1 %, residual < 1e-8 and edge decay |J|, |M| < 1e-3;
- the Si 45° acceptance case `seven_a_si45_fine_case()` (h = λ/6), with the same checks plus
  |1 − R − A_real| < 1 %;
- "PML 7a: flat Si, 45 deg, h = lambda / 4 (not mesh-converged, reported only)" (`seven_a_case("si", 45)`):
  residual and edge-decay checks, but its error against R_beam is only reported (`WARN`), because the λ/4 pass
  is fortuitous (λ/5: +5.9 %). Acceptance for Si 45° is the λ/6 case.

Run 2026-10-14 (win-release, after merging `main` with WP-PML2c), partly beside a debug ctest / pytest; at that
time the Si 45° λ/4 case still carried the 1 % `CHECK` (changed to report-only in the review, results unchanged):

| Test | Result | Time [s] |
|---|---|---|
| Ag 0° | passed (−0.0039 % / −0.0039 %) | 327 |
| Ag 45° | passed (−0.2789 % / −0.1498 %) | 2 491 |
| Si 0° | passed (−0.0218 % / −0.0218 %) | 296 |
| Si 45° (λ/4, now reported only) | within 1 % (−0.7584 % / −0.4117 %), not an acceptance result | 3 201 |
| Si 45° (λ/6, fine) | passed (+0.1755 % / +0.0246 %; 1 − R − A_real 0.21 / 0.17 %) | 1 559 |

The values are identical to the benchmark rows. The fine case's edge-decay check was added in the review; its
measured decay is ≤ 4.9·10⁻⁸ (ROW lines 15/16), far below the 1e-3 limit. The unit/validation suites are unchanged by this WP: 492 ctest in
win-release and win-debug, 121 pytest.

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
# run with C:/msys64/ucrt64/bin first on PATH (libstdc++ DLL clash with Git's mingw64)
B=build/win-release/benchmarks/specklebem_pml_fresnel
$B --material si --theta 0 --pol both                                   # Si 0 deg (defaults: w0 0.5 um, L 1.5 um, h 125 nm)
$B --material ag --theta 0 --pol both
$B --material si --theta 45 --w0 0.75e-6 --patch 7e-6 --pol both       # Si 45 deg
$B --material ag --theta 45 --w0 0.75e-6 --patch 7e-6 --pol both
$B --material si --theta 45 --w0 0.75e-6 --patch 7e-6 --estimate-only  # sizes + dark-band ratios only
$B --material si --theta 45 --pol both --patch 3e-6 --dark-band-tol 1e-2 --h 83.33333333333333e-9  # Si 45 deg, lambda/6
$B --material si --theta 45 --pol both --patch 1.5e-6 --dark-band-tol 1e-1 --h 62.5e-9           # Si 45 deg, lambda/8 series
$B --material ag --theta 45 --pol both --patch 3e-6 --dark-band-tol 1e-2 --acc 1e-6 --stp 12     # quadrature sensitivity
ctest --preset win-release -L validation-large -R "PML 7a"             # the 7a cases as a test
```
