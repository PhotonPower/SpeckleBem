# PML route validation (ADR 0009 decision 7; WP-PML3b: 7a, WP-PML4a: 7b)

The sections from "Setup" to "Validation-large test" are validation 7a (flat Fresnel). Validation 7b
(SPP reflection at the PML start, **met**: |r_spp| ≈ 8·10⁻⁵ at the defaults) is the section
"7b SPP absorption" after them.
Validation 7e (rough Si, PML route vs the uniform box, WP-PML4c) has its own record: `pml_si_crosscheck.md`.

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

## 7b SPP absorption

Acceptance criterion: ADR 0009 decision 7b / `docs/05_validation.md` "PML route" row 7b — a surface plasmon
polariton launched on a flat Ag open surface must be reflected by the PML start with **|r_spp| < 1e-2**.

Executable `benchmarks/pml_spp.cpp` (`specklebem_pml_spp`, `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`, one geometry per
process), shared code `tests/support/pml_spp_support.hpp`, unit tests of the fit `tests/unit/test_pml_spp.cpp`,
test `tests/validation_large/test_pml_spp_large.cpp` (label `validation-large`, release only, memory-guarded).
Raw lines: `pml_validation_rows.txt` (the block "WP-PML4a").

**Verdict: 7b met.** At the defaults (σ_max = 4, m = 2, W_pml = 2λ), h = λ/5, the groove launch gives
**|r_spp| = 8.0·10⁻⁵ (J_x) / 3.4·10⁻⁵ (M_y)**, 125× below the 1e-2 limit and at the measurement floor
(≈ 1–2·10⁻⁴). **The defaults are confirmed:** every profile with W_pml = 2λ or 3λ and m ≥ 2 that was run
(σ_max 2 … 8) is at the floor; m = 1 is ≈ 10× above the measurement floor (1.75·10⁻³ / 1.19·10⁻³), m = 3
is not measurably better. At W_pml = λ the result depends on σ_max and on h (σ_max 4: 1.3·10⁻³ at λ/5,
1.1·10⁻⁴ at λ/6); for σ_max 1 … 8 every value is below 1e-2 (σ_max 0.25 / 0.5 are not: J_x 0.15 / 0.052).
The acceptance value r_spp = max(J_x, M_y) is conservative: the scan conclusions rest on M_y, the
SPP-clean observable (see "Consistency of J_x and M_y" below).

### Method

**Launch.** The open surface (vacuum R1 over Ag R2, ε_r = −9.794 − 0.313j, λ = 500 nm) is flat except for a
Gaussian groove ξ = d exp(−ρ²/r_g²), d = 50 nm (into the metal), r_g = 100 nm, at the origin, illuminated at
normal incidence by the rigorous angular-spectrum beam (w₀ = λ, E along x). Two launches:
- *groove*: the grooved minus the flat solution (same grid, only z differs, so the RWG numbering is the same).
  This is the field scattered by the groove: an SPP along ±x (cos φ pattern), radiation and the
  quasi-cylindrical wave. The errors of the masked incidence at the PML start are common to both solves and
  cancel. Two dense assemblies.
- *local source*: the flat operator with the beam's right-hand side kept only on the test functions whose edge
  midpoint lies within 150 nm of the origin (an impressed source on the interface). It solves with the same
  LU as the flat beam, so it costs one assembly.

They agree (defaults, h = λ/5): |r| J_x 7.97·10⁻⁵ (groove) / 7.90·10⁻⁵ (local), M_y 3.40·10⁻⁵ / 3.51·10⁻⁵,
arg r(J_x) +1.550 / +1.527, arg r(M_y) −0.633 / −0.654, i.e. to 1–3 % in |r| and 0.02 rad in phase
(|r_groove − r_local| ≈ 2·10⁻⁶ for J_x and 1·10⁻⁶ for M_y). Since two different sources and solves give the
same complex r far more closely than the spread over fit variants (the floor below), the ≈ 8·10⁻⁵ /
3.4·10⁻⁵ at fixed fit options is a deterministic reflection of the discretised PML (as seen by this fit),
not fit noise. They agree because the reflection coefficient of the
guided mode is a property of the PML: the source sets only the SPP amplitude and the amount of non-SPP field
(radiation, quasi-cylindrical wave), which the background terms of the fit absorb in M_y (in J_x of the local
source only partly, see "Consistency of J_x and M_y"). The scan therefore uses the local source (half the
cost); the validation-large test uses the groove and checks the M_y agreement (|r_groove − r_local| < 2·10⁻⁴).

The dark band is measured (2.9·10⁻⁴ at pml_start 1.5 µm, w₀ = λ) and accepted with the tolerance relaxed to
1e-2 (`make_termination`): neither launch sees the beam at the PML start (the groove difference cancels it, the
local source is zero there), so the decision-4 check guards nothing here.

**Observables.** Along the x-axis (both half-axes, ρ = |x|) three quantities are sampled, each of which is
one exact cylindrical form on the axis for a TM SPP with E_z ∝ cos φ H₁⁽²⁾(kρ):
- J_x at the midpoints of the edges perpendicular to x (|y| = h/2, averaged; the normal component across an
  edge is the edge coefficient), ∝ H₁⁽²⁾′(kρ);
- M_y at the midpoints of the edges perpendicular to y on y = 0, ∝ H₁⁽²⁾′(kρ);
- div_s J per triangle (centroid), ∝ H₁⁽²⁾(kρ).

**M_y is the clean observable.** It has the smallest fit residual (4·10⁻⁵ at the defaults, J_x 1.4·10⁻⁴,
div J 1.8·10⁻²), its fitted wavenumber stays inside the scan range in every run (k/k_spp 0.967–1.012), and
it passes the geometric check below. In the weak PMLs its reflection follows the round trip through the PML:
with the attenuation exponent A = 2 Re(k_spp) σ_max W_pml/(m + 1) (the amplitude of the wave reflected at the
open outer edge falls by e^−A), |r(M_y)| e^A = 0.052, 0.047, 0.039 for σ_max = 0.25, 0.5, 1 (W = λ, m = 2), i.e. an outer-edge
reflection of ≈ 0.04–0.05 attenuated twice by the PML. J_x is contaminated in weak PMLs: there its fitted
wavenumber runs to the scan bound (k/k_spp = 1.04 at σ_max 0.25 and 0.5), |r| is 7–10× that of M_y and its
phase is not start-referred (geometric check). Presumably an unmodelled reflected non-SPP wave (grazing
radiation returning from the outer edge) enters H_y more than E_x; not investigated further. For A ≥ 6.6
(every run except σ_max ≤ 1 at W = λ) J_x and M_y agree within a factor 3 (J_x/M_y = 0.9–2.9); at W = λ,
σ_max 1 (A = 4.4) J_x is 6.8× M_y (3.2·10⁻³ against 4.7·10⁻⁴). Both are reported; the 7b value r_spp is
the larger of the two. div J (piecewise constant per triangle) is reported only (defaults: groove
2.3·10⁻⁴, local 1.8·10⁻³, residual 1.6–1.8·10⁻²).

**Consistency of J_x and M_y (local-source J_x is not SPP-clean).** For a TM SPP the tangential fields on
the interface satisfy |E_x/H_y| = |κ|/(ωε₀) with κ = (k_spp² − k₀²)^½ = 0.337 k₀, i.e. 0.337 η₀ = 127.0 Ω
(`pml_spp::spp_impedance`). The fitted outgoing amplitudes of a clean SPP must therefore satisfy
|c_out(M_y)/c_out(J_x)| = 127 Ω (with n̂ = −ẑ, M = −n̂ × E and J = n̂ × H give M_y = E_x, J_x = H_y). The study prints this ratio
(`ZOUT` lines, `|out My/Jx|` in the ROW line; `pml_spp::out_impedance`), and refitting all dumps
(`--refit`, no solve) reproduces the values of the run logs: the **groove** launch gives **111 Ω** (0.88 of
the SPP value), the **local source 18–50 Ω** in every run (0.14–0.40; defaults 30.5 Ω, W = λ σ_max 8
18.0 Ω, W = λ σ_max 4 50.2 Ω), i.e. the outgoing J_x column of the local source carries 2.5–7× more field
than its SPP part, presumably the non-SPP near field of the impressed source (not investigated). The
local-source J_x values are therefore not normalised by the SPP amplitude (r = c_in/c_out with a c_out that
is not the SPP's); at the defaults they nevertheless agree with the groove's (7.90 vs 7.97·10⁻⁵), whether
that holds in the other runs is not known. **The scan conclusions rest on M_y**, which shows the same trends
as J_x in every table below; r_spp = max(J_x, M_y) is kept as the conservative acceptance value.

**Fit model.** Between ρ_min = 0.6 µm (k_spp ρ_min = 8.0, above the Hankel expansion's bound 6) and the PML
start a = 1.5 µm, by least squares:

y(ρ) = c_out w(kρ) + c_in w(k(2a − ρ)) + Σ_p c_p e^{−jk₀ρ}(k₀ρ)^{−p},  p ∈ {1, 1.5, 2},

with w = H₁⁽²⁾′ (J_x, M_y) or H₁⁽²⁾ (div J), the reflected wave as the cylindrical wave of the image source
at 2a, and r_spp = c_in/c_out referred to the PML start (both waves have the value w(ka) there). Re k is fitted
(scan ±4 % about k_spp, golden-section refinement) so that the numerical dispersion of the mesh does not leak
into the reflected amplitude. The fitted k/k_spp is 0.97–1.01; J_x and M_y of the same run differ by 1–3 %,
because over a window of ≈ 2 λ_spp k is nearly degenerate with the background columns: k is a nuisance
parameter (with 1e-4 noise on a dense grid the unit test holds r to 1e-4 but k only to 2·10⁻³). Im k =
Im k_spp (the SPP decays over 41.5 µm, ≫ the window). The window holds 9 (M_y) / 10 (J_x) samples per
half-axis at h = λ/5, 11 at λ/6 and 13 at λ/7. The ±x half-axes give identical values: the grid, the groove,
the PML and the beam (up to its sign, as the disc of the local source) are invariant under the rotation by
π about the z-axis (C2), which maps the samples on +x onto those on −x; this symmetry alone fixes the
identity, so the second half-axis carries no extra information. The background terms are needed: without them (`--bg none`) the fit is model-limited
(groove, defaults: |r| 4.2·10⁻³ for both J_x and M_y with fit residuals 1.2·10⁻² / 1.4·10⁻²,
against 8.0·10⁻⁵ / 3.4·10⁻⁵ and 1.4·10⁻⁴ / 4.4·10⁻⁵ with them): the radiated and quasi-cylindrical parts of
the launched field, which decay like (k₀ρ)^−1 … (k₀ρ)^−2 and are not SPP, are then forced into the reflected
column. The unit tests validate the fit on synthetic data: exact recovery of r, k and the background
(≤ 1e-8); relative noise 1e-4 with a fixed seed, on a dense grid (λ/12 over 0.6–2.0 µm, 17–34 samples:
|Δr| < 1e-4 for r = 0, 1e-3, 1e-2) and at the real 7b sampling (h = 100 nm, plane 1.5 µm, M_y 9 samples
0.65–1.45 µm, J_x 10 samples 0.6–1.5 µm, Re k offset ±1.2 %, a background power outside the model, 24 fits
per r): there the few samples amplify the noise about 10× and more so for larger r (through the
nearly degenerate k), max |Δr| = 1.3·10⁻⁴ / 3.8·10⁻⁴ / 9.6·10⁻⁴ (median 5·10⁻⁵ / 7·10⁻⁵ / 3.1·10⁻⁴) for
r = 0 / 1e-3 / 1e-2 (test tolerance 2·10⁻³, which still separates 1e-2 from 0; noise-free the error is
≤ 1.5·10⁻⁵); a background power outside the model ({1.25}, as strong as the SPP at ρ_min: |Δr| < 2·10⁻⁴);
a reflected wave of the planar asymptotic form e^{−jk(2a−ρ)}/(2a − ρ)^½ instead of the model column
(|r| to 3 %, the O(1/(ka)) asymptotic accuracy, ka = 19.9); and the plane shift (r rotates by 2 Re(k) δ).

**Measurement floor.** Refitting the dumps (`--refit`, no solve) with ρ_min = 0.5 / 0.6 / 0.8 µm, background
sets {1.5, 2} / {1, 1.5, 2} / {1, 1.5, 2, 2.5} and with Im k fitted spreads |r| as follows (+x side, min … max):

| Run (h = λ/5 unless marked) | \|r\| J_x | \|r\| M_y |
|---|---|---|
| default, groove | 4.1·10⁻⁵ … 1.2·10⁻⁴ | 1.3·10⁻⁵ … 1.1·10⁻⁴ |
| default, local | 4.0·10⁻⁵ … 1.4·10⁻⁴ | 1.3·10⁻⁵ … 1.2·10⁻⁴ |
| W 2λ (σ_max 2, 3, 5, 6, 8; m 3) and W 3λ | 1.5·10⁻⁵ … 1.9·10⁻⁴ | 3.5·10⁻⁶ … 1.8·10⁻⁴ |
| default at λ/6 and λ/7 | 5.5·10⁻⁶ … 5.0·10⁻⁵ | 4.7·10⁻⁶ … 1.2·10⁻⁴ |
| W λ, σ_max 2 (λ/5; λ/6) | 1.4·10⁻⁵ … 1.4·10⁻⁴; 1.9·10⁻⁵ … 1.1·10⁻⁴ | 1.1·10⁻⁵ … 1.8·10⁻⁴; 1.1·10⁻⁵ … 1.7·10⁻⁴ |
| W λ, σ_max 3 | 1.8·10⁻⁴ … 5.9·10⁻⁴ | 1.4·10⁻⁴ … 6.5·10⁻⁴ |
| W λ, σ_max 4 (λ/5; λ/6) | 8.4·10⁻⁴ … 1.7·10⁻³; 1.3·10⁻⁵ … 1.5·10⁻⁴ | 4.1·10⁻⁴ … 1.2·10⁻³; 5.3·10⁻⁵ … 3.5·10⁻⁴ |
| W λ, σ_max 8 | 1.9·10⁻³ … 5.5·10⁻³ | 2.1·10⁻³ … 8.1·10⁻³ |
| W λ, σ_max 1 | 3.1·10⁻³ … 4.7·10⁻³ | 2.7·10⁻⁴ … 5.8·10⁻⁴ |
| m 1 (W λ σ_max 2 = W 2λ σ_max 4) | 3.7·10⁻⁴ … 3.7·10⁻³ | 8.8·10⁻⁴ … 2.3·10⁻³ |
| W λ, σ_max 2, m 3 | 6.8·10⁻⁴ … 7.9·10⁻⁴ | 2.5·10⁻⁴ … 3.4·10⁻⁴ |
| weak PML σ_max 0.25 (a = 1.5; 1.6 µm) | 0.10 … 0.41; 0.047 … 0.11 | 1.3·10⁻² … 2.0·10⁻²; 1.4·10⁻² … 2.8·10⁻² |

The floor is therefore ≈ 1–2·10⁻⁴: values below it (all W ≥ 2λ runs with m ≥ 2) are "≲ 2·10⁻⁴", not
ordered among themselves. Above the floor the spread is a factor 1.2–3 (m = 1: up to 10 in J_x), which does
not change any conclusion. The spreads are in `pml_validation_rows.txt` (SPREAD lines).

**Geometric check (the fitted reflection is tied to the PML, and the plane reference is correct).** A weak
PML (σ_max = 0.25, W = λ, m = 2, local source) reflects measurably; moving its start by h (a = 1.5 → 1.6 µm,
same W, so the outer edge moves with it) gives |r(M_y)| = 1.716·10⁻² / 2.285·10⁻² with
arg r = −0.5907 / −0.5917: **the start-referred phase changes by 1.0·10⁻³ rad**, whereas referred to a fixed
plane it would rotate by 2 Re(k_spp) h = 2.65 rad. In this weak PML the reflection comes from the outer edge
(the e^A scaling above), so the check does not locate it at the start: it shows that the fitted reflection
moves with the PML (start and edge together), not with a fixed feature of the surface such as the taper end,
and that referring r to the PML start plane is consistent. The magnitude changes by 33 %, within the refit
spreads of the two runs (1.3–2.0 vs 1.4–2.8·10⁻²; the window gains one sample).
J_x fails this check (|r| 0.147 / 0.099, arg +0.86 / +1.31, k at the scan bound), which is why M_y is the
clean observable. This is the second case of the validation-large test (phase change < 0.2 rad, |r| > 5e-3).

### Geometry, mesh, solver

| Parameter | Value |
|---|---|
| Surface | `geometry::make_open_surface_mesh`, patch L = 1 µm (holds the groove), taper 0.5 µm, gap 0.5 µm, PML start a = 1.5 µm, W_pml = λ / 2λ / 3λ (outer half width 2.0 / 2.5 / 3.0 µm), grid h = 100 nm = λ/5 (scan), λ/6, λ/7 (mesh check) |
| SPP | k_spp = (1.05526 − 0.00192j) k₀, λ_spp = 473.8 nm (h = λ/5 is 4.7 points per λ_spp), amplitude decay length 41.5 µm |
| Operator | `Simulation` PML constructor, dense, `formulation::recommend` → ICTF, kernel defaults (`target_accuracy` 1e-5, `stretched_touching_points` 10) |
| Solver | LU in place (LAPACK `zgetrf`, OpenBLAS; without LAPACK Eigen's `PartialPivLU` on an `Eigen::Ref` of the same storage; peak = one matrix, 16 (2N)² bytes), several right-hand sides per factorisation; residual on sampled rows ≤ 9·10⁻¹⁴ |
| Machine | Windows 11, 12 cores / 24 threads, 128 GB, win-release (GCC 16.1, OpenBLAS), 2026-10-14/15, one run at a time, but the machine was shared with other jobs (load not recorded): equal-size runs differ in time by up to 3× (assembly) and 17× (LU) |

### Results (h = λ/5 = 100 nm; local source unless marked)

A = 2 Re(k_spp) σ_max W_pml/(m + 1): round-trip attenuation exponent (outer-edge reflection ∝ e^−A).

| W_pml | σ_max | m | A | \|r\| J_x | \|r\| M_y | 2N | assembly / LU [s] |
|---|---|---|---|---|---|---|---|
| λ | 0.25 | 2 | 1.1 | 1.5·10⁻¹ | 1.7·10⁻² | 9 440 | 48 / 84 |
| λ | 0.5 | 2 | 2.2 | 5.2·10⁻² | 5.1·10⁻³ | 9 440 | 39 / 18 |
| λ | 1 | 2 | 4.4 | 3.2·10⁻³ | 4.7·10⁻⁴ | 9 440 | 48 / 19 |
| λ | 2 | 2 | 8.8 | 4.0·10⁻⁵ | 4.4·10⁻⁵ | 9 440 | 131 / 12 |
| λ | 3 | 2 | 13.3 | 5.9·10⁻⁴ | 2.6·10⁻⁴ | 9 440 | 96 / 5 |
| λ | 4 | 2 | 17.7 | 1.3·10⁻³ | 4.5·10⁻⁴ | 9 440 | 156 / 17 |
| λ | 8 | 2 | 35.4 | 2.6·10⁻³ | 2.1·10⁻³ | 9 440 | 341 / 32 |
| λ | 2 | 1 | 13.3 | 1.75·10⁻³ | 1.19·10⁻³ | 9 440 | 171 / 20 |
| λ | 2 | 3 | 6.6 | 7.7·10⁻⁴ | 3.2·10⁻⁴ | 9 440 | 116 / 17 |
| 2λ | 2 | 2 | 17.7 | 3.9·10⁻⁵ | 1.8·10⁻⁵ | 14 800 | 422 / 51 |
| 2λ | 3 | 2 | 26.5 | 5.6·10⁻⁵ | 2.5·10⁻⁵ | 14 800 | 463 / 65 |
| **2λ (default)** | **4** | **2** | 35.4 | **7.9·10⁻⁵** (groove 8.0·10⁻⁵) | **3.5·10⁻⁵** (groove 3.4·10⁻⁵) | 14 800 | 1 050 / 68 (groove: two assemblies) |
| 2λ | 5 | 2 | 44.2 | 9.1·10⁻⁵ | 4.1·10⁻⁵ | 14 800 | 472 / 16 |
| 2λ | 6 | 2 | 53.0 | 8.1·10⁻⁵ | 3.5·10⁻⁵ | 14 800 | 735 / 61 |
| 2λ | 8 | 2 | 70.7 | 6.3·10⁻⁵ | 4.7·10⁻⁵ | 14 800 | 611 / 21 |
| 2λ | 4 | 1 | 53.0 | 1.75·10⁻³ | 1.19·10⁻³ | 14 800 | 604 / 58 |
| 2λ | 4 | 3 | 26.5 | 2.8·10⁻⁵ | 2.0·10⁻⁵ | 14 800 | 509 / 80 |
| 3λ | 4 | 2 | 53.0 | 3.6·10⁻⁵ | 1.6·10⁻⁵ | 21 360 | 1 010 / 95 |

Peak memory 1.5 GB (2N = 9 440), 3.6 GB (14 800), 7.4 GB (21 360). Fit residuals ≤ 3·10⁻³ (J_x) in all
absorbing PMLs (A ≥ 4).

Findings (measured; the mechanisms are interpretations consistent with the numbers, not fitted):
- **Two reflection mechanisms.** (i) The outer edge, attenuated by e^−A (|r(M_y)| ≈ 0.045 e^−A): it
  dominates for A ≲ 5 (W = λ, m = 2, σ_max ≤ 1; the estimate gives 5.5·10⁻⁴ at σ_max 1 against the measured
  4.7·10⁻⁴). Beyond that it is not visible: at σ_max 2 (A = 8.8) the estimate is 7·10⁻⁶, below the measured
  4·10⁻⁵ at the floor. (ii) The onset of the stretch, discretised on the grid: it grows with
  the profile's steepness at fixed h (W = λ, m = 2: σ_max 3 → 4 → 8 gives M_y 2.6·10⁻⁴ → 4.5·10⁻⁴ →
  2.1·10⁻³) and falls with h (mesh check below). At W = λ the two cross between σ_max 1 and 3, and the
  4·10⁻⁵ measured at σ_max 2 (at the floor) is a sharp optimum (σ_max 1: 3·10⁻³, 3: 6·10⁻⁴ in J_x), not a
  robust operating point.
- **W_pml = 2λ (and 3λ) is a broad optimum:** σ_max 2 … 8 with m = 2 all give J_x 4–9·10⁻⁵ and M_y
  2–5·10⁻⁵, i.e. at the floor; A ≥ 18 makes (i) negligible and the onset is spread over 10 cells.
- **m = 1 is ≈ 10× above the measurement floor** (1.75·10⁻³ / 1.19·10⁻³ in J_x / M_y against a floor of
  ≈ 1–2·10⁻⁴; the m = 2 defaults, 7.9·10⁻⁵ / 3.5·10⁻⁵, are below the floor, so no larger factor is
  measured): the linear profile has a kink (σ′ jumps) at the start. W = λ with σ_max 2 and W = 2λ with σ_max 4 have the same slope σ_max/W and give **identical**
  values to four digits from two independent assemblies: the reflection is made at the start, and the wave
  never reaches the end (A ≥ 13).
- **m = 3 is not measurably better** than m = 2 at W = 2λ (2.8 vs 7.9·10⁻⁵ in J_x, 2.0 vs 3.5·10⁻⁵ in M_y,
  both at the floor). At W = λ it is worse (M_y 3.2·10⁻⁴, J_x 7.7·10⁻⁴ against 4.4·10⁻⁵ / 4.0·10⁻⁵ for
  m = 2), and neither mechanism accounts for it: the outer-edge estimate 0.045 e^−6.6 ≈ 6·10⁻⁵ is 5–13×
  below, and the onset of m = 3 is smoother than that of m = 2. The excess is unexplained (possibly the
  steeper end of the m = 3 profile; not investigated).

### Mesh check (local source)

| Profile | h | 2N | \|r\| J_x | \|r\| M_y | \|r\| div J | assembly / LU [s] | peak [GB] |
|---|---|---|---|---|---|---|---|
| default (W 2λ, σ 4, m 2) | λ/5 | 14 800 | 7.9·10⁻⁵ | 3.5·10⁻⁵ | 1.8·10⁻³ | 1 050 / 68 (groove) | 3.6 |
| default | λ/6 | 21 360 | 2.2·10⁻⁵ | 2.5·10⁻⁵ | 1.6·10⁻⁴ | 728 / 47 | 7.4 |
| default | λ/7 | 29 120 | 1.1·10⁻⁵ | 2.8·10⁻⁵ | 1.3·10⁻⁴ | 1 022 / 165 | 13.7 |
| W λ, σ 2, m 2 | λ/5 | 9 440 | 4.0·10⁻⁵ | 4.4·10⁻⁵ | 1.8·10⁻³ | 131 / 12 | 1.5 |
| W λ, σ 2, m 2 | λ/6 | 13 632 | 8.2·10⁻⁵ | 4.9·10⁻⁵ | 2.2·10⁻⁴ | 164 / 38 | 3.1 |
| W λ, σ 4, m 2 | λ/5 | 9 440 | 1.3·10⁻³ | 4.5·10⁻⁴ | 6.3·10⁻³ | 156 / 17 | 1.5 |
| W λ, σ 4, m 2 | λ/6 | 13 632 | 7.4·10⁻⁵ | 1.1·10⁻⁴ | 2.0·10⁻⁴ | 248 / 102 | 3.1 |

- At the defaults the reflection stays at the floor under refinement (λ/5 → λ/6 → λ/7: J_x 7.9 → 2.2 →
  1.1·10⁻⁵, M_y 3.5 → 2.5 → 2.8·10⁻⁵, all inside the refit spread).
- At W = λ with σ_max = 4 the 1.3·10⁻³ of λ/5 drops to 1·10⁻⁴ at λ/6: the onset reflection (ii) is a
  discretisation effect of a steep profile on few cells (5 cells across W = λ at λ/5, 6 at λ/6), as in a
  discretised PML; the W = λ dependence on σ_max in the table above is thus mesh-dependent, and the defaults
  (10 cells at λ/5) are not.
- div J drops by 10× from λ/5 to λ/6 (1.8·10⁻³ → 1.6–2.2·10⁻⁴): its larger values at λ/5 are the
  piecewise-constant sampling, not the PML.

### Validation-large test

`tests/validation_large/test_pml_spp_large.cpp`, two cases, label `validation-large`, RUN_SERIAL:
- "PML 7b: SPP on flat Ag reflected at the PML start below 1e-2" (`seven_b_case()`: defaults, groove,
  h = λ/5, 2N = 14 800, two dense assemblies): residual < 1e-8; for J_x and M_y of both launches and both
  half-axes fit residual < 1e-3, |Re k/Re k_spp − 1| < 0.03 and |r| < 1e-2; groove and local launch agree in
  M_y, |r_groove − r_local| < 2·10⁻⁴ on both half-axes; the acceptance check
  r_spp = max(|r(J_x)|, |r(M_y)|) of the groove launch < 1e-2. The ROW lines it prints carry the
  |out(M_y)/out(J_x)| diagnostic.
- "PML 7b check: a weak PML's reflection is measured and referred to the PML start" (σ_max 0.25, W = λ,
  local source, a = 1.5 and 1.6 µm, 2N = 9 440 and 10 416): |r(M_y)| > 5·10⁻³ for both, start-referred phase
  change < 0.2 rad (2 Re(k_spp) h = 2.65 rad is required to exceed 2), magnitude ratio within ±50 %.

Run 2026-10-15 (win-release, `ctest --preset win-release -L validation-large -R "PML 7b" --output-on-failure`,
the machine shared with other jobs as above): **both passed**, with values identical to the benchmark rows
(groove J_x 7.967·10⁻⁵, M_y 3.404·10⁻⁵; check case M_y 1.716·10⁻² / 2.285·10⁻²).

| Test | Result | Time [s] |
|---|---|---|
| PML 7b check (2N = 9 440 and 10 416; assembly 58 + 44 s) | passed (5 assertions) | 144 |
| PML 7b (2N = 14 800; assembly 557 + 416 s, LU 67 s, peak 3.6 GB) | passed (26 assertions) | 1 044 |

Rerun 2026-10-15 after the review fixes (fit residual < 1e-3, the groove/local M_y check, the
|out(M_y)/out(J_x)| diagnostic in the ROW lines; same command and machine conditions): **both passed** with
identical |r| values; |out(M_y)/out(J_x)| = 111.4 Ω (groove) and 30.5 Ω (local) at the defaults, 38.4 /
22.1 Ω in the check case (SPP: 127.0 Ω).

| Test | Result | Time [s] |
|---|---|---|
| PML 7b check (assembly 25 + 29 s) | passed (5 assertions) | 69 |
| PML 7b (assembly 414 + 416 s, LU 45 s, peak 3.6 GB) | passed (28 assertions) | 876 |

### Conclusion on the defaults

**σ_max = 4, m = 2, W_pml = 2λ confirmed** (no change proposed). |r_spp| = 8·10⁻⁵ ≪ 1e-2, at the
measurement floor, in the middle of a broad optimum (W = 2λ: σ_max 2 … 8 all at the floor) and stable under
mesh refinement. m = 1 would raise the reflection to ≈ 10× the measurement floor (1.2–1.75·10⁻³); m = 3
gains nothing measurable. W_pml = λ would also
pass 7b at h = λ/5 for every σ_max in 1 … 8 (worst 3.2·10⁻³ at σ_max 1, 2.6·10⁻³ at 8; σ_max 4:
1.3·10⁻³), and it would cut the PML ring's area roughly in half, but its reflection depends on σ_max and h
by more than 10×; whether a 10⁻³ SPP reflection matters for a rough-surface far field is for 7c (WP-PML4) to
show, not for 7b.

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

# 7b (SPP absorption). Common geometry: patch 1 um, taper 0.5 um, PML start 1.5 um, h 100 nm.
S=build/win-release/benchmarks/specklebem_pml_spp
G="--pml-start 1.5e-6 --h 1e-7"
$S $G --pml 1e-6 --sigma 4 --dump d.csv --summary rows.txt             # defaults, groove launch (+ local source)
$S $G --launch source --pml 0.5e-6 --sigma 2                           # scan point: W = lambda, sigma_max 2 (local source)
$S $G --launch source --pml 1e-6 --sigma 4 --order 3                   # order m = 3
$S --launch source --pml-start 1.6e-6 --h 1e-7 --pml 0.5e-6 --sigma 0.25  # geometric check (with 1.5e-6: the reference)
$S --launch source --pml-start 1.5e-6 --h 8.333333333333e-8 --pml 1e-6 --sigma 4   # mesh check lambda/6 (7.142857142857e-8: lambda/7)
$S --pml-start 1.5e-6 --rho-min 0.8e-6 --refit d.csv                   # refit a dump (no solve); also --bg 1.5,2 / none, --fit-imag
ctest --preset win-release -L validation-large -R "PML 7b"             # the 7b case and the geometric check as a test
```
