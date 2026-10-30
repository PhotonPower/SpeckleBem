# Validation strategy

Every fast or approximate component is tested against a slower exact one. The chain of oracles is:

```
analytic (Mie, Fresnel, static integrals)
   └─► dense SIE (direct LU)
          └─► dense SIE (GMRES)
                 └─► MLFMM / ACA / H-matrix (GMRES)
                        └─► GPU backend
```

## Error metrics

- **Bistatic RCS error** (Eq. 7 of the paper): `ε_rr = sqrt(Σ (E_ref − E_sie)² / M) / max|E_ref|` over `M` angles.
- **Field error** along a line: relative error per point (Fig. 1b style) and its median.
- **Operator error**: `‖Z_fast x − Z_dense x‖ / ‖Z_dense x‖` over 10 random `x`.
- **Power balance** for lossy objects: `P_ext = P_sca + P_abs` within tolerance (independent of any reference solution).
- **Reciprocity**: scattering amplitude `S(k̂_s, k̂_i) = S(−k̂_i, −k̂_s)` for swapped directions.

## Acceptance table

| Test | Reference | Metric | Tolerance | Phase |
|------|-----------|--------|-----------|-------|
| Static integrals on separated triangles | Dunavant-20 | abs/rel | 1e-12 | 1 |
| Static integrals on coincident triangles | nested-quadrature extrapolation | rel | 1e-9 | 1 |
| Height-map statistics | prescribed σ, Lc | rel (20 seeds) | 2 % / 5 % | 1 |
| Dielectric sphere d = 1 µm, n = 1.5, λ/10 | Mie | ε_rr | < 1 % | 2 |
| Ag sphere d = 1 µm, λ/20, PMCHWT + LU | Mie | ε_rr | < 1 % | 2 |
| Mesh refinement λ/8 → λ/16 → λ/32 | Mie | ε_rr monotone decrease | — | 2 |
| Power balance, Ag and Si spheres d = 1 µm, PMCHWT + LU (P_abs = ½ Re ∮ (n̂ × M)·J* dS from the surface currents) | — | |P_ext − P_sca − P_abs| / P_ext | < 1 %: Si at λ/6.6 (icosphere n = 3), Ag at λ/13 (n = 4, `validation-large`); at λ/6.6 the Ag defect is 2.3 % because P_abs ≈ 1.6 % of P_ext | 2 |
| Flat box, tapered beam, 0° and 45° | Fresnel r_p, r_s | rel on |r|² | < 1 % (met, WP22c: `benchmarks/results/fresnel_flat.md`; Ag needs GMRES tol 1e-5) | 2 → 4 |
| Total scattered power, Si sphere d = 1 µm, PMCHWT + LU, icosphere n = 3 (`Simulation::scattered_power`, full-sphere grid 64 × 128 / (\|E₀\|²/2η₁); tests/validation/test_drc_power.cpp) | Mie C_sca | rel | < 2 % (the bound of the Si power-balance case; met, WP25: −1.17 %) | 5 |
| Absolute DRC per steradian, flat Si, PML route, 0°, w₀ = 0.5 µm, h = 125 nm, dense (`Simulation::drc` on the 1° reflection hemisphere; tests/validation_large/test_drc_fresnel_large.cpp, `validation-large`) | Fresnel beam reflectance R_beam | rel on ∫_{k_z<0} D dΩ | < 1 % (met, WP25: −0.022 %; evaluation cost `benchmarks/results/drc_evaluation.md`) | 5 |
| GMRES vs LU, 2·10⁴ unknowns | dense LU | true relative residual ≤ tol (solution difference vs LU documented, κ-limited) | < tol | 3 |
| MLFMM matvec, Si & Ag, 2·10⁴–10⁵ unknowns | dense matvec | rel | < 1e-3 (3 digits), < 1e-5 (5 digits) | 4 |
| Ag sphere d = 4 µm, λ/27¹, MLFMM | Mie | ε_rr normal/parallel planes | ≤ 0.5 % | 4 |
| Power balance, Ag and Si surfaces | — | rel | < 1 % (Ag at mesh ≤ λ/13: the surface P_abs of weakly absorbing Ag converges slowly, see benchmarks/results/mie_sphere_dense.md) | 4 |
| Reciprocity, rough surface | — | rel | < 1 % | 4 |
| Near field Ag sphere, xz-plane² (tests/validation_large/test_ag_near_field_large.cpp, CI companion tests/validation/test_ag_near_field.cpp) | Mie | rel per point \|E_sim − E_Mie\| / \|E_Mie\| | < 0.1 (lit), < 1 (shadow) (met, WP24: d = 4 µm paper mesh, GMRES tol 1e-4 (acceptance): max 0.012 lit / 0.006 shadow; at the paper's tol 1e-3 0.088 / 0.022 (lit margin 1.14×); d = 1 µm λ/26.5, tol 1e-5: 0.005 / 0.005; `benchmarks/results/ag_near_field.md`) | 5 |
| Rigorous beam Maxwell check (`AngularSpectrumBeam`; unit test "angular-spectrum beam: finite-difference Maxwell residual at round-off level", tests/unit/test_angular_spectrum_beam.cpp, asserts < 1e-9, measured ≤ 9e-11) | finite differences (O(h⁴), h = λ/400) | ‖∇×E + jωμH‖ / ‖ωμH‖ | < 1e-8 | 5 |
| Speckle PDF at z = −1 mm⁵ (co-polarised intensity; `post::ks_test_exponential` / `sb.ks_test_exponential` on I/⟨I⟩ **per σ, pooled over the seeds**, decimated to independent samples 2 speckle sizes apart, n ≥ 50 per σ; only fully developed σ, coherent fraction exp(−(2kσ)²) < 1e-2 (σ = 100 nm counts at λ = 500 nm); `specklebem.speckle.ensemble_statistics`, key `ks_rejects_5pct`; σ-pooled numbers are supplementary and never a pass) | Eq. 10 (negative exponential e^{−x}) | one-sample KS with the mean estimated from the sample: level-correct exponential test with the Stephens-modified statistic (D − 0.2/n)(√n + 0.26 + 0.5/√n) against its 5 % critical value 1.094 (`KsResult::rejects(0.05)`; Stephens 1974, D'Agostino & Stephens 1986 ch. 4) | level-correct exponential KS (Stephens-modified statistic) does not reject at 5 % on the decimated I/⟨I⟩ of each fully developed σ (≥ 50 independent samples per σ); the plain asymptotic Kolmogorov p is reported alongside (conservative: ≈ 0.5 % actual level at nominal 5 %). Harness verified on synthetic speckle, WP-S1: `tests/unit/test_speckle_statistics.cpp` (level and power over repetitions), `tests/python/test_speckle.py`. Surfaces: **not met with this configuration — physical, open until a server run** (WP-S2 / WP-S2b, lead decision 2026-10-30; `benchmarks/results/speckle_sweep.md`, `tests/python/test_speckle_dod.py`): with two seeds per σ n = 8 / 18 / 32 / 32 < 50 at σ = 100 / 150 / 200 / 250 nm (gate not evaluable), and the patterns are measurably not negative-exponential: the 1-speckle pool over σ = 100–250 nm (n = 278) rejects (modified 1.73, plain p = 0.006), envelope-normalised contrast C_env = 1.16 / 1.29 / 1.16 at σ = 150 / 200 / 250 nm (ideal speckle through the same analysis 0.95–1.01); supplementary σ-pooled gate (n = 90, the same two maps at every σ, not independent): modified 1.086 < 1.094, p = 0.20, its decision flips under 12 of 20 neighbouring analysis choices (not a pass) | 6 |
| Contrast, γ₁₂ vs σ⁵ (`speckle_contrast`, `correlation_coefficient` = Pearson coefficient of the two intensity patterns; `specklebem.speckle.monotone_decreasing` on the seed means ordered by σ) | monotone decrease | — | — (contrast: **met**, WP-S2 / WP-S2b, `tests/python/test_speckle_dod.py`: co-polarised plane contrast, seed means 10.72 → 5.65 → 3.67 → 2.66 → 2.15 for σ = 50 … 250 nm, strictly decreasing, each seed on its own too; γ₁₂: **not testable at normal incidence, deferred (server)**: the 0°/30° incidence pair runs with the server recipe of `benchmarks/results/speckle_sweep.md`; the seed-to-seed correlation of the plane patterns, 0.976 → 0.854 → 0.625 → 0.465 → 0.329, is monotone but supplementary only, lead decision 2026-10-30) | 6 |
| Speckle size⁵: FWHM of \|μ_A\|² from `field_correlation_from_intensity` (Siegert relation) + `speckle_size`, vs `speckle_size_expected` = c·λz/D with c = 0.8859 (square, side D), 1.0290 (disc, diameter D), 1.0600 (Gaussian spot, D = 2w₀, the beam case); Fraunhofer zone, paraxial observation | λz/D | rel | < 10 % (synthetic propagated speckle, WP-S1: ≤ 2.7 % for the three apertures at two (λ, z, D)); surfaces⁵: **met for 3 of 4 fully developed σ; σ = 200 nm +11.4 %** (WP-S2 / WP-S2b, `tests/python/test_speckle_dod.py`, reference 198.8 µm): +2.8 / +4.4 / +0.4 % at σ = 100 / 150 / 250 nm; σ = 200 nm +11.4 % (the method's two-seed scatter on ideal speckle is ±8–15 % per σ; a strict xfail in the test, tolerance not loosened); one (λ, z, D): the magnitude is tested, the λz/D scaling is not | 6 |
| GPU vs CPU, all kernels | CPU | rel | < 1e-4 (double), documented for single | 7 |
| ACA / H-matrix matvec | dense | rel | < ε set | 8 |
| Imported Gmsh closed mesh | power balance | rel | < 1 % | 8 |

¹ "λ/27" is the paper's label for its 393 216-unknown mesh: an octahedron-based sphere with 8·4⁷ =
131 072 triangles, mean edge 30.25 nm = λ/16.5, √(mean triangle area) = λ/25.5 (WP22a,
`benchmarks/results/ag_sphere_4um_mlfmm.md`). The row means that mesh, not a mean edge of λ/27.

² Zones of the near-field row (a the sphere radius, h the mean edge length): shadow = the
geometric shadow |x| < a, z > 0; lit = every other exterior point, including z ≤ 0, the x-axis and
the forward region beside the sphere; both outside the near-surface band |r| − a < h/2. The
denominator of the per-point error is the local |E_Mie| (`benchmarks/results/ag_near_field.md`).

⁵ Speckle rows 42–44 (WP-S2 / WP-S2b, `benchmarks/results/speckle_sweep.md`; lead decision
2026-10-30). Configuration: λ = 500 nm, vacuum over Si (ε_r = 18.478 − 0.606j), Gaussian heights
with Lc = 1 µm (1/e radius of the correlation), σ = 50, 100, 150, 200, 250 nm, seeds 1 and 2 (the
same two maps scaled by σ), patch L = 8 µm, h = 50 nm, uniform closing box (2N = 741 120),
rigorous beam at normal incidence, E along x, w₀ = L/6 = 1.333 µm, ICTF + MLFMM d₀ = 3, GMRES
10⁻⁴; scattered field on the plane z = −1 mm, 4 × 4 mm², 161² points; co-polarised |E_x|²,
resampled to direction cosines, leave-one-out envelope normalisation, halo window
(`specklebem.speckle_sweep.analyse_ensemble`). Physical explanation of row 42: far-zone speckles
are λ/D wide in direction cosines (D = 2w₀), so at most 2D/λ = 10.7 fit per axis and 3.9–6.7 lie
inside the halo windows (4–16 independent samples per pattern); and the beam spot covers only
(w₀/Lc)² = 1.8 surface correlation areas (spot area π w₀²; 0.9 with the effective area π w₀²/2),
so the far field is a sum of few glints: over-dispersed speckle (C_env > 1) although the coherent
part is negligible; a Kirchhoff phase-screen surrogate of the same configuration shows the same
(C_env 1.18–1.32, σ-pooled gate rejected in 64 % of its realisations). The fully-developed rule
exp(−(2kσ)²) < 1e-2 is kept. Row 42 and the γ₁₂ part of row 43 stay open until the runs of the
record's "Server recipe (to be run by hand)" (Lc = 0.5 µm with four seeds at σ ≥ 150 nm, i.e.
(w₀/Lc)² = 7 and an expected n = 64 per σ; the 0°/30° pair; optionally L = 25 µm); the DoD test
then applies the gates of rows 42–44 to those directories (`SBEM_SPECKLE_SWEEP_DIR`).

Preconditioners change iteration counts, not the solution, so they have no row of their own: a new
preconditioner must reproduce the existing solves to the solver tolerance. For the near-field
block-diagonal preconditioner (WP-PC1, ADR 0011; the default of the "mlfmm" PML route since then,
the only measured case) this is gated by
`tests/validation/test_block_preconditioner_mie.cpp` (Ag sphere d = 1 µm, MLFMM: bistatic RCS within
10⁻⁵ of the Jacobi solve at GMRES tolerance 10⁻⁶, ε_rr unchanged to 10⁻⁵) and by the PML-route rows,
which now run with it (7h fast path vs dense); the rough Ag / Si PML sizes up to 2N = 3·10⁵ reproduce
the Jacobi reflectance to 5·10⁻⁸ / 4·10⁻⁷ (`benchmarks/results/preconditioner.md`).

## Rough-surface specific checks

- **Box-depth sensitivity**: scattered far field changes by < 0.1 % when the closing box depth is doubled (Si and Ag). Documents that the artificial boundary is inert.
- **Edge-effect sensitivity**: far field changes by < 0.5 % when L is increased by 20 % at fixed beam waist.
- Metric of both sensitivity checks (ADR 0006 amendment): L2 difference of |F|² over the reflection
  hemisphere, sqrt(Σ w (I_T − I_R)² / Σ w I_R²) with w = sin θ; ε_rr on the xz/yz cuts is reported
  alongside. Beam waist w₀ ≤ L/4, rigorous (angular-spectrum) beam, box cells ≤ λ₁/5, fine band for Si.
  Meaningful only at sizes where the rim is dark (WP-V2, MLFMM).
- **Flat-surface limit**: σ → 0 recovers the specular Fresnel reflection of the beam.
- **Regression**: height map for `seed = 42`, L = 10 µm is stored in `tests/data`; a change in the generator that alters it must be deliberate.

## PML route (open surface, ADR 0009 decision 7)

For the open rough surface terminated by the complex-coordinate-stretching PML these checks replace the box-depth
check above; the physical model is a rough spot in a polished surface with absorbing surroundings (ADR 0009, lead
decision 1). Tolerances are not loosened relative to the box route.

| Check | Reference | Metric | Tolerance | Status |
|-------|-----------|--------|-----------|--------|
| 7a Flat Ag and flat Si with PML, rigorous beam, 0° and 45°, s and p | beam reflectance R_beam of the same beam on the infinite interface (Fresnel per plane wave) | rel on \|r\|² (R) | < 1 % | met (WP-PML3b, `benchmarks/results/pml_validation.md`): Ag 0°/45° and Si 0° at h = λ/4 with the 1e-4 dark band (≤ 0.28 %; 45°: w₀ = 0.75 µm, pml_start 4.5 µm); Si 45° accepted at the mesh-converged h = λ/6 (+0.18 % / +0.03 %, smaller geometry w₀ = λ, L = 3 µm with the dark band relaxed to 1e-2, as the 7a geometry at λ/6 is infeasible densely; the dark band is a decision-4 configuration check, and at λ/6 the 5.9e-3 and 2.6e-2 geometries agree within 0.05 % / 0.08 %); Si 45° at λ/4 (−0.76 % / −0.41 %) is not mesh-converged (λ/5: +5.9 %) and only reported |
| 7b SPP launched on flat Ag, reflected at the PML start | — | \|r_spp\| | < 1e-2 | met (WP-PML4a, `benchmarks/results/pml_validation.md`): \|r_spp\| = 8.0e-5 (J_x) / 3.4e-5 (M_y) at the PML defaults (σ_max 4, m 2, W_pml = 2λ), λ = 500 nm, Ag, groove launch (d 50 nm, r_g 100 nm, grooved minus flat solution) under the rigorous beam at 0°, h = λ/5, PML start 1.5 µm, fit of outgoing + reflected cylindrical SPP waves plus outgoing background terms along the x-axis (measurement floor ≈ 1–2e-4; λ/6 and λ/7 at the floor too); defaults confirmed (W_pml = 2λ: σ_max 2–8 at the floor; m = 1 ≈ 1e-3; W_pml = λ with σ_max 1–8 ≤ 3.2e-3, σ- and h-dependent, σ_max 0.25 / 0.5 not: 0.15 / 0.052) |
| 7c PML parameters W_pml ×1.5 and σ_max ×1.5 (rough Ag)⁴ | the unperturbed run | far-field change (box-check metric) | < 0.1 % | met (WP-PML4b, `benchmarks/results/pml_rough_ag.md`): W_pml ×1.5 2.5·10⁻⁵, σ_max ×1.5 4.6·10⁻⁵ on rough Ag; quadrature (target accuracy 1e-6, 12 stretched touching points) 3·10⁻⁸; gap 0.5 → 0.875 / 1.75 µm at the same window ≤ 4.4·10⁻⁵ (configuration: footnote ⁴) |
| 7d L ×1.2 at fixed waist⁴ | the unperturbed run | far-field change (box-check metric) | < 0.5 % | met on rough Si (WP-PML4b, `benchmarks/results/pml_rough_ag.md`): 0.24 %; not met on rough Ag: 2.6 % at L ×1.2, the termination is not the cause (flat control 2·10⁻⁷, lossy diagnostic 1.1 %), the change comes from SPP currents in the gap; whether a physical L dependence remains is decided 2026-10-25 (physical, ADR 0010; confirmed by WP-PML6b); 0.5 % not loosened⁴ |
| 7e Si cross-check: PML route vs the validated uniform box, L = 4.5 µm, w₀ = L/6, h = λ/4 on both routes³ | uniform-box far field | agreement within the box's own sensitivity (largest box change under L ×1.2)³ | l2_refl ≤ 0.229 %, ε_rr ≤ 0.058 %, \|ΔR/R\| ≤ 4.1·10⁻⁵ | met (WP-PML4c, `benchmarks/results/pml_si_crosscheck.md`): l2_refl 0.146 %, ε_rr 0.026 / 0.041 %, ΔR/R −1.6·10⁻⁵; L = 3.5 µm at λ/6: 0.186 %³ |
| 7f Huygens-plane far field vs windowed currents, flat Ag and Si (7a geometry; CI size with a thinner PML) | windowed-current far field | L2 metric; R | ≤ 1e-3; R ≤ 1e-4 | met (WP-PML6, `benchmarks/results/pml_huygens.md`) at two sizes, the same gates: CI size (`validation`, `tests/validation/test_pml_huygens.cpp`): the 7a zones with W_pml = λ, σ_max 2, 2N = 7 632, dense LU: Ag / Si L2 2.73·10⁻⁴ / 2.84·10⁻⁴, ΔR/R −1.7·10⁻⁷ / −4.1·10⁻⁷, z_H ×1.5 vs currents 5.3 / 5.4·10⁻⁴, spacing /1.5 vs currents 2.7 / 2.8·10⁻⁴; CI cost two dense solves per material, 4–8 min per material on a loaded 12-core desktop, expect 10–20 min per material on 4-vCPU runners (the 7h CI geometry, window from 1.5 w₀, does not qualify: L2 8·10⁻³, the reflected beam crosses the plane window); 7a geometry (`validation-large`, `tests/validation_large/test_pml_huygens_large.cpp`): dense LU, 2N = 11 440, plane defaults (z_H = λ, spacing λ/4, the current window): Ag / Si L2 2.73·10⁻⁴ / 2.83·10⁻⁴, ΔR/R −1.7·10⁻⁷ / −4.2·10⁻⁷; z_H ×1.5 vs currents 5.3·10⁻⁴ / 5.4·10⁻⁴ (vs the default plane 2.6 / 2.7·10⁻⁴), spacing /1.5 vs the default plane 8.5·10⁻⁷ / 7.4·10⁻⁷, plane quadrature degree 10 vs 6 5·10⁻¹² |
| 7g Huygens-plane far field on rough Ag: 7d (L ×1.2), window ×(1 ± 0.3), z_H ×1.5; rough Si 7e repeated | the unperturbed run / uniform box | L2 metric | 7d < 0.5 %, sensitivities < 0.1 %; 7e within the box's sensitivity | not met (WP-PML6, `benchmarks/results/pml_huygens.md`)⁴: rough Ag U, fast path: 7d with the Huygens plane 2.78 % (own default windows) / 1.83 % (the same absolute window; currents 2.57 %), z_H ×1.5 1.05 %, plane window ×0.7 / ×1.3 1.75 % / 0.66 % (wide-gap run); rough Si 7e: l2_refl 0.220 % and ε_rr 0.038 / 0.044 % met, ΔR/R −5.3·10⁻⁴ not (≤ 4.1·10⁻⁵); both cases run as `validation-large` ctests (2026-10-25: the ctests pass as regression guards, every value reproduced; the row stays not met, rough Ag 58 min, rough Si 61 min). Cause: the truncation of the plane by its window (raised cosine across the 0.5 µm gap, transition Δ = λ, the plane ends at pml_start): grazing directions are missed (θ 75–90°: −67 % of the band power vs the currents) and the window end leaves ≈ π²/(2(kΔ)²) per end of every field crossing it; the sensitivities are as large as the 7d change, so ADR 0010's physical-conclusion clause is not applicable to this result. Default far-field source stays the windowed currents; tolerances not loosened; closed by the lead decision of 2026-10-25 as "not met — physical" (ADR 0010: the L dependence of rough metals is the result; rule gap ≥ 6 λ, quantitative in the θ ≤ 30° cone, confirmed by WP-PML6b on two seeds, footnote ⁴). Diagnostic, not validated: on a 3 µm real gap with Δ = 2–6 λ the plane's z_H ×1.5 falls to 0.49–0.10 % (θ ≤ 30°: 0.21–0.018 %) while 7d stays 2.4–2.5 % (θ ≤ 30°: 2.2–2.4 %) for plane and currents alike |
| 7h Fast path vs dense (compression "mlfmm" on the PML route: MLFMM on the real functions + PML ring, near exact, far ACA since WP-PML5b, GMRES), flat Ag (7a type) and rough Ag (7c geometry); scaling | dense PML matrix and its LU solve | matvec rel; R rel; windowed far field L2 (box-check metric); memory / time at 2N ≈ 2.3·10⁴, 6·10⁴ and the largest size in 50 GB | matvec < 1e-3 (d₀ = 3) / < 1e-5 (d₀ = 5); R ≤ 1e-4; L2 < 1e-3 | met at 2N = 2.3·10⁴ (WP-PML5a, `benchmarks/results/pml_ring.md`): rough Ag U (2N = 22 816, d₀ = 3) matvec 1.5·10⁻⁵, dR 4·10⁻⁸, L2 3.3·10⁻⁶ at 0.55× the dense memory (4.6 vs 8.3 GB) and 1.4× its assembly time; flat Ag 2N = 4 592 (d₀ = 3 / 5) matvec 3.2·10⁻⁴ / 4.9·10⁻⁸, dR ≤ 7·10⁻⁸, L2 ≤ 2·10⁻⁶; 2N = 2 816 matvec 3.9·10⁻⁴ / 1.2·10⁻⁸; U at d₀ = 5 (review round): matvec 1.4·10⁻⁷, dR 5.5·10⁻⁸, L2 1.2·10⁻⁷ at 0.75× the dense memory (6.25 GB); kept through the ACA far ring (WP-PML5b, `pml_ring.md` section 5b): U d₀ = 3 matvec 2.1·10⁻⁵, dR 4.8·10⁻⁸, L2 3.3·10⁻⁶ at 1.5 GB (5a 4.6 GB); d₀ = 5 matvec 2.1·10⁻⁷; flat CI 3.2·10⁻⁴ / 4.0·10⁻⁸; scaling: 2N = 59 600 (L = 8.5 µm) ACA ring vs the exact 5a ring 5.1·10⁻⁵ (1.1·10⁻⁵ of the operator), far ring 1.64 GB instead of 16.7 GB, operator 3.7 GB (dense 57 GB); 2N = 152 960 (L = 16 µm) in 13.3 GB (peak 18 GB), d₀ 3 vs 5 R 1.5·10⁻⁸, L2 1.5·10⁻⁶; largest run 2N = 300 160 (L = 24 µm) in 20.1 GB, peak 33.6 GB (50 GB at 2N ≈ 4·10⁵ projected); rough-Si 7d control through the fast path 0.2412 % (dense 0.2411 %), 7e l2_refl 0.1464 % (dense 0.1464 % = the stored WP-PML4c value; the dense rerun of WP-PML5b was stopped, not completed) |

³ Row 7e (WP-PML4c). Configuration: the validated box configuration of `box_validity_mlfmm.md` (ADR 0009 named
L = 4 µm, w₀ = L/6), seed 1, σ = 50 nm, Lc = 500 nm, normal incidence, x polarisation, λ = 500 nm; one height map
generated at λ/60 and subsampled, so that the patch heights are bitwise identical on both routes (the patch
triangle vertex triples agree too). Tolerance (fixed
before the PML solve): per quantity the largest change of the uniform box alone under its L ×1.2 check (WP-V2/V2b,
seeds 1–2, w₀ = L/5…L/6, and this study's λ/4 partners L ×1.22 / ×1.17: 0.155 / 0.161 %); the ε_rr pass (yz
0.0413 %) rests on that cross-seed maximum (WP-V2b seed 2), as this surface's own partners reach 0.0410 %. The
uniform box at h = λ/4 = 125 nm exceeds its own λ₁/5 = 100 nm cell bound of ADR 0006 (the validated box uses
50 nm), because 7e uses one patch mesh for both routes; the λ/6 and λ/10 box runs satisfy the bound.
Caveat: h = λ/4 is not mesh-converged for the rough-Si far field (λ/6 vs λ/4: 2.0–2.3 %, the same on both routes;
box λ/10 vs λ/6: 1.23 %, second-order convergence, ≈ 0.7 % left at λ/10), so 7e compares the terminations at
equal discretisation. R moves by 0.75–1.5·10⁻³ from λ/4 to λ/6 (both routes alike); λ/4 and λ/6 are within
4·10⁻⁴ of λ/10 at L = 4.5 µm. Mesh check of the cross-check at L = 3.5 µm, w₀ = L/6 (the λ/6 PML route does not
fit 50 GB at 4.5 µm): PML vs box 0.210 % (λ/4) and 0.186 % (λ/6), against the box's own L partners at the same
mesh and L of 0.21–0.26 % and 0.19–0.27 %.

⁴ Rows 7c / 7d (WP-PML4b). Configuration: vacuum over Ag (ε_r = −9.794 − 0.313j) or Si (18.478 − 0.606j),
λ = 500 nm, seed 1, σ = 50 nm, Lc = 500 nm, h = λ/4, patch L = 3.75 µm (L ×1.2: 4.5 µm, the same height map
on the common area), taper 0.5 µm, gap 0.5 µm, PML defaults (W_pml = 2λ, σ_max 4, m 2), rigorous beam at 0°,
w₀ = λ = L/7.5 (dark band 3.3·10⁻⁶), default far-field window (taper end to pml_start), dense ICTF + LU,
2N = 22 816 / 27 472. 7c holds at the relaxed dark band 8.3·10⁻³ too (w₀ = 1.3 µm: σ_max ×1.5 3.7·10⁻⁵).
7d on rough Si: l2 0.241 %, l2 θ ≥ 30° 0.50 %, ε_rr 0.095 / 0.067 %, ΔR/R −1.1·10⁻⁵. 7d on rough Ag: l2
2.57 % (θ ≥ 30°: 6.8 %, ΔR/R +1.8·10⁻⁴). The termination is not the cause (7c, flat control, Si control).
The change comes from SPP currents in the gap (amplitude range 41.5 µm ≫ L; 10 % of the patch current level
in the gap): the window truncates them (1.61 % on the same currents) and the added annulus changes them
(2.42 % at the same absolute window, which still includes the window acting on the changed currents; the two
are not separated). ADR 0010 / WP-PML6 (row 7g) decides whether a physical L dependence remains. Controls:
flat Ag with the same zones, beam and window move 2·10⁻⁷; Im ε_r −6 (shorter SPP range, not Ag) 1.09 %.
Across Ag, lossy Ag and Si the change follows the gap current level (l2 / level 0.26–0.32). The
validation-large test asserts 7d on Si and reports the Ag value (WARN, plus a regression guard on the
documented value; lead decision 2026-10-18, ADR 0010); the 0.5 % is not loosened. Every rough-Ag PML result
must state L, gap and window. Row 7g (WP-PML6, `benchmarks/results/pml_huygens.md`) repeats 7d with the
Huygens-plane far field on the same cases through the fast path (MLFMM d₀ = 3 + PML ring, GMRES 10⁻⁷;
currents 7d 2.57 % reproduced) and fails for the plane window, not for the PML: with the default window
(Δ = gap = λ) the plane's own z_H ×1.5 and window ×(1 ± 0.3) sensitivities are 0.7–1.8 %. Diagnostic (not
validated): with the gap widened to 3 µm and the plane window transition Δ = 2–6 λ, the Huygens plane and
the windowed currents agree at θ ≤ 30° to 0.16–0.018 % and z_H ×1.5 moves it by 0.21–0.018 % there, while
L ×1.2 still changes the far field by 2.2–2.4 % (θ ≤ 30°) / 2.4–2.5 % (hemisphere); the hemisphere is not
converged (z_H ×1.5 0.10 % only at Δ = 6 λ, grazing band θ 75–90° −44 % vs the currents). Whether this
makes the rough-Ag L dependence the physical result (ADR 0010 decision 3, last item) was decided 2026-10-25
(yes; ADR 0010 "Decision 2026-10-25"). Rule (lead decision 2026-10-25, confirmed by WP-PML6b,
`benchmarks/results/pml_huygens.md` section "WP-PML6b"): rough-metal far fields are quantitative only with a gap
≥ 6 λ and in the cone θ ≤ 30°, where on two seeds (rough Ag, σ = 50 nm, Lc = 500 nm, L = 3.75 µm, h = λ/4) the
flank ×0.7 / ×1.3 of the window (taper_end, taper_end + 4 λ) at a fixed inner edge moves the windowed currents by
≤ 8.7·10⁻⁴ (default window ×0.7 only: ≤ 3.9·10⁻⁴) and the Huygens plane by ≤ 3.2·10⁻⁴ while L ×1.2 changes the
far field by 2.2–2.3 % / 1.6 % (18–76×: physical, row 7g closed as "not met — physical"); results outside the
cone or with narrower gaps are qualitative. The currents' margin is 8.7·10⁻⁴ on seed 2, which has 1.6× the gap
currents of seed 1 — stronger-SPP surfaces may need a wider gap. Caveat:
the comparisons are at equal discretisation; h = λ/4 is not mesh-converged for the rough-Si far field (³), the
rough-Ag mesh convergence was not measured.

## Test organisation

- `tests/unit` – fast (< 1 s each in release; assembly tests that call `element_blocks` on an icosphere may take up to ~5 s under the sanitizer preset, documented per test), run on every push.
- `tests/validation` – minutes, labelled `validation`, run on every push for the small cases (`ctest -L '^validation$'`; `-L` is a regex); `tests/validation_large` (dense solves with 2N ≥ 15 k, the 4 µm sphere and 30 µm surfaces later) is labelled `validation-large`, memory-guarded, release-only, nightly / manual (`ctest -L validation-large`).
- `tests/python` – API smoke tests and the statistics tests on small precomputed fields.
- Reference data: `tests/data/README.md`.
