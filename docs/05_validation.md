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
| Speckle PDF at z = −1 mm | Eq. 10 | KS test | p > 0.05 | 6 |
| Contrast, γ₁₂ vs σ | monotone decrease | — | — | 6 |
| Speckle size | λz/D | rel | < 10 % | 6 |
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
| 7d L ×1.2 at fixed waist⁴ | the unperturbed run | far-field change (box-check metric) | < 0.5 % | met on rough Si (WP-PML4b, `benchmarks/results/pml_rough_ag.md`): 0.24 %; not met on rough Ag: 2.6 % at L ×1.2, the termination is not the cause (flat control 2·10⁻⁷, lossy diagnostic 1.1 %), the change comes from SPP currents in the gap; whether a physical L dependence remains is decided by WP-PML6 (row 7g, ADR 0010); 0.5 % not loosened⁴ |
| 7e Si cross-check: PML route vs the validated uniform box, L = 4.5 µm, w₀ = L/6, h = λ/4 on both routes³ | uniform-box far field | agreement within the box's own sensitivity (largest box change under L ×1.2)³ | l2_refl ≤ 0.229 %, ε_rr ≤ 0.058 %, \|ΔR/R\| ≤ 4.1·10⁻⁵ | met (WP-PML4c, `benchmarks/results/pml_si_crosscheck.md`): l2_refl 0.146 %, ε_rr 0.026 / 0.041 %, ΔR/R −1.6·10⁻⁵; L = 3.5 µm at λ/6: 0.186 %³ |

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
must state L, gap and window. Caveat:
the comparisons are at equal discretisation; h = λ/4 is not mesh-converged for the rough-Si far field (³), the
rough-Ag mesh convergence was not measured.

## Test organisation

- `tests/unit` – fast (< 1 s each in release; assembly tests that call `element_blocks` on an icosphere may take up to ~5 s under the sanitizer preset, documented per test), run on every push.
- `tests/validation` – minutes, labelled `validation`, run on every push for the small cases (`ctest -L '^validation$'`; `-L` is a regex); `tests/validation_large` (dense solves with 2N ≥ 15 k, the 4 µm sphere and 30 µm surfaces later) is labelled `validation-large`, memory-guarded, release-only, nightly / manual (`ctest -L validation-large`).
- `tests/python` – API smoke tests and the statistics tests on small precomputed fields.
- Reference data: `tests/data/README.md`.
