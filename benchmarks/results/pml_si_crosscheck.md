# PML route vs the uniform box for rough Si (ADR 0009 decision 7e; WP-PML4c)

Acceptance criterion: ADR 0009 decision 7e / `docs/05_validation.md` "PML route" row 7e — the open-surface PML
route and the validated uniform closing box of ADR 0006 give the same far field for the same rough Si surface,
**within the box's own sensitivity**, with the ADR 0006 box-validity metric.

Executable `benchmarks/pml_si_crosscheck.cpp` (`specklebem_pml_si_crosscheck`, `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`,
one route per process), shared code `tests/support/pml_si_crosscheck_support.hpp`, unit test
`tests/unit/test_pml_si_crosscheck.cpp`, acceptance test `tests/validation_large/test_pml_si_crosscheck_large.cpp`.
Raw lines: block "WP-PML4c" at the end of `pml_validation_rows.txt`; full run logs in `pml_si_crosscheck_logs/`.

**Verdict: 7e met** at the 7e configuration (L = 4.5 µm, w₀ = 0.75 µm = L/6, h = λ/4 on both routes):

| Metric | PML route vs box | Tolerance (box's own sensitivity) | Ratio |
|---|---:|---:|---:|
| l2_refl (L2 of \|F\|² over the reflection hemisphere, w = sin θ; verdict metric) | **0.146 %** | 0.229 % | 0.64 |
| ε_rr, xz / yz cut | **0.026 / 0.041 %** | 0.058 % | 0.71 |
| R (WP25 reflectance) relative | **−1.6·10⁻⁵** | 4.1·10⁻⁵ | 0.39 |

- The two terminations differ by about as much as the box differs from itself when its rim moves. At the same
  mesh the box's L ×1.22 / ×1.17 change is 0.155 / 0.161 %, and the PML–box difference is 0.146 %. The residual
  is the edge effect of the truncation, not a defect of either route. It sits off-specular (θ ≥ 30°: 0.40 % for
  PML–box, 0.47–0.50 % for the box L partners).
- **h = λ/4 is not mesh-converged for the rough-Si far field.** The box route moves by **2.3 %** (l2) from λ/4
  to λ/6, about 15× the inter-route difference, and λ/6 is still 1.2 % from λ/10. The box's L sensitivity is
  unchanged by the mesh (λ/6: 0.136 %). The discretisation error is common to both routes, since they share the
  patch mesh. 7e is therefore a statement about the terminations at equal discretisation. It is not an absolute
  accuracy of either route at λ/4. λ/6 → λ/10 moves the box far field by 1.23 %, consistent with second-order
  convergence and a remaining mesh error of roughly 0.7 % at λ/10.
- **R moves by 0.75–1.5·10⁻³ from λ/4 to λ/6** (+7.5·10⁻⁴ at L = 4.5 µm, +1.47·10⁻³ at L = 3.5 µm, on both
  routes alike). λ/4 and λ/6 are within 4·10⁻⁴ of λ/10 at L = 4.5 µm (λ/4 −3.9·10⁻⁴, λ/6 +3.7·10⁻⁴ relative
  to λ/10; box only, R not monotone in h).
- **PML route at λ/6.** It does not fit the 50 GB budget at L = 4.5 µm (2N = 62 016, 61.5 GB dense). The λ/6
  cross-check was run at L = 3.5 µm, w₀ = L/6 (section "Mesh check of the cross-check").
  - PML–box difference: 0.210 % at λ/4 and **0.186 % at λ/6**. The box's own sensitivity there is 0.21–0.26 %
    (λ/4) and 0.19–0.27 % (λ/6), so the agreement holds at both meshes.
  - Each route moves by the same amount from λ/4 to λ/6: 2.02 % (PML) and 2.03 % (box). This confirms that the
    mesh error is common to both routes.

## Setup

| Parameter | Value |
|---|---|
| Surface | `geometry::generate_gaussian_height_map`, seed 1, σ = 50 nm, Lc = 500 nm, generated **once** on 6.5 × 6.5 µm at λ/60 = 8.33 nm and subsampled to the mesh spacing h (every 15th node for λ/4 = 125 nm, 10th for λ/6, 6th for λ/10), so that every mesh and both routes sample one surface |
| PML route | `geometry::make_open_surface_mesh` of the subsampled map: patch L × L (full heights), raised-cosine taper W_t = 0.5 µm = Lc (heights → 0), flat gap W_g = 0.5 µm, flat PML W_pml = 1 µm = 2λ (σ_max 4, m 2), open outer edge; `Simulation` PML constructor, dense, ICTF, dark-band tolerance 1e-4; LAPACK LU (residual ≤ 3·10⁻¹³; the runs used the test-support `pml_fresnel::DenseLu`, which WP-LU1 replaced by the library's `solver::DenseLu` with the same zgetrf factorisation, now used by the code) or full GMRES (λ/6, tol 1e-8); far field of the real part with the default window (taper_end → pml_start, across the gap) |
| Box route | ADR 0006 uniform box (amendment 2026-10-12b: `rough_surface_box_params(Si, vacuum, λ, σ, h, Uniform)`, box cells = top-face cells, default depth 5.65 µm, Si fine band 3.54 µm, inert in the uniform box) on the central L × L crop of the **same subsampled map**; `check_beam_waist` with the Si fraction 1/6; MLFMM d₀ = 3, global leaf rule, ICTF, no Jacobi, full GMRES tol 1e-5 (the WP-V2 settings; Si noise floor 3.7·10⁻⁵) |
| Beam | one `AngularSpectrumBeam` for every run: λ = 500 nm, vacuum, normal incidence from z < 0, E along x (P at 0°), focus at the origin, w₀ as stated, controlled ball 7 µm (box vertices up to 6.9 µm at L = 5.5 µm; the PML route needs 1.02 √2 pml_start ≤ 5.4 µm). 5 760 plane waves at w₀ = 0.75 µm, grid check 9·10⁻¹² |
| Media | Si ε_r = 18.478 − 0.606j, vacuum R1, exp(+jωt) |
| Far field | `post::far_field`, Dunavant degree 10 (check: degree 8, ≤ 5·10⁻⁸ l2), the box-validity grids (`box_validity_support.hpp`): reflection hemisphere θ = 0.5…89.5° (1°) × φ = 0…357° (3°), xz/yz cuts (0.5°). R = `post::radiated_power` on the 90 × 180 Gauss–Legendre hemisphere grid / `Simulation::incident_power()`, i.e. `Simulation::reflected_power(90, 180, 10) / incident_power()` (WP25) |
| Metric | `pml_si_crosscheck::compare`: l2_refl, l2_off30, ε_rr on the cuts (`box_validity` functions, ADR 0006 amendment decision 5), dR = R_test / R_ref − 1 |
| Machine | Windows 11, 12 cores / 24 threads, 128 GB, win-release (GCC 16.1, OpenBLAS), `OMP_NUM_THREADS=12`, 2026-10-15; one solve of this study at a time, but **another worker's dense rough-Ag PML study and debug test suites ran alongside the whole time** (times inflated, about 2–3× for the dense PML assembly against WP-PML3b) |

**Configuration choice (brief: L ≈ 4 µm with w₀ = L/6, or the validated configuration).**
- The study uses the validated box configuration of `box_validity_mlfmm.md` on both routes: L = 4.5 µm,
  w₀ = 0.75 µm = L/6, seed 1, σ = 50 nm, Lc = 500 nm, normal incidence, x polarisation.
- It also runs the box's L-partner check at this study's own mesh, because the mesh differs (λ/4 = 125 nm here,
  50 nm in the record).
- The surface realisation differs from the record's: the same seed on a different generation grid gives a
  different map. So the record values are statistically, not literally, the same configuration.
- PML-route rules from 7a at normal incidence: w₀ = 0.75 µm ≳ λ, and pml_start = 3.25 µm ≥ 3.5 w₀ = 2.63 µm.
  The dark band is 1.6·10⁻⁸, far below the default 1e-4, which is kept.
- **L ×1.2 is not on the λ/4 grid.** 1.2 L = 5.4 µm puts the rim at 2.7 µm, which is not a multiple of 125 nm.
  The partners bracket it instead: L = 5.25 µm (×1.167) and 5.5 µm (×1.222).
- The uniform box at h = 125 nm has cells above λ₁/5 = 100 nm, the ADR 0006 bound for box cells. The validated
  box is 50 nm. The brief prescribes the same patch mesh on both routes, with h = λ/4 primary. The λ/6 and λ/10
  box runs satisfy the bound.

### Height-map identity

Every run builds both meshes and checks them against the cropped map (`pml_si_crosscheck::height_identity`):
- each node of the common L × L grid in the open mesh;
- the box top-face vertex at that node (the vertex of smallest z: walls and bottom lie below the rim).

Their z must equal the map value **bitwise**. All runs report `bitwise yes`: 0 mismatches on 1 369 nodes (λ/4,
L = 4.5 µm), 2 025 (L = 5.5 µm), 3 025 (λ/6), 841 / 1 849 (L = 3.5 µm at λ/4 / λ/6). The largest x/y offset
between the two meshes' vertices is 4.2·10⁻²² m. The `IDENTITY` lines are in the rows file.

**Triangle vertex triples agree.** The patch triangles of the open mesh are the box top-face triangles, compared
as oriented vertex triples on the common grid (`pml_si_crosscheck::patch_triangle_identity`), not only by count:
2 592 of 2 592 at λ/4, L = 4.5 µm, 1 568 of 1 568 at λ/4, L = 3.5 µm and 3 528 of 3 528 at λ/6, L = 3.5 µm, i.e.
for every configuration solved on both routes. This was checked after the runs on the same meshes (unit test
"the PML runs of the record share heights and triangles with the box", meshing only); the benchmark and the
validation-large test now check it before solving.

The unit test also checks that the identity detects:
- a 1 pm change at one node of the map (both meshes against the map);
- a 1 pm change of one patch vertex of the open mesh (open against map and box);
- an open mesh shifted by a quarter cell (no common node, no matched triangle);
- the reversed open mesh (no matched oriented triple).

## Results at the 7e configuration (L = 4.5 µm, w₀ = 0.75 µm, h = λ/4)

| Route | 2N | Solver | It. | Residual | R | Assembly / solve / post [s] | Peak [GB] (estimate) |
|---|---:|---|---:|---:|---:|---|---:|
| Box (uniform, depth 5.65 µm) | 55 296 | MLFMM d₀ 3 + GMRES | 92 | 9.8·10⁻⁶ | 0.3856955 | 248 / 441 / 83 | 12.7 (15.4) |
| PML (patch, taper, gap, PML) | 27 472 | dense + LAPACK LU | — | 2.7·10⁻¹³ | 0.3856893 | 2 138 / 368 / 15 | 24.3 (24.2) |

PML route diagnostics:
- 5 728 of the 13 736 test functions are masked (42 %).
- The dark band is 1.6·10⁻⁸.
- Edge decay at the truncation: |J| 2.2·10⁻⁷, |M| 4.4·10⁻⁷ of their maxima.
- The far-field degree check is 4.9·10⁻⁸.

The PML route has half the unknowns of the box: no walls or bottom, which hold 86 % of the box triangles at this
mesh. It is still about 3.3× slower here in wall time (2 521 s against 772 s), because of the stretched Si kernels in the PML ring (WP-PML3a: the
touching order reaches p = 16 for |k s| h large) and the dense O(N²) assembly. Against the MLFMM box this is not
a like-for-like cost comparison.

### Comparison and the box's own sensitivity

| Comparison (test vs reference), h = λ/4 | l2_refl | l2_off30 | ε_rr xz / yz | dR |
|---|---:|---:|---:|---:|
| **PML route vs box, L = 4.5 µm** | **0.146 %** | 0.402 % | 0.026 / 0.041 % | −1.6·10⁻⁵ |
| Box L = 5.5 vs 4.5 µm (×1.222) | 0.155 % | 0.498 % | 0.026 / 0.032 % | −3.7·10⁻⁵ |
| Box L = 5.25 vs 4.5 µm (×1.167) | 0.161 % | 0.469 % | 0.021 / 0.041 % | −2.3·10⁻⁵ |

R is printed to seven digits. The R values are 0.3856955 (box 4.5 µm), 0.3856893 (PML route), 0.3856814
(box 5.5 µm) and 0.3856865 (box 5.25 µm).

### Tolerance

**Rule (fixed before the PML route was solved).** For each quantity, the tolerance is the largest change of the
uniform box under its own L ×1.2 check at w₀ = L/5…L/6, σ = 50 nm. The values come from two sources, all
measured for the box alone:

| Source (box only) | l2_refl | ε_rr (max of xz, yz) | \|dR\| (\|dP_refl\| in the record) |
|---|---:|---:|---:|
| WP-V2, seed 1, L/6, 4.5 → 5.4 µm, h = 50 nm | 0.19 % | 0.028 % | 2.5·10⁻⁵ |
| WP-V2b (a), seed 2, L/6 | **0.229 %** | **0.058 %** | **4.1·10⁻⁵** |
| WP-V2b (c), seed 1, L/5 (w₀ = 0.9 µm) — the waist variation | 0.152 % | 0.027 % | 1.6·10⁻⁵ |
| this study, λ/4, ×1.222 (5.5 µm) | 0.155 % | 0.032 % | 3.7·10⁻⁵ |
| this study, λ/4, ×1.167 (5.25 µm) | 0.161 % | 0.041 % | 2.3·10⁻⁵ |
| **Tolerance = max** | **0.229 %** | **0.058 %** | **4.1·10⁻⁵** |

Order of events (commit time and log timestamps, 2026-10-15):
- The constants `pml_si_crosscheck::kTolerances` = {2.29e-3, 5.8e-4, 4.1e-5} were committed at 11:19 (3c6b5b0),
  from the WP-V2/V2b records alone. At that time the λ/4 reference box (11:10–11:22) was still running.
- This study's λ/4 box partners ran afterwards (L = 5.5 µm 11:23–11:35, L = 5.25 µm 11:36–11:52). They did not
  raise any maximum, so the constants stood unchanged.
- The PML route started at 11:53, after all three box runs. The validation-large test asserts these constants.

**The ε_rr pass rests on the cross-seed maximum.** The PML–box ε_rr on the yz cut is 0.0413 %. It passes only
because the tolerance is the largest value over the measured configurations, here WP-V2b (a), seed 2 (0.058 %).
This study's own largest same-surface partner value is 0.0410 % (L ×1.167, yz), which the PML route exceeds by
0.0003 percentage points. For l2_refl (0.146 % against the same-surface partners' 0.155–0.161 %) and R
(1.6·10⁻⁵ against 2.3–3.7·10⁻⁵) the PML route lies below the same-surface partners as well.

**Acceptance test.** `ctest --preset win-release -L validation-large -R "7e" --output-on-failure` passed (12
assertions, 5 993 s on the shared machine; log `pml_si_crosscheck_logs/ctest_validation_large_7e.log`, lines in the
rows file). It was run after merging WP-LU1, i.e. with the library's `solver::DenseLu`. It reproduces the
benchmark runs: box R 0.3856955 and 92 iterations, PML route R 0.3856893, l2_refl 0.1464 %, ε_rr 0.0257 / 0.0413 %,
dR −1.617·10⁻⁵, each identical to the printed digits. The LU residual is 4.5·10⁻¹⁶ (the support-class LU of the
benchmark runs: 2.7·10⁻¹³). A first ctest run with the pre-merge build passed with the same numbers (4 129 s).

Justification:
- The PML route replaces the box rim, walls and bottom by a taper into a flat, absorbing surrounding. What
  differs between the two models is what happens at and beyond the patch edge, under the same footprint.
- The box's sensitivity to moving that edge (its L ×1.2 check, the docs/05 edge-effect check) measures how
  much the far field depends on the edge treatment at this L and w₀. It is the box's own uncertainty, in the
  sense of ADR 0006 amendment 2026-10-13, decision 2, which uses the L ×1.2 change as a stand-in for the edge
  error.
- Over seeds and waists the measured values spread by about ±20 % (0.15–0.23 %, WP-V2b). The maximum over the
  measured configurations is the box's demonstrated sensitivity envelope.
- The depth ×2 change (0.007–0.009 %, WP-V2/V2b) is 25× smaller, so it is not a separate term. The docs/05
  thresholds (depth < 0.1 %, L ×1.2 < 0.5 %) are the box's acceptance limits, not its sensitivity, and are
  **not** used as the tolerance.
- The PML route's discretisation and dark-band errors are not added: tolerances are not widened beyond the
  box's sensitivity.

## Mesh check

### Box route, the same surface at λ/6 (and λ/10)

| Comparison | l2_refl | l2_off30 | ε_rr xz / yz | dR |
|---|---:|---:|---:|---:|
| Box λ/6 vs λ/4, L = 4.5 µm | **2.32 %** | 4.69 % | 0.79 / 0.65 % | +7.5·10⁻⁴ |
| Box λ/6, L = 5.5 vs 4.5 µm (×1.222) | 0.136 % | 0.436 % | 0.032 / 0.026 % | −2.4·10⁻⁵ |
| Box λ/10 vs λ/6, L = 4.5 µm | **1.23 %** | 2.22 % | 0.42 / 0.36 % | −3.7·10⁻⁴ |
| Box λ/10 vs λ/4, L = 4.5 µm | 3.53 % | 6.99 % | 1.20 / 0.99 % | +3.9·10⁻⁴ |

R of the box at L = 4.5 µm:

| h | R |
|---|---:|
| λ/4 | 0.3856955 |
| λ/6 | 0.3859860 |
| λ/10 (= 50 nm, the validated box mesh) | 0.3858450 |

The λ/10 vs λ/4 line comes from the parallel run of this WP on the other branch, whose λ/10 box run has the same
ROW values (2N, iterations, residual, R, P_refl) as the one here, only its times differ; it is in the rows file.

- λ/4 = 1.08 Si wavelengths, which is not enough for the far field of the rough surface: the λ/6 far field
  differs by 2.3 % (l2) and 4.7 % off-specular.
- **λ/6 is not converged either.** λ/10 (h = 50 nm, the cell size of the validated box record) moves the far
  field by another 1.23 % (l2), 2.2 % off-specular. The successive differences 2.32 % and 1.23 % fit
  second-order convergence: (h₄² − h₆²) / (h₆² − h₁₀²) = 1.95 for the measured ratio 1.89 (first order: 1.25).
  With p = 2 the remaining mesh error is roughly 0.7 % at λ/10, 1.9 % at λ/6 and 4 % at λ/4 (an estimate from
  norms of differences, not a signed Richardson extrapolation). An absolute 0.2 % would need about λ/20.
- R moves by +7.5·10⁻⁴ from λ/4 to λ/6 (at L = 3.5 µm: +1.47·10⁻³, both routes), then −3.7·10⁻⁴ from λ/6 to
  λ/10: it is not monotone in h. λ/4 and λ/6 are within 4·10⁻⁴ of λ/10 at L = 4.5 µm.
- The L2 metric of |F|² is far more mesh-sensitive than R: the speckle-like off-specular pattern needs the fine
  mesh.
- 7a found λ/4 sufficient for *flat* Si at normal incidence (R −0.022 %). The local slopes of the rough surface
  give the currents a tangential phase, much like oblique incidence on the flat surface, where 7a needed λ/6.
- The box's L sensitivity does not depend on the mesh (0.155 % at λ/4, 0.136 % at λ/6, ×1.222).
- Box at λ/6, L = 4.5 µm: 2N = 123 120, 151 iterations, 22.3 GB. Box at λ/10: 2N = 341 280, 203 iterations,
  28.0 GB (estimate 35.1 GB), 880 s assembly, 2 922 s solve, 854 s far field. The PML route at λ/10 would need
  477 GB dense.

### Mesh check of the cross-check (L = 3.5 µm, w₀ = L/6 = 0.583 µm)

The PML route at λ/6 needs 61.5 GB dense at L = 4.5 µm and 48.2 GB at L = 4 µm. Both exceed the 50 GB budget,
the latter with margin for nothing. L = 3.5 µm is the largest L whose zone boundaries lie on both the λ/4 and the
λ/6 grids: 2N = 48 240, 37.2 GB dense, solved with GMRES since an LU copy would not fit.

The PML-route rules hold there:
- w₀ = 0.583 µm ≥ λ;
- pml_start = 2.75 µm ≥ 3.5 w₀ = 2.04 µm;
- dark band 1.3·10⁻⁷.

The box partners are L = 4.0 µm (×1.143) and 4.5 µm (×1.286) at the same w₀. 1.2 L = 4.2 µm lies on neither
grid.

| Comparison | h | l2_refl | l2_off30 | ε_rr xz / yz | dR |
|---|---|---:|---:|---:|---:|
| **PML route vs box** | λ/4 | **0.210 %** | 0.518 % | 0.051 / 0.071 % | −4.9·10⁻⁵ |
| Box L = 4.0 vs 3.5 µm | λ/4 | 0.260 % | 0.622 % | 0.052 / 0.082 % | −5.7·10⁻⁵ |
| Box L = 4.5 vs 3.5 µm | λ/4 | 0.213 % | 0.677 % | 0.054 / 0.038 % | −5.4·10⁻⁵ |
| Box λ/6 vs λ/4, L = 3.5 µm | | 2.03 % | 4.20 % | 0.86 / 0.57 % | +1.5·10⁻³ |
| **PML route vs box** | λ/6 | **0.186 %** | 0.415 % | 0.036 / 0.063 % | −4.6·10⁻⁵ |
| Box L = 4.0 vs 3.5 µm | λ/6 | 0.271 % | 0.480 % | 0.049 / 0.089 % | −4.1·10⁻⁵ |
| Box L = 4.5 vs 3.5 µm | λ/6 | 0.189 % | 0.546 % | 0.033 / 0.020 % | −6.2·10⁻⁵ |
| PML route λ/6 vs λ/4, L = 3.5 µm | | 2.02 % | 4.20 % | 0.89 / 0.57 % | +1.5·10⁻³ |

R at L = 3.5 µm:

| h | Box | PML route |
|---|---:|---:|
| λ/4 | 0.3861382 | 0.3861193 |
| λ/6 | 0.3867074 | 0.3866898 |

PML–box against the box's own sensitivity at this L (the larger of the two partners per quantity):

| Mesh | l2_refl | ε_rr | R |
|---|---|---|---|
| λ/4 | 0.210 vs 0.260 % (0.81×) | 0.071 vs 0.082 % (0.87×) | 4.9·10⁻⁵ vs 5.7·10⁻⁵ (0.86×) |
| λ/6 | 0.186 vs 0.271 % (0.69×) | 0.063 vs 0.089 % (0.70×) | 4.6·10⁻⁵ vs 6.2·10⁻⁵ (0.74×) |

At L = 3.5 µm the box is more sensitive to its edge than at 4.5 µm (0.21–0.26 % vs 0.16 %). The PML–box
difference grows with it: 0.210 % against 0.146 %. It stays within the box's own sensitivity at that L (table above).
The fixed 7e constants belong to the L = 4.5 µm configuration and are
not applied to this L; the COMPARE lines in the rows file print them anyway, so their pass/FAIL flags do not
apply to these L = 3.5 µm rows.

**The cross-check holds at λ/6.** At h = λ/6 the PML–box difference is 0.186 % (l2), 0.063 % (ε_rr, yz) and
−4.6·10⁻⁵ (R). Against the box's own sensitivity at the same mesh and L (the larger of the two partners per
quantity: 0.271 %, 0.089 %, 6.2·10⁻⁵) these are 0.69×, 0.70× and 0.74×, i.e. the same picture as at λ/4
(0.81×, 0.87×, 0.86×), slightly tighter.
- Both routes move by the same amount from λ/4 to λ/6: 2.02 % (PML) and 2.03 % (box) in l2, +1.48·10⁻³ /
  +1.47·10⁻³ in R. The 2 % mesh error is therefore a property of the shared patch
  mesh, not of either termination, and the inter-route difference (0.19–0.21 %) does not depend on it.
- The PML route at λ/6 was solved with full GMRES (tol 1e-8, no preconditioner): 233 iterations, true residual
  9.7·10⁻⁹. Its dark band (1.3·10⁻⁷) and edge decay (|J| 6.4·10⁻⁸, |M| 5.5·10⁻⁸) are as at λ/4.

## Costs

| Run | 2N | Assembly | Solve | Post | Peak [GB] |
|---|---:|---:|---:|---:|---:|
| Box λ/4, L 4.5 µm | 55 296 | 248 s | 441 s (92 it.) | 83 s | 12.7 |
| Box λ/4, L 5.5 µm | 71 808 | 244 s | 423 s (93 it.) | 110 s | 16.1 |
| Box λ/4, L 5.25 µm | 67 536 | 272 s | 627 s (92 it.) | 99 s | 14.9 |
| PML λ/4, L 4.5 µm (LU) | 27 472 | 2 138 s | 368 s | 15 s | 24.3 |
| Box λ/6, L 4.5 µm | 123 120 | 177 s | 474 s (151 it.) | 102 s | 22.3 |
| Box λ/6, L 5.5 µm | 159 984 | 205 s | 1 163 s (150 it.) | 296 s | 18.4 |
| Box λ/10, L 4.5 µm | 341 280 | 880 s | 2 922 s (203 it.) | 854 s | 28.0 |
| Box λ/4, L 3.5 µm | 40 320 | 169 s | 228 s (93 it.) | 34 s | 8.9 |
| PML λ/4, L 3.5 µm (LU) | 21 360 | 2 530 s | 206 s | 20 s | 14.7 |
| Box λ/6, L 3.5 µm | 89 712 | 207 s | 550 s (154 it.) | 281 s | 16.2 |
| PML λ/6, L 3.5 µm (GMRES, tol 1e-8) | 48 240 | 6 376 s | 598 s (233 it.) | 57 s | 37.4 |

The w₀ = 0.583 µm box partners (L = 4.0 / 4.5 µm): λ/4 84 / 244 s assembly, 277 / 490 s solve (95 / 94 it.),
10.7 / 12.7 GB; λ/6 261 / 339 s, 535 / 830 s (156 / 154 it.), 19.4 / 22.3 GB (rows file). Peaks are measured
process peaks. All times are on the shared machine (see Setup).

- Wall time per route at the 7e configuration: box 13 min, PML route 42 min.
- The estimates hold: PML 24.15 GB estimated against 24.26 GB measured; box 15.4 GB estimated against 12.7 GB
  measured (0.82, as in WP-V2b).
- The dense PML assembly dominates every PML run. At λ/6 (L = 3.5 µm) it took 1.8 h for 2N = 4.8·10⁴ on the
  loaded machine.

## Conclusion

**What 7e shows.** For the same rough Si surface under the same beam, the open-surface PML route and the
validated uniform closing box give the same far field to within the box's own edge sensitivity:
- 7e configuration (L = 4.5 µm, w₀ = L/6, λ/4): 0.146 % against 0.229 % (l2), 0.041 % against 0.058 % (ε_rr),
  1.6·10⁻⁵ against 4.1·10⁻⁵ (R), with the tolerance fixed before the PML solve. Met. The ε_rr pass uses the
  cross-seed maximum: against this surface's own L partners (0.0410 %) the PML route's 0.0413 % is level, not
  below.
- L = 3.5 µm, w₀ = L/6: 0.210 % (λ/4) and 0.186 % (λ/6), each below the box's L-partner changes at the same
  mesh (0.69–0.87× per quantity). The agreement does not depend on the mesh.
- The difference between the routes behaves like the box's edge effect: it grows as L shrinks at fixed
  w₀/L (0.146 % → 0.21 %), as the box's own L sensitivity does (0.16 % → 0.21–0.27 %), and it sits
  off-specular. Neither termination adds an error of its own that is visible above that level.
- Reading: the two terminations are equivalent to the accuracy at which the box's edge treatment is defined.
  Their difference is the residual edge effect of truncating the rough patch, which the PML route models as a
  rough spot in a polished surface (ADR 0009, lead decision 1) and the box as a finite rough block.

**Caveat: the mesh, not the termination, limits the accuracy.** h = λ/4 (1.08 Si wavelengths) is not
mesh-converged for the rough-Si far field: λ/6 moves it by 2.0–2.3 % (l2) and 4–5 % off-specular, on both
routes alike, and λ/10 moves the box far field by another 1.23 %. The sequence fits second-order convergence,
with a remaining mesh error of roughly 4 % at λ/4, 2 % at λ/6 and 0.7 % at λ/10 for this surface (σ = 50 nm,
Lc = 500 nm); 0.2 % would need about λ/20. 7e is a statement about the terminations at equal discretisation,
where the mesh error cancels (the inter-route difference is the same at λ/4 and λ/6). 7a's λ/4 rule for flat Si
at normal incidence does not carry over to rough Si. R is less mesh-sensitive: it moves by 0.75–1.5·10⁻³ from
λ/4 to λ/6, and λ/4 and λ/6 are within 4·10⁻⁴ of λ/10 at L = 4.5 µm.

**Scope.** One seed and one roughness (σ = 50 nm, Lc = 500 nm), normal incidence, x polarisation. σ = 100 nm,
where the box's own L ×1.2 check fails (WP-V2b), is not covered. The λ/6 cross-check uses L = 3.5 µm, because
the 7e L does not fit densely at λ/6. The PML route at λ/10 (2N ≈ 1.7·10⁵ at L = 4.5 µm, 477 GB dense) is far
beyond the dense prototype; it needs the fast PML ring of WP-PML5.

**Costs are not like for like.** The PML route here is the dense prototype (ADR 0009 decision 6): O(N²)
assembly with the stretched Si kernels in the PML ring, LU or GMRES on a dense matrix. It has about half the
unknowns of the box (no walls or bottom) but takes 3–7× the box's wall time and 1.7–2.3× its peak memory, and its
λ/6 run at L = 4.5 µm (61.5 GB) does not fit where the MLFMM box needs 22 GB. The comparison says nothing about
the cost of a fast PML route, which needs the near-field / ACA treatment of the PML pairs (ADR 0009 decision 6).

**What a user should take from it.**
- For rough Si at w₀ = L/6 the uniform box and the PML route are interchangeable at the 0.2 % level (shown at
  σ = 50 nm, one seed, normal incidence); at present the MLFMM box is the cheaper route.
- The PML route's value is for the cases the box cannot do (metals, ADR 0009). 7e checks its termination
  against an independent one on a material where both are valid; it does not replace the box on Si.
- Choose h from a mesh study of the material and roughness, not from the flat-surface rule: here the mesh error
  (2 % at λ/6, 0.7 % at λ/10) is larger than the termination differences. Keep w₀ ≤ L/6 and
  pml_start ≥ 3.5 w₀ (7a) on the PML route.

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
# run with C:/msys64/ucrt64/bin first on PATH; OMP_NUM_THREADS=12; one solve at a time
B=build/win-release/benchmarks/specklebem_pml_si_crosscheck
$B --route pml --estimate-only                                  # sizes, identity, dark band, no assembly
$B --route box --L 4.5e-6  --tag box_l45_h4  --out runs/box_l45_h4                        # box, 7e configuration
$B --route box --L 5.5e-6  --tag box_l55_h4  --out runs/box_l55_h4  --reference runs/box_l45_h4
$B --route box --L 5.25e-6 --tag box_l525_h4 --out runs/box_l525_h4 --reference runs/box_l45_h4
$B --route pml --L 4.5e-6  --tag pml_l45_h4  --out runs/pml_l45_h4  --reference runs/box_l45_h4  # 7e
H6=83.33333333333333e-9; W=0.5833333333333333e-6
$B --route box --h $H6 --tag box_l45_h6 --out runs/box_l45_h6 --reference runs/box_l45_h4       # box mesh check
$B --route box --h 50e-9 --tag box_l45_h10 --out runs/box_l45_h10 --reference runs/box_l45_h6  # lambda/10, 28 GB
$B --route pml --L 3.5e-6 --w0 $W --h $H6 --pml-solver gmres --tag pml_l35_h6 --out runs/pml_l35_h6 \
   --reference runs/box_l35_h6                                   # cross-check at lambda/6 (L 3.5 um)
$B --compare runs/box_l45_h4 runs/pml_l45_h4                     # stored runs only
ctest --preset win-release -L validation-large -R "7e"           # the acceptance case as a test
```
