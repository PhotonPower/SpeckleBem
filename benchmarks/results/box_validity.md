# Validity of the graded closing box under beam illumination (WP-V1, ADR 0006)

Does the closing box of a truncated rough surface (ADR 0006) change the reflected far field?
Four questions, dense operator, for Ag and Si at λ = 500 nm: (1) box depth ×2 (docs/05: < 0.1 %),
(2) graded vs uniform walls and bottom, (3) Si fine band vs none, (4) L ×1.2 at fixed waist
(docs/05: < 0.5 %). Executable `benchmarks/box_validity.cpp` (`specklebem_box_validity`, built with
`-DSPECKLEBEM_BUILD_BENCHMARKS=ON`).

**Short answer.** At the sizes a dense operator allows (L ≤ 2.4 µm, w₀ ≤ 0.67 µm) neither docs/05
sensitivity check is met. Two contributions are visible, but the study does not separate them
cleanly: (a) the graded box itself: the automatic 400 nm coarse cells (λ/1.25) are a far-field
error of 1.5–2 % against the uniform box (Ag); a 100 nm coarse spacing (λ/5) reduces it to 0.17 %
at L = 1.5 µm but only to 0.59 % at L = 1 µm, and at L = 1 µm with depth ×2 the 100 nm box is
still 1.54 % from the uniform box. λ₁/5 is therefore necessary at fixed depth; it is not shown to
be sufficient for the depth check, whose 0.1 % tolerance is below the 100 nm grading error in
every Ag configuration with a uniform reference. (b) The illumination of the box: at these sizes
the edge of the patch and the walls are lit at the percent level (0.6–0.9 % of the rigorous beam
power crosses z = 0 outside L × L at w₀ = L/3), which shows in the uniform box as well (depth ×2:
1.1 %) and decides L ×1.2 (4–6 %, unchanged by the grading). The depth sensitivity tracks the
illumination of the rim (w₀ = L/4: 3× lower at a beam as wide in the box), which suggests, but
does not prove, that it vanishes at w₀ ≪ L; how small w₀/L, and so how large L, must be is
open (an earlier Rayleigh-range estimate, L ≳ 6–11 µm, is contradicted by the w₀ = L/4 data and
holds at most conditionally on z_R ≫ depth being the controlling quantity). With a 100 nm
box the depth ×2 change is 1.76 % at L = 1 µm (uniform box 1.10 %, 400 nm 3.03 %) and 3.49 % at
L = 2 µm (400 nm 2.25 %, no uniform reference): the grading does not change the depth
sensitivity monotonically with the cell size. The one configuration close to the
depth criterion is Si with the fine band: 0.13 % (ε_rr 0.04 %).

## Setup

| Parameter | Value |
|---|---|
| Wavelength, media | λ = 500 nm, vacuum R1; Si ε_r = 18.478 − 0.606j (δ = 1.129 µm), Ag ε_r = −9.794 − 0.313j (δ = 25.4 nm), exp(+jωt) |
| Surface | Gaussian height map, σ = 50 nm, Lc = 500 nm, h = 50 nm, seed 1, generated on an L_gen × L_gen patch and cropped centrally to L (`--L-gen`), so that L and 1.2 L share the surface on the common area. Families: L = 1.0/1.2 µm (map 1.2 µm), 1.5 µm (map 1.5 µm), 2.0/2.4 µm (map 2.4 µm) |
| Box | depth `default_box_depth` (Ag 2 µm, Si 5.65 µm) or ×2; coarse spacing `box_mesh_size` = 400 nm (explicit, M = 3; the automatic rule would give L/8 and change with L), 200 nm (M = 2), 100 nm (M = 1) or uniform (50 nm); Si fine band `default_box_fine_depth` = 3δ + 3σ = 3.54 µm or none |
| Beam | rigorous angular-spectrum Gaussian beam built in the study (exact vacuum Maxwell solution: Gauss–Legendre × trapezoid quadrature over propagating plane waves with spectrum exp(−k_t² w₀²/4), E-vector x̂ − (k̂·x̂) k̂, E_x = 1 V/m at the focus; grid refined until a 1.5× finer grid changes E by < 1e-10 on the mesh, 1 000–6 000 plane waves); normal incidence, +z, focus at the origin, E along x. Waist w₀ = L/3 of the smaller L of a family, kept fixed for L ×1.2; w₀ = L/4 as a second waist. Control runs with the paraxial `excitation::GaussianBeam` |
| Solver | ICTF, no preconditioner, full GMRES, tolerance 1e-6 (true residual ≤ 1e-6 in every run). Solver noise: the Ag L = 2 µm base at 1e-7 differs from 1e-6 by 1.6e-5 (L2) — 1 000× below the effects measured |
| Far field | `post::far_field`, Dunavant degree 10 (degree-8 check ≤ 1.2e-6 relative); reflection hemisphere (k̂_z < 0) on θ = 0.5…89.5° (1°) × φ = 0…357° (3°), forward hemisphere likewise, cuts in the xz- and yz-planes (θ = −89.5…89.5°, 0.5°) |
| Metrics | **l2_refl** = sqrt(Σ w (I_T − I_R)²/Σ w I_R²), I = \|F\|², w = sin θ, over the reflection hemisphere (the verdict metric); l2_off30 the same for θ ≥ 30°; **ε_rr** (docs/05) = sqrt(mean (I_T − I_R)²)/max I_R along the xz / yz cuts; dP = relative change of the power into the reflection hemisphere |
| Diagnostic | PARTS: power radiated into the reflection hemisphere by the currents of the top face, walls, bottom, rim (top–wall) and foot (wall–bottom) RWG functions alone, relative to the total |
| Machine | Windows 11, 24 threads, 128 GB, MSYS2 UCRT64 GCC 16, OpenBLAS, `win-release` + benchmarks, 2026-10-09/10; other agents' builds and tests shared the machine, so times are indicative |

## Runs (unknowns, cost)

| Run | Material, L, w₀ | Box | 2N | Z [GB] | GMRES it. | Assembly [s] | Solve [s] |
|---|---|---|---:|---:|---:|---:|---:|
| ag1_base / ag1_d2 | Ag 1.0 µm, 0.333 µm | 400 nm, depth 2 / 4 µm | 3 930 / 4 290 | 0.25 / 0.29 | 762 / 1 084 | 1.4 / 1.9 | 11 / 22 |
| ag1_uni / ag1_uni_d2 | Ag 1.0 µm, 0.333 µm | uniform, 2 / 4 µm | 24 000 / 43 200 | 9.2 / 29.9 | 1 151 / 1 759 | 13 / 55 | 547 / 2 485 |
| ag1_b100 / ag1_b100_d2 | Ag 1.0 µm, 0.333 µm | 100 nm, 2 / 4 µm | 8 400 / 13 200 | 1.1 / 2.8 | 1 236 / 1 679 | 7 / 16 | 172 / 675 |
| ag15_base / b200 / b100 / uniform | Ag 1.5 µm, 0.5 µm | 400 / 200 / 100 nm / uniform, 2 µm | 7 584 / 9 072 / 14 850 / 39 600 | 0.9 / 1.3 / 3.5 / 25.1 | 948 / 1 479 / 1 947 / 1 901 | 4 / 4 / 8 / 59 | 46 / 105 / 347 / 2 383 |
| ag2_base / d2 / L24 | Ag 2.0 / 2.0 / 2.4 µm, 0.667 µm | 400 nm, 2 / 4 / 2 µm | 12 450 / 13 050 / 17 280 | 2.5 / 2.7 / 4.8 | 1 257 / 1 792 / 1 574 | 9–16 / 12 / 16 | 145 / 274 / 505 |
| ag2_b200 / b100 | Ag 2.0 µm, 0.667 µm | 200 / 100 nm, 2 µm | 14 640 / 23 280 | 3.4 / 8.7 | 1 872 / 2 273 | 20 / 24 | 408 / 1 147 |
| ag2_b100_d2 / b100_L24 | Ag 2.0 / 2.4 µm, 0.667 µm | 100 nm, 4 / 2 µm | 32 880 / 30 240 | 17.3 / 14.6 | 2 708 / 2 722 | 37 / 30 | 2 667 / 1 995 |
| ag2_w05 / w05_d2 / w05_L24 | Ag 2.0 / 2.0 / 2.4 µm, 0.5 µm | 400 nm | 12 450 / 13 050 / 17 280 | | 1 228 / 1 785 / 1 575 | | 186 / 386 / 372 |
| si1_none / none_d2 / si12_none | Si 1.0 / 1.0 / 1.2 µm, 0.333 µm | 400 nm, no band, 5.65 / 11.3 / 5.65 µm | 4 650 / 5 658 / 5 850 | ≤ 0.6 | 861 / 1 051 / 869 | 10 / 19 / 14 | 17 / 49 / 25 |
| si1_b100 | Si 1.0 µm, 0.333 µm | 100 nm, no band | 17 280 | 4.8 | 404 | 95 | 68 |
| si1_fine / fine_d2 / si12_fine | Si 1.0 / 1.0 / 1.2 µm, 0.333 µm | 400 nm + fine band 3.54 µm | 38 082 / 39 090 / 45 522 | 23.2 / 24.4 / 33.2 | 727 / 1 244 / 734 | 138 / 181 / 207 | 590 / 1 331 / 936 |
| si1_uniform | Si 1.0 µm, 0.333 µm | uniform, 5.65 µm | 59 040 | 55.8 | 299 | 306 | 1 046 |
| si15_none / si15_fine | Si 1.5 µm, 0.5 µm | 400 nm, no band / fine band | 8 544 / 59 520 | 1.2 / 56.7 | 935 / 839 | 26 / 439 | 49 / 5 672 |

Dense memory 16 (2N)² bytes; GMRES Krylov storage ≤ 1.4 GB. The largest systems that fit the
budget (2N ≤ 6·10⁴): Si with the fine band at L = 1.5 µm (2N = 59 520), Ag uniform box at
L = 2 µm (2N = 57 600; not run: ~2.5 h of GMRES at the measured 3 s/iteration, so the uniform
reference was taken at L = 1.5 µm). Si with fine band at L = 1.8 µm (72 810) and the Ag uniform box
at depth ×2 (L = 2 µm: 96 000) do not fit.

## 1. Box depth ×2 (docs/05: < 0.1 %)

| Material, L, w₀ | Box | l2_refl | l2_off30 | ε_rr xz / yz | dP | Verdict |
|---|---|---:|---:|---:|---:|---|
| Ag 1.0 µm, L/3 | uniform | 1.10 % | 2.4 % | 0.45 / 0.31 % | −3.0e-4 | fail (physical) |
| Ag 1.0 µm, L/3 | 100 nm | 1.76 % | 2.0 % | 1.04 / 1.00 % | +2.7e-4 | fail |
| Ag 1.0 µm, L/3 | 400 nm | 3.03 % | 3.2 % | 1.17 / 1.24 % | +8e-5 | fail |
| Ag 2.0 µm, L/3 | 400 nm | 2.25 % | 3.1 % | 0.65 / 0.46 % | −1.2e-4 | fail |
| Ag 2.0 µm, L/3 | 100 nm | 3.49 % | 3.0 % | 1.13 / 1.54 % | −5.6e-4 | fail |
| Ag 2.0 µm, L/4 | 400 nm | 0.77 % | 0.9 % | 0.27 / 0.15 % | −5.5e-5 | fail |
| Ag 2.0 µm, L/3, paraxial beam | 400 nm | 1.82 % | 2.8 % | 0.53 / 0.51 % | +3e-5 | fail |
| Ag 2.0 µm, L/3, Im ε = −1.5 / −6 (diagnostic) | 400 nm | 1.73 / 1.12 % | 2.3 / 1.5 % | 0.50 / 0.31 %; 0.34 / 0.12 % | | fail |
| Si 1.0 µm, L/3 | 400 nm, no band | 0.52 % | 0.8 % | 0.22 / 0.24 % | +4.5e-5 | fail |
| Si 1.0 µm, L/3 | fine band | **0.13 %** | 0.17 % | **0.044 / 0.041 %** | +2.1e-5 | l2 marginal fail, ε_rr pass |

The power into the reflection hemisphere changes by ≤ 6e-4 in every case: the depth changes the
angular distribution, not the reflectance. In the uniform Ag box the top-face share of the
reflected power is unchanged by the depth (0.9933 → 0.9937) while the wall share falls 1.40e-3 →
1.11e-3: the change comes from light reaching the walls.

*What drives it: rim illumination rather than Rayleigh spreading.* At L = 2 µm the paraxial beam
radius in the box is about the same for both waists — w(2 µm) = 0.82 µm (w₀ = L/3,
z_R = 2.79 µm) vs 0.81 µm (w₀ = L/4, z_R = 1.57 µm), w(4 µm) = 1.17 vs 1.37 µm, so the L/4 beam
is as wide or wider in the box, and |E_inc| at the wall foot (x = L/2, z = 4 µm) is 0.28 vs
0.22 V/m — yet the depth sensitivity is 3× lower at L/4 (0.77 vs 2.25 %). What differs is the
illumination of the rim: |E_inc| at (L/2, 0, 0) is 0.126 vs 0.029 V/m (intensity 18× lower), at
(0, L/2, 0) 0.105 vs 0.018 V/m, and the walls and rim radiate 5× / 4× less (PARTS 2.8e-3 vs
5.5e-4, 1.1e-4 vs 2.8e-5). The data therefore point to the light that hits the patch edge and
runs down the walls (which the depth change acts on) rather than to the beam spreading onto the
walls within the box depth. A Rayleigh-range criterion (z_R ≫ depth) is not supported by these
two points and is not derived; whether the rim illumination, i.e. w₀/L, alone controls the
sensitivity is a hypothesis with two data points.

*The grading changes the depth sensitivity, not monotonically.* At L = 1 µm depth ×2 is 1.10 %
in the uniform box, 1.76 % with 100 nm cells and 3.03 % with 400 nm cells; at L = 2 µm it is
2.25 % with 400 nm and 3.49 % with 100 nm cells (no uniform reference: the uniform depth-4 µm box
has 2N = 96 000). The 100 nm box is itself 0.59 % (depth 2 µm) and 1.54 % (depth 4 µm) from the
uniform box at L = 1 µm (section 2), so the depth check of a graded box measures the depth
dependence of the grading error as well as the physical sensitivity, and neither 400 nm nor
100 nm cells isolate the latter. At L = 2 µm the finer box resolves the wall waves (surface
plasmons, λ_spp ≈ 0.95 λ, which 400 nm cells cannot carry), which may explain why it is more
depth-sensitive there; this is not tested. Si with the fine band is the best case: its bottom
radiates nothing (PARTS ≤ 4e-6, δ = 1.13 µm and 5.65 µm depth) and the remaining 0.13 % is the
wall illumination.

Making the silver lossier (diagnostic runs, Ag L = 2 µm, 400 nm box) lowers the depth
sensitivity from 2.25 % to 1.73 % (Im ε = −1.5) and 1.12 % (Im ε = −6, ×19) but does not remove
it. The flat-interface surface-plasmon amplitude decay length, 1/|Im k_spp| with
k_spp = k₀ √(ε/(1 + ε)), is 41.5 µm for the tabulated Ag, 8.9 µm for Im ε = −1.5 and 3.1 µm for
Im ε = −6, i.e. at Im ε = −6 a plasmon running down a wall and back (2 × 2 µm, or 2 × 4 µm at
depth ×2) is damped by e^−1.3 … e^−2.6. Halving the sensitivity with that damping is consistent
with plasmons along the walls carrying about half of it: they are one mechanism, not the only
one, and the rest (the incident beam reaching the walls and corners directly) is not affected by
the loss. The loss also changes the top-face response, so the split is indicative only.

## 2. Graded vs uniform walls and bottom

| Material, L, w₀, depth | Test vs reference | l2_refl | l2_off30 | l2 forward | ε_rr xz / yz | dP |
|---|---|---:|---:|---:|---:|---:|
| Ag 1.5 µm, L/3, 2 µm | 400 nm vs uniform | 1.51 % | 3.9 % | 4.6 % | 0.42 / 0.57 % | +7.3e-4 |
| Ag 1.5 µm, L/3, 2 µm | 200 nm vs uniform | 1.94 % | 3.3 % | 4.1 % | 0.41 / 0.78 % | +1.5e-3 |
| Ag 1.5 µm, L/3, 2 µm | **100 nm vs uniform** | **0.17 %** | 0.39 % | 0.36 % | **0.061 / 0.069 %** | +2.8e-5 |
| Ag 1.0 µm, L/3, 2 µm | 400 nm vs uniform | 1.92 % | 3.9 % | 5.8 % | 0.80 / 0.68 % | +1.1e-3 |
| Ag 1.0 µm, L/3, 2 µm | **100 nm vs uniform** | **0.59 %** | 0.68 % | 0.81 % | 0.31 / 0.27 % | −1.8e-4 |
| Ag 1.0 µm, L/3, 2 µm | 100 nm vs 400 nm | 2.15 % | 3.8 % | 6.4 % | 0.98 / 0.85 % | −1.3e-3 |
| Ag 1.0 µm, L/3, 4 µm | 400 nm vs uniform | 1.88 % | 3.6 % | 5.4 % | 0.50 / 0.73 % | +1.5e-3 |
| Ag 1.0 µm, L/3, 4 µm | **100 nm vs uniform** | **1.54 %** | 1.0 % | 1.6 % | 0.85 / 0.84 % | +3.8e-4 |
| Ag 1.0 µm, L/3, 4 µm | 100 nm vs 400 nm | 3.00 % | 3.5 % | 4.6 % | 1.26 / 1.46 % | −1.1e-3 |
| Ag 2.0 µm, L/3, 2 µm | 200 / 100 nm vs 400 nm | 2.36 / 1.44 % | 4.1 / 4.2 % | 2.2 / 5.9 % | 0.81 / 0.40 % (xz) | +1.8e-4 / −1.1e-3 |
| Ag 2.0 µm, L/3, 4 µm | 100 nm vs 400 nm | 3.34 % | 3.9 % | 9.3 % | 1.20 / 0.90 % | −1.5e-3 |
| Si 1.0 µm, L/3, 5.65 µm | 400 nm no band vs 100 nm no band | 1.97 % | 4.5 % | 2.1 % | 0.77 / 0.65 % | +2.7e-4 |
| Si 1.0 µm, L/3, 5.65 µm | fine band (400 nm below) vs uniform | **0.37 %** | 0.70 % | 0.23 % | 0.13 / 0.14 % | −3.0e-4 |
| Si 1.0 µm, L/3, 5.65 µm | 400 nm no band vs uniform | 0.98 % | 1.4 % | 2.2 % | 0.45 / 0.47 % | −2.2e-4 |
| Si 1.0 µm, L/3, 5.65 µm | 100 nm no band vs uniform | 1.57 % | 3.8 % | 0.21 % | 0.55 / 0.39 % | −4.8e-4 |

PARTS (Ag L = 1.5 µm, share of the reflected power radiated by each part alone):

| Box | top | walls | bottom | rim | foot |
|---|---:|---:|---:|---:|---:|
| uniform | 0.997 | 8.0e-4 | 1.9e-4 | 7.4e-5 | 4.9e-6 |
| 100 nm | 1.017 | 1.4e-3 | 4.4e-4 | 5.1e-5 | 3.2e-5 |
| 200 nm | 0.998 | 3.1e-3 | 1.3e-3 | 1.5e-4 | 2.7e-4 |
| 400 nm | 0.977 | 2.6e-3 | 1.9e-3 | 9.8e-5 | 1.0e-3 |

The coarse cells radiate: 400 nm cells (λ/1.25 in R1) raise the bottom contribution 10× and the
foot (wall–bottom edge) 200× above the uniform box. The shadow-side total field is small, so the
exact bottom currents are small, but the coarse bottom cannot represent the cancellation of the
incident beam (|E_inc| up to 0.8 V/m at the bottom centre, Ag L = 2 µm) and its error radiates
into the reflection hemisphere. 200 nm is not better than 400 nm. 100 nm (λ/5) is much better
but not converged: 0.17 % (ε_rr < 0.07 %) at L = 1.5 µm, 0.59 % (ε_rr 0.31 %) at L = 1 µm and
1.54 % (ε_rr 0.85 %) at L = 1 µm with depth ×2. Its error grows with the depth (longer coarse walls)
and is larger at the smaller patch (both w₀ = L/3), whose narrower beam diverges faster onto the
walls. The grading is discrete
(cells 2^M h): with h = 50 nm the next finer option after M = 1 (100 nm) is M = 0, the uniform
box (`--box-mesh-size 70e-9` already gives M = 0), so no intermediate point between 100 nm and
uniform exists to show convergence. The grading error is in the far field, not in the
reflectance (dP ≤ 1.5e-3).

For Si the uniform box (2N = 59 040, the largest Si run) separates the two parts of the box: the
fine band with 400 nm cells below it is within 0.37 % (ε_rr 0.14 %) of the uniform box, while both
no-band boxes are 1.0 % (400 nm) and 1.6 % (100 nm) off — for Si the walls right under the rim
(field not decayed, |k₂| h ≫ 1) matter more than the coarse cells deeper down, whose shadow-side
bottom radiates nothing (PARTS ≤ 4e-6, the Si interior absorbs the transmitted beam over 5.65 µm).
A fine band combined with 100 nm cells below (2N ≈ 4·10⁴ at L = 1 µm) was not run.

## 3. Si fine band vs none

| L, w₀ | none vs fine band | ε_rr xz / yz | dP | 2N none / fine |
|---|---:|---:|---:|---|
| 1.0 µm, 0.333 µm | 0.88 % | 0.41 / 0.43 % | +8e-5 | 4 650 / 38 082 |
| 1.2 µm, 0.333 µm | 0.44 % | 0.22 / 0.19 % | −1.8e-5 | 5 850 / 45 522 |
| 1.5 µm, 0.5 µm (largest feasible) | 0.66 % | 0.16 / 0.18 % | +2.9e-4 | 8 544 / 59 520 |

Without the fine band the Si walls coarsen to 400 nm from ~100 nm below the rim, where the
transmitted field is still e^−0.1 of its surface value and |k₂| h ≈ 30 (WP15, WP-P2). The fine band
also cuts the depth sensitivity 4× (0.52 % → 0.13 %, section 1). A 100 nm no-band box is not a
substitute (1.8 % from the fine-band result; it does not resolve the Si wavelength of 116 nm
either).

## 4. Edge effect: L ×1.2 at fixed waist (docs/05: < 0.5 %)

| Material, L → 1.2 L, w₀ | Box | l2_refl | l2_off30 | ε_rr xz / yz | dP | Verdict |
|---|---|---:|---:|---:|---:|---|
| Ag 2.0 → 2.4 µm, 0.667 µm (L/3) | 400 nm | 5.95 % | 8.1 % | 2.15 / 1.39 % | +6.2e-3 | fail |
| Ag 2.0 → 2.4 µm, 0.667 µm (L/3) | 100 nm | 5.98 % | 8.3 % | 2.09 / 1.59 % | +7.1e-3 | fail |
| Ag 2.0 → 2.4 µm, 0.5 µm (L/4) | 400 nm | 3.94 % | 4.4 % | 1.36 / 1.82 % | +4.2e-4 | fail |
| Ag, Im ε = −6 (diagnostic), L/3 | 400 nm | 4.53 % | 6.6 % | 1.69 / 1.13 % | +6.4e-3 | fail |
| Si 1.0 → 1.2 µm, 0.333 µm (L/3) | 400 nm, no band | 5.36 % | 8.9 % | 2.43 / 1.92 % | +6.4e-3 | fail |
| Si 1.0 → 1.2 µm, 0.333 µm (L/3) | fine band | 5.10 % | 9.2 % | 2.25 / 1.67 % | +6.5e-3 | fail |

The edge effect is physical: it does not depend on the grading (5.95 vs 5.98 %) or the fine band.
Power of the incident beam that crosses z = 0 outside the L × L patch (S_z integrated over the
plane, `--beam spectrum` as in the study; the paraxial value 1 − erf(√2 L/(2 w₀))² in brackets):

| w₀ | L (w₀ = L/…) | outside L × L | outside 1.2 L × 1.2 L |
|---|---|---:|---:|
| 0.333 µm | 1.0 µm (L/3) | 0.89 % (paraxial 0.539 %) | 0.13 % (0.064 %) |
| 0.5 µm | 1.5 µm (L/3) | 0.69 % (0.539 %) | — |
| 0.5 µm | 2.0 µm (L/4) | 0.019 % (0.013 %) | 0.0005 % (0.0003 %) |
| 0.667 µm | 2.0 µm (L/3) | 0.62 % (0.539 %) | 0.077 % (0.064 %) |

The rigorous beam carries more power past the edges than its paraxial counterpart because its
focal profile is not exactly Gaussian (see "Beam model"). Enlarging the patch moves 0.5–0.8 % of
the beam power from outside to inside it; the measured reflected power rises by 0.6–0.7 %
(dP, comparable), and the truncated aperture reshapes the specular lobe by 5–6 % (L2 of the
intensity). A power fraction is not the far-field L2 metric of docs/05, so the table does not
predict the 0.5 % check by itself; it shows that the patch edge is illuminated at the percent
level. w₀ = L/4 (0.019 % past the edges) still gives 3.9 %, possibly because the narrower beam diverges
faster (z_R = 1.6 µm) and the illumination along the walls grows; at feasible L the two cannot be
separated.

## Beam model (paraxial vs rigorous)

Ag L = 2 µm, w₀ = 0.667 µm, 400 nm box: the paraxial `GaussianBeam` changes the reflected power by
−2.8 % and the far field by 2.9 % (ε_rr 0.92 %) against the angular-spectrum beam of the same
waist. Its Maxwell residual (λ/(π w₀))² = 5.7 % is not a small effect at the waists a dense study
or a 3 w₀ ≤ L patch implies; the depth sensitivity is similar with both (1.8 vs 2.25 %).

"Waist" of the angular-spectrum beam means the scalar spectrum exp(−k_t² w₀²/4). After the
transverse projection e = x̂ − (k̂·x̂) k̂ the E_x spectrum is that Gaussian times (1 − k̂_x²), and
only propagating waves (k_t ≤ k) are kept, so the focal E_x profile is Gaussian only to
O(1/(k w₀)²). At w₀ = 0.333 µm (k w₀ = 4.2) the spectrum is still exp(−4.4) = 1.2 % of its peak
at the k_t = k cutoff, and the truncation matters: the profile has wider tails than
exp(−ρ²/w₀²), which is why the edge fractions of section 4 exceed the paraxial ones most at the
smallest waist (0.89 vs 0.54 %).

## Verdicts against docs/05

| Check | Ag | Si | Reason |
|---|---|---|---|
| Depth ×2 < 0.1 % | fail (0.8–3.5 %; 1.1 % in the uniform box) | fail, marginal with the fine band (0.13 %; ε_rr 0.04 %) | illuminated rim and walls (3× lower at w₀ = L/4), plus the depth-dependent grading error (100 nm: 0.59 → 1.54 % from the uniform box at L = 1 µm) |
| L ×1.2 < 0.5 % | fail (3.9–6.0 %) | fail (5.1–5.4 %) | at w₀ = L/3, 0.6–0.9 % of the rigorous beam power crosses z = 0 outside the patch; divergence along the walls |
| Grading (no tolerance in docs/05) | 400 nm: 1.5–1.9 %; 100 nm: 0.17 % (L = 1.5 µm), 0.59 % (L = 1 µm), 1.54 % (L = 1 µm, depth ×2) vs uniform | fine band + 400 nm: 0.37 %; no band: 1.0–1.6 % vs uniform | coarse cells radiate on the shadow side |

## Recommendations for ADR 0006 (proposed, not applied)

1. **Coarse spacing ≤ λ₁/5 is necessary; whether it suffices for the depth check is not shown.**
   Replace the automatic min(depth/2, L/8, 10 h) → 400 nm by `box_mesh_size` ≤ λ₁/5 (100 nm at
   500 nm, M = 1 for h = 50 nm): at fixed depth, 200–400 nm cells are 1.5–1.9 % from the uniform
   box (Ag), 100 nm cells 0.17 % (L = 1.5 µm) and 0.59 % (L = 1 µm). The decay-based contract of
   ADR 0006 covers only the object side; the exterior side of the walls and the bottom carries the
   incident beam (the bottom must cancel it) and needs the exterior resolution. λ₁/5 is not
   sufficient for the 0.1 % depth check at these sizes: the 100 nm grading error grows to 1.54 % at
   depth ×2 (L = 1 µm), and the depth ×2 change with 100 nm cells is 1.76 % (L = 1 µm; uniform box
   1.10 %) and 3.49 % (L = 2 µm; 400 nm 2.25 %). Remaining gaps: no grading between 100 nm and
   uniform exists for h = 50 nm (M is an integer), no uniform reference at L = 2 µm or depth ×2
   beyond L = 1 µm, and the behaviour at w₀ ≤ L/4 and MLFMM sizes (where the walls are lit less) is
   unknown — the depth check of a graded box must be repeated against a uniform box there before
   λ₁/5 is accepted as sufficient. Cost at L = 10 µm, depth 2 µm, h = 50 nm: bottom 20 000 + lower
   walls ~16 000 triangles ≈ 45 % of the top face (400 nm: 7 %, uniform: 180 %). A refinement that
   keeps λ₁/5 only within the beam footprint at the bottom (radius ~2 w(depth)) and lets the outer
   bottom coarsen is possible but untested.
2. **Fine band mandatory for Si** (weakly absorbing objects): 0.44–0.88 % far-field change without
   it at L = 1–1.5 µm (0.66 % at the largest feasible L = 1.5 µm, 2N = 59 520), and 4× the depth sensitivity. Its cost (box 1 000–1 500 % of the top face at
   L = 1–1.5 µm, 146 % at L = 10 µm) makes dense Si studies with the ADR box infeasible beyond
   L ≈ 1.5 µm.
3. **Waist/L ratio: w₀ ≤ L/4** (better L/5). At L/3, 0.6–0.9 % of the rigorous beam power crosses
   z = 0 outside the patch (paraxial 0.54 %), a percent-level illumination of the patch edge (a
   power fraction, not the far-field L2 metric of the 0.5 % check, but of the same order); at L/4
   it is 0.019 % (paraxial 0.013 %), at L/5 about 1e-6 (paraxial).
4. **Rim illumination (hypothesis):** the depth sensitivity follows the illumination of the patch
   edge, not the Rayleigh range. At L = 2 µm, w₀ = L/4 has the shorter Rayleigh range (1.57 vs
   2.79 µm) and a beam as wide or wider in the box, yet a 3× lower depth sensitivity (0.77 vs
   2.25 %) with an 18× weaker rim intensity. The depth checks are therefore expected to become
   meaningful when the rim is dark, i.e. at small w₀/L (w₀ ≤ L/4 or smaller); the w₀/L at which
   depth ×2 falls below 0.1 % is not known from two data points and must be measured (MLFMM, a
   w₀/L sweep at fixed w₀). An earlier criterion z_R ≥ 8 × depth (L ≳ 6–11 µm) was not derived
   and contradicts the w₀ = L/4 data; it is withdrawn.
5. **Rigorous beam before any quantitative rough-surface result:** the paraxial beam is 2.8 % off in
   reflectance at w₀ = 0.67 µm (the angular-spectrum construction of this study, exact and cheap at
   1 000–6 000 plane waves, could serve as the Phase 5 beam).
6. **docs/05 metric:** state the metric of the two sensitivity checks. The L2 difference of |F|²
   over the reflection hemisphere is ~3× the max-normalised ε_rr on the principal cuts (Si fine band
   depth ×2: 0.13 % vs 0.04 %); with ε_rr the Si fine-band box passes.
7. Minimum depth: no evidence for changing `default_box_depth` (Ag 2 µm, Si 5.65 µm); the Si
   bottom radiates nothing (PARTS ≤ 4e-6) and the Ag bottom only through the grading error.

## What could not be tested at feasible sizes

- The docs/05 checks themselves under realistic illumination (dark rim, w₀ ≤ L/4 or smaller, at
  waists where the beam model is near-paraxial): needs larger L than a dense operator allows, i.e.
  the MLFMM (Phase 4); how large is open (recommendation 4). The study executable runs unchanged
  once `SimulationConfig::compression = "mlfmm"` exists (one option to add).
- Whether the λ₁/5 grading error falls below 0.1 % when the rim is dark (recommendation 1):
  needs a graded-vs-uniform pair at w₀ ≤ L/4 with depth ×2.
- Graded vs uniform for Ag at depth ×2 beyond L = 1 µm (uniform depth-4 µm box at L = 2 µm:
  2N = 96 000) and for Si with the fine band at L ≥ 1.5 µm.
- A coarse spacing between 100 nm and uniform: for h = 50 nm the grading levels are 2^M h, so
  none exists (a finer top mesh would be needed to test the convergence of the grading error).
- Ag surface plasmons at L ≥ L_spp (21 µm intensity, 41.5 µm amplitude): the box is an SPP
  resonator in principle; the lossy-Ag diagnostic (section 1) attributes about half of the depth
  sensitivity at L = 2 µm to them, and their share at MLFMM sizes is open.
- Oblique incidence, s polarisation, other seeds and roughness parameters.

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_box_validity
$B --material si --L 1e-6 --fine-band --mesh-only                     # setup and 2N only
$B --material ag --L 2e-6 --L-gen 2.4e-6 --waist 6.6667e-7 --out ag2_base --summary s.txt
$B --material ag --L 2e-6 --L-gen 2.4e-6 --waist 6.6667e-7 --depth-factor 2 --out ag2_d2 \
   --reference ag2_base                                               # prints the COMPARE line
$B --material ag --L 2.4e-6 --L-gen 2.4e-6 --waist 6.6667e-7 --out ag2_L24   # L x 1.2
$B --material ag --L 1.5e-6 --L-gen 1.5e-6 --waist 5e-7 --box-mesh-size uniform --out ag15_uniform
$B --material si --L 1e-6 --L-gen 1.2e-6 --waist 3.3333e-7 --fine-band --out si1_fine
$B --material ag --L 1e-6 --L-gen 1.2e-6 --waist 3.3333e-7 --box-mesh-size 100e-9 \
   --depth-factor 2 --out ag1_b100_d2 --reference ag1_uni_d2 --reference ag1_b100
$B --compare ag15_uniform ag15_b100 ag15_base                        # stored runs, no solve
```

Options: `--depth-factor`/`--depth`, `--box-mesh-size 400e-9|200e-9|100e-9|uniform|auto`,
`--fine-band`, `--beam spectrum|paraxial`, `--eps-imag` (diagnostic loss override), `--tol`,
`--max-gb`, `--allow-unconverged`. Each process prints `SETUP`, `RESULT` and `PARTS` lines and
writes the far-field intensities, the cut amplitudes and the residual history as `.npy` (`--out`,
not committed). SETUP and RESULT print the object ε_r; runs with `--eps-imag` are marked
DIAGNOSTIC there, in their COMPARE lines and in the `diagnostic` attribute (the Im ε = −1.5 / −6
runs of this record predate the marking). A run whose GMRES did not converge writes nothing,
prints no COMPARE line and exits with status 1, and `--compare` refuses stored runs that did not
converge (attribute `converged`; older runs: true residual ≤ 1e-6), both unless
`--allow-unconverged` is given (COMPARE then marked UNCONVERGED). All runs of this record
converged (true residual ≤ 1e-6). About 9 h of wall time for all runs, one system at a time.
