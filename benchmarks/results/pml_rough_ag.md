# PML route on rough surfaces: validations 7c and 7d (WP-PML4b)

ADR 0009 decision 7, docs/05 "PML route", rows 7c and 7d. Study executable `benchmarks/pml_rough_ag.cpp`,
case code `tests/support/pml_rough_ag_support.hpp`, validation-large test
`tests/validation_large/test_pml_rough_ag_large.cpp`, raw ROW / COMPARE / DRC / DARK lines in
`pml_validation_rows.txt` (block "WP-PML4b"). Measured 2026-10-15 on a 12-core desktop with 128 GB RAM
(OpenBLAS LU). The machine was shared with two or three other dense studies during most runs, so the times
vary by up to 2.5× for the same size; the tables give the measured values. All comparisons are at equal
discretisation (h = λ/4 on every run); h = λ/4 is not mesh-converged for the rough-Si far field (7e: λ/6 moves
it by 2.0–2.3 %), and the rough-Ag mesh convergence was not measured in this WP (a finer grid h yields a
different height map from the generator, so a mesh study needs an interpolated map; open point).

## Summary

- **7c met on rough Ag.** σ_max ×1.5: 4.6·10⁻⁵, W_pml ×1.5: 2.5·10⁻⁵ (L2 metric; limit 1·10⁻³). The
  quadrature (target accuracy 1e-6, 12 stretched touching points) changes the far field by 3·10⁻⁸, the gap
  width (0.5 → 0.875 / 1.75 µm, same window) by ≤ 4.4·10⁻⁵.
- **7d: met on the rough Si control, fails on rough Ag (2.6 %).** The termination is not the cause (7c, the flat
  Ag control with 2·10⁻⁷, the rough Si control with the same map, zones and beam with 2.4·10⁻³, half the 0.5 %).
  The change comes from SPP currents in the gap (rms |J| / |M| there 10 % / 8.8 % of the patch level, against
  ≈ 10⁻¹⁰ on flat Ag): the far-field window truncates them (1.61 % on the same currents) and the added annulus
  changes them (2.42 % at the same absolute window, which still includes the window acting on the changed
  currents; the two are not separated). ADR 0010 / WP-PML6 (row 7g) decides whether a physical L dependence
  remains. A more lossy Ag (Im ε = −6, not Ag) reduces the change to 1.1 %. Across rough Ag, lossy Ag and rough
  Si the 7d change follows the current level that leaves the patch: l2 / (gap J level) = 0.26, 0.31, 0.32 (three
  points, an observation, not a law).
- **Dark band:** the 1e-4 default is not binding at w₀ = λ (ratio 3.3·10⁻⁶); it binds for w₀ ≳ 1 µm at gap
  0.5 µm. Relaxed to the 8.3·10⁻³ level (w₀ = 1.3 µm) it costs 5.2·10⁻⁴ overall (half the 7c tolerance) and
  1.8·10⁻³ off-specular (above 0.1 %). Recommendation: keep 1e-4.
- **W_pml = λ, σ_max 2:** 9.5·10⁻⁵ against the default PML at 0.76× the unknowns (0.57× the dense memory, 4.5× less
  assembly time), but the field at the outer PML edge is 4·10⁻³ (> 10⁻³) and its own 7c pair was not run;
  recommendation: keep W_pml = 2λ as the default.
- **Cost:** memory is not the limit (8–26 GB per run), the assembly time is (18–58 min per run at
  2N = 2.3–4.0·10⁴, ≈ 1.5–2.3 h for a 7c / 7d set). The touching-block cache (WP-PML2 review W4) was not started.
- **Validation-large:** the three ctest cases (7c Ag, 7d Si asserted, 7d Ag reported) passed, 16 181 s.

## Setup

| Item | Value |
|---|---|
| Surface | Gaussian height map σ = 50 nm, Lc = 500 nm, seed 1, grid h = λ/4 = 125 nm, generated on 4.5 + 2·0.5 µm and cropped centrally to L + 2 W_t, so the U and L ×1.2 runs share the surface on the common area (map rms 52.98 nm at L = 3.75 µm, 53.76 nm at 4.5 µm) |
| Zones | patch L = 3.75 µm (U) / 4.5 µm (L ×1.2), taper W_t = 0.5 µm (= Lc), gap W_g = 0.5 µm, PML W_pml = 1 µm = 2λ, σ_max 4, m 2 (the 7b defaults) |
| Media | vacuum over Ag (ε_r = −9.794 − 0.313j, 500 nm); 7d control: the same geometry and map with Si (ε_r = 18.478 − 0.606j) |
| Beam | rigorous angular-spectrum beam, normal incidence, E along x, w₀ = 0.5 µm = λ (= L/7.5), focus at the origin, 3 344 plane waves, region radius 1.02 √2 pml_start (4.147 µm for U), P_inc = 5.494·10⁻¹⁶ W (`AngularSpectrumBeam::power()`). The spectrum is cut at k_t = k where it is still exp(−(k w₀)²/4) = 5.2·10⁻⁵ of its peak (beam log); this cut, not the Gaussian profile, sets the dark band 3.3·10⁻⁶ (U) / 2.8·10⁻⁶ (L ×1.2) at pml_start = 5.75 w₀ (the Gaussian alone gives e^−(pml_start/w₀)² ≈ 5·10⁻¹⁵ there); tolerance 1e-4 |
| Solver | dense operator, ICTF (the automatic `formulation::recommend` choice for Ag and Si), kernel defaults (target accuracy 1e-5, 10 stretched touching points), LU in place (LAPACK), peak = one matrix 16 (2N)² bytes; residual on 64 sampled rows ≤ 7·10⁻¹¹ |
| Far field | `Simulation::far_field_options()` window (raised cosine from taper_end to pml_start) unless stated, Dunavant degree 6 for the source quadrature; reflection hemisphere `post::hemisphere_grid(−1, 90, 180)` (Gauss–Legendre in θ, 2° in φ, solid-angle weights, identical in every run), xz / yz cuts in 0.5° steps |
| Metric | docs/05 box-check metric l2 = sqrt(Σ w (I_T − I_R)² / Σ w I_R²) over the hemisphere (l2_off30: θ ≥ 30° only); ε_rr on the cuts; dR = R_T/R_R − 1. Quadrature note: 7c / 7d evaluate the sums on the Gauss–Legendre θ grid with solid-angle weights, whereas 7e and the box studies use the midpoint θ grid with sin θ weights (`box_validity::hemi_l2`): the same integral with a different quadrature; the cuts and ε_rr are identical |
| R | P_refl / P_inc with P_refl = (1/2η₁) Σ w \|F\|² on the hemisphere grid |
| Gap current level | area-weighted rms of \|J\| (and \|M\|) at the triangle centroids over the Gap zone relative to the same over the Patch zone (`zone_currents`) |

### Estimates (no assembly)

2N = 6M² − 4M with M = 2 · outer half width / h cells per axis. L had to put L/2 and 0.6 L on the 125 nm grid
(L ×1.2 without zone rounding: L/2 = 15 h, 0.6 L = 18 h), hence L = 3.75 µm (and 4.5 µm for 7d). Memory is not
what limits the size at h = λ/4: a 50 GB dense budget would allow 2N ≈ 5.6·10⁴.

| Case | 2N | dense (LU in place) | dark band | masked test functions (of N) |
|---|---:|---:|---:|---:|
| U (L 3.75 µm, gap 0.5 µm, W_pml 1 µm) | 22 816 | 8.33 GB | 3.26e-6 | 5 152 of 11 408 (45 %) |
| W_pml ×1.5 | 29 120 | 13.57 GB | 3.26e-6 | 8 304 of 14 560 |
| L ×1.2 (4.5 µm) / gap 0.875 µm | 27 472 | 12.08 GB | 2.80e-6 | 5 728 of 13 736 |
| W_pml = λ | 17 280 | 4.78 GB | 3.26e-6 | 2 384 of 8 640 |
| gap 1.75 µm | 40 016 | 25.62 GB | 2.06e-6 | 7 072 of 20 008 |

Dark-band scan of U (estimate only, `--scan-waists`, DARK lines in the rows file): at gap 0.5 µm
(pml_start 2.875 µm) the ratio is 8.2·10⁻⁷ at w₀ = 0.75 µm, 3.4·10⁻⁴ at 1 µm, 1.3·10⁻³ at 1.1 µm,
1.6·10⁻² at 1.4 µm; at w₀ = 1.3 µm it is 8.3·10⁻³ (gap 0.5 µm), 1.7·10⁻⁴ (1.5 µm), 5.2·10⁻⁵ (1.75 µm),
1.5·10⁻⁵ (2 µm).

**What limits the size: assembly time, not memory.** The rough runs at the default profile took 1 053–3 507 s of
assembly (18–58 min; 2N = 2.3–4.0·10⁴) and 8.45–25.8 GB peak; a 7c / 7d set (U, W_pml ×1.5, σ_max ×1.5, L ×1.2)
took 5 560 s of assembly in the study runs (1.5 h) and 8 450 s in the acceptance run of the test (2.3 h; the
Ag 7d case solves its own U again, another 1 554 s). The touching-block cache (WP-PML2 review W4) was not started in this WP;
these numbers are the input for that decision.

## Run table

Default window unless stated. "gap J / M": gap current level. Times: assembly (operator + right-hand
side), LU; peak RSS.

| Run | Change vs U | 2N | Peak | Assembly / LU | R | gap J / M | edge J / M |
|---|---|---:|---:|---|---:|---|---|
| u | — | 22 816 | 8.45 GB | 1 143 / 139 s (acceptance runs: 1 653 / 223 s, 1 554 / 297 s) | 0.9783391 | 0.0999 / 0.0877 | 5.7e-6 / 1.6e-5 |
| s15 | σ_max 6 | 22 816 | 8.45 GB | 1 221 / 163 s (acceptance run: 1 763 / 282 s) | 0.9783393 | 0.0999 / 0.0877 | 7.9e-7 / 9.6e-6 |
| w15 | W_pml 1.5 µm | 29 120 | 13.72 GB | 1 920 / 240 s (acceptance run: 3 164 / 783 s) | 0.9783386 | 0.0999 / 0.0877 | 2.2e-8 / 1.6e-7 |
| quad | target accuracy 1e-6, 12 stretched touching points | 22 816 | 8.45 GB | 1 108 / 92 s | 0.9783391 | 0.0999 / 0.0877 | 5.7e-6 / 1.6e-5 |
| wl | W_pml 0.5 µm = λ, σ_max 2 | 17 280 | 4.88 GB | 246 / 33 s | 0.9783389 | 0.0999 / 0.0878 | **4.0e-3 / 4.8e-3** |
| l12 | L 4.5 µm (window 2.75–3.25 µm) | 27 472 | 12.22 GB | 1 276 / 366 s (acceptance run: 1 870 / 317 s) | 0.9785148 | 0.105 / 0.098 | 5.8e-6 / 1.5e-5 |
| ugap | gap 0.875 µm (window 2.375–3.25 µm; extra windows 2.375–2.875, 2.75–3.25) | 27 472 | 12.22 GB | 1 053 / 146 s | 0.9780616 | — | 5.8e-6 / 1.4e-5 |
| uwide | gap 1.75 µm (window 2.375–4.125 µm; extra windows below) | 40 016 | 25.82 GB | 2 754 / 1 458 s | 0.9775432 | 0.089 / 0.079 | 5.9e-6 / 1.3e-5 |
| flat_u | σ = 0 (flat Ag, same grid and zones) | 22 816 | 8.45 GB | 962 / 90 s | 0.9816291 | — | 2.6e-9 / 9.5e-10 |
| flat_l12 | σ = 0, L 4.5 µm | 27 472 | 12.22 GB | 899 / 150 s | 0.9816291 | — | 3.0e-9 / 1.1e-9 |
| lossy_u | Im ε_r −6 (diagnostic) | 22 816 | 8.45 GB | 1 977 / 367 s | 0.7404424 | 0.036 / 0.036 | 1.4e-6 / 2.7e-6 |
| lossy_l12 | Im ε_r −6, L 4.5 µm | 27 472 | 12.22 GB | 2 462 / 451 s | 0.7403862 | 0.034 / 0.037 | 1.2e-6 / 2.5e-6 |
| drel | w₀ 1.3 µm, dark-band tolerance 1e-2 (ratio 8.3e-3) | 22 816 | 8.45 GB | 1 805 / 298 s | 0.9783300 | 0.103 / 0.091 | 5.8e-5 / 1.2e-4 |
| drel_s15 | drel with σ_max 6 | 22 816 | 8.45 GB | 1 979 / 409 s | 0.9783305 | 0.103 / 0.091 | 1.2e-5 / 5.9e-5 |
| dstr | w₀ 1.3 µm, gap 1.75 µm (ratio 5.2e-5, default tolerance; window 2.375–4.125 µm, extra window 2.375–2.875) | 40 016 | 25.82 GB | 3 507 / 1 249 s | 0.9775978 | 0.090 / 0.079 | 4.4e-5 / 8.8e-5 |
| si_u | Si (7d control, acceptance run) | 22 816 | 8.45 GB | 1 527 / 389 s | 0.3871358 | 0.0076 / 0.0078 | 1.8e-7 / 3.5e-7 |
| si_l12 | Si, L 4.5 µm (acceptance run) | 27 472 | 12.22 GB | 2 079 / 242 s | 0.3871315 | 0.0074 / 0.0078 | 1.9e-7 / 3.1e-7 |

The flat runs (rows recorded before the gap current level was added) carry only the reflected beam tail in
the gap (incident amplitude there ≈ e^−(2.375/0.5)² ≈ 10⁻¹⁰). The s15 / w15 gap levels come from the
acceptance runs (the study runs predate the column); the peak of an acceptance run is the process peak
(s15 ran after w15 in the same process). The Si runs exist only as acceptance runs (validation-large test,
ICTF, residual 3.6·10⁻¹¹ / 1.6·10⁻¹²). The far field itself costs 1.6–13.5 s per run (51.9 s for uwide with five
windows).

## 7c: PML parameters (met)

| Comparison | l2 | l2_off30 | ε_rr xz / yz | dR |
|---|---:|---:|---:|---:|
| σ_max ×1.5 (s15 vs u) | **4.55e-5** | 8.1e-5 | 1.6e-5 / 8.3e-6 | +2.0e-7 |
| W_pml ×1.5 (w15 vs u) | **2.48e-5** | 4.7e-5 | 9.0e-6 / 4.5e-6 | −5.7e-7 |
| quadrature (quad vs u) | 3.0e-8 | 8.5e-8 | 7.5e-9 / 6.9e-9 | −6.4e-9 |
| gap 0.875 µm, same window 2.375–2.875 (ugap.w1 vs u) | 4.4e-5 | 7.6e-5 | 1.6e-5 / 8.8e-6 | −9.4e-7 |
| gap 1.75 µm, same window 2.375–2.875 (uwide.w1 vs u) | 4.3e-5 | 8.3e-5 | 1.7e-5 / 6.5e-6 | −9.8e-7 |
| gap 1.75 vs 0.875 µm, same outer window 2.75–3.25 (uwide.w2 vs ugap.w2) | 3.9e-5 | 6.1e-5 | 1.6e-5 / 5.3e-6 | −1.6e-8 |
| W_pml = λ, σ_max 2 (wl vs u) | 9.5e-5 | 1.8e-4 | 3.4e-5 / 1.6e-5 | −2.3e-7 |
| relaxed dark band: σ_max ×1.5 at w₀ 1.3 µm, ratio 8.3·10⁻³ (drel_s15 vs drel) | 3.7e-5 | 1.2e-4 | 5.5e-6 / 2.9e-6 | +5.7e-7 |

Both 7c changes are 20–40× below the 0.1 % tolerance. The SPP currents that reach the PML (gap level 10 %)
are absorbed: the field at the outer PML edge is ≤ 1.6·10⁻⁵ of the patch level, and it falls further with both
parameters (J: 5.7·10⁻⁶ → 2.2·10⁻⁸ for W_pml ×1.5, → 7.9·10⁻⁷ for σ_max ×1.5), i.e. what reaches the outer
edge is already negligible at the defaults. The PML parameters change the far field by ~10⁻⁵ (7b measured
|r_spp| = 8·10⁻⁵ at the defaults on flat Ag), concentrated off-specular (l2_off30 ≈ 2× l2). The quadrature is
inert at the 10⁻⁸ level (at ≈ 1.0× the assembly cost: 1 108 vs 1 143 s), and the PML start position at the
4·10⁻⁵ level when the window is kept, at the inner (2.375–2.875 µm) as at the outer (2.75–3.25 µm) window.

## 7d: L ×1.2 at fixed waist

| Comparison | l2 | l2_off30 | ε_rr xz / yz | dR | Verdict |
|---|---:|---:|---:|---:|---|
| **Si: si_l12 vs si_u** | **2.41e-3** | 5.0e-3 | 9.5e-4 / 6.7e-4 | −1.12e-5 | **pass** (0.48× the 0.5 %) |
| **Ag: l12 vs u** | **2.57e-2** | 6.8e-2 | 9.4e-3 / 6.7e-3 | +1.80e-4 | fail (5× the 0.5 %) |
| flat Ag: flat_l12 vs flat_u | 2.2e-7 | 6.9e-6 | 4.2e-8 / 4.9e-8 | +7.7e-12 | pass |
| Ag, Im ε_r −6: lossy_l12 vs lossy_u | 1.09e-2 | 2.7e-2 | 4.0e-3 / 3.2e-3 | −7.6e-5 | fail (diagnostic) |

The validation-large test asserts 7d on the Si control and reports the Ag value with WARN, plus a regression
guard on the documented Ag value (lead decision 2026-10-18 (ADR 0010); the 0.5 % tolerance is not loosened).

### Why rough Ag fails: analysis

**The termination is not the cause.** The flat control shares grid, zones, PML, beam, window and the L ×1.2 move
of the window and passes with 2·10⁻⁷, so the window acting on currents without a bound wave is inert. 7c shows
the PML inert at 10⁻⁵, and the Si control (no surface plasmons, same rough map, zones, beam and window move)
changes by 2.4·10⁻³ and passes. Its gap current level is 0.0076 (J) / 0.0078 (M) of the patch level, 13× below
Ag's: on Si only the scattered field of the rough spot that runs along the surface near grazing reaches the
annulus and the window. The Si change is the non-SPP part of the finite-patch L dependence; its split into
annulus and window parts was not measured (no Si run with a fixed window). It is of the size the box route shows
for rough Si (L ×1.2 at w₀ = L/6: 0.19 % / 0.23 % on seeds 1 / 2, `box_validity_mlfmm.md`; the 7e box partners
L ×1.22 / ×1.17: 0.155 / 0.161 %, `pml_si_crosscheck.md`), at a smaller waist (L/7.5) and a darker rim. Not the
beam either: the Gaussian amplitude at the patch edge is e^−(L/2w₀)² = 8·10⁻⁷ of its peak (geometric estimate;
not measured at the patch edge itself), and the ratio measured at the PML start, 3.3·10⁻⁶, is the floor set by
the spectrum cut (Setup); not the quadrature (3·10⁻⁸).

**The Ag surface carries barely decaying surface currents outside the patch.** Under the rough patch the beam
(w₀ = λ ≈ Lc, so the spot sees about one correlation cell) launches surface plasmons. Their amplitude decay
length is 1/|Im k_spp| = 41.5 µm (11 L; k_spp = 1.0553 − 0.0019j k₀, flat-interface value), so they cross the
taper, the gap and the window region essentially undamped, with only the cylindrical spreading. The gap
current level is **0.100 (J) / 0.088 (M)** of the patch rms in U and still 0.089 / 0.079 averaged over the
1.75 µm wide gap, against ≈ 10⁻¹⁰ on the flat surface. The bound field decays above the surface with
κ = sqrt(k_spp² − k₀²) = 0.337 k₀: e^(−κz) = 0.12 at z = λ and 2·10⁻⁴ at z = 2 µm, consistent with the
Huygens-plane height z_H ≈ λ of ADR 0010 (bound field ≤ e⁻² at 0.94 λ), whose own sensitivity (z_H ×1.5) row 7g
measures. The PML absorbs them (7b, 7c); the far field is affected in two ways, which these runs do not
separate:

1. **The window truncates their currents.** On one solution (ugap or uwide), moving the 0.5 µm window from
   2.375–2.875 µm to 2.75–3.25 µm changes the far field by **1.61 %** (l2_off30 4.7 %, dR +2.8·10⁻⁶; gap 0.875
   and 1.75 µm identical to 4 digits). The 1 µm window shifted by 0.75 µm (3.125–4.125 vs 2.375–3.375 µm)
   changes it by 0.89 % (l2_off30 3.4 %), about half the 0.5 µm value at twice the width (and twice the
   shift): the window effect falls roughly like 1/W. Widening the window from the taper end to 3.375 µm
   (0.5 → 1 µm) changes it by 0.93 % (dR −3.7·10⁻⁴), to 4.125 µm (1 → 1.75 µm) by another 0.47 %
   (dR −4.4·10⁻⁴), and 2.375–3.25 µm against the default window (0.5 → 0.875 µm, ugap vs u, PML start inert)
   by 0.86 %; the shifted wide window 3.125–4.125 vs 2.375–4.125 µm by 0.47 %. A truncated bound wave
   radiates: Re k_spp − k₀ sin θ ≈ 0.055–1.06 k₀ is not resolved by a raised cosine of one or a few
   wavelengths. If the ≈ 1/W trend holds (an extrapolation of three widths), the 0.1 % level would need a window
   of ≈ 10 µm, a ring larger than the patch, which is not affordable densely. The window changes sit off-specular
   (l2_off30 3–4× l2), and R falls with the window width: R(0.5 / 0.875 / 1 / 1.75 µm) = 0.97834 / 0.97806 /
   0.97798 / 0.97754, i.e. 3–8·10⁻⁴. On the flat surface the same window move changes nothing (2·10⁻⁷), and the
   same window on different gap widths agrees to 4·10⁻⁵ (7c table), so this is not a PML effect either.
2. **The added rough annulus changes them.** With the same absolute window (2.75–3.25 µm) and PML start, the
   patch of 4.5 µm against the patch of 3.75 µm (l12 vs ugap.w2) changes the far field by **2.42 %** (l2_off30
   3.9 %), and R rises by **+1.8·10⁻⁴**. The difference sits in θ = 10–45° (65 % of the squared metric), i.e. in
   the broad reflected lobe of the rough spot, not near grazing. This comparison still includes the window
   acting on the changed SPP currents: one reading is that the extra roughness converts SPP power, which the
   smaller patch lets escape into the PML, into light (physical radiation of the larger spot, which the model
   contains); the other is the same window truncating a changed bound-wave amplitude. These runs do not separate
   the two; ADR 0010 / WP-PML6 (row 7g: 7d with a Huygens-plane far field) decides whether a physical
   L dependence remains.

In U → L ×1.2 both act together (the default window moves outward with the patch): 2.57 %, with l12 vs ugap.w2
(same absolute window: the annulus plus the window acting on the changed currents) 2.42 % and ugap.w2 vs ugap.w1
(same currents, window moved) 1.61 %, which do not add in quadrature (the far-field changes are partly
correlated).

**Loss diagnostic.** Im ε_r = −6 (19× the tabulated −0.313; 10× gives an intensity decay length of
2.3 µm > L/2, so the stronger value was used, as the box study did) shortens the SPP amplitude decay length to
3.1 µm and the intensity decay length to 1.56 µm (0.42 L); larger losses do not shorten it further (the SPP
becomes weakly bound, 2.9 µm at −8 … −10). The gap current level falls 2.8× (0.036 / 0.036), the 7d change
2.4× to **1.09 %** (l2_off30 2.7 %), and dR changes sign (−7.6·10⁻⁵). R falls from 0.978 to 0.740 (lossy_u
vs u: the far field differs by 24 %), so the diagnostic material is not Ag and this is a mechanism check
only. A shorter SPP range thus halves the L ×1.2 change but does not remove it: 1.56 µm is still comparable
with the distance from the beam to the annulus (1.9–2.25 µm) and to the window (2.4–3.25 µm), and the gap
level (0.036) stays 5× above the Si control's.

Across the three materials the 7d change follows the current level that leaves the patch:
l2 / (gap J level of the U run) = 2.57·10⁻² / 0.0999 = 0.26 (Ag), 1.09·10⁻² / 0.0355 = 0.31 (lossy Ag),
2.41·10⁻³ / 0.0076 = 0.32 (Si). The L ×1.2 change therefore tracks how much surface current reaches the
annulus and the window, whatever carries it (SPPs on Ag, the near-grazing scattered field on Si). Three
points make this an observation, not a law.

**What remains open** (not measured, no new solves in this WP):
- whether a physical L dependence remains once the window no longer truncates a bound wave (ADR 0010 /
  WP-PML6, row 7g); if 7g fails, the L dependence is physical and the model statement of ADR 0009 is the result;
- the split of the Ag annulus comparison (2.42 %) into the annulus itself and the window acting on the changed
  currents, and the split of the lossy (and the Si) change into annulus and window parts (no run with a fixed
  window);
- the sign of dR: +1.8·10⁻⁴ on Ag, −7.6·10⁻⁵ on lossy Ag, −1.1·10⁻⁵ on Si. A plausible reading is that the
  rough annulus of the lossy metal absorbs more SPP power than it radiates; the absorbed power was not evaluated;
- whether the lossy remainder is SPP-borne or the non-SPP part that Si shows; the similar l2 / gap-level
  ratio suggests the same per-current coupling, but does not establish it;
- the coupling constant ≈ 0.3 itself, which no model here predicts.

**Comparison with the box route.** The closed-box route failed L ×1.2 on rough Ag at every waist tried
(`box_validity_mlfmm.md`, section 3; σ = 50 nm, L = 4 → 4.8 µm): 4.50 % at w₀ = L/4 and 8.19 % at L/8
(graded 100 nm box), and 2.06 % / 3.52 % with the same Im ε = −6 diagnostic; WP-V3
(`ag_box_mechanism.md`) attributed the box sensitivity to surface waves on the finite block (most likely
SPPs) and left the annulus radiation as a hypothesis, unseparated from a rim artefact. Si passed the
box L ×1.2 at w₀ = L/6 (0.19 %, σ = 50 nm; 0.23 % on seed 2) but failed at σ = 100 nm (1.19 %). The PML-route
Ag L ×1.2 change is 1.7–3.2× smaller than the box's (2.6 % at w₀ = L/7.5 vs 4.5–8.2 %); caveat: the two compare
different L (3.75 vs 4 µm), w₀, h (125 vs 100 / 50 nm) and solver (dense LU vs MLFMM + GMRES). The PML route
shows that the termination is not the cause (7c, flat and Si controls); the remaining change comes from the SPP
currents in the gap as described above, and ADR 0010 / WP-PML6 (row 7g) decides whether a physical
L dependence remains.

## Dark band (decision 4 configuration check)

At the 7c / 7d geometry (w₀ = λ, pml_start = 5.75 w₀) the ratio is 3.3·10⁻⁶, 30× below the 1e-4 default:
the default costs nothing here. That level is the floor of the spectrum cut of the rigorous beam
(exp(−(k w₀)²/4) = 5.2·10⁻⁵ at the cut), not the Gaussian profile. With taper + gap = 2λ and the
7d-motivated w₀ ≤ L/5, pml_start / w₀ = L/(2w₀) + 2λ/w₀ ≥ 2.5 + 2λ/w₀, which is ≥ 3.5 for every w₀ ≤ 1 µm at
normal incidence. The default becomes binding only for wide beams at a narrow gap (w₀ ≳ 1 µm at gap 0.5 µm,
scan above) and relaxes with a wider gap (w₀ = 1.3 µm passes 1e-4 at gap 1.75 µm, 2N 40 016).

The rough-Ag pair at w₀ = 1.3 µm (= L/2.9; the waist warning of `Simulation` fires for both runs, as
pml_start 2.875 / 4.125 µm < 3.5 w₀ = 4.55 µm) measures what the level costs:

| Comparison | l2 | l2_off30 | ε_rr xz / yz | dR |
|---|---:|---:|---:|---:|
| **relaxed vs strict, same window 2.375–2.875 µm** (the masking alone; drel vs dstr.w1) | **5.21e-4** | 1.82e-3 | 1.4e-4 / 5.4e-5 | −1.3e-6 |
| relaxed vs strict, each with its default window (2.375–2.875 vs 2.375–4.125 µm) | 1.39e-2 | 3.17e-2 | 5.3e-3 / 5.4e-3 | +7.5e-4 |
| strict: default window vs 2.375–2.875 µm (width 1.75 vs 0.5 µm) | 1.39e-2 | 3.18e-2 | 5.5e-3 / 5.6e-3 | −7.5e-4 |
| relaxed: σ_max ×1.5 (7c at the 8.3·10⁻³ level; drel_s15 vs drel) | 3.67e-5 | 1.25e-4 | 5.5e-6 / 2.9e-6 | +5.7e-7 |
| relaxed σ_max ×1.5 vs strict, same window | 5.06e-4 | 1.78e-3 | 1.3e-4 / 5.5e-5 | −7.5e-7 |

- **The masking at the 8.3·10⁻³ level changes the far field by 5.2·10⁻⁴ overall** (half the 7c tolerance; R by
  1·10⁻⁶, as in 7a) **and by 1.8·10⁻³ off-specular** (θ ≥ 30°, above 0.1 %), about 10× the gap-width effect at
  w₀ = λ (4·10⁻⁵). If it scaled linearly with the ratio (not measured), 1e-3 would give ≈ 6·10⁻⁵.
- **With the default windows the comparison is dominated by the window** (1.39 % from the width 0.5 → 1.75 µm),
  not by the dark band: on rough Ag the window question of 7d decides the far-field accuracy, the dark band does
  not.
- **The PML sensitivity does not grow at the relaxed level** (σ_max ×1.5: 3.7·10⁻⁵ against 4.6·10⁻⁵ for U), so
  7c holds at both levels. σ_max ×1.5 of the strict w₀ = 1.3 µm run was not run (2N 40 016, ≈ 1.3 h on the
  loaded machine), as the relaxed member already shows no growth.
- **Cost of the strict level for this beam:** 1.75× the unknowns, 3.1× the memory, 1.9× the assembly and 4.2× the
  LU time (shared machine), from the 1.25 µm wider gap that 1e-4 needs at w₀ = 1.3 µm.
- The far-field effect of the dark-band level at oblique incidence (where 7a found the 1e-4 level expensive,
  45°: 2.5× the surface) was not measured.

**Recommendation (coordinator decides): keep 1e-4 as the default.** It is satisfied at no extra cost for the
normal-incidence waist rule pml_start ≳ 3.5 w₀ with w₀ ≥ λ; 7a found R changed by ≤ 0.06 % at 5.9·10⁻³, so a
user who needs a wider beam can relax it per run to 1e-2, at a measured far-field cost of 5.2·10⁻⁴ overall (half
the 7c tolerance) and 1.8·10⁻³ off-specular (above 0.1 %) at the 8.3·10⁻³ level: enough for R, the specular lobe
and 7d's 0.5 %, marginal for 0.1 %-level comparisons. A default of 1e-3 would be a compromise supported only by
the linear-scaling assumption.

## W_pml = λ with σ_max 2 (the option 7b deferred)

On rough Ag U, W_pml = λ, σ_max 2 (the 7b optimum at that width: |r_spp| = 4.0·10⁻⁵ / 4.4·10⁻⁵ (J_x / M_y),
measured at h = λ/5) changes the far field by 9.5·10⁻⁵ (l2_off30 1.8·10⁻⁴, dR −2.3·10⁻⁷), within 7c's 0.1 % with
a 10× margin and 2–4× the default profile's own 7c changes, at 2N 17 280 instead of 22 816 (0.76× the unknowns,
0.57× the dense memory, 4.8 vs 8.3 GB). Its assembly took 246 s instead of 1 108–1 143 s (4.5×); the pair count
explains only 1.7× of that (5 832 vs 7 688 triangles), the rest is attributed to the cheaper stretched touching
quadrature of the thinner, weaker PML ring (σ_max 2 instead of 4; 7b: at fixed W the assembly time grows with
σ_max, and W = λ, σ_max 2 took 131 s against 1 050 s for the default at 0.64× the unknowns) and partly to
machine load (an interpretation, not profiled). The current at the outer PML edge is
4.0·10⁻³ / 4.8·10⁻³ of the patch level (default: 1.6·10⁻⁵), above the 10⁻³ edge-decay check of the validation
tests; what reaches the open edge returns attenuated by e^−A with A = 2 Re k_spp σ_max W_pml/(m + 1) = 8.8 (7b),
visible but small in the far field. 7b found the W = λ reflection σ- and h-dependent by > 10× (σ_max 4: 1.3·10⁻³ at
λ/5 vs 1·10⁻⁴ at λ/6). W = λ was measured here only at h = λ/4, and its own 7c pair (W ×1.5, σ ×1.5 about
W = λ) was not run. **Recommendation: keep W_pml = 2λ as the default;** W_pml = λ, σ_max 2 is an option for
far-field studies at the 1e-4 level when the memory or the assembly time matters, to be confirmed with its own
7c pair before production use. At L = 30 µm the PML ring is ≈ 4 W_pml / L of the surface, 6.7 % for W_pml = λ
against 13 % for 2λ, so the saving shrinks with L; the cost model belongs to the fast-method WP (ADR 0009
decision 6).

## First rough-Ag numbers (U, default window)

| Quantity | Value |
|---|---|
| R (U, default window) | **0.97834**; flat PML run with the same zones 0.98163; beam reflectance R_beam of the infinite flat interface for the same beam 0.98167 (7a, `pml_validation.md`), i.e. the flat PML run is −0.004 % off, as in 7a |
| R_rough / R_flat − 1 | **−0.34 %** (window-dependent by ±5·10⁻⁴, see 7d); the difference is the power the roughness couples into SPPs, which the PML absorbs, plus the extra absorption of the rough metal (interpretation, not separated) |
| Far field rough vs flat | l2 65 %, l2_off30 24 (2 400 %): the specular peak is 3.4× lower (DRC at 0°: 1.89 vs 6.49 sr⁻¹), the remainder is diffuse |
| dark band / masked | 3.26·10⁻⁶ / 5 152 of 11 408 test functions (45 %) |
| 2N / memory / times | 22 816 / 8.45 GB peak (in-place LU) / assembly 1 143 s, LU 139 s, far field 2.3 s (shared machine) |

DRC D = |F|² / (2 η₁ P_inc) in the plane of incidence (xz cut, E in the plane), θ < 0 on the −x side:

| θ | −60° | −40° | −20° | −10° | −5° | 0° | +5° | +10° | +20° | +40° | +60° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rough Ag (u) [1/sr] | 2.80e-3 | 3.27e-3 | 0.466 | 1.411 | 1.782 | 1.890 | 1.632 | 1.345 | 0.970 | 0.227 | 3.85e-2 |
| rough Ag, L 4.5 µm (l12) | 2.14e-3 | 3.57e-3 | 0.468 | 1.397 | 1.811 | 1.862 | 1.641 | 1.352 | 1.005 | 0.182 | 4.13e-2 |
| flat Ag (flat_u) | 1.48e-7 | 6.37e-4 | 0.502 | 3.367 | 5.504 | 6.493 | 5.504 | 3.367 | 0.502 | 6.37e-4 | 1.48e-7 |

The rough spot (σ = 50 nm, one realisation, a beam of one correlation length) spreads the reflected lobe
(peak 1.89 vs 6.49 /sr) and makes it asymmetric (+20°: 0.97, −20°: 0.47), with a diffuse floor of
2–4·10⁻³ /sr at 40–60° where flat Ag gives 10⁻⁷–10⁻³. At θ ≥ 40° the L ×1.2 run differs by 7–24 %, the
specular region by ≤ 2 %. These are single-realisation, single-L, single-window numbers; they are not ensemble
statistics.

Box route for comparison (**estimate** from recorded values, not a run of this WP): WP-V3 (`ag_box_mechanism.md`,
same σ, Lc and seed, but L = 4 µm, w₀ = 1 µm, h = 50 nm, MLFMM, the closed box) recorded P_refl = 1.539·10⁻¹²
(rough, `ag4_d20`) and 1.565·10⁻¹² (flat box, `ag4_flat_d20`), i.e. rough / flat − 1 = −1.6 % there against
−0.34 % here. Qualitatively consistent (roughness lowers R by a few 10⁻³ … 10⁻²); not comparable quantitatively
(other waist, illuminated area, mesh, and the box's own sensitivity).

## Conclusions and what a user must do

1. The PML termination itself is validated on rough Ag (7c, including the PML start position and the
   quadrature) and, on the rough Si control, for L ×1.2 (7d).
2. On rough Ag the windowed far field of a finite rough spot changes by 1–2.6 % in this metric with the spot size
   and with the far-field window (λ = 500 nm, σ = 50 nm, Lc = 500 nm, L ≈ 4 µm). The termination is not the cause
   (7c, flat control, Si control). The change comes from SPP currents in the gap: the window truncates them
   (1.61 % on the same currents) and the added annulus changes them (2.42 % at the same absolute window, which
   still includes the window acting on the changed currents; the two are not separated). ADR 0010 / WP-PML6
   (row 7g) decides whether a physical L dependence remains. Until then rough-Ag far fields of the PML route
   carry an uncertainty of ≈ 1–2.6 % in this metric (dominated by the off-specular part, l2_off30 3–4× l2);
   R itself is affected at ≈ 5·10⁻⁴, and R_rough / R_flat is window-dependent by ±5·10⁻⁴.
3. **State L, gap and window (inner / outer half width) with every rough-Ag PML result**, and compare
   rough-Ag results only at equal L, gap and window (lead decision 2026-10-18, ADR 0010 decision 1). Whether
   ensemble averages over seeds reduce the L dependence was not measured (one seed; on it dR > 0).
4. Next step: the SPP-aware far field of ADR 0010 (WP-PML6, after WP-PML5), a Huygens plane in R1 at z_H ≈ λ,
   where the bound field has decayed to e^(−κ z_H) ≈ 0.12 (κ = 0.337 k₀; 2·10⁻⁴ at 2 µm), validated by its own
   sensitivity (z_H ×1.5, window position ×(1 ± 0.3)) and by 7d repeated with it (row 7g). An extra idea, not
   implemented: a lossy ring (graded Im ε, or an impedance-matched SPP absorber) in the polished region between the
   taper end and the window, which would absorb the SPPs before the window and defines the modified problem "rough
   spot whose SPPs are absorbed at its rim".
5. **Assembly time limits the sizes**, not memory (18–58 min per run at 2N = 2.3–4.0·10⁴ on the shared machine,
   8–26 GB; a 7c / 7d set ≈ 1.5–2.3 h of assembly). The touching-block cache (WP-PML2 review W4) was not started;
   the numbers above are the input for that decision.
6. **Mesh:** h = λ/4 is the 7a rule for flat Ag; it is not shown to be converged for σ = 50 nm roughness (a finer
   grid gives another map from the generator; an interpolated map is needed). Open point.

## Validation-large test

`ctest --preset win-release -L validation-large -R "PML 7[cd]"` (three cases, RUN_SERIAL): 7c on rough Ag
(U, W_pml ×1.5, σ_max ×1.5; CHECK l2 < 1e-3), 7d on rough Si (CHECK l2 < 5e-3), 7d on rough Ag (WARN with
the documented 2.57 %). Each case also checks residual < 1e-8, the dark band and the edge decay < 1e-3.

Acceptance run 2026-10-15 19:13–23:43 (release preset, shared machine, test code of commit 62af436; ROW /
COMPARE lines in the rows file):

| Case | Runs | Measured | Assertions | Result | Time |
|---|---|---|---:|---|---:|
| 7c rough Ag, W_pml ×1.5 and σ_max ×1.5 | u, w15, s15 | l2 2.48e-5 (W), 4.55e-5 (σ) < 1e-3 | 14 | Passed | 7 888 s |
| 7d rough Ag (reported only) | u, l12 | l2 2.5675e-2, WARN "documented 2.5675 %, tolerance 0.5 %" | 8 | Passed | 4 048 s |
| 7d rough Si | si_u, si_l12 | l2 2.41e-3 < 5e-3 | 9 | Passed | 4 246 s |
| total | | | | 3 / 3 passed | 16 181 s |

The acceptance ROW values reproduce the study runs to all printed digits in R, P_refl, residual, edge levels and the COMPARE
metrics (only the times differ). The later commit 3e204ea changed only the check of
`extra_windows`, which none of the three cases uses, so the run was not repeated.

After the review (2026-10-18) the Ag 7d case gained a regression guard `CHECK(m.l2 == Approx(kDocumentedAgL2)
.epsilon(0.1))` on the documented 2.5675 % (a reported value, not an acceptance; the WARN stays). The acceptance
run's value, 2.56748 % (WARN line in the rows file), lies within it, so the case was not rerun for the guard.

## Reproduce

```bash
# build with -DSPECKLEBEM_BUILD_BENCHMARKS=ON; run with C:/msys64/ucrt64/bin first on PATH
B=build/win-release/benchmarks/specklebem_pml_rough_ag.exe
$B --case u   --out ff/u   --summary rows.txt          # also: --case w15 | s15 | l12 | si_u | si_l12
$B --tag quad --acc 1e-6 --stp 12 --out ff/quad --summary rows.txt
$B --tag wl   --pml 0.5e-6 --sigma 2 --out ff/wl --summary rows.txt
$B --tag ugap --gap 0.875e-6 --window 2.375e-6,2.875e-6 --window 2.75e-6,3.25e-6 --out ff/ugap --summary rows.txt
$B --tag uwide --gap 1.75e-6 --window 2.375e-6,2.875e-6 --window 2.75e-6,3.25e-6 \
   --window 2.375e-6,3.375e-6 --window 3.125e-6,4.125e-6 --out ff/uwide --summary rows.txt
$B --tag flat_u --rms 0 --out ff/flat_u --summary rows.txt      # flat_l12: --case l12 --tag flat_l12 --rms 0
$B --tag lossy_u --eps-imag -6 --out ff/lossy_u --summary rows.txt  # lossy_l12: --case l12 ...
$B --tag drel --w0 1.3e-6 --dark-band-tol 1e-2 --out ff/drel --summary rows.txt  # drel_s15: ... --sigma 6
$B --tag dstr --w0 1.3e-6 --gap 1.75e-6 --window 2.375e-6,2.875e-6 --out ff/dstr --summary rows.txt
# DARK lines (no assembly): waist scan at gap 0.5 um, then the gap scan at w0 = 1.3 um
$B --tag scan_gap05 --scan-waists 0.75e-6,1e-6,1.1e-6,1.2e-6,1.3e-6,1.4e-6 --summary rows.txt
$B --tag scan_gap150 --gap 1.5e-6  --scan-waists 1.3e-6 --summary rows.txt
$B --tag scan_gap175 --gap 1.75e-6 --scan-waists 1.3e-6 --summary rows.txt
$B --tag scan_gap200 --gap 2e-6    --scan-waists 1.3e-6 --summary rows.txt
$B --compare ff/l12.ff ff/u.ff --summary rows.txt
```

An extra `--window` must end at or before pml_start; typing the pml_start value itself can exceed the
computed pml_start by round-off and is refused by `post::far_field` after the assembly (one 1 668 s
assembly was lost that way); `run()` now applies the same strict check to the extra windows before the
assembly.
