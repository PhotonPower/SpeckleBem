# Validity of the closing box at MLFMM sizes (WP-V2, ADR 0006)

Follow-up of `box_validity.md` (WP-V1, dense, L ≤ 2.4 µm). Question: is the ADR 0006 closing box
inert at sizes where the patch rim is dark? The two docs/05 rough-surface checks, with the ADR 0006
amendment metric (L2 of |F|² over the reflection hemisphere, w = sin θ; ε_rr on the xz/yz cuts
alongside):

- **depth ×2: far field changes < 0.1 %;**
- **L ×1.2 at fixed waist: < 0.5 %.**

Executable `benchmarks/box_validity.cpp` (`specklebem_box_validity --compression mlfmm`), raw lines
in `box_validity_mlfmm_rows.txt`.

**Short answer.**

- **Si passes both checks once the rim is dark — with the uniform box (one seed, one roughness,
  one L ×1.2 pair).** Uniform (50 nm) box, fine band, w₀ = 0.75 µm: depth ×2 0.032 % at
  w₀/L = 1/4 and 0.007 % at 1/6 (noise floor 0.004 %); L ×1.2 1.28 % at 1/4 (fail) and **0.19 % at
  1/6 (pass)**. Scope: seed 1, σ = 50 nm, Lc = 500 nm, normal incidence, x polarisation; three
  depth pairs and a single L ×1.2 pair at 1/6 (4.5 → 5.4 µm); nothing between w₀/L = 1/4 and 1/6.
  The graded λ₁/5 box is 0.085 % from the uniform box and its depth ×2 change is 0.14 % at
  w₀/L = 1/4: **at w₀/L = 1/4, λ₁/5 grading does not suffice for Si and the uniform box is needed**
  (the graded box was not run at 1/6). The uniform box is also cheaper in memory and iterations at
  MLFMM sizes (Si at L = 3 µm: 17 vs 33 GB, 205 vs 322 iterations).
- **Ag fails both checks at every waist tried, and the dark rim does not help.** L = 4 µm,
  w₀/L = 1/4: depth ×2 0.91 % (graded λ₁/5 box) / **0.82 % (uniform box)**; w₀/L = 1/8 (rim
  intensity 10⁻¹² of the peak): depth ×2 1.89 %; L ×1.2 4.5 % (1/4) and 8.2 % (1/8), i.e. ≥ 8× the
  thresholds. The grading is not the cause (graded vs uniform 0.23–0.36 %; the depth sensitivity
  is the same in both boxes). **The cause is not identified.** The sensitivity is consistent with
  surface plasmons (SPPs) excited by the roughness, but it is not separated from the weakly damped
  cavity resonances of the closed box found by WP-A2 (vacuum inside, Ag walls;
  `ag_absorption.md`, section "WP-A2"; ADR 0006 amendment 2026-10-12), which the loss diagnostic
  damps too: with Im ε = −6 the uniform box gives depth ×2 **0.093 %** (9× lower, at the threshold
  within the noise), the wall radiation falls 9–20× and GMRES needs 251 instead of 1 305
  iterations (depth 4 µm, tol 1e-4), but that loss also raises Re Z_s/|Z_s| from ≈ 1.6 % to
  ≈ 27 % and changes the top face (P_refl −25 %). Section 2 ("Ag: SPPs or box cavity resonances?") gives the evidence on
  both sides and four discriminating experiments, none run. That the Ag box influence decays with
  L / L_spp rather than with w₀/L is a **hypothesis**: no Ag L dependence was measured (L = 4 and
  4.8 µm only, L/L_spp ≈ 0.2), and the lossy diagnostic contradicts its simple form (there
  L = 4 µm ≈ 2.6 L_spp, yet L ×1.2 still changes the far field by 2.06 % / 3.52 %).
- WP-V1's hypothesis (depth sensitivity follows the rim illumination **by the incident beam**)
  holds for Si and is **falsified for Ag**; illumination of the rim and walls by surface waves or
  grazing scattered fields is not excluded.

## Setup

| Parameter | Value |
|---|---|
| Wavelength, media | λ = 500 nm, vacuum R1; Si ε_r = 18.478 − 0.606j, Ag ε_r = −9.794 − 0.313j, exp(+jωt) |
| Surface | Gaussian height map, σ = 50 nm, Lc = 500 nm, h = 50 nm, seed 1, generated on 1.2 L and cropped centrally to L (`--L-gen`), so the base, depth ×2 and L ×1.2 runs of a family share the surface |
| Box | depth `default_box_depth` (Ag 2 µm, Si 5.65 µm) and ×2 (Ag 4 µm, Si 11.29 µm); coarse spacing λ₁/5 = 100 nm (graded, M = 1) or uniform (50 nm); Si always with the fine band 3δ + 3σ = 3.54 µm |
| Beam | rigorous angular-spectrum Gaussian beam of the WP-V1 study (`--beam spectrum`; the spectrum grid check on a probe of mesh vertices is printed in the SETUP line of each run log; the ≤ 1e-12 quoted from those logs is **not recorded** in the committed rows file, which holds only RESULT, PARTS and COMPARE lines; the executable aborts if the check exceeds 1e-8), normal incidence, focus at the origin, E along x. Ag: L = 4 µm, w₀ = 1 µm (L/4) and 0.5 µm (L/8), L ×1.2 = 4.8 µm at the same w₀. Si: w₀ = 0.75 µm, L = 3 µm (L/4) and 4.5 µm (L/6), L ×1.2 = 3.6 / 5.4 µm; one Si depth pair at w₀ = 1 µm, L = 4 µm (L/4) |
| Operator | MLFMM (`Simulation` compression "mlfmm"), d₀ = 3, global leaf rule, automatic exact-part budget; ICTF; Ag with left Jacobi (formulation::recommend), Si without |
| Solver | full GMRES (CGS2); Ag tol 1e-4 (sweep) / 1e-5 (references), Si tol 1e-5 (choice from the noise floor below) |
| Far field | `post::far_field`, Dunavant degree 10; reflection hemisphere θ = 0.5…89.5° (1°) × φ = 0…357° (3°), cuts in the xz/yz planes (0.5°). The metric functions are unit-tested (`tests/unit/test_box_validity.cpp`) |
| Machine | Windows 11, 24 threads (12 per run, two runs at a time, other agents' studies alongside), 128 GB, `win-release` + benchmarks, 2026-10-11/12; times indicative |

The geometry follows the brief's intent (a w₀/L sweep where the rim becomes dark) but not its
suggested sizes (w₀ = 1 µm, L = 4…8 µm for both materials), see "Deviations".

## 1. Noise floor (metric between numerical settings of one configuration)

| Case | Comparison | l2_refl | ε_rr xz / yz |
|---|---|---:|---:|
| Ag L = 4 µm, w₀ = 1 µm, 100 nm box, depth 2 µm (2N = 70 560) | tol 1e-4 vs 1e-5 (d₀ = 3) | 1.8e-4 | 2.3e-5 / 2.6e-5 |
| | tol 1e-3 vs 1e-5 | 1.3e-3 | 3.2e-4 / 5.8e-5 |
| | d₀ = 5 vs 3 (both tol 1e-4) | 1.9e-5 | 4.8e-6 / 4.5e-6 |
| Si L = 3 µm, w₀ = 0.75 µm, uniform box, depth 5.65 µm (2N = 205 920) | tol 1e-4 vs 1e-5 | 4.7e-4 | 2.6e-4 / 1.1e-4 |
| | tol 1e-5 vs 1e-6 | 3.7e-5 | 1.9e-5 / 1.7e-5 |
| | d₀ = 5 vs 3 (both tol 1e-5) | 1.7e-5 | 6.8e-6 / 3.4e-6 |

- The MLFMM accuracy is not limiting: d₀ = 5 changes the far field by 2e-5 (Ag and Si), 50× below
  0.1 %. **d₀ = 3** is used throughout (d₀ = 5 costs 1.6× (Ag) / 2× (Si) the solve time and 2–3×
  the memory: leaves of λ instead of λ/2).
- The solver tolerance dominates the floor. **Si: tol 1e-5** (floor 3.7e-5, 27× below 0.1 %); tol
  1e-4 (4.7e-4) would not be well below the tolerance, the brief's 1e-3 vs 1e-4 pair even less.
- **Ag: tol 1e-4** for the sweep (floor 1.8e-4, 5.5× below 0.1 %; every Ag verdict below is ≥ 0.8 %,
  ≥ 8× the threshold, so the floor never decides a verdict; the Im ε = −6 diagnostic at 0.093 % is
  within the floor of the threshold, see section 2), tol 1e-5 for the references (graded and
  uniform depth-2 µm boxes). Ag at tol 1e-5 needs 3 746 iterations (1 h) with the graded box at
  2N = 7·10⁴, so a sweep at 1e-5 did not fit the 2 h run limit beyond L ≈ 4 µm. ADR 0006's tol ≤
  1e-5 for quantitative Ag reflectance is not contradicted: that criterion concerns R, here the
  power changes by ≤ 3e-5 between tolerances. The mapping from tolerance to error is
  case-dependent (WP22c: about 0.5 % error of R at tol 1e-4 on flat Ag at normal incidence), so
  this single-configuration floor (one Ag case, tabulated ε) does not carry over to other Ag cases
  or to the lossy diagnostic.
- Pairs compare runs with the same settings; a pair's noise is up to ~√2 × the floor.

## 2. Depth ×2 (docs/05: < 0.1 %)

| Material, L, w₀ (w₀/L) | Box | 2N base / ×2 | It. base / ×2 | l2_refl | l2_off30 | ε_rr xz / yz | dP | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| Si 3 µm, 0.75 µm (1/4) | uniform | 205 920 / 368 640 | 205 / 216 | **0.032 %** | 0.053 % | 0.014 / 0.005 % | −4e-6 | **pass** |
| Si 3 µm, 0.75 µm (1/4) | 100 nm + band | 148 320 / 188 640 | 322 / 340 | 0.139 % | 0.28 % | 0.037 / 0.018 % | −2.3e-5 | fail |
| Si 4.5 µm, 0.75 µm (1/6) | uniform | 341 280 / 585 360 | 202 / 220 | **0.0072 %** | 0.032 % | 0.0016 / 0.0014 % | +5e-6 | **pass** |
| Si 4 µm, 1 µm (1/4) | uniform | 293 760 / 510 720 | 202 / 219 | **0.031 %** | 0.076 % | 0.0032 / 0.0015 % | −2e-6 | **pass** |
| Ag 4 µm, 1 µm (1/4) | 100 nm | 70 560 / 89 760 | 2 494 / 2 881 | 0.91 % | 1.48 % | 0.20 / 0.21 % | +2e-5 | fail |
| Ag 4 µm, 1 µm (1/4) | uniform | 153 600 / 230 400 | 2 096 (tol 1e-5) / 1 305 | 0.82 % | 1.42 % | 0.18 / 0.17 % | −4e-5 | fail |
| Ag 4 µm, 0.5 µm (1/8) | 100 nm | 70 560 / 89 760 | 2 640 / 2 921 | 1.89 % | 1.99 % | 0.82 / 1.25 % | +2.6e-4 | fail |
| Ag 4 µm, 1 µm (1/4), Im ε = −6 (diagnostic) | uniform | 153 600 / 230 400 | 235 / 251 | 0.093 % | 0.18 % | 0.010 / 0.008 % | −7e-6 | (diagnostic) |

PARTS (power radiated into the reflection hemisphere by the currents of one part alone, relative to
the total; partial fields interfere). The |E_inc| column comes from the SETUP lines of the run logs
(not committed; **not recorded** in the rows file, which holds only the RESULT, PARTS and COMPARE
lines); the PARTS values are in the rows file.

| Run | walls | bottom | rim | foot | |E_inc| at the rim (x, y) [V/m] |
|---|---:|---:|---:|---:|---|
| Si 3 µm, L/4, uniform, depth 5.65 (tol 1e-4 run `si3_d1_uni`) / 11.29 µm | 1.4e-5 / 1.4e-5 | 1.1e-6 / 1.2e-7 | 3.7e-6 / 3.6e-6 | 7e-9 / 4e-10 | 2.3e-2, 1.8e-2 |
| Si 4.5 µm, L/6, uniform, depth 5.65 / 11.29 µm | 8.2e-7 / 7.7e-7 | 3.1e-7 / 7.6e-8 | 2.9e-7 / 3.0e-7 | 9e-10 / 1e-10 | 2.0e-4, 1.2e-4 |
| Ag 4 µm, L/4, 100 nm, depth 2 / 4 µm | 1.1e-3 / 1.5e-3 | 4.7e-4 / 5.8e-4 | 5.3e-5 / 5.4e-5 | 2.8e-5 / 3.3e-5 | 2.1e-2, 1.8e-2 |
| Ag 4 µm, L/4, uniform, depth 2 / 4 µm | 1.6e-3 (tol 1e-3) / 7.2e-4 | 1.2e-3 / 2.0e-4 | 6.9e-5 / 6.4e-5 | 7e-6 / 3e-6 | 2.1e-2, 1.8e-2 |
| Ag 4 µm, L/8, 100 nm, depth 2 / 4 µm | 2.4e-3 / 3.2e-3 | 1.1e-3 / 1.2e-3 | 1.2e-4 / 1.1e-4 | 6.1e-5 / 6.4e-5 | 1.4e-6, 5.3e-6 |
| Ag 4 µm, L/4, uniform, Im ε = −6, depth 2 / 4 µm | 8.1e-5 / 7.8e-5 | 6.6e-6 / 3.0e-6 | 1.6e-5 / 1.6e-5 | 2e-7 / 4e-8 | 2.1e-2, 1.8e-2 |

*Si.* The box is inert once the rim is dark: from w₀/L = 1/4 to 1/6 the rim field falls 100×,
the wall radiation 17× and the depth sensitivity 4.5× (to twice the noise floor). The bottom does
not radiate (δ = 1.13 µm, PARTS ≤ 1e-6). The same w₀/L = 1/4 at two waists (0.75 µm at L = 3 µm, 1 µm at
L = 4 µm) gives the same depth sensitivity (0.032 / 0.031 %): for Si the ratio w₀/L controls it. The graded box fails the same check at 1/4 (0.14 %): its
depth change is 4× that of the uniform box, so the residual grading error (graded vs uniform
0.085 % at both depths) is itself depth dependent — λ₁/5 is not sufficient for the 0.1 % check.

*Ag.* The depth sensitivity is the same in the uniform and the graded box (0.82 vs 0.91 %), so it
is not a grading error. It does not follow the rim illumination by the incident beam: at
w₀/L = 1/8 the incident field at the rim is 10⁴× weaker than at 1/4, yet the depth sensitivity
doubles (1.89 %) and the walls radiate 2.4e-3 of the reflected power (1/4: 1.1e-3). The box
responds to fields that reach the walls without the beam (illumination by surface waves or grazing
scattered fields is not excluded by this test).

**Ag: SPPs or box cavity resonances?** Two candidate mechanisms; the data of this record do not
separate them.

- *Surface plasmons* excited by the roughness under the beam, running over the top face to the rim
  (amplitude decay 41.5 µm, hardly damped over L/2 = 2 µm), turning onto the walls and reflected at
  the foot and the bottom edges. A narrower beam has a wider angular spectrum and couples more of
  its power into SPPs through the roughness (smaller momentum mismatch k_spp − k_t), which would fit
  the larger relative wall radiation at w₀ = 0.5 µm.
- *Cavity resonances of the closed box* (WP-A2: `ag_absorption.md`, section "WP-A2"; ADR 0006
  amendment 2026-10-12). The complementary problem (vacuum inside the box, Ag outside) has weakly
  damped resonances, damped through Re Z_s, strongly dependent on the box depth D (flat Ag, 45° s,
  L = 3 µm: D = 2.0 → 2.1 µm moves the local absorption error from −2 % to −34 %) and tied to a band
  of 8–12 small eigenvalues of the Jacobi-preconditioned operator. A perfectly non-radiating
  resonant component cannot change the total far field, but WP-A2 checked R and A_ext only at the
  0.1–0.4 % level, which leaves room for a radiating residual of near-resonant modes at the 0.8 %
  level of the depth check. WP-A2's cases were flat and at 45° s; the roughness and the normal
  incidence of this record were not covered there.

What the evidence shows:

- *Im ε = −6 diagnostic.* It cuts the depth sensitivity from 0.82 % to 0.093 % (9×; WP-V1 at
  L = 2 µm, w₀ = L/3 found only 2× because the rim was lit there), the wall radiation 9–20× and the
  bottom radiation 70–180× (depth 4 / 2 µm). It shortens the SPP amplitude decay from 41.5 to
  3.1 µm, but it also raises Re Z_s/|Z_s| from ≈ 1.6 % to ≈ 27 %, which damps the cavity
  resonances by a similar order, and it changes the top face itself (P_refl −25 %). It is
  consistent with both mechanisms and separates neither. The 0.093 % is at the threshold within
  the noise: the Ag floor is 1.8e-4 per run at tol 1e-4, ≈ 2.5e-4 for a pair, and it was
  measured for the tabulated ε only, not for the lossy material.
- *PARTS.* The wall and bottom shares are partial far fields of the box currents. A non-radiating
  cavity component contaminates exactly these partial fields (its parts radiate, their sum does
  not), so the shares cannot attribute the sensitivity to SPPs. Their depth dependence looks
  resonance-like rather than SPP-like: uniform box, depth 2 → 4 µm, walls 1.6e-3 → 7.2e-4 and
  bottom 1.2e-3 → 2.0e-4 (the depth-2 µm uniform PARTS come from the tol 1e-3 run
  `ag4_d2_uni_t3`); graded box, same depths, walls 1.1e-3 → 1.5e-3 and bottom 4.7e-4 → 5.8e-4. The
  SPP loss over the extra 4 µm path down and up the walls is only e^(−4/41.5) ≈ 0.91 in amplitude;
  it cannot explain a 6× drop of the bottom share, nor opposite trends in two boxes of the same
  outer geometry.
- *GMRES.* Lossless Ag needs 1 305 iterations, Im ε = −6 251 (uniform box, depth 4 µm, tol 1e-4).
  This is consistent with SPPs and equally with WP-A2 E3, where the slow Ag convergence comes with
  a band of 8–12 small harmonic Ritz values (|θ| = 0.016–0.1) that WP-A2 attributes to the damped
  cavity resonances; extra loss would move such resonances away from the origin as well (not
  measured, see (iv)). The
  iteration counts do not separate the two mechanisms.

Discriminating follow-up experiments (proposed, **not run**; the run directories with the currents
are not committed, so (i) and (ii) need either those files or a re-solve):

- **(i)** Top-face current spectrum with WP-A2's `strip_spectrum`
  (`tests/support/ag_absorption_support.hpp`, with `subtract` off, since a rough surface has no
  analytic reference): SPPs sit at |k_t| ≈ 1.055 k₀ (> k₀), cavity modes at |k_t| < k₀ (WP-A2:
  0.70 k₀ at 45°). Compare the depth 2 and 4 µm runs.
- **(ii)** `extinction_residual` inside the box (a line above max ξ, WP-A2 placement rule) for the
  Ag depth 2 / 4 µm runs: a depth-independent vacuum standing wave between the top face and the
  bottom plate is the cavity signature (WP-A2 flat-surface levels: > 1 % strong, 0.7–1 % moderate).
- **(iii)** A fine depth sweep D = 2.0 / 2.1 / 2.2 µm at L = 4 µm, w₀/L = 1/4: resonant on/off
  switching of the depth sensitivity points to the cavity; a smooth dependence points to SPPs.
- **(iv)** Harmonic Ritz values (`GmresParams::keep_hessenberg`, `tests/support/krylov_spectrum.hpp`),
  lossless vs Im ε = −6: whether the small-eigenvalue band leaves the origin with the loss and
  tracks the iteration counts.

## 3. Edge effect: L ×1.2 at fixed waist (docs/05: < 0.5 %)

| Material, L → 1.2 L, w₀ (w₀/L) | Box | 2N | It. (1.2 L) | l2_refl | l2_off30 | ε_rr xz / yz | dP | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| Si 3 → 3.6 µm, 0.75 µm (1/4) | uniform | 205 920 → 257 472 | 202 | 1.28 % | 1.35 % | 0.51 / 0.44 % | +1.2e-4 | fail |
| Si 4.5 → 5.4 µm, 0.75 µm (1/6) | uniform | 341 280 → 432 864 | 199 | **0.19 %** | 0.64 % | 0.028 / 0.022 % | −2.5e-5 | **pass** |
| Ag 4 → 4.8 µm, 1 µm (1/4) | 100 nm | 70 560 → 95 040 | 2 680 | 4.50 % | 10.7 % | 0.73 / 0.60 % | −1.5e-3 | fail |
| Ag 4 → 4.8 µm, 0.5 µm (1/8) | 100 nm | 70 560 → 95 040 | 2 905 | 8.19 % | 12.9 % | 1.51 / 1.35 % | −2.7e-3 | fail |
| Ag 4 → 4.8 µm, 1 µm (1/4), Im ε = −6 (diagnostic) | uniform | 153 600 → 202 752 | 243 | 2.06 % | 4.57 % | 0.41 / 0.39 % | −9.7e-4 | (diagnostic) |
| Ag 4 → 4.8 µm, 0.5 µm (1/8), Im ε = −6 (diagnostic) | uniform | 153 600 → 202 752 | 251 | 3.52 % | 5.30 % | 0.70 / 0.60 % | −1.9e-3 | (diagnostic) |

*Si.* At w₀/L = 1/4 the patch edge is lit (|E_inc| 2e-2 at the rim) and enlarging the patch
changes the far field by 1.3 %; at 1/6 (rim 2e-4) the change is 0.19 %, below the 0.5 % tolerance
(its off-specular part, 0.64 % for θ ≥ 30°, is where the remaining edge contribution shows).

*Ag.* 4.5 % and 8.2 %, mostly off-specular (11–13 % for θ ≥ 30°), and larger at the smaller waist.
As for the depth, w₀/L does not control it. Two readings, not separated: moving the rim moves the
place where SPPs are scattered out, which reshapes the diffuse far field; or enlarging the box
shifts its cavity resonances (WP-A2: they recur every ~0.2–0.5 µm in L and D), which changes the
response of the whole box. The Im ε = −6 diagnostic halves the edge effect (2.1 % / 3.5 %) but does
not remove it, and it is again larger at the smaller waist. An SPP reading would explain the
remainder by plasmons that still reach the rim (2–2.4 µm away) with ~50 % of their amplitude at a
3.1 µm amplitude decay; but at that loss L_spp (intensity) ≈ 1.56 µm, so L = 4 µm ≈ 2.6 L_spp, and
the check still fails by 4–7× the threshold. This contradicts a simple "the box influence decays
with L / L_spp" picture, which therefore stays a hypothesis (no Ag L dependence was measured
beyond the 4 → 4.8 µm pairs). A lossier diagnostic would also change the top-face response more.

## 4. Graded (λ₁/5 = 100 nm) vs uniform (50 nm) box at w₀/L = 1/4

| Material, L, w₀, depth | l2_refl | l2_off30 | ε_rr xz / yz | dP | Cost graded / uniform (2N, it., peak) |
|---|---:|---:|---:|---:|---|
| Si 3 µm, 0.75 µm, 5.65 µm | 0.085 % | 0.11 % | 0.030 / 0.018 % | +2e-6 | 148 320 / 205 920; 322 / 205 it.; 33 / 17 GB |
| Si 3 µm, 0.75 µm, 11.29 µm | 0.084 % | 0.19 % | 0.021 / 0.008 % | −1.6e-5 | 188 640 / 368 640; 340 / 216 it.; 42 / 31 GB |
| Ag 4 µm, 1 µm, 2 µm (both tol 1e-5) | 0.23 % | 0.38 % | 0.053 / 0.047 % | +5e-5 | 70 560 / 153 600; 3 746 / 2 096 it.; 7 / 10 GB |
| Ag 4 µm, 1 µm, 4 µm (both tol 1e-4) | 0.36 % | 0.54 % | 0.055 / 0.072 % | +2e-5 | 89 760 / 230 400; 2 881 / 1 305 it.; 7 / 11 GB |

Depth ×2 within each box: Si graded 0.139 % vs uniform 0.032 %; Ag graded 0.91 % vs uniform 0.82 %.

**Verdict: at w₀/L = 1/4, λ₁/5 does not suffice; the uniform box is needed for the 0.1 % depth
check** (w₀/L = 1/6 and darker rims untested with the graded box: the grading error might fall with
the rim illumination). For Si the graded box fails the depth check that the uniform box passes. For
Ag the grading error (0.23–0.36 %, growing with the depth as in WP-V1) is a fraction of the Ag box
sensitivity, so it does not change the Ag verdict, but it is above 0.1 % too. The uniform box is also not more
expensive where it matters: for Si the graded box's 100 nm cells raise r_max and with it the
global-rule leaf size (362 vs 181 nm), so the graded box needs more memory (33 vs 17 GB) and more
iterations at fewer unknowns (ADR 0008's 2026-10-11 decision observed the same for Si at L ≥ 4 µm);
for Ag the uniform box converges in 0.56× (tol 1e-5) / 0.45× (tol 1e-4) the iterations of the graded
box, and its solve was faster despite 2.2–2.6× the unknowns.

## Verdicts against docs/05

| Check | Si (uniform box, fine band) | Ag |
|---|---|---|
| Depth ×2 < 0.1 % | **pass** at w₀/L ≤ 1/4 (0.032 %; 1/6: 0.007 %; three depth pairs); graded λ₁/5 box fails at 1/4 (0.14 %) | **fail** (0.82–1.89 %, ≥ 8× the threshold) at w₀/L = 1/4 and 1/8, L = 4 µm, uniform and graded box; the Im ε = −6 diagnostic gives 0.093 %, at the threshold within the noise (≈ 2.5e-4 per pair, not measured for the lossy material) |
| L ×1.2 < 0.5 % | **pass at w₀/L = 1/6** (0.19 %, one pair: 4.5 → 5.4 µm); fail at 1/4 (1.28 %) | **fail** (4.5 % at 1/4, 8.2 % at 1/8) |
| Where it passes | w₀ ≤ L/6 with the uniform box (both checks), for one seed, one roughness (σ = 50 nm, Lc = 500 nm), normal incidence, x polarisation, one L ×1.2 pair; nothing tested between 1/4 and 1/6; depth alone already at L/4 | not at L = 4–4.8 µm for any waist tried; cause not identified (SPPs and the WP-A2 box cavity resonances not separated); w₀/L is not the lever |
| Graded vs uniform | λ₁/5 insufficient at w₀/L = 1/4 (0.085 %, depth ×2 0.14 %; 1/6 untested) | λ₁/5 0.23–0.36 % from uniform (below the Ag box sensitivity, above 0.1 %) |

## Recommendations for ADR 0006 (proposed, not applied)

1. **Uniform box (box cells = top-face cells) for quantitative rough-surface results**, Si and Ag.
   At w₀/L = 1/4, λ₁/5 grading is not sufficient for the 0.1 % depth check (Si: 0.14 % vs 0.03 %
   uniform; Ag: grading error 0.23–0.36 %); at w₀/L = 1/6 the graded box was not tested. The
   recommendation stands on cost grounds as well: at MLFMM sizes the uniform box is the cheaper
   choice (smaller r_max → λ/2.8 leaves; Si at L = 3 µm 17 vs 33 GB and 205 vs 322 iterations; Ag
   0.45–0.56× the iterations), in line with the ADR 0008 decision of 2026-10-11 for Si. The λ₁/5 rule
   can stay as the minimum for qualitative work and dense studies.
2. **Si: w₀ ≤ L/6 at normal incidence** (L ×1.2: 0.19 % at L/6, 1.28 % at L/4; depth ×2 passes at
   L/4 already). Evidence base: one seed, one roughness (σ = 50 nm, Lc = 500 nm), x polarisation,
   three depth pairs and a single L ×1.2 pair (4.5 → 5.4 µm); nothing between 1/4 and 1/6, so the
   bound is sufficient on this evidence, not shown to be necessary. By WP22c's angle rule this
   becomes w₀ ≤ L cos θ_in / 6 (untested at oblique incidence). The current w₀ ≤ L/4 rule is enough
   for the flat Fresnel limit (WP22c) but not for the edge-effect check of a rough surface.
3. **Ag: the docs/05 sensitivity checks are not met at L = 4–4.8 µm, and the waist is not the
   lever** (fail at ≥ 8× the thresholds at every waist tried). The cause is not identified: SPPs
   excited by the roughness and the weakly damped box cavity resonances of WP-A2 (ADR 0006
   amendment 2026-10-12) are both consistent with the data, and the Im ε diagnostic damps both
   (section 2). Options for the coordinator, none tested:
   (a) state that Ag rough-surface far fields from a truncated patch carry a ~1 % (depth) / ~5 %
   (edge) box uncertainty at the sizes tested; whether larger L helps (L ≫ L_spp ≈ 21 µm intensity,
   2N ~ 10⁷) is open and worth testing only if (i)/(ii) of section 2 point to SPPs;
   (b) **conditional on (i)/(ii) showing SPP-dominated box currents**: an SPP-absorbing termination
   of the walls (e.g. a lossy rim band or lossy wall material, analogous to a PML/taper — changes the
   problem and needs its own validation). Its target must not be the 0.093 % of the lossy diagnostic
   without a tol 1e-5 rerun (that value is at the threshold within the noise). If the cavity
   resonances contribute, an absorbing termination of the walls does not address them; the remedy
   would then be the resonance-free formulation named in WP-A2 (a combined-field variant such as
   JMCFIE for the metal, which does not share the interior resonances of the exterior operator);
   (c) a periodic/tapered surface model for metals.
   Recommendation: run (i)–(iii) of section 2 first ((i) and (ii) are post-processing of saved
   currents, (iii) three solves at L = 4 µm), then choose between (b) and the formulation route.
4. **Hypothesis of WP-V1 (rim illumination by the incident beam controls the depth sensitivity):
   confirmed for Si, falsified for Ag.** Illumination of the rim and walls by surface waves or by
   grazing scattered fields is not excluded for Ag. Replace the open item in the 2026-10-10
   amendment accordingly.
5. **Solver settings for box-sensitivity studies:** d₀ = 3 is sufficient (d₀ = 5 changes the metric by
   2e-5). GMRES tol 1e-5 for Si (1e-4 leaves 4.7e-4, half the tolerance). For Ag, tol 1e-4 for
   screening only; for any run within about 3× of a threshold, tol 1e-5 or a noise floor measured
   for that configuration ≤ 1e-4 (the 1.8e-4 floor of section 1 holds for one Ag case and does not
   carry over: the tolerance-to-error mapping is case-dependent). Quantitative Ag reflectance:
   ≤ 1e-5 (unchanged).
6. **docs/05:** the two sensitivity checks are meaningful for weakly absorbing / dielectric objects at
   w₀ ≤ L/6 (on the evidence of recommendation 2); for plasmonic metals add a note that the checks
   were not met at L = 4–4.8 µm for any waist, that the box sensitivity does not decay with w₀/L,
   and that the remedy (larger L, an SPP-absorbing termination or a resonance-free formulation) is
   open until the experiments (i)–(iv) of section 2 separate SPPs from the box cavity resonances.

## Deviations from the brief and assumptions

- **Sizes.** Ag at w₀ = 1 µm with L = 6 and 8 µm was not run: the Ag GMRES at tol 1e-4 needs
  2 500–2 900 iterations already at L = 4 µm (47–78 min per run on the shared machine), so L = 6/8 µm
  (2N 1.4–3.4·10⁵, ~3 000+ iterations) would exceed the 2 h run limit. Instead the w₀/L sweep was done
  at fixed L = 4 µm with w₀ = 1 and 0.5 µm (1/4, 1/8: the rim intensity falls by 10⁸), which tests the
  rim hypothesis directly (same mesh and box, only the illumination changes); for Ag it falsified
  the control by the rim illumination of the incident beam, which makes larger L at w₀ = 1 µm
  uninformative for the waist question. Si at w₀ = 1 µm fits only at L = 4 µm with depth ×2
  (uniform box: the run `si4w1_d2_uni_t5` recorded a peak estimate of 50.55 GB, the setup model
  plus the Krylov basis at that run's `--max-iter`, which the rows file does not record; the 56 GB
  quoted here before was a planning estimate at a larger, unrecorded `--max-iter` (the Krylov basis
  grows by 0.82 GB per 100 iterations at 2N = 510 720, so the difference corresponds to ≈ 670 more
  iterations); the L = 5 µm planning estimates of 62–72 GB are not recorded either), so the Si
  sweep uses
  w₀ = 0.75 µm with L = 3 and 4.5 µm (1/4, 1/6) and their 1.2 L partners, plus one depth pair at
  w₀ = 1 µm, L = 4 µm (1/4).
- **Noise floor tolerances.** Si was measured between 1e-4/1e-5/1e-6 (the brief's 1e-3 vs 1e-4 would
  not have been well below the tolerance, see section 1); Ag between 1e-3/1e-4/1e-5. The Ag d₀ = 5
  run used tol 1e-4 (a d₀ = 5 run at 1e-5 would have taken ~1.5–2 h); the d₀ effect (1.9e-5) is
  measured against the d₀ = 3 run at the same tolerance.
- The Ag uniform depth-4 µm run used tol 1e-4 against the tol 1e-5 depth-2 µm reference (floor
  1.8e-4 vs an effect of 8e-3); the graded-vs-uniform pair at depth 4 µm compares tol 1e-4 runs.
- Beam: the WP-V1 study beam (`--beam spectrum`), not `excitation::AngularSpectrumBeam`; both are exact
  plane-wave sums with the same amplitude spectrum, but they are not identical beams: the study
  beam's polarisation vector e = x̂ − (k̂·x̂)k̂ is not renormalised per plane wave (the library's
  is), and the library beam truncates the spectrum at a different α_max. Both runs of every pair use
  the same beam.
- Diagnostic runs with `--eps-imag -6` are marked DIAGNOSTIC in their RESULT/COMPARE lines and do
  not enter any verdict.

## What was not run (capped)

- Ag at L ≥ 6 µm (any waist) and Ag at tol 1e-5 beyond the L = 4 µm references (run-time limit).
- Ag L ×1.2 with the uniform box (the graded result, 4.5 %, is 10× the grading error).
- Si graded box at w₀/L = 1/6; Si at L ≥ 5 µm with depth ×2 (memory estimate > 60 GB).
- Oblique incidence, s polarisation, other seeds and roughness parameters.
- The experiments (i)–(iv) of section 2 that would separate SPPs from the WP-A2 box cavity
  resonances (proposed in the review of this record, after WP-A2 landed).

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_box_validity
M="--compression mlfmm --max-iter 6000"
# estimate (memory guard: MLFMM setup peak model + Krylov basis at --max-iter <= --max-gb)
$B $M --material si --fine-band --L 4.5e-6 --L-gen 5.4e-6 --waist 0.75e-6 --depth-factor 2 \
   --box-mesh-size uniform --tol 1e-5 --max-iter 300 --estimate-only
# noise floor (Ag)
$B $M --material ag --L 4e-6 --L-gen 4.8e-6 --waist 1e-6 --depth 2e-6 --box-mesh-size 100e-9 \
   --tol 1e-5 --out ag4_d2_b100
$B $M --material ag --L 4e-6 --L-gen 4.8e-6 --waist 1e-6 --depth 2e-6 --box-mesh-size 100e-9 \
   --tol 1e-4 --out ag4_d2_b100_t4 --reference ag4_d2_b100              # add --d0 5 for the d0 floor
# depth x2 and L x 1.2 (Ag, w0/L = 1/4; w0 = 0.5e-6 for 1/8; --box-mesh-size uniform for the pair)
$B $M --material ag --L 4e-6 --L-gen 4.8e-6 --waist 1e-6 --depth 4e-6 --box-mesh-size 100e-9 \
   --tol 1e-4 --out ag4_d4_b100_t4 --reference ag4_d2_b100_t4
$B $M --material ag --L 4.8e-6 --L-gen 4.8e-6 --waist 1e-6 --depth 2e-6 --box-mesh-size 100e-9 \
   --tol 1e-4 --out ag48_d2_b100_t4 --reference ag4_d2_b100_t4
# Si (w0 = 0.75 um; L = 3 / 3.6 um and 4.5 / 5.4 um with --L-gen 3.6e-6 / 5.4e-6)
$B $M --material si --fine-band --L 4.5e-6 --L-gen 5.4e-6 --waist 0.75e-6 --depth-factor 1 \
   --box-mesh-size uniform --tol 1e-5 --max-iter 400 --out si45_d1_uni_t5
$B $M --material si --fine-band --L 4.5e-6 --L-gen 5.4e-6 --waist 0.75e-6 --depth-factor 2 \
   --box-mesh-size uniform --tol 1e-5 --max-iter 300 --out si45_d2_uni_t5 --reference si45_d1_uni_t5
# loss diagnostic (damps SPPs and box cavity resonances alike): --eps-imag -6 (DIAGNOSTIC)
$B --compare ag4_d2_uni_t5 ag4_d2_b100                                # stored runs, no solve
```

Every run of this record is one process (`OMP_NUM_THREADS=12`), with `--summary` collecting the
RESULT/PARTS lines and `--history` the GMRES residuals; all converged (true residual ≤ 1.2 × tol).
About 21 h of process time (30 runs) in two parallel streams over 12 h of wall time.
