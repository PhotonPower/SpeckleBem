# ADR 0006: Rough patches are closed objects (box with rough top)

- **Status**: accepted, amended 2026-10-10 (WP-V1; to be re-examined for lossless substrates)
- **Date**: 2026-10-08
- **Phase**: 1

## Context
A two-region SIE needs a closed interface. A finite rough patch of an infinite substrate is open. Options: (a) close the patch into a box; (b) open-surface PMCHWT-type approximations (not rigorous); (c) half-space (Sommerfeld) Green's function for the substrate (rigorous and exact for the infinite plane, but expensive kernels and a different fast method); (d) periodic unit cell (changes the physics).

## Decision
- (a): the patch is the top face of a closed box. Side walls and bottom are flat. The box depth defaults to the larger of 10 absorption lengths of the object material and a fixed minimum (≈ 2 µm), and is user-overridable.
- The beam is tapered (Gaussian, waist < L/3) so that the illumination at the side walls is negligible.
- Two sensitivity tests (depth ×2, L ×1.2) are part of the validation suite (docs/05_validation.md).
- For Si and Ag at 500 nm this is rigorous to the stated tolerance because the interior field decays before reaching the artificial boundary.

## Consequences
- Lossless or weakly absorbing substrates (glass) are *not* covered by this decision; they require (c) or an absorbing-layer construction and get their own ADR when scheduled.
- The extra side/bottom triangles add unknowns (a few percent for L = 30 µm, depth 2 µm). Measured with the graded box of WP2b (automatic coarse spacing min(depth/2, L/8, 10 h) = 500 nm, rounded to 2³ h = 400 nm): for L = 10 µm, mesh 50 nm, depth 2 µm the closing box has 4 500 wall + 1 250 bottom = 5 750 triangles, 7.2 % of the 80 000 top-face triangles (uniform WP2 box: 144 000, 180 %), and 3.4 % for L = 30 µm.
- Graded box, physics contract (WP2b, WP2c). The coarse walls and bottom plate do not resolve the field; they are valid only below a depth of a few field decay lengths δ = λ₀ / (2π |Im n₂|) under the rough interface (`material::field_decay_length`; at 500 nm Si δ = 1.13 µm, Ag δ = 25 nm). Strip-height bound right under the rim, with and without a fine band: every vertical wall edge between the rim and the anchor row (h_g + e below the rim; h_g = √(dx dy), the top-face spacing for a square grid; e ≥ 0 the local dip of the rim below its level-M interpolation, 0 for a smooth rim) is at most 2 h_g long, and the horizontal wall edges there have the top-face spacing. Without a fine band the first coarsened wall row lies about one h_g below the anchor row, i.e. 2 h_g + e below the rim (100 nm at 50 nm spacing for a smooth rim), and the full coarse spacing is reached 2^M h_g below it (400 nm in the reference case). That targets strongly absorbing objects (Ag: first coarse row at ≈ 4 δ). Weakly absorbing objects get the fine band `box_fine_depth` (WP2c): the walls keep the top-face spacing down to z = box_fine_depth and grade only below. box_fine_depth is measured from the mean plane z = 0, so below a rim point at height ξ the band reaches only z_f − ξ deep; `Simulation` (WP14) passes 3 δ + 3 σ. Cost for Si (3 δ = 3.39 µm, σ = 50 nm, Lc = 500 nm, h = 50 nm): L = 10 µm 70 fine rows, closing box 146 % of the top face at depth 4 µm and 147 % at depth 5.65 µm (10 intensity absorption lengths), against 260 % and 326 % for the uniform box; L = 30 µm 50 % against 153 % and 175 %. `box_mesh_size` must not be below the top-face spacing (values < h/√2 are warned about and give the uniform box): the interior wavelength λ₀/|n₂| is resolved by the top-face mesh, never by the box.
- Graded box, rough rims (WP2c). The wall rows below the rim no longer follow the rim: an anchor row, piecewise linear between the level-M rim nodes and at least h_g below the rim in every column, is reached through a relaxation band, and all deeper rows interpolate between the anchor row and the bottom plate. Every 2:1 transition cell is then a shear of the flat one and cannot invert for any roughness. The relaxation band splits every wall column evenly into n_p = max(1, ⌈(z_S − z_rim) / (2 h_g)⌉) segments and stitches adjacent columns (each triangle has one edge on a vertical column, so none can invert), which enforces the 2 h_g strip-height bound above while adding nodes only where the rim dips (a global relaxation row costs 2 % of the top face at L = 10 µm, h = 50 nm; with one global row, as first implemented, the strips under σ = 250 nm, Lc = 100 nm rims were 500–620 nm tall). Measured on generated maps (L = 10 µm, h = 50 nm, depth 2 µm, automatic M = 3, seeds 1–5; closing box relative to the top-face triangles; R = the largest number of relaxation nodes in one column): σ = 50 nm, Lc = 500 nm R = 0, 7.19 %; σ = 250 nm, Lc = 500 nm R = 2, 7.9–8.1 %; σ = 50 nm, Lc = 100 nm R = 2–3, 8.4–8.5 %; σ = 250 nm, Lc = 100 nm R = 11–13 (about 2 700 relaxation nodes), 13.7–14.0 %. The last corner (rms slope 3.5) exceeds the 10 % target of the backlog: resolving the region right under the rim takes priority over the box budget (coordinator decision of the WP2c review); the unit test bounds it at 14.5 %. Under WP2b (rows following the rim, global M reduced by inverting cells) σ = h, Lc = 2 h gave the uniform box (180 %) and σ = 250 nm, Lc = 500 nm mostly M = 1 (34 %). The wall aspect ratios of very rough rims are set by the rim steps themselves, as in the uniform box (the stitched band stays at 0.3–0.5 × the uniform walls for σ = 250 nm, Lc = 100 nm); the anchor row of a steep rim additionally shears the 2:1 cells (σ = 250 nm, Lc = 500 nm: up to 1.9 × the worst uniform-box wall triangle in a 20-seed sweep at L = 4 µm). The global-M guard remains as a safety net: it reduces M with one warning when the rows below the anchor row are squeezed (a column with less than a quarter of the mean height between anchor row and bottom plate, e.g. a deep rim pit next to the bottom plate) or the graded walls are more than max(4, 2 × the uniform-box walls); an unlucky seed of a very rough, steep rim is expected to trigger the latter (σ = 250 nm, Lc = 500 nm, L = 4 µm, depth 2 µm: 1 of 20 seeds falls back from M = 3 to M = 2). A fine band that leaves no room for the grading reduces M as requested, logged with SBEM_INFO.

## Amendment 2026-10-10 (WP-V1: validity under beam illumination)

Evidence: `benchmarks/results/box_validity.md` (dense operator, Ag and Si at 500 nm, L ≤ 2.4 µm). The
decay-based contract above covers only the object side of the walls and the bottom. The exterior side
carries the incident beam: in the shadow the bottom must cancel E_inc, and coarse cells cannot. Measured:
400 nm cells (the automatic rule) are a 1.5–1.9 % far-field error against the uniform box; 100 nm
(λ₁/5) 0.17–0.59 % at fixed depth, but 1.5 % at depth ×2 (Ag, L = 1 µm). At these sizes neither docs/05
sensitivity check passes (depth ×2: Ag 0.8–3.5 %, uniform box 1.1 %, Si with fine band 0.13 %;
L ×1.2: 3.9–6.0 %), partly because the beam lights the rim (w₀ = L/3).

Decisions (project lead, 2026-10-10):
1. **Coarse spacing ≤ λ₁/5** (λ₁ = wavelength in R1; 100 nm at 500 nm in vacuum) replaces the automatic
   min(depth/2, L/8, 10 h) as the default whenever the wavelength is known (`Simulation` helpers, Python).
   This is a necessary condition, **not shown sufficient**: the graded box is validated only once the
   docs/05 checks pass at MLFMM sizes (WP-V2). The old automatic rule stays available for meshes built
   without a wavelength and logs a warning that it is not validated under illumination.
2. **Fine band mandatory for weakly absorbing objects** (Si: 0.4–0.9 % far-field change without it and a
   4× higher depth sensitivity); `Simulation` passes box_fine_depth = 3 δ + 3 σ as before, now required.
3. **Beam waist w₀ ≤ L/4** (was < L/3): at L/3 the rigorous beam carries 0.6–0.9 % of its power past the
   patch edges, at L/4 0.02 %. Larger waists are rejected with an exception unless explicitly allowed.
4. **Rigorous beam:** quantitative rough-surface results use an exact (angular-spectrum) Gaussian beam;
   the paraxial beam is 2.8 % off in reflectance at w₀ = 0.67 µm and stays for qualitative work.
5. **Metric of the sensitivity checks** (docs/05): the L2 difference of |F|² over the reflection
   hemisphere (sin θ weights) is the verdict metric; ε_rr on the xz/yz cuts is reported alongside.
Implementation notes (WP-B1): λ₁ = λ₀/|n₁| (conservative for lossy backgrounds; metallic backgrounds
   are rejected); the cap is applied by the generator with the actual grid spacing L/round(L/h) and rounded
   down to a 2^M level, strictly (a patch whose spacing is slightly above h may fall back to M = 0, the
   uniform box — choose L as a multiple of h to avoid it; no tolerance, the rule is already only necessary);
   "weakly absorbing" (fine band required) means 3 δ > 2 h, i.e. the field has not decayed to e⁻³ at the
   first coarsened wall row of a band-less box; if the band reaches the bottom the helper uses the uniform
   box with a warning (`RoughBoxParams::uniform_fallback`).
6. Default depths unchanged (Ag 2 µm, Si 5.65 µm): the Si bottom radiates nothing, the Ag bottom only
   through the grading error.

Open (WP-V2): whether λ₁/5 suffices and at which w₀/L the depth check falls below 0.1 % (hypothesis: the
depth sensitivity follows the rim illumination, not the Rayleigh range); Ag surface plasmons carry about
half of the Ag depth sensitivity at L ≤ 2 µm and become relevant at L ≳ 20 µm.

## Amendment 2026-10-11 (WP22c: oblique incidence, solver tolerance)

- **Waist at oblique incidence:** the footprint is 1/cos θ_in longer along x; w₀ = L/4 at 45° loses 0.63 % of
  the beam power past the patch (WP22c). Rule: **w₀ ≤ L cos θ_in / 4** (equal edge loss at all angles;
  at 45° w₀ = L/5 was measured sufficient, edge loss 0.06–0.1 %). `check_beam_waist` gets the incidence
  angle (follow-up).
- **Solver tolerance for quantitative Ag results:** a true relative residual of 1e-4 leaves ~0.5 % error in
  the specular reflectance of Ag (flat, 0°); **GMRES tol ≤ 1e-5 for quantitative Ag reflectance**. Si is
  accurate at 1e-4.
- Open: the Ag absorbed power from the currents is ~40 % low at 45° (energy balance open by 0.3–0.7 %)
  while R meets the criterion — investigate before absorption is used quantitatively.

## Amendment 2026-10-12 (WP-A1, WP-A2: absorbed power of lossy metals)

- **Ag absorption is reported only as A_ext** (far field plus extinction, WP-A1: +0.18 % vs Fresnel), never
  from the currents (½ Re∮(n̂×M)·J*). Reason (WP-A2): the closed box has weakly damped cavity resonances
  of the complementary problem (vacuum inside the box, Ag outside); the discretisation places them and
  excites them, and the resulting non-radiating current component spoils A_local by up to −53 %
  (L = 2.5 µm; D = 2.0 → 2.1 µm moves it from −2 % to −34 %), while R, the far field and A_ext stay within
  0.4 % in all 13 cases. Unchanged with d₀ = 5 at L = 2.5 µm (not MLFMM-driven); the resonances are too
  dense for a box-dimension rule.
- Diagnostic (proposed, follow-up WP-A3): rms of the extinction-theorem residual inside the box on a line
  above max ξ; > 1 % strong, 0.7–1 % moderate mode. Calibrated on flat Ag at 45° s only; a rough-surface
  calibration precedes a library function. A resonance-free formulation is an open option.

## Amendment 2026-10-12b (WP-V2: box checks at MLFMM sizes; lead decision)

Decided by the project lead on the WP-V2 results (`benchmarks/results/box_validity_mlfmm.md`):

1. **Uniform box by default for quantitative rough-surface runs** (Si and Ag): box cells = top-face
   cells (`box_mesh_size` = h, M = 0). The graded λ₁/5 box is insufficient for Si at w₀/L = 1/4
   (depth ×2 0.14 % vs 0.03 % uniform) and is more expensive at MLFMM sizes (Si L = 3 µm: 33 vs 17 GB,
   322 vs 205 iterations; Ag about twice the iterations). The graded box stays available as an explicit
   option (cheaper dense studies, convergence checks). Supersedes decision 1 of the 2026-10-10 amendment
   as the default; λ₁/5 remains the upper bound for any explicit coarse spacing.
2. **Weakly absorbing objects (3δ > 2h, e.g. Si): w₀ ≤ L cos θ_in / 6** (was L cos θ_in / 4). At L/4 the
   edge check fails (L ×1.2: 1.28 %), at L/6 both checks pass (0.19 % / 0.007 %). Provisional: one seed,
   one roughness (σ = 50 nm), one L ×1.2 pair; to be confirmed on a second seed and σ = 100 nm.
   Strongly absorbing objects keep L cos θ_in / 4 (no validated tighter rule; see 3).
3. **Plasmonic metals (Ag): no quantitative rough-surface speckle results until WP-V3.** Ag fails both
   docs/05 checks at every waist tried (depth ×2 0.8–1.9 %, L ×1.2 4.5–8.2 % at L = 4 µm); the cause
   (surface plasmons vs the box cavity resonances of the 2026-10-12 amendment) is open. Reflectance of
   flat Ag (WP22c) and A_ext remain validated.

## Amendment 2026-10-13 (WP-V2b: waist rule confirmation; lead decision)

Decided by the project lead (option A of the WP-V2b record, `benchmarks/results/box_validity_mlfmm.md`,
section "WP-V2b"):

1. **Decision 2 of the 2026-10-12b amendment holds for σ ≤ 50 nm** (no longer provisional there): at
   σ = 50 nm a second seed passes at w₀ = L/6 (depth ×2 0.009 %, L ×1.2 0.23 %), and L/5 passes on seed 1
   (0.007 % / 0.15 %). L/6 stays the library default for weakly absorbing objects; it is not relaxed to
   L/5 on one seed.
2. **σ = 100 nm: no waist rule.** At L = 4.5 µm the depth check passes (0.03 %) but L ×1.2 fails at
   w₀ = L/6 (1.19 %) and L/8 (1.28 %): a tighter waist is not the lever (measured). Likely cause
   (hypothesis, not separated from radiation by the added rough annulus): fields scattered by the
   roughness reach the rim independently of the beam. Box-route Si results at σ = 100 nm are reported with
   an **edge uncertainty of about 1.2–1.3 %** (the docs/05 L ×1.2 change at L ≈ 4.5 µm as a stand-in for
   the edge error, not a measured bound). Whether it falls with the beam-to-rim distance at larger L
   (e.g. the L = 30 µm server run) is not measured; no follow-up is scheduled (lead decision). The PML
   route (ADR 0009) is the alternative termination if a tighter bound is needed later.
