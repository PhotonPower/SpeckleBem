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
