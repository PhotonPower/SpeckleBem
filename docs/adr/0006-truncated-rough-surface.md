# ADR 0006: Rough patches are closed objects (box with rough top)

- **Status**: accepted (to be re-examined for lossless substrates)
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
- Graded box, physics contract (WP2b). The coarse walls and bottom plate do not resolve the field; they are valid only below a depth of a few field decay lengths δ = λ₀ / (2π |Im n₂|) under the rough interface. The first coarsened wall row lies 2 h_g below the rim (h_g = √(dx dy), the top-face spacing for a square grid) and the full coarse spacing is reached 2^M h_g below it (400 nm in the reference case). The automatic rule therefore targets strongly absorbing objects (Ag at 500 nm: δ ≈ 27 nm, first coarse row at ≈ 4 δ). Weakly absorbing objects (Si at 500 nm: δ ≈ 1.1 µm) need the uniform box (`box_mesh_size = mesh_size`) or a fine band down to ≈ 3 δ; that band (`box_fine_depth`) is the follow-up WP2c. `box_mesh_size` must not be below the top-face spacing (values < h/√2 are warned about and give the uniform box): the interior wavelength λ₀/|n₂| is resolved by the top-face mesh, never by the box.
- Graded box, rough rims. The wall rows follow the rim and the number of coarsening levels M is global, so a few steep rim cells (2:1 transition cells that would invert) reduce the grading of the whole box, down to the uniform box (one `SBEM_WARN` per mesh). Measured on generated maps (L = 10 µm, h = 50 nm, depth 2 µm, automatic M = 3, seeds 1–10; closing box relative to the top-face triangles): σ = h with Lc = 2 h always gives M = 0 (uniform box, 180 %), Lc = 3 h gives M = 0 or 1 (mean 127 %), Lc = 4 h M = 1–3 (17 %), Lc ≥ 5 h keeps M = 3 (7.3 %); at Lc = 10 h, σ = 4 h mostly gives M = 2 (14 %) and σ = 5 h (250 nm) mostly M = 1 (34 %). The planned fix is WP2c: wall rows decoupled from the rim below a fine band, so that rim roughness no longer limits M.
