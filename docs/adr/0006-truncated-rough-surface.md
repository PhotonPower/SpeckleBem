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
- The extra side/bottom triangles add unknowns (a few percent for L = 30 µm, depth 2 µm).
