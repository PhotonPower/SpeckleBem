---
name: rough-box-grading
description: Non-obvious design facts of the graded rough-surface closing box (WP2b/WP2c) - anchor row, stitched relaxation band, shape safety net, measured box fractions and aspect ratios
metadata:
  type: project
---

Graded rough box (src/geometry/rough_surface.cpp, WP2c + review fix 2026-10-09):
- Rows below the rim follow an anchor row z_S that is piecewise linear between level-M rim
  nodes (z~ + D + h_g); that alone keeps 2:1 cells non-invertible.
- Coordinator decision (WP2c review C1): every vertical wall edge between rim and anchor row
  <= 2 h_g, even if the box exceeds the 10 % budget. Implemented as per-column relaxation
  nodes n_p = ceil(gap / 2 h_g) with adjacent columns stitched (zipper; a triangle with one
  edge on a vertical column can never invert). Global rows (old R <= 1 cap) were far too
  expensive: one level-0 row = 2 % of the top face at L = 10 um / 50 nm.
- Measured at L = 10 um, depth 2 um, seeds 1-5: (sigma, Lc) = (50, 500) nm 7.19 %;
  (250, 500) 7.9-8.1 %; (50, 100) 8.4-8.5 %; (250, 100) 13.7-14.0 % (> 10 %, flagged).
- Tried and rejected: coordinate descent lowering the anchor node values (minimal gaps)
  saved only 0.5 pp on (250, 100) and pushed more (250, 500) seeds over the aspect net.
- Aspect: stitched band is 0.3-0.5 x the uniform walls for (250, 100); for (250, 500) the
  anchor slope shears 2:1 cells up to 1.9 x uniform, seed 9 (L = 4 um) exceeds 2 x and falls
  back M 3 -> 2 (expected, tested). Safety net = max(4, 2 x uniform walls).
- Si field decay length at 500 nm is 1.129 um (amplitude); intensity absorption length is half.
- WP-B1 (2026-10-10, ADR 0006 amendment): level count M = round(log2(h_c / h_b)) rounds to
  the NEAREST level, so a lambda_1/5 cap must be passed as an exact 2^M h_b (rounded down),
  else 150 nm -> 200 nm. Defaults at L = 10 um, h = 50 nm, seed 1 (top 80 000 triangles):
  Ag depth 2 um M = 1 box 47.5 % (old 400 nm rule 7.19 %, uniform 180 %); Si 5.65 um + band
  3.54 um (73 rows) 195.5 % (old rule + band 153.4 %, uniform 326 %). The new "automatic rule
  not validated" warning changes warning counts in old tests that build graded meshes.
- WP-B1 review: the generator spacing is L / round(L / h), not mesh_size (L = 1.07 um,
  h = 50 nm -> 50.95 nm, so an explicit 2 h = 100 nm gives 101.9 nm cells > lambda_1/5).
  Any lambda-based cap must use the actual h_b, i.e. pass exterior_wavelength and leave
  box_mesh_size unset; precompute nothing from mesh_size. lambda_1 = lambda_0 / |n_1|.

**Why:** these numbers drove kMaxRelaxationEdge, kAspectSlack, kMinColumnScale and the test bounds.
**How to apply:** when touching the box grading or its tests, re-run the hidden
`[.stats]` test in test_rough_surface.cpp before changing a constant.
