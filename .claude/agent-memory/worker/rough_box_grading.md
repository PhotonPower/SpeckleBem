---
name: rough-box-grading
description: Non-obvious design facts of the graded rough-surface closing box (WP2b/WP2c) - anchor row, relaxation rows, shape safety net, measured box fractions and aspect ratios
metadata:
  type: project
---

Graded rough box (src/geometry/rough_surface.cpp, WP2c 2026-10-09):
- Rows below the rim follow an anchor row z_S that is piecewise linear between level-M rim
  nodes; that alone makes 2:1 cells non-invertible. Relaxation strips rim -> z_S are quads with
  vertical edges (only column monotonicity needed).
- Box budget: one level-0 row costs 2 % of the top face at L = 10 um / 50 nm, so the relaxation
  cap is 1 (R = 2 pushed sigma = 250 nm maps to 11 % > the 10 % acceptance).
- Wall aspect for rough rims is dominated by the rim shear (uniform WP2 box: 400-1500 for
  sigma = 250 nm, Lc = 100 nm). Graded rows reach up to 1.7 x the uniform walls for
  sigma = 250 nm, Lc = 500 nm (anchor slope shears 2:1 cells), so a strict "<= uniform" shape
  check breaks M = 3; the safety net uses max(4, 2 x uniform). Shallow boxes (depth 1 um,
  sigma = 250 nm) squeeze rows into slivers (3.4 x) and fall back.
- Si field decay length at 500 nm is 1.129 um (amplitude); intensity absorption length is half.

**Why:** these numbers drove the constants kMaxRelaxationRows, kAspectSlack, kMinColumnScale.
**How to apply:** when touching the box grading or its tests, re-run the hidden
`[.stats]` test in test_rough_surface.cpp before changing a constant.
