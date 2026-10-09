---
name: touching-quadrature-findings
description: Error sources of Duffy/log-Gauss outer rules for touching triangle pairs and reference-convergence facts (WP7c), for future kernel/quadrature work
metadata:
  type: project
---

Measured in WP7c (2026-10-09) against the relative-coordinate reference in test_operators.cpp:
- Three distinct error sources of apex-graded Duffy outer rules: (1) source edges from the shared
  vertex at small elevation over the test triangle (sharp folds) -> near-singular *ray* in the
  angular coordinate; (2) obtuse apex angle / long opposite side -> complex zeros of |w(v)|^2
  close to [0,1]; (3) source vertices not at the apex near the *far side* of a piece -> only more
  radial (u) points help (log-Gauss is sparse near u = 1). Diagnose with separate u / v point
  counts: if more angular points do not help, it is radial.
- "Floors" that survive every outer refinement were reference-limited cases; check the
  reference n vs n + 8 first. Shared-edge reference at 30-degree folds needs n = 44 (n = 28 is off
  by up to 9e-7 for obtuse shapes); shared-vertex n = 28 converged to 4e-11, n = 16 not enough at
  30 degrees.
- Relative K error of nearly coplanar pairs (179 degrees) is amplified ~1/sin(fold): small
  absolute errors look large.
- Debug (ASan) cost: shared-edge reference n = 16 ~1.3 s, n = 28 ~6 s; RHS field evaluations
  (GaussianBeam) ~1e-4 s each under load: keep RHS unit studies to tiny meshes. Release is fast:
  the whole fold sweep (~100 pairs, n = 36..52 references, 3 regions) runs in ~25 s, so iterate
  quadrature experiments in release via a temporary hidden case + temporary add_test entry.
- WP7c review round: doubly obtuse shared edges (both triangles obtuse at A) need more points on
  the pieces of the *other* apex B (both directions: 16 x 14); far vertex projecting next to a
  shared vertex needs 16 x 16 on that apex's pieces. Near-vertex shapes at folds 90-95 degrees
  stay at ~1.4e-7 with the WP7b table (bitwise guarantee prevents changing them). A pair that is
  adaptive for another reason (aspect splits) can be *worse* than the WP7b table (10 vs 12
  angular points): give its pieces the extra points too.
- Reference shared-edge n = 20 suffices (<= 1e-9) at 60 degrees for the doubly obtuse and
  near-B shapes (fast unit checks, Si only).

- WP7d (2026-10-09): exact thresholds sit on structured-grid geometry: the apex zero of a right
  isosceles piece is at exactly kZeroMin = 0.5 (not only cos < 0) -> use relative tolerances.
  A hinge with C and C' both at 90 degrees over A and |AC| = |AC'| has K = 0 by symmetry (relative
  K errors ~1e5 are meaningless): use unequal legs in sweep shapes. Box-rim shared vertex (top
  test, wall source with its rim edge near the top plane) has a small K block (~6e-4 |k|a|L|):
  10 radial points gave 1.2e-7, 14 gave 2e-10. Ray partition with S on side AC near A needs
  more points on the *other* apex's pieces (far side passes A). The release fold sweep takes
  ~35-80 s, so try kernel heuristics there directly (temporary getenv overrides + hidden case).
- Small rough box for kernel tests: mesh_size 0.7 a (a = sweep size), L = 4 h, sigma 0.1 h,
  Lc 2 h, depth 2 h, box_mesh_size = h, seed 7 -> 128 triangles; references n 20 / 16 converged
  to 1e-10; one pair per TEST_CASE keeps win-debug ~2.5 s.

**Why:** these took most of the WP7c turn budget to discover.
**How to apply:** kernel/quadrature WPs (touching pairs, MLFMM near field, k-aware quad_degree_sing).
See [[build-pitfalls-windows]].
