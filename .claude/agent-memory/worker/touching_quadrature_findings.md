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

**Why:** these took most of the WP7c turn budget to discover.
**How to apply:** kernel/quadrature WPs (touching pairs, MLFMM near field, k-aware quad_degree_sing).
See [[build-pitfalls-windows]].
