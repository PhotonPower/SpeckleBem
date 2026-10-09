# ADR 0004: Singularity subtraction with analytic static integrals

- **Status**: accepted
- **Date**: 2026-10-08
- **Phase**: 1–2

## Context
Galerkin RWG matrix entries contain `1/R` (`G`) and `1/R²` (`∇G`) singularities for coincident, edge- and vertex-adjacent triangles. The paper uses "direct integration methods" (Hänninen, Taskinen & Sarvas 2006). Accuracy of these entries bounds the accuracy of the whole method, and they are the most expensive near-field entries.

Alternatives: Duffy transform / radial-angular transforms (purely numerical, robust, more quadrature points); fully analytic Galerkin formulas (complex, only for some kernels); singularity cancellation schemes.

## Decision
- Singularity subtraction: subtract the first terms of the Taylor expansion of `e^{−jkR}` (`1/R`, `R` and, for `∇G`, `∇(1/R)` and `∇R`) and integrate them analytically over the source triangle with the Wilton/Graglia/Hänninen formulas; integrate the smooth remainder with Dunavant rules; the outer (test) integral is numerical with a rule chosen by proximity class.
- Keep a Duffy-transform implementation as an independent cross-check in the test suite.
- Proximity classes: `identical`, `shared_edge`, `shared_vertex`, `near` (distance < 2× size), `far`; quadrature degrees are `OperatorOptions` parameters.
- Quadrature degrees for all classes except `far` must use Dunavant rules with all points inside the triangle and positive weights (degrees 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19; `kernels::triangle_rule_is_positive_interior`). The subtracted remainder is only finitely smooth at R = 0 and the analytic static potential is non-analytic across the source triangle's edges, so a test point outside its triangle samples the wrong branch; `OperatorOptions` validation rejects the other degrees for near and touching pairs.
- Defaults since WP7b (`kernels::OperatorOptions`): touching pairs use the graded outer rule
  with `outer_grading_levels = 4` (0 restores the WP7 scheme) and the smooth-remainder degree
  `quad_degree_sing = 10`; near and far pairs choose their Dunavant degree per pair for
  `target_accuracy = 1e-5` within [`quad_degree_far`, `quad_degree_near`] = [3, 19]
  (`target_accuracy = 0` uses fixed degrees, `quad_degree_near = 8` then gives the WP7
  near degree). Shared-edge / shared-vertex blocks are
  averaged over both orderings above `symmetrize_touching_above_kh = 1`.
- Since WP7c: shared-edge and shared-vertex pairs with a fold below 90 degrees, an obtuse
  angle at a shared vertex or an elongated test triangle use a fold-adaptive outer rule for the
  analytic part
  (`fold_adaptive = true`; all other pairs, including every touching pair of the Mie
  icospheres, keep the WP7b table bitwise), and the right-hand side has its own Dunavant degree
  `quad_degree_rhs = 8` (before: `quad_degree_near` = 19).

## Consequences
- Near-field accuracy (WP7b, measured against independent references in
  `tests/unit/test_operators.cpp`, relative to the block norm of L and of K separately, triangles
  with |k| h <= 0.78). Touching pairs: the inner subtraction is exact to ~5e-11; the outer
  (test) integral of the analytic part uses Duffy sub-triangles graded towards the singular set
  (all edges of an identical triangle, the shared edge, the shared vertex) with generalised
  Gauss-Legendre-log tensor rules (exact for x^j and x^j log x; `outer_grading_levels`, default 4:
  384 / 288 / 100 points for identical / shared-edge / shared-vertex pairs), the smooth remainder
  a Dunavant outer rule (degree 13 for identical triangles so that outer and inner points
  differ), and the first non-smooth gradient term (k^4/8) R (r' - r) is also integrated
  analytically. Result: L and K within 3e-9 of a Sauter-Schwab-type relative-coordinate
  reference (self-converged to 1e-12; it replaces the hp-graded polar reference, which needs
  ~10^8 kernel evaluations per pair for the log-singular K integrand), raw swap asymmetry
  below 5e-10, so shared-edge and shared-vertex blocks are averaged over both orderings only
  above |k| h = `symmetrize_touching_above_kh` (default 1; there the remainder error,
  ~(|k| h)^4, dominates); identical L blocks are always symmetrised (free). The 1e-8 level
  holds for shared edges with dihedral angles >= ~90 degrees and non-obtuse angles at the
  shared vertices. With the WP7b table alone sharper folds were near-singular (60 degrees
  1.5e-8 to 3.4e-7 at level 4, 30 degrees 6.8e-8 to 4.5e-5, still 3.7e-6 at level 6 for a
  skewed pair); resolved by the WP7c fold-adaptive rule (below). `target_accuracy` does not
  govern touching pairs: their smooth remainder uses the fixed `quad_degree_sing` and its
  error grows like (|k| h)^4 (~1e-9 at |k| h = 0.78, ~1e-6 at |k| h = 4); a k-aware
  `quad_degree_sing` is a follow-up. The WP7 scheme (one Dunavant outer rule, both orderings averaged: ~2e-4 for L and
  ~1e-2 of the K block) remains available with `outer_grading_levels = 0`. Near and far pairs:
  the Dunavant degree is chosen per pair from an empirical error envelope in
  kappa = sqrt((h/D)^2 + (0.15 |k| h)^2) so that the block error is below `target_accuracy`
  (default 1e-5) within [`quad_degree_far`, `quad_degree_near`] = [3, 19]; at the class boundary
  of the n = 4 Mie mesh this gives 4.4e-7 (n = 1.5) and 2.5e-6 (Ag) (5.8e-8 and 4.4e-7 with
  1e-6) instead of 4.3e-3 and 2.5e-2 with the fixed degrees 3 / 8. The default 1e-5 was chosen
  over 1e-6 because the dense n = 3 Mie assembly is 20-38 % faster and eps_rr changes by at
  most 1e-8 (absolute). Cost (release, icosphere n = 3 at 500 nm, |k| h ~ 1-5, so shared-edge
  and shared-vertex blocks are averaged; mean per `element_blocks` call, WP7 options ->
  defaults): identical 30 -> 125 µs (~4.2x; 3.5-4.3x by material in the slow cases), shared edge 55-75 -> 175-200 µs (2.4-3.5x), shared
  vertex 55 -> 90-105 µs (1.6-1.8x), near 7-9 -> 9-12 µs (1.2-1.7x), far 1.0 -> 2.6-7.6 µs
  (2.6-8x; the most for the Ag interior, where the selection resolves blocks that the
  exp(-Im k R) decay makes negligible). Far-pair cost by target (vacuum / n = 1.5 / Ag
  interior): 1e-4 1.8 / 2.1 / 3.9 µs, 1e-5 2.6 / 5.0 / 6.6 µs, 1e-6 5.0 / 5.1 / 10.4 µs. Far
  pairs dominate dense assembly: the n = 3 icosphere (2N = 3840) takes 6.6 s (n = 1.5), 7.7 s
  (Ag) and 11.8 s (Si) with the 1e-5 default on 3 threads (5.2 / 5.9 / 9.1 s on 4), against
  6.3 s, 7.9 s and 14 s with 1e-6 in an earlier run and 2.7 s with the WP7 options (3 threads;
  the paired comparison 1e-5 vs 1e-6 on one machine gave 20-38 % less assembly time). On a
  24-core Windows machine (win-release, 24 OpenMP threads) the n = 3 assemblies take 1.1 /
  1.3 / 1.9 s (n = 1.5 / Ag / Si) and the n = 4 ones (2N = 15 360) 7.5 s (n = 1.5) and 11 s
  (Ag). Against the WP7 options the Mie eps_rr (xz / yz) changes by at most 0.03 percentage
  points on the n >= 3 meshes (n = 2 Ag yz improves from 1.51 % to 1.37 %): n = 4 0.073 % / 0.074 % (n = 1.5; WP7 0.074 % / 0.076 %) and 0.078 % / 0.057 % (Ag;
  unchanged), n = 3 0.263 % / 0.274 % (n = 1.5), 0.348 % / 0.256 % (Ag) and 0.554 % / 0.396 %
  (Si; WP7 0.58 % / 0.42 %), while the Si power-balance defect drops from 0.42 % to 0.007 %
  (`benchmarks/results/mie_sphere_dense.md`).
  Touching outer points at level 4: 384 / 288 / 100 (identical / shared edge / shared vertex)
  instead of 25 (Dunavant degree 10); worst touching error by level 0..4: 2.7e-2, 8.8e-3,
  5.6e-5, 3.1e-7, 3.0e-9.
- Fold-adaptive outer rule (WP7c, `OperatorOptions::fold_adaptive`, algorithm in
  `src/kernels/operators.cpp`). Analysis: in the Duffy coordinates r = x + u w(v) of a piece with
  apex x (a shared vertex) every singular feature of the analytic part through x is a function
  of the squared distance of w(v) from it, a quadratic in v; its complex zero v_f + j q is where
  the angular quadrature sees the singularity. Three error sources were identified by the
  dihedral sweep (30 to 179 degrees x regular / skewed / obtuse x shared edge / shared vertex,
  both orderings, vacuum / Si / Ag, against the relative-coordinate reference):
  (1) sharp folds: a source edge from the shared vertex at a small elevation over the test
  triangle (q ~ elevation, i.e. ~ sin(fold) for a regular shape and smaller for skewed ones),
  near-singular along its projection, a ray from the apex — this, not the far vertex, is what
  limited the WP7b table (the far vertex lies r sin(fold) above the test plane, 0.4 a at 30
  degrees); (2) obtuse angles at the apex: the zeros of |w(v)|^2 lie at
  v_f + j h/|p2 - p1| (h the height of x over the side), 0.36 side lengths from [0, 1] for 106
  degrees, which no apex-graded rule resolves (7e-7 at level 4, 5e-8 at level 6 for a 120-degree
  test triangle); (3) radial near-singularities from source vertices not at the apex close to
  the far side of a piece (vertex A seen from the pieces of apex B behind an obtuse angle; the
  far source vertex of 30-degree shared-vertex folds), which only the number of radial points
  reduces. Options considered: auto-raising `outer_grading_levels` (level 6 still leaves 3.7e-6
  for the skewed 30-degree hinge and does not touch (2)); grading towards the near-singular
  vertex (addresses neither the ray (1) nor (2)); re-splitting the Duffy sub-triangles at the
  singular directions — chosen: pieces are split at the real part of every zero closer than
  0.5 side lengths to [0, 1] (geometric splits towards zeros outside the interval), zeros within
  1e-3 of an end grade that side with the log-Gauss rule; for shared edges with a fold clearly
  below 90 degrees (lambda_C of the projected far vertex > 0.1) the pieces are first cut along
  the projections of both source edges (rays A S, B S), unless S is within 0.2 |AB| of a shared
  vertex (then a ray from one apex would end next to the other singular vertex, error 1e-5).
  Points per piece: 6 + 2 l radial (log-Gauss) x 2 + 2 l angular, each at most 16 (14 x 10 at
  level 4; 12 radial points left the obtuse and 30-degree shared-vertex cases at 1e-7 to 2e-6).
  Source (3) needs more points on two kinds of shared-edge pieces (review of WP7c): with the
  angle at A obtuse in both triangles (doubly obtuse, 106 and 120 degrees: 4.3e-8 to 6.3e-7 at
  level 4 with 14 x 10, a regression against the WP7b table at 45 degrees, 3.9e-8 -> 1.3e-7)
  the pieces of apex B use the points of level l + 2 (16 x 14; 16 x 10 and 14 x 14 left 5.3e-7
  and 2.4e-7), and with the projected far vertex within 0.2 |AB| of a shared vertex (the
  near-vertex partition and the WP7b partition of folds just below 90 degrees: 6.2e-7 at 60
  degrees, 5.4e-7 at 75 degrees for the far vertex 0.27 |AB| from B) the pieces of that apex use
  level l + 3 (16 x 16; 16 x 14 left 8e-8). Hard bound per ordering: 48 pieces x 16 x 16 =
  12 288 outer points (WP7b table at level 4: 288 / 100); measured at most 2 376 for both
  orderings after WP7d (75-degree hinge with right angles at A; `kernels::touching_rule_info`
  reports the counts); WP7c: 2 624 at the doubly obtuse 75-degree hinge, now 2 176.
  Pairs that need no split and are not doubly obtuse keep the WP7b table (the near-vertex extra
  points never make a pair adaptive by themselves; they apply only to pairs that use the
  adaptive pieces anyway), so their blocks and the Mie
  results are bitwise unchanged (tested): folds >= 90 degrees unless the apex point feature
  splits a piece (shared edge: |AB| h_C < |MC|^2 with the foot inside the side, roughly the
  median from C longer than AB or an obtuse angle at A or B; shared vertex: 4 area < |BC|^2,
  roughly an obtuse angle at A) or both triangles are obtuse at the same shared vertex; every
  touching pair of an icosphere qualifies. The backlog criterion (WP7c (1): touching-pair error
  <= 1e-7 down to 30 degrees) is met for all swept shapes below 90 degrees; folds >= 90 degrees
  are bitwise WP7b only for the pairs above, the obtuse shapes there change (WP7b table up to
  7.8e-7 at 179 degrees). Measured at level 4: <= 8.3e-8 over the sweep (regular, skewed,
  obtuse, doubly obtuse and near-B shared edges, regular, skewed and obtuse shared vertices;
  WP7b table: up to 4.5e-5 skewed edge, 4.2e-5 near-B edge (45 degrees), 3.1e-5 obtuse edge,
  8.7e-5 obtuse vertex); raw asymmetry <= 2e-7. Cost (release, Si, time per `element_blocks`
  call against the WP7b table, both orderings, quietest of three runs; single-pair maxima vary
  by up to x2 under machine load, the point ratios are the deterministic measure): folds below
  90 degrees mean x1.9 for shared edges (at most x3.3, doubly obtuse 75 degrees; x2.8 without
  the extra points) and x1.6 for shared vertices (at most x3.2, obtuse 30 degrees); folds >= 90
  degrees x1.3
  (shared edges, at most x2.5 doubly obtuse) and x1.2 (shared vertices, at most x1.7 obtuse);
  outer points of the analytic part mean x2.5 / x2.6, at most x4.6 / x6.3 (shared edge /
  vertex below 90 degrees). These are the WP7c measurements; the WP7d sweep gives: shared edges
  below 90 degrees points mean x2.58 (at most x4.12), time mean x1.9 (at most x2.88); shared
  vertices below 90 degrees points mean x1.88 (at most x4.9); shared edges >= 90 degrees points
  x1.92 / x3.4, time x1.55 / x2.48; shared vertices >= 90 degrees time x1.19 / x1.65. The mean meets the backlog's x2; the maxima (rare doubly obtuse and
  obtuse shared-vertex pairs) exceed it and are accepted. Not covered: (a) the near-B shape at
  folds of 90 to ~95 degrees keeps the WP7b table, 1.4e-7 at 90 degrees (similar shapes up to
  1.5e-7; < 1e-7 from ~95 degrees); extending the near-vertex points to these pairs would leave
  the WP7b table also for regular shapes at the 0.2 |AB| boundary; (b) the fold features switch
  off discontinuously at lambda_C = 1e-6 (89.9999 degrees); (c) source features neither through
  the apex nor the far vertex (e.g. the opposite source edge of a shared-vertex pair passing
  close over the test triangle at folds below 30 degrees) only enter through the radial point
  number.
- Structured grids (WP7d review follow-up). Rough-surface top faces consist of right isosceles
  cell halves and the box rim has right angles in the top and the wall triangles; the exact
  tests of WP7c (cosine < 0 for obtuse angles, zero distance < 0.5 for splits; the apex zero of
  a right isosceles piece lies exactly at 0.5) chose a rule by rounding (sweep shape with 90
  degrees at A in both triangles: 908 to 1 696 points, depending on +-1e-9 perturbations). An
  angle now counts as obtuse only if its cosine is below -1e-6 and a zero splits only if it is
  closer than 0.5 (1 - 1e-6): right angles take the rule of acute ones, which meets the bound
  there (exact and perturbed shapes select the same rule; <= 5e-8 at level 4). The split
  tolerance also matters for generic shapes: a child piece created by a geometric split has its
  zero at exactly kZeroMin, so before WP7d rounding decided whether it split again (rules
  changed with the overall scale); now it does not split, so some generic rules use fewer pieces
  (regular 75 degrees: 7 -> 6 pieces; obtuse shared vertex 30 degrees: 1 260 -> 980 points),
  with all sweep bounds kept. Box rim: a
  shared-vertex pair of a top test triangle and a wall source triangle whose rim edge runs
  close to the top plane outside the wedge reached 1.2e-7 of its (small) K block with the WP7b
  table (10 radial points; the rim vertex is a radial near-singularity ~0.47 side lengths away),
  2e-10 with the adaptive piece (14 radial points). Shared vertices with a steep source
  triangle (|n . n'| < 0.7, folds ~46 to ~134 degrees) and a source vertex less than 11.5
  degrees above the test plane therefore use the adaptive piece (140 instead of 100 points).
  This also applies to the reversed ordering of the sweep's regular and skewed shared vertices
  at 90 and 120 degrees (vertex B on the hinge lies in the test plane), which so leave the WP7b
  table; icosphere pairs are unaffected (bitwise, tested). All touching pairs of a 4 x 4 rough
  box (top/top, top/wall, bottom/wall and wall/wall corners; Si): <= 1.6e-8 at sigma = 0.1 h and
  <= 2.0e-8 at sigma = 0.3 h (before WP7d: up to 2.4e-7 and 6.7e-7 — the WP7c bound of 1e-7 did
  not hold on boxes). Rim pairs
  with rim slopes steeper than ~11 degrees against the top triangle keep the WP7b table. The
  new sweep shape with 90 degrees at A in both triangles revealed a gap of the ray partition at
  a 75-degree fold (S on the side A C, 0.26 |AB| from A, just outside the 0.2 |AB| near-vertex
  radius: 3.8e-7, WP7b table 7.3e-8): the far sides of the pieces of apex B pass close to A, a
  radial near-singularity; with S within 0.3 |AB| of A (B) the pieces of the other apex now use
  level l + 3 (4.7e-8). Sweep at level 4 including the right-angle shapes: <= 8.3e-8 (near-B
  at 90 degrees 1.4e-7, the accepted deviation, pinned by a test); right-angle shapes <= 4.7e-8.
- Right-hand side (WP7c): own Dunavant degree `quad_degree_rhs`, default 8. Relative error of
  <f_m, E_inc> / <f_m, H_inc> against degree 20 for a plane wave and a paraxial Gaussian beam
  (w0 = lambda) on an icosphere and a rough-surface box at h = lambda/10 (lambda/27): degree 4
  5.2e-7 (9.8e-9), 5 5.0e-9 (3.0e-11), 6 7.1e-11 (1.8e-13), 8 7.5e-14 (4.5e-16); degree 19
  (the WP7b right-hand side) is not needed. Degree 6 would meet the 1e-8 criterion; 8 (the WP7
  rule) keeps 1e-9 even at h ~ lambda/3 (9.4e-10 against degree 19) for 16 instead of 73 field
  evaluations per triangle.
- Analytic formulas must be verified carefully (unit tests vs extrapolated nested quadrature) — a Phase 1 deliverable.
- The same routines serve dense, MLFMM near-field and ACA pivot evaluation.
