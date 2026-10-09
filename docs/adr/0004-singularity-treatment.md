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
  near degree); `quad_degree_near` is also the right-hand-side rule. Shared-edge / shared-vertex blocks are
  averaged over both orderings above `symmetrize_touching_above_kh = 1`.

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
  holds for shared edges with dihedral angles >= ~90 degrees. Sharper folds are near-singular
  (the far vertex of one triangle comes close to the other, away from the graded edge) and the
  error depends on the triangle shapes, dominated by K: 60 degrees 1.5e-8 to 3.4e-7 at level 4
  (3.4e-9 at level 6), 30 degrees 6.8e-8 to 4.5e-5 at level 4 (up to 3.7e-6 at level 6 for a
  skewed pair). Sharper folds therefore need `outer_grading_levels` 5 to 6, and below ~45
  degrees a grading towards the near-singular vertex (follow-up). `target_accuracy` does not
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
  defaults): identical 30 -> 125 µs (~4x), shared edge 55-75 -> 175-200 µs (2.4-3.5x), shared
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
  points: n = 4 0.073 % / 0.074 % (n = 1.5; WP7 0.074 % / 0.076 %) and 0.078 % / 0.057 % (Ag;
  unchanged), n = 3 0.263 % / 0.274 % (n = 1.5), 0.348 % / 0.256 % (Ag) and 0.554 % / 0.396 %
  (Si; WP7 0.58 % / 0.42 %), while the Si power-balance defect drops from 0.42 % to 0.007 %
  (`benchmarks/results/mie_sphere_dense.md`).
  Touching outer points at level 4: 384 / 288 / 100 (identical / shared edge / shared vertex)
  instead of 25 (Dunavant degree 10); worst touching error by level 0..4: 2.7e-2, 8.8e-3,
  5.6e-5, 3.1e-7, 3.0e-9.
- Analytic formulas must be verified carefully (unit tests vs extrapolated nested quadrature) — a Phase 1 deliverable.
- The same routines serve dense, MLFMM near-field and ACA pivot evaluation.
