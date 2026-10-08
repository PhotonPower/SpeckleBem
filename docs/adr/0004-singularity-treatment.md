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
  below 5e-10, so touching blocks are averaged over both orderings only above
  |k| h = `symmetrize_touching_above_kh` (default 1; there the remainder error, ~(|k| h)^4,
  dominates). The WP7 scheme (one Dunavant outer rule, both orderings averaged: ~2e-4 for L and
  ~1e-2 of the K block) remains available with `outer_grading_levels = 0`. Near and far pairs:
  the Dunavant degree is chosen per pair from an empirical error envelope in
  kappa = sqrt((h/D)^2 + (0.15 |k| h)^2) so that the block error is below `target_accuracy`
  (default 1e-6) within [`quad_degree_far`, `quad_degree_near`] = [3, 19]; at the class boundary
  of the n = 4 Mie mesh this gives 6e-8 (n = 1.5) and 4e-7 (Ag) instead of 4e-3 and 2.5e-2 with
  the fixed degrees 3 / 8. Cost (release, icosphere n = 3 at 500 nm, |k| h ~ 1-5, so touching
  blocks are averaged; mean per `element_blocks` call, WP7 options -> defaults): identical
  30 -> 130 µs (~4x), shared edge 60-75 -> 175-235 µs (2.4-3.2x), shared vertex 57 -> 90-97 µs
  (1.6x), near 7-9 -> 15-21 µs (2-3x), far 1.0 -> 5.5-5.9 µs (5-6x; 9.8 µs, 10x, for the Ag
  interior, where the selection resolves blocks that the exp(-Im k R) decay makes negligible).
  Far pairs dominate dense assembly: the n = 3 icosphere (2N = 3840, 3 threads) takes 6.3 s
  (n = 1.5), 7.9 s (Ag) and 14 s (Si) instead of 2.7 s. Far-pair cost by target (vacuum / n = 1.5
  / Ag interior): 1e-4 1.8 / 2.1 / 3.9 µs, 1e-5 2.6 / 5.0 / 6.6 µs, 1e-6 5.0 / 5.1 / 10.4 µs.
  Touching outer points at level 4: 384 / 288 / 100 (identical / shared edge / shared vertex)
  instead of 25 (Dunavant degree 10); worst touching error by level 0..4: 2.7e-2, 8.8e-3,
  5.6e-5, 3.1e-7, 3.0e-9.
- Analytic formulas must be verified carefully (unit tests vs extrapolated nested quadrature) — a Phase 1 deliverable.
- The same routines serve dense, MLFMM near-field and ACA pivot evaluation.
