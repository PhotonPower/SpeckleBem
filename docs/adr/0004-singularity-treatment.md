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
- Near-field accuracy to ~1e-6 relative with moderate quadrature cost.
- Analytic formulas must be verified carefully (unit tests vs extrapolated nested quadrature) — a Phase 1 deliverable.
- The same routines serve dense, MLFMM near-field and ACA pivot evaluation.
