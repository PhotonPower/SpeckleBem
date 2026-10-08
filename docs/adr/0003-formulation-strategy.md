# ADR 0003: Formulations are scalar weight strategies; preconditioning is orthogonal

- **Status**: accepted
- **Date**: 2026-10-08
- **Phase**: 0

## Context
The paper shows that the choice among PMCHWT, ICTF, MCTF and a diagonal left preconditioner decides whether GMRES converges at all, and that the best choice depends on the material (ICTF for Si, preconditioned for Ag). We need to compare formulations on identical discretisations and switch per material.

## Decision
- `formulation::Formulation` returns only `(a1, a2, b1, b2)` from `(η1, η2)` (Table 1). JMCFIE (which needs the normal equations) is reserved as a `Kind` and will extend the interface with a flag for the N-MFIE/N-EFIE blocks when implemented.
- The left diagonal preconditioner is a `solver::Preconditioner`, not part of the formulation.
- `formulation::recommend(ε_r)` encodes the paper's finding (Re ε_r < 0 → diagonal preconditioner) and is the default `"auto"` in the Python API.

## Consequences
- Formulation studies are a loop over `Kind` with everything else fixed.
- Adding a formulation that changes the *operators* (not just weights) requires a small assembler extension, documented when JMCFIE is added.
