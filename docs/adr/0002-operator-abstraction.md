# ADR 0002: The impedance matrix is an abstract linear operator

- **Status**: accepted
- **Date**: 2026-10-08
- **Phase**: 0

## Context
We will use several representations of `Z`: dense (validation), MLFMM, later ACA/H-matrices, and GPU-resident variants. Solvers and post-processing must not depend on which one is in use. The paper's split `Z = Z_near + Z_far` must be expressible.

## Decision
- `op::LinearOperator` with `apply(x, y)`, `rows()`, `cols()`, `describe()`, `memory_bytes()`.
- `op::CompressionStrategy::build(Problem) → shared_ptr<LinearOperator>` as the factory interface; one strategy per compression method.
- Composition through `SumOperator` (and, later, `ScaledOperator`, `BlockOperator` for the 2×2 J/M block structure).
- Preconditioners are a separate interface (`solver::Preconditioner`) and may themselves wrap operators (H-LU).
- Element integration (`kernels::element_blocks`) is shared by all strategies so that operator comparisons isolate the compression error.

## Consequences
- Solvers are oblivious to compression and backend.
- Operators can be exposed to Python/SciPy as `LinearOperator`.
- Block structure (J/M, regions) is handled by the assembler composing operators, which costs a few extra vector adds per matvec — acceptable.
