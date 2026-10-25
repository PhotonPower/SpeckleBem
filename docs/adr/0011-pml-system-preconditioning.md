# ADR 0011: Preconditioning of the PML-route systems

- **Status**: accepted 2026-10-25 (lead note `docs/lead/2026-10-22-next-work-packages.md`, decision 2)
- **Date**: 2026-10-25
- **Phase**: 5 / 8

## Context

The fast PML route (ADR 0008 amendment, WP-PML5a–c) solves the ICTF system of rough-metal open surfaces with
full GMRES and the Jacobi (diagonal) preconditioner (WP13). Measured on rough Ag (`benchmarks/results/pml_ring.md`,
WP-PML5b): 602 / 1 715 / 2 679 iterations at 2N = 2.3·10⁴ / 1.5·10⁵ / 3.0·10⁵, i.e. iterations ≈ N^0.55. At
L = 30 µm (2N ≈ 4.5·10⁵) this extrapolates to ≈ 3 400 iterations, a Krylov basis of ≈ 26 GB and a peak of ≈ 60 GB;
with the operator setup now 0.3× the dense assembly (5c) and the matvec 0.1 s at 2N = 2.3·10⁴ (5b), the solve —
matvecs plus the CGS2 orthogonalisation, which grows quadratically with the iteration count — is the dominant cost
of the planned many-solve studies (realisations, polarisations, angles). The growth is attributed to the long-range
SPP coupling of the metal surface (ADR 0010 context) that a diagonal preconditioner cannot see.

Alternatives: (i) a block-diagonal preconditioner from the near-field matrix (one block per octree leaf box, or one
level up), the `BlockDiagonalPreconditioner` planned in `solver/preconditioner.hpp` since WP13; (ii) a sparse
approximate inverse or an incomplete factorisation of the near matrix (more setup, less predictable); (iii) a
Schur-complement / Calderón-type preconditioner (Phase 8, formulation-level); (iv) restarted GMRES (cuts memory,
not iterations; convergence of ICTF with restarts is not established); (v) single-precision storage of the Krylov
basis (memory and orthogonalisation time only, WP-G2).

## Decision

1. **WP-PC1 implements (i):** `solver::BlockDiagonalPreconditioner` with one block per octree box over the J and M
   coefficients of the box's basis functions (2n_b × 2n_b), block level a parameter (leaf or one level up, both
   measured). Entries from the near-field sparse matrix of `MlfmmOperator` (`near_operator()`) plus, on the PML
   route, the near part of the PML ring (`PmlRingOperator::near_operator()`); the union must contain every same-box
   entry exactly once (tested against the dense matrix on a small case; a dense-route constructor with the same box
   partition exists for the tests). LU per block (`Eigen::PartialPivLU`), factorised in parallel; `apply` writes
   block-disjoint output, so results are bitwise independent of the thread count. Left preconditioning as for
   Jacobi; the WP13 invariance test (iterates invariant under the formulation's region-independent block-row
   scaling, PMCHWT / ICTF / MCTF) is repeated.
2. **Configuration:** a `Simulation` preconditioner option (none / diagonal / block-diagonal) that keeps
   `diagonal_preconditioner` working; Python mirror; `report()` states the choice. **The default stays Jacobi**
   until the measurement decides.
3. **Gate and default switch:** on the three rough-Ag PML sizes above (iterations, solve time, setup time, memory,
   fitted exponent), one rough-Si case and one Mie sphere case (results unchanged within the solver tolerance): the
   PML route switches to the block-diagonal default if the iterations drop by ≥ 1.5× at the largest size *and* the
   total solve time (setup + solve) drops. Otherwise the result is recorded and the lead decides; a local block
   preconditioner may give a constant factor without changing the exponent if the long-range SPP coupling drives the
   growth — then (iii) is the next lever.
4. **WP-G2 (v) is conditional** on the WP-PC1 measurement: only if the orthogonalisation is still > ≈ 20 % of the
   solve time at the largest size or the basis memory is a constraint. Panels in `std::complex<float>`, all
   accumulations in double, the CGS2 structure and the fixed row-block reduction order unchanged (bitwise
   thread-count independent; not bitwise equal to the double path); gates: converged true residual ≤ tolerance,
   iterations within +2 % of the double path, far field within 1·10⁻⁴ relative.
5. Restarted GMRES (iv) is not pursued now; (ii) and (iii) need a new decision.

## Consequences

- New operator-level code in `solver/` with a dependency on the near-field structure of `mlfmm` (block partition
  from the octree): the preconditioner is built by `Simulation` (which sees both layers) from the operator's near
  parts; `solver/` itself stays below `mlfmm/` in the layer order by receiving the blocks, not the operator.
- Setup cost: LU of ≈ N_boxes blocks of size 2n_b (n_b ≈ 30–60 per leaf): O(N · (2n_b)²), seconds; memory
  O(N · 2n_b) complex, a few per cent of the operator.
- Determinism and the Jacobi path are unchanged; every existing bitwise reference stays valid with the default.
- Revisit after the WP-PC1 measurement (gate 3) and again after Phase 8's Calderón work.
