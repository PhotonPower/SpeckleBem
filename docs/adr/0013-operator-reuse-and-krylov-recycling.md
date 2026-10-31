# ADR 0013: Operator reuse across excitations and Krylov recycling for many-solve studies

- **Status**: accepted 2026-10-31 (user decision: "mach 1 und 2" — WP-SC5 kernel speed-up and the Krylov-recycling
  item proposed in ADR 0012 decision 3; lead design)
- **Date**: 2026-10-31
- **Phase**: 8 (preconditioning / solver cost), serves Phase 6 (speckle statistics) and the server studies

## Context

ADR 0012 closed the near-field-inverse family of preconditioners at its ceiling (WP-SC3/SC4: the ILUT of the near
blocks is the default of the "mlfmm" PML route; iterations 23 / 68 / 140 at 2N = 22 816 / 152 960 / 300 160, a
constant factor 6–10× over the blocks, the growth exponent ≈ 0.6 unchanged). The per-solve cost is now dominated by
the matrix-vector products (24 µm: 140 × 1.45 s = 203 s of 233 s GMRES) and, per *surface*, by the assembly: U 161 s,
16 µm 689 s, 24 µm 1 227 s against 7 s / 129 s / 333 s for setup + GMRES.

The studies that remain to be run solve **the same surface for several excitations**:

- the incidence-angle pair 0° / 30° of the memory-effect-aligned angular correlation (WP-S2 record, candidate b1,
  deferred to the server); angle sweeps for the memory effect itself;
- polarisation pairs (s / p) per surface for the co-/cross-polarised statistics;
- the beam-waist and plane-sampling checks of docs/05 on one surface.

Today every excitation is a new `Simulation`: the operator, the MLFMM tree, the near matrices and the preconditioner
are rebuilt per excitation although only the right-hand side changes (`op::assemble_rhs(problem)` is already a
separate function; the MLFMM operator, the PML ring and the ILUT do not depend on the excitation — the excitation
enters through the right-hand side, the validity checks (wavelength, background, controlled region) and the
post-processing (incident power, far-field options)).

Two levers, in order of certainty:

1. **Operator reuse** — solve several excitations on one assembled operator and preconditioner. Saves the assembly
   and the preconditioner setup per additional excitation (at 24 µm ≈ 1 330 s of 1 660 s per solve). No numerical
   risk: the right-hand side is re-assembled by the same function, the solve is the same.
2. **Krylov recycling** — reduce the iterations of the second and later solves on the same operator. Options:
   (a) the previous solution as the initial guess (zero cost; helps only when the solutions are correlated — nearby
   angles, not s / p); (b) deflation of the slowest eigencomponents with a recycled subspace (GCRO-DR, Parks et al.
   2006: k harmonic Ritz vectors of the preconditioned operator kept between solves, the new solve runs GMRES on
   the projected operator (I − C Cᴴ) M⁻¹Z with C = M⁻¹Z U orthonormal). Memory 32 k n B (k = 40 at 2N = 300 160:
   0.4 GB); cost per iteration + one projection; deterministic if the subspace update is. Whether (b) pays depends
   on the spectrum of the ILUT-preconditioned operator: WP-A2 (E3, Jacobi on the box route) found a band of 8–12
   harmonic Ritz values with |θ| < 0.1 near the origin — if such a band survives the ILUT, deflating it removes
   the slow components; if the ILUT already compresses the spectrum, recycling gains little. Not known: measure
   first, as ADR 0012 did.

Not chosen: block GMRES over all right-hand sides at once (needs all excitations in advance, memory × number of
right-hand sides, a different solver); solving the adjoint / reciprocity tricks (change the physics interface).

## Decision

1. **WP-KR1 — operator reuse and the measurement of the recycling potential.**
   - `Simulation::set_excitation(std::shared_ptr<excitation::Excitation>)`: the same checks as the constructors
     (null, wavelength relative 10⁻¹², background, controlled region, PML configuration warnings), re-assembles the
     right-hand side only (`op::assemble_rhs`, masked in the PML as today), keeps the operator, the near matrices and
     the preconditioner, and resets the solution state (`solved()` false; post-processing throws until the next
     `solve()`). `excitation()` returns the current one; every post-processing quantity that depends on the
     excitation (incident power, far-field options, DRC) reads the current excitation. Before `assemble()` it simply
     replaces the excitation. Python: `sim.set_excitation(beam)`; `sim.config` unchanged.
   - `solver::GmresParams`/`gmres`: an optional initial guess x₀ (`std::optional<VectorXc>` or a separate overload;
     the residual history then starts at |b − A x₀| / |b|; x₀ = 0 remains the default and the only automatic
     behaviour). `Simulation::solve` gets an explicit option to start from the previous solution
     (`SolveOptions::initial_guess = Previous`); never automatic.
   - Measurement (driver `benchmarks/preconditioner_study.cpp` or a new `benchmarks/recycling_study.cpp`, record
     `benchmarks/results/recycling.md`): on rough Ag PML at U and 16 µm (and Si U), one operator, the excitation
     sequence 0° s → 30° s → 0° p (the b1 pair and the polarisation pair): (i) wall time per excitation with reuse
     vs a fresh `Simulation` (assembly + setup saved); (ii) iterations of the second and third solve with x₀ = 0 and
     x₀ = previous; (iii) the harmonic Ritz spectrum of the ILUT-preconditioned operator from the first solve's
     Hessenberg (`keep_hessenberg`: the count of |θ| < 0.1 and < 0.3, σ_min); (iv) a **deflation probe**: with
     `keep_basis` of the first solve, form the k = 10 / 20 / 40 harmonic Ritz vectors, and run the second solve as
     GMRES on the projected operator (a test-grade GCRO step in the driver, not library code) — the iteration count
     it reaches is the upper bound of what WP-KR2 can deliver. Results must agree with the fresh solves to the
     solver tolerance (R, far-field L2 as in the preconditioner records).
   - **Gate for WP-KR2:** the deflation probe reduces the second solve's iterations by ≥ 1.5× against x₀ = 0 at
     16 µm (68 → ≤ 45) for the 0° / 30° pair, at k ≤ 40, with R within the tolerance. If only x₀ = previous gains
     (correlated excitations), it stays as the explicit option and WP-KR2 is not started.
2. **WP-KR2 — GCRO-DR in `solver/` (conditional on the gate):** `solver::RecycledGmres` (or `gmres` with a
   `RecycleSpace&` argument): recycled subspace U (k columns), C = Op U orthonormal; per solve: project the initial
   residual, Arnoldi on (I − C Cᴴ) Op, update U from the harmonic Ritz vectors of the augmented Hessenberg; left
   preconditioning as today; deterministic (fixed k, fixed selection rule, no thread-dependent reductions in the
   small dense parts); memory and timing in the result. Explicit option `SolveOptions::recycle = true` with
   `recycle_dimension k`; never automatic in this WP. Unit tests: identical result to full GMRES on the first solve
   (no recycle space yet) to rounding, convergence of the second solve to the same x within the tolerance, the
   projected residual orthogonal to C, determinism across thread counts. Measurement as WP-KR1 (iii)–(iv) replaced
   by the real implementation; default switch for the many-solve drivers only after the record (lead decision,
   same rule as ADR 0012 decision 5).
3. **WP-SC5** (ILUT kernel speed-up; backlog row) is independent and runs in parallel; its only interface with this
   ADR is the setup time saved once per operator.
4. **Conventions:** no new dependencies; `solver/` stays below `simulation.hpp`; results of every reused solve
   within the solver tolerance of a fresh solve (tests at CI size, records at study size); no silent default change
   (reuse is an explicit call, the initial guess and the recycling explicit options).

## Consequences

- Easier: angle and polarisation studies on one surface cost one assembly + one preconditioner setup plus the
  solves; the server queue for b1 (0° / 30° pairs) halves its assembly work; the Python sweep driver can loop over
  excitations inside one `Simulation`.
- Harder: `Simulation` gains mutable excitation state (the post-processing must be checked for cached per-excitation
  data); the GMRES interface grows (x₀, later the recycle space); the driver gains a study.
- Revisit: after WP-KR1's record — WP-KR2 only if the gate holds; the recycling default for the sweep drivers after
  WP-KR2's record. If the ILUT spectrum shows no deflatable band, the item closes with operator reuse and the
  initial-guess option, and the remaining per-solve lever is the matvec itself (Phase 7: threads / GPU).
