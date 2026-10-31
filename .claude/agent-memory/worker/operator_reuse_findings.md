---
name: operator-reuse-findings
description: WP-KR1 set_excitation / GMRES x0 / recycling study - overload trap (Eigen expr -> std::function), 30 deg dark band at U, bitwise reuse, ILUT spectrum has no small harmonic Ritz values
metadata:
  type: project
---

WP-KR1 (2026-10-31, branch wp/kr1-operator-reuse), `Simulation::set_excitation`,
`solver::gmres(A, b, x0, M, p, cb)`, `SolveOptions::initial_guess`, driver
`benchmarks/recycling_study.cpp`, record `benchmarks/results/recycling.md`.

- **Overload trap**: an overload `gmres(A, b, M, p, const VectorXc& x0, cb)` is ambiguous for
  `VectorXc::Zero(n)`: Eigen expressions are callable with (Index, Index), so they convert to
  `std::function<void(int, Real)>` too. x0 went to the third position instead.
- Reuse is **bitwise**: set_excitation + solve equals a fresh Simulation (rhs, its, currents, R,
  far field, DRC) on the tiny mlfmm PML case, the dense box case and at study size (smoke 2N
  16 016: dR 0, l2 0). Assembly is deterministic, so bitwise asserts are safe.
- **Dark band at 30 deg** with u_case w0 = lambda: U 1.44e-4 (> 1e-4 target, grazing waves of
  the cut spectrum since w0 cos theta < lambda_1), 16 um 5.5e-5 (passes). Study runs U / Si U
  with --dark-band-tolerance 2e-4 (documented). `--estimate-only` prints DARK lines cheaply.
- x0 = previous for s -> s(30 deg) or s -> p is useless (r0 1.2-1.3, its +1).
- Smoke (rough Ag L 2.5 um, 2N 16 016): ILUT harmonic Ritz values all |theta| >= 0.69 (19 its),
  parent's >= 0.31 (140 its); deflation probe with all 19 Ritz vectors: 19 its (no gain).
- GMRES x0 = the solution with a looser tol -> 0 iterations; with the same tol the explicit
  residual can exceed the Arnoldi estimate, so tests use a looser tol for the 0-it check.
- Review fixes (2026-10-31, wp/kr1-operator-reuse-review-fixes): two excitations at
  lambda (1 -+ 0.8e-12) both pass the constructor check but after assemble() op::validate
  refuses the second (omega kept, rel 1e-12); unit-tested. `save_hdf5` takes the excitation and
  the dark-band ratio from `_save_snapshot()` (same lock as currents/rhs): any new
  per-excitation file attribute must come from the snapshot too. A duck-typed proxy whose
  `_save_snapshot()` calls `set_excitation()` right after emulates the race deterministically.
- spdlog's `fmt::format` is not used anywhere in the library (spdlog may be built on
  std::format); prepare log strings with ostringstream when they must be built before a commit.
- Recycling-run assemblies were 5-16 % slower than the same-day WP-SC4 idle SETUP rows
  (compare without ILUT setup: the study driver's `assemble` excludes the preconditioner, the
  recycling driver's includes it). Cause unknown; quote both when comparing timings.

**Why:** WP-KR2 / sweep drivers decide on recycling from these numbers.
**How to apply:** reuse set_excitation for angle/polarisation pairs; do not expect x0 gains for
uncorrelated excitations. See [[ilut-preconditioner-findings]], [[pml-fresnel-findings]].
