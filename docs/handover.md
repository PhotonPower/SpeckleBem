# Handover: state of the project and how to continue

Written by the coordinator agent on 2026-10-08 at the end of the first multi-agent session.
Read this first when taking over on a new machine, then `CLAUDE.md` and `docs/backlog.md`.

## Where things stand

- `main` is green (debug preset with `-Werror` + ASan/UBSan, release preset, GitHub Actions CI) and
  is the only long-lived branch. Every work package (WP) was squash-merged from a `wp/<id>-<slug>`
  branch after a review; the WP branches are kept on the remote for history only and can be deleted.
- Phase 1 (geometry and discretisation) is complete; Phase 2 (dense SIE solver) is complete up to the
  Mie validation (WP11). The dense PMCHWT + LU solver reproduces the Mie bistatic RCS of dielectric and
  Ag spheres (d = 1 µm, λ = 500 nm) to ε_rr ≈ 0.07 % at λ/13 and conserves power to 0.2 %
  (`benchmarks/results/mie_sphere_dense.md`).
- Follow-ups merged after Phase 2: WP9a (frequency only in `Problem::omega`), WP2b (graded box walls
  and bottom, closing box 7 % of the top face). WP7b (graded outer quadrature for touching pairs and
  k-aware degree selection) is the last item of this session; see the section at the end.
- Test counts on `main`: `ctest --preset debug -LE validation-large` and the release counterpart list
  ~192 tests; `validation-large` (dense 2N = 15 k solves, release only, memory-guarded, RUN_SERIAL)
  adds 3 cases, of which the Ag λ/20 sphere (2N ≈ 61 k, ~121 GB) skips below 200 GB of RAM.

## Open decisions for the project lead (recorded in docs/backlog.md)

1. **WP-F, formulation weights.** With Table 1 as transcribed in `formulation.hpp`, the combination
   coefficients a_i/η_i and b_i η_i are region-independent, so ICTF and MCTF are block-row scalings of
   PMCHWT: identical currents with the direct solver, identical systems after the Jacobi
   preconditioner. If the paper's ICTF/MCTF use region-dependent weights, correct Table 1 (and
   `formulation::recommend`) before the Phase 3 formulation study.
2. **WP12 Fresnel flat-interface check** is deferred to Phase 4 (needs > 10⁵ unknowns; dense matrix
   > 100 GB). **Ag sphere at λ/20** needs a node with ≥ 200 GB (test exists, self-skipping).
3. **WP7b default accuracy**: `OperatorOptions::target_accuracy` trades far-pair cost against
   class-boundary accuracy (1e-6: ~4× dense assembly cost, boundary error ~1e-7; 1e-5: ~2×, ~1e-5).

## Follow-ups in the backlog (todo)

- WP2c: decay-aware fine band (`box_fine_depth`) and rim-decoupled grading rows for the rough-surface
  box (Si needs fine walls down to ~3 µm; rough rims currently fall back towards the uniform box).
- Phase 3 is not yet broken down: `solver::gmres` + preconditioners (`gmres.hpp`, `preconditioner.hpp`
  declared), `Simulation` driver (`simulation.hpp`, pimpl), formulation/convergence study
  (depends on WP-F), rigorous or paraxial `GaussianBeam` already exists for excitation.
  Suggested WPs: WP13 GMRES + Identity/Diagonal preconditioners with tests vs `solve_direct`;
  WP14 `Simulation` driver + Python bindings for mesh/excitation/Simulation (docs/10);
  WP15 convergence-study script and `benchmarks/results/`; WP16 `io::result_writer` (HDF5 optional).

## How the multi-agent workflow works here

- `.claude/settings.json` starts every session as the `coordinator` agent (`.claude/agents/`), with
  `worker` (implements one WP in its own git worktree, never pushes) and `reviewer` (read-only,
  builds and tests itself) as subagents. The coordinator writes the briefs, runs the reviews, applies
  review fixes through the same worker, resolves `src/CMakeLists.txt` / `tests/CMakeLists.txt`
  conflicts at squash time, updates `CHANGELOG.md` and `docs/backlog.md`, and pushes `main`.
- Worker briefs that worked well: fully self-contained (files, formulas with signs, acceptance
  numbers, test list, constraints, report format); physics is always checked by an independent
  reference in the test (Mie currents, brute-force quadrature, hp-graded references), never by the
  implementation's own formulas.
- Pitfalls workers hit (also in the "Notes for workers" section of the backlog): Eigen's complex
  `cross()` conjugates; fetched Eigen must be a system include; Ubuntu's `libspdlog-dev` + fmt 9.1
  breaks `-Werror` on GCC 13; Catch2 3.4 needs `SKIP_RETURN_CODE 4`; GCC 13.3 `-O3` vectoriser bug in
  the box mesher (reproducer in `docs/compiler_notes/`); `-L validation` is a regex (use
  `-L '^validation$'`); running two `validation-large` cases concurrently needs > 15 GB.
- Worker turn limit: one WP7b worker exhausted its 150 turns before committing; the coordinator
  committed its worktree state as a WIP commit and re-dispatched. Prefer smaller WPs (≤ 800 lines)
  and tell workers to commit early.

## Machine notes

- The session machine had 4 cores, 15 GB, no BLAS/LAPACK (Eigen LU, 2N = 15 360 in ~5 min),
  GCC 13.3, Clang 18, CMake 3.28, Python 3.13 (NumPy, pytest installed by pip). On a machine with
  OpenBLAS the dense solves are much faster; CI (ubuntu-24.04) has OpenBLAS.
- GitHub access in the session went through the Claude GitHub connector; `gh` had no valid token.
  Issues #1–#12 and #14 are closed or resolved by merge commits (close #14 manually if still open);
  #13 (WP12) stays open as deferred.
