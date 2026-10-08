# SpeckleBem — session guide

SpeckleBem is a headless C++20 library (Python bindings via pybind11) that computes polarised
light scattering from spheres and Gaussian rough surfaces rigorously with surface integral
equations (RWG/Galerkin, PMCHWT/ICTF/MCTF) and fast BEM compression (MLFMM first).
Authoritative documents: `docs/01_project_plan.md` (phases, definition of done),
`docs/02_architecture.md` (layers), `docs/03_theory_sie.md` (equations),
`docs/05_validation.md` (acceptance table), `docs/06_conventions.md` (binding conventions),
`docs/07_coding_guidelines.md` (style, review checklist), `docs/adr/` (decisions),
`docs/backlog.md` (work packages and their status), `docs/handover.md` (state at the last session end,
open decisions, how to continue).

## Build and test

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
cmake --preset debug   && cmake --build --preset debug   && ctest --preset debug   # -Werror + ASan/UBSan
ctest --preset release -L validation      # validation tests only (-LE validation = unit only)
scripts/format.sh                         # clang-format (+ ruff if installed) before committing
pip install -e python/[dev]               # Python package, editable (scikit-build-core)
PYTHONPATH=build/release/python:python python3 -m pytest tests/python -q   # CI-style, no install
```

Eigen, spdlog, Catch2 and pybind11 come from system packages when present, otherwise
`cmake/Dependencies.cmake` fetches them (FetchContent). BLAS/LAPACK are optional; without
them Eigen is used (slower, same results). Build directories are `build/<preset>/`.
Tests are Catch2 v3: unit tests in `tests/unit/test_<module>.cpp` (tag `[module]`),
validation tests in `tests/validation/` (label `validation`), Python tests in `tests/python/`.

## Hard rules (deviations are bugs)

- Time convention `exp(+jωt)`: outgoing waves `exp(−jkR)`, passive media `Im(ε_r) ≤ 0`,
  `Im(k) ≤ 0`; data in `n + ik` form (optics) is conjugated on import.
- SI units everywhere in C++ and Python (metres, seconds, V/m, A/m). No implicit µm/nm.
- R1 = exterior/background, R2 = object. The surface normal points **out of R2 into R1**;
  triangles are counter-clockwise seen from R1. `J = n̂ × H`, `M = −n̂ × E` on the R1 side.
- Unknown vector `x = [J; M]`, each block of length N = number of interior edges.
- Rough surfaces: mean plane `z = 0`, object occupies `z > ξ(x,y)`, light comes from `z < 0`
  travelling in `+z`. Spheres: centred at the origin, plane wave in `+z`, E along `x`.
- Layer dependencies point downwards only (docs/02). `reference/` depends only on `core` and
  `material`. Namespaces mirror directories; files `snake_case`, types `PascalCase`,
  private members end with `_`. Use `Real`, `Complex`, `Index` (int64) from `core/types.hpp`.
- New `.cpp` files are listed explicitly in `src/CMakeLists.txt` (no GLOB); new test files in
  `tests/CMakeLists.txt`. Uncomment the pre-listed placeholder lines when implementing them.
- Warning-free under the project warning set; the `debug` preset enables `-Werror` and
  sanitizers, and CI builds with `-Werror` on GCC and Clang.
- Every piece of functionality ships with tests: unit tests with fixed seeds, plus a validation
  test whenever a physical result is produced. Acceptance criteria live in
  `docs/05_validation.md`; do not loosen tolerances to make a test pass.
- Exceptions for user/programmer errors, never silent fallbacks. Logging via `SBEM_INFO` etc.,
  no `std::cout` in the library. Dependencies beyond Eigen/spdlog/BLAS/OpenMP need an ADR.
- Never commit tokens, credentials or machine-specific paths.

## Branches, commits, workflow

- `main` is always green (build + tests). All work happens on `wp/<id>-<slug>` branches
  (e.g. `wp/01-mesh-topology`), one branch per work package, merged by the coordinator only
  after a green build, green tests and a review.
- Commit messages: `<layer>: <imperative summary>` (CONTRIBUTING.md), e.g.
  `geometry: add edge topology and closedness check`. Layers: core, geometry, basis, kernels,
  material, excitation, formulation, operator, mlfmm, solver, post, reference, backend, io,
  python, tests, docs, tooling.
- Update `CHANGELOG.md` ([Unreleased]) and `docs/backlog.md` when a work package lands.
- Multi-agent setup: `.claude/agents/coordinator.md` (plans, assigns, reviews, merges),
  `.claude/agents/worker.md` (implements one WP on its own branch in a worktree),
  `.claude/agents/reviewer.md` (read-only review before merge).
