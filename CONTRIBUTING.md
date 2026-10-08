# Contributing

## Workflow

1. Every change goes through a branch and a pull request against `main`; CI must be green.
2. Work is organised by the phases in `docs/01_project_plan.md`. Open an issue per work package and reference it in the PR.
3. Design changes that affect more than one layer get an ADR in `docs/adr/` (copy `template.md`).
4. New source files are added explicitly to `src/CMakeLists.txt` (no globbing).
5. New functionality ships with tests: unit tests for components, a validation test when a physical result changes.

## Code style

- C++20, `clang-format` (config in repo), `clang-tidy` clean for new code. Run `scripts/format.sh` before committing.
- Python: `ruff`, type hints, NumPy docstrings.
- Details: `docs/07_coding_guidelines.md`.

## Commit messages

`<layer>: <imperative summary>` – e.g. `mlfmm: add diagonal translation operator`, `docs: clarify region naming`.

## Secrets

Never commit tokens, credentials or machine-specific paths. `.gitignore` blocks `*token*`; CI uses repository secrets.
