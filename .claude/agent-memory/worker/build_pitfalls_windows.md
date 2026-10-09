---
name: build-pitfalls-windows
description: Windows/MSYS2 build and test pitfalls met while implementing WPs (GCC false positives, sanitizer-run timings, worktree sandbox limits)
metadata:
  type: project
---

- UCRT64 GCC 16 `-Wnull-dereference` false positive: assigning a scalar into a freshly
  allocated Eigen vector (`g = VectorXc::Zero(m); g(0) = beta;`) warns inside `<complex>`.
  Workaround: build it in one expression (`g = VectorXc::Unit(m, 0) * beta;`).
- A helper used only under `#ifdef NDEBUG` (validation-large cases SKIP in debug) trips
  clang `-Wunused-function` in win-debug: mark it `[[maybe_unused]]`.
- win-debug is -O0 Eigen + ASan: dense assembly of icosphere n = 2 (2N = 960) ~4.5 s, and 200
  GMRES iterations at n = 960 take ~15 s. For unit tests use icosphere n = 1 at lambda = 1 um
  (same h/lambda, 2N = 240, ~1-1.5 s per assembly). Split heavy cases into separate TEST_CASEs.
- Timings under `ctest -j` or while other workers build fluctuate 5x; rerun serially before
  concluding a test is too slow.
- `ctest -R` is case-sensitive: name test cases so the brief's `-R <module>` filter matches.
- Worktree sandbox: Bash commands with shell variables as sed args, `cd && exe "[tag]" > file`,
  or heredoc + mv chains are refused. Use Edit/Write for file changes; to see Catch2 INFO
  values, temporarily switch INFO to WARN (sed with literal path) and run `ctest -V -R ...`.
- No `python3` on PATH in Git Bash (Windows Store alias only).

**Why:** cost several rebuild cycles in WP13 (2026-10-09).
**How to apply:** check these before the first debug build of a new WP. See [[gmres-observations]].
