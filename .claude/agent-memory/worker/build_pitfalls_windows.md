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
- Since WP7b the OperatorOptions defaults (outer_grading_levels = 4, target_accuracy = 1e-5)
  make cheap unit-test quadrature 2-10x slower under ASan: test helpers that lower the degrees
  must also set `outer_grading_levels = 0; target_accuracy = 0.0;` (as `wp7_options()`). With
  that, a 2N = 240 driver case (assembly + 56 GMRES it or LU) is ~1 s serial in win-debug.
- For exact-comparison tests (sparse vs dense entries, refactoring checks) accuracy is
  irrelevant: WP7 scheme with quad_degree_far = 1, near = 2, sing = 2 (lowest positive-interior)
  is ~5x cheaper than wp7_options under ASan (icosphere n = 1 dense ~0.3 s; WP19a).
- Timings under `ctest -j` or while other workers build fluctuate 5x; rerun serially before
  concluding a test is too slow.
- `ctest -R` is case-sensitive: name test cases so the brief's `-R <module>` filter matches.
- Use Edit/Write for file changes; to see Catch2 INFO values, temporarily change INFO to WARN
  with Edit and run `ctest -V -R ...`.
- No `python3` on PATH in Git Bash (Windows Store alias only).
- `scripts/format.sh` uses the clang-format from the registry Python's Scripts dir; `ruff` is not
  installed, so Python files must be kept within 100 columns by hand (python/pyproject.toml).
- Test helper executables built in `build/<preset>/tests/` can be located from Python via the
  build-tree package: `Path(specklebem.__file__).parents[2] / "tests"` (WP16 npy fixture).
- `std::istreambuf_iterator` file reads trip GCC 13/14 `-O3 -Wnull-dereference` on Linux CI
  (not GCC 16 here): read files with `fs::file_size` + `in.read` instead.
- GCC 13 `-O3 -Wnull-dereference` also fires on `std::vector<T> v(static_cast<size_t>(i) + 2, x);
  v[i] = ...` with a signed `int i` (path i = -2 -> size 0 -> null data). Do the size/index
  arithmetic in `std::size_t` from non-negative inputs (WP18 test_plane_wave, CI-only failure).
- Reproducing Linux CI GCC 13 diagnostics locally: winlibs ships a standalone GCC 13.3.0 UCRT
  zip (github.com/brechtsanders/winlibs_mingw, release `13.3.0posix-11.0.1-ucrt-r1`); unzip it
  into the scratchpad and compile single TUs with the CI flags + `-Werror`, taking include
  paths from `build/win-release/compile_commands.json` (`-isystem` for `_deps`). MSYS2's
  repo only keeps GCC >= 14.2 packages. It reproduced the WP18 failure exactly.
- `std::filesystem::rename(tmp, target)` replaces an existing target on Windows with both
  win-release (libstdc++) and win-debug (libc++): temp-file + rename overwrites work (WP16).
- Python edit scripts written through a heredoc: put C++ snippets with backslash escapes in
  raw strings (`r'''...'''`), otherwise `\x..`/`\0` become real bytes and matches fail.

**Why:** cost several rebuild cycles in WP13 (2026-10-09).
**How to apply:** check these before the first debug build of a new WP. See [[gmres-observations]].
