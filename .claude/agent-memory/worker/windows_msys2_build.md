---
name: windows-msys2-build
description: Pitfalls of the native Windows/MSYS2 build (win-release/win-debug presets)
metadata:
  type: project
---

Native Windows build since WP-W1 (2026-10-09): `win-release` = MSYS2 UCRT64 GCC 16, `win-debug` =
CLANG64 Clang 22 + libc++ + ASan/UBSan. Details and lessons are in docs/backlog.md "Notes for
workers / Windows"; the non-obvious ones:

- Git Bash's PATH has Git's `/mingw64/bin` (old libstdc++-6.dll) before `/c/msys64/ucrt64/bin`.
  Any build/test outside the presets (plain `cmake --build build/x`) dies with 0xc0000139 at
  `catch_discover_tests` time. Always use `--preset win-*` or prepend the MSYS2 bin dir.
- Windows CMake's FindPython picks the registry CPython 3.11; pin Python3_EXECUTABLE and
  PYTHON_EXECUTABLE (pybind11 2.13 uses the old FindPythonLibsNew).
- `ctest -j 12` oversubscribes OpenMP: dense validation tests 48 s in parallel vs 3-6 s serial.
- Several agents share the machine: timings (assembly, OpenBLAS LU) vary up to ~9x / ~90x
  under their builds. Before timing, poll `ps -W` until no cc1plus/clang++/specklebem_ process
  runs for ~30 s, and repeat. The session scratchpad is shared with sibling agents too: use a
  private subdirectory for logs.
- A worktree checked out before main had `.gitattributes` has a CRLF working copy (index LF);
  after merging main, delete those files and `git checkout -- .` to get LF (bash scripts).
- `-L validation -LE validation-large` selects the same tests as `-L '^validation$'`.
- Catch2 runs hidden (`[.]`) cases when a tag filter like `[kernels]` matches them; add
  `~[.]` (`'[kernels]~[.]'`) to time only the regular cases (hidden slow ones take minutes
  under ASan).
- MSYS2 Python defaults to cp1252: always `open(..., encoding='utf-8')` (or PYTHONUTF8=1).
  `open(p, 'w')` truncates before the encode error, so a failed write empties the file
  (restore with `git checkout -- <file>`). Text-mode writes emit CRLF: pass
  `newline='\n'`. An open for write can also fail with EINVAL while a concurrent build
  reads the file; just retry.
- GCC 16 (win-release) false positive: `vec.insert(end, n, x)` followed by
  `vec.insert(end, {a, b, c})` triggers -Warray-bounds in stl_uninitialized.h; push_back.
- win-debug (-O0, ASan, Eigen asserts) runs per-triangle Eigen checks at ~200 us/triangle:
  test helpers like check_box over 50k triangles take ~5 s; keep big per-triangle checks
  release-only (`#ifdef NDEBUG`) and run a small analogue in debug.

**Why:** these cost several iterations in WP-W1 and WP2c.
**How to apply:** any WP that builds or tests on this Windows machine.
