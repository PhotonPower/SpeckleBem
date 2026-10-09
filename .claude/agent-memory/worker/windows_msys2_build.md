---
name: windows-msys2-build
description: Pitfalls of the native Windows/MSYS2 build (win-release/win-debug presets) and of running commands from a worktree-isolated agent on Windows
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

**Why:** these cost several iterations in WP-W1.
**How to apply:** any WP that builds or tests on this Windows machine.
