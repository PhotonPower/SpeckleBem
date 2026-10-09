# ADR 0007: File paths are UTF-8 strings

- **Status**: accepted
- **Date**: 2026-10-09
- **Phase**: 3

## Context
Public functions that touch the file system (`geometry::read_mesh`/`write_*`, `io::open_npy_directory`,
`io::open_hdf5`) take `const std::string&` paths and build `std::filesystem::path` from them. On Linux and
macOS a narrow string is interpreted as UTF-8 (the de-facto file-system encoding). On Windows, the
development platform since WP-W1, `std::filesystem::path(std::string)` interprets the bytes in the active
ANSI code page (cp1252 here), so a UTF-8 path with non-ASCII characters (e.g. `C:/Users/Jürgen/run`) is
garbled, and `path.string()` used in error messages can throw for characters outside the code page.
Python (WP14b) hands strings to pybind11 as UTF-8, so this would surface as soon as users pass their own
paths.

Alternatives: (a) `std::filesystem::path` in the public API (natural in C++, but pybind11 needs
`pybind11/stl/filesystem.h` and the Python side would see `os.PathLike` conversions with platform-specific
narrowing); (b) `std::wstring` on Windows (platform-dependent API); (c) keep `std::string`, define it as
UTF-8 everywhere and convert internally.

## Decision
- Every `std::string` that denotes a file or directory path in the public C++ API is **UTF-8 on every
  platform**. Header documentation states this.
- Conversion to `std::filesystem::path` goes through one helper in `core`:
  `std::filesystem::path core::path_from_utf8(std::string_view)` (via `std::u8string`) and its inverse
  `std::string core::path_to_utf8(const std::filesystem::path&)` (via `path::u8string()`), used for error
  messages and logging. No `path(std::string)` or `path::string()` on user-supplied paths elsewhere.
- File contents are unaffected (binary formats, UTF-8 JSON/ASCII meshes).
- No new dependency.

## Consequences
- Python passes `str`/`os.fspath()` results straight through; non-ASCII paths work on Windows.
- Existing callers with ASCII paths are unchanged. Callers that relied on ANSI-code-page narrow strings on
  Windows (none in the repository) would have to convert to UTF-8.
- Implementation: a small follow-up (WP-P1) adds the two helpers with tests (round trip of a non-ASCII
  directory name on Windows and Linux) and switches `mesh_io` and `io::result_writer`; it must land before
  WP14b exposes file I/O to Python.
