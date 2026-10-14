---
name: python-hdf5-io
description: WP23 HDF5 via h5py on the Python side - MSYS2 h5py versions, UTF-8 paths, build mirror of .py files, ruff config scope, snapshot lock, attr/group order, test design
metadata:
  type: project
---

HDF5 is Python-only (lead decision 2026-10-13): `python/specklebem/io.py` (save_hdf5,
load_hdf5, npy_to_hdf5, HistoryRecorder), `Simulation.save` assigned to the pybind11 class in
`__init__.py`. The C++ `open_hdf5` stub, `has_hdf5` and `SPECKLEBEM_ENABLE_HDF5` were removed in
the WP23 review fixes (2026-10-14).

Findings (2026-10-14):
- MSYS2 UCRT64 `mingw-w64-ucrt-x86_64-python-h5py` installs h5py 3.16 / HDF5 2.1.1 (pulls
  hdf5 + aws-c-* libs); native non-ASCII (UTF-8) file names work with `h5py.File(str)` on
  Windows, no file-object workaround needed. '/' is allowed in HDF5 attribute names.
- New pure-Python package files must be added to `SPECKLEBEM_PYTHON_FILES` in
  `python/CMakeLists.txt` (configure_file mirror into `build/<preset>/python/specklebem`);
  otherwise tests with `pythonpath=build/.../python python` import the stale build copy.
- Root `ruff.toml` (line-length 100) now covers `tests/python`; `python/pyproject.toml` covers
  `python/`. A user-level ruff 0.16 (Python311 Scripts) enables many extra rules that the tree
  does not satisfy; lint only changed files with `--select E,F,W,B,UP`.
- save_hdf5 reads the Simulation state through `Simulation._save_snapshot()` (one `locked()`:
  last solve + params, currents if `solved()`, rhs if `assembled()`); never wrap Simulation
  calls in `except RuntimeError` (it swallows "Simulation busy" / re-entrant errors).
- A mapping written as attrs + sub-groups reads back with attributes first, then sub-groups:
  HDF5 keeps the namespaces apart, so mixed insertion order is not preserved.
- `tempfile.mkstemp` creates 0o600 files; io.py chmods the temp file to 0o666 & ~umask
  (umask read once at import) before the rename.
- The npy fixture (`npy_dir`, session scope) lives in `tests/python/conftest.py`.
- CI requires h5py via `SPECKLEBEM_REQUIRE_H5PY=1` (like SPECKLEBEM_REQUIRE_SCIPY).

**Why:** future WPs touching Python I/O or adding Python modules hit the same traps.
**How to apply:** check these before adding Python files or extending the HDF5 layout
(bump `FORMAT_VERSION` for incompatible changes). See [[python-bindings]].
