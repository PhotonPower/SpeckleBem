---
name: python-bindings
description: pybind11 module pitfalls in SpeckleBem (warning flags order, py::vectorize const-ref error, read-only Eigen views, running pytest)
metadata:
  type: project
---

- The `_specklebem` module links `SpeckleBem::warnings` (since WP14b1). UCRT64 GCC 16 at -O3
  then emits `-Wnull-dereference` / `-Wmaybe-uninitialized` false positives inside
  pybind11/numpy.h, Eigen and libstdc++. `target_compile_options(... -Wno-...)` does NOT work:
  the target's own options come before the interface options of linked libraries on the command
  line. Source-file `COMPILE_OPTIONS` come last, so the suppression lives in
  `set_property(SOURCE ... APPEND PROPERTY COMPILE_OPTIONS ...)` in python/CMakeLists.txt.
  Clang (win-debug) needs no suppression.
- `py::vectorize` on a lambda with a `const T&` first parameter does not compile with pybind11
  2.13 (reinterpret_cast casts away const in numpy.h). Broadcast by hand with
  `numpy.broadcast_arrays` + c_style `array_t::ensure` (also lets you release the GIL).
- Returning `const Vertices&` with `reference_internal` gives a read-only zero-copy view whose
  base keeps the C++ object alive (tested after `del mesh`).
- pybind11 default translation already maps invalid_argument/domain_error -> ValueError and
  runtime_error/logic_error -> RuntimeError; out_of_range -> IndexError unless a local
  translator is registered (bindings.cpp maps it to ValueError).
- New pure-Python files must be listed in `SPECKLEBEM_PYTHON_FILES` (python/CMakeLists.txt) or
  the build-tree package misses them.
- Docstrings (WP14b1f): every bound object has a NumPy-style docstring with units; keep that
  for new bindings (tests assert units in `__doc__`). clang-format does not wrap `R"doc(...)"`
  raw strings and ruff is not installed on the Windows machine: check <= 100 columns and
  trailing spaces by hand (awk 'length > 100').
- Non-finite inputs: `common.hpp::require_all_finite` before releasing the GIL; use it for new
  point/angle evaluation bindings.

**Why:** found while implementing WP14b1 (2026-10-09); each cost a rebuild cycle.
**How to apply:** check before adding bindings in WP14b2 and later. See [[build-pitfalls-windows]].
