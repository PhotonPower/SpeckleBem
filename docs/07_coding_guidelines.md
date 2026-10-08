# Coding guidelines

## C++

- **Standard**: C++20. No compiler extensions. Builds warning-free under the project warning set (`cmake/CompilerWarnings.cmake`); CI enables `-Werror`.
- **Style**: `.clang-format` (Google base, 4 spaces, 100 columns). Run `scripts/format.sh`.
- **Headers**: `#pragma once`, one public type per header where practical, `@file` doc comment stating the purpose and the equations/papers implemented. Include order: project, third-party, system.
- **Namespaces** mirror directories (docs/06_conventions.md).
- **Types**: use the aliases in `core/types.hpp` (`Real`, `Complex`, `Index`, `VectorXc`, …). Never `int` for sizes/indices that can exceed 2³¹ (N can reach 10⁷).
- **Ownership**: value semantics for data, `std::unique_ptr` for pimpl, `std::shared_ptr` only for operators shared between composites. No raw owning pointers. `Problem` holds non-owning pointers to objects whose lifetime the `Simulation` guarantees.
- **Errors**: exceptions (`std::invalid_argument`, `std::runtime_error`, `std::logic_error`) for programmer/user errors; never silent fallbacks. Physics tolerances exceeded → warning in the log + flag in the result struct.
- **Parallelism**: OpenMP for shared-memory loops; parallel regions live in the backend or in assembly/translation loops, not in low-level kernels. No `std::thread` ad hoc. Deterministic reductions where results feed tests.
- **Numerics**: no premature optimisation in kernels before the dense oracle exists. Keep a slow reference path (`_reference` suffix or `#ifdef SPECKLEBEM_REFERENCE_PATH`) wherever an optimised path replaces it.
- **Logging**: `SBEM_INFO` etc.; timings via `ScopedTimer`. No `std::cout` in the library.
- **Dependencies**: Eigen, spdlog, BLAS/LAPACK, OpenMP; optional CUDA, HDF5. Anything else needs an ADR.
- **GPU code**: `.cu` files only under `src/backend/gpu/`; no CUDA types in public headers.

## Python

- Package `specklebem` wraps `_specklebem`. Thin, documented wrappers; NumPy in/out, zero-copy where Eigen allows (`py::array` ↔ `Eigen::Ref`).
- `ruff` formatting, type hints, NumPy-style docstrings. Public API listed in `docs/10_python_api.md`; changes there are reviewed like C++ API changes.
- Tests with pytest; heavy numerics are tested in C++, Python tests check plumbing and statistics helpers.

## Testing

- Unit tests next to the layer they cover (`tests/unit/test_<module>.cpp`), Catch2 v3, tags `[module]`.
- Validation tests labelled `validation`; large ones `validation-large`.
- A bug fix comes with a regression test.
- Random inputs use fixed seeds.

## Documentation

- Equations in docs use the conventions of `06_conventions.md`; cite the source (paper, equation number).
- Public headers are the API documentation (Doxygen-style comments); generate with `doxygen` when needed (config to be added in Phase 6).
- Every design decision affecting more than one layer → ADR.

## Review checklist

1. Does it build with `-Werror` on GCC and Clang?
2. Tests added / updated? Do validation tolerances still hold?
3. Does the change respect layer dependencies (downward only)?
4. Units, time convention, normal direction, region naming correct?
5. Memory: any new O(N²) allocation?
6. Docs / CHANGELOG updated?
