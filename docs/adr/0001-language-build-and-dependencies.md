# ADR 0001: C++20 core, CMake build, minimal dependencies, Python via pybind11

- **Status**: accepted
- **Date**: 2026-10-08
- **Phase**: 0

## Context
The reference implementation is Fortran 90 with MKL. We need a maintainable, extensible code base that runs headless on Linux workstations and clusters, can later use GPUs, and is driven from Python.

Alternatives: Fortran (fast, but weak abstraction and GPU ecosystem), Julia (attractive, but deployment and C++-interop story weaker for a long-lived library), Python+Numba (not fast enough for singular quadrature and MLFMM inner loops), Rust (possible, but CEM/BLAS/CUDA ecosystem thinner).

## Decision
- C++20 with Eigen for linear algebra, BLAS/LAPACK (OpenBLAS/MKL) through Eigen, OpenMP for shared memory, spdlog for logging.
- CMake ≥ 3.24 with presets; system packages first, FetchContent fallback for small dependencies; heavy ones (MKL, CUDA, HDF5) are found only.
- Python bindings with pybind11, packaged with scikit-build-core.
- Catch2 for C++ tests, pytest for Python.
- No Boost, no in-house linear algebra.

## Consequences
- One toolchain for CPU and CUDA (nvcc consumes C++20 headers).
- Dependency count stays small; adding one requires an ADR.
- Fortran reference code (if available) can be used for cross-checks but is not linked.
