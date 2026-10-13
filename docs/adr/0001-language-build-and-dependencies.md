# ADR 0001: C++20 core, CMake build, minimal dependencies, Python via pybind11

- **Status**: accepted
- **Date**: 2026-10-08
- **Phase**: 0

## Context
The reference implementation is Fortran 90 with MKL. We need a maintainable, extensible code base that runs headless on Linux workstations and clusters, can later use GPUs, and is driven from Python.

Alternatives: Fortran (fast, but weak abstraction and GPU ecosystem), Julia (attractive, but deployment and C++-interop story weaker for a long-lived library), Python+Numba (not fast enough for singular quadrature and MLFMM inner loops), Rust (possible, but CEM/BLAS/CUDA ecosystem thinner).

## Decision
- C++20 with Eigen for linear algebra, BLAS/LAPACK (OpenBLAS/MKL) through Eigen, OpenMP for shared memory, spdlog for logging.
- CMake ≥ 3.25 with presets (amended 2026-10-09, WP-W1: was 3.24; FetchContent `SYSTEM` and preset schema v6 need 3.25); system packages first, FetchContent fallback for small dependencies; heavy ones (MKL, CUDA, HDF5) are found only.
- Python bindings with pybind11, packaged with scikit-build-core.
- Catch2 for C++ tests, pytest for Python.
- No Boost, no in-house linear algebra.

## Consequences
- One toolchain for CPU and CUDA (nvcc consumes C++20 headers).
- Dependency count stays small; adding one requires an ADR.
- Fortran reference code (if available) can be used for cross-checks but is not linked.

## Amendment 2026-10-13 (WP23: HDF5 on the Python side; lead decision)

- **HDF5 is written and read only in Python** (`specklebem.io`, h5py, optional extra `specklebem[io]`); the C++
  library has no HDF5 dependency and keeps its dependency-free `.npy` directory writer (`io::open_npy_directory`).
  The former optional C++ HDF5 stub, its CMake option and `has_hdf5()` are removed (WP23 review fixes). "Heavy
  ones … found only" above now refers to MKL and CUDA.
- Reason: the Phase 5 DoD ("round trip HDF5 → Python (h5py) without loss") is met without a C++ dependency;
  Python is the user interface (docs/02), and h5py handles UTF-8 paths on Windows (ADR 0007).
- Consequence: C++-only users get `.npy` results; `specklebem.io.npy_to_hdf5` converts them losslessly.
