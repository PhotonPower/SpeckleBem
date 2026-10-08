# ADR 0005: Execution backend interface for CPU and GPU

- **Status**: accepted
- **Date**: 2026-10-08
- **Phase**: 0 (interface), 7 (CUDA implementation)

## Context
GPU acceleration is a stated later goal. Retrofitting it into algorithm code is expensive; designing every algorithm for the GPU from day one slows Phases 1–6.

## Decision
- A small `backend::Backend` interface that grows kernel by kernel. Phase 0 has only `gemv`; Phase 4 adds the MLFMM stage kernels; Phase 7 adds device buffers and the CUDA implementation.
- Algorithms are written so that their hot loops are already "kernel-shaped": data in contiguous SoA arrays indexed by (level, box, direction), no per-element allocation, no virtual calls inside loops.
- CUDA is an opt-in CMake option; `.cu` sources live only under `src/backend/gpu/`.
- Results must agree with the CPU path to 1e-4 (double); single precision only with a passing validation test.

## Consequences
- Early phases pay only a layout discipline, not a GPU implementation.
- Switching vendors (HIP/SYCL) means a second backend, not a rewrite.
