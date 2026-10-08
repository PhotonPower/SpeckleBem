# GPU strategy

GPU support is an **opt-in build** (`SPECKLEBEM_ENABLE_CUDA`) that never changes results beyond the tolerances in docs/05_validation.md. The CPU path remains the reference and is always built.

## Where the time goes (expected, from the paper and MLFMM literature)

| Stage | CPU cost | GPU suitability |
|---|---|---|
| Near-field assembly (`element_blocks` over leaf-neighbour pairs) | O(N · n_near) with heavy quadrature, 30–60 % of setup | excellent: independent pairs, high arithmetic intensity |
| Radiation-pattern precomputation | O(N · 2L²) | excellent |
| Translation stage | O(#interaction pairs · 2L²) per matvec, typically the largest matvec share | excellent: batched pointwise complex multiply-add |
| Interpolation / anterpolation | O(#boxes · L² log L) | good (FFT in φ, sparse in θ) |
| Near-field matvec (block-sparse) | memory-bound SpMV | good (cuSPARSE BSR) |
| GMRES orthogonalisation | O(iter · N) | good (cuBLAS), but Krylov basis memory limits |
| Field evaluation on large grids | O(N · N_points) or tree-accelerated | excellent |

## Design

- `backend::Backend` interface. Algorithms call `backend->kernel(...)`; the CPU implementation uses OpenMP/BLAS, the CUDA one cuBLAS/cuSPARSE/custom kernels.
- **Explicit data residency.** `DeviceBuffer<T>` with `to_device()`, `to_host()`; a `LinearOperator` can be "device resident" so that the whole GMRES loop runs on the GPU and only the residual norm comes back per iteration.
- **Structure of arrays** for patterns and translators (`[level][box][direction][component]`), 64-byte aligned, so the same layout serves CPU SIMD and GPU coalescing.
- **Mixed precision**: patterns and translators in `float2` when the requested accuracy is ≤ 3 digits and the matvec-vs-dense test passes; near-field blocks and Krylov vectors in double.
- **Batching**: translation grouped by unique translator (≤ 316 per level) → one batched GEMM-like kernel per group.
- **Multi-GPU** is out of scope until single-GPU is validated; the level-wise decomposition of MLFMM maps naturally to it later.

## Order of implementation (Phase 7)

1. `gemv` / dense operator on device (sanity, tooling, tests).
2. Near-field assembly kernel (biggest setup win; quadrature tables in constant memory, one thread block per test triangle).
3. Translation kernel.
4. Aggregation / disaggregation including interpolation.
5. Device-resident GMRES.
6. Field evaluation.

Each step: CPU/GPU comparison test (rel < 1e-4 in double), timing entry in `benchmarks/results/`.

## Non-CUDA options

The backend interface is vendor-neutral; a SYCL or HIP backend would be a second implementation of the same interface. Not planned, but not precluded.
