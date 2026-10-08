# Architecture

## Guiding principles

1. **One geometric type.** Everything is a `TriangleMesh`. Sphere, rough patch, STL import are *generators*; the solver never knows which.
2. **The matrix is an operator, not an array.** `op::LinearOperator::apply` is the only thing solvers need. Dense, MLFMM, ACA, H-matrix and GPU-resident operators are interchangeable (ADR 0002).
3. **Formulation = scalar weights.** The physics of combining EFIE/MFIE for two regions (Table 1 of the paper) is a tiny strategy object; it never touches integration or compression code (ADR 0003).
4. **Hot kernels behind a backend.** Assembly, translation, matvec and field evaluation call `backend::Backend`; CPU and CUDA implementations coexist (ADR 0005).
5. **Headless.** No plotting, no GUI, no interactive state in the core. Python is the user interface; HDF5 is the exchange format.
6. **Validation first.** The dense operator and the Mie series are permanent oracles; every fast path is tested against them.

## Layer diagram

```
          ┌─────────────────────────────────────────────┐
 Python   │  specklebem (pybind11)  — NumPy / SciPy / h5py │
          └───────────────────────┬─────────────────────┘
                                  │
          ┌───────────────────────▼─────────────────────┐
 Driver   │  Simulation  (config → assemble → solve → post)│
          └──┬──────────┬───────────┬──────────┬────────┘
             │          │           │          │
 Physics  geometry   excitation  material   formulation
             │          │           │          │
          ┌──▼──────────▼───────────▼──────────▼────────┐
 Discret. │  basis (RWG)  ·  kernels (G, L, K, quadrature, singularities) │
          └───────────────────────┬─────────────────────┘
                                  │  element_blocks()
          ┌───────────────────────▼─────────────────────┐
 Operator │  op::Problem → CompressionStrategy::build()   │
          │   ├ DenseStrategy   → DenseOperator            │
          │   ├ MlfmmStrategy   → Sparse(near) + MLFMM(far)│
          │   ├ AcaStrategy     → H-matrix of ACA blocks   │
          │   └ HMatrixStrategy → H-matrix (+ H-LU precond)│
          └───────────────────────┬─────────────────────┘
                                  │  LinearOperator::apply
          ┌───────────────────────▼─────────────────────┐
 Solver   │  GMRES  +  Preconditioner (diag / block / Schur / H-LU) │
          └───────────────────────┬─────────────────────┘
                                  │  currents [J; M]
          ┌───────────────────────▼─────────────────────┐
 Post     │  fields · RCS · DRC · speckle statistics · io  │
          └───────────────────────┬─────────────────────┘
                                  │
          ┌───────────────────────▼─────────────────────┐
 Backend  │  CpuBackend (OpenMP, BLAS)  │  CudaBackend    │
          └─────────────────────────────────────────────┘
```

Dependencies point downwards only. `reference/` (Mie) depends on nothing but `core` and `material`.

## Key abstractions

### `geometry::TriangleMesh`
Vertices, triangles, derived edge topology and per-triangle cache. Invariants: closed, 2-manifold, consistently oriented, normals into R1. Generators:
- `make_icosphere` – subdivided icosahedron, projected to the sphere.
- `make_rough_surface_mesh` – structured triangulation of `z = ξ(x, y)` on an n×n grid plus side walls and bottom plate. The closing box is a numerical necessity for a two-region SIE; its influence is controlled by (i) beam tapering, (ii) a depth several absorption lengths below the mean plane and (iii) a sensitivity test (docs/05_validation.md). For lossless dielectrics in later phases an absorbing lower half or a half-space Green's function will be needed — the generator interface leaves room for both.
- `io::read_mesh` – arbitrary closed meshes.

### `basis::RwgSpace`
Maps edges → basis functions; provides values, divergence, and the per-triangle support list that element assembly iterates over. N = number of interior edges.

### `kernels::element_blocks`
Given a triangle pair and region parameters, returns 3×3 Galerkin blocks of `L_i` and `K_i`. Internally chooses the quadrature degree by proximity class and applies singularity subtraction for touching triangles. This is the only place where integration happens; all operators (dense near-field, MLFMM near-field, ACA pivot rows/columns) call it. It is therefore the first candidate for the GPU.

### `formulation::Formulation`
Returns `(a1, a2, b1, b2)` for the two coupled equations. The assembler composes
```
Z = [ (a1/η1) L1 + (a2/η2) L2        −(a1/η1) K1 − (a2/η2) K2      ]
    [  b1 η1 K1 + b2 η2 K2             (b1 η1/η1²) L1 + (b2 η2/η2²) L2 ]
```
with the sign and principal-value conventions documented in docs/03_theory_sie.md. A diagonal left preconditioner is an orthogonal choice (`solver::DiagonalPreconditioner`).

### `op::LinearOperator` and `op::CompressionStrategy`
`Problem` bundles everything needed to assemble (space, materials, formulation, excitation, kernel options). A `CompressionStrategy` turns a `Problem` into a `LinearOperator`. Composite operators (`SumOperator`) build `Z_near + Z_far`. Operators report `memory_bytes()` and `describe()` for planning and logs.

### `mlfmm::Octree` / `mlfmm::MlfmmOperator`
The octree groups RWG functions by edge midpoint. It is shared with ACA/H-matrix cluster trees in Phase 8 (the octree *is* a cluster tree with geometric admissibility). The MLFMM operator holds per-level sampling, translators, and radiation patterns; `apply` runs aggregation → translation → disaggregation for each of the four blocks and both regions.

### `solver::gmres`
Operator-only Krylov solver with left preconditioning, residual callback, full or restarted. Results carry the full history for convergence studies.

### `post::*`
Field evaluation (direct and tree-accelerated), scattering observables, speckle statistics on regular grids (FFT-based autocorrelation).

### `backend::Backend`
Minimal interface now (`gemv`); grows with each kernel that gets a GPU implementation (ADR 0005). Device buffers and explicit host↔device transfers are added in Phase 7.

### `Simulation`
Pimpl-based driver: holds mesh, excitation, config; builds `RwgSpace`, `Problem`, the chosen operator, RHS and preconditioner; runs the solver; offers `SurfaceSolution` to post-processing. This is what the Python `Simulation` class wraps.

## Data flow of one run

```
mesh ──► RwgSpace ──► Problem(materials, formulation, excitation)
                         │
                         ├─ assemble_rhs ──► b (2N)
                         ├─ assemble_diagonal ──► preconditioner
                         └─ CompressionStrategy::build ──► Z (LinearOperator)
gmres(Z, b, M) ──► x = [J; M] ──► SurfaceSolution ──► fields / RCS / statistics ──► HDF5
```

## Extension points

| Want to add… | Implement | Touch nothing else in |
|---|---|---|
| a new geometry | a function returning `TriangleMesh` | kernels, operators, solver |
| a new formulation (e.g. JMCFIE) | `Formulation` subclass (+ normal-equation support in assembler if needed) | compression |
| a new compression scheme | `CompressionStrategy` + `LinearOperator` | solver, post |
| a new preconditioner | `Preconditioner` subclass | operators |
| a GPU kernel | a method on `Backend` + CUDA implementation | algorithms |
| a new observable | function in `post/` | everything upstream |
| multiple objects / regions | extend `Problem` to a region graph; assembler composes block operators | kernels, compression |

## Memory model (orders of magnitude at N = 10⁶, λ/10 mesh)

- Unknowns: 2N complex = 32 MB per vector; GMRES with 500 iterations without restart stores ~16 GB of Krylov vectors → restart or compressed basis is a Phase 7 knob.
- `Z_near`: with ~100 elements per leaf and 27 near boxes, ≈ 2.7·10³ interactions per element × 4 blocks × 16 B ≈ 170 GB if stored naively for both regions — the dominant cost. Mitigations: skip interior near-field beyond skin depth only when provably below tolerance, single-precision storage, symmetry (`L` blocks are symmetric in Galerkin), recompute mode.
- MLFMM far-field: O(N log N) with the sampling-point count 2L² per box per level.

These numbers drive the Phase 7 priorities.
