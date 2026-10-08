# Project plan

**Objective.** A headless C++20 library with Python bindings that computes, rigorously, the scattering of polarised light from (a) spheres as the Mie reference and (b) planar patches with prescribed Gaussian roughness, scaling to millions of unknowns via MLFMM; later extensible to arbitrary closed surfaces, other compression schemes and GPU execution.

**Non-goals (for now).** GUI, multilayer substrates, periodic boundary conditions, time-domain, inverse problems. These are kept *possible* by the architecture but are not scheduled.

The plan is organised in phases. Each phase has deliverables, a *definition of done* (DoD) with measurable criteria, and the ADRs it depends on. Phases 1–3 may overlap; Phase 4 (MLFMM) starts only when Phase 2's dense solver is validated, because the dense operator is the oracle for the compressed one.

---

## Phase 0 — Foundation (this scaffold)

**Deliverables**
- Repository layout, CMake build with presets, dependency resolution, warnings-as-errors preset, CI (GCC + Clang, Release + Debug/ASan).
- `core/`: types, constants, logging, timers. `material/`: complex permittivity, reference Ag/Si, dispersive interpolation. `formulation/`: PMCHWT/ICTF/MCTF weights. `operator/`: `LinearOperator`, dense and sum operators. `backend/`: CPU backend.
- pybind11 module skeleton, Python package, pytest.
- Documentation set and ADR 0001–0005.

**DoD**: `cmake --preset release && cmake --build --preset release && ctest` green locally and in CI; `import specklebem` works.

## Phase 1 — Geometry and discretisation

**Deliverables**
- `geometry::TriangleMesh`: topology (edges, edge→triangle), orientation check and repair, closedness check, bounding box, quality report.
- `make_icosphere` / `make_sphere`.
- Rough-surface generator (`generate_gaussian_height_map`, Eq. 8–9 via FFT convolution; direct convolution as test oracle), closed box mesh (`make_rough_surface_mesh`), import of external height maps.
- Mesh import/export: STL (binary/ASCII), OBJ, Gmsh v4.1 ASCII.
- `basis::RwgSpace`, Dunavant triangle rules (degree 1–20), Gauss–Legendre.
- Analytic static integrals (`kernels/singularity.hpp`) for identical / edge-adjacent / vertex-adjacent triangle pairs.

**DoD**
- Euler characteristic 2 for sphere and closed box meshes; all edges have exactly two triangles.
- Generated height maps reproduce target σ within 2 % and Lc within 5 % (ensemble of 20 seeds).
- RWG: surface divergence is ±l/A; normal continuity across the shared edge; Σ over support integrates to l (unit test against closed form).
- Static integrals agree with 20-point Dunavant on well-separated triangles to 1e-12, and converge where quadrature does not (coincident triangles).
- Rough-surface mesh for L = 10 µm, h = 50 nm generates in < 5 s.

## Phase 2 — Dense SIE solver and Mie validation

**Deliverables**
- `kernels::element_blocks`: Galerkin blocks of `L_i` and `K_i` for a triangle pair, with singularity subtraction (ADR 0004) and adaptive quadrature degree by proximity.
- `op::DenseStrategy`: full 2N×2N assembly for both regions combined by the formulation weights; `assemble_rhs`; `assemble_diagonal`.
- `solver::solve_direct` (dense LU; Eigen with BLAS acceleration, LAPACK `zgesv` optional later).
- `reference::MieSolution` (Bohren–Huffman, `exp(+jωt)` adapted; near/far field, bistatic RCS, cross sections, Wiscombe n_max).
- `post::scattered_field`, `far_field`, `bistatic_rcs` (direct evaluation).
- `excitation::PlaneWave`.

**DoD** (see docs/05_validation.md for the exact metrics)
- Dielectric sphere (n = 1.5, d = 1 µm, λ = 500 nm): bistatic RCS error ε_rr < 1 % vs Mie at λ/10 mesh; error decreases monotonically with refinement.
- Ag sphere d = 1 µm: ε_rr < 1 % at λ/20 mesh with PMCHWT via direct solver.
- Operator symmetry/reciprocity checks: `Z_PMCHWT` is complex-symmetric up to quadrature error (< 1e-6 relative).
- Flat interface limit: a large flat box illuminated by a tapered beam reproduces Fresnel reflection coefficient within 1 % for p and s at 0° and 45°.

## Phase 3 — Iterative solution and formulation study

**Deliverables**
- `solver::gmres` (complex, full and restarted, callback for residual history), `DiagonalPreconditioner`, `IdentityPreconditioner`.
- Convergence-study script reproducing the qualitative behaviour of Fig. 2(a,b): PMCHWT stalls, ICTF converges for Si, diagonal-preconditioned ICTF/MCTF converges fastest for Ag.
- Automatic recommendation (`formulation::recommend`) wired into `Simulation`.
- `excitation::GaussianBeam` (paraxial with first-order longitudinal components and `θ_in`; the rigorous angular-spectrum beam follows in Phase 5).

**DoD**
- GMRES reproduces the direct solution to residual tolerance on a 20 k-unknown problem.
- Documented iteration counts for Ag/Si, L = 10 µm, σ = 50 nm, Lc = 500 nm (dense operator, ~2–5·10⁴ unknowns) stored under `benchmarks/results/`.

## Phase 4 — MLFMM

**Deliverables**
- `mlfmm::Octree`: Morton ordering, adaptive leaf size (~100 elements, ≥ λ/4 box), near / interaction lists.
- Plane-wave expansions on the unit sphere (Gauss–Legendre in θ, uniform in φ, L from the excess-bandwidth formula with configurable accuracy), radiation/receiving patterns of RWG functions for `L` and `K` operators in both regions.
- Upward pass (interpolation + phase shift), diagonal translation with precomputed translators per level and relative offset, downward pass (anterpolation).
- `op::SparseOperator` for near interactions; `MlfmmOperator = Z_near + Z_far`.
- Memory and timing report per level; `MlfmmStrategy` selectable from `Simulation`.
- Treatment of the lossy interior wavenumber (docs/04_theory_mlfmm.md §"Lossy media"): per-level validity check of the expansion and automatic fallback to direct near-field evaluation for levels where the expansion is not accurate to the target.

**DoD**
- MLFMM matvec vs dense matvec: relative error < 1e-3 (default) and < 1e-5 with `accuracy_digits = 5`, on sphere and rough-surface meshes with 2·10⁴–10⁵ unknowns, for both Si and Ag.
- Ag sphere d = 4 µm, λ/27 mesh (393 k unknowns): ε_rr ≤ 0.5 % in both scattering planes (paper: 0.26 % / 0.37 %).
- Scaling exponent γ (time per solve vs N) ≤ 1.5 for Si rough surfaces, ≤ 2.0 for Ag, over N = 5·10⁴ … 10⁶.
- Si 30×30 µm², σ = 100 nm, λ/10 mesh (≈ 2.2 M unknowns) solves on a 256 GB node.

## Phase 5 — Post-processing and I/O

**Deliverables**
- Field evaluation on planes, cylinders and arbitrary point clouds; MLFMM-accelerated evaluation for large observation grids.
- Bistatic RCS, DRC on a cylinder (Fig. 9), co/cross-polarised intensities, total scattered power.
- Rigorous Gaussian beam as an angular-spectrum superposition of plane waves (exact Maxwell solution) replacing the paraxial model; `θ_in` support.
- HDF5 result writer (mesh, currents, fields, metadata, solver history); `.npy` fallback.

**DoD**
- Near field of the Ag sphere in the xz-plane matches Mie with the error profile of Fig. 1(b) (relative error < 0.1 along x, < 1 in the shadow).
- Rigorous beam satisfies `∇×E = −jωμH` and `∇·E = 0` at sampled points to 1e-8 relative.
- Round trip HDF5 → Python (`h5py`) without loss.

## Phase 6 — Python API and speckle statistics

**Deliverables**
- Full pybind11 coverage: meshes (NumPy zero-copy), excitations, `Simulation`, operators (`matvec` exposed for SciPy `LinearOperator` interop), post-processing, Mie.
- `post::speckle_statistics`: contrast (Eq. 12), PDF (Eq. 10), angular correlation (Eq. 11), intensity autocorrelation and `μ_A` (Eq. 13), speckle size, comparison helper for Eq. 14.
- Examples reproducing Figs. 5–9 of the paper as Python scripts under `examples/python/`.
- API documentation (docs/10_python_api.md finalised, docstrings).

**DoD**
- Fully developed speckle at z = −1 mm shows negative-exponential PDF (KS test p > 0.05 against Eq. 10).
- Contrast and angular correlation decrease monotonically with σ over 50–250 nm (two seeds each).
- Speckle size from `μ_A` scales as λz/D within 10 %.

## Phase 7 — Performance and GPU

**Deliverables**
- Profiling report (assembly, near matvec, far matvec per stage, field evaluation).
- OpenMP tuning: thread-parallel assembly over leaf pairs, NUMA-aware allocation, BLAS batching for translation.
- CUDA backend (ADR 0005): near-field assembly, translation stage, aggregation/disaggregation, field evaluation; host/device buffer management; mixed precision where validated.
- Optional out-of-core / recompute mode for `Z_near` when memory is the limit.

**DoD**
- ≥ 3× end-to-end speed-up on a single GPU vs 12-core CPU for the 393 k-unknown Ag sphere; identical results to 1e-4 relative.
- 30×30 µm² Si case fits in 128 GB host memory with recompute mode.

## Phase 8 — Alternative compression and general geometry

**Deliverables**
- `aca::AcaStrategy` (kernel-independent, naturally handles complex k — a candidate for the lossy interior operator); `hmatrix::HMatrixStrategy` with approximate H-LU as a preconditioner for metals; Schur-complement preconditioner; evaluation of FFT/butterfly-based schemes for very large electrical sizes.
- General closed meshes from STL/Gmsh in the whole pipeline; multiple objects (region graph) as an extension of the two-region formulation.
- Benchmark suite comparing compression schemes on identical problems (accuracy, memory, time).

**DoD**
- Each strategy passes the same matvec-vs-dense test as MLFMM.
- Ag rough surface iteration count reduced by ≥ 2× with H-LU or Schur preconditioner vs diagonal.
- An imported Gmsh mesh of a non-trivial closed shape (e.g. rough cylinder) solves and conserves power (absorbed + scattered = extinguished, within 1 %).

---

## Milestones (indicative order, not dates)

| M | Content | Phases |
|---|---------|--------|
| M1 | Meshes and bases validated | 0–1 |
| M2 | Dense solver matches Mie | 2 |
| M3 | Iterative solver, formulation choice automated | 3 |
| M4 | MLFMM matches dense, 4 µm Ag sphere matches Mie | 4 |
| M5 | Full post-processing, HDF5 | 5 |
| M6 | Python-complete, paper figures reproduced | 6 |
| M7 | GPU acceleration | 7 |
| M8 | Second compression method, general geometries | 8 |

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MLFMM plane-wave expansion breaks down for the strongly lossy interior (Ag) at large box sizes | wrong far interactions, divergence | per-level accuracy test against direct evaluation; cap interior expansion at the level where decay makes far terms negligible *to machine precision* (a numerical, not physical, truncation — documented and tested); ACA as kernel-independent alternative (Phase 8) |
| Low-frequency breakdown for very fine meshes (λ/27) | ill-conditioning | minimum leaf box λ/4; many elements per leaf; diagonal/block preconditioner |
| `Z_near` memory at 2 M unknowns (four operator blocks, two regions) | out-of-memory | symmetry exploitation, single-precision storage option, recompute mode (Phase 7) |
| Slow convergence for metals (γ ≈ 1.9) | long runtimes | H-LU / Schur preconditioners (Phase 8) |
| Artificial closing of the rough patch (side walls, bottom) | spurious reflections | box depth ≫ absorption length (Si at 500 nm: ~0.6 µm; Ag: ~15 nm); tapered beam; check sensitivity to box depth in tests |
| Paraxial beam violates Maxwell's equations at the few-percent level | not rigorous | angular-spectrum beam in Phase 5; paraxial only for early development |
| Singular-integral accuracy limits overall accuracy | plateau in convergence | analytic subtraction (ADR 0004); cross-check against Duffy transform |

## Work packages ready to start

1. `geometry/mesh.cpp` – topology and quality (Phase 1)
2. `geometry/rough_surface.cpp` – FFT height-map generator and closed-box mesher
3. `basis/rwg.cpp`, `kernels/quadrature.cpp`
4. `kernels/singularity.cpp` – analytic static integrals with unit tests
5. `reference/mie.cpp` – independent of the BEM, can proceed in parallel
