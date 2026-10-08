# SpeckleBem

**Rigorous simulation of polarised light scattering from rough and arbitrarily shaped surfaces, using surface integral equations (SIE) and fast boundary-element compression (MLFMM first, others later).**

SpeckleBem solves Maxwell's equations *exactly* (up to discretisation error) for a homogeneous object of complex permittivity embedded in a homogeneous background, by discretising only the interface with RWG basis functions and solving the coupled tangential EFIE/MFIE system iteratively. The dense impedance matrix is never formed for large problems; a pluggable compression layer (multilevel fast multipole method, later ACA / H-matrices) provides the matrix–vector product. The resulting surface currents give near fields, far fields, speckle statistics and scattering coefficients.

The design follows Fu, Daiber-Huppert, Frenner & Osten, *Optics and Lasers in Engineering* 162 (2023) 107438 ([docs/references.md](docs/references.md)) and generalises it: the geometry, the formulation, the compression scheme and the execution backend (CPU / GPU) are independent, replaceable layers.

## Goals

- **Rigorous** – no physical approximation (no Kirchhoff, no thin-layer, no periodicity, no paraxial shortcuts in the solver).
- **Large** – millions of unknowns on a workstation (30×30 µm² at λ = 500 nm ≈ 2 M unknowns in the reference paper).
- **Polarised** – full vector fields, co- and cross-polarised speckle.
- **General geometry** – spheres (Mie reference) and rough planar patches first; any closed triangle mesh (STL / Gmsh) later.
- **Headless** – C++20 core library, Python front end (pybind11 / NumPy), HDF5 output; no GUI.
- **GPU-ready** – hot kernels sit behind a backend interface; CUDA is an opt-in build.

## Repository layout

```
SpeckleBem/
├── CMakeLists.txt, CMakePresets.json   build system (C++20, CMake ≥ 3.24)
├── cmake/                               dependency and warning modules
├── include/specklebem/                  public headers (one directory per layer)
│   ├── core/          types, logging, timers
│   ├── geometry/      TriangleMesh, sphere, rough-surface generator, mesh I/O
│   ├── basis/         RWG basis functions
│   ├── material/      complex permittivity, dispersive tables
│   ├── excitation/    plane wave, Gaussian beam
│   ├── kernels/       Green's function, quadrature, singularity treatment, L/K operators
│   ├── formulation/   PMCHWT / ICTF / MCTF weights, preconditioner recommendation
│   ├── operator/      LinearOperator interface, dense & sparse operators, assembler
│   ├── compression/   mlfmm/ (Phase 4), aca/ and hmatrix/ (Phase 8)
│   ├── solver/        GMRES, preconditioners, direct LU
│   ├── postprocessing/ fields, RCS, DRC, speckle statistics
│   ├── reference/     Mie series
│   ├── backend/       CPU (OpenMP/BLAS) and CUDA execution backends
│   └── io/            HDF5 / NumPy result writers
├── src/                                 implementations, mirrors include/
├── python/                              pybind11 module + `specklebem` package
├── tests/                               unit/, validation/ (Catch2), python/ (pytest), data/
├── examples/                            C++ and Python usage examples
├── benchmarks/                          scaling benchmarks (Fig. 2c of the paper)
├── docs/                                plan, architecture, theory, conventions, ADRs
└── scripts/                             formatting, reference-data generation
```

## Documentation

| Document | Content |
|---|---|
| [docs/01_project_plan.md](docs/01_project_plan.md) | Phases, milestones, deliverables, acceptance criteria, risks |
| [docs/02_architecture.md](docs/02_architecture.md) | Layers, key abstractions, data flow, extension points |
| [docs/03_theory_sie.md](docs/03_theory_sie.md) | Stratton–Chu SIE, tangential formulations, RWG/Galerkin discretisation |
| [docs/04_theory_mlfmm.md](docs/04_theory_mlfmm.md) | Multilevel fast multipole method as applied here |
| [docs/05_validation.md](docs/05_validation.md) | Validation strategy and quantitative acceptance criteria |
| [docs/06_conventions.md](docs/06_conventions.md) | Units, time convention, coordinates, polarisation, region naming |
| [docs/07_coding_guidelines.md](docs/07_coding_guidelines.md) | C++ / Python style, testing, review rules |
| [docs/08_gpu_strategy.md](docs/08_gpu_strategy.md) | How and when kernels move to the GPU |
| [docs/09_compression_methods.md](docs/09_compression_methods.md) | MLFMM vs ACA vs H-matrices vs butterfly; selection criteria |
| [docs/10_python_api.md](docs/10_python_api.md) | Target Python API |
| [docs/adr/](docs/adr/) | Architecture decision records |

## Building

```bash
# Dependencies: C++20 compiler, CMake ≥ 3.24, Ninja, Eigen 3.4, BLAS/LAPACK (OpenBLAS or MKL),
# OpenMP, Python ≥ 3.10 with NumPy. spdlog, Catch2 and pybind11 are fetched automatically if absent.
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Python package (editable, builds the extension via scikit-build-core):

```bash
pip install -e python/[dev]
python -c "import specklebem as sb; print(sb.__version__, sb.silver_500nm().n)"
```

Optional: `-DSPECKLEBEM_USE_MKL=ON`, `-DSPECKLEBEM_ENABLE_CUDA=ON`, `-DSPECKLEBEM_ENABLE_HDF5=ON`.

## Status

Phase 1 (geometry and discretisation) complete: mesh topology and icospheres, Gaussian rough-surface generator and closed box mesh, STL/OBJ/Gmsh I/O, RWG basis, Dunavant and Gauss–Legendre quadrature, analytic static integrals, Mie reference solution. Phase 2 (dense SIE solver, Mie validation) in progress. See the [project plan](docs/01_project_plan.md), the [backlog](docs/backlog.md) and [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT – see [LICENSE](LICENSE).
