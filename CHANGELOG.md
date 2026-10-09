# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com), versioning: [SemVer](https://semver.org).

## [Unreleased]

### Added
- python (WP14b2): `Simulation` (keyword mapping onto `SimulationConfig`: formulation,
  preconditioner, solver, compression, `gmres=dict(...)`, `kernels=dict(...)`; unknown strings
  and keys raise `ValueError`), `assemble()` / `solve(tol, max_iter, restart, side, callback)`
  with the GIL released and Python exceptions from the callback propagated, `SolveResult`,
  `report()`, resolved `formulation` / `preconditioner`, `currents` and `rhs()` (copies),
  `operator()` as `LinearOperator` (`matvec`, `@`, `as_scipy()`; keeps the Simulation alive),
  `field`, `far_field`, `bistatic_rcs`; free functions `plane_grid`, `cylinder_grid`,
  `intensity`, `polarized_intensity`, `differential_reflection_coefficient`; mesh file I/O
  (`read_mesh` / `write_mesh` and the STL / OBJ / Gmsh variants) and `open_npy_directory` /
  `ResultWriter` with `str` / `os.PathLike` UTF-8 paths. Tests `tests/python/test_simulation.py`
  (GMRES vs direct, Mie n = 1.5 at icosphere n = 2 with eps_rr < 2 %, callbacks, keep-alive,
  SciPy GMRES on `as_scipy()`, GIL release) and `test_io.py` (round trips under a non-ASCII
  directory). CI installs SciPy (apt `python3-scipy`, MSYS2 `python-scipy`).
- python (WP14b1f): `TriangleMesh.quality()` returns `MeshQuality` (with `__repr__`) and
  `quality_report()` the C++ text report, as in C++; NumPy-style docstrings with units and
  conventions on every bound class, function and method; `ValueError` for non-finite points in
  field evaluation (`electric_field`, `magnetic_field`, Mie fields) and non-finite angles in
  `bistatic_rcs`. Tests for input layouts (int32/uint32, Fortran order, strided views, nested
  lists, empty mesh), keep-alive of `Mie.a_n`/`b_n` and docstring units.
- kernels (WP7c): fold-adaptive outer rule for shared-edge and shared-vertex pairs
  (`OperatorOptions::fold_adaptive`, default on): the test triangle is cut into Duffy pieces at
  the singular directions through the shared vertex (projected source edges of folds below
  90 degrees, obtuse apex angles), with the points of level l + 2 / l + 3 on the pieces of
  doubly obtuse shared edges (both triangles obtuse at the same vertex) and of far vertices
  projecting within 0.2 |AB| of a shared vertex; touching-pair error <= 8.3e-8 for dihedral
  angles 30 to 179 degrees with regular, skewed, obtuse, doubly obtuse and near-vertex
  triangles (WP7b: up to 4.5e-5 skewed / 4.2e-5 near-vertex / 8.7e-5 obtuse at 30 to 45
  degrees; not covered: near-vertex shapes at folds of 90 to ~95 degrees keep the WP7b table,
  <= 1.5e-7). Cost against the WP7b rule (time per call, folds below 90 degrees): mean x1.9
  (shared edges, at most x3.3 for doubly obtuse pairs) and x1.6 (shared vertices, at most
  x3.2); hard bound 48 pieces x 16 x 16 outer points per ordering. Folds >= 90 degrees whose
  pieces need no split (all Mie icosphere pairs) bitwise unchanged.
  `kernels::touching_rule_info` reports the deterministic outer point counts. Slow dihedral
  sweep and right-hand-side degree study registered as the ctest entries
  `kernels: fold sweep (slow)` and `assembler: rhs degree study (slow)` (label `slow`,
  excluded from CI and the standard runs). The log-Gauss tables are checked against all 2n
  moments.
- operator (WP7c): `OperatorOptions::quad_degree_rhs` (default 8, positive-interior), the
  Dunavant degree of `op::assemble_rhs` (before: `quad_degree_near` = 19); relative error
  <= 7.5e-14 against degree 20 for plane waves and Gaussian beams on lambda/10 meshes.
- python (WP14b1): bindings for `TriangleMesh` (read-only zero-copy `vertices`/`triangles`/`edges`
  views kept alive by the mesh, `quality_report()` as `MeshQuality`), `make_icosphere`,
  `make_sphere`, `HeightMap`, `generate_gaussian_height_map`, `make_rough_surface_mesh`,
  `make_mesh_from_height_map` (keyword arguments incl. `box_depth`, `box_mesh_size`,
  `box_fine_depth`), the Python class `RoughSurface`, `DispersiveMaterial`, `field_decay_length`,
  `PlaneWave` / `GaussianBeam` (`shared_ptr` holders, vectorised `electric_field` /
  `magnetic_field`) and `Mie` (broadcast `bistatic_rcs`, `scattered_E`, cross sections); GIL
  released in mesh generation and field loops; invalid input raises `ValueError`. C++:
  `geometry::MeshQuality`, `TriangleMesh::quality()` and `geometry::to_string(MeshQuality)`
  (`quality_report()` unchanged). The module now builds with the project warning set (GCC's
  `-Wnull-dereference`/`-Wmaybe-uninitialized` false positives in pybind11/Eigen headers are
  disabled per source). Tests `tests/python/test_geometry.py`, `test_physics.py`.
- core (WP-P1, ADR 0007): file paths are UTF-8 strings on every platform.
  `core::path_from_utf8` (validating, via `std::u8string`; malformed UTF-8 or a NUL byte throws
  `std::invalid_argument`), `core::path_to_utf8` (via `path::u8string()`) and
  `core::find_invalid_utf8` in `core/path.hpp`. `geometry::io` mesh readers/writers and
  `io::open_npy_directory` open files only through these helpers and name paths in UTF-8 in
  error messages, so non-ASCII paths work on Windows (previously interpreted in the ANSI code
  page). Tests: `tests/unit/test_path.cpp`, and mesh (STL/OBJ/Gmsh) and `.npy` directory round
  trips under a directory named `sbem_Jürgen_µm_路径_🙂` in `test_mesh_io.cpp` /
  `test_result_writer.cpp`.
- solver (WP14a): `Simulation` driver (`src/simulation.cpp`, pimpl) for the dense strategy. Owns
  mesh, RWG space, formulation, excitation, `op::Problem` (ω of the excitation), operator,
  right-hand side, Jacobi diagonal (built lazily, only for a Jacobi-preconditioned GMRES solve)
  and solution; automatic formulation / preconditioner via `formulation::recommend`
  (`SimulationConfig::formulation` and `diagonal_preconditioner` are `std::optional`);
  `SolverKind::Gmres` (left/right, Jacobi or none; `solve(GmresParams, cb)` overrides
  `config.gmres` for one call, a non-converged solve keeps its solution and logs a warning) or
  `SolverKind::Direct` (dense LU, mapped onto `GmresResult` with `residual_history` =
  {LU residual}); `compression` defaults to `"dense"` (other strategies throw until Phase 4/8);
  constructor checks (closed and outward-oriented mesh, 2N ≤ `op::kMaxDenseUnknowns` (now public),
  wavelength and background vs the excitation, implemented formulation (JMCFIE rejected), kernel
  options, GMRES parameters); idempotent `assemble()`; `report()` with mesh size, choices, memory,
  timings, iterations and residuals; accessors `problem()`, `formulation_kind()`,
  `diagonal_preconditioner()`, `num_unknowns()`, `system_operator()`, `rhs()`. Unit tests
  `tests/unit/test_simulation.cpp`; validation `tests/validation/test_simulation_mie.cpp` (WP11
  dielectric sphere through the driver: direct reproduces the WP11 ε_rr and stays below 1 % vs
  Mie, GMRES/ICTF within 1e-4).
- io (WP16): `io::open_npy_directory`, a dependency-free `ResultWriter` that stores vectors and
  matrices as NumPy `.npy` files (format 1.0, little-endian, C order, `<f8`/`<c16`), meshes as
  `<group>/vertices.npy` (`<f8`) and `<group>/triangles.npy` (`<i8`, 0-based) and attributes in
  one `attributes.json` (rewritten on every call, shortest round-trip Reals, strict UTF-8);
  validated names (`[A-Za-z0-9_.-]`, `/` for groups, Windows device names such as `CON` or
  `com1.x` rejected on all platforms); every file is written to `<file>~tmp` and renamed over the
  target, so targets are always complete; `open_hdf5` throws until HDF5 support lands. Unit tests
  `tests/unit/test_result_writer.cpp` (independent NPY reader, bitwise data); NumPy round trip
  `tests/python/test_result_writer.py` via the helper `specklebem_npy_fixture` (CI sets
  `SPECKLEBEM_NPY_FIXTURE`, so a missing helper fails instead of skipping).
- geometry (WP2c): rough-surface box with optional decay-aware fine band (`box_fine_depth`) and a
  per-column stitched relaxation band under rough rims (vertical wall edges ≤ 2 h_g, M = 3 kept for
  σ ≤ 250 nm, Lc ≥ 100 nm; box 7–14 % of the top face at L = 10 µm); `material::field_decay_length`.
- kernels (WP7b): graded outer quadrature for touching triangle pairs (generalised Gauss-log rules on
  Duffy sub-triangles, `OperatorOptions::outer_grading_levels`, default 4) and k-aware near/far degree
  selection (`target_accuracy`, default 1e-5, within [`quad_degree_far`, `quad_degree_near`] = [3, 19]);
  touching-pair error 2e-4 → 3e-9, class-boundary error 4e-3 → 4e-7; Mie benchmark re-measured on the
  24-core Windows machine (`benchmarks/results/mie_sphere_dense.md`).
- solver (WP13): `solver::gmres`, complex GMRES (Saad & Schultz; MGS Arnoldi with one
  re-orthogonalisation pass, Givens rotations, happy-breakdown and singular-Krylov handling),
  full or restarted (`GmresParams::restart`), left or right preconditioning
  (`PreconditionerSide`, default left: the stopping criterion is the preconditioned residual,
  invariant under row scaling with Jacobi, issue #15), `GmresResult::true_relative_residual`;
  `DiagonalPreconditioner` (validated, stores 1/d). Unit tests `tests/unit/test_gmres.cpp`
  (LU agreement, termination in ≤ n steps, breakdown, restart, row-scaling invariance, right
  preconditioning, errors, dense BEM sphere systems for Si/ICTF and Ag/PMCHWT + Jacobi);
  validation-large case `tests/validation_large/test_gmres_vs_lu_large.cpp` (Si rough-surface
  box, 2N = 19 968, GMRES vs LU).
- tooling (WP-W1): native Windows build with MSYS2: presets `win-release` (UCRT64 GCC, OpenMP,
  OpenBLAS) and `win-debug` (CLANG64 Clang/libc++, ASan + UBSan, `-Werror`), a Windows MSYS2
  UCRT64 CI job, `.gitattributes` (LF checkouts everywhere), Windows memory guard of the dense
  validation tests (`tests/support/system_memory.cpp`). CMake ≥ 3.25 (FetchContent `SYSTEM`);
  fetched Eigen/spdlog/Catch2/pybind11 are system includes; fetched spdlog 1.14.1 → 1.15.3 and
  Catch2 3.7.1 → 3.16.1 (Clang 22 compatibility; the fetched Catch2 runs only in the Windows job,
  Linux CI uses the system Catch2 3.4). CI also runs on pushes to `wp/**` branches.
- geometry (WP1, #1): `TriangleMesh` edge topology (interior edges with plus/minus triangles,
  boundary-edge count, non-manifold and duplicate-triangle rejection), orientation check and
  repair with per-component outward orientation, closedness, signed volume, bounding box,
  quality report; `make_icosphere` / `make_sphere`; unit tests `tests/unit/test_mesh.cpp`.
- reference (WP5, #6): `MieSolution`, Bohren–Huffman Mie series adapted to exp(+jωt) (conjugated
  coefficients and fields), near-field n_max criterion, scattered/internal near fields,
  bistatic RCS, scattering and extinction cross sections; unit tests against the BH sample case,
  the Rayleigh limit and surface boundary conditions; validation tests labelled `validation`.
- basis (WP3b, #4): `RwgSpace`, Rao–Wilton–Glisson basis functions on interior edges with O(1)
  triangle → basis lookup, values, divergences and signed supports; unit tests for normal
  continuity, flux and divergence identities, open meshes and `flip_normals`.
- kernels (WP3a, #3): Dunavant triangle rules of degree 1–20 (re-solved to double precision,
  documented degrees with negative weights or exterior points, `triangle_rule_is_positive_interior`)
  and Gauss–Legendre rules of any order; exactness tests up to degree 20 and n = 100.
- geometry (WP2, #2): Gaussian rough-surface height maps (Eqs. 8–9, FFT and direct convolution,
  exact σ normalisation, Lc as 1/e radius of the autocorrelation, portable RNG), `HeightMap`
  statistics, closed box mesh with rough top (`make_rough_surface_mesh`,
  `make_mesh_from_height_map`); 20-seed statistics test, seed-42 regression reference in
  `tests/data/`, closedness/orientation/timing tests.
- tooling: `.clang-format` puts `<unsupported/Eigen/...>` into the third-party include block;
  `docs/compiler_notes/` documents a GCC 13.3 -O3 loop-vectoriser store-permutation bug that the
  box mesher works around.
- geometry (WP6, #7): mesh import/export for binary and ASCII STL (exact vertex welding),
  OBJ and Gmsh v4.1 ASCII with bit-exact text output, atomic writes, line-ending/BOM tolerance,
  fixtures and adversarial-input regression tests.
- kernels (WP4, #5): analytic static integrals over a flat triangle (∫1/R, ∫(r'−r)/R, ∫R,
  ∫(r'−r)R, ∫∇'(1/R) with solid-angle normal part and in-plane principal value, stable log
  branches, boundary finite-part flag) and `classify` proximity classes; tests against composite
  Dunavant, graded Duffy references, solid-angle jump and invariances.
- excitation (WP8, #9): `PlaneWave` and paraxial `GaussianBeam` (q-parameter form with first-order
  longitudinal components, θ_in rotation about y, p/s polarisation; Maxwell residual (λ/πw₀)²/2);
  solver (WP8): `solve_direct` with column equilibration, iterative refinement and
  `DirectSolveInfo` diagnostics (Eigen LU, BLAS-accelerated when available).
- post (WP10, #11): Stratton–Chu near-field evaluation in R1 and R2 (region by generalised winding
  number), `total_field`, `far_field`, `bistatic_rcs`, grids, polarised intensities and relative DRC;
  the representation signs are verified against exact Mie surface currents projected onto RWG
  (unit test n = 2 → 3, validation test n = 3 → 4, O(h²) convergence).
- kernels (WP7, #8): `element_blocks`, Galerkin L and K blocks for a triangle pair in mixed-potential
  form with singularity subtraction of the 1/R and R terms (ADR 0004), proximity-class quadrature,
  symmetrised touching pairs, series remainder near R = 0, `jump_block` for the ±½ n̂ × f term,
  `validate(OperatorOptions)`; tests against brute-force and polar references, symmetry, static
  limit; hidden `[.slow]` sweeps; allocation test in a separate executable.
- operator (WP9, #10): `DenseStrategy::build` (2N × 2N block system of docs/03 with formulation
  weights and the effective jump terms K₁ = K^PV − ½ I, K₂ = K^PV + ½ I; deterministic OpenMP by
  triangle colouring), `assemble_rhs`, `assemble_diagonal` (bitwise equal to the diagonal),
  `Problem::omega` and `op::validate`; first dense Mie solves: ε_rr = 0.28 % (n = 1.5) and 0.36 % (Ag)
  on an icosphere of 1 280 triangles; tests for symmetry, per-region jump signs, exact-current residuals.
- tests (WP11, #12): Mie validation of the dense solver in both scattering planes with refinement
  (ε_rr 0.89 % → 0.27 % → 0.074 % for n = 1.5, 1.95 % → 0.35 % → 0.078 % for Ag on icospheres
  n = 2/3/4), power balance (optical theorem, far-field integral, surface-current absorption), new
  `validation-large` ctest label (release-only, memory-guarded), results in
  `benchmarks/results/mie_sphere_dense.md`.
- tooling: `.github/workflows/ci.yml` had an invalid flow-mapping `env` since the scaffold so CI
  never ran; fixed. CI runs `-L '^validation$'`; `validation-large` is nightly/manual.
- post (WP9a): `SurfaceSolution::omega` removed; the frequency comes only from `op::Problem::omega`
  (checked against an attached excitation to 1e-12 relative); post-processing tests use
  `excitation::PlaneWave`. `validation-large` tests run serially (RUN_SERIAL).
- geometry (WP2b, #14): graded side walls and bottom plate of the rough-surface box
  (`RoughSurfaceParams::box_mesh_size`, automatic coarse spacing min(depth/2, L/8, 10 h), conforming
  2:1 transition strips, decay-based validity contract, rough-rim guard); reference case L = 10 µm,
  50 nm, depth 2 µm: closing box 7.2 % of the top face instead of 180 % (224 000 → 85 750
  triangles); uniform fallback bit-identical to the previous mesh.
- build: fetched Eigen is a system include and no longer pollutes the CMake package registry.
- tooling: coordinator/worker/reviewer agent definitions, `CLAUDE.md`, `docs/backlog.md`.
- Project scaffold: CMake build with presets, dependency resolution, warnings, CI (GCC/Clang, Release/Debug+sanitizers).
- Public header layout for all planned layers (geometry, basis, kernels, formulation, operator, compression, solver, postprocessing, reference, backend, io).
- Implemented: core types and constants, logging, timer, reference materials (Ag/Si at 500 nm), dispersive material interpolation, PMCHWT/ICTF/MCTF weights and material-based recommendation, scalar Green's function and gradient, dense and sum operators, CPU backend.
- pybind11 module with materials and formulation enums; Python package skeleton.
- Unit tests (Catch2) and Python tests (pytest).
- Documentation: project plan, architecture, SIE and MLFMM theory, validation, conventions, coding guidelines, GPU strategy, compression survey, Python API, ADRs.

### Changed
- material (WP14b1f): `DispersiveMaterial` throws `std::invalid_argument` for non-finite,
  non-positive or not strictly increasing wavelengths, non-finite indices and indices with
  `Im(n) > 0` (the message says that exp(+jwt) needs `n - jk` and that `n + ik` optics data must
  be conjugated); `at_wavelength` throws for a non-finite wavelength and now accepts the table
  endpoints. Python: these raise `ValueError`.
- python (WP14b1f): `TriangleMesh.quality_report()` returns the text report (`str`) instead of a
  `MeshQuality`; use `quality()` for the numbers. `quality()`/`quality_report()` no longer
  release the GIL.
