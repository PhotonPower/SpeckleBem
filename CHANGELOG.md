# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com), versioning: [SemVer](https://semver.org).

## [Unreleased]

### Added
- mlfmm (WP17): `mlfmm::Octree` over the RWG edge midpoints: root cube from the padded mesh
  vertex bounding box, anchored at its lower corner (flat or thin geometry stays one box layer
  thick), uniform depth (all leaves on the finest level; the first level on which every box
  holds <= `max_elements_per_leaf`, bounded by `max_levels` <= 21 and the floor
  `(1 - kMinBoxSizeTolerance)`·`min_box_size_lambda`·λ with `kMinBoxSizeTolerance = 1e-2`, so
  the 4 µm sphere at 500 nm, root 8 λ, reaches λ/4 leaves on 6 levels), Morton-ordered boxes
  per level with integer coordinates `Box::ijk`, contiguous element ranges (`permutation()` /
  `inverse_permutation()`), `find_box(level, ijk)`, near lists (same-level neighbours, self
  excluded) and interaction lists (<= 189, levels >= 2), `summary()`. Tests check the
  permutation, range nesting, Morton order, root box, box geometry and leaf criteria on
  icospheres, a thin rough box and flat plates (exactly flat and with ±1e-15 m jitter: one box
  layer, identical leaves), the near and interaction lists against an independent pairwise
  classification from `ijk`, list symmetry and completeness over all leaf pairs (each pair near
  or translated on exactly one level), the λ/4 floor (4 µm sphere regression) and `max_levels`
  bounds, input errors, and the build time (122,880 basis functions in ~0.03 s, release).
- solver / tooling (WP15, Phase 3 DoD): rough-box helpers `default_box_depth(object, λ)` =
  max(10 intensity absorption lengths δ/2, 2 µm) (`kMinBoxDepth`; Si 5.65 µm, Ag 2 µm) and
  `default_box_fine_depth(object, λ, σ)` = 3δ + 3σ in `simulation.hpp` (ADR 0006; a lossless
  dielectric object throws `std::invalid_argument`), with `[simulation]` unit tests. Study
  executable `benchmarks/formulation_convergence.cpp` (`SPECKLEBEM_BUILD_BENCHMARKS`): PMCHWT,
  ICTF and MCTF without and with left Jacobi on Si and Ag Gaussian rough surfaces (σ = 50 nm,
  Lc = 500 nm, h = 50 nm, L = 3.2 µm, 2N ≈ 3·10⁴, dense, full GMRES, tol 1e-3); record
  `benchmarks/results/formulation_convergence.md` and histories
  `benchmarks/results/formulation_convergence_histories.csv`. Si: ICTF 219, MCTF 254, PMCHWT 315,
  Jacobi 272 iterations; Ag: ICTF 622, MCTF 651, PMCHWT 886, Jacobi 1069; the three
  Jacobi-preconditioned formulations iterate identically (≤ 5e-13, issue #15). Fu et al.'s
  "PMCHWT does not converge" and "Jacobi best for Ag" are not reproduced at this size.
- mlfmm (WP18): plane-wave machinery (`compression/mlfmm/plane_wave.hpp`, `interpolation.hpp`):
  `truncation_order` (ADR 0008 §2, Re k bandwidth), `SphereSampling` (Gauss-Legendre x uniform
  phi, contiguous SoA directions / theta_hat / phi_hat, weights summing to 4 pi),
  `spherical_hankel2` (h_l^(2) of complex argument, Re z > 0, Im z <= 0, upward recurrence,
  measured <= 1.4e-14 relative for l <= 120, |z| in [0.05, 80]), `legendre_p`, the diagonal
  translator `translator` (exp(+jwt), prefactor -jk/4pi verified by the addition theorem),
  `expansion_error` (worst-case corner-to-corner check over the nearest interaction-list
  offsets, building block of the ADR 0008 §6 lossy-region policy; Ag-like k gives >> 1) and
  `SphereInterpolator` (separable local Lagrange, polar reflection with even/odd parity for
  scalar / theta-phi components, anterpolation = exact transpose). Measured: random points
  meet 10^-d0 for box edges >= lambda/2 (d0 = 3, 5; Si-like k from lambda/4), lambda/4 boxes
  lose up to one digit; the corner worst case stays above 10^-d0 at the ADR order (1e-3 ...
  9e-2); 0.1 x 10^-d0 interpolation needs p ~ 12-14 (d0 = 3) / ~ 20-22 (d0 = 5), not p = 6.
- mlfmm (WP18 review, ADR 0008 amendments): `interpolation_order(d0)` (measured table: 14 for
  d0 <= 3, 22 for d0 <= 5, throws above 5) and `leaf_sampling_order(L, p)` = max(L, p - 1);
  `SphereInterpolator` has no default order any more and accepts any p <= 2 n_theta_src
  (stencils over the pole-reflected nodes); the weight convention of the disaggregation (apply
  the parent weights before anterpolating) is documented. `expansion_error` gains the seeded
  statistical mode (`ExpansionCheck::random`, `pairs`, `seed`; platform-independent points) and
  `search_truncation_order` implements the per-level order search (smallest L >= formula
  meeting 10^-d0, breakdown detection). Measured: real k needs formula + 1 only at lambda/4 for
  d0 = 5; lambda/4 at d0 = 7 is not achievable (minimum 4e-7); Ag-like k reaches d0 = 3 for
  boxes <= 0.25 lambda_0 at L = 9 ... 22 and breaks down from 0.5 lambda_0. `spherical_hankel2`
  raises `std::underflow_error` when e^{-jz} underflows (instead of returning zeros), the
  translator raises `std::overflow_error` for non-finite coefficients or values. New tests:
  complex-z Hankel check up to |z| = 80 against Miller j_l / upward y_l with their Wronskian,
  lambda/4 interpolation with the oversampled leaf, order search, and a `slow` sweep
  (`mlfmm: plane-wave accuracy sweep (slow)`) printing the measured tables.
- kernels / operator (WP-P2): dense assembly performance on rough boxes. Profile benchmark
  `benchmarks/dense_assembly_profile.cpp` (per-class element_blocks timings, near/far degree and
  touching-rule statistics, schedule makespan, same-process before/after comparison) and record
  `benchmarks/results/dense_assembly_profile.md`. Root causes: OpenMP static schedule (one
  thread got all coarse box rows, ×2.5 / ×4.5 the ideal time), degree cap 19 for every object-
  region pair with a 400 nm box cell, C-library sincos/exp in the near/far loop. Fixes: dynamic
  largest-first schedule (bitwise identical matrix); `OperatorOptions::decay_aware_target`
  (default on; WP7c part 4: lossy regions bound the near/far error by target × the undamped
  magnitude bound of the block, ADR 0004; lossless regions bitwise unchanged);
  `OperatorOptions::fast_plain_kernel` (default on; branch-free vectorised sin/cos/exp in
  `kernels/fast_math.hpp`, ≤ 1 ulp, blocks within 2.7e-15; operators.cpp built with
  `-fno-math-errno -fno-trapping-math`); `kernels::plain_rule_degree` (diagnostics); Python
  `kernels=dict(decay_aware_target=...)`. WP15 systems: Si 1513–1535 s → 242 s, Ag 1056–1293 s
  → 35 s (24 threads); the Si box stays ~5–10× the N²-scaled sphere figure because of its coarse
  box cells (|k_Si| h ≈ 30, see the record).
- python (WP14b3): calls on one `Simulation` from several Python threads are serialised by a
  per-object timed mutex taken with the GIL released (re-entrant calls from a `solve()` callback
  raise `RuntimeError`; from inside a callback another busy `Simulation` raises "Simulation
  busy" instead of waiting, so two solves calling into each other cannot deadlock; a main
  thread waiting for the lock checks for Ctrl-C every 50 ms); GMRES solves are interruptible
  with Ctrl-C (`PyErr_CheckSignals()` per iteration with a callback, every 50 ms on the main
  thread without one); `exterior=None` now means `excitation.background` (was vacuum). Tests
  for concurrent `solve()` (alternating tolerances) against `field()` / `bistatic_rcs()` /
  `currents` / `report()` with exact comparison to the two possible results, a concurrent first
  `solve()` on an unassembled Simulation, cross-Simulation callbacks, re-entrancy, interruption
  of a solve and of a lock wait via `_thread.interrupt_main()`, a dielectric background and the
  `SolveResult.x` keep-alive. `op::LinearOperator::apply` documents that it must be safe to
  call concurrently. CI sets `SPECKLEBEM_REQUIRE_SCIPY=1`, which makes the SciPy test fail
  instead of skip without SciPy.
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
- kernels (WP7d): the fold-adaptive rule classifies right angles at shared vertices (rough
  top faces, box rim) with a relative tolerance of 1e-6 instead of exact sign / distance tests,
  so exact and +-1e-9-perturbed right angles select the same rule (that of acute angles);
  shared-vertex pairs with a steep source triangle that has a vertex less than 11.5 degrees
  above the test plane (box rim) use the adaptive piece (rim K error 1.2e-7 -> 2e-10); with the
  ray partition and S within 0.3 |AB| of a shared vertex the pieces of the other apex get the
  points of level l + 3 (right angles at A, 75-degree fold: 3.8e-7 -> 4.7e-8). Icosphere
  blocks are bitwise unchanged. Tests: right-angle shapes in the fold sweep and fast rule /
  accuracy checks, top-face and rim pairs of a rough box against the reference, the near-B
  deviation pinned, the 16 x 16 point cap at grading levels 1 to 6. Doc fixes (ADR 0004).
- material (WP14b1f): `DispersiveMaterial` throws `std::invalid_argument` for non-finite,
  non-positive or not strictly increasing wavelengths, non-finite indices and indices with
  `Im(n) > 0` (the message says that exp(+jwt) needs `n - jk` and that `n + ik` optics data must
  be conjugated); `at_wavelength` throws for a non-finite wavelength and now accepts the table
  endpoints. Python: these raise `ValueError`.
- python (WP14b1f): `TriangleMesh.quality_report()` returns the text report (`str`) instead of a
  `MeshQuality`; use `quality()` for the numbers. `quality()`/`quality_report()` no longer
  release the GIL.
