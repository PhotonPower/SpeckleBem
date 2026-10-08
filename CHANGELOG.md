# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com), versioning: [SemVer](https://semver.org).

## [Unreleased]

### Added
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
- build: fetched Eigen is a system include and no longer pollutes the CMake package registry.
- tooling: coordinator/worker/reviewer agent definitions, `CLAUDE.md`, `docs/backlog.md`.
- Project scaffold: CMake build with presets, dependency resolution, warnings, CI (GCC/Clang, Release/Debug+sanitizers).
- Public header layout for all planned layers (geometry, basis, kernels, formulation, operator, compression, solver, postprocessing, reference, backend, io).
- Implemented: core types and constants, logging, timer, reference materials (Ag/Si at 500 nm), dispersive material interpolation, PMCHWT/ICTF/MCTF weights and material-based recommendation, scalar Green's function and gradient, dense and sum operators, CPU backend.
- pybind11 module with materials and formulation enums; Python package skeleton.
- Unit tests (Catch2) and Python tests (pytest).
- Documentation: project plan, architecture, SIE and MLFMM theory, validation, conventions, coding guidelines, GPU strategy, compression survey, Python API, ADRs.
