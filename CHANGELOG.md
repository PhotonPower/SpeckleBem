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
- build: fetched Eigen is a system include and no longer pollutes the CMake package registry.
- tooling: coordinator/worker/reviewer agent definitions, `CLAUDE.md`, `docs/backlog.md`.
- Project scaffold: CMake build with presets, dependency resolution, warnings, CI (GCC/Clang, Release/Debug+sanitizers).
- Public header layout for all planned layers (geometry, basis, kernels, formulation, operator, compression, solver, postprocessing, reference, backend, io).
- Implemented: core types and constants, logging, timer, reference materials (Ag/Si at 500 nm), dispersive material interpolation, PMCHWT/ICTF/MCTF weights and material-based recommendation, scalar Green's function and gradient, dense and sum operators, CPU backend.
- pybind11 module with materials and formulation enums; Python package skeleton.
- Unit tests (Catch2) and Python tests (pytest).
- Documentation: project plan, architecture, SIE and MLFMM theory, validation, conventions, coding guidelines, GPU strategy, compression survey, Python API, ADRs.
