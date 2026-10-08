# Backlog

Work packages (WP) derived from the phases in [01_project_plan.md](01_project_plan.md).
Maintained by the coordinator agent (`.claude/agents/coordinator.md`); one WP is one worker,
one branch `wp/<id>-<slug>`, typically 200–800 lines including tests. Acceptance criteria are
taken from the phase's definition of done and the acceptance table in
[05_validation.md](05_validation.md). Status values: `todo`, `in progress`, `review`, `done`.

## Phase 1 — Geometry and discretisation

| WP | Title | Phase | Depends on | Files | Acceptance criterion | Status | Branch | Issue |
|----|-------|-------|------------|-------|----------------------|--------|--------|-------|
| WP1 | Mesh topology, orientation, closedness, quality report; icosphere generator | 1 | — | `src/geometry/mesh.cpp`, `src/geometry/sphere.cpp`, `tests/unit/test_mesh.cpp` (+ `src/CMakeLists.txt`, `tests/CMakeLists.txt`) | Euler characteristic V − E + F = 2 for icospheres (subdivisions 0–4); every edge has exactly two triangles; `is_closed()` true for the sphere and false after removing a triangle; orientation check detects a flipped triangle and repair restores consistency; normals of the icosphere point outward (into R1, `n̂·(c − 0) > 0`); `flip_normals()` reverses all; `bounding_box` exact; areas sum to 4πr² within the discretisation error that decreases with refinement; `quality_report()` reports min/max/mean edge length, aspect ratio, non-manifold edge count; `make_sphere` meets the target edge length within a factor 2 | done (squash-merged) | `wp/01-mesh-topology` | [#1](https://github.com/PhotonPower/SpeckleBem/issues/1) |
| WP2 | Gaussian rough-surface generator and closed box mesh | 1 | WP1 | `src/geometry/rough_surface.cpp`, `tests/unit/test_rough_surface.cpp`, `tests/data/` regression entry | Height maps reproduce σ within 2 % and Lc within 5 % over an ensemble of 20 seeds (Eqs. 8–9, docs/05); FFT and direct convolution agree to 1e-10 relative on a small grid; `make_rough_surface_mesh` yields a closed, consistently oriented mesh (χ = 2, all edges two triangles) with normals out of R2 (object `z > ξ`) into R1 (`z < ξ`); default box depth per ADR 0006; L = 10 µm, mesh 50 nm generates in < 5 s (release); `seed = 42`, L = 10 µm regression check against stored reference; `make_mesh_from_height_map` round-trips a user height map | in progress | `wp/02-rough-surface` | [#2](https://github.com/PhotonPower/SpeckleBem/issues/2) |
| WP3a | Triangle (Dunavant) and interval (Gauss–Legendre) quadrature rules | 1 | — | `src/kernels/quadrature.cpp`, `tests/unit/test_quadrature.cpp` | `triangle_rule(d)` for d = 1…20: weights sum to 1, points inside the triangle, integrates all monomials x^a y^b with a + b ≤ d exactly (1e-13 relative); `gauss_legendre(n)` for n = 1…64 integrates polynomials of degree 2n − 1 exactly (1e-13); invalid degree throws `std::invalid_argument`; rules are cached and returned by const reference | todo | `wp/03a-quadrature` | [#3](https://github.com/PhotonPower/SpeckleBem/issues/3) |
| WP3b | RWG basis function space | 1 | WP1 | `src/basis/rwg.cpp`, `tests/unit/test_rwg.cpp` | `size()` = number of interior edges; `divergence(n, t)` = +l/A on T⁺ and −l/A on T⁻; normal component of f_n is continuous across the shared edge (equal to 1 on the edge, zero on the other two edges); ∫_{T±} ∇·f_n dS = ±l_n (closed form); `support(t)` lists exactly the ≤ 3 basis functions on t with correct signs; `value()` returns zero for unsupported (n, t) | in progress | `wp/03b-rwg` | [#4](https://github.com/PhotonPower/SpeckleBem/issues/4) |
| WP4 | Analytic static integrals and proximity classification | 1 | WP1, WP3a | `src/kernels/singularity.cpp`, `tests/unit/test_singularity.cpp` | `static_integrals` (∫ 1/R, ∫ (r' − r)/R, ∫ ∇'(1/R)) agree with Dunavant-20 numerical integration to 1e-12 (abs/rel) on well-separated triangles (docs/05); on coincident / edge- / vertex-adjacent configurations the analytic value agrees with nested-quadrature extrapolation to 1e-9 relative; observation point in the plane of the triangle and at a vertex handled without NaN; `classify` returns identical / shared_edge / shared_vertex / near / far correctly on an icosphere | todo | `wp/04-singularity` | [#5](https://github.com/PhotonPower/SpeckleBem/issues/5) |
| WP5 | Mie reference solution | 1 (needed in 2) | — | `src/reference/mie.cpp`, `tests/unit/test_mie.cpp`, `tests/validation/test_mie_reference.cpp` | Bohren–Huffman coefficients adapted to exp(+jωt) (`Im(n) ≤ 0`, outgoing `exp(−jkr)`); Wiscombe n_max; Bohren–Huffman Appendix A sample case (m = 1.55, x = 5.213: Q_sca = Q_ext = 3.10543, Q_back = 2.92534 within 1e-4); Rayleigh limit (x = 0.01, Ag) for C_sca and C_abs against the closed form within 1e-3; optical theorem; `C_ext − C_sca = C_abs ≥ 0` for Ag and Si; `bistatic_rcs` consistent with `|r E_s|²` from `scattered_E` at large r (1e-6 relative); backward/forward scattering symmetry for a small sphere; tangential continuity of E (and H) at the surface between `internal_E` and `scattered_E + E_inc` (1e-6 relative) | in progress | `wp/05-mie` | [#6](https://github.com/PhotonPower/SpeckleBem/issues/6) |
| WP6 | Mesh import/export: STL (binary/ASCII), OBJ, Gmsh v4.1 ASCII | 1 | WP1 | `src/geometry/mesh_io.cpp`, `tests/unit/test_mesh_io.cpp`, small fixtures in `tests/data/` | Round trip write → read of an icosphere for each format preserves vertex count (after welding of duplicate STL vertices), triangle count and orientation to 1e-12 relative; `read_mesh` dispatches by extension and throws on unknown extension or malformed file; imported mesh passes `is_closed()` | todo | `wp/06-mesh-io` | [#7](https://github.com/PhotonPower/SpeckleBem/issues/7) |

### Schedule

- Wave 1 (parallel): WP1, WP5. WP3a also has no dependencies and may join as a third worker.
- Wave 2 (after WP1 is merged): WP2, WP3b, WP3a (if not started earlier).
- Wave 3: WP4 (after WP3a), WP6.
- Phase 1 is complete (milestone M1) when all seven WPs are `done` and the Phase 1 definition
  of done in docs/01 is met.

## Phase 2 and later

To be broken down when Phase 1 nears completion. Candidates in dependency order:
`kernels::element_blocks` (WP7, needs WP3b + WP4), `excitation::PlaneWave` (WP8),
`op::DenseStrategy` + RHS + diagonal (WP9), `solver::solve_direct` (WP10),
`post::scattered_field` / `far_field` / `bistatic_rcs` (WP11), Mie validation test
(WP12, needs WP5 + WP9–WP11), Fresnel flat-interface validation (WP13, needs WP2).

## Notes for workers (lessons learned)

- Without system packages, Eigen/spdlog/Catch2/pybind11 are fetched by `cmake/Dependencies.cmake`.
  The fetched Eigen is marked as a system include so that `-Werror` does not trip on Eigen
  internals; do not add Eigen include paths by hand.
- BLAS/LAPACK are optional; the test machine may not have them (Eigen fallback, same results).
- `-Wconversion`/`-Wsign-conversion` are on: use `Index` (int64) for sizes, cast explicitly
  when indexing Eigen with `int` and when mixing `std::size_t` and `Index`.
