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
| WP2 | Gaussian rough-surface generator and closed box mesh | 1 | WP1 | `src/geometry/rough_surface.cpp`, `tests/unit/test_rough_surface.cpp`, `tests/data/` regression entry | Height maps reproduce σ within 2 % and Lc within 5 % over an ensemble of 20 seeds (Eqs. 8–9, docs/05); FFT and direct convolution agree to 1e-10 relative on a small grid; `make_rough_surface_mesh` yields a closed, consistently oriented mesh (χ = 2, all edges two triangles) with normals out of R2 (object `z > ξ`) into R1 (`z < ξ`); default box depth per ADR 0006; L = 10 µm, mesh 50 nm generates in < 5 s (release); `seed = 42`, L = 10 µm regression check against stored reference; `make_mesh_from_height_map` round-trips a user height map | done (squash-merged) | `wp/02-rough-surface` | [#2](https://github.com/PhotonPower/SpeckleBem/issues/2) |
| WP3a | Triangle (Dunavant) and interval (Gauss–Legendre) quadrature rules | 1 | — | `src/kernels/quadrature.cpp`, `tests/unit/test_quadrature.cpp` | `triangle_rule(d)` for d = 1…20: weights sum to 1, integrates all monomials x^a y^b with a + b ≤ d exactly (1e-13 relative); all points inside and all weights positive except the documented Dunavant degrees 3, 7, 11, 15, 16, 18, 20 (queried via `triangle_rule_is_positive_interior`); `gauss_legendre(n)` for n = 1…64 integrates polynomials of degree 2n − 1 exactly (1e-13); invalid degree throws `std::invalid_argument`; rules are cached and returned by const reference | done (squash-merged) | `wp/03a-quadrature` | [#3](https://github.com/PhotonPower/SpeckleBem/issues/3) |
| WP3b | RWG basis function space | 1 | WP1 | `src/basis/rwg.cpp`, `tests/unit/test_rwg.cpp` | `size()` = number of interior edges; `divergence(n, t)` = +l/A on T⁺ and −l/A on T⁻; normal component of f_n is continuous across the shared edge (equal to 1 on the edge, zero on the other two edges); ∫_{T±} ∇·f_n dS = ±l_n (closed form); `support(t)` lists exactly the ≤ 3 basis functions on t with correct signs; `value()` returns zero for unsupported (n, t) | done (squash-merged) | `wp/03b-rwg` | [#4](https://github.com/PhotonPower/SpeckleBem/issues/4) |
| WP4 | Analytic static integrals and proximity classification | 1 | WP1, WP3a | `src/kernels/singularity.cpp`, `tests/unit/test_singularity.cpp` | `static_integrals` (∫ 1/R, ∫ (r' − r)/R, ∫ ∇'(1/R)) agree with Dunavant-20 numerical integration to 1e-12 (abs/rel) on well-separated triangles (docs/05); on coincident / edge- / vertex-adjacent configurations the analytic value agrees with nested-quadrature extrapolation to 1e-9 relative; observation point in the plane of the triangle and at a vertex handled without NaN; `classify` returns identical / shared_edge / shared_vertex / near / far correctly on an icosphere | done (squash-merged) | `wp/04-singularity` | [#5](https://github.com/PhotonPower/SpeckleBem/issues/5) |
| WP5 | Mie reference solution | 1 (needed in 2) | — | `src/reference/mie.cpp`, `tests/unit/test_mie.cpp`, `tests/validation/test_mie_reference.cpp` | Bohren–Huffman coefficients adapted to exp(+jωt) (`Im(n) ≤ 0`, outgoing `exp(−jkr)`); Wiscombe n_max; Bohren–Huffman Appendix A sample case (m = 1.55, x = 5.213: Q_sca = Q_ext = 3.10543, Q_back = 2.92534 within 1e-4); Rayleigh limit (x = 0.01, Ag) for C_sca and C_abs against the closed form within 1e-3; optical theorem; `C_ext − C_sca = C_abs ≥ 0` for Ag and Si; `bistatic_rcs` consistent with `|r E_s|²` from `scattered_E` at large r (1e-6 relative); backward/forward scattering symmetry for a small sphere; tangential continuity of E (and H) at the surface between `internal_E` and `scattered_E + E_inc` (1e-6 relative) | done (squash-merged) | `wp/05-mie` | [#6](https://github.com/PhotonPower/SpeckleBem/issues/6) |
| WP6 | Mesh import/export: STL (binary/ASCII), OBJ, Gmsh v4.1 ASCII | 1 | WP1 | `src/geometry/mesh_io.cpp`, `tests/unit/test_mesh_io.cpp`, small fixtures in `tests/data/` | Round trip write → read of an icosphere for each format preserves vertex count (after welding of duplicate STL vertices), triangle count and orientation to 1e-12 relative; `read_mesh` dispatches by extension and throws on unknown extension or malformed file; imported mesh passes `is_closed()` | done (squash-merged) | `wp/06-mesh-io` | [#7](https://github.com/PhotonPower/SpeckleBem/issues/7) |

| WP2b | Graded / coarsened side walls and bottom plate of the rough-surface box | 1 (follow-up) | WP2 | `src/geometry/rough_surface.cpp`, tests | WP2 meshes walls and bottom at the top-face spacing (L = 10 µm, 50 nm, depth 2 µm: 80 k top + 80 k bottom + 64 k wall triangles, i.e. ≈ 1.8× the top-face count, versus "a few percent" in ADR 0006). Coarsen the bottom plate and the lower part of the walls (field there is negligible by construction) with a conforming transition, keep closedness/orientation tests green, and either meet the ADR 0006 estimate or amend its Consequences section. Validation: box-depth and edge-effect sensitivity tests (docs/05) unchanged | done (squash-merged) | `wp/02b-box-grading` | [#14](https://github.com/PhotonPower/SpeckleBem/issues/14) |

### Schedule

- Wave 1 (parallel): WP1, WP5. WP3a also has no dependencies and may join as a third worker.
- Wave 2 (after WP1 is merged): WP2, WP3b, WP3a (if not started earlier).
- Wave 3: WP4 (after WP3a), WP6.
- Phase 1 (milestone M1) is complete: WP1, WP2, WP3a, WP3b, WP4, WP5, WP6 merged (main 6c1bf9f,
  115 tests). Open follow-up: WP2b.

## Phase 2 — Dense SIE solver and Mie validation

Feasibility note (dense oracle, this 15 GB / 4-core machine without BLAS; CI runners have 7 GB):
a dense `Z` with 2N unknowns needs 16·(2N)² bytes, i.e. 2N = 20 k → 6.4 GB. The Phase 2 DoD cases
scale as follows: dielectric sphere d = 1 µm at λ/10 (icosphere n = 4, 5 120 triangles, ≈ λ/13):
2N ≈ 15 k → 3.8 GB, feasible but slow without BLAS (`validation-large`); Ag sphere d = 1 µm at
λ/20 with icospheres means n = 5 (20 480 triangles, ≈ λ/26): 2N ≈ 61 k → 121 GB incl. the LU copy, not feasible here (needs a ≥ 200 GB node or the Phase 4 MLFMM; the test exists and skips by its memory guard);
flat box for the Fresnel limit (needs L ≳ 3 w₀ ≳ 10 λ at λ/10): 2N ≳ 10⁵ → > 100 GB, **not
feasible dense** and therefore deferred to Phase 4 (MLFMM). Unit-level tests use icospheres with
n ≤ 3 (2N ≤ 3 840).

| WP | Title | Phase | Depends on | Files | Acceptance criterion | Status | Branch | Issue |
|----|-------|-------|------------|-------|----------------------|--------|--------|-------|
| WP7 | `kernels::element_blocks`: Galerkin L and K blocks for a triangle pair with singularity subtraction | 2 | WP3a, WP3b, WP4 | `src/kernels/operators.cpp`, `tests/unit/test_operators.cpp` | Mixed-potential form of docs/03 (gradient moved onto the test function, K as principal value plus the explicit ±½ n̂ × f jump on coincident triangles); proximity class from `classify`; `OperatorOptions` degrees validated (positive-interior only for non-far classes, ADR 0004); far pairs agree with brute-force double Dunavant quadrature to 1e-10; touching pairs: the inner subtraction agrees with an independent Duffy/polar cross-check (same outer rule) to 1e-6 and converges with the smooth-remainder degree, while the outer-rule error at the default degree (~2e-4 for L, ~1e-4 of ½I for K) is documented in `operators.hpp` and ADR 0004 (follow-up WP7b); L blocks symmetric under (test, source) swap to 1e-12; static limit k → 0 of the vector-potential part matches the static integrals; complex k (Ag, Si) handled; allocation-free inner loop | done (squash-merged) | `wp/07-element-blocks` | [#8](https://github.com/PhotonPower/SpeckleBem/issues/8) |
| WP8 | Excitations and direct solver: `excitation::PlaneWave`, paraxial `GaussianBeam`, `solver::solve_direct` | 2 | — | `src/excitation/excitation.cpp`, `src/solver/direct.cpp`, `tests/unit/test_excitation.cpp`, `tests/unit/test_direct.cpp` | PlaneWave: |E| = |e0|, H = k̂ × E / η1, `exp(−j k k̂·r)`, finite-difference Maxwell check `∇×E = −jωμH` to 1e-8; GaussianBeam (paraxial, Phase 2 model, Phase 5 replaces it): waist/focus/θ_in/polarisation per header, reduces to the plane wave for w₀ → ∞, Maxwell residual documented (few percent, flagged in the log); `solve_direct` = Eigen `PartialPivLU` (BLAS-accelerated when available; LAPACK `zgesv` path deferred, not buildable here) with column equilibration and iterative refinement, residual ‖Zx − b‖/‖b‖ < 1e-12 on random 200×200 systems, singular matrix throws | done (squash-merged) | `wp/08-excitation-direct` | [#9](https://github.com/PhotonPower/SpeckleBem/issues/9) |
| WP9 | Dense assembly: `op::DenseStrategy`, `assemble_rhs`, `assemble_diagonal` | 2 | WP7, WP8 | `src/operator/assembler.cpp`, `tests/unit/test_assembler.cpp` | 2N × 2N block system of docs/03 with formulation weights (a_i/η_i, b_i η_i), OpenMP over triangle pairs with deterministic results, `Z_PMCHWT` complex-symmetric to < 1e-6 relative (docs/01), diagonal equals the diagonal of the dense matrix, RHS by Dunavant quadrature of ⟨f_m, E_inc⟩ / ⟨f_m, H_inc⟩ converging with degree; assembly of an n = 3 icosphere (2N = 3 840) in < 30 s in release (measured 2.7 s on 3 threads; ~140 s under sanitizers, documented) | done (squash-merged) | `wp/09-dense-assembly` | [#10](https://github.com/PhotonPower/SpeckleBem/issues/10) |
| WP10 | Post-processing: scattered/total near field, far field, bistatic RCS, polarised intensities | 2 | WP3b | `src/postprocessing/fields.cpp`, `src/postprocessing/scattering.cpp`, `tests/unit/test_fields.cpp` | Stratton–Chu evaluation from `[J; M]` with the sign conventions of docs/03 in R1 and R2; `far_field` is the r → ∞ limit of `scattered_field` (1e-6 at r = 10⁴ λ with Richardson); `bistatic_rcs` = 4π|F|²/|E_inc|²; `plane_grid`/`cylinder_grid` layouts; `polarized_intensity` decomposition; unit test with the currents of a Mie solution projected onto RWG (J = n̂ × H, M = −n̂ × E from WP5) reproduces the Mie far field within the projection error that decreases with refinement | done (squash-merged) | `wp/10-postprocessing` | [#11](https://github.com/PhotonPower/SpeckleBem/issues/11) |
| WP11 | Mie validation of the dense solver | 2 | WP5, WP9, WP10 | `tests/validation/test_mie_sphere_dense.cpp` | docs/05: dielectric sphere n = 1.5, d = 1 µm, λ = 500 nm: ε_rr < 1 % vs Mie (icosphere n = 4, label `validation-large`), monotone decrease over n = 2, 3, 4 (label `validation` for n ≤ 3); Ag sphere d = 1 µm with PMCHWT + LU: ε_rr at n = 3 and n = 4 documented, the λ/20 case (2N ≈ 35 k) labelled `validation-large` and skipped when memory is insufficient; power balance for Ag within 1 %; results stored under `benchmarks/results/` | done (squash-merged; Ag λ/20 case wired up, needs ≥ 200 GB) | `wp/11-mie-validation` | [#12](https://github.com/PhotonPower/SpeckleBem/issues/12) |
| WP12 | Fresnel flat-interface validation (tapered beam, 0° and 45°, p and s) | 2 → 4 | WP2, WP8, WP10, MLFMM | `tests/validation/test_flat_interface_fresnel.cpp` | Deferred: dense system size > 10⁵ unknowns (see feasibility note); implemented with the MLFMM operator in Phase 4 | blocked | `wp/12-fresnel` | [#13](https://github.com/PhotonPower/SpeckleBem/issues/13) |

| WP2c | Decay-aware fine band and rim-decoupled grading rows for the rough-surface box | 1/3 (follow-up) | WP2b | `src/geometry/rough_surface.cpp`, `rough_surface.hpp`, tests | Optional `box_fine_depth`: walls keep the top spacing down to ~3 field decay lengths δ = λ₀/(2π|Im n₂|) (Si at 500 nm: ~3 µm), then grade; `Simulation` passes it from the material. Interior rows follow the piecewise-linear level-M rim interpolation with a few level-0 relaxation rows (quads with vertical edges) so that 2:1 cells cannot invert for any roughness; the global-M guard stays as a safety net. Acceptance: default rough surfaces with σ ≤ 250 nm, Lc ≥ 100 nm at 50 nm spacing keep M = 3 (box ≤ 10 % of the top); all closedness/orientation/aspect tests green | todo | `wp/02c-box-fine-band` | — |
| WP7b | Graded outer quadrature for touching pairs and k-aware degree selection | 2/3 (follow-up) | WP7 | `src/kernels/operators.cpp`, `tests/unit/test_operators.cpp`, ADR 0004 | Sauter–Schwab or graded/Duffy outer rule for identical, shared-edge and shared-vertex pairs: L and K blocks accurate to 1e-8 against the hp-graded reference, raw (unsymmetrised) swap asymmetry < 1e-9; far/near degree chosen from |k|·distance and h/distance so that the class-boundary error is ≤ the touching error (currently far-degree-3 K errors reach 1e-3 dielectric / 1e-2 Ag on a λ/13 mesh); cost per touching pair ≤ 3× current | review done, fix round committed, merge pending (see docs/handover.md) | `wp/07b-graded-outer` | — |
| WP-F | Decision: formulation weights. Resolved 2026-10-09 by the coordinator from the sources: Table 1 of Fu et al. 2023 is transcribed exactly in `formulation.hpp`; Fu et al. state that the three Jacobi-preconditioned formulations show no obvious difference ("we thus refer them simply as the preconditioned formulation"), which is exactly the consequence of region-independent a_i/η_i, b_i η_i; Karaosmanoğlu et al. 2017 define MCTF as a = b = 1, c = d = η_o η_p, i.e. also a block-row scaling of PMCHWT (identity terms cancel). Only CTF (a = 1/η_o, b = 1/η_p) is region-dependent and is not needed (inaccurate for plasmonics per Karaosmanoğlu). Consequences: code and `recommend` unchanged; direct-solver currents identical for all three, so the Mie validation covers them; differences appear only in unpreconditioned GMRES (WP15) | 3 (decision) | WP9 | `docs/03_theory_sie.md` | Table 1 confirmed against the sources (Solís et al. 2015 Table 1; Karaosmanoğlu & Ergül 2019 Eq. 1) | done (decision by the project lead) | — | [#15](https://github.com/PhotonPower/SpeckleBem/issues/15) |
| WP9a | Single source of truth for ω: `Problem::omega` replaces `SurfaceSolution::omega`; replace the test-local `TestPlaneWave` by `excitation::PlaneWave` | 2 (follow-up) | WP9, WP10 | `include/specklebem/operator/assembler.hpp`, `include/specklebem/postprocessing/fields.hpp`, `src/postprocessing/fields.cpp`, `tests/support/fields_test_support.hpp` | `Problem::omega` exists and `post::*` prefers it (done in WP9); remaining: remove `SurfaceSolution::omega` before the Phase 3 `Simulation` driver; tests unchanged in outcome | done (squash-merged) | `wp/09a-omega` | — |

### Phase 2 schedule

- Wave 1 (parallel, after WP4 is merged): WP7, WP8, WP10.
- Wave 2: WP9 (after WP7 + WP8), then WP11 (after WP9 + WP10). Done: WP7, WP8, WP9, WP10, WP11 merged
  (main, 187 tests). Phase 2 DoD status: dielectric sphere ε_rr 0.074 % at λ/13 (target < 1 % at λ/10) met;
  Ag sphere 0.078 % at λ/13 met, the λ/20 case awaits a large node; symmetry met; Fresnel (WP12) deferred.
- Phase 3 breakdown: see the Phase 3 table.

## Phase 3

| WP | Title | Phase | Depends on | Files | Acceptance criterion | Status | Branch | Issue |
|----|-------|-------|------------|-------|----------------------|--------|--------|-------|
| WP13 | `solver::gmres` (complex, full and restarted, left/right preconditioning) with `IdentityPreconditioner` / `DiagonalPreconditioner` | 3 | WP8, WP9 | `src/solver/gmres.cpp`, `src/solver/preconditioner.cpp`, `tests/unit/test_gmres.cpp`, `tests/validation_large/test_gmres_vs_lu_large.cpp` | Full GMRES agrees with LU on random and dense BEM systems; left-Jacobi iterations invariant under row scaling (issue #15); docs/05 "GMRES vs LU, 2·10⁴ unknowns": Si rough-surface box (2N = 19 968), ICTF unpreconditioned, tol 1e-6, converged with true residual ≤ 1e-6 | review | `wp/13-gmres` | — |

## Platform and tooling

| WP | Title | Phase | Depends on | Files | Acceptance criterion | Status | Branch | Issue |
|----|-------|-------|------------|-------|----------------------|--------|--------|-------|
| WP-W1 | Native Windows build with MSYS2 (`win-release` UCRT64 GCC + OpenBLAS, `win-debug` CLANG64 Clang + ASan/UBSan), `.gitattributes` LF checkout, fetched Eigen as `SYSTEM` for CMake 4, Windows memory guard, Windows CI job, CI on `wp/**` pushes | tooling | — | `CMakePresets.json`, `cmake/Dependencies.cmake`, `.gitattributes`, `.github/workflows/ci.yml`, `tests/support/system_memory.*` | Both win presets warning-free, 192/192 ctest (`-LE validation-large`), 13/13 `^validation$`, pytest green; Linux CI unchanged and green | done (squash-merged) | `wp/w1-windows-msys2` | — |

## Notes for workers (lessons learned)

- Coordinator tooling: `SendMessage` is not available in the Windows sessions, so a finished worker cannot be resumed. Trivial review fixes (docs/comments, < 20 lines) are applied by the coordinator on the WP branch; substantive fixes go to a new worker whose brief contains the original brief, the review findings and the branch head.
- Without system packages, Eigen/spdlog/Catch2/pybind11 are fetched by `cmake/Dependencies.cmake`.
  The fetched Eigen is marked as a system include so that `-Werror` does not trip on Eigen
  internals; do not add Eigen include paths by hand.
- BLAS/LAPACK are optional; the test machine may not have them (Eigen fallback, same results).
- Ubuntu 24.04's `libspdlog-dev` links the system fmt 9.1 whose headers hit a GCC 13 `-Warray-bounds`
  false positive under `-Werror`; use the fetched spdlog (do not install `libspdlog-dev`, or pass
  `-DCMAKE_DISABLE_FIND_PACKAGE_spdlog=ON`). Clang needs `libomp-dev` for OpenMP.
- Eigen's `cross()` conjugates for complex vectors (it is the Hermitian cross product); never use it on
  phasors. Write the cross product of complex vectors by hand or cast to a real/imag split.
- The K operator uses the source-point gradient ∇'G = −`grad_green` (docs/03); E^s = −L J + K M.
- `-Wconversion`/`-Wsign-conversion` are on: use `Index` (int64) for sizes, cast explicitly
  when indexing Eigen with `int` and when mixing `std::size_t` and `Index`.

### Windows (MSYS2, WP-W1)

- Presets `win-release` (UCRT64 GCC, libstdc++, OpenMP, OpenBLAS) and `win-debug` (CLANG64 Clang,
  libc++, ASan + UBSan, `-Werror`); commands and packages in README.md "Windows (MSYS2)". Two
  environments because MinGW GCC has no sanitizers and UCRT64's Clang has no ASan runtime.
  Never mix them: objects/DLLs of UCRT64 (libstdc++) and CLANG64 (libc++) are not ABI compatible.
- Line endings: `.gitattributes` forces LF (`* text=auto eol=lf`, `*.stl -text`). Without it,
  `core.autocrlf=true` checks out CRLF and bash scripts and byte-compared fixtures break. Open
  files that are compared byte by byte with `std::ios::binary` (text mode on Windows writes CRLF).
- DLL lookup: a Windows executable finds its DLLs in its own directory, then on `PATH`. Git for
  Windows puts `/mingw64/bin` (an older `libstdc++-6.dll`) early on `PATH` in Git Bash; a test
  executable that picks it up dies with `0xc0000139` (entry point not found), already at build
  time because `catch_discover_tests` runs the executable. The presets prepend
  `$env{MSYS2_ROOT}/<env>/bin` for configure, build and test; outside the presets, put the MSYS2
  `bin` first yourself. The Python extension's DLLs are found next to MSYS2's `python.exe`; use
  the Python of the same environment as the compiler (`PYTHONPATH` separator is `;` there).
- LLP64: `long` is 32 bit on Windows. Use `Index` / `std::int64_t` / `std::size_t` for sizes and
  byte counts, `std::stoll` rather than `std::stol`, never `%ld`. No POSIX-only headers in shared
  code (`sysconf`, `getrusage`, `<unistd.h>`; `std::aligned_alloc` does not exist on Windows,
  `_aligned_malloc` needs `_aligned_free`). Keep `<windows.h>` out of headers: it defines the
  macros `near` and `far` (collide with `kernels::Proximity`) and `min`/`max`; the platform code
  of the tests lives in `tests/support/system_memory.cpp`.
- Fetched dependencies are declared `SYSTEM` (CMake ≥ 3.25) so their headers do not trip
  `-Werror`. Setting `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` on the fetched Eigen target by hand
  breaks Eigen's `install(EXPORT)` with CMake 4. Clang 22 needs spdlog ≥ 1.15 (fmt 11; fmt 10's
  `FMT_STRING` fails to compile) and Catch2 ≥ 3.14 (`__COUNTER__` is flagged by
  `-Wc2y-extensions` at the macro expansion site, which `SYSTEM` cannot suppress).
- CMake's FindPython also searches the Windows registry: pin `Python3_EXECUTABLE` /
  `PYTHON_EXECUTABLE` to MSYS2's python (the presets and the CI job do).
- Running dense validation tests with `ctest -j 12` oversubscribes the cores (each test runs an
  OpenMP region on all cores): they take ~48 s each in parallel vs 3–6 s serially in win-release.
