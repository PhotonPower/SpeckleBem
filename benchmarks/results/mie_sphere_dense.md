# Mie validation of the dense SIE solver (WP11, re-measured after WP7b)

Phase 2 acceptance tests of `docs/01_project_plan.md` and `docs/05_validation.md`.

- Sphere radius 0.5 µm (d = 1 µm) in vacuum, λ = 500 nm, plane wave k̂ = +z, E along x (1 V/m).
- Icosphere meshes (`make_icosphere`), RWG, PMCHWT, dense assembly (`op::DenseStrategy`,
  default `OperatorOptions`: since WP7b graded outer rule for touching pairs with
  `outer_grading_levels` 4, near/far degree selection with `target_accuracy` 1e-5 in
  [3, 19], ADR 0004), `solver::solve_direct` (Eigen `PartialPivLU`).
- ε_rr of docs/05: E_ref = √σ_Mie, E_sie = √σ_SIE, RMS over 181 angles θ = 0…180°, divided by
  max E_ref; xz-plane (φ = 0, `bistatic_rcs` plane normal +y) and yz-plane (φ = π/2, plane
  normal −x).
- Power balance: P_sca = ∫|F|²/(2η₁) dΩ (Gauss–Legendre 64 in cos θ × 128 φ), P_ext from the
  optical theorem C_ext = −(4π/k₁) Im[F(ẑ)·x̂*] (exp(+jωt)), P_abs = ½ Re ∮ (n̂ × M)·J* dS from
  the RWG currents (exact Dunavant degree 2). Defect = |P_ext − P_sca − P_abs| / P_ext.
- Machine (this record): Windows 11, 24 cores, 128 GB RAM, MSYS2 UCRT64 GCC 16.1, OpenBLAS
  0.3.33 (`EIGEN_USE_BLAS`, the LU's block products run in OpenBLAS with its default
  threading), `win-release` preset, 24 OpenMP threads for the assembly (default
  `omp_get_max_threads()`). Branch `wp/07b-graded-outer` after merging main 0bcc97d,
  2026-10-09.
- Timings are indicative: other agents' builds and tests shared the machine during some runs.
  The table gives two runs on an otherwise idle machine (no compiler or test process running
  at start and end) as "run A / run B". Ag n = 4: one of the idle runs gave 22.5 s assembly
  (disturbed), a separate repeat 10.6 s; the table shows the repeat and the other idle run.
  Two further runs under load took up to ~9× (assembly) and ~90× (LU of 2N = 960) longer.
  ε_rr, rcond, residuals and the power balance are identical in all runs (the assembly is
  bitwise thread-count independent).
- Memory: peak working set of the test process (`peak_rss_bytes`, `PeakWorkingSetSize` on
  Windows), which includes Z and the LU copy (estimate 2 · 16 · (2N)² bytes); the n = 2 / 3 /
  4 values are each size's process-wide peak.

## Bistatic RCS error and timings

| Material | Icosphere n | Triangles | 2N | Mesh size h (mean edge) | ε_rr xz | ε_rr yz | Assembly [s] | LU [s] | rcond | Peak memory [GB] |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| n = 1.5 | 2 | 320 | 960 | λ/3.3 | 0.87 % | 0.88 % | 0.19 / 0.18 | 0.11 / 0.10 | 5.1e-6 | 0.05 |
| n = 1.5 | 3 | 1 280 | 3 840 | λ/6.6 | 0.263 % | 0.274 % | 1.09 / 1.05 | 2.4 / 3.1 | 3.6e-7 | 0.51 |
| n = 1.5 | 4 | 5 120 | 15 360 | λ/13.2 | **0.073 %** | **0.074 %** | 7.5 / 7.6 | 39 / 47 | 1.4e-7 | 7.62 |
| Ag | 2 | 320 | 960 | λ/3.3 | 1.93 % | 1.37 % | 0.31 / 0.32 | 0.12 / 0.11 | 3.0e-7 | 0.05 |
| Ag | 3 | 1 280 | 3 840 | λ/6.6 | 0.348 % | 0.256 % | 1.29 / 1.27 | 2.4 / 3.1 | 9.8e-8 | 0.51 |
| Ag | 4 | 5 120 | 15 360 | λ/13.2 | **0.078 %** | **0.057 %** | 10.6 / 11.2 | 37 / 47 | 7.5e-8 | 7.62 |
| Ag | 5 | 20 480 | 61 440 | λ/26.5 | skipped (needs ~121 GB > 60 % of 128 GB) | | | | | |
| Si | 3 | 1 280 | 3 840 | λ/6.6 | 0.554 % | 0.396 % | 1.90 / 1.80 | 2.5 / 3.2 | 4.0e-7 | 0.51 |

LU residuals |Zx − b|/|b| ≤ 2.7e-15 in all cases. ε_rr decreases monotonically n = 2 → 3 → 4
in both planes (roughly O(h²): factor 3.3–5.5 per refinement). DoD status: dielectric
"λ/10" (n = 4, λ/13) ε_rr < 1 % — met; Ag "λ/20 PMCHWT + LU" — the λ/20 mesh (n = 5) needs
~121 GB and SKIPs by the 60 % memory guard on this 128 GB machine; at λ/13 ε_rr is already
0.08 %.

## Power balance (cross sections vs Mie)

Mie: Ag C_ext = 2.4676e-12 m², C_sca = 2.4277e-12 m², C_abs = 3.989e-14 m² (1.6 % of C_ext);
Si C_ext = 1.9563e-12, C_sca = 1.3544e-12, C_abs = 6.019e-13; n = 1.5 C_ext = C_sca = 1.8468e-12.

| Material | n | Defect | C_ext vs Mie | C_sca vs Mie | C_abs (surface) vs Mie | C_ext − C_sca vs Mie C_abs |
|---|---:|---:|---:|---:|---:|---:|
| n = 1.5 | 3 | 0.22 % | +0.17 % | +0.17 % | 4.0e-15 m² (0.22 % of C_ext) | 1.2e-17 m² |
| n = 1.5 | 4 | 0.026 % | +0.082 % | +0.082 % | 4.7e-16 m² (0.026 % of C_ext) | 1.3e-19 m² |
| Ag | 2 | 48 % | −3.6 % | −5.8 % | −1.05e-12 m² (−2730 %) | +128 % |
| Ag | 3 | 2.4 % | −0.89 % | −0.91 % | −1.90e-14 m² (−148 %) | −0.07 % |
| Ag | 4 | **0.18 %** | −0.18 % | −0.18 % | 3.53e-14 m² (−11.4 %) | −0.15 % |
| Si | 3 | **0.007 %** | −0.87 % | −1.17 % | −0.21 % | −0.18 % |

The absorbed power of the Ag sphere is only 1.6 % of the extinguished power (Im ε_r = −0.313),
so the surface integral ½ Re ∮ (n̂ × M)·J* is a small in-phase part of large reactive surface
fields and converges slowly with the RWG discretisation; the far-field quantities and
C_ext − C_sca converge much faster. The Ag defect at n = 3 does not change when the quadrature
degrees are raised (WP11: far/near/singular 8/12/14, near factor 6: defect 2.4 %; WP7b graded
touching rule: 2.4 %), i.e. it is discretisation, not assembly, error. The projected exact
Mie currents give C_abs −27 % / −6.4 % / −1.6 % of Mie at n = 2 / 3 / 4, consistent with this.
The Si balance, by contrast, improved with the WP7b near-field accuracy (defect 0.42 % →
0.007 %, C_abs −1.2 % → −0.2 %).

## History: WP11 record (WP7 kernel options, other machine)

Commit a291ed3 (branch `wp/11-mie-validation`), 2026-10-08; 4 cores (Intel Xeon @ 2.1 GHz,
4 OpenMP threads), 15.7 GiB RAM, GCC 13.3, Linux `release` preset, **no BLAS/LAPACK** (Eigen
only); kernel options of WP7 (one Dunavant outer rule for touching pairs, fixed far / near
degrees 3 / 8). Not comparable in time with the record above (different machine, threads and
LU backend).

| Material | n | ε_rr xz | ε_rr yz | Assembly [s] | LU [s] | Defect |
|---|---:|---:|---:|---:|---:|---:|
| n = 1.5 | 2 | 0.89 % | 0.89 % | 0.32 | 0.18 | — |
| n = 1.5 | 3 | 0.27 % | 0.28 % | 2.0 | 6.7 | 0.22 % |
| n = 1.5 | 4 | 0.074 % | 0.076 % | 25.2 | 303 | 0.024 % |
| Ag | 2 | 1.95 % | 1.51 % | 0.22 | 0.16 | 50 % |
| Ag | 3 | 0.35 % | 0.25 % | 2.0 | 6.6 | 2.3 % |
| Ag | 4 | 0.078 % | 0.057 % | 24.4 | 304 | 0.16 % |
| Si | 3 | 0.58 % | 0.42 % | 2.1 | 6.5 | 0.42 % |

With the WP7b defaults the per-pair assembly cost rose (ADR 0004: n = 3 assembly 6.6 / 7.7 /
11.8 s for n = 1.5 / Ag / Si on 3 threads of a Linux machine, against 2.7 s with the WP7
options); on the 24-thread machine above it is 1.1 / 1.3 / 1.9 s.

## Reproduce

```bash
cmake --preset release && cmake --build --preset release      # Windows: win-release
ctest --preset release -L '^validation$'      # n = 2, 3 cases
ctest --preset release -L validation-large -V # n = 4 cases (7.6 GB each), n = 5 skips < ~200 GB RAM
# single cases with all reported values:
build/release/tests/specklebem_validation_tests "[mie_dense]"
build/release/tests/specklebem_validation_large_tests "[mie_dense]"
```

Tests: `tests/validation/test_mie_sphere_dense.cpp`,
`tests/validation_large/test_mie_sphere_dense_large.cpp`, helpers in
`tests/support/mie_dense_test_support.hpp`.
