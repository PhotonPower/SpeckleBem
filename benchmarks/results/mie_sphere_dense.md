# Mie validation of the dense SIE solver (WP11)

Phase 2 acceptance tests of `docs/01_project_plan.md` and `docs/05_validation.md`.

- Sphere radius 0.5 µm (d = 1 µm) in vacuum, λ = 500 nm, plane wave k̂ = +z, E along x (1 V/m).
- Icosphere meshes (`make_icosphere`), RWG, PMCHWT, dense assembly (`op::DenseStrategy`,
  default `OperatorOptions`), `solver::solve_direct` (Eigen `PartialPivLU`, no BLAS).
- ε_rr of docs/05: E_ref = √σ_Mie, E_sie = √σ_SIE, RMS over 181 angles θ = 0…180°, divided by
  max E_ref; xz-plane (φ = 0, `bistatic_rcs` plane normal +y) and yz-plane (φ = π/2, plane
  normal −x).
- Power balance: P_sca = ∫|F|²/(2η₁) dΩ (Gauss–Legendre 64 in cos θ × 128 φ), P_ext from the
  optical theorem C_ext = −(4π/k₁) Im[F(ẑ)·x̂*] (exp(+jωt)), P_abs = ½ Re ∮ (n̂ × M)·J* dS from
  the RWG currents (exact Dunavant degree 2). Defect = |P_ext − P_sca − P_abs| / P_ext.
- Machine: 4 cores (Intel Xeon @ 2.1 GHz, 4 OpenMP threads), 15.7 GiB RAM, GCC 13.3, release
  preset, **no BLAS/LAPACK** (Eigen only). Commit a291ed3 (branch `wp/11-mie-validation`),
  2026-10-08.
- Memory: peak RSS of the test process (`getrusage`), which includes Z and the LU copy
  (estimate 2 · 16 · (2N)² bytes).

## Bistatic RCS error and timings

| Material | Icosphere n | Triangles | 2N | Mesh size h (mean edge) | ε_rr xz | ε_rr yz | Assembly [s] | LU [s] | rcond | Peak memory [GB] |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| n = 1.5 | 2 | 320 | 960 | λ/3.3 | 0.89 % | 0.89 % | 0.32 | 0.18 | 5.1e-6 | 0.04 |
| n = 1.5 | 3 | 1 280 | 3 840 | λ/6.6 | 0.27 % | 0.28 % | 2.0 | 6.7 | 3.6e-7 | 0.50 |
| n = 1.5 | 4 | 5 120 | 15 360 | λ/13.2 | **0.074 %** | **0.076 %** | 25.2 | 303 | 1.4e-7 | 7.65 |
| Ag | 2 | 320 | 960 | λ/3.3 | 1.95 % | 1.51 % | 0.22 | 0.16 | 3.2e-7 | 0.04 |
| Ag | 3 | 1 280 | 3 840 | λ/6.6 | 0.35 % | 0.25 % | 2.0 | 6.6 | 9.8e-8 | 0.50 |
| Ag | 4 | 5 120 | 15 360 | λ/13.2 | **0.078 %** | **0.057 %** | 24.4 | 304 | 7.5e-8 | 7.65 |
| Ag | 5 | 20 480 | 61 440 | λ/26.5 | skipped (needs ~121 GB) | | | | | |
| Si | 3 | 1 280 | 3 840 | λ/6.6 | 0.58 % | 0.42 % | 2.1 | 6.5 | 4.1e-7 | 0.50 |

LU residuals |Zx − b|/|b| ≤ 2.5e-15 in all cases. ε_rr decreases monotonically n = 2 → 3 → 4
in both planes (roughly O(h²): factor 3.2–3.7 per refinement). DoD status: dielectric
"λ/10" (n = 4, λ/13) ε_rr < 1 % — met; Ag "λ/20 PMCHWT + LU" — the λ/20 mesh (n = 5) does not
fit on this machine; at λ/13 ε_rr is already 0.08 %.

## Power balance (cross sections vs Mie)

Mie: Ag C_ext = 2.4676e-12 m², C_sca = 2.4277e-12 m², C_abs = 3.989e-14 m² (1.6 % of C_ext);
Si C_ext = 1.9563e-12, C_sca = 1.3544e-12, C_abs = 6.019e-13; n = 1.5 C_ext = C_sca = 1.8468e-12.

| Material | n | Defect | C_ext vs Mie | C_sca vs Mie | C_abs (surface) vs Mie | C_ext − C_sca vs Mie C_abs |
|---|---:|---:|---:|---:|---:|---:|
| n = 1.5 | 3 | 0.22 % | +0.20 % | +0.12 % | 5.5e-15 m² (0.3 % of C_ext) | 1.5e-15 m² |
| n = 1.5 | 4 | 0.024 % | +0.085 % | +0.076 % | 6.0e-16 m² (0.03 % of C_ext) | 1.6e-16 m² |
| Ag | 2 | 50 % | −4.0 % | −6.3 % | −1.09e-12 m² (−2840 %) | +134 % |
| Ag | 3 | 2.3 % | −0.89 % | −0.91 % | −1.59e-14 m² (−140 %) | +0.32 % |
| Ag | 4 | **0.16 %** | −0.18 % | −0.18 % | 3.58e-14 m² (−10.3 %) | −0.09 % |
| Si | 3 | **0.42 %** | −1.0 % | −1.5 % | −1.2 % | +0.15 % |

The absorbed power of the Ag sphere is only 1.6 % of the extinguished power (Im ε_r = −0.313),
so the surface integral ½ Re ∮ (n̂ × M)·J* is a small in-phase part of large reactive surface
fields and converges slowly with the RWG discretisation; the far-field quantities and
C_ext − C_sca converge much faster. The Ag defect at n = 3 does not change when the quadrature
degrees are raised (far/near/singular 8/12/14, near factor 6: defect 2.4 %), i.e. it is
discretisation, not assembly, error. The projected exact Mie currents give C_abs −27 % / −6.4 %
/ −1.6 % of Mie at n = 2 / 3 / 4, consistent with this.

## Reproduce

```bash
cmake --preset release && cmake --build --preset release --parallel 3
ctest --preset release -L '^validation$'      # n = 2, 3 cases (~70 s for the whole label)
ctest --preset release -L validation-large -V # n = 4 cases (~6 min each, 7.7 GB), n = 5 skips < ~200 GB RAM
# single cases with all reported values:
build/release/tests/specklebem_validation_tests "[mie_dense]"
build/release/tests/specklebem_validation_large_tests "[mie_dense]"
```

Tests: `tests/validation/test_mie_sphere_dense.cpp`,
`tests/validation_large/test_mie_sphere_dense_large.cpp`, helpers in
`tests/support/mie_dense_test_support.hpp`.
