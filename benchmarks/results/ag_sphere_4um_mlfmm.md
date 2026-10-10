# Ag sphere d = 4 µm with MLFMM vs Mie (WP22a, Phase 4 DoD)

Phase 4 acceptance test of `docs/01_project_plan.md` ("Ag sphere d = 4 µm, λ/27 mesh (393 k
unknowns): ε_rr ≤ 0.5 % in both scattering planes (paper: 0.26 % / 0.37 %)") and the
`docs/05_validation.md` row "Ag sphere d = 4 µm, λ/27, MLFMM".

**Verdict: met.** ε_rr = **0.023 % (xz) / 0.016 % (yz)** on the paper's 393 216-unknown mesh at
the paper's GMRES tolerance 1e-3 (criterion ≤ 0.5 %; Fu et al. 2023: 0.26 % / 0.37 %), 442
GMRES iterations (paper: 424), 15.6 min wall time (paper: 188 min on 12 threads), peak working
set 33.8 GB.

## Setup

- Sphere centred at the origin in vacuum, λ = 500 nm, Ag ε_r = −9.794 − 0.313j
  (`material::silver_500nm`, exp(+jωt)), plane wave k̂ = +z, E along x, 1 V/m (docs/06).
- `Simulation` with compression `"mlfmm"`, `accuracy_digits` d₀ = 3, automatic leaf rule (λ/4
  leaves, r_max/a ≤ 0.6) and automatic exact-part budget (2 × near field), default kernel options;
  formulation per `formulation::recommend` (ICTF + left Jacobi; with the Jacobi preconditioner
  PMCHWT, ICTF and MCTF are the same system, docs/03 / issue #15, so this is the paper's
  "preconditioned" formulation); full GMRES (no restart), x₀ = 0, tolerance on the monitored
  (left-preconditioned) relative residual.
- ε_rr of docs/05: E_ref = √σ_Mie, E_sie = √σ_SIE, RMS over θ = 0…180° in 181 points (1°), divided
  by max E_ref; xz-plane (φ = 0, `bistatic_rcs` plane normal +y) and yz-plane (φ = π/2, plane
  normal −x). Mie: `reference::MieSolution` (Bohren–Huffman, conjugated to exp(+jωt)). As a
  sampling check also over 1801 points (0.1°): the values agree to ≤ 6 % (relative) with the
  181-point ones (table "ε_rr 1801").
- Meshes: the paper's 131 072 triangles / 393 216 unknowns are 8 · 4⁷, i.e. an
  **octahedron-based** sphere, not an icosphere (20 · 4ⁿ: n = 6 gives 2N = 245 760, n = 7
  2N = 983 040). The study builds it (`make_octasphere` in
  `tests/support/ag_sphere_mlfmm_support.hpp`, midpoint subdivision with projection as
  `make_icosphere`): mean edge **30.25 nm = λ/16.5** (min 24.5 nm, max 38.3 nm = λ/13.1).
  The paper's "λ/27" does not correspond to the mean edge of this mesh at λ = 500 nm (an h of
  λ/27 = 18.5 nm needs the icosphere n = 7 with 983 040 unknowns, whose near field alone is
  ~76 GB at λ/4 leaves); √(triangle area) is 19.6 nm = λ/25.5, which is probably the paper's
  measure. Mesh sensitivity is covered by the icosphere n = 6 (λ/13.2) on the same sphere.
- Machine: Windows 11, 24 cores, 128 GB RAM (shared: another study, WP-V1, ran dense systems of
  up to ~35–45 GB throughout), MSYS2 UCRT64 GCC 16.1, OpenBLAS, `win-release`, 24 OpenMP threads.
  Branch `wp/22a-ag-sphere-4um`, 2026-10-10. Timings are indicative (the WP-V1 study ran
  concurrently; runs marked † overlapped with builds or test runs of this WP).
- Executable: `specklebem_ag_sphere_mlfmm` (`benchmarks/ag_sphere_mlfmm.cpp`,
  `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`), one case per process, e.g.
  `specklebem_ag_sphere_mlfmm --mesh octa --n 7 --d 4e-6 --tol 1e-3`; `--estimate-only` prints
  the mesh, octree and near-field estimate without assembling. validation-large test:
  `tests/validation_large/test_ag_sphere_mlfmm_large.cpp` (d = 1 µm icosphere n = 5, and the
  4 µm case, memory-guarded on the *available* memory).

## Results

ε_rr in %, "assembly" = `Simulation::assemble` (octree, far setup incl. exact part, near field,
right-hand side, Jacobi diagonal), "solve" = GMRES wall time, memory in GB (operator parts as
stored; peak = process peak working set).

| d [µm] | Mesh | Triangles | 2N | h (mean edge) | tol | ε_rr xz | ε_rr yz | ε_rr 1801 xz / yz | GMRES it | Assembly [s] | Solve [s] (s/it) | Near | Exact R2 | Far | Peak |
|---:|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | ico n = 4 | 5 120 | 15 360 | λ/13.2 | 1e-3 | 0.192 | 0.145 | 0.192 / 0.145 | 74 | 6.1 | 10.9 (0.15) | 0.30 | 0.18 | 0.12 | 1.19* |
| 1 | ico n = 4 | 5 120 | 15 360 | λ/13.2 | 1e-6 | **0.0785** | **0.0570** | 0.0777 / 0.0559 | 316 | 5.6 | 45.7 (0.14) | 0.30 | 0.18 | 0.12 | 1.19* |
| 1 | octa n = 5 | 8 192 | 24 576 | λ/16.5 | 1e-3 | 0.141 | 0.096 | 0.141 / 0.096 | 114 | 4.9 | 7.9 (0.07) | 0.75 | 0.40 | 0.17 | 2.09 (2.84*) |
| 1 | ico n = 5 | 20 480 | 61 440 | λ/26.5 | 1e-3 | 0.167 | 0.154 | 0.166 / 0.153 | 100 | 22.3 | 41.0 (0.41) | 4.70 | 2.12 | 0.41 | 16.6* |
| 1 | ico n = 5 | 20 480 | 61 440 | λ/26.5 | 1e-5 | **0.0185** | **0.0133** | 0.0183 / 0.0130 | 388 | 28.3† | 124 (0.32)† | 4.70 | 2.12 | 0.41 | 11.9 |
| 2 | octa n = 6 | 32 768 | 98 304 | λ/16.5 | 1e-3 | 0.0388 | 0.0342 | 0.0384 / 0.0337 | 282 | 45.4 | 225 (0.80) | 3.07 | 1.65 | 0.70 | 11.5* |
| 4 | ico n = 6 | 81 920 | 245 760 | λ/13.2 | 1e-3 | 0.0416 | 0.0300 | 0.0406 / 0.0283 | 332 | 51.9† | 201 (0.61)† | 4.73 | 2.91 | 1.91 | 14.0 |
| **4** | **octa n = 7** | **131 072** | **393 216** | **λ/16.5** | **1e-3** | **0.0228** | **0.0155** | 0.0224 / 0.0148 | **442** | **91.4** | **780 (1.76)** | **12.24** | **6.76** | **2.84** | **33.8** |
| 4 | octa n = 7 | 131 072 | 393 216 | λ/16.5 | 1e-4 | TOL4_XZ | TOL4_YZ | TOL4_FINE | TOL4_IT | TOL4_ASM† | TOL4_SOLVE† | 12.24 | 6.76 | 2.84 | TOL4_PEAK |

\* measured before the `SparseOperator` copy fix of this WP (see "Memory"): the peak contained a
third copy of the near field; the d = 1 µm octa n = 5 case was repeated after the fix (2.84 →
2.09 GB, identical ε_rr and iterations).

Checks against the existing records:
- d = 1 µm, icosphere n = 4 at tol 1e-6 reproduces the dense PMCHWT + LU values of
  `mie_sphere_dense.md` (0.0784 % / 0.0569 %) to the third digit: the MLFMM (d₀ = 3) adds no
  visible error to ε_rr.
- d = 1 µm at λ/26.5 (icosphere n = 5, the docs/05 "λ/20" case whose dense LU needs ~121 GB):
  0.0185 % / 0.0133 % at tol 1e-5, 4.2× below the λ/13 values (O(h²)); the solve takes ~2.5 min
  and 11.9 GB (validation-large test).

## Big run in detail (d = 4 µm, octa n = 7, tol 1e-3)

- Octree: 6 levels, λ/4 leaves (125 nm), 4544 leaves, r_max = 33.1 nm (r_max/a = 0.27).
- R1 (vacuum): expansion on all far levels (λ/4 … 2λ boxes).
- R2 (Ag, α = −Im k₂ = 39.2 µm⁻¹, x*(3)/α = 0.30 µm): no expansion level; per box pair
  truncation below δ = 10⁻⁴ (742 084 380 basis pairs dropped, logged) or exact evaluation:
  189 972 exact box pairs, 169 070 700 basis pairs (860 per basis function) = **6.76 GB**
  (40 B/pair). The box-pair bound (911 M pairs, 36.4 GB) exceeded the automatic budget
  (2 × the 12.24 GB near-field estimate = 24.5 GB), so the per-basis count decided (169 M
  pairs, 6.76 GB); **the budget did not fire** and `max_exact_far_bytes` was left automatic.
  (The library log prints these in MiB: 34 759 / 23 344 / 6 451 MB.)
- Near field: 509 825 424 entries (1297 per row) = **12.24 GB**.
- Far tables and one apply workspace: 2.84 GB; operator total 21.85 GB; full-GMRES Krylov basis
  (443 vectors) 2.79 GB; **peak working set 33.8 GB** (reached while the near field is moved into
  its operator, see below).
- Times: MLFMM operator 71.3 s (far setup incl. exact part 13.5 s, of which exact assembly
  8.5 s; near field 57.8 s), Jacobi diagonal 20.1 s, right-hand side 0.04 s; GMRES 442 it in
  780 s (1.76 s/it including the Gram–Schmidt growth: 1.0 s/it over the first 100 iterations);
  RCS evaluation (181 + 1801 angles, both planes) 24.5 s. Total ≈ 15.6 min.
- GMRES history (monitored residual): 3.3e-2 (50), 1.7e-2 (100), 6.4e-3 (200), 2.6e-3 (300),
  1.2e-3 (400), 1.0e-3 (442); true residual 1.04e-3.

## Scaling at fixed mesh size (octa, h = λ/16.5, tol 1e-3)

| d [µm] | 2N | near / row | exact pairs / basis | near [GB] | exact [GB] | it | s/it |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 576 | 1275 | 820 | 0.75 | 0.40 | 114 | 0.07 |
| 2 | 98 304 | 1303 | 840 | 3.07 | 1.65 | 282 | 0.80 |
| 4 | 393 216 | 1297 | 860 | 12.24 | 6.76 | 442 | 1.76 |

Near field and exact part grow linearly with N at fixed h (the exact interactions reach
x*/α = 0.30 µm, independent of the sphere size). The ADR 0008 WP21 projection for this case
(near ≈ 15 GB, exact ≈ 15 GB at d₀ = 3) was conservative: 12.2 GB and 6.8 GB measured.

## Sensitivities

- **GMRES tolerance.** At d = 1 µm the tolerance 1e-3 dominates ε_rr (0.19 % vs 0.078 % at
  1e-6 on λ/13; 0.17 % vs 0.019 % at 1e-5 on λ/26.5). At d = 4 µm (ε_rr is normalised by the
  forward peak, which grows with the size) the tolerance-1e-3 value is 0.023 % / 0.016 % and
  TOL4_SENTENCE
- **Mesh.** d = 4 µm at λ/13.2 (icosphere n = 6, 2N = 245 760): 0.042 % / 0.030 %; at λ/16.5
  (octa n = 7): 0.023 % / 0.016 % — both an order of magnitude below the criterion.
- **d₀ = 5** was not run: with the d₀ = 5 leaf rule (λ/2 leaves) the near-field estimate is
  48.9 GB (`--estimate-only`), plus an exact part of ~2× the d₀ = 3 one and the transient near
  copy, i.e. ≳ 110 GB peak, beyond what this shared 128 GB machine had available (≈ 55–68 GB).
  It is not needed for the DoD (the d₀ = 3 MLFMM error is invisible in ε_rr, see the d = 1 µm
  tol 1e-6 check).

## Memory: SparseOperator copy fix

The ramp-up showed peak working sets of 3.5–3.8 × the near field. Cause: Eigen 3.4.0 (the
FetchContent version) has no move constructor for `SparseMatrix`, so
`SparseOperator(Matrix Z) : Z_(std::move(Z))` copied the matrix, and three copies of the near field
were alive at the end of `assemble_sparse` (local, by-value parameter, member). The constructor
now swaps (`Z_.swap(Z)`), leaving two transient copies (the by-value parameter is still copied
from the caller's matrix; removing it needs a signature change). At 2N = 393 216 the fix saves
12 GB of peak memory.

## Comparison with the paper

| | Fu et al. 2023 | SpeckleBem (this record) |
|---|---|---|
| Unknowns | 393 216 | 393 216 (same 8 · 4⁷-triangle sphere) |
| Formulation | preconditioned | ICTF + left Jacobi (= PMCHWT + Jacobi) |
| GMRES tolerance | 1e-3 | 1e-3 |
| Iterations | 424 | 442 |
| ε_rr (two planes) | 0.26 % / 0.37 % | 0.023 % (xz) / 0.016 % (yz) |
| Wall time | 188 min (2 × Xeon E5-2627, 12 threads) | 15.6 min (24 cores) |
| Memory | — | 21.9 GB operator, 33.8 GB peak |

The paper does not say which plane is "normal" and which "parallel"; both of our values are
an order of magnitude below either of its values.
