# Benchmarks

Target reference points (Fu et al. 2023, 2x Xeon E5-2627, 12 threads):

| Case | N (unknowns) | Formulation | Iterations | Wall time |
|------|-------------:|-------------|-----------:|----------:|
| Ag sphere, d = 4 µm, λ/27 mesh | 393,216 | precond. | 424 | 188 min |
| Ag surface 10×10 µm², σ = 100 nm | ~240,000 | precond. | – | 46 min |
| Si surface 10×10 µm², σ = 100 nm | ~240,000 | ICTF | – | 12 min |
| Si surface 30×30 µm² | 2,160,000 | ICTF | – | (hardware limit) |

Complexity exponents to beat: γ ≈ 1.35 (Si rough), γ ≈ 1.9 (Ag rough).

Each benchmark writes a JSON record (N, levels, iterations, time per matvec,
peak memory) to `benchmarks/results/` so regressions are visible in CI history.

Curated result records (Markdown, tracked in git despite the global `results/` ignore rule):

- [`results/mie_sphere_dense.md`](results/mie_sphere_dense.md) — Mie validation of the dense
  PMCHWT + LU solver (WP11, Phase 2 DoD): ε_rr in the xz- and yz-planes, power balance,
  assembly / LU timings and memory per icosphere level.
- [`results/formulation_convergence.md`](results/formulation_convergence.md) — formulation
  convergence study (WP15, Phase 3 DoD; Fu et al. 2023 Fig. 2a,b): GMRES iteration counts of
  PMCHWT / ICTF / MCTF without and with the left Jacobi preconditioner on Si and Ag Gaussian rough
  surfaces (dense operator, L reduced to 3.2 µm). Histories (every 10th iteration):
  [`results/formulation_convergence_histories.csv`](results/formulation_convergence_histories.csv).
- [`results/dense_assembly_profile.md`](results/dense_assembly_profile.md) — dense assembly
  profile of the WP15 rough boxes and icospheres (WP-P2): time per pair class, region and
  triangle size, near/far degrees, schedule makespan, before/after timings
  (`dense_assembly_profile.cpp`).

- [`results/ag_sphere_4um_mlfmm.md`](results/ag_sphere_4um_mlfmm.md) — Ag sphere d = 4 µm with
  MLFMM vs Mie (WP22a, Phase 4 DoD) on the paper's 393 216-unknown mesh: ε_rr in both planes,
  iterations, setup/solve times, memory per component (near field, Ag exact part, far), ramp-up
  over d = 1 / 2 / 4 µm, tolerance and mesh sensitivity (`ag_sphere_mlfmm.cpp`).

## Executables (`-DSPECKLEBEM_BUILD_BENCHMARKS=ON`)

- `specklebem_ag_sphere_mlfmm` (`ag_sphere_mlfmm.cpp`): the WP22a study, one case per process.
  Options `--mesh octa|ico`, `--n <subdivisions>`, `--d <diameter m>`, `--digits <d0>`,
  `--tol <t>`, `--max-iter <n>`, `--exact-budget-gb <G>`, `--formulation pmchwt|ictf|mctf`,
  `--jacobi 0|1`, `--estimate-only` (mesh, octree and near-field estimate, no assembly). The
  4 µm case (`--mesh octa --n 7 --d 4e-6`) needs ~34 GB and ~16 min on 24 cores.

- `specklebem_formulation_convergence` (`formulation_convergence.cpp`): the WP15 study. Options
  `--materials si,ag`, `--formulations pmchwt,ictf,mctf`, `--L <m>`, `--seed <n>`,
  `--max-iter <n>`, `--tol <t>`, `--out <dir>` (.npy directory with the residual histories, the
  meshes and attributes), `--csv <file>` (appended, `--csv-stride` 10), `--waist-factor`
  (w0 = L / factor, default 3), `--fine-band` (ADR 0006 fine band 3δ + 3σ), `--mesh-only`
  (print the setup and the unknown count, no assembly). Each dense system (2N ≈ 3·10⁴) needs
  ~16 GB; the systems are processed one after the other.
- `specklebem_dense_assembly_profile` (`dense_assembly_profile.cpp`): the WP-P2 profile.
  Options `--mesh si|ag|sphere`, `--material si|ag|glass|vacuum`, `--L <m>`,
  `--box-mesh-size <m>`, `--sphere-n <n>`, `--radius <m>`, `--stride <n>`, `--profile`
  (per-class element_blocks timings and degree statistics), `--build R` (R timed
  DenseStrategy::build runs), `--compare R` (pre-WP-P2 pair loop in the same process),
  `--micro` (single-thread cost per far pair and degree), `--threads T`, `--target <t>`,
  `--no-decay` / `--libm` (switch off `decay_aware_target` / `fast_plain_kernel`),
  `--schedule-threads T`. See `results/dense_assembly_profile.md`.
