# Formulation convergence study, dense operator (WP15, Phase 3 DoD)

Reproduction attempt of Fu et al. 2023, Sec. 4 / Fig. 2(a,b): GMRES convergence of PMCHWT,
ICTF and MCTF without preconditioner and with the (left) Jacobi preconditioner on Gaussian
rough surfaces of Ag and Si at λ = 500 nm. Executable `benchmarks/formulation_convergence.cpp`
(`specklebem_formulation_convergence`, built with `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`); residual
histories (every 10th iteration plus the last) in
[`formulation_convergence_histories.csv`](formulation_convergence_histories.csv).

## Setup

| Parameter | Value |
|---|---|
| Wavelength | λ = 500 nm, vacuum background (R1) |
| Materials (R2) | Si ε_r = 18.478 − 0.606j, Ag ε_r = −9.794 − 0.313j (exp(+jωt)) |
| Surface | Gaussian height map (Eqs. 8–9), σ = 50 nm, Lc = 500 nm, seed 1, `use_fft` |
| Patch | **L = 3.2 µm** (paper: 10 µm), 65 × 65 grid, h = 50 nm (λ/10); realised rms 58.7 nm, estimated Lc 606 nm (a 6.4 λ patch holds only ~6 × 6 correlation areas) |
| Closing box (ADR 0006) | depth `default_box_depth`: Si 5.65 µm (10 intensity absorption lengths, δ = 1.129 µm), Ag 2 µm (minimum, δ = 25.4 nm); graded with the automatic coarse spacing (M = 3, 400 nm), **no fine band** |
| Mesh, Si | 10 400 triangles (top face 8 192, box 2 208 = 27.0 %), N = 15 600, **2N = 31 200**, Z = 15.6 GB |
| Mesh, Ag | 9 760 triangles (top face 8 192, box 1 568 = 19.1 %), N = 14 640, **2N = 29 280**, Z = 13.7 GB |
| Excitation | paraxial `GaussianBeam`, normal incidence (+z, from z < 0), w₀ = L/3 = 1.067 µm, focus at the origin (z = 0), p-polarised (E along x), 1 V/m; Maxwell residual reported by the class: (λ/(π w₀))² = 0.0223 |
| Discretisation | RWG Galerkin, default `OperatorOptions` (WP7b: graded outer rule, target accuracy 1e-5) |
| Solver | full GMRES (no restart), x₀ = 0, tolerance 1e-3 on the monitored relative residual, max_iter 2000; Jacobi = `op::assemble_diagonal`, left side (the paper's choice), so the monitored residual of the Jacobi runs is the preconditioned one |

One `Simulation` per material and formulation assembles Z once; the unpreconditioned solve is
`Simulation::solve`, the Jacobi solve `solver::gmres` with a `DiagonalPreconditioner` on the same
operator. All systems were solved one after the other.

Machine and build: Windows 11, 24 cores, 128 GB RAM, MSYS2 UCRT64 GCC 16.1.0, OpenBLAS 0.3.33,
`win-release` preset with `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`, 24 OpenMP threads. Code: branch
`wp/15-convergence-study` after merging main **889fb51** (WP-P1; WP7c not yet included),
2026-10-09. Other agents' builds and validation tests shared the machine during all runs, so the
times are indicative only (iteration counts and residuals do not depend on the load).

## Results

| Material | Formulation | Preconditioner | Iterations | Monitored residual | True residual \|b − Zx\|/\|b\| | Assembly [s] | Solve [s] | s / it |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Si | PMCHWT | none | 315 | 9.99e-4 | 9.99e-4 | 1535 | 207 | 0.66 |
| Si | ICTF | none | **219** | 9.97e-4 | 9.97e-4 | 1513 | 137 | 0.62 |
| Si | MCTF | none | 254 | 9.91e-4 | 9.91e-4 | 1535 | 147 | 0.58 |
| Si | PMCHWT | Jacobi | 272 | 9.99e-4 | 3.11e-2 | (same Z) | 187 | 0.69 |
| Si | ICTF | Jacobi | 272 | 9.99e-4 | 4.30e-3 | (same Z) | 195 | 0.72 |
| Si | MCTF | Jacobi | 272 | 9.99e-4 | 4.31e-3 | (same Z) | 162 | 0.60 |
| Ag | PMCHWT | none | 886 | 1.00e-3 | 1.00e-3 | 1163 | 536 | 0.60 |
| Ag | ICTF | none | **622** | 9.97e-4 | 9.97e-4 | 1056 | 307 | 0.49 |
| Ag | MCTF | none | 651 | 9.97e-4 | 9.97e-4 | 1240 | 352 | 0.54 |
| Ag | PMCHWT | Jacobi | 1069 | 9.95e-4 | 9.20e-3 | (same Z) | 778 | 0.73 |
| Ag | ICTF | Jacobi | 1069 | 9.95e-4 | 1.82e-3 | (same Z) | 695 | 0.65 |
| Ag | MCTF | Jacobi | 1069 | 9.95e-4 | 1.82e-3 | (same Z) | 585 | 0.55 |

All twelve runs converged to 1e-3 (none reached max_iter). The Jacobi diagonal took < 1 s.
"True residual" is measured in each formulation's own (unscaled) system; for the Jacobi runs
the iterates are identical across the formulations, so their true residuals differ only by the
row weighting (PMCHWT's rows are less balanced, hence 3.1e-2 / 9.2e-3 against 4.3e-3 / 1.8e-3).

Monitored residual at selected iterations:

| Run | 10 | 50 | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| Si PMCHWT | 1.36e-1 | 4.11e-2 | 1.37e-2 | 3.12e-3 | — | — |
| Si ICTF | 8.42e-2 | 2.11e-2 | 5.46e-3 | 1.22e-3 | — | — |
| Si MCTF | 1.08e-1 | 2.93e-2 | 1.05e-2 | 1.80e-3 | — | — |
| Si Jacobi (all three) | 6.52e-2 | 1.43e-2 | 5.79e-3 | 1.64e-3 | — | — |
| Ag PMCHWT | 1.19e-1 | 4.83e-2 | 1.90e-2 | 7.69e-3 | 1.66e-3 | — |
| Ag ICTF | 1.14e-1 | 4.45e-2 | 2.44e-2 | 7.67e-3 | 1.48e-3 | — |
| Ag MCTF | 9.64e-2 | 3.09e-2 | 1.83e-2 | 7.20e-3 | 1.67e-3 | — |
| Ag Jacobi (all three) | 2.26e-1 | 1.29e-1 | 9.37e-2 | 6.21e-2 | 1.97e-2 | 1.22e-3 |

No history shows a plateau; all decrease steadily.

## Issue #15: the three Jacobi-preconditioned formulations

ICTF and MCTF are block-row scalings of PMCHWT (docs/03), so left Jacobi makes the three
systems identical. Measured over the full histories (`max_k |r_k − r_k^PMCHWT| / r_k^PMCHWT`):

| Material | ICTF+Jacobi vs PMCHWT+Jacobi | MCTF+Jacobi vs PMCHWT+Jacobi | Iterations |
|---|---:|---:|---|
| Si | 4.9e-13 | 4.5e-13 | 272 / 272 / 272 |
| Ag | 2.4e-13 | 9.9e-14 | 1069 / 1069 / 1069 |

Identical to rounding, as predicted (and as Fu et al. observe: "no obvious difference"). The Si
values come from the `ISSUE15` lines of the single Si process; the Ag formulations ran in separate
processes, so the Ag values were computed afterwards from the full-precision `.npy` residual
histories of the three runs (the CSV keeps only 6 digits).

## Fu et al. 2023, Fig. 2 statements

1. **PMCHWT does not converge** (both materials) — **not reproduced** at this size: PMCHWT
   converges in 315 (Si) and 886 (Ag) iterations. It is, however, the slowest unpreconditioned
   formulation for both materials (1.44× ICTF for Si, 1.42× ICTF for Ag).
2. **ICTF converges fastest for Si without preconditioner** — **reproduced**: ICTF 219 against
   MCTF 254, Jacobi 272 and PMCHWT 315 iterations.
3. **The (Jacobi-)preconditioned formulation is best for Ag** — **not reproduced** (and the
   comparison favours Jacobi: its monitored residual is the preconditioned one; its true residuals
   at stop are 1.8–31× larger): Jacobi needs
   1069 iterations against ICTF 622, MCTF 651 and PMCHWT 886; it is behind from the start
   (9.4e-2 against 1.8–2.4e-2 after 100 iterations).
4. **MCTF similar to ICTF for Si** — **reproduced**: 254 against 219 iterations (+16 %), with
   similar histories.
5. **MCTF much slower than ICTF for Ag** — **not reproduced**: 651 against 622 iterations
   (+5 %).
6. **The three preconditioned formulations behave alike** — **reproduced** exactly (identical
   histories to ≤ 5e-13, see issue #15 above).

Possible reasons for the differences (not investigated in WP15): the patch is 3.1× smaller
than the paper's (2N ≈ 3·10⁴ against roughly 2.4·10⁵ for L = 10 µm; iteration counts and the
PMCHWT ill-conditioning grow with the problem size); the closing box is a much larger part of
the mesh (19–27 % against ~7 % at L = 10 µm) and is graded with 400 nm cells on the lower walls
and the bottom plate, whose diagonal entries differ strongly from the 50 nm top-face entries
(this affects Jacobi in particular); a 6.4 λ patch has only a few correlation areas, so the
realised surface statistics deviate from the nominal ones; the paraxial beam; and Fu et al.'s
own discretisation and quadrature details. For Ag, surface plasmons excited by the roughness propagate ~20 µm on flat Ag (ε = −9.794 − 0.313j)
— far more than L — and reach the side walls (relevant to WP-V1). A size sweep (MLFMM, Phase 4) is needed to see
whether PMCHWT stalls and Jacobi overtakes ICTF for Ag at larger L.

## Deviations from the paper and from ADR 0006

- L = 3.2 µm instead of 10 µm (dense operator; 2N within 2.5–3.5·10⁴; L = 3.4 µm gave
  2N = 35 106 for Si).
- Si box without the ADR 0006 fine band (3δ + 3σ = 3.54 µm): with it the Si mesh at L = 3.2 µm
  has 46 688 triangles (box 470 % of the top face), 2N = 140 064, Z ≈ 314 GB — infeasible
  dense. Without it the Si walls coarsen to 400 nm from ~100–400 nm below the rim, where the field
  is still e^{−0.1…−0.35} of its surface value and the Si interior wavelength λ/|n₂| ≈ 116 nm is not
  resolved (|k₂|h ≈ 22). The Si system is therefore not the rigorous ADR 0006 problem; this may
  affect its iteration counts, not only the field accuracy in the box.
- Paraxial Gaussian beam (the rigorous angular-spectrum beam is Phase 5); waist L/3 as in
  ADR 0006.

## Notes

- Dense assembly took 1050–1540 s per system (24 threads, shared machine), much more than the
  N² extrapolation of the sphere records (2N = 15 360 in ~8–11 s, `mie_sphere_dense.md`). Not
  analysed here; the large Si triangles with |k₂|h ≈ 22 (k-aware degree selection) are the prime
  suspect — see WP-P2 (dense assembly performance) in docs/backlog.md.
- WP7c check: after merging main 601abb9 (WP7c fold-adaptive touching rule and own RHS degree)
  Ag ICTF was re-run: 622 iterations without and 1069 with Jacobi, as above; the residual
  histories differ by at most 3.2e-6 (none) and 3.4e-5 (Jacobi) relative. The table, measured
  before WP7c, therefore stands (assembly 1293 s in the re-run).

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_formulation_convergence
$B --materials si,ag --L 3.2e-6 --mesh-only                      # setup and unknown counts only
$B --materials si --L 3.2e-6 --seed 1 --max-iter 2000 --tol 1e-3 \
   --out <dir_si> --csv benchmarks/results/formulation_convergence_histories.csv
$B --materials ag --formulations pmchwt --L 3.2e-6 --seed 1 --out <dir_ag_pmchwt> --csv ...
# likewise --formulations ictf and mctf for Ag (run separately here, one process each)
```

Each process prints a `SETUP` line, one `RESULT` line per solve and `ISSUE15` lines (when the
formulations of a material run in one process), and writes the residual histories, the mesh and
attributes as a `.npy` directory (`--out`; not committed). About 6 h of wall time in total on
the shared machine (assembly dominates).
