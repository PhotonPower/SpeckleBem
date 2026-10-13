# Formulation convergence study, dense operator (WP15, Phase 3 DoD)

Reproduction attempt of Fu et al. 2023, Sec. 4 / Fig. 2(a,b): GMRES convergence of PMCHWT,
ICTF and MCTF without preconditioner and with the (left) Jacobi preconditioner on Gaussian
rough surfaces of Ag and Si at λ = 500 nm. Executable `benchmarks/formulation_convergence.cpp`
(`specklebem_formulation_convergence`, built with `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`); residual
histories (every 10th iteration plus the last) in
[`formulation_convergence_histories.csv`](formulation_convergence_histories.csv).

The MLFMM size sweep L = 3.2 … 10 µm (WP22b2, Phase 4) that revisits the WP15 deviations is the
section [WP22b2: MLFMM size sweep](#wp22b2-mlfmm-size-sweep) below; it updates the verdicts of
the section "Fu et al. 2023, Fig. 2 statements".

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

## WP22b2: MLFMM size sweep

The WP15 deviations (statements 1, 3, 5) were measured at one size, L = 3.2 µm, with the dense
operator, the paraxial beam w₀ = L/3 and a box graded to 400 nm cells. WP22b2 repeats the four
solves per material with the MLFMM at L = 3.2, 5, 7 and 10 µm (10 µm is the paper's size) on the
uniform box, and at 3.2 µm compares it with the graded λ₁/5 = 100 nm box (same MLFMM, beam and
GMRES; Ag: all three unpreconditioned formulations and Jacobi). The size sweep and the box
comparison change one thing at a time against each other, but neither reproduces the WP15
setup: dense vs MLFMM, beam and the 400 nm coarse cells are not separated.

Executable `specklebem_mlfmm_scaling` (`benchmarks/mlfmm_scaling.cpp`, options `--formulation`,
`--jacobi`, `--history-file`); residual histories (every 10th iteration plus the last, 4 digits,
one `HISTORY` line per solve) in
[`formulation_convergence_mlfmm_histories.txt`](formulation_convergence_mlfmm_histories.txt), the
same at every iteration with 17 digits in
[`formulation_convergence_mlfmm_histories_full.txt`](formulation_convergence_mlfmm_histories_full.txt),
the `ROW` lines (times, memory, true residual) in
[`formulation_convergence_mlfmm_rows.txt`](formulation_convergence_mlfmm_rows.txt), the per-run
logs in [`wp22b2_logs/`](wp22b2_logs/); tables below from
`benchmarks/formulation_sweep_summary.py`.

### Setup

| Parameter | Value |
|---|---|
| Wavelength, materials | λ = 500 nm, vacuum R1; Si ε_r = 18.478 − 0.606j, Ag ε_r = −9.794 − 0.313j (as WP15) |
| Surface | Gaussian, σ = 50 nm, Lc = 500 nm, h = 50 nm, seed 1; **L = 3.2, 5, 7, 10 µm** |
| Closing box (ADR 0006 with amendments, ADR 0008 decision 2026-10-11) | **uniform**: `box_mesh_size` = h = 50 nm on walls and bottom plate (the library default since ADR 0006 amendment 2026-10-12b / WP-B3; the study support `mlfmm_scaling_support.hpp` pins `BoxGradingMode::Graded` with an explicit coarse spacing, default 100 nm, so `--box-mesh-size 50e-9` stays required to get it); depth `default_box_depth` (Si 5.65 µm, fine band 3δ + 3σ, Ag 2 µm). At L = 3.2 µm also the **graded** λ₁/5 = 100 nm box (Si with fine band) |
| 2N (uniform box) | Si 222 720 / 391 200 / 614 880 / 1 022 400; Ag 110 592 / 216 000 / 369 600 / 672 000 |
| 2N (graded box, 3.2 µm) | Si 160 128, Ag 48 000 |
| Excitation | rigorous `AngularSpectrumBeam`, normal incidence (+z), w₀ = L/4 (Si wider than the WP-B3 default L/6: `--waist-factor 4 --allow-wide-beam`; outside the provisional Si limit L/6 of ADR 0006 amendment 2026-10-12b; this is a convergence study, not a quantitative R), focus at the origin, E along x |
| Operator | MLFMM, d₀ = 3, automatic leaf rule (q = 1) and exact-part budget |
| Solver | full GMRES (CGS2, no restart), x₀ = 0, tolerance 1e-3 on the monitored residual, `--max-iter` 4000 or a cap (below); Jacobi = left diagonal preconditioner of `Simulation` (monitored residual = preconditioned one) |
| Solves per (material, L) | PMCHWT, ICTF, MCTF without preconditioner, PMCHWT + Jacobi (the three Jacobi variants are identical, issue #15; all three run at L = 3.2 µm as a check). Review round: Ag 10 µm ICTF + Jacobi (true-residual comparison) and Ag 3.2 µm graded MCTF and PMCHWT |

One case per process, run one after the other (2026-10-12/13, win-release, GCC 16.1, 24 threads,
Ryzen 9 5900X, 128 GB). Before each case the driver waited until the 1-min load average was below
5 at two checks 30 s apart (`load0` = second value, 0.99–4.8); four cases started after the 10 min
limit at higher load (Si 3.2 µm MCTF 11.6 and MCTF + Jacobi 146, Ag 3.2 µm PMCHWT + Jacobi 184,
Ag 5 µm ICTF 9.6): their times are indicative only. Iteration counts and residuals do not depend
on the load. Peak memory ≤ 80.2 GB (Si 10 µm PMCHWT). The three review-round runs (2026-10-13,
merged main 6dc692e, same build settings) ran one after the other on the otherwise idle machine
without the load wait.

**Capped runs** (one tool command may run at most 2 h): Ag PMCHWT at L = 7 µm (3 300 iterations)
and 10 µm (2 200), chosen from the cost model t(k) ≈ a k + c k² (c ≈ 1 ns × 2N, CGS2) to fit the
2 h limit. Not reached: the Ag 10 µm MCTF cap (2 300), the Si 10 µm caps (PMCHWT and Jacobi 450,
ICTF and MCTF 1 000) and the Si 7 µm PMCHWT and Jacobi caps (700); all other runs used
`--max-iter 4000`. All runs except the two capped Ag PMCHWT runs converged to 1e-3.

### Iterations per material, formulation and L

Iterations (solve time in s); uniform box. Capped runs: ≥ cap, with the residual reached.

| Si | L = 3.2 µm | 5 µm | 7 µm | 10 µm |
|---|---:|---:|---:|---:|
| 2N | 222 720 | 391 200 | 614 880 | 1 022 400 |
| PMCHWT | 369 (928) | 369 (1 549) | 374 (2 926) | 405 (5 336) |
| ICTF | 75 (165) | 70 (291) | **65** (445) | **68** (888) |
| MCTF | **74** (157) | 70 (291) | 67 (461) | 70 (893) |
| Jacobi (PMCHWT + Jacobi) | 333 (708) | 333 (1 398) | 256 (1 764) | 280 (3 574) |
| setup [s] / peak [GB] (ICTF) | 53 / 17.9 | 76 / 30.9 | 135 / 59.5 | 189 / 74.8 |

| Ag | L = 3.2 µm | 5 µm | 7 µm | 10 µm |
|---|---:|---:|---:|---:|
| 2N | 110 592 | 216 000 | 369 600 | 672 000 |
| PMCHWT | 2 127 (829) | 3 423 (3 642) | ≥ 3 300, 1.51e-3 (5 928) | ≥ 2 200, 2.29e-3 (5 143) |
| ICTF | 347 (74) | 419 (173) | 490 (374) | 643 (884) |
| MCTF | 505 (115) | 737 (345) | 1 161 (1 113) | 2 039 (4 533) |
| Jacobi (PMCHWT + Jacobi) | **346** (82) | 423 (176) | **470** (355) | **597** (807) |
| setup [s] / peak [GB] (ICTF) | 13 / 4.5 | 25 / 7.6 | 46 / 17.0 | 73 / 25.7 |

Ratios to ICTF (iterations):

| Ratio | Si 3.2 | Si 5 | Si 7 | Si 10 | Ag 3.2 | Ag 5 | Ag 7 | Ag 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PMCHWT / ICTF | 4.92 | 5.27 | 5.75 | 5.96 | 6.13 | 8.17 | > 6.73 | > 3.42 |
| MCTF / ICTF | 0.99 | 1.00 | 1.03 | 1.03 | 1.46 | 1.76 | 2.37 | **3.17** |
| Jacobi / ICTF | 4.44 | 4.76 | 3.94 | 4.12 | 1.00 | 1.01 | 0.96 | **0.93** |

The Jacobi ratios compare iteration counts at the same monitored tolerance, which for Jacobi is
the preconditioned residual (see statement 3 for the comparison at equal true residual).

Full table (all runs; monitored residual at stop; true residual |b − Zx|/|b| of the
formulation's own unpreconditioned system, from the `ROW` lines (for Jacobi runs it differs from
the monitored, preconditioned residual); s/it, setup, peak). Review-round runs: Ag 3.2 µm graded
MCTF and PMCHWT, Ag 10 µm ICTF + Jacobi:

| Material | Box | L [um] | 2N | Formulation | Iterations | Converged | Residual | True res. | Solve [s] | s/it | Setup [s] | Peak [GB] |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| ag | 100 nm | 3.2 | 48000 | PMCHWT | 3461 | yes | 0.000999 | 0.000999 | 1167 | 0.337 | 9 | 6.6 |
| ag | 100 nm | 3.2 | 48000 | ICTF | 571 | yes | 0.000998 | 0.000998 | 110 | 0.192 | 9 | 4.3 |
| ag | 100 nm | 3.2 | 48000 | MCTF | 783 | yes | 0.000999 | 0.000999 | 158 | 0.202 | 9 | 4.5 |
| ag | 100 nm | 3.2 | 48000 | PMCHWT + Jacobi | 960 | yes | 0.000999 | 0.00844 | 209 | 0.218 | 11 | 4.6 |
| ag | 50 nm | 3.2 | 110592 | PMCHWT | 2127 | yes | 0.000999 | 0.000999 | 829 | 0.390 | 14 | 7.7 |
| ag | 50 nm | 3.2 | 110592 | ICTF | 347 | yes | 0.000999 | 0.000999 | 74 | 0.213 | 13 | 4.5 |
| ag | 50 nm | 3.2 | 110592 | MCTF | 505 | yes | 0.000996 | 0.000997 | 115 | 0.229 | 13 | 4.8 |
| ag | 50 nm | 3.2 | 110592 | PMCHWT + Jacobi | 346 | yes | 0.000996 | 0.00638 | 82 | 0.236 | 35 | 4.5 |
| ag | 50 nm | 3.2 | 110592 | ICTF + Jacobi | 346 | yes | 0.000996 | 0.00107 | 74 | 0.214 | 17 | 4.5 |
| ag | 50 nm | 3.2 | 110592 | MCTF + Jacobi | 346 | yes | 0.000996 | 0.00107 | 74 | 0.213 | 17 | 4.5 |
| ag | 50 nm | 5 | 216000 | PMCHWT | 3423 | yes | 0.000999 | 0.001 | 3642 | 1.064 | 24 | 18.1 |
| ag | 50 nm | 5 | 216000 | ICTF | 419 | yes | 0.000999 | 0.000999 | 173 | 0.414 | 25 | 7.6 |
| ag | 50 nm | 5 | 216000 | MCTF | 737 | yes | 0.001 | 0.001 | 345 | 0.468 | 24 | 8.7 |
| ag | 50 nm | 5 | 216000 | PMCHWT + Jacobi | 423 | yes | 0.000994 | 0.00539 | 176 | 0.416 | 32 | 7.6 |
| ag | 50 nm | 7 | 369600 | PMCHWT | 3300 | **no (cap)** | 0.00151 | 0.00151 | 5928 | 1.796 | 46 | 33.8 |
| ag | 50 nm | 7 | 369600 | ICTF | 490 | yes | 0.000999 | 0.000999 | 374 | 0.764 | 46 | 17.0 |
| ag | 50 nm | 7 | 369600 | MCTF | 1161 | yes | 0.001 | 0.001 | 1113 | 0.959 | 46 | 21.0 |
| ag | 50 nm | 7 | 369600 | PMCHWT + Jacobi | 470 | yes | 0.001 | 0.0062 | 355 | 0.756 | 60 | 16.9 |
| ag | 50 nm | 10 | 672000 | PMCHWT | 2200 | **no (cap)** | 0.00229 | 0.00228 | 5143 | 2.338 | 73 | 42.6 |
| ag | 50 nm | 10 | 672000 | ICTF | 643 | yes | 0.000999 | 0.000999 | 884 | 1.375 | 73 | 25.7 |
| ag | 50 nm | 10 | 672000 | MCTF | 2039 | yes | 0.000999 | 0.000999 | 4533 | 2.223 | 73 | 40.8 |
| ag | 50 nm | 10 | 672000 | PMCHWT + Jacobi | 597 | yes | 0.000999 | 0.00544 | 807 | 1.351 | 97 | 25.3 |
| ag | 50 nm | 10 | 672000 | ICTF + Jacobi | 597 | yes | 0.000999 | 0.00108 | 808 | 1.353 | 102 | 25.3 |
| si | 100 nm | 3.2 | 160128 | ICTF | 105 | yes | 0.000993 | 0.000993 | 323 | 3.079 | 106 | 35.5 |
| si | 100 nm | 3.2 | 160128 | PMCHWT + Jacobi | 467 | yes | 0.000999 | 0.00744 | 1579 | 3.381 | 91 | 36.4 |
| si | 50 nm | 3.2 | 222720 | PMCHWT | 369 | yes | 0.000997 | 0.000997 | 928 | 2.515 | 58 | 18.9 |
| si | 50 nm | 3.2 | 222720 | ICTF | 75 | yes | 0.000964 | 0.000964 | 165 | 2.200 | 53 | 17.9 |
| si | 50 nm | 3.2 | 222720 | MCTF | 74 | yes | 0.000997 | 0.000997 | 157 | 2.120 | 50 | 17.8 |
| si | 50 nm | 3.2 | 222720 | PMCHWT + Jacobi | 333 | yes | 0.000996 | 0.00956 | 708 | 2.126 | 56 | 18.8 |
| si | 50 nm | 3.2 | 222720 | ICTF + Jacobi | 333 | yes | 0.000996 | 0.00117 | 713 | 2.142 | 56 | 18.8 |
| si | 50 nm | 3.2 | 222720 | MCTF + Jacobi | 333 | yes | 0.000996 | 0.00118 | 711 | 2.135 | 127 | 18.8 |
| si | 50 nm | 5 | 391200 | PMCHWT | 369 | yes | 0.000999 | 0.000999 | 1549 | 4.198 | 76 | 32.8 |
| si | 50 nm | 5 | 391200 | ICTF | 70 | yes | 0.000974 | 0.000974 | 291 | 4.164 | 76 | 30.9 |
| si | 50 nm | 5 | 391200 | MCTF | 70 | yes | 0.000967 | 0.000967 | 291 | 4.161 | 76 | 30.9 |
| si | 50 nm | 5 | 391200 | PMCHWT + Jacobi | 333 | yes | 0.001 | 0.013 | 1398 | 4.199 | 89 | 32.5 |
| si | 50 nm | 7 | 614880 | PMCHWT | 374 | yes | 0.000992 | 0.000992 | 2926 | 7.823 | 136 | 62.5 |
| si | 50 nm | 7 | 614880 | ICTF | 65 | yes | 0.000979 | 0.000979 | 445 | 6.844 | 135 | 59.5 |
| si | 50 nm | 7 | 614880 | MCTF | 67 | yes | 0.000971 | 0.000971 | 461 | 6.883 | 135 | 59.5 |
| si | 50 nm | 7 | 614880 | PMCHWT + Jacobi | 256 | yes | 0.000998 | 0.0171 | 1764 | 6.892 | 164 | 61.4 |
| si | 50 nm | 10 | 1022400 | PMCHWT | 405 | yes | 0.000998 | 0.000998 | 5336 | 13.175 | 180 | 80.2 |
| si | 50 nm | 10 | 1022400 | ICTF | 68 | yes | 0.000956 | 0.000956 | 888 | 13.059 | 189 | 74.8 |
| si | 50 nm | 10 | 1022400 | MCTF | 70 | yes | 0.000958 | 0.000958 | 893 | 12.755 | 185 | 74.8 |
| si | 50 nm | 10 | 1022400 | PMCHWT + Jacobi | 280 | yes | 0.000997 | 0.017 | 3574 | 12.766 | 218 | 77.9 |

### Residual histories

Monitored residual at selected iterations (log-linear interpolation between the stored samples
every 10 iterations):

| Run | 10 | 50 | 100 | 200 | 500 | 1000 | 2000 | 3000 | 4000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ag 100 nm L = 3.2 um PMCHWT | 2.25e-01 | 4.48e-02 | 2.19e-02 | 8.53e-03 | 2.11e-03 | 1.78e-03 | 1.42e-03 | 1.09e-03 | - |
| ag 100 nm L = 3.2 um ICTF | 2.04e-01 | 4.07e-02 | 1.63e-02 | 6.45e-03 | 1.35e-03 | - | - | - | - |
| ag 100 nm L = 3.2 um MCTF | 1.68e-01 | 5.17e-02 | 2.59e-02 | 1.13e-02 | 2.49e-03 | - | - | - | - |
| ag 100 nm L = 3.2 um PMCHWT + Jacobi | 3.41e-01 | 1.24e-01 | 5.27e-02 | 2.21e-02 | 4.49e-03 | - | - | - | - |
| ag 50 nm L = 3.2 um PMCHWT | 1.23e-01 | 2.90e-02 | 1.47e-02 | 5.32e-03 | 2.38e-03 | 1.79e-03 | 1.10e-03 | - | - |
| ag 50 nm L = 3.2 um ICTF | 1.21e-01 | 2.00e-02 | 8.38e-03 | 2.80e-03 | - | - | - | - | - |
| ag 50 nm L = 3.2 um MCTF | 1.18e-01 | 2.77e-02 | 1.31e-02 | 5.34e-03 | 1.02e-03 | - | - | - | - |
| ag 50 nm L = 3.2 um PMCHWT + Jacobi | 1.58e-01 | 2.19e-02 | 8.77e-03 | 2.81e-03 | - | - | - | - | - |
| ag 50 nm L = 3.2 um ICTF + Jacobi | 1.58e-01 | 2.19e-02 | 8.77e-03 | 2.81e-03 | - | - | - | - | - |
| ag 50 nm L = 3.2 um MCTF + Jacobi | 1.58e-01 | 2.19e-02 | 8.77e-03 | 2.81e-03 | - | - | - | - | - |
| ag 50 nm L = 5 um PMCHWT | 1.21e-01 | 3.95e-02 | 2.21e-02 | 8.46e-03 | 3.02e-03 | 2.90e-03 | 1.66e-03 | 1.04e-03 | - |
| ag 50 nm L = 5 um ICTF | 1.23e-01 | 2.18e-02 | 9.91e-03 | 3.63e-03 | - | - | - | - | - |
| ag 50 nm L = 5 um MCTF | 1.20e-01 | 3.43e-02 | 1.93e-02 | 8.59e-03 | 2.11e-03 | - | - | - | - |
| ag 50 nm L = 5 um PMCHWT + Jacobi | 1.41e-01 | 2.24e-02 | 1.02e-02 | 3.62e-03 | - | - | - | - | - |
| ag 50 nm L = 7 um PMCHWT | 1.25e-01 | 4.10e-02 | 2.27e-02 | 9.76e-03 | 3.06e-03 | 2.82e-03 | 2.01e-03 | 1.52e-03 | - |
| ag 50 nm L = 7 um ICTF | 1.14e-01 | 1.89e-02 | 8.88e-03 | 4.09e-03 | - | - | - | - | - |
| ag 50 nm L = 7 um MCTF | 1.11e-01 | 3.49e-02 | 2.13e-02 | 1.16e-02 | 4.02e-03 | 1.33e-03 | - | - | - |
| ag 50 nm L = 7 um PMCHWT + Jacobi | 1.24e-01 | 1.91e-02 | 8.69e-03 | 3.73e-03 | - | - | - | - | - |
| ag 50 nm L = 10 um PMCHWT | 1.45e-01 | 5.56e-02 | 3.44e-02 | 1.61e-02 | 4.39e-03 | 3.50e-03 | 2.43e-03 | - | - |
| ag 50 nm L = 10 um ICTF | 1.29e-01 | 2.53e-02 | 1.29e-02 | 5.92e-03 | 1.59e-03 | - | - | - | - |
| ag 50 nm L = 10 um MCTF | 1.38e-01 | 4.77e-02 | 3.06e-02 | 1.93e-02 | 8.30e-03 | 3.48e-03 | 1.05e-03 | - | - |
| ag 50 nm L = 10 um PMCHWT + Jacobi | 1.44e-01 | 2.54e-02 | 1.27e-02 | 5.42e-03 | 1.39e-03 | - | - | - | - |
| ag 50 nm L = 10 um ICTF + Jacobi | 1.44e-01 | 2.54e-02 | 1.27e-02 | 5.42e-03 | 1.39e-03 | - | - | - | - |
| si 100 nm L = 3.2 um ICTF | 1.08e-01 | 6.55e-03 | 1.17e-03 | - | - | - | - | - | - |
| si 100 nm L = 3.2 um PMCHWT + Jacobi | 1.20e-01 | 3.73e-02 | 1.55e-02 | 5.88e-03 | - | - | - | - | - |
| si 50 nm L = 3.2 um PMCHWT | 1.11e-01 | 6.62e-03 | 2.09e-03 | 1.93e-03 | - | - | - | - | - |
| si 50 nm L = 3.2 um ICTF | 6.46e-02 | 3.40e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 3.2 um MCTF | 1.04e-01 | 3.45e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 3.2 um PMCHWT + Jacobi | 9.90e-02 | 1.65e-02 | 7.01e-03 | 2.58e-03 | - | - | - | - | - |
| si 50 nm L = 3.2 um ICTF + Jacobi | 9.90e-02 | 1.65e-02 | 7.01e-03 | 2.58e-03 | - | - | - | - | - |
| si 50 nm L = 3.2 um MCTF + Jacobi | 9.90e-02 | 1.65e-02 | 7.01e-03 | 2.58e-03 | - | - | - | - | - |
| si 50 nm L = 5 um PMCHWT | 9.90e-02 | 5.37e-03 | 1.98e-03 | 1.86e-03 | - | - | - | - | - |
| si 50 nm L = 5 um ICTF | 5.62e-02 | 2.72e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 5 um MCTF | 1.02e-01 | 2.75e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 5 um PMCHWT + Jacobi | 8.77e-02 | 1.23e-02 | 5.73e-03 | 2.37e-03 | - | - | - | - | - |
| si 50 nm L = 7 um PMCHWT | 9.42e-02 | 4.86e-03 | 1.93e-03 | 1.82e-03 | - | - | - | - | - |
| si 50 nm L = 7 um ICTF | 4.38e-02 | 2.02e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 7 um MCTF | 8.84e-02 | 2.35e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 7 um PMCHWT + Jacobi | 6.80e-02 | 8.96e-03 | 3.94e-03 | 1.53e-03 | - | - | - | - | - |
| si 50 nm L = 10 um PMCHWT | 1.04e-01 | 5.85e-03 | 2.01e-03 | 1.84e-03 | - | - | - | - | - |
| si 50 nm L = 10 um ICTF | 4.72e-02 | 2.23e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 10 um MCTF | 9.54e-02 | 2.73e-03 | - | - | - | - | - | - | - |
| si 50 nm L = 10 um PMCHWT + Jacobi | 7.66e-02 | 9.75e-03 | 4.36e-03 | 1.68e-03 | - | - | - | - | - |

- **Ag PMCHWT** converges in a staircase: plateaus at an almost constant residual, separated by
  drops. Every 200 iterations at L = 7 µm: 9.8e-3, 3.6e-3, 2.89e-3,
  2.83e-3, 2.82e-3, 2.79e-3, 2.59e-3, 2.17e-3, 2.03e-3, 2.02e-3, 2.01e-3, 1.97e-3, 1.74e-3,
  1.55e-3, 1.52e-3, 1.52e-3 (3 000; ×0.975 over the last 500 iterations before the 3 300 cap);
  at L = 10 µm: 1.6e-2, 5.8e-3, 3.9e-3, 3.55e-3, 3.50e-3, 3.48e-3, 3.45e-3, 3.30e-3 (1 400),
  2.87e-3, 2.43e-3, 2.29e-3 (2 200; ×0.73 over the last 500). Plateau lengths from the
  full-precision histories, a plateau being a stretch in which the residual falls by less than
  3 % per 100 iterations: L = 3.2 µm 480, 430, 395 iterations (from iteration 390, 1 095,
  1 676); 5 µm 585, 608, 561, 507 (from 472, …); 7 µm 785, 813, ≥ 582 (cut by the cap; from
  553, …); 10 µm 933 (from 676), then a second one from 2 037 cut by the cap. The first plateau
  sets in at 2.5e-3 (3.2 µm), 3.1e-3 (5 µm), 3.0e-3 (7 µm) and 3.7e-3 (10 µm) after 390–680
  iterations; the first plateaus lengthen from ≈ 480 (3.2 µm) to ≈ 930 iterations (10 µm). Their
  cause is not established. Open hypotheses: the WP-A2 box cavity resonances (near-null modes
  of the truncated Ag box, ADR 0006 amendment 2026-10-12), and the MLFMM approximation error
  (d₀ = 3; see "Box effect": the dense WP15 histories have no plateau).
- **Si PMCHWT** stalls at 1.8–2.1e-3 from iteration ≈ 100 to ≈ 350 at every L (100: 2.1e-3,
  2.0e-3, 1.9e-3, 2.0e-3; 200: 1.93e-3, 1.86e-3, 1.82e-3, 1.84e-3 at L = 3.2, 5, 7, 10 µm)
  and then converges in 369–405 iterations: the stall length grows little with L.
- ICTF, MCTF and the Jacobi runs decrease steadily for both materials (no plateau).

### Jacobi identity check (issue #15) at L = 3.2 and 10 µm

ICTF + Jacobi and MCTF + Jacobi against PMCHWT + Jacobi, full-precision histories
(`--history-full-file`, every iteration, 17 digits; `max_k |r_k − r_k^PMCHWT| / r_k^PMCHWT`):

| Material | ICTF + Jacobi | MCTF + Jacobi | Iterations |
|---|---:|---:|---|
| Si | 1.1e-8 | 8.0e-9 | 333 / 333 / 333 |
| Ag | 3.4e-12 | 5.4e-12 | 346 / 346 / 346 |
| Ag, L = 10 µm (review run W1) | 4.7e-9 | — | 597 / 597 / — |

Identical, as predicted. The Si difference (1e-8, dense WP15: 5e-13) is larger than the dense
one; **hypothesis, not tested**: rounding amplified by the Krylov process, since each
formulation assembles its own scaled near-field and far-field blocks (different summation
order, presumably a 1e-16 effect on the operator) and the Si PMCHWT + Jacobi history contains
a long slow phase (2.6e-3 at 200 → 1e-3 at 333) that would magnify such perturbations.
The true residuals in the formulations' own systems differ by the row weighting (Si 9.6e-3 /
1.17e-3 / 1.18e-3, Ag 6.4e-3 / 1.07e-3 / 1.07e-3 for PMCHWT / ICTF / MCTF at 3.2 µm; Ag 10 µm
5.4e-3 / 1.08e-3 for PMCHWT / ICTF).

### Box effect at L = 3.2 µm

| Material | Box | 2N | ICTF | PMCHWT + Jacobi | Jacobi / ICTF | ICTF setup + solve [s] | Jacobi setup + solve [s] |
|---|---|---:|---:|---:|---:|---:|---:|
| Si | graded 100 nm (fine band) | 160 128 | 105 | 467 | 4.45 | 106 + 323 | 91 + 1 579 |
| Si | uniform 50 nm | 222 720 | 75 | 333 | 4.44 | 53 + 165 | 56 + 708 |
| Ag | graded 100 nm | 48 000 | 571 | 960 | **1.68** | 9 + 110 | 11 + 209 |
| Ag | uniform 50 nm | 110 592 | 347 | 346 | **1.00** | 13 + 74 | 35 + 82 |
| Ag | WP15 dense, graded 400 nm, paraxial w₀ = L/3 | 29 280 | 622 | 1 069 | 1.72 | — | — |

Ag without preconditioner (review round, C2: graded MCTF and PMCHWT added; MLFMM d₀ = 3,
rigorous beam w₀ = L/4 in both MLFMM rows):

| Ag, L = 3.2 µm | 2N | ICTF | MCTF | PMCHWT | MCTF / ICTF | PMCHWT / ICTF |
|---|---:|---:|---:|---:|---:|---:|
| graded 100 nm box, MLFMM | 48 000 | 571 | 783 | 3 461 | **1.37** | **6.06** |
| uniform 50 nm box, MLFMM | 110 592 | 347 | 505 | 2 127 | 1.46 | 6.13 |
| WP15: graded 400 nm box, dense, paraxial w₀ = L/3 | 29 280 | 622 | 651 | 886 | 1.05 | 1.42 |

- The uniform box needs fewer iterations for both materials although it has 1.4× (Si) and 2.3×
  (Ag) the unknowns: Si −29 % (ICTF and Jacobi alike), Ag −39 % (ICTF), −36 % (MCTF), −39 %
  (PMCHWT) and −64 % (Jacobi). It is also cheaper overall: smaller leaves and near field (Si
  peak 17.9 against 35.5 GB) and shorter solves.
- For Ag, the graded box penalises Jacobi specifically (1.68× ICTF, close to the WP15 ratio 1.72
  on the 400 nm box); on the uniform box Jacobi equals ICTF. **Attribution from one paired
  comparison** (L = 3.2 µm, seed 1): the WP15 statement-3 deviation is attributed mainly to the
  graded box rather than to the patch size. A plausible mechanism, not tested: the diagonal
  entries of the 100–400 nm box cells differ strongly from those of the 50 nm top face, so the
  Jacobi row weights unbalance the system instead of equilibrating it. WP15 also differs from the
  MLFMM runs in the beam (paraxial L/3), the operator (dense) and the coarse cells (400 nm); these
  are not separated.
- For the unpreconditioned Ag ratios (statements 1 and 5) the graded-vs-uniform choice at λ₁/5 is **not** the cause: on the
  graded λ₁/5 box MCTF/ICTF = 1.37 and PMCHWT/ICTF = 6.06, almost the uniform-box 1.46 and 6.13,
  far from WP15's 1.05 and 1.42. The graded-box PMCHWT history shows the same staircase
  (plateaus of 703, 623, 564, 510 iterations from iteration 611, by the 3 %-per-100-iterations
  criterion above), whereas no WP15 history has a plateau. What this separates: graded λ₁/5 vs
  uniform box at fixed operator, beam and GMRES. What it does not separate: the WP15 400 nm coarse
  cells (the λ₁/5 box has 100 nm cells), the dense vs MLFMM operator and the paraxial L/3 vs
  rigorous L/4 beam. **Hypothesis, not tested:** the MLFMM approximation (d₀ = 3, block errors of
  order 1e-3) perturbs the ill-conditioned PMCHWT spectrum enough to cause the plateaus, whose
  levels (1.1–3.7e-3 for Ag, ≈ 2e-3 for Si) are of that order; Si PMCHWT also stalls at ≈ 2e-3
  with the MLFMM (iterations 100–350) but not in the dense WP15 run. Counter-indication: the
  finest-level R1 block check of the graded 3.2 µm run (2.9e-4, run log) is ≈ 18× smaller than
  that of the uniform run (5.2e-3), yet its plateaus start at a similar level (1.9e-3 against
  2.5e-3). The WP-A2 box-cavity resonances remain the other open hypothesis. A d₀ = 5 run (or a
  dense run with the rigorous beam) at 3.2 µm would decide it.
- For Si, the box changes the level, not the ranking (Jacobi/ICTF 4.4 for both boxes).

### Fu et al. 2023, Fig. 2 statements versus L (updated verdicts)

1. **PMCHWT does not converge** —
   - Ag: **trend consistent** (PMCHWT impractical, cost grows with L; residual at iteration
     2 000: 1.10e-3 / 1.66e-3 / 2.01e-3 / 2.43e-3 for L = 3.2 / 5 / 7 / 10 µm; 6–8× the ICTF
     iterations where it converges: 2 127 at 3.2 µm, 3 423 at 5 µm). **Non-convergence not
     demonstrated**: both caps (3 300 at 7 µm, residual 1.5e-3; 2 200 at 10 µm, 2.3e-3) lie
     below the 5 µm count of 3 423, and the 10 µm residual was still decreasing (×0.73 over its
     last 500 iterations; ×0.975 at 7 µm). The staircase plateaus lengthen with L (≈ 400 →
     ≈ 930 iterations).
   - Si: **not reproduced at any L**. 369, 369, 374 and 405 iterations at L = 3.2, 5, 7, 10 µm, 4.9–6.0× ICTF,
     with a size-independent plateau at ≈ 2e-3 (iterations 100–350). PMCHWT is the slowest Si
     formulation, as in WP15.
2. **ICTF converges fastest for Si without preconditioner** — **reproduced at every L**, tied
   with MCTF (75/74, 70/70, 65/67, 68/70); 4–6× faster than PMCHWT and Jacobi. ICTF
   iterations do not grow with L (75, 70, 65, 68).
3. **Jacobi is best for Ag** — **marginal: reproduced at 10 µm by a small margin (−3 % at equal
   true residual, one seed); at 7 µm only on the preconditioned residual (−4 %; the
   equal-true-residual comparison was not measured there)**. On the uniform
   box Jacobi equals ICTF at 3.2 and 5 µm (346/347, 423/419) and wins at 7 µm (−4 %) and at the
   paper's 10 µm (597 against 643, −7 %); it beats MCTF (3.4× at 10 µm) and PMCHWT by far.
   **Stopping-criterion caveat:** left Jacobi stops on the *preconditioned* residual. In the
   ICTF system the ICTF + Jacobi solutions have true residuals 1.07e-3 (3.2 µm) and 1.08e-3
   (10 µm, review run W1: 597 iterations, identical to PMCHWT + Jacobi to 4.7e-9), against
   ≤ 1e-3 for plain ICTF. Plain ICTF (whose monitored residual equals its true residual to the
   6 recorded digits) reaches those true residuals after 336 (3.2 µm) and 617 (10 µm)
   iterations, so at equal true residual Jacobi needs +3 % (3.2 µm) and −3 % (10 µm)
   iterations: the 10 µm advantage shrinks from 7 % to 3 %. On the graded box (WP15 and here at
   3.2 µm) Jacobi is 1.7× slower than ICTF; the WP15 deviation is attributed mainly to the box
   grading (one paired comparison, see "Box effect"); the small Jacobi advantage itself appears
   only with growing L.
4. **MCTF similar to ICTF for Si** — **reproduced at every L** (ratio 0.99–1.03).
5. **MCTF much slower than ICTF for Ag** — **reproduced from L = 5 µm, growing with L**: MCTF/ICTF
   = 1.46, 1.76, 2.37, 3.17 at L = 3.2, 5, 7, 10 µm. The WP15 +5 % at 3.2 µm is **not** a size
   effect (the MLFMM gives 1.46 at the same L) and **not** a graded-vs-uniform box effect (graded
   λ₁/5 box: 1.37, review run); it comes from the remaining setup differences of WP15 (dense
   operator, paraxial L/3 beam, 400 nm coarse cells), which are not separated (see "Box effect").
6. **The three preconditioned formulations behave alike** — **reproduced** (identity to 5e-12
   Ag and 1e-8 Si at 3.2 µm, 4.7e-9 Ag ICTF at 10 µm, see above).

At the paper's size, five of six statements hold at 10 µm; statement 1 holds for neither material
in the strict sense (Si PMCHWT converges at every size tested; Ag PMCHWT is impractical, but its
non-convergence is not demonstrated). Further qualifications: the Jacobi advantage for Ag is
small (7 % at 10 µm on the preconditioned residual, 3 % at equal true residual; statement 3) and was seen only on the
uniform box; the Ag verdicts rest on a closing box that fails the docs/05 box checks for Ag (ADR
0006 amendment 2026-10-12b, decision 3; cause open, WP-V3); the Si runs use w₀ = L/4, wider than
the provisional Si limit L/6 (convergence study, not a quantitative R).

### Recommendation for `formulation::recommend` (proposed, not changed)

Current choice (`src/formulation/formulation.cpp`): ICTF for dielectrics, ICTF + left Jacobi for
Re ε_r < 0 (metals).

- **Si: ICTF — confirmed** (fastest at every L, MCTF equivalent; measured at w₀ = L/4, wider than
  the provisional Si limit L/6 of ADR 0006 amendment 2026-10-12b; whether the counts
  change at L/6 was not measured).
- **Ag: ICTF + Jacobi — not contradicted, marginal on the uniform box** (≤ ICTF at every L on
  the preconditioned residual, −7 % at 10 µm; at equal true residual +3 % at 3.2 µm and −3 % at
  10 µm, see statement 3), but **contradicted on the
  graded λ₁/5 box** (+68 % iterations at 3.2 µm, and WP15's +72 %). The uniform box is already
  the library default (ADR 0006 amendment 2026-10-12b, WP-B3), so the default Ag setup is the one
  on which Jacobi is not contradicted. What remains open is the `formulation::recommend` question for
  explicitly graded boxes: `recommend` sees only ε, so it cannot know the box; where a graded box
  is chosen for Ag, plain ICTF would be better (e.g. by passing the box grading to `recommend`
  or by documenting the override `diagonal_preconditioner = false`). A coordinator/ADR decision.
- Caveat for all Ag conclusions: Ag rough surfaces currently fail the docs/05 box checks (depth
  ×2, L ×1.2) on uniform and graded boxes alike; the cause (surface plasmons vs the WP-A2 box
  cavity resonances) is open (WP-V3), and no quantitative rough-Ag results are produced until
  then (ADR 0006 amendment 2026-10-12b, decision 3). The iteration counts above are properties of
  the systems solved, which include that box.
- Side result for the Phase 4 Ag scaling DoD (WP22b3, graded box: 1 394 iterations at L = 4 µm,
  2 272 at 11 µm): on the uniform box Ag needs 470–643 iterations at L = 7–10 µm, and
  2N = 672 000 at L = 10 µm solves in 807 s. The open Ag time-per-solve exponent should be
  re-measured on the uniform box.

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

WP22b2 (MLFMM size sweep; one case per process, ≈ 17 h in total on the quiet machine, plus
≈ 40 min for the three review runs):

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_mlfmm_scaling
H=benchmarks/results/formulation_convergence_mlfmm_histories.txt
F=benchmarks/results/formulation_convergence_mlfmm_histories_full.txt
$B --material ag --L 10e-6 --box-mesh-size 50e-9 --beam angular-spectrum --estimate-only
# per material, L (3.2e-6 5e-6 7e-6 10e-6) and (formulation, jacobi) in
# (pmchwt,0) (ictf,0) (mctf,0) (pmchwt,1); at L = 3.2 um also (ictf,1) (mctf,1).
# Ag runs use the default waist L/4; Si runs need --waist-factor 4 --allow-wide-beam (since
# WP-B3 the Si default is L/6, and the recorded Si runs used L/4).
$B --material ag --L 10e-6 --box-mesh-size 50e-9 --beam angular-spectrum --max-iter 4000    --formulation ictf --jacobi 0 --history --history-file $H --history-full-file $F > ag10_ictf.log
$B --material si --waist-factor 4 --allow-wide-beam --L 10e-6 --box-mesh-size 50e-9    --beam angular-spectrum --max-iter 1000 --formulation ictf --jacobi 0    --history --history-file $H --history-full-file $F > si10_ictf.log
# caps: Ag PMCHWT --max-iter 3300 (7 um) / 2200 (10 um), Ag 10 um MCTF 2300, Si 7 um PMCHWT and
# Jacobi 700, Si 10 um PMCHWT and Jacobi 450, Si 10 um ICTF and MCTF 1000; others 4000
# graded box at 3.2 um: --box-mesh-size 100e-9 with (ictf,0) and (pmchwt,1) for both materials
# (Si with --waist-factor 4 --allow-wide-beam), and for Ag also (mctf,0) and (pmchwt,0) (review C2)
# review W1: Ag 10 um uniform box with (ictf,1)
grep -h '^ROW' *.log > benchmarks/results/formulation_convergence_mlfmm_rows.txt
python benchmarks/formulation_sweep_summary.py $H    --rows benchmarks/results/formulation_convergence_mlfmm_rows.txt --full $F
```

The summary script prints the full table (with the true residual of each formulation's own
system from the `ROW` lines), the snapshots, the capped runs and (with `--full`) the Jacobi
identity check. It refuses duplicate records of one run (key: material, box, L, waist,
formulation, Jacobi). The raw per-run logs, the estimate logs and the batch files of the sweep
are in [`wp22b2_logs/`](wp22b2_logs/); the batch files predate WP-B3 and carry no waist option,
so their Si lines need `--waist-factor 4 --allow-wide-beam` with current code. The 40 `HISTORY`
lines of the original sweep lack the `waist=` and `max_iter=` keys (added in the review round;
all those runs are at waist 4, as their `ROW` lines' `waist=4` shows); the three review runs
carry them.
