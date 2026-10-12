---
name: si-large-area-findings
description: WP22d Si sigma=100 nm large-area facts - proxy 10.5 um numbers, WP22e server results (2x slower, B overestimated the time growth by ≈ 35 % (measured growth ≈ 26 % below B), R_coarse), sigma 100 vs 50 costs, octree leaf quantisation, extrapolation rule, server kit figures, review fixes (256 GB reasoning, timing matvecs, Ag series, tiny Ag test cost)
metadata:
  type: project
---

WP22d (2026-10-11): Phase 4 "Si 30x30 um" split into a 128 GB proxy plus a 1 TB server run kit
(`scripts/server_run_kit/`, record `benchmarks/results/si_large_area.md`).

- Proxy Si L = 10.5 um, sigma 100 nm, uniform 50 nm box, 2N = 1 098 720: 90 it, solve 1 464 s,
  peak 89.5 GB; L = 11 um model 120.7 GB (> 110 GB limit). Uniform box beats graded q = 0.8 here.
- sigma 100 vs 50 nm at equal L: iterations +23-33 %, solve +48-49 %, r_max 64-65 -> 74-81 nm.
  Si L = 1 um sigma 100 needs 116 iterations (smoke size; small patches iterate more).
- Octree root = patch edge L, so leaf = L/2^k: L = 30 um gets 234 nm leaves (model 823 GB) while
  L = 32 um with q = 0.8 gets 125 nm (546 GB). Root padding proposed, not applied.
- Graded/uniform iteration ratio at equal L (WP22b3 sigma 50): 1.39-1.46.
- Extrapolation rule used in the record: component model (estimator memory x measured s/GB) for
  memory and setup; union with the 2N power law for matvec/solve. Power law alone underestimates
  when the leaf grows (falls below 0.72 x model, the floor of all measured rows).
- (peak - Krylov)/model: 0.72-0.86 over WP22b3 + WP22d rows.

Review fixes (2026-10-11, `wp/22d-review-fixes`):
- 256 GB reasoning: docs/01 "2.2 M" = open top face (2N 2.16 M); ADR 0006 expected the box at a
  few % (3.4 % measured, 400 nm cells); +85-175 % now from lambda1/5 + Si fine band, 5.65 um
  depth, uniform-box recommendation; model ~138 kB/unknown at 234 nm leaves.
- run() times 2 (matvecs + 1) full-matvec equivalents before GMRES (12 at default 5): 23-31 min at
  L = 30 um, so large server runs use --matvecs 1. Post (--far-field 1) ~ 324 s x 2N / 1.1e6.
- Ag DoD closure: a server Ag series (L = 4/7/11/17.5 um) with its own gamma, not one point
  (Ag solve bandwidth-bound; Si server factor does not transfer).
- Tiny Ag MLFMM tests: near-field assembly dominates (~4 ms per unknown single-threaded); the
  default 2 um Ag box walls are most unknowns. Case::box_depth (test-only) = 0.5 um made the
  power-summary smoke test ~0.6 s (was 28 s).
- Slurm cgroup v2: limit on the job ancestor, task cgroup says "max": walk to the root, min.
- Never edit a sh script while a run of it is in progress (dash reads the file incrementally;
  the running smoke run died with a syntax error at its tail).

Server results (WP22e part 1, runs of 2026-10-12; 2 x 8-core EPYC 9015, 32 threads incl. SMT,
OMP_PLACES=threads / spread, interleave): Si L = 30 um solved, 95 it, peak 697 GB; L = 20 um 95 it, 250.4 GB.
- (peak - Krylov)/model 0.79 / 0.863 / 0.84 (L = 2/20/30): run 2 slightly ABOVE the 0.86 band.
- Server ~2x SLOWER than the desktop for Si (far matvec/GB 2.5x at 6 levels, post/unknown
  1.86-2.55x: desktop post/unknown is NOT constant, 2.16e-4 s at L = 4 um vs 2.92-2.95e-4 at
  6/10.5 um; server constant 5.48-5.50e-4; setup/GB 2-5x); only the sparse near matvec is 1.4x
  faster/GB (bandwidth-bound is a hypothesis). Cause untested. SMT alone does NOT distinguish the
  machines: the desktop also ran 2 threads/core (24 on 12, unpinned). Differences: pinning,
  2 sockets + interleave, clock, GCC 14 vs 16. Neither build uses -march (AVX-512 unused).
  Suggested check: run 1 with --threads 16 and with --numa none.
- Method B overestimated time growth 20 -> 30 um by ~35 %: cost per GB FALLS with larger leaves
  (far matvec 0.54 -> 0.40 s/GB, near setup 25 -> 15 s/GB). Memory B held.
- R vs R_coarse 2.1 % at L = 30 um: far field structure width lambda/(pi w0) = 1.2 deg < 2 deg
  coarse grid; diffuse speckle carries R (coherent factor exp(-(4 pi sigma/lambda)^2) ~ 2e-3).
  1 - R - A with R_coarse goes negative (-7e-3): use the power balance to pick the grid.
- solve_s includes one extra apply after the last iteration (true residual).
- si_large_area_extrapolation.py ignores machine=server rows (filters series=wp22d / sigma 50).

**Why:** server results will come back later and must be compared against these figures.
**How to apply:** when recording server rows, check against the stated ranges; server rows carry
`machine=server` (own series in mlfmm_scaling_fit.py). See [[mlfmm-scaling-findings]].
