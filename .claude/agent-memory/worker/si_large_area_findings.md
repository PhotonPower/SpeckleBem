---
name: si-large-area-findings
description: WP22d Si sigma=100 nm large-area facts - proxy 10.5 um numbers, sigma 100 vs 50 costs, octree leaf quantisation, extrapolation rule, server kit figures, review fixes (256 GB reasoning, timing matvecs, Ag series, tiny Ag test cost)
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

**Why:** server results will come back later and must be compared against these figures.
**How to apply:** when recording server rows, check against the stated ranges; server rows carry
`machine=server` (own series in mlfmm_scaling_fit.py). See [[mlfmm-scaling-findings]].
