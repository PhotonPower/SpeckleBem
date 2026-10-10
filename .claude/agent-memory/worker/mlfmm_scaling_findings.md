---
name: mlfmm-scaling-findings
description: WP22b1 measured MLFMM scaling on Si/Ag rough boxes (100 nm box cells) - leaf size set by box r_max, Si pattern memory limit, Ag GMRES MGS share, timing noise, peak model
metadata:
  type: project
---

Measured 2026-10-10 (win-release, 24 cores, shared 128 GB machine), sigma 50 nm, Lc 500 nm,
h 50 nm, d0 = 3, full GMRES tol 1e-3, record benchmarks/results/mlfmm_scaling.md:
- Leaf rule uses the GLOBAL r_max: 100 nm box cells give r_max 116-141 nm -> leaf floor
  195-235 nm, root (edge ~ max(L, depth)) halving rounds up to 219-406 nm. Leaf size jumps with
  L (Si L <= 6 um: ~360 nm, L = 7-8 um: 219-250 nm) -> non-monotone memory per unknown.
- Automatic box rule (200-400 nm cells): r_max 230-490 nm, leaves lambda/0.7 ... lambda/0.35,
  near field ~ dense (Si L = 4 um: 307 GB est.); Ag auto also 2.5x more iterations.
- Si: iterations flat (90-154), far matvec 85-95 % (R2 leaf order 45-60), leaf patterns 60-70 %
  of peak; 2N = 5.2e5 -> 79 GB peak; 1e6 needs ~240-310 GB. gamma_solve 0.73.
- Ag: 1000-1700 iterations (ICTF+Jacobi), serial MGS in solver::gmres = 42-66 % of solve
  (~9 GB/s effective at k = 1700, 2N = 2.9e5); 2N = 2.9e5 solve 55 min; exact part <= 0.8 GB with
  big leaves. GMRES(100) at 43k: 2.8x iterations, same time. gamma_solve 1.01-1.6 by range.
- Peak model 2 near + 1.1 leaf patterns + 2 exact + 1 GB is within 5 %; patterns =
  N 2(L+1)^2 32 B per expanded region (mlfmm::leaf_sampling; do not skip lossy-but-expanding Si).
- Timing noise from other agents: single medians up to 2x off; time the full/near/far applies
  interleaved; Si L = 1 solve 111 s vs 180 s in two runs.

**Why:** Phase 4 DoD scaling and sizing of later large rough-surface runs (Phase 5).
**How to apply:** budget memory with the model; expect Si limit ~5e5 unknowns until a local leaf
rule exists; Ag solves beyond ~3e5 need > 2 h with the current GMRES. See
[[mlfmm-operator-findings]], [[ag-sphere-mlfmm-findings]], [[gmres-observations]].
