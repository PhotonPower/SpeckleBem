---
name: mlfmm-scaling-findings
description: WP22b1/WP22b3 MLFMM scaling on Si/Ag rough boxes - leaf size set by box r_max, uniform box best Si route to 1e6, Ag solve cost model and 2 h cap, local leaf q=0.8 effect, loadavg caveat
metadata:
  type: project
---

Measured 2026-10-10 (win-release, 24 cores, shared 128 GB machine), sigma 50 nm, Lc 500 nm,
h 50 nm, d0 = 3, full GMRES tol 1e-3, record benchmarks/results/mlfmm_scaling.md:
- Leaf rule uses the GLOBAL r_max: 100 nm box cells give r_max 116-141 nm -> leaf floor
  195-235 nm, root (edge ~ max(L, depth)) halving rounds up to 219-406 nm. Leaf size jumps with
  L (Si L <= 6 um: ~360 nm, L = 7-8 um: 219-250 nm) -> non-monotone memory per unknown.
- Automatic box rule without the lambda1/5 cap (200-400 nm cells, `--box-mesh-size uncapped`):
  r_max 230-490 nm, leaves lambda/0.7 ... lambda/0.35, near field ~ dense (Si L = 4 um: 307 GB
  est.); Ag L = 4 um: 10x near, 3.8x solve time.
- Si: iterations fall 154 -> 90-106 (w0 = L/4 = lambda/2 at L = 1 um is non-paraxial and lights
  the walls; box/top ratio falls with L); far matvec 85-95 % (R2 leaf order 45-60); leaf
  patterns 40-61 % and far memory 43-74 % of the (pre-d00317a) peak; 2N = 5.2e5 -> 79 GB peak
  with 100 nm box cells. gamma_solve 0.73 +- 0.07 single sweep, 0.88 with min-of-repeats.
- Uniform box (`--box-mesh-size 50e-9`) shrinks r_max to 64-67 nm, leaves 156-181 nm: Si L = 10 um
  2N = 1.02e6 ~73 GB near + far (87 GB peak model) -> the route to Si 1e6 today (estimate only).
- After d00317a the measured peak after setup = near + far within 1 % (no near transient).
- Ag: 1000-1700 iterations (ICTF+Jacobi), serial MGS in solver::gmres = 42-66 % of solve
  (~9 GB/s effective at k = 1700, 2N = 2.9e5); exact part <= 0.8 GB with big leaves.
  gamma_solve 1.20 +- 0.18 (5 sizes), 1.43 from 7e4; last interval 2.2 (noise-sensitive).
  Ag L = 3 um solve: 325 s quiet vs 511-619 s loaded (deterministic 1004 iterations).
- Peak model after d00317a: 1.5 near + 1.1 (leaf patterns + far tables) + 2 exact + 1 GB.
  Far tables (translators per distinct offset, 128 B x boxes x dirs workspace) matter at large
  leaves: Si uncapped L = 1.5 um R2 leaf L = 109 -> 24 GB patterns; old model missed them (-32 %).
- Timing noise from other agents: single medians up to 2x off; time the full/near/far applies
  interleaved; repeat runs and take the minimum (fit script does this per quantity).

WP22b3 (2026-10-11, CGS2 GMRES, angular-spectrum beam, quiet machine):
- Si uniform box (50 nm) is the best Si route: 64-85 iterations (graded 100 nm box: 96-104 at the
  same L, beam irrelevant), L = 10 um 2N = 1.02e6 in 954 s / 74.8 GB; gamma_solve 1.09 +- 0.04.
- Ag (Jacobi) iterations grow gamma ~0.3 (2272 at 4.2e5); solve time t(k) = a + matvec k + c k^2
  with c/2N = 0.89-1.03 ns (CGS2 23-27 GB/s); b within 11 % of matvec only for L >= 4 um (L = 3 um
  +27 %: per-iteration overhead). Full solves at 7.6e5 / 1.02e6 need 2.6-5 h: do not fit
  a 2 h background command -> cap with --max-iter (gives a conv=0 ROW). Residual extrapolation over
  the last 300 its: log-linear = low, power law = high bracket. Back-test caveat: the low end is
  not a strict lower bound - at the 1800 cap of L = 11 um it overshot the true 2272 by 2 its.
  Commit the GMRES histories with any projection (benchmarks/results/mlfmm_scaling_ag_histories.txt
  + benchmarks/mlfmm_scaling_projection.py); a review asked for it when only logs held them.
- Ag iteration counts scatter (one surface per L, non-monotone: L5 < L4); local gamma_it 9->11 um
  0.80 vs global 0.27, so a single interval's slope says little. Ag DoD stays open by measurement
  until the L = 17.5 um converged solve (WP22d server kit).
- Local leaf rule q = 0.8 (Si 100 nm box L = 4/8 um): peak -34/-26 %, setup +32/+20 %, same
  iterations/matvec. Peak after setup can be < near + far (far workspace pages first touched in
  the matvec); quote the end-of-run peak. Default q stays 1 (coordinator 2026-10-11): uniform box
  is the recommended Si setup, q = 0.8 the opt-in memory option for graded boxes.
- Cygwin /proc/loadavg reads 16-170 in-run (run means 24-84) from our own 24-thread run: only the
  pre-start value is a foreign-load indicator; after a run the 1-min average needs 2-10 min to
  fall below 5.

**Why:** Phase 4 DoD scaling and sizing of later large rough-surface runs (Phase 5).
**How to apply:** budget memory with the model (check far_est_gb vs far_gb rows); Si beyond
~5e5 unknowns needs the uniform box or a local leaf rule; Ag solves beyond ~4e5 need > 2 h even
with CGS2 (Jacobi iterations are the limit). See [[mlfmm-operator-findings]], [[ag-sphere-mlfmm-findings]],
[[gmres-observations]].
