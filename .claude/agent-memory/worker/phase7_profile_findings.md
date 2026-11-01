---
name: phase7-profile-findings
description: WP-PF1 Phase 7 profile - shared-desktop load detection (foreign job, foreign builds, queued runners), 24 logical threads, stage shares, thread scaling (matvec saturates at 12, Si ACA tail), runner gating
metadata:
  type: project
---

WP-PF1 (formerly WP-P1 in the branch names; 2026-10-31 / 11-01), driver `benchmarks/profile_study.cpp` (`--no-solve` for scaling runs), record
`benchmarks/results/profile.md`; library stats `MlfmmOperator::setup_stats/apply_stats`,
`MlfmmFarOperator::setup_stats/apply_stats`, `PmlRingOperator::apply_stats` (on/off switch,
mutex-guarded sums, bitwise-neutral).

- **Idle check by process names is not enough**: the desktop also runs foreign compute jobs
  (another code, in a series; one process, 40-100 % of one logical thread) with a foreign build
  (not ours) in between, the coordinator builds/tests the main checkout (cc1plus, ctest,
  specklebem_*_tests from build/ outside the worktree), and a queued runner of an earlier session
  can start a study minutes later. A first U run was 35 % slower (assembly 218.7 vs 161.9 s) with
  zero name hits; a 16 um run overlapping the coordinator's ctest had orth 27 s vs 0.6 s idle.
  Check total CPU too: `typeperf` with the localised total-CPU performance counter, 5 samples
  (the English counter path fails on a localised system); the per-process counter of the same
  localised family gives 100 = one logical thread.
- Gate each timed run on: no cc1plus / ctest / *_tests / pytest / other profile_study, then wait
  ~90 s and re-check (another runner starts within seconds of a run's end); log a 30 s CPU trace
  (typeperf -si 30 -o csv) to compute busy threads per stage window.
- The desktop has **24 logical threads** (12 cores, SMT): omp_get_max_threads() = 24 by default,
  so the earlier records (WP-SC3..SC5, KR1) ran at 24 threads, not 12.
- Background bash tasks die at 2 h: chain long series as separate tasks; killing a series bash
  leaves its current child run alive (useful to stop a queue without losing the run).
- Measured (24 threads, loaded 1.0-1.25x idle): assembly 93-97 % PML ring (ACA far ring 47 % Ag U /
  74 % Ag 16 um / 81 % Si U; near ring 50 / 18 / 12 %), everything else ~7 % (7.3 / 7.1 / 3.1 %). GMRES 87-99 % matvec
  (far passes 50-58 % Ag, 95 % Si; ACA ring apply 20 % Ag), ILUT apply 10-12 % Ag, orth ~1 %.
  Plane field 161^2 = 4.2x GMRES at Ag U. Matvec fastest at 12 threads (bandwidth; SMT +5-19 %).
  Si ACA far ring 12 -> 24 threads only 1.03x (tail of long blocks on ~5 threads), Ag 1.59x.
  Si far R2 plan (order searches, 31-38 s) does not scale with threads at all.
- Repeatability between identical 24-thread runs: stage-level shares agree within 1.3 pp
  (assembly) / 0.4 pp (matvec near / far / ring in GMRES), but the per-pass far shares do not
  (Ag U in GMRES +-4 pp, standalone up to 12 pp, e.g. translation 19.0 vs 8.8 %): quote per-pass
  shares only as rough. Review round: a "replica bitwise NO" must throw, not print (driver now
  does); "parts = 100 %" of the matvec is circular (library other = total - parts), the
  informative numbers are "other" and "far other".

**Why:** Phase 7 decisions (threads / GPU / algorithmic) rank on these shares.
**How to apply:** for timed runs on this machine gate as above and quote the thread count;
compare shares, not seconds. See [[operator-reuse-findings]], [[ilut-kernel-findings]], [[pml-ring-findings]].
