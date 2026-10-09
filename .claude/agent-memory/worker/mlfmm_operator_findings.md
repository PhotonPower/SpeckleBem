---
name: mlfmm-operator-findings
description: WP20b full MLFMM (near + far) measurements - matvec errors vs dense at d0 = 3 / 5, r_max/a rule, timings, memory, workspace-pool speedup, test-cost lessons
metadata:
  type: project
---

Measured in WP20b (2026-10-10, win-release, 24 cores; tests/validation_large/test_mlfmm_vs_dense_large.cpp),
vacuum exterior, lambda0 = 500 nm, dense oracle target 1e-6, max over 3 random x:
- d0 = 3, lambda/4 leaves (r_max/a ~ 0.57-0.59): icosphere R = 1 um n=4 (2N 15360) 1.7e-4 (n=1.5
  PMCHWT), 2.1e-4 (Si ICTF); rough box 2 um x 0.3 um @ 50 nm (2N 24960) 7.2e-5 (n=1.5 ICTF),
  4.3e-5 (Si PMCHWT). 2N = 88320 box: 2.9e-4 on 128 exact rows (assemble_sparse rows oracle).
- **d0 = 5 achieved on multilevel trees** with r_max/a <= 0.29: lambda/2 leaves 3.7e-6 / 6.5e-6
  (sphere), 6.7e-6 / 4.2e-6 (box); lambda leaves on a 4 um plate @ 100 nm: 5.5e-7.
- Costs: dense assembly (target 1e-6) 25-100 s for 1.5-2.5e4; MLFMM setup 3-27 s, apply
  0.3-0.8 s (dense matvec 0.25-0.7 s at these sizes, so no apply win below ~3e4). d0 = 5 with
  lambda/2 leaves: near field up to 1.5 GB at 2N = 2.5e4 (vs 0.23 GB for lambda/4).
  2N = 88320: setup 31 s, apply 2.0 s, 5.5 GB (dense would be 119 GB).
- Workspace pool (in-pass zeroing, no per-call alloc): far apply of 2N = 61440 sphere 1.52 -> 0.66 s.
- GMRES via Simulation: Si sphere 2N 3840 currents 4.6e-5; rough box 7680 ICTF+Jacobi 3.2e-4
  (329 its, ~22 s each -> moved to validation-large).
- Unit tests comparing MLFMM to dense need accurate far entries: the cheap 1-point far rule
  differs from the radiation patterns by ~3 % in K columns; target_accuracy = 1e-5 with cheap
  touching pairs fixes it but costs ~90 s under ASan -> accuracy checks release-only.

**Why:** ADR 0008 d0 = 5 status, WP21/WP22 leaf policy and test budgets.
**How to apply:** d0 = 5 needs leaves >= lambda/2 and meshes with r_max/a <= 0.3; see
[[mlfmm-multilevel-findings]], [[mlfmm-octree-findings]].
