---
name: sweep-reuse-findings
description: WP-KR3 speckle sweep run_sequence - tiny Si box test costs, small reuse saving on the Si box route, lead decision series (ii) THETA=30 default, p-only analysis, ruff state, s-pol tag
metadata:
  type: project
---

Findings from WP-KR3 (2026-10-31, branch wp/kr3-sweep-operator-reuse):

- Tiny Si box case (L = 1 um) is NOT cheap with the real operator: the box is 5.6 um deep, and the
  MLFMM far operator of the Si interior dominates. h = 100 nm (2N 14 880): ~65-72 s assembly
  (far 2.2 GB), 35-66 s per solve (78 it at tol 1e-4). h = 200 nm (2N 3780): ~65 s assembly
  (far 56 s, 1.7 GB), ~20 s per solve (31 it at tol 1e-3).
- Cheap recipe used by tests/python/test_speckle_sweep.py: wrap `_specklebem.Simulation` (both
  run_case and run_sequence get it) with kernels quad_degree_far=1/near=2/sing=2,
  outer_grading_levels=0, target_accuracy=0 AND mlfmm automatic_leaf_size=False,
  min_box_size_lambda=4 (2 levels, all near, no far): 0.5 s assembly, ~1.3 s solve (~50 it) at
  tol 1e-2. Reduced kernels with the automatic octree at h = 100 nm ran > 10 min: avoid.
  min_box_size_lambda=2 (3 levels) brings the far back: 116 s.
- Bitwise: run_sequence (set_excitation) vs fresh run_case per case gave identical currents, E,
  intensity, far_F, residual history, R, it on the box route with MLFMM (also with full kernels).
- On the Si box route (no preconditioner) the setup is small: 8 um record 111-218 s of 27-50 min
  per case, so operator reuse saves little there; the big saving is on the PML/ILUT route
  (792 of ~850 s at 16 um, WP-KR1). Say so when a brief assumes assembly dominates.
- The speckle sweep Python files (module, driver, tests) were not ruff-formatted on main (15 files
  tree-wide; CI does not run ruff); WP-KR3 put the reformat in a separate droppable last commit.
- CaseParams.tag() had no polarisation: s / p pairs collided in one directory; now `_pols` for s
  (p tags unchanged).
- Python edits need `cmake --build --preset win-release` before pytest: the build-tree package
  holds copies of the .py files and comes first on the pytest pythonpath.

- LEAD DECISION 2026-10-31 (KR3 review): series (ii) keeps THETA=30 as the default (0 deg partners =
  the record's desktop seed-1 solves, decision b1; 5 cases, 6-12 h server); THETA=0,30 is only the
  option "same-operator pair" (+ the five 0 deg solves: 2.75 h desktop = 24.4+25.8+35.2+32.1+47.9 min
  solve+plane+far, x 1.9-2.5 = 5-7 h; series 11-19 h). Do not sell operator reuse on the Si box
  route; Ag PML 16 um saves ~780 s per extra excitation (792 s fresh vs 11-22 s loaded swap).
- speckle_sweep_stats.py / the DoD test take |E_x|^2 (p co-pol): since the KR3 review they keep
  pol == "p" only (scan_cases: s skipped + listed, JSON skipped_cases) and refuse mixed theta_deg
  unless --theta-deg selects one (the DoD fixture selects 0). Any new list option in the driver
  needs the same check in the analysis.
- The real-solver kr3 fixture with 0/30 x p/s (4 cases per path): fixture 4.1 s -> 8.7 s. The
  WP-KR1 PML test (test_operator_reuse.py) does have an MLFMM far operator (3 levels, 66 MB), so it
  covers the far path under set_excitation; the kr3 reduced operator has none (2 levels).
- Scratchpad gotcha: a stray scratchpad/py.py shadows pytest's `import py` when a script runs from
  the scratchpad root; run helper scripts from a subdirectory.
- Stats script and DoD test were not ruff-formatted before the KR3 review (format-only commit).

Related: [[operator-reuse-findings]], [[speckle-sweep-findings]].
