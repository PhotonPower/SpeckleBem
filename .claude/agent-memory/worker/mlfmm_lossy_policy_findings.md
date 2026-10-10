---
name: mlfmm-lossy-policy-findings
description: WP21 measurements of the MLFMM region policy - block-check errors vs 10^-d0 for lossless/Si, Ag decisions and exact-part sizes, leaf-rule consequences
metadata:
  type: project
---

Measured in WP21 (2026-10-10, win-release, slow ctest "mlfmm: region policy sweep (slow)"),
lambda0 = 500 nm, vacuum exterior, kBlockCheckPairs = 3 nearest offset classes, <= 24
(corner-near half + spread) bases per box:
- **Block check (max rel. Frobenius error of L/K, nearest offsets) exceeds 10^-d0 for lossless
  regions** even where the docs/05 matvec meets it: R1 at lambda/2 leaves 8.8e-4 (d0 = 3),
  4.0e-4 (d0 = 5, matvec ~4e-6); lambda/4 leaves 1.35e-2 (d0 = 3, matvec ~2e-4); small sphere
  (R = 0.5 um) level-2 boxes 2-5e-3. So the block check gates only lossy regions (ADR 0008
  WP20a/b amendments: matvec criterion governs lossless); Si interior (a = 2-4 lambda_i) 3e-9 ...
  3.5e-5, always below 10^-d0.
- Ag interior (alpha = 39/um): search not achievable at lambda/4 and lambda/2 leaves (errors
  1e3 ... 1e9). Decay bound on support bounding boxes: exact box pairs only at the leaf level
  (+ level leaf-1 at d0 = 5 on R = 1 um), coarser levels all truncated. Exact part before the
  per-basis-pair refinement: 0.4-2 GB at 2N = 1.5-2.5e4 (larger than the d0 = 3 near field).
- Leaf rule r_max/0.3 moves the R = 1 um icosphere n = 4 and the 2 um / 50 nm box to lambda/2
  leaves at d0 = 3; lambda/4 leaves need r_max <= 37.5 nm (R = 0.5 um n = 4, box mesh 35 nm).

**Why:** WP22 (393k Ag sphere) sizing and any revisit of the block-check gate.
**How to apply:** expect the Ag exact part to scale like the leaf-level interaction lists
(~3x the near field on surfaces); see [[mlfmm-operator-findings]].
