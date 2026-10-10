---
name: mlfmm-lossy-policy-findings
description: WP21 (+ review round) measurements of the MLFMM region policy - block-check errors, Ag exact-part sizes (40 B/pair storage), x*(d0) fallback criterion, budget, leaf-ratio 0.6/0.3 consequences
metadata:
  type: project
---

Measured in WP21 (2026-10-10, win-release, slow ctest "mlfmm: region policy sweep (slow)"),
lambda0 = 500 nm, vacuum exterior, kBlockCheckPairs = 3 nearest offset classes, <= 24 bases per
box:
- **Block check (max rel. Frobenius error of L/K, nearest offsets) exceeds 10^-d0 for lossless
  regions** even where the docs/05 matvec meets it: R1 at lambda/2 leaves 8.8e-4 (d0 = 3),
  4.0e-4 (d0 = 5, matvec ~4e-6); lambda/4 leaves 1.35e-2 (d0 = 3, matvec ~2e-4). So the block
  check gates only lossy regions (matvec criterion governs lossless).
- Ag interior (alpha = 39/um): search not achievable at lambda/4 and lambda/2 leaves. Exact box
  pairs on the two finest levels, coarser levels truncated.
- Review round: truncation needs alpha d >= x*(d0) (11.76 at d0 = 3, **16.69** at d0 = 5, not
  16.65). Moderate loss is often accepted by the expansion via the "<= 2 x lossless analogue"
  rule (n = 1.5 - 1j at lambda/4 leaves: block 2.9e-3 vs 2.7e-3); n = 1.5 - 0.6j at lambda/2
  leaves is rejected -> cause lossy_region (x*/alpha = 6.2 leaf edges > 4). n = 1.5 - 2j on a
  small sphere (R = 0.25 um) passes x*/alpha <= 4 a yet has 77 % of N^2 exact: small objects
  are covered by the budget, not the criterion.
- Exact part with (L, K) storage (40 B/pair vs 96 B for four combined entries, 2.4x less): Ag
  icosphere R = 1 um sub 5 (2N = 61440, d0 = 3, lambda/4 leaves): 18.0e6 pairs = 687 MB (box-pair
  bound 9.3e7 = 5x too high -> the per-basis count decides; it takes 0.1 s), near 1133 MB, setup
  20 s, peak RSS 4.9 GB, exact-rows error 1.1e-4. Exact/near ~0.6 -> automatic budget 2 x near.
- Leaf ratio 0.6 (d0 <= 3) / 0.3 (d0 > 3): the 50 nm / 70 nm meshes return to lambda/4 leaves at
  d0 = 3 (88 320 box: 5.5 GB, peak RSS 6.4 GB, 2.9e-4).

- WP21f: automatic budget now min(max(2 near, 1 GiB), 16 (2N)^2). Both regions forced exact
  (40 B/pair each, 80 B vs dense 64 B per pair) exceed it once far pairs > 80 % of N^2 (4-level
  icosphere sub 3: 94.6 %, 266 MB vs 225 MB): forced-exact diagnostics need an explicit budget.
- Jump terms: R1 + R2 vs dense does NOT detect a flipped jump sign in assemble_region_sparse
  (cancels); per-region vs dense with the other weights zeroed gives 2 (ICTF) / 3e-3..7e-3
  (PMCHWT) relative max entry error with the flip.

**Why:** WP22 (393k Ag sphere) sizing: projected exact part ~15 GB at d0 = 3, ~30 GB at d0 = 5
(scale pairs/row ~586 by (h_ref/h)^2 and by (x*(d0)/x*(3))^2).
**How to apply:** use these numbers for WP22 memory planning; see [[mlfmm-operator-findings]].
