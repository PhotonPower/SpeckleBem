---
name: mlfmm-octree-findings
description: Measured MLFMM octree facts (WP17): root/anchoring, lambda/4 floor tolerance, leaf sizes on 4 um spheres, plate/thin-box layers, list sizes, test timings
metadata:
  type: project
---

Measured on the WP17 uniform-depth octree after the review fix (2026-10-09):

- Root = padded **vertex** bounding cube (edge e(1+2e-6)), anchored at the lower corner. An
  icosphere has vertices at the poles, so the 4 um sphere root is exactly 8 lambda at 500 nm.
  The earlier midpoint-centred root was 7.99 lambda and lost the lambda/4 level at an exact
  floor; now a level is admitted at >= (1 - kMinBoxSizeTolerance = 1e-2) * floor.
- 4 um sphere, lambda = 500 nm, max_elements_per_leaf = 100: n=4: 4 levels (lambda leaves,
  max 47); n=5: 5 levels (lambda/2, max 54); n=6 (122,880 RWG): 6 levels, 4568 lambda/4 leaves,
  max 58 (was 215 at lambda/2); n=7 (491,520): 6 levels, lambda/4 leaves still floor-bound,
  max 211 mean 104 (was 845). n=4 with max 4 per leaf: 6 levels, 3660 leaves, max 4.
- Centring cut thin geometry at its mid-plane: a 2 um plate with +-1e-15 jitter had 2048 leaves
  vs 1024 flat (twice the boxes per level); anchored: both 4^l per level, identical. Thin rough
  box (L 2 um, depth 0.3 um): level 1 had 8 boxes, now 4. Note: grid-aligned midpoints exactly
  on box faces shift boxes with the pad, so leaf depth on regular plates can change by one.
- Surface meshes fill only part of the 6x6x6 parent neighbourhood: interaction lists reach
  ~61, never near 189 (sphere n=3, small leaves); near lists ~13 mean, 18 max.
- Brute-force leaf-pair completeness: 13M pairs ~1 s release; keep <= 1e6 pairs (n=3,
  max 6 per leaf: ~800 leaves) for win-debug (~1.8 s). Build of 122,880 RWG: 0.03 s release.

**Why:** WP18/19 test budgets and the sampling/near-field cost depend on leaf sizes.
**How to apply:** when choosing octree params or test meshes for MLFMM tests; see
[[gmres-observations]] for solver budgets.
