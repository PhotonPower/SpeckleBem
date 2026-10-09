---
name: mlfmm-octree-findings
description: Measured MLFMM octree facts (WP17): leaf depth vs lambda/4 bound on spheres, surface list sizes, build/test timings
metadata:
  type: project
---

Measured on the WP17 uniform-depth octree (2026-10-09):

- Root = midpoint bounding cube, slightly smaller than the sphere diameter (midpoints lie inside
  the sphere). 4 um icosphere n=6 (122,880 RWG) at lambda = 500 nm: root 7.999 lambda, so the
  lambda/4 level is just inadmissible and the leaves stop at lambda/2 with 2..215 elements
  (mean 106) for max_elements_per_leaf = 100; 5 levels, 1160 leaves. Build 0.03 s release.
- Surface meshes fill only part of the 6x6x6 parent neighbourhood: interaction lists reach
  ~61, never near 189 (sphere n=3, small leaves); near lists ~13 mean, 18 max.
- Brute-force leaf-pair completeness: 13M pairs ~1 s release; keep <= 1e6 pairs (n=3,
  max 6 per leaf: 824 leaves) for win-debug (~1.8 s). Structure checks over three meshes
  ~3.4 s in win-debug.

**Why:** WP18/19 test budgets and the sampling/near-field cost depend on leaf sizes.
**How to apply:** when choosing octree params or test meshes for MLFMM tests; see
[[gmres-observations]] for solver budgets.
