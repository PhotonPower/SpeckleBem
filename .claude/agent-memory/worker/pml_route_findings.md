---
name: pml-route-findings
description: WP-PML3a masked RHS / windowed far field / PML Simulation - cost of Si in the PML (k-aware touching order), dark band needs a2 >~ 3.5 w0 with w0 >~ lambda, test-design pitfalls
metadata:
  type: project
---

WP-PML3a (2026-10-13): `op::pml_masked_functions`, masked `assemble_rhs`, `post::CurrentWindow`,
`Simulation(OpenSurfaceMesh, PmlTermination, ...)`, `Simulation.with_pml` in Python.

- **Cost is set by |k s| h in the PML**: the stretched touching rule adds
  ceil(0.75 |k| s_max h) - 4 points (cap 16). Si (|k| = 5.4e7/m) at sigma_max 4, h = 100 nm hits
  p = 16: 128-triangle open surface (56 PML triangles) took 24 s dense assembly in release;
  eps 2.25 + sigma_max 1 at the same mesh: 0.1 s. Structural tests must use a low-index object
  and small sigma_max (or stretched_touching_points 4 does not help: the k raise dominates).
- **Dark band**: rigorous beam needs w0 >~ lambda (spectrum cut at k_t = k), and 1e-4 needs
  a2 = pml_start >~ 3.5 w0, so a2 >~ 1.75 um at 500 nm; small unit-test surfaces never pass the
  default 1e-4 (relax dark_band_tolerance to 0.9 in structural tests). Measured: w0 0.5 um,
  a2 1.75 um -> ratio 1.27e-5.
- PML masks ~60 % of the test functions on the brief geometry (patch 1.5, taper 0.5, gap 0.5,
  PML 1 um, h 125 nm: 3424 of 5720).
- Smoke (flat Si, patch 1.5/taper 0.5/gap 0.5/PML 1 um, h 125 nm, w0 0.5 um, sigma 4 m 2, ICTF
  dense + LU, 2N 11440): R_sim 0.388091 vs R_beam 0.388176 (-0.022 %), edge |J|, |M| 1.5e-9 /
  6.5e-9 of max; assembly 1502 s, LU 351 s (both under a concurrent debug build/ctest).
  Test: tests/validation_large/test_pml_smoke_large.cpp (SPECKLEBEM_PML_SMOKE_H_NM).
- GCC 16 -O3 gave a location-less `-Wnull-dereference` in a Catch2 test body after inlining a
  helper (reference radiation integral); `[[gnu::noinline]]` on the helper silenced it.

**Why:** WP-PML3b (7a) and WP-PML4 run dense Si/Ag PML cases; cost and dark-band limits size them.
**How to apply:** budget dense PML runs by PML triangles x p^4; see [[stretched-kernel-findings]],
[[open-surface-pml-findings]].
