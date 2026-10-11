---
name: ag-absorption-findings
description: WP-A1 Ag absorbed power from currents at 45 deg (flat box) - local (n x M).J* low at L = 4 um only (spurious +-k sin(theta) standing wave, M phase +4-6 mrad); P_ext - P_sca correct; not solver, not MLFMM-generated
metadata:
  type: project
---

Measured 2026-10-11 (WP-A1, benchmarks/ag_absorption.cpp, record benchmarks/results/ag_absorption.md),
Ag 45 deg s, h = 50 nm, ICTF + Jacobi, MLFMM, L/w0 = 5:
- L = 2 / 3 / 4 um: local A -0.1 % / -2.1 % / -43 % vs A_beam; A_ext = (P_ext - P_refl - P_fwd)/P_inc
  within +0.2 % in all three. Formula exact (degree 2 = 6), walls <= 1.3e-4.
- L = 4 um: refinement to true residual 1.07e-6 leaves A unchanged (not solver; at 0 deg it WAS solver).
  Error spectrum along x: equal |M_err| peaks at +0.70 k0 and -0.70 k0 (mirror of the specular current)
  -> spurious standing wave; absent at L = 3 um.
- MLFMM: d0 = 3 vs 5 identical at L = 3 um; L = 4 um exact-row residual 2.9e-5, MLFMM error on the
  analytic currents 4.4e-5 and spectrally flat (no mirrored content). d0 >= 4 at L = 4 um: 500 nm
  leaves, 17.6 GB setup -> not run. Cause (intrinsic box mode vs amplified MLFMM error) open.
- RT0 interpolant of the analytic currents: 12-13 % pointwise error, but integrated A within -1..-3 %;
  strip-local A has +-20 % antisymmetric interpolation artefacts (do not read local A per strip).
- Tooling: --refine (x += GMRES(Z, b - Zx)), --save/--load-currents, --row-check / --strip-check via
  op::assemble_sparse (exact rows: 150 rows x all columns in ~1 s).

**Why:** Ag absorption from surface currents amplifies mrad current errors by ~60 (Re Z_s/|Z_s| = 1.6 %).
**How to apply:** use P_ext - P_sca for Ag absorption; check A_ext and the M-vs-J phase before blaming
discretisation; GCC 16 -Warray-bounds false positive with make_shared<PlaneWave> in a TU that also
includes the AngularSpectrumBeam header -> [[gnu::noinline]] helper. See [[fresnel-flat-findings]].
