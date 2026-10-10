---
name: angular-spectrum-beam-findings
description: WP-E1 AngularSpectrumBeam facts - plane-wave counts, indicative eval cost, aliasing outside R, convergence, constants mismatch, paraxial/power O(f^2) coefficients
metadata:
  type: project
---

Measured 2026-10-10 (WP-E1, `excitation::AngularSpectrumBeam`, tol 1e-10, lambda 500 nm):
- Plane-wave counts (R = region radius): w0 = 1 um, R = 4 um: ~1 500-1 600; w0 = 2.5 um,
  R = 11 um: 1 664; w0 = 5 um, R = 41 um: 3 608; R = 80 um: 14 100 (~ (R/w0)^2 scaling).
- Cost (indicative only, timed on a shared busy machine, not to be quoted in committed docs
  without a serial re-time on an idle machine): a few ns per wave per point for E (SSE2, 4
  partial sums), below libm sin+cos; `fields()` (E+H) ~1.2x E. Since the review round the RHS
  assembly and post::total_field call `Excitation::fields()` (one pass).
- Outside the controlled ball the finite sum aliases (trapezoid in phi): ghosts ~1e-4 at 2 R
  and ~0.1 at 4 R (w0 = 2.5 um, R = 4 w0). Simulation therefore rejects meshes with a vertex
  beyond controlled_radius(); evaluations beyond 1.2 R warn once per beam object.
- Grid convergence is super-exponential: one 1.5x refinement jumps from > 1e-10 to ~1e-13, so
  joint 1.5x steps overshoot; per-direction ~25 % steps save 10-60 % waves.
- Core constants are CODATA-rounded: eta0 / (mu0 c0) - 1 = 3e-11, so exact plane-wave sums
  show |curl E + j w mu H| ~ 3e-12 k |E|, not 1e-16. Use the mismatch as the tolerance floor.
- FD Maxwell check needs h = lambda/400 with Richardson (lambda/50 gives 3e-7 truncation);
  achieves ~8e-11 (docs/05 row references that unit test).
- Leading-order coefficients (f = lambda_1/(pi w0)): max |E - E_paraxial| over rho <= 2 w0,
  |zeta| <= z_R/4 ~ 0.35-0.37 f^2 (ratio 0.2496 when w0 doubles); on-axis phase gradient
  -k (1 - f^2/2) (Gouy); power (1 + f^2/2) pi w0^2/(4 eta) (projection + cos alpha).
- Eigen `cross` on complex vectors conjugates: a Poynting test with `e.cross(h)` passed by
  luck at the waist (real fields); use an explicit unconjugated cross with conj(H).
- A std::atomic member makes a class non-copyable; the beam wraps its warn-once flag in a tiny
  struct whose copy ctor gives a fresh flag, so the beam stays copyable.

**Why:** test budgets and tolerances for anything using the rigorous beam (rough-surface
validation, Phase 5).
**How to apply:** pick region_radius >= the max vertex distance from the focus (Simulation
throws otherwise) and cover near-field observation points; budget beam RHS cost as
waves x points x one sincos. See [[box-validity-findings]].
