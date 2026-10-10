---
name: angular-spectrum-beam-findings
description: WP-E1 AngularSpectrumBeam facts - plane-wave counts, eval cost, convergence behaviour, constants mismatch, paraxial/power O(f^2) coefficients
metadata:
  type: project
---

Measured 2026-10-10 (WP-E1, `excitation::AngularSpectrumBeam`, tol 1e-10, lambda 500 nm):
- Plane-wave counts (R = region radius): w0 = 1 um, R = 4 um: ~1 500-1 600; w0 = 2.5 um,
  R = 11 um: 1 664; w0 = 5 um, R = 41 um: 3 608; R = 80 um: 14 100. Cost ~6.6 ns per wave per
  point for E (SSE2, 4 partial sums), scalar study loop 8 ns, libm sin+cos 11.5 ns; `fields()`
  (E+H) ~1.2x E. The assembler calls electric_field and magnetic_field separately (2x cost).
- Grid convergence is super-exponential: one 1.5x refinement jumps from > 1e-10 to ~1e-13, so
  joint 1.5x steps overshoot; per-direction ~25 % steps save 10-60 % waves.
- Core constants are CODATA-rounded: eta0 / (mu0 c0) - 1 = 3e-11, so exact plane-wave sums
  show |curl E + j w mu H| ~ 3e-12 k |E|, not 1e-16. Use the mismatch as the tolerance floor.
- FD Maxwell check needs h = lambda/400 with Richardson (lambda/50 gives 3e-7 truncation);
  achieves ~8e-11.
- Leading-order coefficients (f = lambda_1/(pi w0)): max |E - E_paraxial| over rho <= 2 w0,
  |zeta| <= z_R/4 ~ 0.35-0.37 f^2 (ratio 0.2496 when w0 doubles); on-axis phase gradient
  -k (1 - f^2/2) (Gouy); power (1 + f^2/2) pi w0^2/(4 eta) (projection + cos alpha).
- Eigen `cross` on complex vectors conjugates: a Poynting test with `e.cross(h)` passed by
  luck at the waist (real fields); use an explicit unconjugated cross with conj(H).

**Why:** test budgets and tolerances for anything using the rigorous beam (rough-surface
validation, Phase 5).
**How to apply:** pick region_radius to cover the mesh; budget beam RHS cost as waves x points
x 2 x 7 ns. See [[box-validity-findings]].
