---
name: drc-power-findings
description: WP25 absolute DRC / radiated power - GL-in-theta grid weight sums not exact for small n_theta, projected-Mie power errors, Si sphere C_sca, debug costs, bitwise promotion of the WP22c grid
metadata:
  type: project
---

WP25 (2026-10-14): `post::SphereGrid`, `hemisphere_grid`, `sphere_grid`, `radiated_power`, `drc`,
`drc_polarized`, absolute cylinder DRC, `GaussianBeam::power()`, `Simulation::incident_power /
drc / radiated_power / reflected_power / scattered_power`.

- Grid = Gauss-Legendre in theta (not in cos theta): weight sum is only the GL rule for
  int sin(theta): hemisphere rel error 0.11 (n=1), 2e-8 (n=4), rounding from n=7; sphere 0.57
  (n=1), rounding from n=8. Polynomials in cos theta are spectral, not exact (sphere needs ~24
  nodes for cos^6 at 1e-13). Do not claim "exact" in tests.
- The library grid reproduces the old fresnel_flat helper bitwise (same op order; checked with a
  temporary test), so `fresnel_flat::HemisphereGrid` is now an alias of `post::SphereGrid`.
- Projected Mie currents (fields_test MieSetup) give poor total power at 500 nm, n = 2
  (h ~ lambda/3.3): C_sca -26 %; at lambda = 1 um n = 2: -4.6 %. Solved Si sphere n = 3 via
  Simulation (default options): C_sca -1.17 % vs Mie (2 % bound), 64x128 far field 1.9 s / 4 thr.
- win-debug: Simulation dense default options at icosphere n = 2 + LU + far field ~490 s; use
  n = 1 with cheap degrees (9 s).
- Flat Si PML case (2N 11440, h 125 nm): int D dOmega -0.022 % vs R_beam (same as WP-PML3a
  smoke); assembly 1560 s + LU 62 s at 4 threads on the loaded machine. Far field cost 12 ns per
  (direction x source point) at 4 threads: 1 deg grid 9 s all triangles, 3.7 s windowed;
  extrapolated 2N = 1e6: ~790 s per 1 deg grid at 4 threads.
- Cylinder absolute DRC vs far-field DRC: deviation exactly ~1/r (6.9e-2 at 10 um ... 3e-4 at
  2.56 mm for a 1 um sphere at lambda 1 um).

- WP25 review (2026-10-14): AngularSpectrumBeam::power / GaussianBeam::power = 1 + f^2/2 +
  0.313 f^4 (f = lambda/(pi w0 n1); same unit on-axis amplitude, no renormalisation); test
  tolerance f^4 passes at w0 = 5 and 10 lambda with region_radius = w0 (cheap grid).
- PML-route exception rule (docs/10): valid call / wrong state (field, bistatic_rcs,
  scattered_power) -> std::logic_error / RuntimeError; k_z > 0 directions -> invalid_argument /
  ValueError (Simulation::far_field, drc, radiated_power). pybind maps std::logic_error to
  RuntimeError via the std::exception fallback.
- Cheap structural solves in win-debug (loaded machine, serial): tiny rough box 0.4 um, h 100 nm,
  depth 0.3 um, cheap quadrature + LU ~4 s; tiny 8x8-cell rough PML surface ~4.5-6 s; the
  Gaussian generator needs Lc >= 2h and the PML taper >= Lc, so a hand-made white-noise map
  (nominal Lc = h) keeps the PML case at 8x8 cells. No CI-run validation test solved a flat box
  before the review: tests/validation/test_flat_interface_fresnel.cpp has only reference checks
  plus the tiny API-agreement box (reflected_power bitwise == support hemisphere_power).

**Why:** reference for Phase 6 DRC / power post-processing and its tests.
**How to apply:** size grids n_theta >= 8; see [[fresnel-flat-findings]], [[pml-route-findings]].
