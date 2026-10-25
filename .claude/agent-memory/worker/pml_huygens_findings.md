---
name: pml-huygens-findings
description: WP-PML6 Huygens-plane far field of the PML route — 7f/7g numbers, plane-window truncation diagnosis, extended-plane (wide gap) study, run costs, driver/guard facts
metadata:
  type: project
---

WP-PML6 (ADR 0010; record benchmarks/results/pml_huygens.md). Status 2026-10-25: 7f met, 7g not met, default
far-field source stays `FarFieldSource::Currents`; lead decision on the follow-up pending.

- 7f (flat Ag/Si, dense, 2N 11 440): Huygens vs currents 2.7/2.8e-4, dR <= 4e-7; z_h x1.5 2.6e-4; spacing /1.5
  8e-7. Plane fields 0.3 s (729 pts).
- 7g (rough Ag U, fast path, 602/669/705 its): 7d Huygens 2.78 % (own windows) / 1.83 % (same abs window),
  currents 2.57 %; z_h x1.5 1.05 %; window x0.7/x1.3 1.75/0.66 %. Si 7e with plane: l2/eps pass, dR -5.3e-4 fails.
- Cause = plane window (raised cosine across the 0.5 um gap, Delta = lambda, plane bounded by pml_start):
  grazing band 75-90 deg -67 % vs currents; point-source error ~ pi^2/(2(k Delta)^2) per end (measured 0.28 /
  4.7e-2 / 1.9e-2 / 9.8e-3 / 4.2e-3 / 2.3e-3 at Delta = 1/2/3/4/6/8 lambda). Not physics, so ADR clause n/a.
- Extended-plane diagnostic (ugap3 = U with 3 um gap, l12gap3 = Lx1.2 with 2.625 um gap; 2N 62 016, ~890-900
  its, ~17 + 26 min per pair): z_h x1.5 falls 1.05->0.10 % (theta<=30: 0.81->0.018 %) for Delta 1->6 lambda,
  plane vs currents at theta<=30 0.018 % at 6 lambda, but 7d stays 2.4-2.5 % (theta<=30 2.2-2.4 %) for BOTH
  sources -> L dependence looks physical in the cone; hemisphere not converged (grazing -44 %).
- The plane cannot go over the PML: exterior_scattered_field and huygens_plane(s,opt) throw beyond the stretch
  start, and real observation points over the PML are not valid for the stretched-source representation
  (needs stretched observation points + convergence theory). Beyond the outer PML edge: nothing at all.

- 7f premise (review round 2026-10-25): the plane matches the currents only if the reflected beam has decayed
  before the plane window. Flat Ag, w0 = lambda, Delta = lambda: window from 1.5 w0 (7h CI geometry, 2N 4 592)
  L2 8.1e-3 / dR -5.4e-4 (Si alike); from 2 w0 (2N 6 016) 2.1e-3; from 2.5 w0 (7a zones) 2.7e-4. A thin PML
  (W = lambda, sigma_max 2) and a 60 x 120 hemisphere change the 7a values by < 1 %: CI case flat_ci_case(),
  2N 7 632, ~2-8 min per material depending on load (dense LU, 1.9 GB). 7a-size 7f is validation-large now.
- Validation-large 7g timings on a loaded machine: rough Ag 3 471 s (identical values); rough Si box + PML
  > 1 h when loaded: run Si and Ag in separate background commands (2 h stop).
- detail::stretched_distance (fields.hpp) throws logic_error for Im R~ > 0: Im R~ > 0 needs the observation
  point beyond the source on its side; a point between the start and the source still has Im R~ <= 0.

**Why:** the lead must decide between accepting physical L dependence (theta<=30 cone), validating a wide-gap
plane (window x(1+-0.3) not yet repeated there, one seed only), or a plane over the PML.
**How to apply:** for any follow-up WP reuse `benchmarks/pml_huygens.cpp --study 7g --cases ugap3,l12gap3`
(delta_windows: tags .HD<d>/.Hz15D<d>/.CD<d>/.HabsD<d>) and the scratch cone script idea (theta-cone L2 and band
powers from `--out` far-field files). Benchmarks are OFF by default: configure win-release with
-DSPECKLEBEM_BUILD_BENCHMARKS=ON. Test pitfall: Simulation zone half widths (1.875e-6 + 0.5e-6) are not bitwise
the literal 2.375e-6; compare windows with a 1e-12 margin or take them from size_of(). Validation-large 7g Ag case
~55 min on a loaded machine (3 fast solves). See [[pml-rough-ag-findings]], [[pml-ring-findings]].
