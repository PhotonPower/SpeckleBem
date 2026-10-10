---
name: box-validity-findings
description: WP-V1 measured facts on the rough closing box under beam illumination (Ag/Si, dense) - grading error vs depth, rim-illumination-driven depth sensitivity, SPP share, edge power, beam model, GMRES cost, study pitfalls
metadata:
  type: project
---

Measured 2026-10-09/10 (WP-V1, benchmarks/box_validity.cpp, record benchmarks/results/box_validity.md):
- Grading error (L2 of |F|^2, reflection hemisphere, Ag, vs uniform box): 400 nm cells 1.5-1.9 %,
  200 nm no better; 100 nm (lambda/5) 0.17 % at L = 1.5 um but 0.59 % at L = 1 um and 1.54 % at
  L = 1 um with depth x2 -> lambda/5 necessary, NOT shown sufficient for the 0.1 % depth check.
  Grading levels are 2^M h (h = 50 nm): nothing between 100 nm (M=1) and uniform (70 nm -> M=0).
- Depth x2 (Ag): uniform 1.10 %, 100 nm 1.76 %, 400 nm 3.03 % at L = 1 um; 400 nm 2.25 %, 100 nm
  3.49 % at L = 2 um -> grading changes the depth sensitivity non-monotonically.
- Depth sensitivity follows RIM illumination, not Rayleigh range: w0 = L/4 at L = 2 um has shorter
  z_R and an equally wide beam in the box but 3x lower sensitivity (rim intensity 18x lower). The
  old "z_R >= 8 x depth, L >~ 6-11 um" rule was underived and withdrawn (review W2).
- Lossy-Ag diagnostic (Im eps -1.5 / -6; SPP amplitude decay 8.9 / 3.1 um vs 41.5 um real Ag)
  halves the depth sensitivity: SPPs carry about half, one mechanism but not the only one.
- Rigorous-beam power through z = 0 outside the L x L patch: w0 = L/3: 0.89 % (w0 0.333 um),
  0.69 % (0.5 um), 0.62 % (0.667 um); paraxial 0.539 %; L/4: 0.019 %. Wider tails because the
  E_x spectrum is Gaussian x (1 - kx^2) with a k_t = k cutoff. A power fraction is not the L2 metric.
- Paraxial GaussianBeam at w0 = 0.67 um: reflectance -2.8 % vs the rigorous angular-spectrum beam.
- GMRES ICTF tol 1e-6: Ag needs 1 000-2 700 it (more with finer boxes), Si 400-1 250; tol 1e-7
  changes the far field by only 1.6e-5. Ag uniform 2N = 39 600: 40 min solve on the shared machine.
- Pitfalls: a background loop survives TaskStop (kill the bash parent and the exe by Windows PID,
  `ps -W` column 4); run study binaries from a private copy so the build tree can relink; the machine
  may sleep overnight (a run "hung" 8 h in assembly). Git Bash /proc/meminfo has no MemAvailable
  (use MemFree).

**Why:** sets budgets and the evidence base for ADR 0006 grading/waist rules; the review showed
single fixed-depth comparisons overstate conclusions.
**How to apply:** box/beam changes, MLFMM-size re-runs of the docs/05 rough-surface checks: always
pair graded-vs-uniform at both depths before claiming a grading suffices. See
[[rough-box-grading]], [[gmres-observations]], [[windows-msys2-build]].
