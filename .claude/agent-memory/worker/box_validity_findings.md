---
name: box-validity-findings
description: WP-V1 measured facts on the rough closing box under beam illumination (Ag/Si, dense) - grading error, depth/edge sensitivity mechanisms, beam model, GMRES cost, study pitfalls
metadata:
  type: project
---

Measured 2026-10-09/10 (WP-V1, benchmarks/box_validity.cpp, record benchmarks/results/box_validity.md):
- 400 nm coarse box cells (lambda/1.25 in R1) = 1.5-1.9 % far-field error (L2 of |F|^2, reflection
  hemisphere) vs the uniform box for Ag; 200 nm no better; 100 nm (lambda/5) 0.17 %. Coarse bottom
  and wall-bottom edges radiate (per-part far-field share 10x / 200x the uniform box).
- At dense sizes (L <= 2.4 um, w0 <= 0.67 um, z_R <= 2.8 um) depth x2 (Ag 1.1 % even uniform) and
  L x1.2 (4-6 %) are dominated by the beam reaching the walls and w0 = L/3 losing 0.5 % of power
  past the edges -> docs/05 checks need L >~ 6-11 um. Si with fine band depth x2: 0.13 % (eps_rr 0.04 %).
- Lossy-Ag diagnostic (Im eps x19) halves the sensitivities only: SPP circulation not dominant at L <= 2.4 um.
- Paraxial GaussianBeam at w0 = 0.67 um: reflectance -2.8 % vs a rigorous angular-spectrum beam.
- GMRES ICTF tol 1e-6: Ag needs 1 000-2 700 it (more with finer boxes), Si 400-1 250; tol 1e-7
  changes the far field by only 1.6e-5. Ag uniform 2N = 39 600: 40 min solve on the shared machine.
- Pitfalls: a background loop survives TaskStop (kill the bash parent and the exe by Windows PID,
  `ps -W` column 4); run study binaries from a private copy so the build tree can relink; the machine
  may sleep overnight (a run "hung" 8 h in assembly).

**Why:** sets budgets and the evidence base for ADR 0006 grading/waist rules.
**How to apply:** box/beam changes, MLFMM-size re-runs of the docs/05 rough-surface checks. See
[[rough-box-grading]], [[gmres-observations]], [[windows-msys2-build]].
