---
name: pml-rough-ag-findings
description: WP-PML4b validations 7c/7d on rough Ag (PML route) - 7c met (1e-5..1e-4), 7d Si 0.24 % pass / Ag 2.6 % (SPPs: gap J 10 %), l2 ~ 0.3 x gap level, flat/lossy controls, W = lambda, dark band, ctest costs, foreign-batch pitfall
metadata:
  type: project
---

WP-PML4b (2026-10-15, finished 2026-10-18), benchmarks/pml_rough_ag.cpp, tests/support/pml_rough_ag_support.hpp,
record benchmarks/results/pml_rough_ag.md. Case U: rough Ag sigma 50 nm, Lc 500 nm, seed 1, L 3.75 um
(map generated at 4.5 um, cropped), h lambda/4, w0 = lambda, taper/gap 0.5 um, W_pml 2 lambda, sigma 4.
Formulation is the automatic `formulation::recommend` = ICTF for Ag AND Si (not PMCHWT).

- 2N 22 816 (8.3 GB), assembly 1 100-2 000 s depending on load, LU 90-390 s; L x1.2: 27 472, 12.1 GB.
- **7c met**: sigma x1.5 4.6e-5, W x1.5 2.5e-5, gap 0.875 um 4.4e-5 (L2 metric); quadrature acc 1e-6/stp 12
  3e-8 (inert); W_pml = lambda with sigma 2: 9.5e-5 (2N 17 280, 4.8 GB) but edge decay 4e-3 > 1e-3.
- **7d**: rough Si control 0.241 % (pass, gap J 0.0076); Ag 2.57 % (l2_off30 6.8 %), dR +1.8e-4.
  Flat Ag 2e-7. Same solve, window moved outward 1.6 %; l12 vs ugap at the same window 2.4 %.
  Window effect ~ 1/W (0.5 um window shifted 1.61 %, 1 um window shifted 0.89 %; widths 0.5->1->1.75 um
  0.93 % / 0.47 %); R(window width 0.5/0.875/1/1.75 um) = 0.97834/0.97806/0.97798/0.97754.
- **Wording rule (review 2026-10-18, ADR 0010)**: do NOT call the Ag L dependence "physics of the patch";
  the 2.42 % annulus comparison still contains the window acting on changed SPP currents (not separated).
  Write: termination not the cause; window truncates SPP currents; WP-PML6 (row 7g, Huygens plane)
  decides whether a physical L dependence remains. The 7d Ag test has a regression guard (Approx 10 %).
  Lossy (Im eps -6) 1.09 %, gap J 0.036, dR -7.6e-5 (sign flips, unexplained).
- **Scaling observation**: l2(L x1.2) / gap-J level of U = 0.26 (Ag), 0.31 (lossy), 0.32 (Si): the L
  sensitivity tracks the surface current that leaves the patch. Useful to predict 7d from one solve.
- Loss diagnostic: Im eps -6 (x19): SPP amplitude decay 3.1 um, intensity 1.56 um (< L/2); x10 gives
  2.3 um intensity (not enough); amplitude decay saturates ~2.9 um for larger loss. R 0.740.
- Dark band: binds only for w0 >~ 1 um at gap 0.5 um (ratio 3.4e-4 at 1 um); relaxed 8.3e-3 vs strict
  5.2e-5 at w0 1.3 um: far field 5.2e-4. Default 1e-4 recommended kept.
- validation-large "PML 7[cd]" (3 cases) = 16 181 s on the shared machine (7c 7 888 s: u + w15 + s15).
- Rough DRC at w0 = lambda: peak 1.89 vs flat 6.49 /sr, asymmetric (+20: 0.97, -20: 0.47); R 0.97834 vs 0.98163.
- **Pitfall**: previous workers' background batches kept running in the shared scratchpad after their
  "turn limit" (quad, wl, uwide, drel runs appeared while one worked; a foreign ctest log was touched days
  later). Check the process list for foreign specklebem/rough_ag executables before planning runs, and
  reuse their rows instead of duplicating solves.
- Extra far-field windows must end at or before pml_start strictly; run() checks this before assembly
  since 3e204ea (a round-off overshoot once cost a 1 668 s assembly).
- Catch2 WARN output wraps at 80 columns: re-join by dropping the 2-space indent, no space after a line
  ending in '.', '-', '/', one space otherwise (verified against the benchmark's unwrapped ROW lines).

**Why:** later rough-Ag PML work (statistics, MLFMM/ACA PML) inherits this L-dependence and these costs.
**How to apply:** state L, gap and window with every rough-Ag result; validate terminations on Si or
flat Ag, not rough Ag. See [[pml-spp-findings]], [[ag-box-mechanism-findings]], [[pml-fresnel-findings]],
[[pml-si-crosscheck-findings]].
