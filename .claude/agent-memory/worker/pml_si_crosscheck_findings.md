---
name: pml-si-crosscheck-findings
description: WP-PML4c validation 7e (rough Si, PML route vs uniform box, L 4.5 um, w0 L/6, h lambda/4) - result, tolerance rule, shared-map design (fine generation + subsampling), grid constraints of L x 1.2, costs
metadata:
  type: project
---

WP-PML4c (2026-10-15), benchmarks/pml_si_crosscheck.cpp, tests/support/pml_si_crosscheck_support.hpp,
record benchmarks/results/pml_si_crosscheck.md.

- **Result (h = lambda/4, seed 1, sigma 50 nm, Lc 500 nm, L 4.5 um, w0 0.75 um):** PML route vs uniform
  box l2_refl 0.146 %, eps_rr 0.026 / 0.041 %, dR -1.6e-5; the box's own L x 1.22 / x 1.17 change at
  the same h is 0.155 / 0.161 % (record at 50 nm: 0.15-0.23 %). So the two terminations differ by
  about as much as the box differs from itself when its rim moves: the residual is the edge effect.
- **Tolerance rule (fixed before the PML solve):** per quantity, the max of the box's L x 1.2 changes
  (WP-V2/V2b records, seeds 1-2, L/5-L/6; plus this WP's lambda/4 partners): l2 0.229 %, eps 0.058 %,
  dR 4.1e-5. Run the box partners first, then the PML route.
- **Same surface at several h:** generate once at lambda/60 on 6.5 um and subsample (factor 15 / 10 /
  6 for lambda/4 / lambda/6 / lambda/10); generating at h directly gives a different realisation per h.
  L/2 and 1.2 L/2 must both be grid lines: at 125 nm with L = 4.5 um, 5.4 um is impossible (5.25 /
  5.5 um used); L = 5 um would give exact x1.2 at lambda/4 and lambda/6.
- **Costs (shared machine):** PML 2N 27 472 dense assembly 2138 s (Si PML, loaded machine), LAPACK LU
  368 s, peak 24.3 GB; box lambda/4 2N 55 296 MLFMM 248 s + 92 it 441 s, 12.7 GB (patterns 8.8 GB:
  global leaf 360 nm at h 125 nm). PML lambda/6 at L 4.5 um = 2N 62 016 = 61.5 GB dense: infeasible
  within 50 GB.
- Box uniform at h = 125 nm exceeds the lambda_1/5 cell rule (validated box is 50 nm): state it.
- **lambda/4 is NOT mesh-converged for rough Si far fields** (sigma 50 nm, Lc 500 nm): lambda/6 moves l2 by
  2.0-2.3 %, off-specular 4-5 %, R +7.5e-4..1.5e-3, identically on both routes (PML 2.02 %, box 2.03 % at
  L 3.5 um). Box lambda/10 vs lambda/6: 1.23 % more (2nd-order fit: ~4 / 2 / 0.7 % left at lambda/4 / 6 / 10,
  0.2 % needs ~lambda/20); R not monotone in h. The inter-route difference is mesh-independent. Box lambda/10
  L 4.5 um: 2N 341 280, 203 it, 28 GB real (35 est), ~78 min on a loaded machine.
- 7e ctest (validation-large) passed twice: 69 min / 100 min on the shared machine (PML dense assembly 48-73
  min dominates); solver::DenseLu residual 4.5e-16 vs 2.7e-13 of the old support LU, same far field digits.
- **lambda/6 cross-check at L 3.5 um, w0 L/6 (2026-10-15):** PML vs box 0.186 % (box L partners 4.0 / 4.5 um:
  0.271 / 0.189 %); lambda/4 0.210 % (0.260 / 0.213 %). PML-box difference grows as L shrinks, like the box's
  own edge sensitivity. PML lambda/6 dense 2N 48 240: 37.2 GB, assembly 6376 s (loaded machine), full GMRES
  tol 1e-8 233 it 598 s, unpreconditioned.
- **R accuracy wording (review W3):** R moves 0.75-1.5e-3 lambda/4 -> lambda/6 (L 4.5 / 3.5 um, both routes);
  only "lambda/4 and lambda/6 within 4e-4 of lambda/10 at L 4.5 um" holds. Do not write "R within 4e-4" bare.
- **eps_rr pass rests on the cross-seed max (review W4):** PML yz 0.0413 % vs same-surface partner max 0.0410 %;
  only the WP-V2b seed-2 value 0.058 % makes it pass. kTolerances were committed (11:19) before the partner runs
  (11:23-11:52) and the PML run (11:53): state such timelines from log timestamps + commit time.
- L 3.5 um (w0 0.583) is the largest L on both lambda/4 and lambda/6 grids with PML lambda/6 <= 50 GB.
- Identity check is now also by oriented vertex triples (patch_triangle_identity); the TriangleMesh constructor
  re-orients a single flipped triangle, so test orientation sensitivity with a fully reversed mesh.
  The unit case meshing the three record configurations (781^2 map at lambda/60) costs 0.6 s release but
  74 s win-debug (ASan, ctest -j 8).
- After merging WP-LU1: test support must use solver::DenseLu(Z, LuBackend::Auto/Eigen), backend() ==
  LuBackend::Lapack; pml_fresnel::DenseLu no longer exists (silent build break on cross-branch merges).

**Why:** 7e closes the PML-route validation for Si; reruns or Ag/Si comparisons reuse the design.
**How to apply:** for any cross-route comparison share one fine-generated map; pick L so that the
L x 1.2 partner is on the grid; fix tolerances from reference-only runs first. See
[[pml-fresnel-findings]], [[box-validity-mlfmm-findings]].
