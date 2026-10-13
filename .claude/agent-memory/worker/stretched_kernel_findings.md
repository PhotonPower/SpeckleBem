---
name: stretched-kernel-findings
description: WP-PML2 stretched (PML) kernels - anisotropic ridge of 1/R~, ridge-split Sauter-Schwab rules, shared-edge base subdivision, accuracy/cost numbers, test-geometry pitfalls
metadata:
  type: project
---

WP-PML2 (2026-10-13), `src/kernels/stretched_operators.cpp`, tests `test_stretched_kernels.cpp`.

- **Anisotropic ridge**: with s_x != s_y (PML strips: sigma_y = 0) the kernel 1/R~ has a
  near-singular *direction* (|dy| ~ sigma |dx|), relative angular width
  eps = sqrt(lmin/lmax) of M = f_r f_r^T + f_i f_i^T (~1/(1+sigma^2), 0.059 at sigma 4). It hits
  every pair, not just touching ones. Plain tensor Gauss: touching 1e-3 at n = 20 (sigma 4);
  near pairs along the ridge 1e-4 with 100x100 points at D = 1.5 h.
- Fix for touching coplanar pairs: split the Sauter-Schwab parameter domains along the ridge
  planes (eigenvector direction, NOT d_y = Re(c) d_x: that line is off by O(eps) and stalls
  convergence) and grade geometrically (ratio 2, finest 0.5 eps, p points per panel).
- Plain WP7b shared-edge reference rule converges slowly on whole facets (1e-7 at n = 10 even
  isotropic); one midpoint subdivision of the cone bases gives 3e-12. Reused in tests as reference?
  -> the test_operators.cpp shared-edge reference has the same slow rate.
- Test geometry pitfall: sigma' jumps at x_p + W (profile end) -> grids crossing it converge
  algebraically (looked like a quadrature failure). Keep test grids inside [x_p, x_p + W].
- k-aware floor: Ag in the PML has |k s| h ~ 5-8 at h = lambda/10; p_eff = p + ceil(0.75 kh) - 4.
- Measured (p = 10): uniform stretch vs L(ks) 3e-11 (edge), 3e-9 (identical, reference-limited);
  strip sigma 1: 8e-9, 2.6: 3e-7, 3.4: 6e-7; corners ~1e-8; near/far target-reached <= 9e-8,
  pairs at the degree cap up to 1e-4 (deep strip, along the ridge).
- Rule tuning (finishing worker, same day): error is set by the plain (n) and radial (nr) points;
  the graded-panel points ng converge by 6 (ng 6 vs 10: same error to 3 digits, 35-50 % fewer
  points), ng 4 / kFine 1 hurt identical pairs (4e-8). Default now n = nr = p, ng = ceil(0.6 p).
  Cost is dominated by graded panels (n = 6 barely cuts shared-vertex points).
- Dense cost: 2888-triangle open surface (h 50 nm, PML 0.5 um, sigma 4, vac/Ag): 2.7 s plain vs
  ~420 s stretched (run beside another test), 9800 shared-vertex pairs at 6e5 points dominate;
  58k of 3.9M stretched near/far pairs hit the degree cap (1.5 %; the near/far unit test picks
  ridge-aligned pairs, there 150 of 300). Main open item for WP-PML5.
- Debug (ASan) costs: stretched touching pair ~0.1-1 s at -O0; keep debug cases to a few pairs,
  cheap options (stretched_touching_points = 4) for structural dense tests.

- Review fixes (2026-10-13): K of tilted x flat-PML near/far pairs is small (|K| eta/|L| ~ 0.01)
  and its relative error runs ~20x the L error; one far pair (D/h 4.1, degree 9) gave K 1.05e-5
  > target 1e-5 (test uses CHECK_NOFAIL + 1e-4 regression bound; heuristic change left to the
  coordinator). Capped pairs: Ag deep strip 4.9e-4 (36/50). Shared-vertex rule is symmetric
  under test/source exchange to 1e-14 (both-orderings check via reversed triangle numbering only
  probes shared edges, ~1e-12). Subdivided WP7b shared-edge reference: 5e-12 at n = 24.
  Hidden `[.]` scratch cases need a temporary add_test line to run through ctest.

- WP-PML2c (2026-10-13, idle 5900X-class 12 cores): [stretched_cost] 296.6 s -> 90.8 s; PML3a
  Si smoke h 250 nm 287.5 s -> 44.5 s (R_sim unchanged). Once-per-unordered-pair + transposes
  halves evaluations (7.0e10 -> 3.5e10), batching/fastmath/shared geometry give the rest. Graded
  points max(ceil(0.6p), ceil(p_eff/2)): Ag corners 2.6e-10 -> 6.3e-10, 9.8e-11 -> 9.5e-10,
  strips/vacuum unchanged (kept). Remaining: shared vertex 55 %, far 35 % (almost all Ag: 1.5e6 of
  3.9e6 far pairs at the degree cap, no decay relaxation for stretched pairs -> next lever).
  Stretched fastmath vs C library on the small OpenCase: 3e-15. Clang -Werror flags unused
  inline helpers in anonymous namespaces (GCC does not): build win-debug before reporting.
  Catch2 wraps WARN lines at 80 cols: grep -A3 when extracting sweep rows from ctest -V logs.

- WP-PML2c review fixes: debug (ASan) cost of the small OpenCase (72 triangles) is the stretched
  touching pairs: 1-thread dense ~7-11 s even with degrees 1/2/2 and an x-only strip, a
  single assemble_diagonal ~4-6 s; sigma_max 1 vs 2 barely changes it. Debug-time levers that
  worked: Z_ref rows restricted to a lower-y triangle band, per-region checks on a stride.
  test_concurrent_calls_are_serialised (pytest) can fail under heavy machine load (passes alone).

**Why:** these took most of the WP-PML2 iterations; WP-PML3/4/5 build on the stretched path.
**How to apply:** PML accuracy/cost questions, fast PML ring (WP-PML5), profile choice
(smaller sigma_max with wider W reduces the ridge). See [[open-surface-pml-findings]].
