---
name: mlfmm-single-level-findings
description: WP19b measured single-level FMM (RWG patterns) block errors vs dense, pattern quadrature calibration, far-field factors, test-cost lessons
metadata:
  type: project
---

Measured in WP19b (2026-10-09, tests/unit/test_patterns.cpp, slow sweep `[patterns_sweep]`):

- Far-field factors verified against element_blocks: L = (w mu k / 16 pi^2) sum w T R.V,
  K = (k^2 / 16 pi^2) sum w T R.W, W = khat x V (W_theta = -V_phi, W_phi = V_theta). The extra
  1/4pi comes from G = e^{-jkR}/(4 pi R) on top of the addition-theorem -jk/4pi. Flipped K sign
  gives relative error exactly 2.
- Single-level block error (8 pairs, nearest + farthest partners) at the search order with the
  enlarged diagonal depends only on a/lambda_medium and rmax/a: a = lambda/4: 1.3e-2 (d0 = 3);
  lambda/2: 1.3e-3 (d0 = 3), 3.6e-4 (d0 = 5); 0.75 lambda: 1.7e-4 / 6.7e-5; lambda: 4.6e-5 /
  1.5e-5; 1.5 lambda: 3e-6 / 1.1e-6. The statistical (volume-random) order search reports
  "achievable" in all these cases: it underestimates nearest-offset RWG block errors.
- Flat rough box (surface fills the facing box faces): at lambda/2 the nearest pair has a
  minimum over L of 1.3e-4 (L = 22, then breakdown), independent of mesh 62.5/31.25 nm.
- Pattern quadrature envelope E_d = 3 (|k|h/4)^{d+1}/(d+1)! (calibrated, conservative 1-2 digits
  at |k|h >= 2); target 0.01 x 10^-d0 suffices.
- Cost: the pattern loop with Eigen dot + std::exp(Complex) was ~10x slower in win-debug than
  plain double loops with cos/sin (rough box test 11 s -> 1.6 s). Dense oracle: set
  target_accuracy = 0.01 x 10^-d0, not 1e-9.

**Why:** WP20/21 must pick truncation orders / leaf sizes; the search alone is not enough.
**How to apply:** for MLFMM defaults and test tolerances, see also [[mlfmm-plane-wave-accuracy]].
