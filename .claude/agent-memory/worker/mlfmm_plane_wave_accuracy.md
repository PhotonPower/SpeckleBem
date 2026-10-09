---
name: mlfmm-plane-wave-accuracy
description: Measured accuracy of the ADR 0008 truncation formula, corner worst case, Lagrange interpolation order, Hankel recurrence (WP18), plus Eigen/GCC pitfalls met there
metadata:
  type: project
---

Measured in WP18 (2026-10-09, tests/unit/test_plane_wave.cpp, real k = 2pi, box edge a in lambda):

- ADR formula L = kD + 1.8 d0^{2/3}(kD)^{1/3}, D = sqrt(3) a, random points in boxes 2a apart:
  <= 10^-d0 for a >= lambda/2 (d0 = 3, 5); at lambda/4 up to 1.6e-3 (d0=3) / 2.4e-4 (d0=5)
  depending on the seed. Si-like k (n = 4.3 - 0.07j) passes from lambda0/4 (kD large).
- Corner-to-corner worst case (d/X = 0.87, offset (2,0,0)) never reaches 10^-d0 at the ADR L:
  8.6e-2 (lambda/4) ... 1.2e-3 (4 lambda) for d0 = 3. Min over L (low-frequency breakdown):
  ~1e-2 at lambda/4, ~1e-3 at lambda, 2e-4 at 2 lambda, 1e-5 at 4 lambda. An ADR §6 policy
  based on this corner check rejects almost every level; WP21 needs a different criterion.
- Local Lagrange interpolation child (L_c) -> parent of exp(-jk khat.d), |d| = D_c/2: p = 6 gives
  only 1e-3..1e-2; 0.1 x 10^-d0 needs p ~ 12-14 (d0 = 3), ~ 20-22 (d0 = 5). Error grows with box
  size (ADR sampling tends to 4 points per pattern wavelength).
- h_l^(2) upward recurrence: <= 1.4e-14 relative for l <= 120, |z| in [0.05, 80]. The Bessel-
  polynomial closed form in long double is a usable reference only where its cancellation
  sum|t|/|sum| <= 1e6 (fails for |z| >~ 25 at large l); add a long double recurrence check.
- theta/phi components flip sign across the pole reflection (odd parity) - interpolation must know.

Pitfalls: Eigen `a.dot(b)` conjugates `a` (use `cwiseProduct(b).sum()` for bilinear sums).
GCC 16 -Wnull-dereference false positive also on `std::vector<Real> p(n); p[0] = 1.0;` and on a
lambda returning `(M * u).cast<Complex>().cwiseProduct(w)`: build with push_back / in-place loops.
win-debug: Eigen expression loops over ~1e4 directions are ~30x slower; plain pointer loops and
release-only large-box sweeps kept unit cases < 1 s.

**Why:** these numbers decide WP21 parameters (truncation, p, lossy-region policy).
**How to apply:** check before choosing MLFMM defaults or test tolerances; re-measure if
truncation_order or the sampling changes. See [[gmres-observations]].
