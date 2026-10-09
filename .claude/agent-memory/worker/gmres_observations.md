---
name: gmres-observations
description: Measured GMRES iteration counts / residuals on SpeckleBem systems (WP13) for choosing test tolerances and budgets
metadata:
  type: project
---

Measured 2026-10-09 with `solver::gmres` (WP13, full GMRES, win-release):
- Sphere r = 0.5 um, icosphere n = 2, 500 nm, tol 1e-8: Si ICTF unpreconditioned 93 it,
  |x - x_LU|/|x_LU| 1.1e-7; Ag PMCHWT + left Jacobi 193 it, true residual 1.7e-7.
- Icosphere n = 1 at 1 um: Si ICTF 53 it (err 1.1e-6), Ag PMCHWT+Jacobi 46 it (err 1.1e-7).
- Icosphere n = 3 (2N = 3840), 500 nm, lossless n = 1.5, ICTF unpreconditioned, tol 1e-8
  (WP14a Simulation driver): 201 it, 2.2 s; eps_rr differs from the LU value by ~1e-8
  (LU 2.7 s, dense assembly 0.6 s, win-release).
- Si rough-surface box (L 1.6 um, h 50 nm, depth 0.5 um, uniform box, 2N = 19 968), ICTF
  unpreconditioned, tol 1e-6: 330 it in 87 s (~0.26 s/it, dense gemv memory-bound), true
  residual 9.7e-7, err vs LU 5.3e-6; LU 85 s, rcond 8.9e-8; assembly 10 s; peak RSS 12.8 GB.
- Left Jacobi is exactly row-scaling invariant (histories agree to rounding); unpreconditioned
  GMRES on a row-scaled (1e-3..1e3) system stalls (21 it vs 1e-2 after 63 it).
- Arnoldi residual estimate equals the explicit true residual to ~1e-12 relative for
  well-conditioned systems. The *absolute* gap is ~1e-17..1e-16 (right Jacobi GMRES(5) on the
  1e-3..1e3 row-scaled n = 200 system), so a "monitored == true to 1e-10 relative" check only
  holds for residuals >~ 1e-6; at tol 1e-12 the gap was 1e-5 relative.

- WP15 rough box (L 3.2 um, h 50 nm, sigma 50 / Lc 500 nm, seed 1, graded box, no fine band,
  Gaussian beam w0 = L/3, tol 1e-3, full GMRES): Si 2N = 31 200 -> ICTF 219, MCTF 254, PMCHWT
  315, left Jacobi 272 it; Ag 2N = 29 280 -> ICTF 622, MCTF 651, PMCHWT 886, Jacobi 1069 it.
  ~0.5-0.7 s/it (shared machine). Left-Jacobi histories of the 3 formulations agree to 5e-13.
  Left-Jacobi true residual at stop is 2-30x the monitored 1e-3 (PMCHWT rows worst).
- Dense assembly of those rough boxes took 1050-1540 s each (WP7b defaults, shared machine),
  ~5x beyond N^2 scaling of a 2N = 4320 smoke case (4-8 s): budget ~25 min per 3e4 system.

**Why:** gives realistic budgets/tolerances for later solver, MLFMM and formulation-study WPs.
**How to apply:** use as baseline when setting max_iter or tolerances; re-measure if the
assembler quadrature changes (WP7b). See [[build-pitfalls-windows]].
