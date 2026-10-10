---
name: fresnel-flat-findings
description: WP22c flat-box vs Fresnel measurements (Ag/Si, 0/45 deg, MLFMM d0=3) - errors vs R_beam, Ag needs GMRES tol 1e-5, waist L/5 at 45 deg, GMRES counts, hemisphere quadrature, study pitfalls
metadata:
  type: project
---

Measured 2026-10-10 (WP22c, benchmarks/fresnel_flat.cpp, record benchmarks/results/fresnel_flat.md),
MLFMM d0 = 3, h = 50 nm, lambda_1/5 box, Si fine band:
- All 8 docs/05 cases within 1 % of R_beam at GMRES tol 1e-4: Ag L = 4 um 0 deg -0.52 % (w0 = 1 um),
  45 deg p/s -0.07/-0.09 % (w0 = 0.8); Si L = 6 um 0 deg +0.11/+0.10 % (w0 = 1.5), 45 deg -0.14/-0.06 %
  (w0 = 1.2).
- **Ag at tol 1e-4 is solver-limited**: Ag 0 deg -0.52 % -> -0.011 % at tol 1e-5 (1 049 -> 1 657 it),
  spurious absorption (A 0.032 vs Fresnel 0.018, energy balance -0.8 %) vanishes. Before that test the
  error looked like discretisation (grew with w0 and with h = 40 nm: -0.78 %; unchanged by d0 = 5 and
  dense) - check GMRES tol and the energy balance R + A = 1 first. Si closed to 1e-3 at 1e-4.
- Reference: R_beam vs plane-wave |r|^2 ~1e-6 for Ag 0 deg, Si 45 deg p -0.45 % at w0 = 1.2 um (1/w0^2).
  Edge loss at 45 deg with w0 = L/4 is 0.6 % -> use L/5 (0.06-0.1 %).
- GMRES (ICTF, Ag + Jacobi, tol 1e-4): Ag 0 deg ~1 050-1 120 it, 45 deg 1 630-1 650; Si 118-208.
  Si L = 6 um 2N = 339 120 peak 37 GB (6-level octree, cheaper than L = 4 um: 56 GB estimate).
- Midpoint-in-theta hemisphere rule was 5e-4 off; Gauss-Legendre in theta exact to 1e-14 at 1 deg.
- Pitfalls: a bash command with several cases hits the 2 h background limit and kills the running
  case (one case per command); runs need the MSYS2 DLLs next to a private exe copy; the scratchpad
  directory is shared with other sessions (use unique log names).

**Why:** baseline for Phase 5 rough-surface runs (Ag tolerance) and any re-run of the flat limit.
**How to apply:** use GMRES tol 1e-5 for Ag quantitative reflectance; w0 <= L/5 at oblique incidence.
See [[box-validity-findings]], [[angular-spectrum-beam-findings]], [[gmres-observations]].
