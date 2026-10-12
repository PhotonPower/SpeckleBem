---
name: ag-absorption-findings
description: WP-A1 Ag absorbed power from currents at 45 deg (flat box) - local (n x M).J* low at L = 4 um only (spurious +-k sin(theta) standing wave, M phase +4-6 mrad); P_ext - P_sca correct; non-radiating near-null mode; WP-A2: intrinsic cavity mode of the box (see ag-null-mode-findings)
metadata:
  type: project
---

Measured 2026-10-11 (WP-A1, benchmarks/ag_absorption.cpp, record benchmarks/results/ag_absorption.md),
Ag 45 deg s, h = 50 nm, ICTF + Jacobi, MLFMM, L/w0 = 5 (needs --allow-wide-beam since WP-B2):
- L = 2 / 3 / 4 um: local A -0.1 % / -2.1 % / -43 % vs A_beam; A_ext = (P_ext - P_refl - P_fwd)/P_inc
  within +0.2 % in all three. Formula exact (degree 2 = 6), walls <= 1.3e-4.
- h = 50 nm limits the local product to ~1-3 % even without the mode (interpolant -2.7 %, L = 3 um -2.1 %).
- L = 4 um: refinement to true residual 1.07e-6 leaves A unchanged (not solver; at 0 deg it WAS solver).
  Error spectrum along x: equal raw |M_err| peaks at +0.70 k0 and -0.70 k0 (rect window, reference
  spectrum not printed; ~1 % inferred) -> spurious standing wave; absent at L = 3 um.
- Why A_ext and R miss it: they depend only on the field radiated into R1; a current that is the trace of
  a source-free vacuum field inside the box radiates nothing (exactly blind), while the local product's
  cross term with the Ag interior field is -1/2 Re[j w eps0 (eps_r* - 1) int E*.dE dV] (|eps' - 1| ~ 11).
  Leading hypothesis: (nearly) non-radiating near-null mode of the discrete PMCHWT-family system.
- Row/strip checks (exact-row residual 2.9e-5, MLFMM error 4.4e-5 spectrally flat) show a near-null mode
  exists; they do NOT decide intrinsic vs MLFMM-error excitation (gain needed only a few hundred).
  Origin open; d0 = 5 at L = 4 um ~21 GB (17.6 GB setup + 3.1 GB Krylov). Follow-ups E1-E5 in the record.
- A_ext subtracts terms ~2 P_inc (~160x A): far-field quadrature must be good to ~1e-5 for 0.1 % in A.
- Profile bins of 250 nm cannot resolve the +-k beat (period lambda/(2 sin theta) = 354 nm): alias.
- RT0 interpolant of the analytic currents: 12-13 % pointwise error, but integrated A within -1..-3 %;
  strip-local A has +-20 % antisymmetric interpolation artefacts (do not read local A per strip).
- Tooling: --refine (x += GMRES(Z, b - Zx)), --save/--load-currents, --row-check / --strip-check via
  op::assemble_sparse (exact rows: 150 rows x all columns in ~1 s).

**Why:** Ag absorption from surface currents amplifies mrad current errors by ~60 (Re Z_s/|Z_s| = 1.6 %).
**How to apply:** use P_ext - P_sca for Ag absorption; check A_ext and the M-vs-J phase before blaming
discretisation; a component invisible to far-field observables points at non-radiating currents, not at
"no overlap with the beam"; GCC 16 -Warray-bounds false positive with make_shared<PlaneWave> in a TU that
also includes the AngularSpectrumBeam header -> [[gnu::noinline]] helper. See [[fresnel-flat-findings]].
