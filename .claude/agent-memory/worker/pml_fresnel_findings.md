---
name: pml-fresnel-findings
description: WP-PML3b validation 7a (flat Si/Ag + PML vs R_beam) - oblique dark band rule, Si oblique needs h <= lambda/6 (lambda/5 +5.9 %), PML/window/quadrature inert, LAPACK LU, costs, batch pitfalls
metadata:
  type: project
---

WP-PML3b (2026-10-13/14), benchmarks/pml_fresnel.cpp, tests/support/pml_fresnel_support.hpp,
record benchmarks/results/pml_validation.md.

- **Dense LU**: `solver::solve_direct` (Eigen PartialPivLU + BLAS) 351 s at 2N = 11 440; LAPACK
  zgetrf (OpenBLAS, `extern "C"` under SPECKLEBEM_HAVE_BLAS_LAPACK) 8-65 s, 444-534 s at 46 112.
  One factorisation serves s and p (2nd RHS via op::assemble_rhs of a 2nd Simulation).
- **Oblique dark band** (default 1e-4): at 45 deg w0 = lambda fails at ANY patch (>= 2e-3): the
  rigorous beam's spectrum cut at k_t = k puts grazing waves exp(-(k w0 cos th)^2/4) along z = 0;
  plus defocus along the footprint. 7a uses w0 0.75, L 7 um (pml_start 4.5, 5.2e-5, ~2x margin),
  2N 46 112 at h 125 nm (Si assembly 2216 s, Ag 1054 s, peak 68 GB); smallest passes are L 6.5 um
  (8.9e-5, 2N 42 000) and w0 0.7 / L 7 (7.7e-5) - the review flagged calling L 7 "cheapest". R
  barely cares: dark 5.9e-3 vs 5e-5 -> <= 0.06 % at lambda/4; 5.9e-3 vs 2.6e-2 at lambda/6 ->
  0.05 % (p) / 0.08 % (s). Literal 7a geometry at lambda/6: 2N 104 016, 173 GB dense (infeasible).
- **Record keeping**: the --estimate-only DARK scan was printed to the terminal only; the review
  asked for the raw lines. Tee every scan into a log and append it to the rows file.
- Review W2: a not-mesh-converged pass must not be an acceptance CHECK (Si 45 deg lambda/4 is now
  WARN-only in the validation-large test; acceptance is the lambda/6 case).
- **Si oblique mesh**: 45 deg Si lambda/4 -0.70 %, lambda/5 **+5.9 %** (p), lambda/6 +0.18 %; power
  balance 1-R-A_real (A_real = (n x M).J* inside pml_start) 6 % -> 0.2 % tracks it. Si 0 deg and Ag
  converge monotonically (lambda/4 fine). h ~ lambda/n_Si (116 nm) suspected.
- PML inert for flat R: W x1.5, sigma x1.5, quadrature (acc 1e-6, stp 12) all <= 2e-7; gap x1.5
  ~1e-5. A_real useless for Ag (negative at 45 deg lambda/4).
- **Batch pitfall**: TaskStop on a background `bash script.sh` leaves the script (and its wait
  loops) running; editing a running script makes bash read new lines at the old offset -> duplicate
  concurrent runs. Never overwrite a script that a live bash still reads; check `ps -ef`.

**Why:** WP-PML4 (7b-7e) and rough-Ag PML runs reuse these sizes, rules and pitfalls.
**How to apply:** scan dark_band_ratios() (--estimate-only) before assembling; use h <= lambda/6
for Si off-normal; budget 2N ~ 4.6e4 per 45 deg case at the strict dark band. See
[[pml-route-findings]], [[stretched-kernel-findings]].
