---
name: pml-wide-gap-findings
description: WP-PML6b wide-gap rule on rough Ag (gap 3 um, 2N 62 016) - cone theta<=30 window sensitivities per seed, 7d values, seed-2 margin, costs, flank-scaling choice, driver/run pitfalls
metadata:
  type: project
---

WP-PML6b (2026-10-25), `benchmarks/pml_huygens.cpp --study 6b`, support `pml_huygens::six_b_*`, record
`benchmarks/results/pml_huygens.md` section "WP-PML6b". Cases wide_gap3_case(seed) / wide_gap3_patch_case(seed),
seeds {1, 2}; seed 1 reproduces the WP-PML6 extended-plane diagnostic bitwise (886 / 904 its).

- Window study = flank x0.7 / x1.3 of (taper_end, taper_end + 4 lambda) at fixed inner edge, both sources. The 7g
  way (scale both half widths) does not fit: inner edge would enter the patch, outer the PML. L x1.2 x1.3 window
  ends 5.35 um < pml_start 5.375 (tight).
- Cone theta<=30 sensitivities: currents 4.0e-4 (seed 1) / 8.7e-4 (seed 2), Huygens 3.0 / 3.2e-4; 7d cone
  2.26 % / 1.58 % (ratio 56 / 18). Seed 2 gap currents 1.6x seed 1 -> currents sensitivity doubles: the 1e-3
  gate has little margin for stronger-SPP surfaces. Hemisphere: currents 0.17-0.59 % (not converged).
- Seed 2 far field differs 49-51 % from seed 1 (independent surface).
- Cost per 62 016 case on the shared machine: assembly 374-1250 s, solve 322-616 s, peak 4.5 GB, 13-33 min.
- Driver lesson: with --in, a process re-emitting the partner's comparisons duplicates rows; emit only for
  cases solved in the process. Keep a copy of the exe outside the build dir for long runs (Windows locks it).

**Why:** later rough-metal PML statistics inherit the gap >= 6 lambda / theta <= 30 rule and its margins.
**How to apply:** quote the cone numbers with seed; for new surfaces check the currents' window sensitivity
first (largest consumer of the margin). See [[pml-rough-ag-findings]], [[pml-spp-findings]].
