---
name: formulation-sweep-findings
description: WP22b2 MLFMM formulation sweep (PMCHWT/ICTF/MCTF/Jacobi, L = 3.2-10 um, uniform box) - iteration counts, Ag PMCHWT staircase, graded vs uniform box (C2 review runs), Jacobi true residual, Si waist pin, run costs
metadata:
  type: project
---

Measured 2026-10-12/13 (WP22b2, record benchmarks/results/formulation_convergence.md section
"WP22b2"; raw logs benchmarks/results/wp22b2_logs/), MLFMM d0 = 3, rigorous beam w0 = L/4,
tol 1e-3, uniform 50 nm box (box_mesh_size = h):
- Si iterations: ICTF 75/70/65/68, MCTF 74/70/67/70, PMCHWT 369/369/374/405 (plateau ~2e-3 from
  it 100 to 350, size-independent), PMCHWT+Jacobi 333/333/256/280 at L = 3.2/5/7/10.
- Ag: ICTF 347/419/490/643, Jacobi 346/423/470/597, MCTF 505/737/1161/2039, PMCHWT 2127/3423/
  >3300 (1.5e-3)/>2200 (2.3e-3). Staircase plateaus (<3 % drop per 100 its) ~400 (3.2 um) to
  ~930 its (10 um). Capped runs do NOT show non-convergence (caps < 5 um count; still falling).
- Review C2 (graded 100 nm box, Ag 3.2 um): ICTF 571, MCTF 783, PMCHWT 3461 (also staircase),
  Jacobi 960. Ratios MCTF/ICTF 1.37 and PMCHWT/ICTF 6.06 ~ uniform 1.46/6.13, far from WP15 dense
  1.05/1.42 -> WP15 ratios are not a graded-vs-uniform box effect; open: dense vs MLFMM, paraxial
  L/3 beam, 400 nm cells. Hypothesis: d0 = 3 MLFMM error causes the PMCHWT plateaus (dense WP15
  had none); a d0 = 5 run at 3.2 um would decide. Jacobi deficit IS box-related (1.68x graded).
- Review W1: left Jacobi stops on the preconditioned residual. Ag ICTF+Jacobi true ICTF-system
  residual 1.07e-3 (3.2 um), 1.08e-3 (10 um, 597 its); plain ICTF reaches those at 336 / 617 its
  -> Jacobi advantage at 10 um is 3 % at equal true residual (7 % nominal).
- Jacobi identity: Ag 5e-12 (3.2 um), 4.7e-9 (10 um ICTF), Si 1e-8 (dense was 5e-13).
- Costs: Ag uniform L = 10 um 2N = 672k: ICTF 884 s, MCTF 2039 its 4533 s; Ag PMCHWT 3300 its at
  L = 7 um 5928 s; Si L = 10 um (1.02e6) PMCHWT 405 its 5336 s, 80 GB peak; ICTF 888 s. Ag 3.2 um
  graded PMCHWT 3461 its 1167 s (2N = 48k).
- Since WP-B3 (ADR 0006 amendment 2026-10-12b) the uniform box is the library default, but the
  scaling support still pins BoxGradingMode::Graded with default 100 nm: pass
  --box-mesh-size 50e-9 explicitly. Si default waist is now L/6: reproducing the Si L/4 runs needs
  --waist-factor 4 --allow-wide-beam (the old batch files lack it). See [[server-kit-waist]].
- Driver: after each run the 1-min load needs ~1-10 min to fall below 5 (foreign load). Editing
  a bash driver script while it runs breaks its tail. A stopped 2 h command leaves the exe
  running: next driver must wait for it.

**Why:** settles/qualifies the WP15 deviations; informs formulation::recommend for graded boxes.
**How to apply:** use the uniform box for Ag MLFMM runs; cap Ag PMCHWT runs; budget
t(k) ~ a k + c k^2 (c ~ 1 ns x 2N); compare Jacobi vs unpreconditioned at equal TRUE residual.
See [[mlfmm-scaling-findings]], [[gmres-observations]].
