---
name: mlfmm-local-leaf-findings
description: WP21L local leaf rule measurements - elevated ratio rho/2 needed, d0=5 lambda/2-leaf limitation, quantile 0.99 inert for Si boxes, test costs
metadata:
  type: project
---

Measured 2026-10-11 (WP21L, branch wp/21l-local-leaf-rule):

- **Elevated ratio must be rho/2** (kElevatedSupportRatioFactor): with the ADR wording (leaf ratio
  rho on levels above the leaf) graded box 1.5 x 2 um Si, quantile 0.5, d0 = 3 gave matvec error
  5.0e-2; rho/2 gave 1.5e-4. Mechanism: the level order grows beyond k r_min and translators amplify
  the interpolation error of the children's fields; exact leaf patterns are immune.
  Consequence: no function ever lives on level D-1 (rho_e a_{D-1} = rho a_D); elevation is >= 2
  levels, so trees need >= 5 levels for elevated functions with home level >= 2.
- **d0 = 5 with lambda/2 leaves fails on far-heavy geometries**: graded box 1 x 1 x 4 um (far part
  ~8 % of |Zx|): local rule 5.6e-5 (Si) / 6.4e-5 (Ag), all from leaf-leaf vacuum interactions
  (elevated part 7e-7 / 1e-6); the global rule passes (4.5e-6) only because r_max forces lambda
  leaves. WP19b block error at lambda/2 is ~4e-4, so the matvec criterion holds only when far/total
  is small. Open coordinator decision (a_min(5) or binding lossless block check).
- **Default quantile 0.99 elevates nothing in the WP22b1 Si boxes**: coarse 100 nm cells are 7.7 %
  (L = 4 um) / 10.5 % (L = 8 um) of the functions; q <= 0.9 / 0.8 needed. Estimates: Si L = 4 um
  peak 55 -> 35 GB, L = 8 um 84 -> 60 GB (near 14.3 -> 6.4 / 16.2 -> 7.5 GB).
- On small tall boxes (box functions > 50 %) the local rule costs MORE memory (near 0.24 -> 0.99 GB).
- Debug (ASan) costs: n^2 partition loop over 1600 bases ~36 s; MlfmmOperator on a 2N = 4320
  graded box with n = 1.5 ~2 min; keep Si interiors out of unit tests (L ~ 50 patterns).

Related: [[mlfmm-lossy-policy-findings]], [[mlfmm-scaling-findings]].
