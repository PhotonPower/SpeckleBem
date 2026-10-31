---
name: schur-preconditioner-findings
description: WP-SC2 near-field Schur complement (ADR 0012 d2) on rough Ag PML - iterations equal to parent blocks; why (K near coupling ~0 on near-planar surfaces); implementation pitfalls
metadata:
  type: project
---

WP-SC2 (2026-10-31): `solver::SchurComplementPreconditioner` (Ã block LU over J box blocks, S̃ =
D - C Ã⁻¹ B on D's pattern, only its M diagonal blocks formed, B/C borrowed from the operator's
near matrices). Rough Ag PML U: **184 iterations = parent's 184** (early gate <= 92 failed);
tiny PML 46 = 46; Ag sphere box route 261 vs parent 266. Memory half of parent (66 vs 131 MB at U).

**Why:** on (nearly) planar surfaces the K-operator near entries (B, C = J-M coupling) of coplanar
pairs vanish (f_m . ((r-r') x f_n) = 0), so exact inter-box B/C adds nothing; the WP-SC1 nearlu
gain lives in the inter-box A/D (T-operator) coupling. Measured (driver diagnostics, Ag U):
nearlu_bc (A, D box-diagonal, B, C full; the ceiling of the Schur family) 184 = parent; nearlu_ad
(A, D full, B, C box-diagonal) 21 vs nearlu 20; Si 48 / 14 / 13. ||B|| ||C|| / (||A|| ||D||) = 6e-4
(scale-free J-M coupling). Stage 2 (16/24 um) not run.

**How to apply:** a near-field preconditioner for rough metal PML systems must capture inter-box
A/D coupling (e.g. overlapping blocks, ILU/SPAI of Z_n), not the J-M Schur split.

Pitfalls found:
- Eigen PartialPivLU::rcond() with an exactly zero pivot can return a finite value (division by
  zero inside the estimator): check the U diagonal for exact zeros. BlockDiagonalPreconditioner
  has the same latent gap (not fixed in WP-SC2).
- Bitwise equality across source splits needs canonical (sorted) row/column order of the dense
  pieces fed to GEMM: OpenBLAS rounding depends on the row position in the panel.
- GCC 16 -Warray-bounds false positive: shared_ptr<PlaneWave> destructor in a TU that also
  includes angular_spectrum_beam.hpp; use a stack PlaneWave.
- Invariance test: M⁻¹Z equal to 1.7e-15, residual histories equal to 1e-15 up to ~30 iterations,
  then Krylov rounding grows ~10x per 3-4 iterations: compare only the first 40.
