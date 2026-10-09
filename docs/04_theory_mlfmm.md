# Multilevel fast multipole method

Reference implementation plan for `compression/mlfmm/`, following Song–Lu–Chew (1997), Sheng et al. (1998) for the two-region dielectric case, and Gumerov et al. (2003) for the data structures (as in the paper).

## Idea

Split `Z = Z_near + Z_far`. Near interactions (boxes adjacent at the leaf level) are integrated exactly and stored sparse. Far interactions are evaluated through the plane-wave (diagonal) form of the addition theorem:
```
e^{−jk|r_m − r_n|} / |r_m − r_n|  ≈  (−jk/4π) ∮ e^{−jk·(r_m − r_M)}  T_L(k, r_MN, k̂)  e^{−jk·(r_N − r_n)} d²k̂
```
with the translation operator
```
T_L(k, r, k̂) = Σ_{l=0}^{L} (−j)^l (2l+1) h_l^{(2)}(kr) P_l(k̂ · r̂)
```
(`h_l^{(2)}` for `exp(+jωt)`). The three factors become **aggregation** (sources → box centre), **translation** (box M → box N), **disaggregation** (box centre → observers).

## Data structures

- **Octree** over RWG edge midpoints, root = bounding box of the mesh, Morton ordering so that the elements of a box are contiguous. Leaf criterion: ≤ `max_elements_per_leaf` (≈100) **and** box size ≥ `min_box_size_lambda · λ` (λ/4) to avoid low-frequency breakdown. Paper: 120 elements per smallest box, 5 levels for the 4 µm sphere.
- **Near list**: the 3×3×3 neighbourhood at the leaf level. **Interaction list**: children of the parent's neighbours that are not neighbours themselves (≤ 189 boxes).
- **Sampling** on the unit sphere per level: `L_m` Gauss–Legendre points in θ × `2L_m` uniform in φ → `2L_m²` directions (paper: `L_m = 21` for the 10×10 µm² surfaces).
- **Truncation** `L = kD + 1.8 d0^{2/3} (kD)^{1/3}` with `D` the box diagonal and `d0` the requested digits (`MlfmmParams::accuracy_digits`).

## Radiation and receiving patterns

With the addition theorem above (source s near box centre C_s, observer o near C_o; prefactor
−jk/4π; translator without that prefactor, `mlfmm::translator`), the source side carries
e^{−jk k̂·(C_s − s)} = e^{+jk k̂·(s − C_s)} and the observer side e^{−jk k̂·(o − C_o)}. For each leaf box
and each RWG `f_n` inside it, precompute for every direction `k̂`:
```
radiation (source side):  V_n(k̂) = ∫_{supp f_n} (I − k̂k̂) f_n(r') e^{+jk k̂·(r' − r_box)} dS'   (for L)
                          W_n(k̂) = k̂ × V_n(k̂)                                              (for K)
receiving (observer side): R_m(k̂) = ∫_{supp f_m} (I − k̂k̂) f_m(r)  e^{−jk k̂·(r − r_box)} dS
                                   = V_m(−k̂)            (also for complex k: no conjugation)
```
Both components (θ̂, φ̂) are stored. The same quantities with `k_2` serve region R2. The receiving
pattern is the radiation pattern evaluated at the opposite direction (the projector I − k̂k̂ is even
in k̂); with the (I − k̂k̂) projection the far-field form of the gradient terms of L is included
(transverse part only). Conjugation is never used: for complex k it would be wrong.

## Passes per matvec

1. **Aggregation (upward)**: leaf radiation `Σ_n x_n V_n(k̂)`; at each coarser level interpolate the child's pattern to the parent's finer sampling (Lagrange/global interpolation; later FFT in φ) and shift the phase `e^{−jk·(r_child − r_parent)}`.
2. **Translation**: for every box pair in interaction lists, multiply pointwise by the precomputed `T_L` (translators depend only on the level and the integer offset → ≤ 316 unique per level).
3. **Disaggregation (downward)**: phase-shift and anterpolate (transpose of interpolation) from parent to child; at the leaves integrate against receiving patterns with the quadrature weights.
4. **Add near part**: `y += Z_near x`.

Each of the four blocks (`L1, L2, K1, K2`, combined by the formulation weights) reuses the same tree and translators per region; `L` and `K` differ only in the pattern (`V` vs `k̂ × V`), so aggregation can be done once per region per matvec and both operators obtained from the same outgoing pattern.

## Lossy media

The interior wavenumber `k_2` is complex. Consequences:
- `h_l^{(2)}(k_2 r)` grows and the plane-wave sampling of `e^{−jk_2·r}` becomes exponentially ill-conditioned for `Im(k_2)·D` large. For Ag (`Im(n) ≈ −3.1`, skin depth ≈ 13 nm) the interior expansion is useless beyond the leaf level; for Si (`Im(n) ≈ −0.07`, absorption length ≈ 0.6 µm) it is usable for a few levels.
- Policy: compute, per level and region, the expansion error on a test pair against direct integration; use the expansion only where the error is below the target. For deeper levels the interior interaction decays as `e^{−|Im k_2| r}`; it is retained **only if above the tolerance**, otherwise dropped *as a numerical truncation below the requested accuracy* (reported in the log). This keeps the method rigorous to the stated tolerance.
- Phase 8 alternative: ACA/H-matrix for the interior operator — kernel-independent, no stability issue with complex `k`.

## Complexity and memory

`O(N log N)` operations and memory for the far part when the truncation `L` grows linearly with box size (high-frequency regime). Reported in `MlfmmOperator::describe()` per level: boxes, sampling points, translators, pattern memory, time per stage.

## Validation hooks

- `MlfmmOperator` vs `DenseOperator` matvec on the same `Problem` (docs/05_validation.md).
- Single-level FMM special case (levels = 2) as a stepping stone before the full multilevel implementation.
- Translator symmetry and reciprocity checks on random box pairs.
