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

- **Octree** (`mlfmm::Octree`) over RWG edge midpoints. Root = the mesh **vertex** bounding box (`TriangleMesh::bounding_box()`) turned into a cube of edge `e (1 + 2·10⁻⁶)` (`e` = largest extent), **anchored at the lower corner** `lo − 10⁻⁶ e` rather than centred, so flat or thin geometry stays in one layer of boxes instead of being cut at its mid-plane; points on the upper root faces are clamped into the last box. Morton ordering, so that the elements of a box are contiguous.
- **Uniform depth**: all leaves lie on the finest level `D`, so the passes need only level-wise lists (no adaptive U/V/W/X lists). `D` is the first level on which every box holds ≤ `max_elements_per_leaf` (≈100) elements, unless a further split would make the box edge smaller than the floor or exceed `max_levels`; then the leaves may hold more. Floor: a level is admitted if its box edge is ≥ `(1 − kMinBoxSizeTolerance) · min_box_size_lambda · λ` with `kMinBoxSizeTolerance = 10⁻²` (default λ/4). The floor only guards against the gradual low-frequency breakdown — the truncation uses the actual enlarged box diagonal and the accuracy is checked per level (ADR 0008) — so it need not be exact; the tolerance keeps a root of 7.99 λ from losing its λ/4 level. Example: the 4 µm sphere at λ = 500 nm has an 8 λ root and 6 levels down to λ/4 leaves (icosphere n = 6: ≤ 58 elements per leaf). Paper: 120 elements per smallest box, 5 levels for the 4 µm sphere.
- **Near list** (every level): the same-level boxes of the 3×3×3 neighbourhood, **excluding the box itself**; the self-interaction is near as well (leaf-self and leaf-near-list blocks are assembled exactly). **Interaction list** (levels ≥ 2): children of the parent's near boxes that are not near themselves (≤ 189 boxes; the parent's own children are always near). Every leaf pair is either near (self or near list) or translated at exactly one level.
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

1. **Aggregation (upward)**: per region, two outgoing fields per leaf box, F_J = Σ_n x_J,n V_n(k̂) and F_M = Σ_n x_M,n V_n(k̂); at each coarser level interpolate the child's fields to the parent's sampling (local Lagrange, p from `interpolation_order(d₀)`, odd pole parity for θ̂/φ̂ components) and shift the phase by `e^{+jk k̂·(r_child − r_parent)}` (radiation convention).
2. **Translation**: for every box and every box in its interaction list, multiply pointwise by the precomputed translator times the quadrature weights of that level, w_q T_L(k, c_obs − c_src, k̂_q) (≤ 316 unique offsets per level); the incoming fields therefore carry the level's weights.
3. **Disaggregation (downward)**: shift the parent's incoming field by `e^{−jk k̂·(r_child − r_parent)}` and anterpolate (exact transpose of the interpolation) to the child — the parent weights are already in the field, no child weights are applied. At the leaves the reception is an **unweighted** sum over directions of the receiving patterns R_m(k̂) = V_m(−k̂) against U_J = e c_L G_J − e c_K k̂×G_M and U_M = h c_K k̂×G_J + m c_L G_M (c_L = ωμk/16π², c_K = k²/16π², e = a/η, h = bη, m = b/η); k̂× is applied once at reception (far_operator.hpp).
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
