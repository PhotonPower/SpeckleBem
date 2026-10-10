# ADR 0008: MLFMM design (near/far split, sampling, interpolation, lossy regions)

- **Status**: accepted
- **Date**: 2026-10-09
- **Phase**: 4

## Context
Phase 4 replaces the dense operator by `Z = Z_near + Z_far` with a multilevel fast multipole far part
(docs/04_theory_mlfmm.md, ADR 0002). Several choices cut across `kernels`, `operator` and `mlfmm` and
must be fixed before the work packages start: how basis functions are grouped and the near/far split
is defined, how directions are sampled and interpolated between levels, how translators are evaluated
for complex wavenumbers, how the strongly lossy interior of metals is handled without giving up the
"rigorous to a stated tolerance" contract, and where parallelism lives. The dense operator stays the
oracle (docs/05): every MLFMM result is checked against it.

## Decision
1. **Grouping and split.** RWG functions are grouped by edge midpoint into an octree (Morton order,
   leaf ≤ `max_elements_per_leaf` and box edge ≥ `min_box_size_lambda`·λ₁, default λ/4). A basis pair
   (m, n) is *near* iff their leaf boxes coincide or touch (3×3×3 neighbourhood); near entries are the
   exact Galerkin entries assembled from `kernels::element_blocks` (the same code as `DenseStrategy`,
   so Z_near equals the dense matrix on its pattern up to summation order) and stored in
   `op::SparseOperator`. All other pairs are *far* and go through the expansion. Because a basis
   function extends up to its support radius beyond its midpoint, the truncation uses the box
   diagonal enlarged by twice the largest support radius of the level.
2. **Sampling and truncation.** Per level and region: L = kD + 1.8 d₀^{2/3} (kD)^{1/3}
   (d₀ = `accuracy_digits`, D the enlarged box diagonal, k = Re k for the bandwidth), directions =
   (L+1) Gauss–Legendre points in θ × 2(L+1) uniform points in φ.
3. **Interpolation.** Between levels: separable local Lagrange interpolation (θ with polar
   reflection, φ periodic) of order p (default 6, chosen so the interpolation error is ≤ 0.1 × the
   expansion target; measured in tests); anterpolation is its exact transpose. FFT/global
   interpolation is a later option (`use_fft_interpolation`).
4. **Translators.** T_L(k, r, k̂) = Σ_{l≤L} (−j)^l (2l+1) h_l^{(2)}(kr) P_l(k̂·r̂) for exp(+jωt), with
   spherical Hankel functions of complex argument from an in-house recurrence (no new dependency),
   validated against series and asymptotic values. Precomputed per (level, region, integer offset),
   ≤ 316 offsets per level.
5. **Both regions, same tree.** One octree; per region its own k, L per level, patterns and
   translators. Radiation patterns of `L` and `K` share the aggregated vector (V and k̂ × V).
6. **Lossy regions (rigour contract).** For each region and level an a-priori check compares the
   expansion of a worst-case box pair in the interaction list against direct evaluation of the Green's
   function; the expansion is used only where the relative error is ≤ 10^{−d₀}. Where it is not,
   either (a) the interaction magnitude bound exp(−|Im kᵢ| d_min)/d_min (d_min = minimum
   centre-to-centre distance of the interaction list minus the enlarged diameter) is ≤ 10^{−(d₀+1)}
   relative to the self term, in which case the region-i far interaction of that level is dropped —
   a numerical truncation below the requested accuracy, logged with the bound — or (b) the pairs are
   moved to the near part and integrated directly. Silent dropping is not allowed.
7. **Parallelism and layout.** OpenMP over boxes inside each pass (these are the "translation loops"
   of docs/07); no parallel regions in kernels. Patterns stored as contiguous arrays indexed by
   (level, box, direction, component) per ADR 0005, so stages can become backend kernels in Phase 7.
   Results must be deterministic for a given thread count; tests compare against dense with a
   tolerance, not bitwise.
8. **Validation gates.** Single-level FMM (two well-separated boxes) against the dense far blocks
   before the multilevel passes; MLFMM matvec vs dense < 1e-3 (d₀ = 3) and < 1e-5 (d₀ = 5) on spheres
   and rough surfaces, Si and Ag (docs/05).

## Consequences
- Z_near reuses the validated kernels and inherits their accuracy; no second integration path.
- The lossy-region policy keeps the method rigorous to the stated tolerance for Ag, at the price of
  either more near-field work or documented truncation; both are visible in `describe()` and the log.
- Local Lagrange interpolation is simple and parallel-friendly but costs O(p²) per sample; if it
  dominates at large N, the FFT option is the planned upgrade.
- The enlarged-diameter truncation slightly over-samples small leaf boxes; acceptable for λ/4 leaves.

## Amendment 2026-10-09 (WP18 measurements)

WP18 measured the plane-wave building blocks (tests/unit/test_plane_wave.cpp):
- Local Lagrange interpolation at the §2 sampling (close to 4 points per pattern wavelength) with
  p = 6 reaches only 1e-3 … 1e-2; 0.1 × 10^{−d₀} needs p ≈ 12–14 for d₀ = 3 and p ≈ 22 for d₀ = 5.
- The corner-to-corner worst case of a two-box separation stays above 10^{−d₀} at the formula
  order for every box size (1e-3 … 9e-2 at d₀ = 3), and raising L is capped by the low-frequency
  breakdown (best ≈ 1e-2 at λ/4, 1e-3 at λ, 1e-5 at 4λ). Random points inside the boxes meet
  10^{−d₀} for boxes ≥ λ/2; at λ/4 the formula loses up to one digit.

Decisions replacing the corresponding parts of §2, §3 and §6:
- **§3 interpolation order:** p is a function of d₀ taken from a measured table (default
  p = 14 for d₀ ≤ 3, p = 22 for d₀ ≤ 5, extended by measurement for other d₀; still overridable).
  The FFT/global option remains the planned upgrade if interpolation dominates.
- **§2/§6 per-level accuracy check:** statistical, not worst-case corner: the maximum relative
  error over a fixed, seeded set of random source/observer points inside the boxes for the
  nearest interaction offsets (the norm-wise error that the docs/05 matvec criterion measures).
  The matvec-versus-dense comparison of docs/05 remains the authoritative acceptance test.
- **§2 truncation order per level:** the smallest L ≥ the formula value for which the statistical
  check meets 10^{−d₀}; if no such L exists before the breakdown onset (detected by the check
  getting worse with growing L), that level is not used as a leaf level (the leaf level is
  coarsened, i.e. the effective `min_box_size_lambda` rises for this d₀ — expected at λ/4 for
  d₀ = 5). For complex k the same search is the §6 validity test; where it fails at every L the
  §6 alternatives (documented truncation by the decay bound, or near-field fallback) apply.

## Amendment 2026-10-09 (WP18 review)

- The interpolation order of the table can exceed the number of θ nodes of a coarse leaf sampling
  (λ/4 leaves: n_θ = 9 at d₀ = 3, 12 at d₀ = 5). The leaf sampling is therefore oversampled when
  needed: L_leaf = max(L from §2 / the amendment search, p − 1), and the interpolator accepts any
  order whose stencil fits the extended (pole-reflected) θ range. The cost is a larger leaf
  sampling only on the finest level.
- Pattern phase convention: radiation patterns carry e^{+jk k̂·(r' − r_box)}, receiving patterns
  e^{−jk k̂·(r − r_box)} = radiation pattern at −k̂ (docs/04 corrected; no conjugation, also for
  complex k).
- Special functions never fail silently: spherical Hankel values that underflow or overflow the
  double range raise exceptions, and translators are checked for finiteness.

## Amendment 2026-10-09 (WP18 review round)

- `mlfmm::search_truncation_order` implements the per-level search of the first amendment
  (statistical check, breakdown detection). For strongly lossy media the bandwidth formula with
  Re k gives far too small orders (Ag: 2–3); the search finds usable orders for small boxes
  (Ag at d₀ = 3: L ≈ 9 / 13 / 22 for 0.05 / 0.1 / 0.25 λ₀) and reports breakdown for larger
  ones. The §6 policy therefore always uses the search result, never the formula alone, for
  complex k; the formula only provides the starting order.
- The interpolation order is required explicitly (`interpolation_order(d₀)`: 14 for d₀ ≤ 3, 22
  for d₀ ≤ 5; d₀ > 5 must be measured first). When the oversampled leaf sampling equals the
  parent sampling (λ/4 leaves at d₀ = 3) the leaf-to-parent interpolation is the identity and may
  be skipped.

## Amendment 2026-10-09 (WP19b single-level measurements)

Single-level FMM far blocks of RWG functions against the dense entries (relative Frobenius error at the
order found by `search_truncation_order`): leaf edge a = λ/2: 1.3e-3 (d₀ = 3) and 3.6e-4 (d₀ = 5);
10^{−3} is met from a ≈ 0.75 λ, 10^{−5} from a ≈ 1.5 λ (lossy Si from ≈ 1 λ); the errors depend only on
a/λ (in the medium) and r_max/a and come from corner-near basis pairs at the nearest interaction
offsets, which the volume-random check rarely samples. Decisions:
- **Per-level acceptance check:** block-based — sampled far blocks of real basis functions (`far_block`
  vs dense entries) at the nearest interaction offsets present in the tree, or an equivalent point check
  with points on box faces and corners. The volume-random check remains a fast pre-screen for the order
  search only.
- **Leaf level:** the finest level that passes the block check at 10^{−d₀} for the chosen L (coarser
  leaves otherwise), unless WP20 shows that the docs/05 matvec criterion (norm-wise, dominated by the
  exact near field) is met with finer leaves; WP20 measures both and the result is recorded here before
  WP21 fixes the policy. Expected default: λ/2–0.75 λ leaves for d₀ = 3.
- Pattern quadrature target 0.01 × 10^{−d₀} (`PatternOptions::target_accuracy`).

## Amendment 2026-10-09 (WP20a multilevel measurements: leaf policy for d₀ = 3)

Multilevel far operator vs dense (4-level icospheres, r_max/a ≈ 0.53, vacuum exterior, n = 1.5 or Si
interior, PMCHWT, d₀ = 3): far-only error 1.7e-3 / 4.6e-4 / 1.6e-4 for λ/4 / λ/2 / 0.75 λ leaves;
**full-matvec error (docs/05 metric) 2.2e-4 / 1.7e-4 / 7.8e-5** — below 1e-3 at every leaf size; the
multilevel errors equal the single-level block errors at the same a/λ (interpolation adds nothing
visible). Decisions:
- **d₀ = 3:** the acceptance criterion is the docs/05 full-matvec error; λ/4 leaves are allowed (they
  keep the near field at ~15 GB for the 393 k-unknown WP22 case instead of 60–135 GB at 0.75 λ). The
  block-based check of the WP19b amendment is to be reported per level from WP20b/WP21 on (WP20a's
  `describe()` reports the volume-random search error only) and does not by itself
  coarsen the leaves for d₀ = 3.
- **d₀ = 5:** not yet established on a multilevel tree — on the test meshes (r_max/a ≈ 0.5) the order
  search finds no order meeting 1e-5. WP20b measures it on finely meshed problems (r_max/a ≤ 0.3,
  2N ≥ 2·10⁴); until then d₀ = 5 is documented as unverified.
- The volume-random search is pessimistic for r_max/a ≳ 0.5; a minimum leaf size relative to r_max
  (e.g. a ≥ 3.5 r_max for d₀ = 5) is to be decided with the WP20b data.

## Amendment 2026-10-10 (WP20b: full MLFMM operator vs dense)

`MlfmmOperator` (near + far) against the dense operator, relative matvec error over three random
vectors (λ₀ = 500 nm, vacuum exterior; reproduced by the WP20b review):

| case | 2N | interior | d₀ = 3, λ/4 leaves (r_max/a ≈ 0.57) | d₀ = 5, λ/2 leaves (r_max/a ≈ 0.29) |
|---|---|---|---|---|
| icosphere R = 1 µm | 15 360 | n = 1.5 / Si | 1.7e-4 / 2.1e-4 | 3.7e-6 / 6.5e-6 |
| rough box 2 × 2 × 0.3 µm, 50 nm | 24 960 | n = 1.5 / Si | 7.2e-5 / 4.3e-5 | 6.7e-6 / 4.2e-6 |
| rough plate 4 µm, 100 nm, λ leaves | 22 080 | n = 1.5 | 5.4e-6 | 5.5e-7 |
| rough box 4 µm, 50 nm (exact rows) | 88 320 | Si | 2.9e-4 | — |

Both docs/05 criteria (< 1e-3 at d₀ = 3, < 1e-5 at d₀ = 5) are met for dielectric and Si interiors.
Decisions: **d₀ = 5 is verified** on multilevel trees for leaves ≥ λ/2 with r_max/a ≤ 0.3; λ/4 leaves
remain limited to d₀ = 3. Leaf rule: a ≥ max(a_min(d₀), r_max / 0.3) with a_min = λ/4 (d₀ = 3),
λ/2 (d₀ = 5); WP21 applies it automatically. The d₀ = 5 near field with λ/2 leaves is ~6.6× the λ/4 one
(1.5 GB vs 0.23 GB at 2N = 2.5·10⁴) — relevant for WP22 sizing. Lossy interiors (Ag) are still rejected
by the order search; WP21 implements §6. The per-level block check is reported from WP21 on.

## Amendment 2026-10-10 (WP21: lossy-region policy, leaf rule, exact-part budget)

Replaces §6 and the leaf rule of the WP20b amendment. Per region and level, from the leaf upwards
(expansion levels are contiguous: once a level does not use the expansion, no coarser level does):
1. **Expansion** if the order search is achievable and, for **lossy** regions, the block check passes
   (≤ 10^{−d₀} or ≤ 2 × the block error of the lossless analogue k = Re k_i, whose own search must be
   achievable). For lossless regions the block check is reported only (the WP20a/b acceptance by the
   full-matvec error stands).
2. Otherwise per box pair (and inside exact box pairs per basis pair), with d = distance between the
   bounding boxes of the support vertices (a true lower bound on |r − r'|) and α = −Im k:
   δ = (1 + αd) e^{−αd} bounds every region-i entry relative to its undamped bound U (|G| ≤ δ/(4πR),
   |∇G| ≤ δ(1 + |Re k|R)/(4πR²)). **Truncation** if δ ≤ 10^{−(d₀+1)} — counted and logged with the bound,
   never silent; otherwise **exact** evaluation in a per-region sparse part storing (L_i, K_i) per basis
   pair (40 B/pair, weights applied in apply).
3. The exact fallback is allowed only if x*(d₀)/α ≤ 4·a_l, with x* the root of (1 + x)e^{−x} =
   10^{−(d₀+1)} (x*(3) = 11.76, x*(5) = 16.69) and a_l the box edge of the first level without
   expansion; otherwise `TruncationOrderError` (cause `lossy_region`; `mesh_or_leaf_size` when
   r_max/a > `max_support_ratio(d₀)`; `digits` otherwise).
4. **Exact-part budget:** estimated before any allocation (box-pair bound, then the exact per-basis
   count); default max(2 × near-field bytes, 1 GiB), `MlfmmParams::max_exact_far_bytes`; exceeding it
   raises `TruncationOrderError` (cause `exact_part_too_large`).
5. **Leaf rule:** a ≥ max(a_min(d₀), r_max / max_support_ratio(d₀)) with a_min = λ/4 and ratio 0.6 for
   d₀ ≤ 3, a_min = λ/2 and ratio 0.3 for d₀ ≤ 5 (applied automatically; the user value is a lower bound).

Measured (Ag, vacuum exterior): icosphere 2N = 15 360 matvec 1.0e-4 (d₀ = 3, λ/4) / 4.1e-6 (d₀ = 5,
λ/2), exact part 170 / 100 MB; rough box 2N = 21 216: 6.4e-5 / 2.1e-6; icosphere 2N = 61 440 (exact rows):
1.1e-4, exact part 687 MB vs near field 1133 MB; Ag GMRES through Simulation matches dense (RCS 3.3e-6).
WP22 projection (Ag sphere 4 µm, λ/27, N ≈ 196 k): exact part ≈ 15 GB (d₀ = 3) / 30 GB (d₀ = 5) next to a
near field of ≈ 15 GB (d₀ = 3).
