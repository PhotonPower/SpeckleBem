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
