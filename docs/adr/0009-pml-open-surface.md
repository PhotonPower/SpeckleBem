# ADR 0009: Open rough surface terminated by a complex-coordinate-stretching PML

- **Status**: accepted 2026-10-13 (project lead: open points 1, 3, 4 decided; 2 settled by validation 7c)
- **Date**: 2026-10-13
- **Phase**: 4 → 5 (prerequisite for quantitative rough Ag, ADR 0006 amendment 2026-10-12b decision 3)

## Context

ADR 0006 closes the finite rough patch into a box. For Si this is validated: uniform box, w₀ ≤ L cos θ/6, provisional
(amendment 2026-10-12b). For Ag it is not. WP-V3 (`benchmarks/results/ag_box_mechanism.md`) found that the Ag box
sensitivity (depth ×2: 0.4–1.9 %, L ×1.2: 4.5–8.2 % at L = 4 µm) is a physical surface-wave effect of the finite Ag
block, not a numerical artefact.
- A wave with |k_t| ≈ k₀…1.055 k₀ runs down the walls and radiates at the foot. Its depth phase follows
  β(θ) = k_w + k₀ cos θ, with k_w = 0.96–1.07 k₀.
- The rim scatters it, and the added rough annulus radiates.
- The SPP amplitude decay length at 500 nm is 41.5 µm, so neither a deeper box nor a larger L (≲ 30 µm) removes it.
- The box cavity resonances of WP-A2 exist but do not drive the far-field sensitivity.

The project lead has good experience with a PML by complex coordinate stretching at the boundary of the BEM domain
(dense PMCHWT for plasmonic structures on infinite substrates: x̃ = x + i∫σ dx under exp(−iωt), a polynomial
profile of order 2, σ_max 3–5). WP-V3 listed an absorbing termination (R1) as the recommended remedy.

**What any termination can represent.** At L ≪ L_spp no local termination approximates the semi-infinite rough
surface. Its SPPs would scatter off roughness outside the patch for tens of µm. A termination therefore defines a
**modified, well-posed problem**: a rough patch embedded in a smooth (flat) interface of the same materials, whose
outer part absorbs everything that reaches it. The docs/05 box checks must be read as checks of that model, not of
the infinite rough surface. The model is physical in its own right: a finite rough spot on a polished surface.

Options:
- **(a) Keep the closed box and add a lossy impedance frame** (WP-V3 R1, first variant). It absorbs bound SPPs but
  not free grazing waves (WP-V3 W3), and it needs a new PMCHWT/IBC junction (SPP 1.050 vs 1.055 k₀ mismatch).
- **(b) An open surface: rough patch → flat continuation → complex coordinate stretching (PML) → truncation.** It
  needs no closing box, so no walls, no bottom and no cavity. The stretching damps every wave travelling outwards
  in the stretched direction, bound or free, by exp(−Re(k_t) Im Δx̃). This is the approach the lead has used.
- **(c) A layered-medium (Sommerfeld) Green's function** for the flat surroundings. Exact for the flat part, but
  every kernel and the fast method change (already rejected in ADR 0006 for cost).

## Decision

Option **(b)**, introduced as an **alternative termination** next to the closed box. The box stays for Si and for
regression. The PML route becomes the default for plasmonic objects once it passes validation (below).

1. **Geometry.**
   - The open surface is z = ξ(x, y) on the patch |x|, |y| ≤ L/2. A taper zone of width W_t, at least one correlation
     length, blends ξ smoothly to 0 (C¹, e.g. a raised-cosine window on ξ).
   - The flat continuation z = 0 extends to |x|, |y| ≤ L/2 + W_t + W_g + W_pml (W_g: an optional flat gap) and ends
     with an open edge, where the currents have decayed.
   - The mesh stays **real**. Triangles in the PML are flat, at z = 0.
   - RWG functions on the outer boundary edges are omitted (zero normal current there). This is admissible because
     the currents at the truncation are below the PML target.
2. **Stretching as a map, not as complex vertices.**
   - r̃ = S(r) = (x̃(x), ỹ(y), z) acts only in the PML (z = 0 there):
     x̃ = x − j sgn(x) ∫_{x_p}^{|x|} σ(t) dt for |x| > x_p, with σ(t) = σ_max ((t − x_p)/W_pml)^m.
     **This is the exp(+jωt) form.** The lead's exp(−iωt) code uses x̃ = x + i∫σ dx; the sign flips, so that
     outgoing e^{−jkx} decays.
   - Basis functions are mapped by the Piola transform, f̃ dS̃ = DS f dS and (∇̃·f̃) dS̃ = (∇·f) dS with
     DS = diag(s_x, s_y, 1), s = 1 − jσ. The Galerkin integrals keep the real quadrature, the real RWG functions and
     their divergences. They change only by:
     - the complex distance R̃ = |S(r) − S(r')|, evaluated on the principal branch Re R̃ ≥ 0;
     - the Jacobian factors DS in the vector products and in the gradient of G;
     - nothing else.
   - The map is a geometry-level object (`geometry::CoordinateStretch`, identity outside the PML). The kernels take
     it as an optional parameter. With no stretch present, the code paths and results are bit-identical to today
     (regression guard).
3. **Kernels and singular integrals.**
   - G(R̃) = e^{−jkR̃}/(4πR̃) for both regions (k₁, and complex k₂).
   - Singularity extraction stays on the real distance R (ADR 0004). The smooth remainder G(R̃) − 1/(4πR) contains
     the stretching: in the PML, near r = r', R̃ ≈ |DS (r − r')|, so the remainder is O(1/R) anisotropic, not smooth.
     The singular part must therefore be extracted with the **stretched** local metric, 1/(4π|DS(r_c)(r − r')|) with
     DS frozen at the element, or integrated with Duffy / Sauter–Schwab directly on the analytic kernel.
   - Which of the two is the implementation WP's choice, with a convergence test against high-order quadrature.
   - This is the one place where the stretching touches ADR 0004.
4. **Excitation.**
   - The incident field is the analytic continuation of the beam, evaluated at r̃, on test functions inside the
     taper and gap, and **zero on test functions in the PML**.
   - This requires a dark band: |E_inc| at the PML start below the PML target (≈ 1e-4 relative), which with the
     rigorous beam means w₀ ≲ (L/2 + W_t + W_g)/2. The waist rule is restated for the PML route after validation.
     *Provisional rule (WP-PML3a, superseding the estimate above):* the enforced amplitude ratio
     |E_inc|(PML start) / max |E_inc|(patch) ≤ 1e-4 (amplitude, i.e. 1e-8 in intensity) needs
     pml_start ≳ 3.5 w₀ with w₀ ≳ λ for the rigorous beam (w₀ ≲ (L/2 + W_t + W_g)/2 only reaches ≈ 2e-2); to be
     confirmed by 7a (WP-PML3b), including the larger footprint at oblique incidence.
     *Rule after 7a (WP-PML3b, measured, `benchmarks/results/pml_validation.md`):* at 0° the provisional rule holds
     (w₀ = λ, pml_start = 3.5 w₀: ratio 1.3e-5). At oblique incidence it is not sufficient (measured), and
     **w₀ cos θ ≳ λ** is needed. Explanation, consistent with but not fitted to the measured ratios: the rigorous
     beam's spectrum, cut at k_t = k about the beam axis, puts grazing plane waves of relative amplitude
     ≈ exp(−(k w₀ cos θ)²/4) into the surface, and the beam defocuses along its footprint (Rayleigh range
     π w₀²/λ), so pml_start grows faster than w₀/cos θ. At 45°, w₀ = 0.5 µm never
     reaches 1e-4 (≥ 2e-3 up to pml_start 3.75 µm). The smallest measured passes are w₀ = 0.75 µm with
     pml_start = 4.25 µm (L = 6.5 µm, 8.9e-5, 2N = 4.2·10⁴ at λ/4) and w₀ = 0.7 µm with pml_start = 4.5 µm
     (7.7e-5); 7a uses w₀ = 0.75 µm with pml_start = 4.5 µm (L = 7 µm, 5.2e-5 / 4.0e-5 for p / s, ≈ 2× below
     1e-4), chosen with margin, i.e. 2N = 4.6·10⁴ at λ/4. **7a does not need the 1e-4 level.** At a dark-band ratio of
     5.9e-3 (tolerance relaxed) R changed by ≤ 0.06 % against the 5e-5 geometry (2.6e-2: ≤ 0.15 %, Si 45°, λ/4; at λ/6 the 5.9e-3 and 2.6e-2 geometries agree within 0.05 % (p) / 0.08 % (s)). The 1e-4 default is therefore
     conservative for the far field. Relaxing it for rough-surface runs (which would cut the 45° surface by ≈ 2.5×)
     is left to WP-PML4 (7c/7d at both levels). Mesh found by 7a: h = λ/4 suffices for Ag and for Si at normal
     incidence. **Si at oblique incidence needs h ≤ λ/6** (≈ 0.7 λ/n_Si). At λ/4 it passed only by chance:
     λ/5 was +5.9 % off. Indicator: the power balance 1 − R − A_real.
   - Not evaluating the continued angular spectrum inside the PML avoids its exp(+k Im x̃) growth.
   - *Implementation (WP-PML3a).* `op::pml_masked_functions` / `op::assemble_rhs`: a test function is masked (its J
     and M entries are exactly 0) iff one of its two support triangles has a vertex strictly inside the stretched
     zone (`CoordinateStretch::in_pml`: |x| > x_p or |y| > y_p). An RWG function straddling the Gap/Pml line (one
     triangle in the gap, one in the PML) is therefore masked; one whose triangles only touch the start line is
     not. Every other entry is bitwise the unstretched one (the stretch is the identity on its support, so the
     incident field is evaluated at the real r), and the incident field is evaluated only on triangles carrying an
     unmasked function. `Simulation` (PML constructor) measures the dark band as max |E_inc| on the PML-start
     square (z = 0, spacing ≤ λ₁/8) over max |E_inc| on the patch vertices and rejects ratios above
     `PmlTermination::dark_band_tolerance` (default 1e-4); the excitation's controlled ball must cover the
     vertices of the triangles the masked right-hand side integrates.
5. **Far field and power.**
   - Far fields are computed from the currents on the real part of the surface only (patch, taper, gap), with a
     smooth window across the gap. The PML currents are excluded (their radiation integral grows like
     exp(+k Im x̃)).
   - The window and gap width are validated by the flat-surface test (7a). A Huygens-plane alternative is kept as a
     fallback if the windowed far field does not pass.
   - *Implementation (WP-PML3a).* `post::FieldOptions::window` (`post::CurrentWindow{inner_half, outer_half}`)
     weights the currents at each quadrature point by a raised-cosine C¹ window of t = max(|x|, |y|) (1 up to
     inner_half, 0 from outer_half). `post::far_field` requires it for a non-identity stretch, with outer_half ≤ the
     stretch start; `Simulation` uses inner_half = taper_end, outer_half = pml_start (across the gap, so W_g > 0 is
     required). Near fields (`scattered_field`, `total_field`) refuse the stretch and the window.
   - *Validated by 7a (WP-PML3b).* The windowed far field passes with W_g = 0.5 µm (= λ). The Huygens-plane
     fallback is not needed. Gap/window ×1.5 (0.75 µm) changes R by ≤ 1.2·10⁻⁵ relative (Si and Ag, 45°, λ/4), and
     W_pml ×1.5 or σ_max ×1.5 by ≤ 2·10⁻⁷ (preview of 7c; flat surface, no SPP channel).
   - A_ext (ADR 0006 amendment 2026-10-12) remains the absorption measure. A_local over the patch is reported only as
     a diagnostic.
6. **Fast method.**
   - The **prototype is dense** (validation only, 2N ≲ 6·10⁴).
   - At scale the PML ring is a large share of the surface: W_pml ≈ 1–2 µm against L = 30 µm gives about 4 W/L ≈
     20 % of the area. The MLFMM plane-wave translations are not valid for complex coordinates.
   - The fast version therefore treats every interaction with a PML element in the near-field / ACA path
     (H-matrix blocks with complex distances) and keeps the MLFMM for real–real pairs.
   - That is a separate WP with its own decision, taken after the dense prototype, together with the cost model.
     It amends ADR 0008, not this ADR.
7. **Validation (replaces the docs/05 box checks for the PML route).**
   - **(a) Flat interface:** a flat Ag and a flat Si surface with PML under the rigorous beam, 0° and 45°, s and p,
     must reproduce Fresnel r_s, r_p to < 1 % on |r|². This is the PML analogue of WP22c and the gate for 5.
   - **(b) SPP absorption:** an SPP launched on a flat Ag surface (dipole or a sub-wavelength groove) must be
     reflected by the PML start with |r_spp| < 1e-2.
   - **(c) PML-parameter sensitivity:** far field change < 0.1 % for W_pml ×1.5 and for σ_max ×1.5. This replaces
     depth ×2.
   - **(d) L ×1.2 sensitivity**, unchanged, read for the "rough spot in a polished surface" model.
   - **(e) The Si cross-check:** the PML route against the validated uniform box at L = 4 µm, w₀ = L/6 (agreement
     within the box's own sensitivity).
   - docs/05 gets a "PML route" block. Its tolerances are not loosened.

## Consequences

- **Removes for metals:**
  - the walls, the bottom plate, the cavity resonances (WP-A2) and the A_local defect of the closed box;
  - for Si, the fine band and the box (about 50–150 % extra unknowns).
- **Adds:**
  - the PML ring, about 20 % of the area at L = 30 µm, more at small L;
  - a stretch-aware kernel path;
  - a different near-field share for the fast method.
- **Changes the physical model**, which must be stated in every result of the PML route: a rough spot in a polished
  surface. The truncated box was neither this nor the infinite rough surface. For Ag, at L ≪ 40 µm, no affordable
  model is the infinite rough surface.
- **Layers touched:** geometry (open mesh with taper and PML zones, `CoordinateStretch`), basis (boundary-edge
  handling of the open mesh), kernels (complex distance, Jacobians, singular part), excitation (masking), post
  (windowed far field), `Simulation` (choice of termination). This is why it needs this ADR (docs/02 layering:
  the stretch lives in geometry and is passed downward).
- **Work packages:**
  - WP-PML1: geometry and stretch map with unit tests;
  - WP-PML2: stretch-aware dense kernels and singular part, with quadrature convergence tests;
  - WP-PML3: excitation masking and windowed far field, validation 7a;
  - WP-PML4: SPP absorption 7b and sensitivities 7c/7d/7e, Ag rough-surface results;
  - WP-PML5 (later): fast path for the PML ring (ADR 0008 amendment).
- **Revisit after WP-PML3:** if 7a fails because of the far-field window, switch decision 5 to the Huygens plane. If
  the PML start reflects SPPs above 1e-2 at affordable W_pml, revisit the profile (order m, σ_max) or a hybrid with
  option (a).

## Lead decisions (2026-10-13)

1. **Model accepted:** "rough spot in a polished surface with absorbing surroundings" is the physical model for
   quantitative rough-Ag speckle on the PML route.
2. **Profile defaults:** start from m = 2, σ_max = 3–5 (the lead's experience); W_pml and the far-field window are
   fixed by validation 7a–7c (no lead value given; start with W_pml = 2 λ₁ and test 1.5×).
3. **Dense first:** the prototype is dense (decision 6); the fast PML ring (WP-PML5) is scheduled only after WP-PML4.
4. **Si stays on the box:** the validated uniform closing box (ADR 0006, amendment 2026-10-12b) remains the Si route;
   the PML route targets plasmonic objects. Check 7e is a cross-check only, not a migration.

## Open points (original draft, kept for the record)

1. **Model statement.** Is "rough spot in a polished surface with absorbing surroundings" acceptable as the physical
   model for quantitative rough-Ag speckle (and later for Si)?
2. **Profile defaults.** Start from your earlier values (m = 2, σ_max = 3–5, W_pml = ?) and confirm by 7c. What
   PML thickness in λ did you use, and was the far field taken from the physical part only?
3. **Scope.** Is a dense prototype first, with the fast PML ring later as its own decision (6), the right order? Or
   is a fast version needed before any Ag result?
4. **Box for Si.** Keep the validated box for Si, or move Si to the PML route as well once 7e passes?
