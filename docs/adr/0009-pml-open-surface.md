# ADR 0009: Open rough surface terminated by a complex-coordinate-stretching PML

- **Status**: proposed (coordinator draft 2026-10-13; open points for the project lead at the end)
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

## Decision (proposed)

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
   - Not evaluating the continued angular spectrum inside the PML avoids its exp(+k Im x̃) growth.
5. **Far field and power.**
   - Far fields are computed from the currents on the real part of the surface only (patch, taper, gap), with a
     smooth window across the gap. The PML currents are excluded (their radiation integral grows like
     exp(+k Im x̃)).
   - The window and gap width are validated by the flat-surface test (7a). A Huygens-plane alternative is kept as a
     fallback if the windowed far field does not pass.
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
  - WP-P1: geometry and stretch map with unit tests;
  - WP-P2: stretch-aware dense kernels and singular part, with quadrature convergence tests;
  - WP-P3: excitation masking and windowed far field, validation 7a;
  - WP-P4: SPP absorption 7b and sensitivities 7c/7d/7e, Ag rough-surface results;
  - WP-P5 (later): fast path for the PML ring (ADR 0008 amendment).
- **Revisit after WP-P3:** if 7a fails because of the far-field window, switch decision 5 to the Huygens plane. If
  the PML start reflects SPPs above 1e-2 at affordable W_pml, revisit the profile (order m, σ_max) or a hybrid with
  option (a).

## Open points for the project lead

1. **Model statement.** Is "rough spot in a polished surface with absorbing surroundings" acceptable as the physical
   model for quantitative rough-Ag speckle (and later for Si)?
2. **Profile defaults.** Start from your earlier values (m = 2, σ_max = 3–5, W_pml = ?) and confirm by 7c. What
   PML thickness in λ did you use, and was the far field taken from the physical part only?
3. **Scope.** Is a dense prototype first, with the fast PML ring later as its own decision (6), the right order? Or
   is a fast version needed before any Ag result?
4. **Box for Si.** Keep the validated box for Si, or move Si to the PML route as well once 7e passes?
