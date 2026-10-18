# ADR 0010: SPP-aware far field of the PML route (Huygens surface in R1)

- **Status**: accepted 2026-10-18 (project lead followed the coordinator's recommendation; implementation after
  WP-PML5)
- **Date**: 2026-10-18
- **Phase**: 5

## Context

ADR 0009 decision 5 computes the far field of the PML route from the surface currents on the real part of the
surface (patch, taper, gap) with a raised-cosine window across the gap. Validation 7a (flat Ag and Si) and 7e
(rough Si vs the uniform box, 0.146 % against the box's own 0.229 %) pass with it, and 7c (W_pml ×1.5, σ_max ×1.5
on rough Ag: 2.5·10⁻⁵ / 4.6·10⁻⁵) shows that the PML itself is inert.

Validation 7d (L ×1.2 at fixed waist) **fails on rough Ag at 2.6 %** against 0.5 % (WP-PML4b,
`benchmarks/results/pml_rough_ag.md`), while the flat-Ag control gives 2·10⁻⁷ and rough Si passes. The mechanism is
not the termination: the roughness launches surface plasmon polaritons (decay length ≈ 41 µm at λ = 500 nm) that
run across the patch and into the flat gap (rms |J|, |M| in the gap ≈ 10 % / 9 % of the patch values). The far-field
window then truncates a bound wave: evaluated on the *same* currents, moving the 0.5 µm window outward by 0.375 µm
changes the far field by 1.6 %, widening it from 0.5 to 1 µm by 0.93 % and from 1 to 1.75 µm by 0.47 % — a
convergence like 1/W. The radiation integral of a truncated bound wave does not converge with the window; it
converges only when the truncation happens where the wave has decayed (the PML), which the window cannot include
(the PML currents' radiation integral diverges, ADR 0009 decision 5). A more lossy Ag (Im ε_r = −6 instead of the tabulated −0.313, i.e.
19×, SPP range shortened) halves the L dependence (1.1 %) but does not remove it.

Alternatives considered:

- **(a) Huygens surface in R1.** Evaluate E and H on a closed-enough surface above the real part of the structure
  (a plane z = z_H over patch + taper + gap, where the bound SPP field has decayed exponentially with
  exp(−|κ| z), κ = sqrt(k_spp² − k₀²) ≈ 0.34 k₀ → 1/e at ≈ 0.47 λ), and radiate the equivalent currents
  n̂ × H, −n̂ × E from that plane with the same window. The bound wave is filtered by distance, not by truncation.
  Cost: one near-field evaluation per Huygens quadrature point (O(N_H · N_triangles)), then a cheap plane
  radiation integral; the Huygens plane needs its own window and a stated z_H. Model-free.
- **(b) Subtract the guided wave from the currents before the window.** Fit and remove the SPP mode
  (k_spp known analytically for the flat gap) from J, M in the gap. Cheap, but model-dependent (mode shape,
  two-dimensional cylindrical spreading, several modes for structured metals) and silent when the model is wrong.
- **(c) Document only.** Report rough-Ag far fields with L and window stated, no claim of L independence.

## Decision

1. **(c) now**: every rough-metal result of the PML route states L, the window and the gap width; docs/05 row 7d is
   met on rough Si and documented as failed-with-cause on Ag; the 0.5 % tolerance is not loosened.
2. **(a) as WP-PML6 after WP-PML5**: an SPP-aware far field by a Huygens plane in R1 for the PML route:
   - `post::huygens_far_field` (or an option of `post::far_field`): E, H on a planar quadrature grid at z = z_H
     above the real part (patch + taper + gap) from the surface currents (`post::total_field`-style evaluation,
     scattered field only), then the radiation integral of the equivalent currents with the raised-cosine window
     in the plane; the incident field is not included (scattered far field as today).
   - Defaults from measurement: z_H ≈ λ (bound field ≤ e⁻² at 0.94 λ), grid spacing ≤ λ/4 in the plane, window as
     for the currents; both made configurable and validated by their own sensitivity (z_H ×1.5, spacing /1.5).
   - `Simulation::far_field` on the PML route switches to the Huygens evaluation by default once validated; the
     windowed current far field stays available as `FarFieldSource::Currents` for the comparison and for Si.
   - Fast path: the plane evaluation is O(N_H · N) direct; with WP-PML5 (fast PML ring) the near-field evaluation
     on the plane uses the same compressed operator (MLFMM for real–real pairs).
3. **Validation (docs/05 "PML route" block, new rows 7f–7g):**
   - 7f: flat Ag and Si (7a geometry): Huygens far field vs windowed-current far field within 1·10⁻³ relative in the
     L2 metric and R within 1·10⁻⁴ (no bound wave: both must agree).
   - 7g: rough Ag (the WP-PML4b surface): 7d with the Huygens far field < 0.5 %, window position ×(1 ± 0.3) and
     z_H ×1.5 each < 0.1 %; rough Si 7e repeated with the Huygens far field within the box's own sensitivity.
   - If 7g fails: the L dependence is physical (re-scattering of SPPs at the roughness changes the radiating
     currents with L, not only the window) — then the model statement of ADR 0009 ("rough spot in a polished
     surface") is the result and (b) is not pursued either.

## Consequences

- Rough-metal far fields of the PML route are not quantitative until WP-PML6; rough Si and all flat cases are
  unaffected (7a, 7e pass with the windowed currents).
- Adds a near-field evaluation stage to the far-field path of the PML route: cost ≈ N_H points × N triangles
  (≈ (L/λ·4)² points; for L = 4 µm, z_H = λ: ≈ 1·10³ points × 7·10³ triangles, seconds), plus the plane radiation
  integral per direction (cheap).
- Layers touched: post (new evaluation), simulation (default switch, options, Python mirror) — hence this ADR.
- ADR 0009 decision 5 is amended: the windowed current far field remains the validated route for flat surfaces and
  rough Si; for rough metals it is superseded by this ADR once WP-PML6 is validated.
- Revisit: if WP-PML5 changes the operator interface, the Huygens evaluation must use the compressed near-field
  product rather than the dense currents loop.
