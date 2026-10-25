# ADR 0010: SPP-aware far field of the PML route (Huygens surface in R1)

- **Status**: accepted 2026-10-18 (project lead followed the coordinator's recommendation; implementation after
  WP-PML5); implemented in WP-PML6 (section "Result": 7f met, 7g not met, default unchanged); amended
  2026-10-25 (extended plane, lead decision 2026-10-22; withdrawn the same day — see the decision of
  2026-10-25: the L dependence of rough metals is accepted as physical, row 7g closed as "not met — physical")
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

## Result (WP-PML6, 2026-10)

- **Status:** implemented, 7f met, 7g not met; the default far-field source of the PML route stays the windowed
  currents (`FarFieldSource::Currents`); the Huygens plane is available as `FarFieldSource::Huygens` /
  `FieldOptions::huygens` (`post::huygens_far_field`, `post::exterior_scattered_field` with the PML source
  triangles radiating through the stretched Green's function). Record `benchmarks/results/pml_huygens.md`.
- **7f (flat Ag / Si, dense LU):** Huygens vs currents L2 2.73·10⁻⁴ / 2.83·10⁻⁴, ΔR/R −1.7·10⁻⁷ / −4.2·10⁻⁷; z_H ×1.5
  2.6 / 2.7·10⁻⁴ (vs the currents 5.3 / 5.4·10⁻⁴), spacing /1.5 8.5 / 7.4·10⁻⁷, plane quadrature 5·10⁻¹²; plane
  evaluation 0.3 s (729 points), 1.5 s at the rough U case (2 025 points).
- **7g (rough Ag U, fast path):** 7d with the Huygens plane 2.78 % (own default windows) / 1.83 % (the same absolute
  window), currents 2.57 %; z_H ×1.5 1.05 %, plane window ×0.7 / ×1.3 1.75 % / 0.66 %; spacing /1.5 4·10⁻⁵, gap
  width at a fixed window 3·10⁻⁵ (inert). Rough Si 7e with the plane: l2_refl 0.220 % and ε_rr 0.038 / 0.044 %
  pass, ΔR/R −5.3·10⁻⁴ fails (4.1·10⁻⁵; the same −5.2·10⁻⁴ against the currents of the same solve).
- **Diagnosis:** the plane window, not the physics. The plane ends at pml_start and its raised cosine spans the gap
  (Δ = λ): grazing directions are missed (θ 75–90°: −67 % band power vs the currents) and the window end leaves
  ≈ π²/(2(kΔ)²) per end of every field crossing it (point-source unit test 4.7·10⁻² / 9.8·10⁻³ / 2.3·10⁻³ at
  Δ = 2 / 4 / 8 λ, 0.28 at Δ = λ). The plane's own sensitivities are as large as the 7d change, so the clause
  "if 7g fails, the L dependence is physical" (decision 3) does not apply to this result. Tolerances not loosened.
- **Extended-plane diagnostic (not validated, for the lead decision):** the plane cannot reach over the PML without
  code changes and a theory for stretched observation points (record, section "Diagnostic: extended plane"); on
  the U surface with a 3 µm real gap and Δ = 1 / 2 / 3 / 4 / 6 λ: z_H ×1.5 1.05 / 0.49 / 0.28 / 0.18 / 0.10 %
  (θ ≤ 30°: 0.81 / 0.21 / 0.073 / 0.042 / 0.018 %), Huygens vs currents at θ ≤ 30° 0.97 / 0.16 / 0.079 / 0.038 /
  0.018 %, and 7d with the plane 2.78 / 2.55 / 2.48 / 2.45 / 2.43 % (θ ≤ 30°: 2.68 / 2.39 / 2.32 / 2.30 / 2.28 %),
  with the currents 2.4 % at every Δ ≥ 2 λ. In the cone θ ≤ 30° the plane converges and the L ×1.2 change of
  ≈ 2.3 % remains for both sources (the condition of the clause met there); the hemisphere is not converged
  (grazing band −44 % at Δ = 6 λ).
- **Plane evaluation kept direct:** the plane fields are the direct O(N_H · N_tri) evaluation of
  `exterior_scattered_field` (OpenMP over the plane points), not the compressed near-field operator foreseen
  by decision 2 ("Fast path") and the "Revisit" consequence: 0.3 s at 729 points (7f), 1.5 s at 2 025 (U), 12.5 s at 7 225 points (the Δ = 6 λ
  wide-gap plane, 2N = 62 016; 10.8–21.6 s across the runs of that plane depending on the machine load), against
  assembly and solve times of 10³ s on the same cases. The compressed path stays a revisit item if planes of
  ≫ 10⁴ points become the default.
- **Decision taken (lead, 2026-10-25, section "Decision 2026-10-25" below):** the L dependence is accepted as
  physical on the wide-gap evidence (θ ≤ 30° cone), `FarFieldSource::Currents` stays the default, the gap ≥ 6 λ
  rule is confirmed by WP-PML6b (window ×(1 ± 0.3) on the wide plane, a second seed); the plane over the PML is
  not pursued. Rough-metal PML results state L, gap and window.

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

## Amendment 2026-10-25: extended Huygens plane (lead decision 2026-10-22, option (A))

**Finding of WP-PML6 (first measurement, `benchmarks/results/pml_huygens.md`).** 7f is met (flat Ag / Si: Huygens vs
windowed currents 2.7·10⁻⁴ / 2.8·10⁻⁴, ΔR ≤ 4·10⁻⁷). 7g is not met *and not in the sense of decision 3*: on rough Ag
7d with the Huygens far field is 2.8 % (currents 2.6 %), but the sensitivities are large (z_H ×1.5 1.05 %, window
×0.7 / ×1.3 1.75 % / 0.66 %, Huygens vs currents 1.4 %), and 7e with the Huygens far field loses power (ΔR −5·10⁻⁴
against 4·10⁻⁵). Cause: with the plane window identical to the currents' window (raised-cosine flank Δ = gap = λ
over the real part) the flank cuts through the propagating field crossing the plane at z_H — at z_H = λ radiation
under θ crosses the plane only ≈ z_H tan θ beside its source — and through the bound SPP field, which is still
≈ e⁻² at z_H ≈ λ; a truncated bound field radiates like an edge. The point-source model gives an edge error
≈ π²/(2(kΔ)²) per flank (unit test: 4.7·10⁻² / 9.8·10⁻³ / 2.3·10⁻³ at Δ = 2 / 4 / 8 λ). The clause "if 7g fails"
of decision 3 therefore does not apply: the L dependence cannot be called physical from these runs.

**Decision (option (A)).**
1. The Huygens plane at z = z_H covers the **whole real part** (patch + taper + gap) with window W ≡ 1 and is
   **extended over the PML zone and beyond**; the field above the PML zone is evaluated from the stretched sources
   with the analytically continued kernels (complex distance, Piola currents — `post::exterior_scattered_field`,
   tested: a dipole field continued through the PML to 6·10⁻³).
2. The raised-cosine flank 1 → 0 lies **entirely outside pml_start**: no part of the window cuts through the field
   over the real part. Over the PML the stretch damps the outgoing and the bound field, so the truncation there is
   harmless.
3. Flank width Δ_H from the point-source error model (≈ 3–4 λ for < 0.1 %): Δ_H = 2, 3, 4 λ are measured and
   reported, then the default is fixed *before* 7g is asserted (a discretisation rule, not a fit to 7g). z_H stays
   ≈ λ. The cost of the larger plane (points, time) is recorded next to the results.
4. Gates unchanged (decision 3): 7f stays met; 7g: 7d < 0.5 % with the Huygens far field, z_H ×1.5 and window
   ×(1 ± 0.3) each < 0.1 %, rough Si 7e within the box's own sensitivity. If 7g is met, the Huygens source becomes
   the default of the PML route (decision 2); if not, the result is recorded (result paragraph, docs/05 rows 7f/7g,
   the record), `FarFieldSource::Currents` stays the default and the lead decides — no further far-field variants
   without a lead decision.

**Consequences.** The plane has ≈ ((L + 2 W_pml + 2 Δ_H)/(λ/4))² points instead of ≈ ((L + 2 gap)/(λ/4))²
(U case: ≈ 2·10³ → ≈ 10⁴ points, seconds with the direct evaluation). The stretched-source evaluation of
`exterior_scattered_field` becomes part of the validated far-field path and needs its own sensitivity (z_H ×1.5
covers it). Decision 5 of ADR 0009 is unchanged for flat surfaces and rough Si.

## Decision 2026-10-25 (project lead, coordinator's recommendation): option (A) withdrawn, L dependence accepted as physical

**Why option (A) is withdrawn.** The WP-PML6 continuation and its review (`benchmarks/results/pml_huygens.md`)
showed that the amendment above rested on a wrong premise: the stretched-source evaluation is valid only at
observation points *inside* the stretch start. For a real observation point over the PML zone and a stretched
source between it and pml_start, Im R̃² = 2(x − x′)F(x′) > 0, so the principal R̃ has Im R̃ > 0 and the kernel
grows; near x′ ≈ x the branch cut is crossed. Stretched observation points r̃ = S(r) would give the valid PML field,
but radiating it with real weights is a wavenumber-dependent taper, not the far field, and with the deformed kernel
exp(jk k̂·r̃) dS̃ it is exact only for spectral content with k′ₓ > k sin θ — theory first. Neither variant recovers the
grazing band, whose power is absorbed in the PML and not radiated through a finite plane at z_H (−67 % in
θ = 75–90° at the U case). The only gain would be a smaller window-end error, which the wide-gap measurement
already delivers.

**What the measurements establish.** With a real gap of 3 µm (pml_start 5.375 µm, 2N = 62 016) the plane converges
in the cone θ ≤ 30° for flank widths Δ ≥ 3–4 λ (z_H ×1.5: 0.04–0.07 %, Huygens vs currents 0.04–0.08 %,
window end 3.8·10⁻⁴), and there L ×1.2 still changes the far field by 2.2–2.4 % — for the Huygens plane with its
own and with the fixed window, at z_H ×0.5 and ×1.5, with and without the PML sources, **and for the windowed
currents (2.41 %)**: 20–40× the method's own uncertainty. This is the "if 7g fails" branch of decision 3 with its
condition (small sensitivities) met.

**Decision.**
1. The L dependence of the far field of rough *metal* surfaces on the PML route is accepted as physical
   (re-scattering of the roughness-launched SPPs within the illuminated spot and its surroundings): the model
   statement of ADR 0009 ("a rough spot in a polished surface of size L") is the result, and L, the gap and the
   window are part of every rough-metal result. Row 7g of docs/05 is closed as "not met — physical, documented".
2. `FarFieldSource::Currents` stays the default of the PML route; the Huygens plane remains an option and a
   diagnostic (7f met). Decision 2 of 2026-10-18 (default switch) is void.
3. **Configuration rule for rough metals** (to be validated by WP-PML6b and then written into ADR 0009 and docs/05
   footnote ⁴): the flat gap between the taper and pml_start must be ≥ 6 λ (3 µm at 500 nm) so that the windowed
   current far field is converged in the cone θ ≤ 30° (window-end sensitivity ≤ 1·10⁻³, Huygens vs currents
   ≤ 1·10⁻³); results outside that cone and with narrower gaps are reported as qualitative.
4. **WP-PML6b (confirmation, small):** on the wide-gap plane (U surface with a 3 µm gap and its L ×1.2 partner)
   the window ×(1 ± 0.3) sensitivity of both sources, a second roughness seed, and the 7d value with the windowed
   currents; acceptance: window sensitivities ≤ 1·10⁻³ in the θ ≤ 30° cone, 7d ≥ 10× the sensitivities for both
   seeds (confirming the physical origin), the rule of item 3 recorded with its numbers.
5. No further far-field variants (plane over the PML, mode subtraction, smoother windows) without a new lead
   decision.

**Consequences.** Rough-metal far fields of the PML route are quantitative in the θ ≤ 30° cone with a gap ≥ 6 λ
(≈ 2.7× the unknowns of the λ-gap configuration at the U size; the fast path makes this affordable: assembly
0.3× dense, 886–904 GMRES iterations at 2N = 62 016). Speckle statistics of rough metals must state L and be read
as those of a spot of size L. The open questions of ADR 0009 (point 4, box for Si) are unaffected.
