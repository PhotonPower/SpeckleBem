# Ag absorbed power at oblique incidence (WP-A1, follow-up of WP22c)

Finding under investigation (`benchmarks/results/fresnel_flat.md`): on the flat Ag patch (ε_r = −9.794 −
0.313j, λ = 500 nm, h = 50 nm, L = 4 µm, w₀ = 0.8 µm, ADR 0006 box, rigorous beam, MLFMM d₀ = 3) the
reflectance R is within 0.1 % of Fresnel at 45° s and p, but the absorbed power from the currents,
A_sim = P_abs/P_inc with P_abs = ½ Re ∮ (n̂ × M)·J* dS, was 0.0073 instead of 1 − |r|² = 0.0127 at 45° s
(0.0234 vs 0.0252 at 45° p), while at 0° it closed (A_sim = Fresnel at tol 1e-5).

Executable `benchmarks/ag_absorption.cpp` (`specklebem_ag_absorption`, the fresnel_flat case plus the
diagnostics of `tests/support/ag_absorption_support.hpp`; checks in
`tests/validation/test_ag_absorption.cpp`). Machine: Windows 11, 24 cores, win-release (GCC 16.1,
OpenBLAS), 2026-10-11, shared with other agents; every run ≤ 6 GB peak.

## Verdict

- **The absorption formula and its quadrature are correct**; the walls and the bottom plate absorb
  ≤ 1.3·10⁻⁴ (≤ 2 % of A); the GMRES tolerance is not the cause (iterative refinement to a true residual
  of 1.07·10⁻⁶ leaves A unchanged). At h = 50 nm the local product is limited to **~1–3 % even without
  the spurious component**: the RWG interpolant of the analytic currents gives −2.7 % of A on the top
  face and the L = 3 µm solution −2.1 %. h = 50 nm is sufficient for R and A_ext, not for a sub-percent
  local A.
- **The absorbed power obtained from the far field and the extinction of the same currents is correct**:
  A_ext = (P_ext − P_refl − P_fwd)/P_inc = 0.012716 against A_beam = 0.012693 (+0.18 %) in the very case
  where the local product gives 0.00726. R and A_ext close the energy balance; only the local product
  ½ Re (n̂ × M)·J* of the solved currents is wrong.
- **Cause**: a few-milliradian phase error of M relative to J (+4.0 mrad at the patch centre, +6.2 mrad
  over the whole top face), carried by a spurious **standing-wave component** of the currents with the
  tangential wavenumbers ±k₀ sin θ (error peaks of equal height at +0.70 k₀ and at the mirrored
  −0.70 k₀, of order 1 % of the specular current; inferred, see Diagnostics), which appears **only at
  L = 4 µm** (L = 2 and 3 µm at the same h, d₀, tolerance and L/w₀ are within −0.1 % and −2.1 %).
  Because Re Z_s/|Z_s| = 1.6 % for Ag, 1 mrad changes A by ≈ 6 %.
- **Why R and A_ext miss it, and what that implies (leading hypothesis)**: P_ext, P_refl and P_fwd
  depend only on the field the currents radiate into R1 (P_ext is the overlap of that field with the
  incident beam, P_refl and P_fwd are far-field powers). A current component (δJ, δM) = (n̂ × δH,
  −n̂ × δE) that is the trace of a source-free **vacuum** field (δE, δH) inside the box radiates nothing
  into R1 (extinction theorem; R1 is vacuum too), so A_ext and R are **exactly blind** to it, whatever
  its amplitude. The local product is not: with the true currents the trace of the Ag interior field
  (E, H), the divergence theorem over the box volume V gives for the cross term
  −½ Re ∮ (δE × H* + E × δH*)·n̂ dS = −½ Re[jωε₀ (ε_r* − 1) ∫_V E*·δE dV] (the magnetic terms cancel),
  which vanishes for a vacuum interior and is weighted by |ε_r − 1| ≈ 11 for Ag. Together with the
  1/(Re Z_s/|Z_s|) ≈ 60 sensitivity of A to the phase of M, the observations (A_local −43 %, A_ext and R
  within 0.2 %) therefore imply that the spurious component is (nearly) **non-radiating — in the null
  space of the exterior operator**: an interior-resonance-like near-null mode of the discrete
  PMCHWT-family system at high contrast, constrained only through the Ag (R2) operator. This is the
  leading hypothesis; it is not yet tested directly (E1 below).
- **Origin of the excitation: open.** The solution satisfies the exact Galerkin rows at the centre to
  2.9·10⁻⁵, and the MLFMM operator error on the analytic currents is broadband 4.4·10⁻⁵ with no mirrored
  (−k₀ sin θ) content. These row and strip checks show that a near-null mode exists (a percent-level
  component with a residual footprint ≤ 3·10⁻⁵); they do **not** decide between excitation by the
  discretisation error (intrinsic to the discrete L = 4 µm system) and excitation by the MLFMM error:
  a near-null mode amplifies the projection of any broadband error onto it, and the gain needed is only
  a few hundred (≈ 10⁻²/(3–4·10⁻⁵) ≈ 250–350, ≤ 700 if the component is 2 %). A dense or d₀ ≥ 4
  reference at L = 4 µm (≥ 17 GB setup, 77 GB dense) was outside this work package's 10 GB budget; see
  the proposed follow-up experiments.
- **Consequence**: the local surface absorption ½ Re ∮ (n̂ × M)·J* is **not a reliable observable for Ag**
  (it amplifies mrad-level current errors by 1/(Re Z_s/|Z_s|) ≈ 60); use **A_ext = P_ext − P_sca** (far
  field + extinction integral) for the absorbed power of weakly lossy metals, and the local value only as
  a sensitive consistency probe. No library bug was found; no library function was changed.

## Diagnostics

All at 45° s, h = 50 nm, ICTF + left Jacobi, MLFMM, L/w₀ = 5 (w₀ = L/5 as WP22c), tolerance 1e-5 unless
stated. A_beam = 1 − R_beam − (share of k_z ≤ 0 components) from the infinite-interface beam
reflectance; A_patch = reference absorbed flux through the L × L patch.

| Case | 2N | it | R_sim/R_beam − 1 | A_sim (local) | A_ext | A_beam | top / walls / bottom | dφ(M − J) centre / top [mrad] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| L = 2 µm, w₀ = 0.4, d₀ = 3 | 22 800 | 1 680 | −0.161 % | 0.012651 (−0.1 %) | 0.012709 | 0.012664 | 0.012483 / 1.7e-4 / 4e-7 | −0.24 / +0.09 |
| L = 3 µm, w₀ = 0.6, d₀ = 3 | 43 200 | 2 066 | −0.127 % | 0.012423 (−2.1 %) | 0.012697 | 0.012689 | 0.012395 / 2.8e-5 / 1e-7 | +0.18 / +0.27 |
| L = 3 µm, w₀ = 0.6, **d₀ = 5** | 43 200 | 2 066 | −0.127 % | 0.012423 (−2.1 %) | 0.012697 | 0.012689 | identical to d₀ = 3 (≤ 10⁻⁶) | +0.17 / — |
| **L = 4 µm, w₀ = 0.8, d₀ = 3** (WP22c case) | 69 600 | 2 757 | −0.082 % | **0.007257 (−42.8 %)** | **0.012716** | 0.012693 | 0.007385 / **−1.3e-4** / 5e-7 | **+4.03 / +6.28** |

Iterative refinement of the L = 4 µm case (x ← x + GMRES(Z, b − Z x), same MLFMM operator):

| Stage | true residual | R_sim/R_beam − 1 | A_sim | A_ext | top: A_sim/A_ref − 1 | dφ(M − J) centre / inner / top [mrad] |
|---|---:|---:|---:|---:|---:|---|
| GMRES tol 1e-4 (1 630 it) | 1.20e-4 | −0.089 % | 0.006671 | 0.012785 | −0.465 | +4.51 / +5.85 / +7.01 |
| refinement (1 953 it) | 8.39e-6 | −0.081 % | 0.007355 | 0.012705 | −0.410 | +3.95 / +5.13 / +6.16 |
| refinement (2 194 it) | **1.07e-6** | −0.081 % | **0.007348** | 0.012707 | −0.411 | +3.95 / +5.14 / +6.17 |

The error is converged with respect to the solver: the solution of the discrete (MLFMM) system itself
carries it. (At 0° the WP22c tolerance study did change A, 0.0316 → 0.0183 from 1e-4 to 1e-5: there the
error was solver error; at 45° it is not.)

Per-region detail of the L = 4 µm case (tol 1e-5): c_X = ∫ X_sim·X_ref* / ∫ |X_ref|² against the
analytic interface currents (every plane wave of the beam reflected with its own Fresnel coefficients):

| Region | A_sim/A_ref − 1 | interpolant A/A_ref − 1 | \|c_J\|, arg c_J | \|c_M\|, arg c_M | ‖J − J_ref‖/‖J_ref‖ (interpolant) |
|---|---:|---:|---|---|---|
| centre (ellipse w₀/2 cos θ × w₀/2) | −0.259 | −0.012 | 1.0069, −2.69 mrad | 0.9910, +1.34 mrad | 0.126 (0.124) |
| inner \|x\|, \|y\| ≤ L/4 | −0.342 | −0.025 | 1.0081, −3.00 mrad | 0.9890, +2.22 mrad | 0.129 (0.127) |
| whole top face | −0.418 | −0.027 | 1.0097, −3.80 mrad | 0.9872, +2.48 mrad | 0.132 (0.128) |

The same table at L = 3 µm: centre −0.012 (interpolant −0.006), |c_J| = 0.9992, |c_M| = 0.9995,
arg c_M − arg c_J = +0.18 mrad. The pointwise RWG error (12–13 %) is the first-order RT0 interpolation
error and is the same for the solution and the interpolant; it averages out in integrated quantities.

Spatial spectrum along x of the current error X_sim − X_ref on the strip |y| ≤ L/8 (top-triangle
centroids, `--spectrum`): at L = 4 µm |M_err| peaks at k_x = +0.70 k₀ (1.01·10⁻¹⁴, the specular current)
**and at −0.70 k₀ (1.04·10⁻¹⁴)**, where the reference has nothing; the broadband floor (RT0 pointwise
error) is ≈ 1–2·10⁻¹⁵. At L = 3 µm neither peak is present (floor only). These magnitudes are **raw**
DFT values: area-weighted sums Σ_t A_t X_err(c_t) e^{j k_x x_t} over the strip [V·m for M], with a
**rectangular window** (the strip is cut at x = ±L/2; the resolution is 2π/L = 0.125 k₀ at L = 4 µm, so
the ±0.70 k₀ peaks, 1.4 k₀ apart, are well separated, but part of the floor is leakage). The reference
spectrum was not printed in these runs, so the relative size of the mirrored peak is only inferred: if
the +0.70 k₀ error peak is the amplitude deviation of the specular current (|c_M| − 1 = −1.3 % on the top
face), the mirrored peak of equal height is ≈ 1 % of the specular reference spectrum. `--spectrum` now
also prints the J and M error spectra relative to the reference spectrum at +k₀ sin θ and accepts
`--spectrum-window hann` (not re-run). In the strip profiles (`--profile`) the phase error at L = 4 µm
oscillates along x and grows smoothly towards |y| → L/2; the 16 bins of 250 nm do not resolve the beat of
a ±k₀ sin θ pair (period λ/(2 sin θ) = 354 nm, aliased), so no period is quoted. At L = 3 µm it varies
smoothly and antisymmetrically like the interpolant's.

Operator checks (`--row-check`, `--strip-check`; exact rows from `op::assemble_sparse`): at L = 4 µm the
MLFMM matvec on the solution differs from the exact rows of the 148 central basis functions by 2.1·10⁻⁵
(J rows) / 2.9·10⁻⁵ (M rows) and the exact residual there is 2.9·10⁻⁵ (L = 3 µm: 1.1·10⁻⁵ / 6.8·10⁻⁶,
exact residual 8.3·10⁻⁶; d₀ = 5: 8·10⁻⁷ / 1.1·10⁻⁶). On the analytic (pure +k sin θ) currents along the
strip |y| ≤ 50 nm (558 basis functions) the MLFMM error is 4.4·10⁻⁵ relative and spectrally flat
(1.3·10⁻⁶ of the specular peak at −0.70 k₀).

## Findings per hypothesis

| Hypothesis | Test | Result |
|---|---|---|
| (a) formula / quadrature / sign / ½ / normal | Dunavant degree 2 vs 6 per part; independent A_ext = P_ext − P_sca (the extinction integral equals the optical theorem to 10⁻¹⁰ for arbitrary currents, unit test); conventions checked against the transmitted-wave impedance (unit test) | **correct**: degree 2 = degree 6 to all digits; A_ext = A_beam + 0.18 % in the failing case |
| (b) closing box, power past the edges | per-part split; A_patch | walls −1.3·10⁻⁴ (L = 4) / +2.8·10⁻⁵ (L = 3) / +1.7·10⁻⁴ (L = 2), bottom ≤ 5·10⁻⁷; A_patch = A_beam within 0.1 %: **not the cause** (the negative wall absorption at L = 4 is a symptom of the same spurious currents) |
| (c) discretisation at the skin-depth scale | RWG interpolant of the analytic currents; L = 2 / 3 µm at the same h | interpolant −1.2 … −2.7 %; L = 2, 3 µm at h = 50 nm −0.1 / −2.1 %: the L = 4 µm error (−43 %) is not a local resolution limit, but h = 50 nm limits the local product to **~1–3 %** even without the spurious component (sufficient for R and A_ext, not for a sub-percent local A) |
| (d) formulation, Jacobi, solver tolerance | refinement to true residual 1.07·10⁻⁶; ICTF / MCTF are row scalings of PMCHWT (docs/03: identical currents with a direct solver, identical Jacobi-preconditioned systems) | **not the cause at 45°** (A unchanged from 8·10⁻⁶ to 1·10⁻⁶ true residual) |
| MLFMM accuracy | d₀ = 3 vs 5 at L = 3 µm; exact-row checks at L = 4 µm | d₀ irrelevant at L = 3 µm; at L = 4 µm the operator error is 2–4·10⁻⁵ and has no mirrored content. This shows that a near-null mode exists, but does not exclude MLFMM error as its excitation (a gain of a few hundred suffices, see Verdict): **origin open**; d₀ ≥ 4 at L = 4 µm (500 nm leaves, near field 7.3 GB, setup 17.6 GB) **not run** (budget) |

## Recommendations

1. Report the absorbed power of Ag (and other weakly lossy metals, Re Z_s ≪ |Z_s|) as
   **A_ext = (P_ext − P_sca)/P_inc**, with P_ext = ½ Re ∮ (E_inc·J* + M·H_inc*) dS and P_sca from the far
   field of both hemispheres (`ag_absorption::extinction_power`, the study's `DIAG extinction` line). The
   local ½ Re ∮ (n̂ × M)·J* is a sensitive probe of mrad-level current errors, not an observable; the WP22c
   energy-balance diagnostic 1 − R − A_local is therefore not a closure check for Ag at oblique incidence.
   **Accuracy requirement:** A_ext is a difference of terms ≈ 160× larger than A (P_ext ≈ P_refl + P_fwd
   ≈ 2 P_inc: the reflected beam and the forward shadow lobe), so a 0.1 % accuracy of A needs P_ext and
   the far-field powers to ~10⁻⁵ relative: the forward lobe must be resolved by the hemisphere grid
   (`ff_dtheta`, the cone refinement around the lobe) and the source quadrature (`ff_degree`) must be
   converged at that level. The recorded +0.18 % used the WP22c settings (ff_dtheta 1°, ff_degree 6,
   0.1 rad cone) and was not checked for convergence at 10⁻⁵.
2. A library function `post::absorbed_power` is **not** proposed now: the local formula is the one that
   fails, and the robust variant needs the incident field and both far-field hemispheres; once a
   quantitative absorption observable is needed (Phase 5), add `post::extinction_power` (beam / plane wave)
   next to `far_field` with the test of `tests/validation/test_ag_absorption.cpp`, and with the 10⁻⁵
   far-field accuracy requirement of recommendation 1 as part of its tests (forward-lobe grid and
   quadrature convergence).
3. Open: test the non-radiating near-null-mode hypothesis and decide whether its excitation is
   intrinsic (discretisation error) or MLFMM error, with the experiments below (cheapest first).

## Proposed follow-up experiments

Not run (WP-A1 budget; to be scheduled as a follow-up work package). Memory estimates from the recorded
runs (every recorded run ≤ 6 GB peak; L = 4 µm, d₀ = 3: 2N = 69 600, Krylov basis 2 757 × 2N × 16 B ≈
3.1 GB).

| | Experiment | Decides | Memory |
|---|---|---|---|
| E1 | Far field of the saved L = 4 µm currents near the retro direction (−k₀ sin θ tangential, the direction a radiating ∓k component would feed) and the **extinction-theorem residual inside the box** (E_inc plus the R1-side representation of the currents at interior points, which must vanish; a non-radiating component shows up there as its own vacuum field), L = 3 vs 4 µm | non-radiating hypothesis directly: retro lobe at the edge-diffraction level of L = 3 µm and an interior residual with ±k₀ sin θ structure at L = 4 µm only | `--load-currents`, no assembly: < 1 GB |
| E2 | Re-solve with a perturbed right-hand side b + δb, δb broadband random (fixed seed) with ‖δb‖/‖b‖ = 3·10⁻⁵ (the size of the MLFMM and exact-row errors), L = 3 and 4 µm; normalised `--spectrum` of the change of the solution | amplification of broadband errors into the ±k₀ sin θ mode (gain of a few hundred at L = 4 µm, ≈ 1 at L = 3 µm, if the mode is the mechanism) | as the solve: ≤ 6 GB per case |
| E3 | Harmonic Ritz values from the GMRES Hessenberg matrix (needs the Hessenberg exposed by `solver::gmres` or a copy in the study) at L = 3 and 4 µm | an isolated small eigenvalue of the (Jacobi-preconditioned) system at L = 4 µm only, and its Ritz vector's spectrum | solve + Hessenberg (2 757² × 16 B ≈ 0.12 GB): ≤ 6 GB |
| E4 | L sweep 2.0–3.5 µm (step 0.25 µm, w₀ = L/5) and a box-depth sweep at L = 3 µm (needs a `--box-depth` option; the depth is `default_box_depth` today), logging A_local − A_ext and the normalised ±k₀ sin θ peak; d₀ = 3 vs 5 at the L where the mode appears | resonance-like dependence on the box dimensions (a mode of the box, not of the patch alone); with d₀ = 5 at that L, intrinsic vs MLFMM-excited | d₀ = 3: ≤ 6 GB per case; d₀ = 5 at L ≤ 3.5 µm as the recorded L = 3 µm d₀ = 5 run (≤ 6 GB) |
| E5 | L = 4 µm, 45° s with d₀ = 5, only if E2–E4 leave the excitation open | intrinsic (mode unchanged) vs MLFMM-excited (mode gone or reduced by the operator-error ratio) at the recorded case | setup 17.6 GB (near field 7.3 GB) + Krylov ≈ 3.1 GB ≈ 21 GB |

## Reproduce

All recorded cases use w₀ = L/5 at 45°, above the oblique-incidence limit w₀ ≤ L cos θ/4 that
`check_beam_waist` enforces since WP-B2 (ADR 0006 amendment 2026-10-11), so they need
`--allow-wide-beam` (as `benchmarks/fresnel_flat.cpp`; edge loss of these cases 0.06–0.1 %):

```
specklebem_ag_absorption --material ag --L 2e-6 --w0 0.4e-6 --theta 45 --pol s --allow-wide-beam   # tol 1e-5 default
specklebem_ag_absorption --material ag --L 3e-6 --w0 0.6e-6 --theta 45 --pol s --allow-wide-beam \
    [--d0 5] --row-check 0.15e-6
specklebem_ag_absorption --material ag --L 4e-6 --w0 0.8e-6 --theta 45 --pol s --allow-wide-beam \
    --max-iter 3500 --row-check 0.2e-6 --save-currents x_L4.bin
specklebem_ag_absorption --material ag --L 4e-6 --w0 0.8e-6 --theta 45 --pol s --allow-wide-beam \
    --tol 1e-4 --refine 1e-5,1e-6
specklebem_ag_absorption ... --allow-wide-beam --load-currents x_L4.bin --profile 16 | --spectrum | \
    --strip-check 0.05e-6
```
