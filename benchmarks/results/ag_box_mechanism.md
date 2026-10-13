# Ag box sensitivity: surface plasmons or box cavity resonances? (WP-V3, ADR 0006)

Follow-up of `box_validity_mlfmm.md` (WP-V2), section 2: a truncated rough Ag patch (λ = 500 nm,
ε_r = −9.794 − 0.313j) fails both docs/05 box checks at L = 4 µm (depth ×2: 0.8–1.9 %, L ×1.2:
4.5–8.2 %), and WP-V2 could not tell whether surface plasmons (SPPs) or the weakly damped cavity
resonances of the closed box (vacuum inside, Ag walls; WP-A2, `ag_absorption.md`) cause it. This
record runs the four discriminating experiments proposed there ((i)–(iv)), the MLFMM test of the
WP22b2 PMCHWT plateaus ((v)) and three additional checks, decides the mechanism and proposes
remedies for the coordinator's ADR.

Labels: **measured** = a number from a run of this record or a committed earlier record;
**estimated** = derived by a stated calculation; **hypothesis** = an interpretation not tested
directly.

## Short answer

- **Mechanism: surface waves on the closed Ag box, launched by the roughness under the beam, most
  likely surface plasmons (SPPs); a physical property of the truncated Ag block, not a numerical
  artefact. The box cavity resonances are present in the currents but do not cause the far-field
  box sensitivity of the rough box** (in the flat box one depth, D = 2.0 µm, is resonant and that
  resonance does show in the far field, see the flat-box bullet). Confidence: high that the depth
  sensitivity is radiated by the walls, the foot and the bottom (not by the top face), that it is
  reproduced in a second box discretisation (magnitude 1.24 vs 1.27 %, correlation 0.93) and that
  the cavity resonances do not drive it in the rough box; medium that the wall waves are SPPs
  rather than grazing light (the wall spectra favour SPPs, the depth-phase fit leans towards
  k_w ≈ k₀; both have |k| ≈ k₀ … 1.055 k₀). For L ×1.2 two contributions are not separated
  (**hypothesis** (a): SPPs scattered or reflected at the rim, a truncation artefact; alternative
  (b): real radiation of the SPPs scattered by the added rough annulus, which the semi-infinite
  surface also has).
- (i) **measured**: the currents on the walls and the bottom plate are dominated by SPP-like waves
  (66–80 % of the spectral power above k₀: 4 × 4 µm faces 75–80 %, the 4 × 2 µm walls at D = 2 µm
  72 %, flat 66 %; centroid 1.06–1.08 k₀ against Re k_spp = 1.055 k₀). Windowed plane-wave
  references (**estimated**): in a 4 × 4 µm window an SPP gives 77–81 % above k₀ and a k₀ wave
  50–56 %; in the 4 × 2 µm wall window an SPP gives 66–83 % and a k₀ wave 51–59 %, depending on the
  direction. The walls at D = 2 µm are thus closer to an SPP than to a k₀ wave, but less clearly
  than the 4 × 4 µm faces. The part of the top-face current that changes with the depth (1.6–2.2 %
  of it) is ≈ 60 % SPP-like (53–57 % above k₀), the part that changes with L is SPP-like (76–79 %
  above k₀). A cavity mode would sit below k₀ (a 0.7 k₀ wave: 0 % above k₀).
- (ii) **measured**: the extinction-theorem residual inside the box is 0.6–1.0 % at L = 4 µm (1.4–1.5 %
  at 4.8 µm; flat box 0.04–0.24 %; Im ε = −6: 0.15–0.28 %), i.e. a moderate cavity component exists
  (WP-A2 levels), but it is **not depth-resonant** (0.76–1.14 % on the top line over the ten depths
  D = 2.0 … 2.4 and 4.0 µm, without relation to the far-field change) and not enhanced at the bottom
  plate, while the far field changes by up to 1.17 % between neighbouring depths.
- (iii) **measured**: the far-field change does not switch on and off with D. Against D = 2.0 µm it
  is 0.70 / 1.17 / 1.14 / 0.76 / 0.40 / 0.72 / 1.10 / 1.06 % for D = 2.05 … 2.40 µm (steps of
  0.05 µm) and 0.82 % for 4.0 µm: a smooth oscillation with a period of ≈ 0.25 µm, every small step
  as large as a doubling of the depth; no depth is an odd one out (against D = 4.0 µm every depth of
  the sweep differs by 0.74–0.82 %). One travelling wave F(D) = F_∞ + A e^(−jβD) explains 90–99 %
  of the depth-dependent far field in every 10° band. Read as β(θ) = k_w + k₀ cos θ, the per-band
  k_w falls from 1.074 k₀ (0–10°) to 0.958 k₀ (80–90°) (mean 1.005 ± 0.045 k₀); a free fit
  β = a + b k₀ cos θ gives a = 0.93 k₀, b = 1.11 (**estimated**). A wave running down the walls at
  ≈ k₀ and radiated from depth D; the fit leans towards k_w ≈ k₀ rather than Re k_spp = 1.055 k₀.
- (iv) **measured**: the band of small harmonic Ritz values (|θ| = 0.02–0.1 at k = 1 000) is the same
  within ~15 % for all ten depths D = 2.0 … 4.0 µm and L = 4.8 µm (the flat box has a similar
  band); it does not move with D (as box cavity modes would). With Im ε = −6, GMRES converges
  (k = 341 / 364) before such a band forms; at equal k = 250 the values are similar (0.10 vs
  0.087); whether the lossy spectrum has the band is not tested. The flat runs show a D-dependent
  smallest Ritz value (σ_min 2.26e-3 at D = 2.0 µm vs 9.95e-3 at 2.1 µm, at their converged
  k = 797 vs 741, i.e. not at equal k), consistent with the flat-box resonance of (ii).
- (v) **measured**: PMCHWT at d₀ = 5 reproduces the d₀ = 3 residual history of WP22b2 to 0.15 % at
  every iteration (2 126 vs 2 127 iterations, the same three plateaus). **The MLFMM error does not
  cause the PMCHWT staircase.** The box-cavity hypothesis for the PMCHWT plateaus is not tested for
  PMCHWT; the ICTF + Jacobi near-origin band of (iv) does not move with D.
- Additional checks (**measured**): the far-field difference pattern F(4 µm) − F(2 µm) is the same in
  the graded 100 nm and the uniform 50 nm box (complex correlation 0.93) and between tolerances
  1e-4 and 1e-5 (0.9999), while numerical noise patterns correlate ≤ 0.16; the loss diagnostic
  removes the SPP part of the depth-dependent top current (11 % above k₀) and cuts it 11×.
- **Flat box** (same rim illumination, no roughness; **measured**): 6.5× weaker wall currents, 55×
  less wall radiation. Its cavity component switches with D (residual 0.19 % at 2.0 µm, 0.044 % at
  2.1 µm, 0.054 % at 4.0 µm), and the far field follows: D = 2.1 and 4.0 µm agree to 0.034 %, both
  differ from D = 2.0 µm by 0.24 %. D = 2.0 µm is the odd one out, so in the flat box the depth ×2
  change measured from 2.0 µm (0.24 %) is that resonance, and away from it the flat-box depth effect
  is 0.034 % (< 0.1 %). The rough box has no such odd depth ((iii)); its depth effect (0.4–1.2 %)
  is 1.7–5× the resonant flat value and 12–35× the non-resonant one. The first version of this
  record read the flat case as "the resonance shows in the non-radiating residual, not in the far
  field"; the comparison 2.1 vs 4.0 µm (review round) shows that this is wrong.
- **Remedy (input for an ADR, section below)**: the box sensitivity is physics of the finite block
  (SPP decay length 41.5 µm in amplitude, 10× the patch), so neither a resonance-free formulation
  nor a tighter solver or MLFMM setting removes it. Because SPPs launched under the beam also
  scatter into real radiation off roughness far outside a small patch, the docs/05 target
  "approximate the semi-infinite surface" cannot be reached with any local termination at
  L ≪ L_spp. Recommended: an SPP-absorbing termination (graded lossy impedance frame on the outer
  top face plus walls and bottom, or a PML by complex coordinate stretching at the boundary of the
  BEM domain), prototyped in its own WP with an ADR; it defines a modified, well-posed problem
  (rough patch in SPP-absorbing surroundings), not an approximation of the semi-infinite rough
  surface, and the ADR must say what the frame model represents. Until then, Ag rough far fields
  carry a stated box uncertainty. Depth averaging (measured here) reduces the depth effect only
  about 3× (to 0.23–0.30 % between disjoint groups of 2–5 depths) and does not touch the edge
  effect.

## Setup

| Parameter | Value |
|---|---|
| Configuration | WP-V2 Ag uniform-box configuration: Gaussian height map σ = 50 nm, Lc = 500 nm, h = 50 nm, seed 1, generated on 4.8 µm and cropped to L (`--L-gen 4.8e-6`), uniform box (box cells = h), depth D as stated (default 2 µm) |
| Beam | WP-V2 rigorous angular-spectrum beam (`--beam spectrum`, grid check ≤ 1.6e-14), normal incidence, E along x, w₀ = 1 µm (L/4 at L = 4 µm) |
| Operator / solver | MLFMM d₀ = 3 (global leaf rule), ICTF + left Jacobi (formulation::recommend), full GMRES, **tol 1e-5** (ADR 0006 for Ag), max 4 000 iterations; the depth-2 µm run reproduces WP-V2's `ag4_d2_uni_t5` bit for bit (2 096 iterations, true residual 1.077e-5, P_refl 1.539014e-12) |
| Loss diagnostic | Im ε_r = −6 (`--eps-imag -6`, DIAGNOSTIC lines), as in WP-V2 |
| Flat reference | σ = 0 (`--sigma 0`), otherwise identical |
| Diagnostics | `benchmarks/box_validity.cpp` `--mechanism --ritz --save-currents`, `--load-currents … --diff`, `--depth-fit`; helpers in `tests/support/ag_box_mechanism_support.hpp` (unit-tested, `tests/unit/test_ag_box_mechanism.cpp`) |
| Machine | Windows 11, 12 cores / 24 threads, 128 GB, win-release (GCC 16, OpenBLAS), quiet machine (this study only), 2026-10-13 |

Re k_spp / k₀ = Re √(ε/(1+ε)) = 1.0553 (**estimated**, flat interface), amplitude decay length
1/|Im k_spp| = 41.5 µm, i.e. ≈ 10 L at L = 4 µm; with Im ε = −6 it is 3.1 µm.

## Runs

| Run | L, D [µm] | 2N | It. | Solve [s] | P_refl [W] | PARTS walls / bottom | docs/05 l2_refl vs reference |
|---|---|---:|---:|---:|---:|---|---|
| `ag4_d20` | 4, 2.0 | 153 600 | 2 096 | 1 408 | 1.539014e-12 | 7.48e-4 / 2.43e-4 | (reference) |
| `ag4_d205` | 4, 2.05 | 155 520 | 1 947 | 1 073 | 1.539135e-12 | 7.35e-4 / 2.28e-4 | 0.70 % vs 2.0 |
| `ag4_d21` | 4, 2.1 | 157 440 | 1 918 | 1 061 | 1.539182e-12 | 7.17e-4 / 2.37e-4 | 1.17 % vs 2.0 |
| `ag4_d22` | 4, 2.2 | 161 280 | 2 272 | 1 513 | 1.538864e-12 | 6.87e-4 / 2.20e-4 | 0.76 % vs 2.0, 1.15 % vs 2.1 |
| `ag4_d40` | 4, 4.0 | 230 400 | 2 287 | 1 930 | 1.538896e-12 | 7.01e-4 / 1.93e-4 | **0.82 %** vs 2.0 (WP-V2: 0.82 % at tol 1e-4) |
| `ag48_d20` | 4.8, 2.0 | 202 752 | 2 102 | 1 547 | 1.536517e-12 | 9.83e-4 / 4.02e-4 | **4.52 %** vs `ag4_d20` (WP-V2 graded: 4.50 %) |
| `ag4_d20_lossy6` | 4, 2.0, Im ε = −6 | 153 600 | 341 | 110 | 1.158868e-12 | 8.0e-5 / 6.1e-6 | (reference) |
| `ag4_d40_lossy6` | 4, 4.0, Im ε = −6 | 230 400 | 364 | 155 | 1.158862e-12 | 7.7e-5 / 2.6e-6 | 0.092 % vs lossy 2.0 (WP-V2 at tol 1e-4: 0.093 %) |
| `ag4_flat_d20` | 4, 2.0, σ = 0 | 153 600 | 797 | 340 | 1.564524e-12 | 1.35e-5 / 8.9e-6 | (reference) |
| `ag4_flat_d21` | 4, 2.1, σ = 0 | 157 440 | 741 | 245 | 1.564755e-12 | 1.70e-5 / 6.0e-6 | 0.24 % vs flat 2.0 |
| `ag4_flat_d40` | 4, 4.0, σ = 0 | 230 400 | 1 030 | 548 | 1.564654e-12 | 1.58e-5 / 5.4e-6 | **0.24 %** vs flat 2.0; **0.034 %** vs flat 2.1 |

The fine-sweep runs D = 2.05 … 2.4 µm are listed in (iii). In all 18 solves, one at a time: 16 box
runs (the table plus `ag4_d215`, `ag4_d225`, `ag4_d23`, `ag4_d235`, `ag4_d24`) and the two (v)
runs; summed GMRES solve time 4.5 h (box) + 0.5 h ((v)) ≈ 5.0 h (RESULT lines of `wp_v3_logs/`), ≈ 6 h
wall clock for the box runs with assembly, far fields and the `--mechanism` diagnostics. (The first
version of this record said "21 box runs, ≈ 9.5 h of solves"; the logs hold one solve per tag, so
the count was wrong.) The SETUP lines of the committed logs predate the `sigma` field (added in the review round);
σ is in the RESULT-line tags (`ag4_flat_*`: σ = 0) and in the stored attribute `rms_roughness`.

Raw lines: `ag_box_mechanism_rows.txt` (RESULT, PARTS, MECH, RITZ, DIFF), spectra
`ag_box_mechanism_spectra.txt`, interior residual `ag_box_mechanism_interior.txt`, harmonic Ritz
values `ag_box_mechanism_ritz.txt`, logs `wp_v3_logs/`.

## (v) MLFMM error and the PMCHWT plateaus (WP22b2)

Ag, L = 3.2 µm, uniform 50 nm box, rigorous beam w₀ = L/4, PMCHWT without preconditioner,
tol 1e-3, `benchmarks/mlfmm_scaling.cpp --digits 5`, compared with the committed d₀ = 3 history
(`formulation_convergence_mlfmm_histories_full.txt`) by `benchmarks/ag_box_mechanism_histories.py`.

| | d₀ = 3 (WP22b2) | d₀ = 5 (this record) |
|---|---:|---:|
| iterations to 1e-3 | 2 127 | 2 126 |
| residual at it. 100 / 400 / 800 / 1 200 / 1 600 / 2 000 | 1.471e-2 / 2.442e-3 / 2.355e-3 / 1.478e-3 / 1.281e-3 / 1.102e-3 | 1.469e-2 / 2.442e-3 / 2.355e-3 / 1.478e-3 / 1.281e-3 / 1.102e-3 |
| ratio d₀ 5 / 3 over all iterations | | 0.99852 … 1.00004 |
| plateaus (< 3 % drop per 100 it.) | 390–870, 1 095–1 527, 1 676–2 071 (1 307 it.) | 390–870, 1 095–1 526, 1 676–2 070 (1 305 it.) |
| leaf, near field, peak | 200 nm, 2.8 GB, 7.7 GB | 400 nm, 12.5 GB, 18.3 GB |
| solve time | 829 s | 1 654 s |

**Verdict (measured): the plateaus are a property of the PMCHWT system, not of the MLFMM error.**
Reducing the operator error (d₀ 3 → 5: block check of the finest far level 5.0e-4 at d₀ = 5,
a = 0.8 λ, against 5.2e-3 at d₀ = 3, a = 0.4 λ, and 1.2e-3 at d₀ = 3, a = 0.8 λ; i.e. 2–10×,
**estimated** from the block checks in `wp_v3_logs/ag32_pmchwt_d05.log` and
`wp22b2_logs/ag32_pmchwt.log`; the nominal 10^−d₀ targets differ by two orders, the first version of
this record quoted that as the error reduction) changes the residual history by at most 0.15 %. The second
hypothesis of WP22b2 (box cavity resonances) is **not tested for PMCHWT** by this record: the
ICTF + Jacobi near-origin band of (iv) does not move with D, and the ICTF depth sweep shows no
resonant switching (iii), but no PMCHWT run at another depth was made. What causes the PMCHWT
staircase stays open (**hypothesis**: the unbalanced J/M block scaling of unpreconditioned PMCHWT
for |ε| ≈ 10, which Jacobi removes, combined with the near-origin band of (iv)).

ICTF (no preconditioner) at d₀ = 5 (optional run of the brief): 347 iterations, as at d₀ = 3
(residual ratio 0.9965 … 1.0002 at every iteration; 187 s, peak 15.1 GB), so the PMCHWT / ICTF
iteration ratio stays 6.13 (2 126 / 347; WP22b2: 2 127 / 347).

## (i) Spatial spectra of the box currents

Area-weighted 2D DFT of J and M at the triangle centroids of each face, separable Hann window over
the face, grid |k_u|, |k_v| ≤ 2.5 k₀ in steps of k₀/32, power binned in 0.025 k₀ annuli
(SPEC lines). Columns: fraction of the power above k₀; within Re k_spp ± 0.05 k₀; in
0.3–0.95 k₀ (propagating, off-specular); power-weighted centroid in 0.8–1.4 k₀.

Window response (**estimated**, single plane waves in the Hann window of each face, same grid,
direction φ = 0–90° in 15° steps; `benchmarks/ag_box_mechanism_window.py`,
`wp_v3_logs/window_references.log`; the first version of this record gave the 4 × 4 µm rows only,
with 48–52 % in the k_spp ± 0.05 column for the SPP):

| Window | wave | > k₀ | at k_spp ± 0.05 | 0.3–0.95 k₀ | centroid [k₀] |
|---|---|---|---|---|---|
| 4 × 4 µm (top, bottom, walls at D = 4 µm) | SPP 1.055 k₀ | 77–81 % | 47–58 % | 5.5–7.8 % | 1.058 |
| | k₀ | 50–56 % | 40–51 % | 20–26 % | 1.003 |
| | 0.95 k₀ | 25–30 % | 23–31 % | 44–51 % | 0.956 |
| | 0.7 k₀ | 0 % | 0 % | 100 % | 0.83 |
| 4 × 2 µm (walls at D = 2 µm; u along, v down the wall) | SPP 1.055 k₀ | 66–83 % (83 % along, 67 % down the wall) | 27–53 % | 6–25 % | 1.064–1.068 |
| | k₀ | 51–59 % | 26–49 % | 22–38 % | 1.011–1.028 |
| | 0.95 k₀ | 32–40 % | 22–32 % | 45–50 % | 0.964–0.995 |
| | 0.7 k₀ | 0–2 % | 0–2 % | 96–100 % | 0.84–0.89 |

The main-lobe half width is 0.25 k₀ in a 4 µm window and 0.5 k₀ across a 2 µm one, so the centroid,
not the peak bin, locates a ring, and the 4 × 2 µm walls separate an SPP from a k₀ wave less well
(the ranges of the two touch: 66 % vs 59 %).

| Face, run | > k₀ (J / M) | at k_spp ± 0.05 | 0.3–0.95 k₀ | centroid 0.8–1.4 [k₀] |
|---|---|---|---|---|
| top, `ag4_d20` | 3.6 / 4.1 % | 1.9 / 2.3 % | 35 / 34 % | 1.013 / 1.012 |
| walls, `ag4_d20` | 72 / 71 % | 29 / 29 % | 17 / 18 % | 1.080 / 1.077 |
| walls, `ag4_d40` | 80 / 79 % | 49 / 49 % | 6.6 / 6.9 % | 1.064 / 1.062 |
| bottom, `ag4_d20` | 76 / 76 % | 47 / 48 % | 8.6 / 8.7 % | 1.061 / 1.058 |
| bottom, `ag4_d40` | 75 / 77 % | 47 / 49 % | 10 / 8.6 % | 1.057 / 1.061 |
| walls, flat `ag4_flat_d20` | 66 / 65 % | 30 / 30 % | 25 / 26 % | 1.067 / 1.063 |
| bottom, flat | 79 / 80 % | 53 / 54 % | 7.2 / 7.1 % | 1.063 / 1.063 |
| walls, Im ε = −6 | 69 / 69 % | 29 / 29 % | 19 / 20 % | 1.075 |
| bottom, Im ε = −6 | 56 / 57 % | 36 / 38 % | 18 / 17 % | 1.053 |
| **top, `ag4_d40` − `ag4_d20`** (rms 1.57 / 1.76 % of J / M) | **53 / 53 %** | 28 / 30 % | 30 / 30 % | **1.061 / 1.049** |
| **top, `ag4_d21` − `ag4_d20`** (rms 1.95 / 2.22 %) | **54 / 57 %** | 30 / 33 % | 29 / 26 % | 1.048 / 1.051 |
| top, `ag4_d22` − `ag4_d20` (rms 1.97 / 2.10 %) | 53 / 55 % | 29 / 32 % | 29 / 27 % | 1.053 / 1.048 |
| top, flat D 4 − 2 µm (rms 0.34 / 0.35 %) | 62 / 65 % | 42 / 44 % | 9.8 / 8.5 % | 1.060 / 1.066 |
| **top, `ag48_d20` − `ag4_d20`**, common 4 µm square (rms 7.95 / 10.1 %) | **76 / 79 %** | 44 / 46 % | 12 / 9.4 % | **1.064 / 1.068** |
| top, lossy D 4 − 2 µm (rms 0.14 / 0.14 %) | 11 / 9.3 % | 5.0 / 5.0 % | 65 / 67 % | 0.989 / 0.985 |

Wall and bottom spectra use the 4 µm × D (walls) and 4 µm × 4 µm (bottom) windows, combined over
the four walls by their raw powers. Current amplitudes (rms): top J 1.65e-3 A/m, walls 1.27e-4,
bottom 6.4e-5 (rough, D = 2 µm); flat: walls 1.96e-5, bottom 1.20e-5. The flat top-face difference
D 4 − 2 µm contains the D = 2.0 µm flat-box resonance of (ii) (the flat far fields at 2.1 and 4.0 µm
agree to 0.034 %).

Findings (**measured**):

- **The box is covered by SPP-like surface waves.** On the 4 × 4 µm faces (bottom, walls at
  D = 4 µm) the spectra match a single SPP plane wave in the same window (75–80 % above k₀ against
  77–81 %, a k₀ wave gives 50–56 %; centroid 1.06 k₀). The 4 × 2 µm walls at D = 2 µm (72 %, flat
  66 %, centroid 1.07–1.08 k₀) lie inside the SPP range of their window (66–83 %) and above the
  k₀-wave range (51–59 %), but the two ranges touch, so there the spectra favour an SPP without
  excluding a mixture with grazing light. This holds for the rough and the flat surface and
  survives Im ε = −6 (shorter SPP decay, still above k₀). A cavity field of the vacuum interior has
  |k_t| ≤ k₀ on every face.
- **The roughness launches them.** Flat and rough boxes have the same rim illumination by the beam
  (|E_inc| 2.1e-2 at the rim), but the rough box has 6.5× the wall current and 55× the wall
  radiation (PARTS 7.5e-4 vs 1.35e-5). (WP-V2: the narrower beam, w₀ = 0.5 µm, whose wider angular
  spectrum lets the roughness reach k_spp more easily, doubled the depth sensitivity.)
- **What changes with the box on the top face is SPP.** The L-dependent part of the top current
  (8–10 % of it) has the spectrum of a single SPP in the window (76–79 % above k₀ against
  77–81 %); the depth-dependent part (1.6–2.2 %) is about
  60 % SPP-like and 40 % propagating (0.3–0.95 k₀; a mixture of SPPs and sub-k₀ components, among
  them the cavity component of (ii)). With Im ε = −6 the depth-dependent top current falls 11×
  and loses its SPP part (11 % above k₀).
- Which face radiates the change (DIFFFF lines; partial far fields interfere, so this is
  indicative only): the far field of the top-face current difference alone is nearly orthogonal
  to the depth change of the far field (correlation 0.03 for D 4 − 2 µm, 0.04 for 2.1 − 2.0 µm),
  so the depth change is radiated where the walls, the foot and the bottom are (the "rest",
  1.17 |dF|); the L change is radiated mainly by the top-face functions (top part 1.03 |dF|,
  correlation 0.91), which include the outer annulus of the larger patch.
- **The L change mixes two contributions that these runs do not separate.** (a) A truncation
  artefact (**hypothesis**): SPPs scattered into radiation or reflected at the rim of the 4 µm
  patch; supported by the 8–10 % change of the current on the common 4 µm square, which SPPs
  reflected at the rim would cause. (b) Real radiation from the added rough annulus: the SPPs
  launched under the beam (amplitude decay 41.5 µm) are also scattered into radiation by the
  roughness outside the 4 µm square, as they would be on the semi-infinite surface; the
  top-face dominance of the DIFFFF split fits this as well. The direct illumination of the annulus
  is small (|E_inc| ≤ 2e-2 of the peak beyond the 4 µm rim), so (b) is presumably driven mainly by
  the SPPs (**hypothesis**; the direct scattering of the beam tail by the annulus is not
  separated). (b) is physics of the semi-infinite surface, not an artefact (see the remedy
  section).

## (ii) Extinction-theorem residual inside the box

rms|E_inc + E₁| / rms|E_inc| on lines inside R2 (WP-A2 placement rule: z_p = max ξ + 3h = 0.287 µm
for this surface, |x|, |y| ≤ L/2 − 3h, 20 nm spacing; further x-lines at z_p + 0.5 and + 1.0 µm and
at D − 3h; the vertical line x = y = 0), normalised by rms|E_inc| on the z_p line.

| Run | x-line z_p | z_p + 0.5 | z_p + 1.0 | bottom line D − 3h | y-line z_p | vertical |
|---|---:|---:|---:|---:|---:|---:|
| `ag4_d20` | 0.97 % | 0.90 % | 0.83 % | 0.97 % | 1.00 % | 0.91 % |
| `ag4_d21` | 0.79 % | 0.76 % | 0.80 % | 0.73 % | 1.05 % | 0.70 % |
| `ag4_d22` | 0.84 % | 0.72 % | 0.67 % | 0.77 % | 0.83 % | 0.60 % |
| `ag4_d40` | 0.99 % | 0.86 % | 0.79 % | 0.90 % | 0.87 % | 0.79 % |
| `ag48_d20` | 1.43 % | 1.43 % | 1.40 % | 1.49 % | 1.55 % | 1.35 % |
| `ag4_d20_lossy6` | 0.20 % | 0.15 % | 0.15 % | 0.18 % | 0.20 % | 0.28 % |
| `ag4_d40_lossy6` | 0.18 % | 0.17 % | 0.16 % | 0.15 % | 0.19 % | 0.20 % |
| `ag4_flat_d20` (z_p = 0.15 µm) | 0.19 % | 0.19 % | 0.20 % | 0.18 % | 0.19 % | 0.24 % |
| `ag4_flat_d21` | 0.044 % | 0.046 % | 0.050 % | 0.036 % | 0.042 % | 0.037 % |
| `ag4_flat_d40` | 0.054 % | 0.033 % | 0.043 % | 0.048 % | 0.044 % | 0.052 % |

The top-line values of the other fine-sweep depths are in the (iii) table (0.76–1.14 %).

Findings (**measured**):

- A cavity component exists: 0.6–1.0 % at L = 4 µm and 1.4–1.5 % at L = 4.8 µm, against
  0.15–0.28 % with Im ε = −6 and 0.04–0.24 % for the flat box. On WP-A2's (flat, 45°,
  s-calibrated) scale that is a moderate (L = 4) to strong (L = 4.8) mode; the rough-surface floor
  is not calibrated (an upper bound of the floor is ≈ 0.2 %, from the lossy runs; the lossless
  rough-surface residual is 3–5× above it). Roughness raises the residual as it raises the wall
  SPPs (flat vs rough at D = 2 µm: 0.19 vs 0.97 %); whether the cavity component is fed by the
  plasmons or by the roughness directly is not separated here.
- It is **not depth-resonant**: across D = 2.0–2.2 µm and 4.0 µm its level varies by ±15 % with no
  switching and no enhancement at the bottom plate (WP-A2 at L = 3 µm, flat, 45° s: D = 2.0 → 2.1 µm
  moved it from 0.46 % to 1.6 %). The vertical profile is no clean standing wave (several
  components; INTZ lines), and the dominant lateral wavenumber of the residual changes from line to
  line (|k| = 0–0.9 k₀).
- In the **flat** box the cavity component does behave resonantly: 0.19 % at D = 2.0 µm, 4× less
  at 2.1 and 4.0 µm (0.04–0.05 %), and the far field follows it: 2.1 and 4.0 µm against 2.0 µm
  both give 0.24 %, but 2.1 against 4.0 µm gives 0.034 % (`wp_v3_logs/compare_review.log`). D = 2.0 µm
  is the odd one out: the flat-box depth ×2 change of 0.24 % is the radiating effect of the
  resonance at 2.0 µm (or of what accompanies it), away from it the flat box changes by 0.034 %.
  The first version of this record concluded "the resonance shows in the non-radiating residual,
  not in the far field"; that is wrong for the flat box. The flat Ritz values agree: the smallest
  singular value of the Hessenberg is 2.26e-3 at D = 2.0 µm against 9.95e-3 at 2.1 µm (at the
  converged k = 797 and 741, not at equal k; (iv)). In the rough box no depth of the sweep is an
  odd one out (against D = 4.0 µm all differ by 0.74–0.82 %, (iii)).
- It does **not track the far-field sensitivity**: the largest far-field step (2.0 → 2.1 µm,
  1.17 %) goes with a smaller residual (0.97 → 0.79 %), and L ×1.2 (4.5 % far-field change) raises
  it by 50 %, as a change of the box cavity would, without telling which of the two changes the far
  field. A purely non-radiating component cannot change the far field (WP-A2); its radiating
  discretisation residue would be discretisation-dependent, which the additional check (A) excludes
  for the rough box. The flat box shows that a resonant depth can still change the far field by
  ≈ 0.24 % (previous bullet); in the rough box such a contribution would have to hide in the smooth
  0.4–1.2 % oscillation of (iii), which none of the depths breaks.

## (iii) Fine depth sweep

D = 2.0 … 2.4 µm in steps of 0.05 µm (the brief's 2.0 / 2.1 / 2.2 µm and the optional 2.3 / 2.4 µm,
plus the half steps for the wave fit) and 4.0 µm; every run at tol 1e-5 against `ag4_d20`
(**measured**, COMPARE lines; noise floor of a pair at tol 1e-5 ≈ 3e-5, WP-V2):

| D [µm] | 2.05 | 2.10 | 2.15 | 2.20 | 2.25 | 2.30 | 2.35 | 2.40 | 4.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l2_refl vs 2.0 µm | 0.70 % | 1.17 % | 1.14 % | 0.76 % | **0.40 %** | 0.72 % | 1.10 % | 1.06 % | 0.82 % |
| l2_off30 | 1.05 % | 1.81 % | 2.05 % | 1.74 % | 1.18 % | 0.99 % | 1.40 % | 1.73 % | 1.42 % |
| ε_rr xz / yz | 0.10 / 0.13 % | 0.21 / 0.21 % | 0.21 / 0.19 % | 0.10 / 0.10 % | 0.03 / 0.03 % | 0.12 / 0.15 % | 0.21 / 0.21 % | 0.18 / 0.17 % | 0.19 / 0.17 % |
| GMRES iterations | 1 947 | 1 918 | 2 090 | 2 272 | 2 059 | 2 024 | 2 023 | 2 029 | 2 287 |
| interior residual, z_p line | 0.76 % | 0.79 % | 0.96 % | 0.84 % | 0.77 % | 1.14 % | 0.98 % | 0.84 % | 0.99 % |

(D = 2.0 µm: 2 096 iterations, interior residual 0.97 %; 2.2 vs 2.1 µm: 1.15 %.)

- **No on/off switching.** Every step of 0.05–0.1 µm changes the far field by about as much as
  doubling the depth; the change oscillates with D (minimum 0.40 % at ΔD = 0.25 µm, maxima near
  ΔD = 0.1–0.15 and 0.35–0.4 µm), i.e. with a period of ≈ 0.25 µm, close to λ/2. A resonance of a
  high-Q box mode would produce a peak at one depth and a near-zero change elsewhere (WP-A2's
  A_local: −2 % at 2.0 µm, −34 % at 2.1 µm, −0.4 % at 2.5 µm).
- The interior residual varies between 0.76 and 1.14 % without relation to the far-field change
  (its maximum, 2.3 µm, has a below-average far-field change).

**Travelling depth-wave fit** (`--depth-fit`, the nine runs D = 2.0 … 2.4 µm, complex far fields
of the reflection hemisphere weighted by √sin θ; model F(D) = F_∞ + A e^(−jβD) with one β per θ
band, residual relative to the depth-dependent part; **measured**):

| θ band [°] | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| depth-dependent part / mean | 0.47 % | 0.74 % | 0.72 % | 0.93 % | 1.6 % | 1.9 % | 4.6 % | 6.4 % | 8.1 % |
| best β / k₀ | 2.070 | 2.030 | 1.940 | 1.840 | 1.670 | 1.545 | 1.410 | 1.230 | 1.045 |
| residual of the fit | 1.2 % | 1.4 % | 2.5 % | 5.4 % | 9.1 % | 10 % | 6.2 % | 5.5 % | 4.4 % |
| β/k₀ − cos θ_centre | 1.074 | 1.064 | 1.034 | 1.021 | 0.963 | 0.971 | 0.987 | 0.971 | 0.958 |

(Five depths 2.0–2.2 µm give the same picture: 1.079, 1.074, 1.039, 1.021, 0.988, 1.006, 1.017,
1.006, 1.003.)

- **One travelling wave explains 90–99 % of the depth-dependent far field in every band**, and its
  depth wavenumber is close to **β(θ) = k_w + k₀ cos θ**: that is the phase of a wave that runs
  down the walls with wavenumber k_w and is radiated at depth D (the wall foot / bottom-plate edge)
  towards the reflection hemisphere (optical path k₀ cos θ · D back to the top).
- **k_w is not constant** (**estimated** from the table): the per-band value falls with θ, from
  1.074 k₀ (0–10°) and 1.064 k₀ (10–20°) to 0.96–0.99 k₀ beyond 40° (mean 1.005 ± 0.045 k₀ over
  the nine bands; five depths: 1.026 ± 0.032 k₀). A free least-squares fit β = a + b k₀ cos θ_centre
  over the nine bands gives a = 0.93 ± 0.02 k₀ and b = 1.11 ± 0.03 (rms misfit 0.023 k₀; five
  depths: a = 0.98 k₀, b = 1.07), against a = k_w, b = 1 for the model; with b fixed to 1 the best
  k_w is 1.005 k₀ (rms misfit 0.042 k₀), and k_w = Re k_spp = 1.055 k₀ fits worse (0.066 k₀). The
  depth fit therefore leans towards k_w ≈ k₀, i.e. towards grazing light along the walls or a mixture,
  not towards Re k_spp; the β grid step (0.005 k₀) and the short span (0.4 µm, **estimated**
  uncertainty ±0.05 k₀ per band) limit this. The SPP identification of the wall waves rests on the
  spectra of (i), which for the 4 × 2 µm walls at D = 2 µm separate SPP and k₀ only marginally.
- Excluded by the fit: a round trip down and up the walls (β = 2 Re k_spp = 2.11 k₀ in every band),
  and a field that does not radiate from depth D (a vacuum cavity standing wave between top and
  bottom contributes no depth phase of the form k₀ cos θ · D, and its far field is zero).

## (iv) Harmonic Ritz values (Jacobi-preconditioned ICTF, GMRES Hessenberg)

Leading (k + 1) × k blocks of the GMRES Hessenberg (`--ritz`): equal-k comparison of solves with
different iteration counts.

| Run | k | smallest three \|θ\| | # \|θ\| < 0.03 / < 0.1 / < 0.3 | σ_min(H̄) |
|---|---:|---|---|---:|
| `ag4_d20` | 1 000 | 0.0205, 0.0259, 0.0388 | 2 / 10 / 34 | 1.11e-2 |
| `ag4_d205` | 1 000 | 0.0229, 0.0248, 0.0372 | 2 / 10 / 33 | 1.29e-2 |
| `ag4_d21` | 1 000 | 0.0202, 0.0260, 0.0381 | 2 / 10 / 34 | 1.26e-2 |
| `ag4_d22` | 1 000 | 0.0198, 0.0211, 0.0343 | 2 / 10 / 34 | 1.03e-2 |
| `ag4_d40` | 1 000 | 0.0214, 0.0264, 0.0392 | 2 / 10 / 32 | 1.37e-2 |
| `ag48_d20` | 1 000 | 0.0228, 0.0249, 0.0399 | 2 / 10 / 32 | 1.41e-2 |
| `ag4_d215` … `ag4_d24` (5 runs) | 1 000 | 0.0196–0.0229, 0.0221–0.0255, 0.0368–0.0409 | 2 / 10 / 33–34 | 1.16–1.31e-2 |
| `ag4_flat_d20` | 797 (converged) | 0.0116, 0.0397, 0.0522 | 1 / 7 / 27 | 2.26e-3 |
| `ag4_flat_d21` | 741 (converged) | 0.0161, 0.0192, 0.0683 | 2 / 7 / 24 | 9.95e-3 |
| `ag4_flat_d40` | 1 000 | 0.0121, 0.0256, 0.0374 | 2 / 8 / 32 | 8.08e-3 |
| `ag4_d20` | 250 | 0.0873, 0.0975, 0.149 | 0 / 2 / 8 | 5.54e-2 |
| `ag4_d20_lossy6` | 250 | 0.101, 0.160, 0.164 | 0 / 0 / 7 | 6.05e-2 |
| `ag4_d20_lossy6` | 341 (converged) | 0.089, 0.135, 0.143 | 0 / 1 / 11 | 5.79e-2 |
| `ag4_d40_lossy6` | 364 (converged) | 0.102, 0.122, 0.137 | 0 / 0 / 12 | 6.38e-2 |

Findings (**measured**; reading = **hypothesis**):

- The band of 10 values with |θ| < 0.1 at k = 1 000 (Re θ of both signs in a regular ladder of
  spacing ≈ 0.01–0.015, Im θ ≈ 0.008–0.02 > 0) is the same within ~15 % for all ten depths
  D = 2.0 … 2.4 and 4.0 µm and for L = 4.8 µm. The flat box has a similar band (and converges in
  0.4× the iterations). Box cavity modes (vacuum standing waves between the top face and
  the bottom plate) would shift with D and double in number from D = 2 to 4 µm; this band does not.
  It is a property of the top-face/ material problem, not of the box dimensions (**hypothesis**:
  the Krylov sampling of a quasi-continuous part of the spectrum near 0, e.g. the weakly damped
  SPP continuum of the Ag interface, which the harmonic Ritz values cover with an evenly spaced
  ladder).
- With Im ε = −6, GMRES converges (k = 341 / 364) before such a band forms; at equal k = 250 the
  values are similar (smallest |θ| 0.10 vs 0.087 lossless), and the iterations fall 6×. Whether the
  lossy spectrum has the band is not tested (the first version of this record read "Im ε = −6
  removes the band", which the equal-k comparison does not support).
- The flat runs show a D-dependent smallest Ritz value: σ_min(H̄) = 2.26e-3 at D = 2.0 µm against
  9.95e-3 at 2.1 µm, at their converged k = 797 and 741 (not at equal k, so only indicative),
  consistent with the flat-box resonance at D = 2.0 µm of (ii).
- For the rough box: a box-independent near-origin band, against discrete box cavity resonances
  dominating the convergence; whether it is the SPP continuum is not shown (the lossy comparison
  above is inconclusive). Not proof: the Krylov space is not invariant (k ≪ 2N), as in WP-A2 E3.

## Additional checks

**(A) Discretisation independence of the depth change** (**measured**, complex cut amplitudes
cut_xz_F / cut_yz_F, `benchmarks/ag_box_mechanism_patterns.py`; the WP-V2 run directories are not
committed, as in WP-V2):

| Difference pattern A vs B | \|dF\|/\|F\| (xz / yz) | correlation with `ag4_d40 − ag4_d20` (xz / yz) |
|---|---|---|
| uniform 50 nm box, tol 1e-5: `ag4_d40 − ag4_d20` | 1.27 / 1.00 % | 1 |
| uniform box, tol 1e-4 / 1e-5 (WP-V2 `ag4_d4_uni_t4 − ag4_d2_uni_t5`) | 1.27 / 1.00 % | 0.9999 / 0.9998 |
| graded 100 nm box (WP-V2 `ag4_d4_b100_t4 − ag4_d2_b100_t4`) | 1.24 / 1.02 % | **0.938 / 0.926** |
| `ag4_d21 − ag4_d20` | 1.48 / 1.37 % | 0.743 / 0.748 |
| noise: tol 1e-4 vs 1e-5 (WP-V2 graded D = 2 µm) | 0.029 / 0.035 % | 0.100 / 0.042 |
| noise: d₀ 5 vs 3 (WP-V2 graded D = 2 µm) | 0.002 / 0.003 % | 0.137 / 0.159 |

The depth effect is the same complex pattern in two box discretisations (100 nm graded and 50 nm
uniform walls and bottom; different iteration histories), while the numerical noise patterns are
uncorrelated with it. A depth effect produced by the discretisation (a discretely placed resonance
amplifying discretisation or MLFMM error) would differ between the two boxes. Remaining caveat:
the top face (50 nm) is the same in both.

**(B) Correction of a WP-V2 reading.** WP-V2 quoted PARTS of the uniform depth-2 µm box from its
tol 1e-3 run (walls 1.6e-3, bottom 1.2e-3) and read a "6× drop of the bottom share" with depth as
resonance-like. At tol 1e-5 the shares are walls 7.5e-4 → 7.0e-4 and bottom 2.4e-4 → 1.9e-4
(D 2 → 4 µm, **measured**): a 20 % drop, no resonance-like behaviour; the tol 1e-3 run's shares
were inflated by the unconverged solution.

**(C) Flat reference** (σ = 0, same box, beam and rim illumination; **measured**): depth ×2 from
2.0 µm changes the far field by 0.24 % (2.1 µm: 0.24 %), 3.5× less than with roughness, but 2.1
and 4.0 µm agree to 0.034 % (`wp_v3_logs/compare_review.log`): the 0.24 % belongs to the resonant
depth 2.0 µm of (ii), and away from it the flat box passes the depth criterion (0.034 % < 0.1 %;
one pair). Wall radiation 1.35e-5 vs 7.5e-4, wall currents 6.5× weaker; the depth-dependent top
current (D 4 − 2 µm, so including the D = 2.0 µm resonance) is SPP-like (62–65 % above k₀,
centroid 1.06 k₀). With a flat top only the lit rim (|E_inc| 2e-2 of the peak at w₀ = L/4)
launches SPPs; the roughness under the beam adds the larger part. This fits
WP-V2's narrower beam (w₀ = L/8, rim dark): the depth sensitivity doubled (1.89 %), because the
wider angular spectrum of the narrow beam lets the roughness reach k_spp more easily
(**hypothesis**, consistent with the numbers, not tested separately here).

**(D) Depth averaging** (intensity averaged over groups of depths, docs/05 metric between the
group averages; `ag_box_mechanism_patterns.py --average`; **measured**):

| Groups | l2_refl of the averages | single-run pairs of the groups (median / max) |
|---|---:|---|
| D 2.00–2.20 (5 runs) vs 2.20–2.40 (5 runs) — **overlapping** (2.20 µm in both) | 0.18 % | 0.80 / 1.20 % |
| D 2.00–2.20 (5) vs 2.25–2.40 (4) (disjoint, review round) | 0.26 % | 0.81 / 1.20 % |
| D 2.00–2.15 (4) vs 2.20–2.40 (5) (disjoint, review round) | 0.25 % | 0.79 / 1.20 % |
| D 2.00–2.15 (4) vs 2.25–2.40 (4) | 0.23 % | 0.79 / 1.20 % |
| D 2.00, 2.10 (2) vs 2.30, 2.40 (2) | 0.26 % | 0.76 / 1.06 % |
| D 2.00, 2.10, 2.20 (3) vs 2.25, 2.35 (2) | 0.30 % | 0.87 / 1.20 % |
| D 2.00–2.20 (5) average vs the single run D = 4.0 µm | 0.53 % | 0.80 / 0.82 % |
| all nine D (2.0–2.4) vs L = 4.8 µm (D 2.0) | 4.40 % | 4.43 / 4.51 % |

(Rows 2, 3: `wp_v3_logs/depth_averaging_review.log`; the others `depth_averaging.log`. The first
version of this record quoted the overlapping row 1 as the best case, 0.18 %.)

Averaging over 0.2 µm of depth (about one period of the depth oscillation near normal incidence,
less than one at grazing angles, where β is smaller) cuts the depth sensitivity about 3×, to
0.23–0.30 % between disjoint groups, not below 0.1 %, and leaves the edge effect unchanged. The
averaged D = 2.00–2.20 µm far field still differs from the single D = 4.0 µm run by 0.53 %. These
groups all lie within 0.4 µm of each other; a real depth ×2 test of averaging needs groups around
2 µm and around 4 µm (not run).

## Mechanism decision

**Surface waves (most likely SPPs) on the closed Ag box** (physical). The roughness under the beam
couples a small part of the incident power into SPPs (Re k_spp = 1.055 k₀, amplitude decay
41.5 µm ≈ 10 L), which run over the top face. Part of them turns onto the walls, runs down to the
foot, where part is scattered into radiation that leaves past the box into the reflection
hemisphere (the depth effect, 0.4–1.2 % for any D step of 0.05–0.4 µm; its phase is close to
k_w D + k₀ cos θ · D with k_w ≈ k₀, (iii)), and continues over the bottom plate. The L effect
(4.5 %) mixes SPPs scattered or reflected at the rim (**hypothesis**, a truncation artefact) with
the real radiation of SPPs scattered by the rough annulus that the larger patch adds (alternative;
(i), last bullet). The truncated patch is physically a finite Ag block whose plasmons see its
whole surface; the semi-infinite rough surface has no rim, walls or foot, but its SPPs are
scattered into radiation by roughness everywhere along their 41.5 µm path. Evidence: (i) spectra,
(iii) no resonant switching and the travelling-wave law of the depth phase, (A) the same depth
pattern in a second box discretisation, (C) the flat reference and the loss diagnostic.

**Box cavity resonances** (complementary problem, WP-A2): present as a moderate non-radiating
component (ii) that spoils A_local (ADR 0006 amendment 2026-10-12 stands). In the rough box they
are not depth-resonant, do not track the far-field changes, are not tied to the GMRES near-origin
band (iv), and no depth of the sweep stands out in the far field: they are not the cause of the
docs/05 failures. In the flat box one depth (2.0 µm) is resonant and does change the far field by
0.24 % ((ii), (C)), so a resonance can radiate at the 0.1 % level; with roughness its contribution
is hidden below the 0.4–1.2 % SPP oscillation.

Confidence: **high** that the depth sensitivity is radiated by the walls, the foot and the bottom
(DIFFFF), that it is a property of the continuous problem in the sense that it is reproduced in a
second box discretisation (magnitude 1.24 vs 1.27 %, correlation 0.93), and that the cavity
resonances are not its cause in the rough box; **medium** that the wall waves are SPPs rather
than grazing light along the walls: the 4 × 4 µm faces are clearly SPP-like, the 4 × 2 µm walls
at D = 2 µm only marginally (72 % above k₀ against 66–83 % for an SPP and 51–59 % for a k₀ wave
in that window), and the depth fit leans towards k_w ≈ k₀ (free fit a = 0.93 k₀, b = 1.11;
k_w = 1.005 ± 0.045 k₀ with b = 1), not Re k_spp; **open** how the L effect splits into rim
artefact and annulus radiation. What would raise it further: a refinement of the top face (the
only part shared by the graded and uniform boxes of (A)); a longer depth span for a sharper k_w;
and the termination prototype of the remedy. Note for that prototype: a lossy surface frame absorbs
bound SPPs, but not free grazing waves travelling in the vacuum along the walls; if part of the
wall waves is grazing light, a surface frame leaves that part, and the prototype must test this
(e.g. the depth sensitivity with the frame on, against the 0.1 % criterion).

## Proposed remedy (input for an ADR)

The coordinator writes the ADR; this section lists the options with their evidence and cost.

**What can be required.** docs/05 asks that the far field of the truncated patch approximate that
of the semi-infinite rough Ag surface to the docs/05 tolerances (depth ×2 < 0.1 %, L ×1.2 < 0.5 %)
under a beam with a dark rim. For Ag at 500 nm this target cannot be reached with any local
termination at L ≪ L_spp = 41.5 µm (the SPP amplitude decay length; **estimated** argument): on
the semi-infinite surface the SPPs launched under the beam are scattered into real radiation by
the roughness along their whole path, far outside a 4–10 µm patch, and that radiation belongs to
its far field; a patch that ends after a few µm cannot contain it, whatever its termination does.
An absorbing termination (frame or PML) therefore **defines a modified, well-posed problem**: a
rough patch of size L in SPP-absorbing surroundings, whose far field is box-independent (depth,
L) if the absorption works, but which is not an approximation of the semi-infinite rough surface.
The SPP radiation from beyond the patch is removed by construction, not approximated. The ADR must
say what the frame model represents (e.g. "rough patch of size L in an absorbing, non-radiating
plane", a finite-patch scattering model) and replace or qualify the docs/05 wording for metals.
Within that modified problem, the SPPs must not come back from the box (depth effect) and must not
be scattered at the rim (edge effect, part (a) of the L effect).

| Option | Addresses | Evidence here | Cost / risk |
|---|---|---|---|
| **R1. Graded SPP-absorbing termination**: a lossy frame on the outer top-face band (where the beam is dark, e.g. ρ > 2.5 w₀) whose loss ramps up smoothly towards the rim, continued over walls and bottom, so that SPPs are absorbed before the rim | depth and edge effect | The loss diagnostic (Im ε = −6 everywhere) cuts the depth effect 9× (0.82 → 0.092 %) and the SPP part of the depth-dependent current 11×; but applied everywhere it also changes the top face (P_refl −25 %) and leaves the edge effect at 2.1 % (the SPPs still reach the rim with ~50 % amplitude at a 3.1 µm decay length) — the absorption must be confined to the frame and strong enough there | Library work: the material is homogeneous per region today. Variants: (a) an impedance boundary condition (Leontovich, Z = η₀/n) on frame, walls and bottom with a graded resistive term, hybridised with PMCHWT on the inner top face; (b) a third region (an absorbing "skirt" material below a skin layer) with junction basis functions. Both need their own validation (flat Fresnel limit, reflection of an SPP at the ramp, then the docs/05 checks). (a) medium, (b) large. A Leontovich IBC with \|n\| ≈ 3.1 supports an SPP at k/k₀ = √(1 − 1/ε) ≈ 1.050 against 1.055 for the PMCHWT top face (**estimated**): a phase mismatch at the PMCHWT/IBC junction that reflects part of every SPP there, which the ramp must keep below the target. (c) **Candidate (project lead's preference): a PML by complex coordinate stretching** at the boundary of the BEM domain (walls, bottom and the outer top band); not evaluated here (needs stretched-coordinate Green's functions or a stretched-geometry formulation; the coordinator writes the ADR). Every variant modifies the problem, not only outside the lit area: it removes the SPP radiation from beyond the patch, which the semi-infinite surface has (see "What can be required"). |
| **R2. Depth averaging** of the intensity over n box depths spanning half an SPP wavelength (≈ 0.24 µm) | depth effect only (averages out the interference of the foot radiation with the rest, keeps its incoherent part) | (D): ≈ 3× smaller, 0.23–0.30 % between disjoint groups of 2–5 depths over 0.2 µm — not below 0.1 %; a depth ×2 test of the averages (groups around 2 and 4 µm) not run | n solves per configuration (n ≈ 3–5); no library change; does not touch the edge effect (4.5 %), so it does not make Ag pass docs/05 by itself |
| R3. Larger L | both, slowly | SPP amplitude decay 41.5 µm: a 0.1 % return needs a path of several decay lengths, L ≳ 50–100 µm (**estimated**; 2N ~ 10⁸, **estimated**) | infeasible |
| R4. Periodic (Floquet) surface model | removes the truncation | — (SPPs remain, as physics of the periodic surface) | very large (periodic Green's function, Ewald, new MLFMM); a different problem (periodic statistics) |
| R5. Resonance-free formulation (JMCFIE-type) | the cavity component of (ii) (A_local, near fields) | the cavity component does not cause the far-field sensitivity (ii), (iv), (A) | medium–large; **does not address the docs/05 failures** — keep it only as an option for A_local |
| R6. Accept and state the uncertainty | — | depth 0.7–1.2 %, L ×1.2 4.5 % at L = 4 µm, w₀ = L/4 (this record and WP-V2) | none; Ag rough far fields stay qualitative / with a box error bar |

**Recommendation.** (1) Keep ADR 0006 amendment 2026-10-12b decision 3 (no quantitative rough Ag
far fields) and record the mechanism: surface waves (most likely SPPs) on the finite block, not
the box cavity resonances; a resonance-free formulation is not the remedy for the far field.
(2) Decide in the ADR what the terminated model represents (a rough patch in SPP-absorbing
surroundings, not the semi-infinite surface), then prototype R1 — a graded resistive impedance
frame (a) or a PML by complex coordinate stretching (c) on the outer top face, walls and bottom —
in its own work package (formulation layer), first on flat Ag (Fresnel limit and an SPP launched
by a single bump: reflection at the frame or PML against an analytic or long-patch reference),
then the depth and L checks on this configuration, including the test whether the termination
also removes the grazing-light part of the wall waves. (3) Until then, report Ag rough-surface
results only with the box uncertainty of R6; R2 is not worth adopting as a rule (≈ 3× for n× the
cost, edge effect untouched). (4) Keep the WP-A2 rule (A_ext only) and treat the JMCFIE-type formulation (R5) as a
separate, lower-priority item for near-field quantities.

Points for the ADR to settle: where the frame starts (ρ > 2.5 w₀ keeps the incident intensity
there below 4e-6 of the peak at normal incidence) and how fast its loss ramps (an abrupt step
reflects SPPs) — at w₀ = L/4 the rim is at 2 w₀, so a frame needs a narrower waist (e.g. w₀ ≤ L/8:
frame from 2.5 w₀ = 0.31 L to the rim) or a larger L, i.e. the frame also changes the waist rule
for metals. The L/8 rule makes sense only together with the frame: WP-V2 measured that w₀ = L/8
alone doubles the depth sensitivity (1.89 % against 0.91 % at L/4, both with the graded 100 nm
box), presumably because the wider
angular spectrum launches SPPs more easily (C). Further points: the PMCHWT/IBC SPP mismatch
(1.055 vs ≈ 1.050 k₀) at the inner edge of an impedance frame; whether the frame is a property of
the box generator (geometry) or of the formulation (a surface-impedance region or PML); what the
terminated model represents (above); and the validation target (depth and L checks below 0.1 % /
0.5 % on this configuration for the modified problem, plus the flat Fresnel limit unchanged within
its WP22c tolerance).

## Deviations from the brief and assumptions

- (i) uses a 2D radially binned spectrum of every face instead of WP-A2's strip spectrum along x
  (`strip_spectrum` needs the analytic flat reference and samples one strip): SPPs launched by a
  rough patch spread in all directions, so the annulus power at |k| ≈ k_spp is the quantity. The
  flat reference was run (cheap: 797 iterations).
- (ii)/(iv) were run for all depths, not only the 2/4 µm pair; the L pair is included.
- (iii) uses steps of 0.05 µm up to 2.4 µm (the optional 2.3/2.4 µm included) so that the depth
  phase can be fitted; every run at tol 1e-5.
- (iv) compares leading Hessenberg blocks at equal k (250, 1 000) instead of the full Hessenberg
  only, because the lossless and lossy solves differ 6× in iterations; a non-plasmonic absorber
  with similar |ε| was not run (the equal-k loss comparison turned out inconclusive, (iv)).
- The "which face radiates the change" split (DIFFFF) and the depth averaging (D) are additions,
  as are the WP-V2-based correlation check (A) and the correction (B).
- The beam is the WP-V2 study beam (`--beam spectrum`), so that `ag4_d20` reproduces the WP-V2
  reference bit for bit; not `excitation::AngularSpectrumBeam`.
- No library change (`src/`, `include/`): all code is in `benchmarks/box_validity.cpp`,
  `tests/support/` and three benchmark scripts.
- Review round (2026-10-13, no new solve; stored runs only): flat 2.1 vs 4.0 µm comparison (the
  flat-box reading of (ii) corrected), rough-box comparisons against D = 4.0 / 2.1 / 2.3 µm, free
  fit of the depth phase and the per-band k_w trend, per-window plane-wave references, the L
  effect split into two not separated contributions, the lossy Ritz reading, disjoint depth
  averaging groups, the run count, and the remedy section ("what can be required", PML
  candidate, IBC mismatch, L/8 only with a frame). Code: attribute checks for `--load-currents`,
  `--diff` and `--depth-fit`, plus / minus consistency check of the face transfer, top-face count
  check of the face classification, probe lines validated before assembly, `--ritz K > --max-iter`
  refused, Hessenberg in the memory guard, sigma in the SETUP line.

## What was not run

- A top-face refinement (h = 25 nm, 2N ≈ 6·10⁵, ~3 h per Ag solve) to vary the only part shared by
  the two boxes of (A).
- Other seeds, σ = 100 nm, oblique incidence, s polarisation, w₀ = L/8 with the uniform box.
- L ×1.2 pairs at several L (L-averaging), a deeper depth span for a sharper k_w.
- Any SPP-absorbing termination (the remedy needs library work and an ADR), including the test
  whether it removes the grazing-light part of the wall waves.
- A depth ×2 test of depth averaging (groups around 2 µm and around 4 µm).
- A PMCHWT run at another depth (box-cavity hypothesis of the PMCHWT plateaus).

## Reproduce

```bash
cmake --preset win-release -DSPECKLEBEM_BUILD_BENCHMARKS=ON && cmake --build --preset win-release
B=build/win-release/benchmarks/specklebem_box_validity
S=build/win-release/benchmarks/specklebem_mlfmm_scaling
R=out/v3/runs; SUM=benchmarks/results/ag_box_mechanism_rows.txt
C="--material ag --L-gen 4.8e-6 --waist 1e-6 --box-mesh-size uniform --compression mlfmm \
   --max-iter 4000 --save-currents --mechanism --summary $SUM"
# (v) MLFMM test of the PMCHWT plateaus (and ICTF for the ratio: --formulation ictf)
$S --material ag --L 3.2e-6 --box-mesh-size 50e-9 --beam angular-spectrum --max-iter 4000 --history \
   --formulation pmchwt --jacobi 0 --digits 5 --history-file benchmarks/results/ag_box_mechanism_histories.txt \
   --history-full-file benchmarks/results/ag_box_mechanism_histories_full.txt > ag32_pmchwt_d05.log
python benchmarks/ag_box_mechanism_histories.py \
   benchmarks/results/formulation_convergence_mlfmm_histories_full.txt \
   benchmarks/results/ag_box_mechanism_histories_full.txt --ref "material=ag L_um=3.2 box=50.000000" \
   --ref-extra "d0=3 " "form=pmchwt jacobi=0" --test-extra "d0=5 " "form=pmchwt jacobi=0"
# solves (one process each, one at a time; tags ag4_d<depth>, ag48_d20, *_lossy6, ag4_flat_*)
$B $C --tag ag4_d20 --out $R/ag4_d20 --L 4e-6 --depth 2.0e-6 --tol 1e-5 --ritz 250,1000
$B $C --tag ag4_d40 --out $R/ag4_d40 --L 4e-6 --depth 4.0e-6 --tol 1e-5 --ritz 250,1000 --reference $R/ag4_d20
#   D = 2.05 ... 2.4 um likewise (--ritz 1000); L = 4.8 um: --L 4.8e-6 --depth 2e-6
#   loss diagnostic: --eps-imag -6 --ritz 250,4000 (the original runs passed 250,100000, which
#   was clamped silently; --ritz K > --max-iter is now refused); flat: --sigma 0
# stored-currents diagnostics (top-face difference spectrum and far-field split; the stored
# attributes must match these options, the --diff run all but L and the depth)
$B --material ag --L 4e-6 --L-gen 4.8e-6 --waist 1e-6 --box-mesh-size uniform --depth 4e-6 \
   --load-currents $R/ag4_d40 --diff $R/ag4_d20 --tag ag4_d40 --summary $SUM
# stored-run comparisons of the review round (wp_v3_logs/compare_review.log)
$B --compare $R/ag4_flat_d40 $R/ag4_flat_d21
$B --compare $R/ag4_d40 $R/ag4_d20 $R/ag4_d205 ... $R/ag4_d24   # likewise against ag4_d21, ag4_d23
# depth-wave fit (the runs must differ in box_depth only), difference-pattern correlations,
# depth averaging (disjoint groups), windowed plane-wave references of (i)
$B --depth-fit $R/ag4_d20 $R/ag4_d205 $R/ag4_d21 $R/ag4_d215 $R/ag4_d22 $R/ag4_d225 $R/ag4_d23 \
   $R/ag4_d235 $R/ag4_d24
python benchmarks/ag_box_mechanism_patterns.py --pair $R/ag4_d40 $R/ag4_d20 --pair $R/ag4_d21 $R/ag4_d20
python benchmarks/ag_box_mechanism_patterns.py --average D2.00-2.20 $R/ag4_d20 $R/ag4_d205 \
   $R/ag4_d21 $R/ag4_d215 $R/ag4_d22 --average D2.25-2.40 $R/ag4_d225 $R/ag4_d23 $R/ag4_d235 \
   $R/ag4_d24 --average D4.0 $R/ag4_d40
python benchmarks/ag_box_mechanism_window.py
```

Committed data: `ag_box_mechanism_rows.txt` (RESULT / PARTS / MECH / RITZ / DIFF lines),
`ag_box_mechanism_compare.txt` (COMPARE / DIFF / DIFFFF), `ag_box_mechanism_spectra.txt`
(SPECSUM / SPEC), `ag_box_mechanism_interior.txt` (INTERIOR / INTZ), `ag_box_mechanism_ritz.txt`
(RITZ / RITZV), `ag_box_mechanism_histories*.txt` ((v)), raw logs `wp_v3_logs/` (including
`depth_fit.log`; review round: `compare_review.log`, `depth_averaging_review.log`,
`window_references.log`, `attribute_checks_review.log`). The run directories (currents, Hessenbergs, far fields; ~2 GB) are not committed;
the correlations of (A) use WP-V2's uncommitted run directories in addition.
