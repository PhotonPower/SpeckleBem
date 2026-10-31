# Operator reuse across excitations and the recycling potential (WP-KR1)

ADR 0013 decision 1. Library code: `Simulation::set_excitation`, `Simulation::excitation`, `SolveOptions` /
`InitialGuess` (`include/specklebem/simulation.hpp`), the initial-guess overload `solver::gmres(A, b, x0, M, p, cb)`
(`include/specklebem/solver/gmres.hpp`), Python `Simulation.set_excitation` and `solve(initial_guess="previous")`;
tests `tests/unit/test_operator_reuse.cpp`, the x0 cases of `tests/unit/test_gmres.cpp`,
`tests/python/test_operator_reuse.py`; driver `benchmarks/recycling_study.cpp`; raw lines `recycling_rows.txt`.

## Setup

| item | value |
|---|---|
| machine | 12-core desktop, 128 GB RAM, win-release, one study process at a time, started on an idle machine |
| load | U and Si U idle throughout; 16 µm: assembly idle, from the 30° swap on another worktree's build and ctest ran (GMRES / probe / swap seconds of that run are loaded; iterations and R are load-independent). The "idle" assemblies are nevertheless slower than the WP-SC4 idle rows of the same operators (`preconditioner_rows.txt`, `assemble` = `Simulation::assemble` without a preconditioner): without the ILUT setup (A + D) and the construction, Ag U 173.6 s vs 161.9 s (**+7 %**), 16 µm 711.3 s (792.2 − 78.4 − 2.4) vs 675.2 s (**+5 %**), Si U 693–726 s (the three runs 695.1 / 703.7 / 727.9 s, ILUT 2.4 s) vs 622.9 s (**+11 to +16 %**; +12 to +17 % against the 619.8 s of WP-SC2). No cause was identified (the runs record no other load); the reuse ratios below use this run's own fresh path and would be ≈ 7–16 % smaller with the WP-SC4 times (≈ 790–3 300 at U / Si U, 71 at 16 µm), which leaves their conclusion unchanged |
| cases | rough PML route of the preconditioner study, `pml_rough_ag::u_case()`: σ = 50 nm, Lc = 500 nm, seed 1, h = λ/4, W_t = W_g = 0.5 µm, W_pml = 2λ, σ_max 4, m 2; rough Ag U (L = 3.75 µm, 2N = 22 816), rough Ag 16 µm (`--patch 16e-6 --map-patch 16e-6`, 2N = 152 960), rough Si U (2N = 22 816) |
| operator | compression "mlfmm", d0 = 3, the default preconditioner of the route (ILUT of the near blocks A and D, fill 2, drop 10⁻², WP-SC4) |
| solver | full GMRES, left preconditioning, tolerance 10⁻⁷ (as the preconditioner records) |
| excitations | rigorous angular-spectrum beam, w0 = λ = 500 nm (the `u_case()` beam), focus at the origin, controlled ball 1.02 √2 pml_start; sequence **0° s → 30° s → 0° p** (s: E ⟂ plane of incidence xz = E along y; 0° p = the `u_case()` beam of the WP-SC3/SC4 rows) |
| beam-waist rule | `check_beam_waist(L, w0, θ, max_fraction = beam_waist_fraction)` passes for all three at every size (Ag U: 0.5 ≤ 0.81 µm at 30°; Si U: 0.5 ≤ 0.54 µm) |
| dark band | 30° s at U (Ag and Si, same geometry and beam): **1.44·10⁻⁴**, above the 10⁻⁴ target (w0 cos θ = 0.87 λ < λ₁: grazing waves of the cut spectrum, WP-PML3b); U and Si U run with `--dark-band-tolerance 2e-4` (all excitations; the 0° beams have 3.3·10⁻⁶). 16 µm: 5.5·10⁻⁵, default tolerance. A study setting for an iteration and reuse measurement, not a validated PML configuration: the reused and the fresh 30° solves see the same right-hand side |
| commands | `specklebem_recycling_study --tag u --dark-band-tolerance 2e-4 --parent --fresh`; `--tag u_L16 --patch 16e-6 --map-patch 16e-6`; `--tag si_u --material si --dark-band-tolerance 2e-4 --fresh` (all with `--summary recycling_rows.txt`) |

Per case: construct + `assemble()` (operator + right-hand side + ILUT) of the first excitation = the fresh path of one
excitation; the first solve (x0 = 0) with `keep_hessenberg` / `keep_basis`; its harmonic Ritz values; the recycle space;
then for 30° s and 0° p: `set_excitation` (timed), the solve with x0 = previous (the currents of the previous excitation,
`InitialGuess::Previous`), the solve with x0 = 0, and the deflation probe for k = 10 / 20 / 40 (clipped to the Krylov
dimension of the first solve). With `--fresh` a fresh `Simulation` of each of the two later excitations (construct +
assemble + solve) is compared with the reused x0 = 0 solve. With left preconditioning (the default, used here) the
residual history of a solve with x0 = previous starts at |M⁻¹(b − Z x₀)| / |M⁻¹b| (right preconditioning:
|b − Z x₀| / |b|), i.e. it keeps the reference of the x0 = 0 solve so that the tolerance means the same; ADR 0013
decision 1 writes |b − A x₀| / |b|. The r0 values below are this preconditioned ratio.

**Deflation probe** (driver only, a test-grade GCRO step, not library code): U = V_k Y with Y the eigenvectors of the k
harmonic Ritz values of smallest |θ| of the first solve (G = H_k + |h_{k+1,k}|² H_k^{-H} e_k e_kᵀ), C̃ = Op U by k
explicit applications of Op = M⁻¹Z, C = Q of the thin QR of C̃, U ← U R⁻¹ (Op U = C). For a new right-hand side:
f = M⁻¹b, r = (I − C Cᴴ) f, GMRES on (I − C Cᴴ) Op with the tolerance tol |f| / |r| (it then stops at the same
preconditioned residual |M⁻¹(b − Z x)| / |M⁻¹b| ≤ 10⁻⁷ as the plain solve), x = y + U Cᴴ (f − Op y). The iteration
count it reaches is the upper bound of what WP-KR2 (GCRO-DR with a subspace from one solve) can deliver.

## Results

### Summary

- **ADR 0013 gate for WP-KR2: not met.** At 16 µm the deflation probe with k = 40 harmonic Ritz vectors of the first
  solve needs **69** iterations for the second solve (30° s) against **71** with x0 = 0 (**1.03×**, gate ≥ 1.5×, i.e.
  ≤ 47); k = 10 / 20: 71 / 70. The third solve (0° p): 67 / 67 / 66 (k = 10 / 20 / 40) against 68. The baselines
  differ: ADR 0013 writes "68 → ≤ 45" with the 0° p count of the WP-SC3/SC4 records as its x0 = 0 baseline, while the
  second solve of this sequence is 30° s, whose own x0 = 0 count 71 gives ≤ 47 (71 / 1.5). On the ADR's own numbers
  the 0° p solve reaches 66 against 68 at k = 40 (1.03×), failing ≤ 45 by the same margin. At U and Si U the probe
  gains at most one iteration even with the whole Krylov space of the first solve (k = 23 / 14).
- **Why: the ILUT already compresses the spectrum.** The harmonic Ritz values of the ILUT-preconditioned operator have
  **no** value with |θ| < 0.3 at any size (smallest |θ| 0.65 at U, 0.43 at 16 µm, 0.86 for Si; σ_min(H̄) 0.51 / 0.26 /
  0.80, σ_max/σ_min 3.5 / 10.9 / 2.1). The parent blocks (WP-PC1 default) at U leave one value below 0.3 (0.298,
  σ_max/σ_min 29.9, 185 iterations). The band of 8–12 values with |θ| < 0.1 that WP-A2 E3 found for Jacobi on the box
  route has no counterpart with the ILUT on the PML route (different route and preconditioner, so not a like-for-like
  comparison): the iterations are spread over a compact cluster around 1, not
  spent on a few slow outliers, and deflating the smallest Ritz values removes almost nothing.
- **x0 = previous does not help** for these excitation pairs: the previous excitation's currents have a residual of
  1.22–1.35 × the x0 = 0 residual (s → s at 30°, s → p), and the solves need the same or up to 4 more iterations
  (16 µm 0° p: 72 vs 68). It stays an explicit option for correlated excitations (nearby angles), as ADR 0013 allows.
- **Operator reuse pays the whole assembly.** `set_excitation` (right-hand side only) costs 0.19–0.21 s at U / Si U
  against 178–179 s / 695–728 s for construct + assemble + ILUT setup of a fresh `Simulation` (ratio 840–3 900), and 10.6 s
  (30° s; loaded) at 16 µm against 792 s (ratio 75). Per additional excitation the cost drops from assembly + solve to
  the solve: at 16 µm 792 s + 60 s → 11 s + 60 s. The fresh-path assemblies of this run are 7 % (Ag U), 5 % (16 µm)
  and 11–16 % (Si U) slower than the WP-SC4 idle values (161.9 / 675.2 / 622.9 s, compared without the ILUT setup;
  no cause identified, see the load row); with those values the ratios would be ≈ 790–3 300 and 71, qualitatively the
  same.
- **Reused solves reproduce fresh solves bitwise.** At U and Si U the fresh `Simulation` of 30° s and 0° p gives the
  same iterations and R to every printed digit (dR 0, L2 0) as the reused x0 = 0 solve; the reused 0° p solves equal
  the WP-SC4 records (Ag U 0.97833914, 16 µm 0.97780528 with 68 iterations, Si U 0.38713580). x0 = previous and every
  probe agree with the x0 = 0 solve within |ΔR| ≤ 6.0·10⁻⁹ (relative dR ≤ 7.0·10⁻⁹), far-field L2 ≤ 3.8·10⁻⁸, all
  well inside the 10⁻⁷ criterion, so the probe is correct.

### ADR 0013 gate (second solve at 16 µm, 30° s after 0° s)

| method | iterations | ratio to x0 = 0 | R | dR vs x0 = 0 |
|---|---|---|---|---|
| x0 = 0 | 71 | 1.00 | 0.9569863827 | — |
| x0 = previous (0° s currents) | 71 | 1.00 | 0.9569863828 | +9.9·10⁻¹¹ |
| probe k = 10 | 71 | 1.00 | 0.9569863838 | +1.1·10⁻⁹ |
| probe k = 20 | 70 | 1.01 | 0.9569863831 | +3.9·10⁻¹⁰ |
| probe k = 40 | 69 | **1.03** | 0.9569863822 | −5.3·10⁻¹⁰ |
| gate | ≤ 47 | ≥ 1.5 | within 10⁻⁷ | |

The gate row uses this sequence's 30° s baseline (71 / 1.5 → ≤ 47); the ADR's "68 → ≤ 45" refers to the 0° p
baseline, which the third solve tests: 66 at k = 40 against 68 (1.03×), equally far from ≤ 45.

**Not met** (1.03× at k = 40). WP-KR2 is not started under the ADR's rule; the item closes with operator reuse and the
explicit initial-guess option, and the remaining per-solve lever is the matvec itself.

### Spectrum of the preconditioned operator (first solve, 0° s)

| case | preconditioner | its | Krylov k | count \|θ\| < 0.1 | count \|θ\| < 0.3 | three smallest \|θ\| | largest \|θ\| | σ_min(H̄) | σ_max/σ_min |
|---|---|---|---|---|---|---|---|---|---|
| Ag U | ILUT (default) | 23 | 23 | 0 | 0 | 0.649, 0.678, 0.701 | 1.62 | 0.511 | 3.53 |
| Ag U | parent blocks | 185 | 185 | 0 | 1 | 0.298, 0.363, 0.405 | 3.97 | 0.227 | 29.9 |
| Ag 16 µm | ILUT (default) | 69 | 69 | 0 | 0 | 0.435, 0.438, 0.469 | 2.14 | 0.263 | 10.9 |
| Si U | ILUT (default) | 14 | 14 | 0 | 0 | 0.864, 0.919, 0.946 | 1.47 | 0.798 | 2.11 |

Recycle space (DEFLATION lines): k = 23 of 23 at U (cond(R) 39, build 2.8 s), k = 40 of 69 at 16 µm (cond(R) 171,
build 54 s = 40 applications of Op + QR, U and C 0.20 GB), k = 14 of 14 for Si (cond(R) 47, build 13.7 s).

### Per case and excitation

Iterations (GMRES seconds; 16 µm from the 30° swap on loaded):

| case | excitation | swap s | fresh path s | x0 = 0 | x0 = previous (r0) | probe k = 10 | k = 20 | k = 40 | R (x0 = 0) |
|---|---|---|---|---|---|---|---|---|---|
| Ag U | 0° s | — | 179.0 | 23 (2.9) | — | — | — | — | 0.9782743224 |
| Ag U | 30° s | 0.195 | 179.0 (fresh run 178.9) | 24 (3.1) | 24 (1.24) | 24 | 23 | 23 (k = 23) | 0.9712044165 |
| Ag U | 0° p | 0.213 | 179.0 (fresh run 177.6) | 23 (3.1) | 24 (1.30) | 23 | 23 | 23 (k = 23) | 0.9783391394 |
| Ag 16 µm | 0° s | — | 792.2 | 69 (59.9) | — | — | — | — | 0.9729110350 |
| Ag 16 µm | 30° s | 10.6 | 792.2 | 71 (80.9) | 71 (1.25) | 71 | 70 | 69 | 0.9569863827 |
| Ag 16 µm | 0° p | 22.0 | 792.2 | 68 (59.7) | 72 (1.32) | 67 | 67 | 66 | 0.9778052808 |
| Si U | 0° s | — | 727.9 | 14 (14.9) | — | — | — | — | 0.3861121943 |
| Si U | 30° s | 0.191 | 727.9 (fresh run 695.1) | 14 (14.5) | 14 (1.22) | 14 | 14 (k = 14) | 14 (k = 14) | 0.4297980289 |
| Si U | 0° p | 0.188 | 727.9 (fresh run 703.7) | 14 (14.6) | 14 (1.35) | 14 | 14 (k = 14) | 14 (k = 14) | 0.3871357984 |

The 16 µm swap seconds (10.6 / 22.0 s) and the later GMRES / probe seconds of that run were measured while another
worktree built and tested; the right-hand side of the angular-spectrum beam is the only work of the swap (sum over its
plane waves at the quadrature points), 0.2 s at U.

Agreement with the x0 = 0 solve of the same excitation (dR relative, |ΔR| absolute, L2 of the windowed hemisphere
|F|²):

| case | excitation | x0 = previous | probe k = 10 | k = 20 | k = 40 | fresh Simulation |
|---|---|---|---|---|---|---|
| Ag U | 30° s | +3.4·10⁻⁹ / 1.4·10⁻⁸ | +2.1·10⁻⁹ / 1.3·10⁻⁸ | +3.4·10⁻¹⁰ / 2.2·10⁻⁸ | +5.9·10⁻⁹ / 1.9·10⁻⁸ | 0 / 0 |
| Ag U | 0° p | −1.0·10⁻⁹ / 1.7·10⁻⁸ | −3.6·10⁻⁹ / 1.6·10⁻⁸ | −1.6·10⁻⁹ / 1.8·10⁻⁸ | −2.0·10⁻⁹ / 1.8·10⁻⁸ | 0 / 0 |
| Ag 16 µm | 30° s | +9.9·10⁻¹¹ / 1.4·10⁻⁸ | +1.1·10⁻⁹ / 1.2·10⁻⁸ | +3.9·10⁻¹⁰ / 1.2·10⁻⁸ | −5.3·10⁻¹⁰ / 1.2·10⁻⁸ | (not run; 0° p = WP-SC4 row) |
| Ag 16 µm | 0° p | −8.9·10⁻¹⁰ / 2.2·10⁻⁸ | +6.4·10⁻¹¹ / 1.1·10⁻⁸ | +2.5·10⁻⁹ / 1.2·10⁻⁸ | +6.2·10⁻¹⁰ / 1.2·10⁻⁸ | (not run) |
| Si U | 30° s | +3.5·10⁻⁹ / 3.7·10⁻⁸ | +3.8·10⁻⁹ / 3.2·10⁻⁸ | +7.0·10⁻⁹ / 3.6·10⁻⁸ | (= k 14) | 0 / 0 |
| Si U | 0° p | +4.6·10⁻⁹ / 3.7·10⁻⁸ | +5.9·10⁻¹⁰ / 1.1·10⁻⁸ | −1.2·10⁻⁹ / 1.4·10⁻⁸ | (= k 14) | 0 / 0 |

Comparison with the WP-SC4 records (the 0° p beam is the `u_case()` beam): Ag U R 0.9783391394 vs 0.97833914 (23
iterations both), 16 µm 0.9778052808 vs 0.97780528 (68 both), Si U 0.3871357984 vs 0.38713580 (14 both).

## Interpretation and consequences

- Operator reuse is the lever ADR 0013 expected: for the angle / polarisation studies on one surface the assembly and
  the preconditioner setup are paid once (16 µm: 792 s once, then ≈ 70 s per excitation instead of ≈ 850 s).
- Krylov recycling with a subspace from one solve is not worth building for the ILUT-preconditioned PML systems: the
  probe is the best a GCRO-DR with k ≤ 40 vectors of the first Krylov space can reach for the second solve, and it is
  within 3 % of plain GMRES. A GCRO-DR that keeps refining U over many solves could approximate the slowest
  eigenvectors better, but the spectrum offers no isolated slow part to deflate (no |θ| < 0.3), so the expected gain
  stays small; measured iteration growth with size (23 → 69) comes from the cluster widening, not from outliers.
- The initial guess "previous" only helps for strongly correlated right-hand sides; for s / p and 0° / 30° pairs its
  residual exceeds that of x0 = 0.
