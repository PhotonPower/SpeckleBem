# Handover: state of the project and how to continue

Updated by the coordinator agent on 2026-10-11 (second session on the Windows machine). Read this first, then `CLAUDE.md` and `docs/backlog.md`.

## Where things stand

- `main` is green on the Windows presets (`win-release` UCRT64 GCC 16 + OpenBLAS, `win-debug` CLANG64
  Clang 22 + ASan/UBSan) and in GitHub Actions (Ubuntu GCC 13 Release, Ubuntu Clang Debug, Windows MSYS2,
  format check). CI also runs on pushes to `wp/**` branches; it is the only Linux check (no WSL here).
- **Phase 1 and Phase 2** complete (Fresnel check WP12 deferred to Phase 4; Ag λ/20 sphere needs a
  ≥ 200 GB node).
- **Phase 3 complete:** GMRES with left/right Jacobi (WP13), `Simulation` driver (WP14a), Python API for
  meshes, materials, excitations, Mie, Simulation, fields, I/O with UTF-8 paths (WP14b1, 14b1f, 14b2, 14b3,
  WP-P1, ADR 0007), `.npy` result writer (WP16), formulation study (WP15,
  `benchmarks/results/formulation_convergence.md`: 3 of 6 Fu et al. Fig. 2 statements reproduced at
  2N ≈ 3·10⁴; issue #15 equivalence of the Jacobi-preconditioned formulations confirmed to 5e-13).
- **Kernel accuracy:** graded touching rule (WP7b), fold-adaptive rule for sharp folds and obtuse angles
  (WP7c), robust right-angle classification and box-rim rule (WP7d): touching-pair error ≤ 1e-7 for folds
  30–179°, ≤ 2e-8 on all rough-box pairs; icosphere blocks bitwise unchanged since WP7b.
- **Phase 4 (MLFMM) in progress:** design in ADR 0008 with six amendments from measurements. Merged:
  WP17 octree, WP18 plane-wave machinery, WP19a near field (`SparseOperator`, entries bitwise equal to
  dense), WP19b patterns and single-level FMM, WP20a multilevel far operator, WP20b full `MlfmmOperator`
  with `Simulation`/Python `compression="mlfmm"`. docs/05 MLFMM criteria met for dielectric and Si
  interiors (d₀ = 3 ≤ 2.9e-4 with λ/4 leaves; d₀ = 5 ≤ 6.7e-6 with λ/2 leaves; up to 2N = 8.8·10⁴). Also
  merged: WP-P2 dense assembly performance (Ag box ×35, Si box ×6 faster), WP7d kernel polish.
- **Phase 4 merged since (2026-10-10/11):** WP21 + WP21f (lossy-region policy), **WP22a** (Ag sphere 4 µm,
  393 216 unknowns: ε_rr 0.023 % / 0.016 % ≤ 0.5 %, DoD met) + WP22a-f, **WP-V1** (closing-box validity study),
  **ADR 0006 amendments** (box cells ≤ λ₁/5, Si fine band, w₀ ≤ L cos θ/4, rigorous beam, metric, GMRES tol ≤ 1e-5
  for quantitative Ag), WP-B1 (box defaults) + WP-B2 (angle-aware waist check), **WP-E1** (rigorous
  angular-spectrum beam, `Excitation::controlled_radius` checked by `Simulation`), **WP22b1** (scaling: DoD
  partially demonstrated — Si γ 0.73–0.88 to 2N = 5.2·10⁵, Ag γ 1.2 to 2.9·10⁵), **WP22c** (Fresnel: all 8
  cases < 1 %, DoD met), **WP-G1** (CGS2 GMRES, 1.6–4.7× faster orthogonalisation).
- **Merged 2026-10-11 (later):** WP21L (local leaf rule, opt-in, d₀ ≤ 3), WP-A1 (Ag absorption: A_ext from
  far field + extinction matches Fresnel; local ½Re∮(n̂×M)·J* fails through a near-null mode), **WP22b3**
  (scaling: **Si met** with the uniform 50 nm box, γ_solve 1.09 over 2N = 5.9·10⁴ … 1.02·10⁶, 954 s / 75 GB at
  10⁶; **Ag measured** γ 1.40 over 7·10⁴ … 4.2·10⁵, full window projected 1.40–1.51; default leaf quantile stays 1,
  ADR 0008 decision 2026-10-11).
- **WP22d merged:** Si proxy L = 10.5 µm (2N = 1.1·10⁶, 1 464 s, 89.5 GB); L = 30 µm extrapolated 600–720 GB,
  3.2–7.1 h; run kit `scripts/server_run_kit/` for the lead's 1 TB server (Si 20/30 µm + server Ag series
  L = 4–17.5 µm). Server Si results recorded (WP22e part 1, below).
- **WP-A2 merged:** the Ag near-null mode is an intrinsic cavity resonance of the closed box (not MLFMM);
  Ag absorption only as A_ext (ADR 0006 amendment 2026-10-12); diagnostic follow-up WP-A3.
- **WP-V2 merged:** Si passes the docs/05 box checks with the uniform box at w₀/L = 1/6 (one configuration);
  Ag fails at every waist, cause open (SPPs vs WP-A2 cavity resonances) → WP-V3. ADR 0006 rule changes
  (uniform box for quantitative runs, w₀ ≤ L/6 for Si) proposed to the lead.
- **ADR 0006 amendment 2026-10-12b (lead decision) + WP-B3 merged:** uniform box by default, w₀ ≤ L cos θ/6 for
  weakly absorbing objects (provisional, WP-V2b), no quantitative rough Ag until WP-V3.
- **Server runs (WP22d kit, commit 1be8f2b, w₀ = L/4):** Si runs 1–3 done 2026-10-12 and recorded (WP22e part 1,
  `benchmarks/results/si_large_area.md`, "Server results"): **Si L = 30 µm, 2N = 5.95·10⁶ solved**, 95 it.,
  solve 5.56 h, peak 697 GB; the server (2 × 8 cores, 32 threads) is ≈ 2 × slower than the desktop for Si.
  Ag series (run 4) recorded in WP22e part 2 (below).
- **Merged:** WP22b2 (fb4b2fc, formulation sweep with MLFMM, L = 3.2–10 µm). WP22e part 1 (6dc692e).
- **WP-V3 merged (b06276e):** the Ag box
  sensitivity is caused by surface waves (most likely surface plasmons) on the finite Ag block,
  radiated at the wall foot for the depth; for L ×1.2 rim scattering (hypothesis) and real
  radiation from the added rough annulus are not separated. Not caused by the box cavity resonances
  in the rough box (the flat box has one resonant depth, 2.0 µm, visible in the far field); PMCHWT
  plateaus unchanged at d₀ = 5. Remedy options for an ADR in `benchmarks/results/ag_box_mechanism.md`
  (SPP-absorbing termination: lossy impedance frame or a PML by complex coordinate stretching; it
  defines a modified problem, not the semi-infinite surface). **Remedy decided: ADR 0009 (accepted 2026-10-13),
  open surface with a complex-coordinate-stretching PML; dense prototype first (WP-PML1…PML4), Si stays on the box.**
- **Merged:** WP-PML1 (88c93fb, open surface + stretch). **WP22e part 2** (server Ag series L = 4–17.5 µm, all
  converged; γ_solve 1.37 ± 0.08 over 2N = 6.96·10⁴ … 1.02·10⁶ on one machine: **Phase 4 scaling item met for Si and Ag**;
  iterations identical to the desktop, server 1.45–1.88× slower for Ag).
- **Merged:** WP-PML2 (stretch-aware dense kernels: Piola factors, complex distance, ridge-split touching rules;
  bitwise unchanged without stretch; dense assembly with a PML ≈ 150× the plain cost, fine for the dense prototype).
- **Merged:** WP-PML3a (PML right-hand side masking, windowed far field, `Simulation` PML constructor, Python
  `Simulation.with_pml`; flat Si smoke −0.022 % vs Fresnel).
- **Merged:** WP-V2b (Si waist rule): w₀ ≤ L/6 confirmed at σ = 50 nm (second seed; L/5 also passes on seed 1);
  at σ = 100 nm L ×1.2 fails (1.19 % at L/6, 1.28 % at L/8, L = 4.5 µm): the waist is not the lever (measured);
  likely cause roughness-scattered fields reaching the rim (hypothesis, not separated from the added annulus).
  **Lead decision 2026-10-13 (option A):** ADR 0006 amendment 2026-10-13: L/6 rule for σ ≤ 50 nm; σ = 100 nm box
  results carry ≈ 1.2–1.3 % edge uncertainty; no follow-up scheduled.
- **Merged:** WP-PML2c (stretched assembly 3.2–6.5× faster, accuracy unchanged).
- **Merged:** WP23 (HDF5 on the Python side, lead decision 2026-10-13: `specklebem.io.save_hdf5` / `load_hdf5`,
  `npy_to_hdf5`, `Simulation.save`, bitwise round trip incl. a PML case; h5py optional extra `specklebem[io]`;
  the C++ HDF5 stub, `SPECKLEBEM_ENABLE_HDF5` and `has_hdf5()` removed; ADR 0001 amendment).
- **Merged:** WP25 (absolute DRC per steradian D = |F|²/(2η₁P_inc), radiated / reflected / scattered power,
  `Simulation::incident_power`, `far_field`, Python `reflectance`; flat Si PML 0° −0.022 % vs R_beam, Si sphere
  −1.17 % vs Mie C_sca). Direct far-field evaluation is cheap (1° hemisphere ≈ 790 s extrapolated at 2N ≈ 10⁶,
  4 threads): no MLFMM far-field acceleration; follow-up only if Phase 6 needs ≥ 10⁵ directions per solve.
  PML-route exception rule: wrong state → `std::logic_error` / RuntimeError, k_z > 0 direction → ValueError.
- **Merged:** WP-PML3b (validation 7a met): Ag 0°/45° and Si 0° at λ/4 ≤ 0.28 %; Si 45° only at the
  mesh-converged λ/6 (+0.18 % / +0.03 %) on a smaller geometry with the dark band relaxed to 1e-2 (the 7a geometry
  at λ/6 needs 173 GB dense); accepted by the coordinator on the measured dark-band insensitivity (≤ 0.08 % at λ/6),
  lead informed 2026-10-14. Rules: w₀ cos θ ≳ λ; 45°: w₀ = 0.75 µm, pml_start 4.5 µm; Si oblique h ≤ λ/6. The flat
  surface does not exercise the PML (sensitivities ≤ 2e-7): 7b (SPP) is the real PML test (WP-PML4). Proposed for
  WP-PML4: a warning (not an error) for oblique Si with h > λ/6 or a waist below the 45° rule; decide the default
  dark band (1e-4 costs ≈ 2.5× surface at 45°, R insensitive) with 7c/7d.
- **Merged:** WP24 (Ag sphere near field vs Mie, docs/05 row met): per-point e = |ΔE|/|E_Mie| < 0.1 outside the
  sphere beyond d_min = h/2 (incl. the x-axis and the forward region beside the sphere), < 1 in the geometric
  shadow. 4 µm (paper mesh) at GMRES tol 1e-4: 0.0124 / 0.0062 (≈ 8× margin); at the paper's tol 1e-3 0.088 / 0.022
  (1.14×: the error is GMRES-limited there, so acceptance runs at 1e-4); 1 µm: 0.0050 / 0.0049; CI n = 3: 0.050 /
  0.086. Lessons: the error rises steeply below ≈ h/2 from the surface (discretisation, not quadrature); direct
  field evaluation ≈ 4.5 ns per point × source sample (24 threads), 4 µm grid ≈ 7–13 min, comparable to the solve.
- **Merged:** WP-PML4a (validation 7b met, defaults confirmed): SPP on flat Ag reflected at the PML start with
  |r_spp| = 7.9e-5 / 3.5e-5 (J_x / M_y) for σ_max 4, m 2, W = 2λ — at the measurement floor (1–2e-4), 125× below
  1e-2; W = 2λ is a broad optimum (σ_max 2–8, stable at λ/6 and λ/7), m = 1 ≈ 10× above the floor, m = 3 no gain.
  W = λ also passes for σ_max 1–8 (≤ 3.2e-3) but varies > 10× with σ and h: kept as a cost option to be judged in 7c
  (WP-PML4b), defaults unchanged (coordinator decision 2026-10-15). Method notes: M_y is the clean observable, the
  local-source J_x carries 2.5–7× non-SPP field (SPP impedance 18–50 Ω vs 127 Ω; groove launch 111 Ω); the 9–10
  sample fit amplifies 1e-4 noise to |Δr| ≲ 1e-3 at r = 1e-2 (unit-tested at the real sampling).
- **Merged:** WP-LU1 (`solver::DenseLu`, LAPACK zgetrf/zgetrs or Eigen, backend reported; `solve_direct` a wrapper).
  Measured on the loaded machine: LAPACK 2–3× faster than Eigen at n = 11 440 (19.4 vs 60.7 s; PML matrix 16.9 vs
  35.7 s), equal on one thread — both run OpenBLAS kernels (`EIGEN_USE_BLAS`), the gap is Eigen's fork–join schedule
  under load (inferred). The 10–40× expected from WP-PML3b was an artefact of a concurrent build. zgecon replaced by
  the Hager/Higham estimator on the backend's solves (63 s → 1.5 s on the PML matrix). Non-finite Z → invalid_argument.
- **Merged:** WP-PML4c (validation 7e met): PML route vs validated uniform box on the same rough Si surface and mesh
  (L = 4.5 µm, w₀ = L/6, h = λ/4): 0.146 % vs the box's own L ×1.2 sensitivity 0.229 % (rule fixed before the PML
  solve); the residual is the common edge effect. **Finding for the lead:** the rough-Si far field is not
  mesh-converged at λ/4, λ/6 or λ/10 (box λ/4 → λ/6 2.3 %, λ/6 → λ/10 1.2 %, second-order ⇒ ≈ 4 / 2 / 0.7 % mesh
  error, ≈ λ/20 for 0.2 %; R moves by 0.75–1.5e-3 per step) — 7a's "λ/4 suffices for Si at normal incidence"
  holds for flat R only. The box-validity *sensitivity* checks are unaffected, but absolute-accuracy claims of
  rough-Si far-field patterns at 50 nm = λ/10 carry ≈ 0.7 %; the PML route beyond λ/6 needs WP-PML5 (fast PML
  ring). Needs an ADR 0006/0009 note on the rough-Si mesh rule (lead decision: target accuracy vs cost).
- **Merged:** WP-PML4b (7c met on rough Ag: σ ×1.5 4.6e-5, W ×1.5 2.5e-5, quadrature 3e-8; 7d met on rough Si 0.24 %,
  failed on rough Ag 2.6 % with the cause documented: the roughness launches SPPs into the flat gap and the
  far-field window truncates a bound wave — flat control 2e-7, window shift 1.6 %, width convergence ≈ 1/W, lossy Ag
  1.1 %; whether a physical L dependence remains is decided by WP-PML6). Lead decision 2026-10-18 (coordinator's
  recommendation): document now, Huygens-plane far field as WP-PML6 after WP-PML5 (ADR 0010; ADR 0009 decision 5
  amended). Dark band 1e-4 kept (relaxed 1e-2 costs 5e-4); W = λ σ 2 an option, not default (own 7c pair pending);
  PML-route configuration warnings in `Simulation`. Rough-metal results state L, gap and window. Assembly time, not
  memory, limits the dense PML sizes (≈ 1.5–2.3 h per 7c/7d set): the touching-block cache (WP-PML2 review W4) is
  still open and becomes WP-PML5's concern.
- **Merged:** WP-PML5a (fast PML route, baseline): compression "mlfmm" on the PML route = unchanged MLFMM on the real
  functions + exact `PmlRingOperator` (near ring on the octree pattern, far ring per leaf block, budget-guarded,
  apply thread-count independent). 7h at the 7c size: d₀ 3 matvec 1.5e-5, R 4e-8, far field 3.3e-6 vs dense,
  d₀ 5 matvec 1.4e-7; 4.6 vs 8.3 GB; setup 1.4× the dense assembly (near ring both orderings → 5c; far ring
  O(N_s·N) → ACA in 5b, needed ≈ 10× at L = 15 µm, 40× at 30 µm).
- **Merged:** WP-PML5c (near-ring assembly cost): canonical-pair cache + translation classes along the PML strips
  (key built in the kernels layer from every input of the touching rule; copies exact), Jacobi diagonal from the
  near parts, `PmlRingPlan` once per operator with a tree fingerprint. Gate met: near ring 0.26–0.41× the dense
  assembly of the same pairs at the 7c size (load varied; idle rerun open), matrix bitwise 5a. Setup is now
  dominated by the far ring (→ WP-PML5b).
- **Merged:** WP-PML5b (ACA far ring): partially pivoted ACA on H-matrix row clusters (η = 2, tolerance
  10^−(d₀+1)), stretched drop bound, integrated with 5c. 7h met through the fast path; `assemble()` 0.31× dense at
  the 7c size, operator 1.5 GB vs 8.3 GB; 2N = 152 960 in 18 GB, 300 160 in 33.6 GB; 7d-Si/7e reproduced. Phase 5
  PML fast path complete (7a–7e, 7h); 7f/7g await WP-PML6.
- **Lead note 2026-10-22** (`docs/lead/2026-10-22-next-work-packages.md`, binding): WP-PML6 option (A) (ADR 0010
  amended 2026-10-25), then WP-PC1 (ADR 0011 written 2026-10-25), WP-PML5d (parallel), WP-G2 conditional;
  deferred/rejected items listed there — do not schedule them.
- **Lead decision 2026-10-25 (option 1, coordinator's recommendation):** ADR 0010 option (A) withdrawn (real
  observation points over the PML are illegitimate: Im R̃ > 0; stretched points need theory and do not recover the
  grazing band); the L dependence of rough-metal far fields is accepted as physical (wide-gap plane converged in
  θ ≤ 30°, 7d ≈ 2.3 % for Huygens and currents); default stays Currents; rule gap ≥ 6 λ for rough metals, to be
  confirmed by WP-PML6b (window ×(1 ± 0.3), second seed). WP-PML6 is merged as "7f met, 7g not met — physical".
- **In flight (2026-10-25):** WP-PML6 review fixes (`wp/pml6-review-fixes`: CI 7f size, 7g large runs, Im R̃ guard,
  wording); WP-PML5d (`wp/pml5d-ring-pair-memo`, step-0 study first); WP-PML6 finishing worker
  on `wp/pml6-huygens-far-field-3` (record, docs, ctest, extended-plane diagnostic), the option (A)
  implementation follows on top; WP-PC1 started after ADR 0011. The earlier WP-PML6 continuation worker
  (worktree agent-a40c1f6891657e9af, deregistered by mistake, see the lesson) may still hand back measurements
  from `scratchpad/pml6b` — data only.
- **Earlier state (sleep 2026-10-20 09:50):** WP-PML6 (`wp/pml6-huygens-far-field`, implementation complete at 0e02fd7;
  continuation `wp/pml6-huygens-far-field-2` at f5bd897 in worker worktree agent-a40c1f6891657e9af: 7g reported as
  unmet with regression guards, docs/10 finished, wide-gap diagnostics; still to do there: record
  `benchmarks/results/pml_huygens.md` + rows, docs/05 rows 7f/7g, ADR 0010 result paragraph, full ctest both presets,
  report). **Findings:** 7f met (flat Ag/Si Huygens vs currents 2.7e-4 / 2.8e-4, dR ≤ 4e-7, z_H ×1.5 2.6e-4); 7g
  **not met and not in the ADR's "physical" sense**: 7d with Huygens 2.78 % (currents 2.57 %), but z_H ×1.5 1.05 %,
  window ×0.7/×1.3 1.75 %/0.66 %, Huygens vs currents at U 1.41 %, 7e with Huygens dR −5e-4 — the plane window
  (raised cosine, transition Δ = gap = λ) truncates the propagating field crossing the plane at z_H (point-source
  edge error ≈ π²/(2(kΔ)²) ≈ 12 % per end at Δ = λ; wide gap 1.375 µm: z_H sensitivity 0.085 % at θ ≤ 30°). Default
  stays `FarFieldSource::Currents`. **Lead decision pending (asked 2026-10-20):** (A) extend the plane beyond the real
  part over the PML zone and beyond with stretched-source field evaluation (already implemented and tested: dipole
  continued through the PML to 6e-3) and fix the window transition Δ ≈ 3–4 λ from the point-source error model
  before rerunning 7g (needs an ADR 0010 amendment; coordinator's recommendation), or (B) document the failure.
- **Lesson (2026-10-20):** never remove an agent worktree on a "stopped at turn limit" notification alone — the
  WP-PML6 worker was still running measurements from scratchpad copies of its executables (process filter on the
  worktree path missed them); check processes by worktree *and* scratchpad path, and prefer leaving the worktree
  until the agent's hand-back report has arrived. Only its memory note was lost. Repeated 2026-10-25 with the
  continuation worker (it had resumed after the wake and was running `scratchpad/pml6b/bin/huygens_v4.exe`): a
  5-day-old 0-byte task output file is **not** evidence that an agent is dead. Pre-removal check (both paths):
  `Get-CimInstance Win32_Process | ? { $_.CommandLine -match '<agent-id>' -or $_.ExecutablePath -match 'scratchpad' }`;
  a "Permission denied" from `git worktree remove` means a process still holds the tree — stop, do not force.
- **Next:** WP-PML6 (Huygens-plane far field for rough metals, ADR 0010; rows 7f/7g), then an idle-machine timing
  rerun of the 5b/5c records and the GMRES preconditioning item (iterations ≈ N^0.5 on rough Ag PML systems:
  602 → 1 715 → 2 679 at 2N = 2.3e4 / 1.5e5 / 3e5; Krylov basis 12.9 GB at 3e5 — restarted GMRES or a near-field
  preconditioner, needs an ADR note); Ag scaling on the uniform box (WP22b4, optional); Phase 6. Open lead items:
  rough-Si mesh rule (7e finding), ADR 0009 open point 4 (box for Si).
- Test counts on `main`: 586 ctest (`-LE "validation-large|slow"`), 129 pytest; labels `slow`, `validation`
  (CI) and `validation-large` (manual, release, memory-guarded).

## Open questions / findings for the project lead

1. **Box validity (WP-V2, WP-V2b):** Si is fine with the uniform box and w₀ ≤ L/6 at σ = 50 nm; at
   σ = 100 nm (the Phase-4 roughness) the L ×1.2 check fails at L = 4.5 µm for every waist tried (1.2–1.3 %,
   WP-V2b; decided 2026-10-13, option A: ADR 0006 amendment, documented edge uncertainty, no follow-up). **Ag rough surfaces do not pass**
   the docs/05 box checks at L = 4 µm (0.8–8 %). WP-V3: the cause is physical (surface waves, most likely
   surface plasmons, on the finite block, decay length 41.5 µm ≫ L), so a resonance-free formulation or
   tighter solver settings do not help. Since SPPs also radiate off roughness far outside a small patch,
   no local termination at L ≪ 41.5 µm approximates the semi-infinite Ag surface: an absorbing frame or
   PML (complex coordinate stretching, the lead's preference) defines a modified problem (rough patch in
   absorbing surroundings), and the ADR must say what it represents. The lead decides on that
   termination (ADR, library work) or a stated box uncertainty for Ag.
2. **Ag absorption**: use A_ext (far field + extinction) only; the local formula is spoiled by intrinsic box
   cavity resonances (WP-A2). A resonance-free formulation would be a larger design change (open).
3. **Phase 4 target "Si 30×30 µm²"**: **met** on the lead's 1 TB server (peak 697 GB, WP22e part 1); docs/01
   reworded to "solves on a 1 TB node" (lead decision 2026-10-13); memory reduction towards 256 GB is the
   optional WP22f. Ag scaling to 10⁶ is closed by measurement
   (WP22e part 2: γ_solve 1.37 ± 0.08 on the server over 2N = 6.96·10⁴ … 1.02·10⁶; the bottom 5·10⁴ … 6.96·10⁴ unmeasured).
4. WP15 deviations from Fu et al.: WP22b2 (MLFMM, L = 3.2–10 µm, uniform box) reproduces five of six Fig. 2
   statements at 10 µm; "PMCHWT does not converge" holds for neither material strictly (Si converges; Ag is
   impractical, non-convergence not demonstrated). The WP15 Ag MCTF/PMCHWT ratios at 3.2 µm are neither a size
   nor a graded-vs-uniform box effect; dense vs MLFMM, beam and 400 nm coarse cells remain unseparated
   (PMCHWT plateaus: not the MLFMM d₀ = 3 error — WP-V3 d₀ = 5 reproduces the history to 0.15 %; the
   box-cavity hypothesis for the PMCHWT plateaus is not tested for PMCHWT; the ICTF + Jacobi near-origin
   band does not move with D; cause open).
   Open decision: `formulation::recommend` for explicitly graded Ag boxes (plain ICTF there). The Ag λ/20
   sphere (size) remains open.

## How the workflow runs on this machine

- Coordinator plans, writes self-contained briefs, has every WP reviewed (reviewer builds/tests
  independently), sends fix rounds to a new worker (SendMessage is unavailable here), squash-merges,
  pushes `main` (`git push origin main` is allowed). Trivial review fixes (< 20 lines) are applied by the
  coordinator at merge.
- Worker pitfalls: see "Notes for workers" in `docs/backlog.md` (GCC 13 `-O3` false positives and the
  local winlibs GCC 13.3 reproduction; wall-clock tests failing under load; MAX_PATH with long scratch
  paths; never record permission-guard workarounds in memory).
- Workers hit the 150-turn limit on large kernel WPs (WP7c, WP-P2): their committed state was usable and
  their reports arrived after background work finished; keep WPs ≤ 800 lines and tell workers to commit
  early.
- Machine: Windows 11, 24 cores, 128 GB, MSYS2 in `C:/msys64` (UCRT64 + CLANG64), no virtualisation.
- After a reboot (2026-10-15) every agent is gone but the worktrees survive: check each in-progress worktree
  (`git status`, `git log main..HEAD`, logs in the session scratchpad), commit the leftovers as WIP, remove dead
  worktrees (`git worktree remove -f -f`), and spawn continuation workers pointed at the *existing* worktree
  (keeps the builds and benchmark binaries). A worker cannot run git against another agent's worktree (guard):
  continuation workers therefore commit on a `<branch>-final`/`-2` branch in their own worktree, which the
  coordinator fast-forwards onto the WP branch. Parallel Bash calls of the coordinator share one cwd: use
  `git -C <path>` and absolute paths, never `cd` in parallel calls.
- While other workers run heavy jobs, the permission classifier may refuse a worker's `cmake --build` ("interfere
  with workloads"); the worker then reports it and the coordinator runs the gate builds and ctest itself (both
  presets) before merging — which it does anyway. Workers never work around a refusal.
- Raw logs of local benchmark runs that a record cites by basename (WP24, WP-PML4a, WP-LU1) are archived in the
  git-ignored `results/local_runs/<wp>/` on the development machine (server logs: `server_runs/`); the session
  scratchpad is not permanent.
- Two branches that are each green can break `main` together (WP-LU1 removed the LAPACK prototypes of
  `pml_fresnel_support.hpp` that WP-PML4a's support header had been using): always build and test `main`
  itself after a squash merge, before pushing; fix forward on `main` with a small commit.
- Literature PDFs live in `docs/papers/` (git-ignored, never commit).
