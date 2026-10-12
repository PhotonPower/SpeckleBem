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
  L = 4–17.5 µm). **Waiting for the lead's server results** (record template in `benchmarks/results/si_large_area.md`).
- **WP-A2 merged:** the Ag near-null mode is an intrinsic cavity resonance of the closed box (not MLFMM);
  Ag absorption only as A_ext (ADR 0006 amendment 2026-10-12); diagnostic follow-up WP-A3.
- **WP-V2 merged:** Si passes the docs/05 box checks with the uniform box at w₀/L = 1/6 (one configuration);
  Ag fails at every waist, cause open (SPPs vs WP-A2 cavity resonances) → WP-V3. ADR 0006 rule changes
  (uniform box for quantitative runs, w₀ ≤ L/6 for Si) proposed to the lead.
- **In flight:** WP22b2 (formulation sweep with MLFMM, L = 3.2–10 µm).
  They share the machine via a lock directory in the session scratchpad (WP22d holds it for timed runs).
- **Next:** WP-V2 (box validity at MLFMM sizes, graded vs uniform box), WP22b2 (formulation sweep with MLFMM),
  then Phases 5/6. A stronger preconditioner than Jacobi would address the Ag iteration growth.
- Test counts on `main`: 420 ctest (`-LE "validation-large|slow"`), 94 pytest; labels `slow`, `validation`
  (CI) and `validation-large` (manual, release, memory-guarded).

## Open questions / findings for the project lead

1. **Box validity (WP-V2):** Si is fine with the uniform box and w₀ ≤ L/6; **Ag rough surfaces do not pass**
   the docs/05 box checks at L = 4 µm (0.8–8 %); quantitative Ag speckle needs WP-V3 (mechanism and remedy).
2. **Ag absorption**: use A_ext (far field + extinction) only; the local formula is spoiled by intrinsic box
   cavity resonances (WP-A2). A resonance-free formulation would be a larger design change (open).
3. **Phase 4 target "Si 30×30 µm²"**: lead decision — proxy on this 128 GB machine plus a run kit; the lead runs
   L = 30 µm on a 1 TB server (WP22d). Ag scaling to 10⁶ is closed by measurement only once that server run (or a
   > 2 h local run) is done.
4. WP15 deviations from Fu et al. (formulation sweep with MLFMM, WP22b2) and the Ag λ/20 sphere (size) remain open.

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
- Literature PDFs live in `docs/papers/` (git-ignored, never commit).
