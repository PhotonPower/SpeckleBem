# Handover: state of the project and how to continue

Updated by the coordinator agent at the end of the second multi-agent session (2026-10-10, first session
on the Windows machine). Read this first, then `CLAUDE.md` and `docs/backlog.md`.

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
- **In flight when the session ended (branches pushed, not merged, not reviewed):**
  - `wp/21-mlfmm-lossy` (head 4014306): lossy-region policy (ADR 0008 §6) for Ag, automatic leaf rule,
    WP20b review follow-ups, Ag validation-large cases. The worker had committed implementation and
    tests but had not reported; continue with a worker that verifies (builds, tests incl. the Ag
    validation-large cases), completes the brief in the WP21 backlog row and reports; then review.
  - `wp/v1-box-validity` (head a4768d0): study executable for the closing-box validity (depth ×2, graded
    vs uniform, Si fine band, edge effect) plus diagnostics (far-field share of top/walls/bottom, a
    diagnostic loss override). No results record yet; the long dense runs were still going. Continue
    with a worker that re-runs the study configurations one per process (each < 2 h), writes
    `benchmarks/results/box_validity.md` with verdicts against docs/05 and recommendations for ADR 0006
    — this is the most important open physics question (see below).
- Next after that: WP22 Phase 4 validation (Ag sphere 4 µm at λ/27, 393 k unknowns; scaling exponents;
  WP12 Fresnel; MLFMM size sweep for the WP15 formulation questions).
- Test counts on `main`: 335 ctest (`-LE "validation-large|slow"`), 83 pytest; labels `slow`, `validation`
  (CI) and `validation-large` (manual, release, memory-guarded).

## Open questions / findings for the project lead

1. **WP-V1 (important):** the graded closing box (ADR 0006) may not be valid as used: (a) the incident
   beam is still strong at the box bottom (z_R ≈ 7 µm), so the coarse bottom must reproduce the
   cancellation of E_inc in the shadow; (b) for Si without the fine band, the 400 nm box cells do not
   resolve the interior wavelength (|k₂|h ≈ 30); (c) for Ag, roughness-excited plasmons (propagation
   ≈ 20 µm) reach the side walls. Needs the docs/05 sensitivity checks before speckle results.
2. **WP15 deviations from Fu et al.:** PMCHWT converges and Jacobi is slowest for Ag at 2N ≈ 3·10⁴; a
   size sweep with MLFMM is needed.
3. **WP12 Fresnel** and the **Ag λ/20 sphere** remain deferred (size).

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
