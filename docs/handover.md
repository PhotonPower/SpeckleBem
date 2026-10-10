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
- **Merged since:** WP21 (lossy-region policy, ADR 0008 WP21 amendment: Ag matvec vs dense 1.0e-4 / 4.1e-6)
  and WP21f (per-region jump test, dense cap on the automatic exact-part budget).
- **In flight (2026-10-10):**
  - `wp/22a-ag-sphere-4um` (pushed, in review): Ag sphere 4 µm with MLFMM on the paper's 393 216-unknown
    mesh (octahedron-based sphere, mean edge λ/16.5): worker-reported ε_rr 0.023 % / 0.016 % at tol 1e-3,
    442 iterations, 15.6 min, peak 33.8 GB — Phase 4 DoD (≤ 0.5 %) met if the review confirms. Includes a
    `SparseOperator` copy fix (−12 GB peak). Open: DoD wording "λ/27" (paper mesh is λ/27 by √area only),
    `SparseOperator(Matrix&&)` signature, d₀ = 5 needs exclusive use of the machine.
  - `wp/v1-box-validity` (record `benchmarks/results/box_validity.md` written; two Si L = 1.5 µm runs
    pending): docs/05 depth and edge checks fail at dense-feasible sizes; 400 nm box cells are a 1.5–2 %
    far-field error (100 nm: 0.17 %); Si needs the fine band; w₀ = L/3 too wide; the paraxial beam is 2.8 %
    off. Seven proposed ADR 0006 changes **await the project lead's decision** (asked 2026-10-10).
- Test counts on `main`: 350 ctest (`-LE "validation-large|slow"`), 83 pytest; labels `slow`, `validation`
  (CI) and `validation-large` (manual, release, memory-guarded).

## Open questions / findings for the project lead

1. **WP-V1 (important, decision pending):** see `benchmarks/results/box_validity.md` (recommendations for
   ADR 0006: box cells ≤ λ₁/5 where illuminated, Si fine band mandatory, w₀ ≤ L/4, docs/05 checks at
   L ≥ 6–11 µm with MLFMM, rigorous angular-spectrum beam before quantitative rough-surface results,
   state the docs/05 metric, keep default depths).
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
