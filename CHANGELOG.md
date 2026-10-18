# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com), versioning: [SemVer](https://semver.org).

## [Unreleased]

### Added
- simulation / tests / benchmarks (WP-PML4b, ADR 0009 decisions 7c/7d): PML-route configuration warnings
  (`SBEM_WARN`, not errors: a weak absorber such as Si at oblique incidence with a patch mesh h > λ₁/6; a beam
  waist below w₀ cos θ ≥ λ₁ or pml_start < 3.5 w₀ / cos θ), unit-tested with a log sink. Validation **7c met** on
  rough Ag (seed 1, σ = 50 nm, Lc = 500 nm, L = 3.75 µm, h = λ/4, 2N = 22 816): σ_max ×1.5 4.6e-5, W_pml ×1.5
  2.5e-5, quadrature 3e-8, gap ×1.75/×3.5 4.4e-5 (limit 1e-3). **7d met on rough Si** (0.24 % < 0.5 %) and
  **failed on rough Ag (2.6 %)**, documented with cause: the roughness launches SPPs into the flat gap (gap rms
  current 10 % of the patch, Si 0.8 %), and the far-field window truncates a bound wave (flat control 2e-7, window
  shift alone 1.6 %, width convergence ≈ 1/W, lossy Ag Im ε ×10 1.1 %) — ADR 0009 decision 5 amended, Huygens-plane
  far field planned as WP-PML6 (ADR 0010). Dark-band default 1e-4 kept (3e-6 at w₀ = λ; binding only for
  w₀ ≳ 1 µm at gap 0.5 µm; relaxed 1e-2 costs 5e-4 in the metric); W_pml = λ, σ_max 2 is a documented option
  (9.5e-5 vs the defaults, 0.76× unknowns, 0.57× memory, edge current 4e-3), not the default. First rough-Ag
  numbers: R = 0.9783 vs flat 0.9816, DRC(0°) 1.89 vs 6.49 sr⁻¹. Benchmark `specklebem_pml_rough_ag`, shared code
  `tests/support/pml_rough_ag_support.hpp`, unit tests, three validation-large cases (7c Ag; 7d Si asserted;
  7d Ag reported), record `benchmarks/results/pml_rough_ag.md`, rows, docs/05 rows 7c/7d + footnote, ADR 0009 note.
- tests / benchmarks (WP-PML4c, ADR 0009 decision 7e): validation 7e met — on the same rough Si surface (seed 1,
  σ = 50 nm, Lc = 500 nm, normal incidence, L = 4.5 µm, w₀ = L/6, bitwise identical height map and patch mesh
  h = λ/4) the PML route and the validated uniform box agree to 0.146 % in the ADR 0006 L2 metric (ε_rr
  0.026 / 0.041 %, ΔR −1.6e-5) against the box's own L ×1.2 sensitivity 0.229 % (ε_rr 0.058 %, R 4.1e-5), fixed
  before the PML solve; λ/6 cross-check at L = 3.5 µm 0.186 % vs 0.19–0.27 %. Finding: the rough-Si far field is
  not mesh-converged at λ/4 or λ/6 (box λ/4 → λ/6 2.3 %, λ/6 → λ/10 1.2 %, second-order: ≈ 4 / 2 / 0.7 % mesh
  error at λ/4 / λ/6 / λ/10, ≈ λ/20 for 0.2 %; R moves by 0.75–1.5e-3), common to both routes.
  Benchmark `specklebem_pml_si_crosscheck` (one route per process, shared fine-generated map subsampled to h,
  `--reference`/`--compare`, memory guard), shared code `tests/support/pml_si_crosscheck_support.hpp`, 6 unit
  tests, validation-large 7e case (4129 s), record `benchmarks/results/pml_si_crosscheck.md` with committed run
  logs, rows, docs/05 row 7e, ADR 0009 note.
- solver (WP-LU1): reusable dense LU `solver::DenseLu` (factor once, solve one or many right-hand sides) with a
  LAPACK zgetrf/zgetrs backend when a configure-time link probe succeeds (`SPECKLEBEM_HAVE_LAPACK`, private to the
  library; `solver::lapack_available()`) and Eigen `PartialPivLU` otherwise or on request (`LuBackend`). Column
  equilibration by powers of two, rcond by the Hager/Higham estimator (LAPACK zlacn2's algorithm) driven by the
  backend's solves instead of zgecon (whose BLAS-1/2 triangular solves cost more than the factorisation on PML
  matrices), up to two refinement steps against the original Z, 32-bit LAPACK integer guard, backend reported in
  `DirectSolveInfo::backend` and `Simulation::report`. `solve_direct` is a thin wrapper with unchanged results;
  `tests/support/pml_fresnel_support.hpp` uses it (7a Si 0° reproduced, −0.022 %). Benchmark `specklebem_dense_lu`
  (`--matrix`, `--repeat`, `--raw`), record `benchmarks/results/dense_lu.md`: LAPACK 19.4 s vs Eigen 60.7 s at
  n = 11 440 and 16.9 vs 35.7 s on the 2N = 11 440 PML matrix (loaded machine, minima; equal on one thread since
  Eigen uses the OpenBLAS kernels through `EIGEN_USE_BLAS`; the difference is the parallel schedule under load);
  the 10–40× of the WP-PML3b observation was not reproduced (that Eigen figure was taken under a concurrent build).
- tests / benchmarks (WP-PML4a, ADR 0009 decision 7b): validation 7b of the PML route met — an SPP launched on flat Ag
  (sub-wavelength groove under the rigorous beam, grooved minus flat solution; or a localized source) is reflected at
  the PML start with |r_spp| = 7.9e-5 (J_x) / 3.5e-5 (M_y) for the default profile (σ_max 4, m 2, W_pml = 2λ), at the
  measurement floor 1–2e-4 and 125× below the 1e-2 limit; the defaults are confirmed (W = 2λ a broad optimum for
  σ_max 2–8, stable at h = λ/6 and λ/7; m = 1 ≈ 10× above the floor; W = λ passes for σ_max 1–8 but varies > 10×
  with σ and h — a cost option for 7c). Benchmark `specklebem_pml_spp` (fit of outgoing + reflected cylindrical SPP
  waves with background terms, `--dump`/`--refit`, SPP-impedance diagnostic |out(M_y)/out(J_x)|), shared code
  `tests/support/pml_spp_support.hpp` (Hankel expansion, fit, in-place LU), 16 unit tests (synthetic recovery incl.
  the real 7b sampling, asymptotic reflected wave, plane shift, impedance), validation-large test (7b case plus a
  weak-PML geometric check with a shifted PML start), record section "7b SPP absorption" in
  `benchmarks/results/pml_validation.md` + rows, docs/05 row 7b, ADR 0009 note.
- tests / benchmarks (WP24, docs/05 row "Near field Ag sphere"): total field of the Ag sphere in the xz-plane vs
  Mie, relative error per point e = |E_sim − E_Mie| / |E_Mie| < 0.1 outside the sphere beyond d_min = h/2 and < 1 in
  the geometric shadow (|x| < a, z > 0). Shared code `tests/support/ag_near_field_support.hpp` (zones, exact
  percentiles, d_min scan), unit tests of the zone logic, CI case (icosphere n = 3, dense: 0.050 / 0.086),
  validation-large cases d = 1 µm (ico n = 5, tol 1e-5: 0.0050 / 0.0049) and d = 4 µm at the paper's 2N = 393 216
  (GMRES tol 1e-4: 0.0124 / 0.0062, margin ≈ 8×; at the paper's tol 1e-3 0.088 / 0.022, margin 1.14×, kept as
  evidence), benchmark `specklebem_ag_near_field`, record `benchmarks/results/ag_near_field.md` (near-surface
  profile, quadrature degrees 6/12/19, d_min choice, evaluation cost ≈ 4.3–4.6 ns per point × source sample on 24
  threads). `ag_sphere_mlfmm_support::run` gains an `after_solve` callback (skipped when GMRES did not converge).
- tests / benchmarks (WP-PML3b, ADR 0009 decision 7a): validation 7a of the PML route met. Benchmark
  `specklebem_pml_fresnel` (flat open surface + PML, rigorous beam, s and p from one LAPACK LU, `--estimate-only`
  dark-band scan), shared code `tests/support/pml_fresnel_support.hpp`, five validation-large cases, record
  `benchmarks/results/pml_validation.md` + rows. Ag 0°/45° and Si 0° at h = λ/4 within 0.28 % of R_beam; Si 45°
  converges only at h ≤ λ/6 (+0.18 % / +0.03 %, λ/8 −0.03 % / −0.07 %; the λ/4 value is reported only). Oblique
  configuration rule w₀ cos θ ≳ λ (45°: w₀ = 0.75 µm, pml_start 4.5 µm); quadrature, gap, W_pml and σ_max
  sensitivities ≤ 1.2e-5. docs/05 "PML route" block and ADR 0009 rule updated.
- post / simulation / python (WP25): absolute observables. `post::SphereGrid`, `hemisphere_grid`, `sphere_grid`
  (Gauss–Legendre θ × trapezoid φ, weights in sr), `post::radiated_power` = (1/2η₁) Σ w |F|², `post::drc` /
  `drc_polarized` (D = |F|²/(2η₁ P_inc) per steradian, ∫_{k_z<0} D dΩ = R), absolute cylinder DRC overload
  (far-zone approximation). `GaussianBeam::power()` = πw₀²/(4η). `Simulation::incident_power`, `far_field`, `drc`,
  `drc_polarized`, `radiated_power`, `reflected_power`, `scattered_power` (PML window applied automatically; on the
  PML route k_z > 0 directions are refused and `scattered_power` is unavailable); Python bindings and
  `Simulation.reflectance`. Validation: Si sphere scattered power vs Mie −1.17 % (< 2 %), flat Si PML 0° ∫D dΩ vs
  R_beam −0.022 % (validation-large); CI check that the public API reproduces the support-code reflectance;
  evaluation cost in `benchmarks/results/drc_evaluation.md`.
- python / io (WP23, lead decision 2026-10-13: HDF5 on the Python side, ADR 0001 amendment): `specklebem.io.save_hdf5` /
  `load_hdf5` (mesh, currents, right-hand side, fields with nested attrs, `/far_fields`, solver history and
  parameters, config / excitation / PML metadata, format version and self-descriptions), `npy_to_hdf5` for the C++
  `.npy` directory, `Simulation.save` (one locked snapshot, errors not swallowed); atomic writes via `mkstemp` +
  `os.replace`; complex data for real datasets refused. h5py is the optional extra `specklebem[io]`; CI installs it
  and requires the HDF5 tests (`SPECKLEBEM_REQUIRE_H5PY`). New `Simulation::assembled()` / `solved()` (C++ and
  Python properties), `PlaneWave::direction()` / `polarization()`. Root `ruff.toml` (line length 100).
- kernels / operator (WP-PML2c): 3.2–6.5× faster dense assembly with a PML stretch. Each unordered stretched pair is
  evaluated once (sub-block transposes supply the other ordering), `kernels::element_blocks_both_regions` shares the
  pair setup and, where both regions pick the same rule, the point loop; batched point loop with fastmath for pairs
  with K = 0 (≤ 7e-14, decided per region; K ≠ 0 bitwise). Graded touching points max(ceil(0.6 p), ceil(p_eff/2))
  instead of ceil(0.6 p) + extra (Ag corners ≤ 1e-9, strips unchanged). Unstretched path bitwise unchanged. Tests:
  pair-by-pair reference assembly, thread-count determinism, fastmath vs C library.
- operator / post / simulation / python (WP-PML3a, ADR 0009 decisions 4–5): the open-surface PML route end to end
  (dense). `op::assemble_rhs` accepts a stretch and masks every test function whose support has a vertex strictly
  inside the stretched zone (`op::pml_masked_functions`; Gap/Pml straddling functions masked; the incident field is
  never evaluated in the PML); `post::CurrentWindow` / `FieldOptions::window` (C¹ raised cosine on max(|x|,|y|)),
  required by `far_field` with a stretch, refused by the near-field functions; `PmlTermination` and the
  `Simulation(OpenSurfaceMesh, PmlTermination, excitation, config)` constructor (dense only; checks orientation,
  gap > 0 (≥ longest edge for order < 1), taper ≥ correlation length on rough patches, dark band ≤ 1e-4 amplitude,
  controlled radius over the unmasked supports); `far_field_window()` / `far_field_options()`. Python:
  `make_open_surface_mesh`, `OpenSurfaceParams`, `OpenSurfaceMesh`, `SurfaceZone`, `PmlTermination`,
  `CurrentWindow`, `Simulation.with_pml`, windowed `far_field` (`bistatic_rcs` refused on the PML route).
  Tests `[pml_rhs]`, `[pml_far_field]`, `[pml_simulation]`, `tests/python/test_pml.py`; validation-large smoke:
  flat Si, 2N = 11 440 (h = λ/4), R 0.388091 vs Fresnel beam 0.388176 (−0.022 %), edge currents ≤ 6.5e-9 of max.
- tests / docs (WP-V2b): Si waist rule confirmation. Uses the `--sigma` option of
  `benchmarks/box_validity.cpp` (WP-V3). Study (`benchmarks/results/box_validity_mlfmm.md`,
  section "WP-V2b"; rows in `box_validity_mlfmm_rows.txt`, raw logs in
  `benchmarks/results/wp_v2b_logs/`): 11 MLFMM solves at L = 4.5 → 5.4 µm, uniform box, tol 1e-5.
  At σ = 50 nm the provisional rule w₀ ≤ L/6 holds for a second seed (depth ×2 0.009 %, L ×1.2
  0.23 %), and L/5 passes both on seed 1 (0.007 % / 0.15 %). At σ = 100 nm the depth check passes (0.03 %) but
  L ×1.2 fails at L/6 (1.19 %) and at L/8 (1.28 %): the waist is not the lever (measured); likely cause
  (hypothesis, not separated from radiation by the added rough annulus): roughness-scattered fields reaching the rim. ADR 0006 change proposed (rule restricted to
  σ ≤ 50 nm, σ = 100 nm caveat, larger-L follow-up), not applied.
- kernels / operator (WP-PML2, ADR 0009 decisions 2–3): stretch-aware dense element blocks for the PML
  route. `op::Problem::stretch` (a `geometry::CoordinateStretch`) is passed by the dense assembler and
  `assemble_diagonal`; pairs with a stretched triangle use Piola-transformed RWG functions and the complex
  distance R~ on the principal branch (exp(+jωt)); near/far pairs a double Dunavant rule with ridge- and
  |k s|-aware degree selection (non-flat pairs with a K margin), touching pairs a Sauter–Schwab type
  relative-coordinate rule on the analytic kernel, split and graded along the near-singular ridge of an
  anisotropic stretch (`OperatorOptions::stretched_touching_points`, `kernels::stretched_pair_info`).
  No stretch, an identity stretch or a pair without PML triangle is bitwise unchanged. Sparse, region
  sparse, RHS, MLFMM and post refuse a non-identity stretch. ADR 0004 note (WP-PML2). Tests
  `[stretched_kernels]` (exact identity L~(k) = s L(ks), touching and near/far convergence, independent
  shared-edge reference, non-flat K vs brute force, reciprocity, decay, refusals, ICTF smoke) and slow
  `[stretched_sweep]`, `[stretched_cost]`.
- geometry (WP-PML1, ADR 0009 decisions 1 and 2): open rough surface terminated by a PML.
  `geometry::make_open_surface_mesh(HeightMap, OpenSurfaceParams)` meshes z = xi w(x) w(y)
  (raised-cosine C^1 taper `taper_window`) on the map's grid out to the truncation square, with
  per-triangle `SurfaceZone` tags (Patch, Taper, Gap, Pml), flat PML at z = 0, n_z < 0 (R2 into
  R1, emitted by the generator) and the outer square as the only boundary (no RWG there); zone
  widths are rounded up to the grid (SBEM_INFO). `geometry::CoordinateStretch` (polynomial
  profile, exp(+j w t) sign x~ = x - j sgn(x) Int sigma, linear continuation beyond the PML end)
  with `map`, `jacobian_diag`, `in_pml`, bit-exact `identity()`; `make_pml_stretch` builds it
  from an `OpenSurfaceMesh`. Unit tests `[open_surface]`, `[coordinate_stretch]`. The closed box
  route is unchanged.
- tests / docs (WP-V3): Ag box sensitivity mechanism. `benchmarks/box_validity.cpp` gains
  `--mechanism` (2D spatial spectra of J and M on the top face, walls and bottom vs |k_t|/k₀;
  extinction-theorem residual on the WP-A2 interior probe lines), `--ritz K1,...` (harmonic Ritz
  values of leading GMRES Hessenberg blocks), `--save-currents`, `--load-currents DIR [--diff DIR_B]`
  (diagnostics of stored runs, spectrum of the top-face current difference and the far-field split
  top face / rest), `--depth-fit` (travelling depth-wave fit of the complex far fields, new
  `refl_F` output), `--sigma`; helpers in `tests/support/ag_box_mechanism_support.hpp`
  (unit-tested, `tests/unit/test_ag_box_mechanism.cpp`); scripts
  `benchmarks/ag_box_mechanism_histories.py`, `ag_box_mechanism_patterns.py` and
  `ag_box_mechanism_window.py`. Study (`benchmarks/results/ag_box_mechanism.md`, 18 solves: 16 box
  runs and 2 (v) runs, raw logs in `wp_v3_logs/`): the Ag box sensitivity is caused by surface waves
  (most likely surface plasmons) on the finite Ag block, radiated at the wall foot for the depth
  (reproduced in a second box discretisation: magnitude 1.24 vs 1.27 %, correlation 0.93); for
  L ×1.2 rim scattering (hypothesis) and real radiation of the SPPs from the added rough annulus are
  not separated. Not caused by the box cavity resonances in the rough box (present in the currents,
  not depth-resonant; the flat box has one resonant depth that does show in the far field, 0.24 %).
  The WP22b2 PMCHWT plateaus are unchanged at d₀ = 5 (not an MLFMM effect; the box-cavity
  hypothesis is not tested for PMCHWT). Remedy options for an ADR: an SPP-absorbing termination
  (lossy impedance frame or a PML by complex coordinate stretching), which defines a modified
  problem (rough patch in absorbing surroundings), since no local termination at L ≪ 41.5 µm can
  approximate the semi-infinite Ag surface. Review round: `--load-currents` / `--diff` /
  `--depth-fit` check the stored attributes against the options (error on mismatch), the face
  transfer checks the plus / minus consistency, `--ritz K > --max-iter` is refused and the kept
  Hessenberg counts in the memory guard; `read_npy` / `read_attributes` moved to
  `tests/support/ag_box_mechanism_support.hpp` (round-trip tests).
- tests / docs (WP22b2): MLFMM formulation size sweep. `benchmarks/mlfmm_scaling.cpp` gains
  `--history-file` / `--history-stride` / `--history-digits` / `--history-full-file` (one
  `HISTORY` line of monitored residuals per solve, with `waist=` and `max_iter=`) and `form=` /
  `jacobi=` keys on the `ROW` line; option parsing moved to `mlfmm_scaling::parse_options`
  (unit-tested; numeric values must be consumed whole, out-of-range values and the history
  options with `--no-solve` / `--estimate-only` are refused, `--jacobi` accepts only 0 or 1),
  `probe_history_files` checks the history paths before the assembly; new `history_indices`,
  `history_record`, `append_line`, `formulation_key`, `Result::residual_history`;
  `benchmarks/formulation_sweep_summary.py` (refuses duplicate records; pytest);
  `mlfmm_scaling_fit.py` groups by `form` / `jacobi`. Study
  (`benchmarks/results/formulation_convergence.md`, section WP22b2; raw logs in
  `benchmarks/results/wp22b2_logs/`, full-precision histories committed): PMCHWT / ICTF / MCTF /
  Jacobi at L = 3.2, 5, 7, 10 µm for Si and Ag on the uniform 50 nm box (rigorous beam w₀ = L/4
  (Si pinned past the WP-B3 default), d₀ = 3, tol 1e-3), 43 solves. At 10 µm five of six Fu et
  al. Fig. 2 statements hold; "PMCHWT does not converge" holds for neither material in the strict
  sense: Si PMCHWT converges in 369–405 iterations at every L; Ag PMCHWT is impractical (residual
  at iteration 2 000: 1.1e-3 → 2.4e-3 from 3.2 to 10 µm, 6–8× the ICTF iterations), but its
  non-convergence is not demonstrated (both caps lie below the 5 µm count, the 10 µm residual was
  still decreasing). Ag MCTF/ICTF grows 1.46 → 3.17 with L; Ag Jacobi beats ICTF by 7 % at
  10 µm on the preconditioned residual, 3 % at equal true residual. The WP15 Ag Jacobi deficit is
  attributed to the graded box (one paired comparison at 3.2 µm: graded 100 nm box Jacobi 1.68×
  ICTF, uniform 1.00×); the WP15 Ag MCTF/PMCHWT ratios are neither a size nor a graded-vs-uniform
  box effect (graded 1.37 / 6.06, uniform 1.46 / 6.13, WP15 1.05 / 1.42; dense vs MLFMM, beam and
  400 nm cells not separated). The uniform box also cuts Ag iterations (ICTF −39 %, Jacobi −64 %
  at 3.2 µm). Ag verdicts rest on a box that fails the docs/05 Ag box checks (WP-V3 open).
- docs (WP22e part 2): server Ag series (run 4: Ag L = 4 / 7 / 11 / 17.5 µm, σ = 50 nm, graded 100 nm box,
  rigorous beam, ICTF + left Jacobi, d₀ = 3, tol 1e-3), four `row.txt` lines in `mlfmm_scaling_rows.txt`
  (block "WP22d server") and decimated histories `mlfmm_scaling_ag_server_histories.txt`; all converged
  (1 394 / 1 518 / 2 272 / 2 920 it.), **γ_solve 1.37 ± 0.08 over 2N = 6.96·10⁴ … 1.02·10⁶** on one
  machine (95 % interval 1.00–1.73; Phase 4 Ag scaling item met, the bottom 5·10⁴ … 6.96·10⁴ not
  measured); iterations and residuals identical to the desktop at equal L; server/desktop solve time
  1.45–1.88; peak 94.3 GB at 2N = 1.02·10⁶. `mlfmm_scaling_projection.py` gains `--back-tests` and
  `--capped` (default output unchanged). Record in `benchmarks/results/si_large_area.md`.
- docs (WP22e part 1): WP22d server results of the Si runs 1–3 (lead's 1 TB server, 2 × 8-core
  EPYC 9015, commit 1be8f2b, w₀ = L/4) in `benchmarks/results/si_large_area.md` ("Server results"
  and assessment) and the three `row.txt` lines in `mlfmm_scaling_rows.txt` (block "WP22d server").
  **Si 30 × 30 µm², σ = 100 nm, 2N = 5 947 200 solved**: 95 iterations (tol 1e-3), setup 79 min,
  solve 5.56 h, wall 8.0 h, peak 697 GB (extrapolated 600–720 GB); L = 20 µm: 95 iterations, peak
  250.4 GB (Si 30 µm solved on the 1 TB server; docs/01 "256 GB node" wording not met, closure is
  the lead's decision). (peak − Krylov) / model 0.79 / 0.86 / 0.84; the server is about 2 × slower
  than the desktop for Si (far matvec 2.5 ×, power summary 1.9–2.5 × per unit of work; cause
  untested: both machines ran 2 threads per core, so SMT alone does not distinguish them; pinning,
  two sockets with interleave, clock and compiler differ; thread count and NUMA policy not
  varied); the component-model
  extrapolation overestimated the 20 → 30 µm time growth by ≈ 35 %; R vs R_coarse differ by 2.1 % at
  L = 30 µm (far-field grid too coarse for the narrow beam). Ag series (run 4) pending.
- tests / docs (WP-V2): closing-box validity at MLFMM sizes (`benchmarks/results/box_validity_mlfmm.md`,
  raw lines `box_validity_mlfmm_rows.txt`). `specklebem_box_validity` gains `--compression mlfmm`,
  `--d0`, `--leaf-quantile`, `--jacobi`, the MLFMM memory guard (`--estimate-only`, `--force`),
  `--parts` and `--history`; grids, docs/05 metric, options and the memory guard decision
  (`krylov_basis_gb`, `check_memory_guard`) moved to `tests/support/box_validity_support.hpp` with
  unit tests (`[box_validity]`). Noise floor: d₀ = 5 vs 3 2e-5; GMRES tol 1e-5 (Si, floor 4e-5) /
  1e-4 (Ag screening, floor 1.8e-4 for one case). **Si with the uniform box passes depth ×2
  (0.032 % at w₀/L = 1/4, 0.007 % at 1/6) and L ×1.2 at w₀/L = 1/6 (0.19 %; 1.28 % at 1/4) — one
  seed, one roughness, one L ×1.2 pair; at w₀/L = 1/4 the λ₁/5 graded box does not suffice (0.085 %
  from uniform, depth ×2 0.14 %; 1/6 untested).** **Ag fails both checks at every waist tried**
  (depth ×2 0.82–1.89 %, L ×1.2 4.5–8.2 % at L = 4 µm, w₀/L = 1/4 and 1/8; uniform and graded box
  alike; ≥ 8× the thresholds). The Ag sensitivity and its slow GMRES are consistent with surface
  plasmons but not separated from the WP-A2 box cavity resonances (ADR 0006 amendment 2026-10-12),
  which the Im ε = −6 diagnostic (depth ×2 0.093 %, at the threshold within the noise) damps too;
  four discriminating experiments are listed, none run. WP-V1's hypothesis (rim illumination by the
  incident beam) holds for Si only. ADR 0006 recommendations (uniform box, Si w₀ ≤ L/6; for metals
  an SPP-absorbing termination only if the follow-ups show SPP-dominated box currents, otherwise a
  resonance-free formulation) proposed, not applied.
- solver / python / tests (WP-A2): `GmresParams::keep_hessenberg` returns the unrotated
  (k + 1) x k Arnoldi Hessenberg of the last GMRES cycle in `GmresResult::hessenberg` (default
  off, iterates bitwise unchanged); Python `gmres=dict(keep_hessenberg=True)` and
  `SolveResult.hessenberg`; `tests/support/krylov_spectrum.hpp` (Ritz, harmonic Ritz and
  singular values). `benchmarks/ag_absorption.cpp`: `--box-depth`, `--hessenberg`
  (`--save-hessenberg`, `--no-ritz`), `--perturb`, `--retro`, `--interior` (extinction-theorem
  residual inside the box) and a MODE line; support `strip_spectrum`, `mode_peaks`,
  `extinction_residual`, `fresnel_flat::Case::box_depth` / `keep_hessenberg`. Study result
  (`benchmarks/results/ag_absorption.md`, section WP-A2): the Ag near-null mode is
  **intrinsic** — a non-radiating damped cavity mode of the truncated box (vacuum standing wave
  between top face and bottom plate in the extinction residual), unchanged with d₀ = 5 at
  L = 2.5 µm (d₀ = 5 not run at L = 4 µm), not driven by spectrally flat errors (supporting
  evidence; the L = 4 µm perturbation solve did not converge), switching with L and the box
  depth (A_local +0.1 … −53 %); R, the far field and A_ext are unaffected.
- tests / tooling / docs (WP22d): large-area Si for the Phase 4 target "Si 30×30 µm²".
  **Proxy measured** on the 128 GB machine (`benchmarks/results/si_large_area.md`): Si L = 10.5 µm,
  σ = 100 nm, 2N = 1 098 720, uniform 50 nm box, rigorous beam, ICTF, d₀ = 3, tol 1e-3: 90
  iterations, setup 225 s, solve 1 464 s, peak 89.5 GB (near 16.3, far 71.0, Krylov 1.6 GB),
  R = 0.365, A = 0.634 (tol 1e-3, not a quantitative reflectance; balance 1e-3); σ = 100 nm
  points at L = 4 / 6 µm against the σ = 50 nm series (iterations +23 / +33 %, solve +48 / +49 %,
  peak +9 / +5 %). **Extrapolated, not measured** (component model for memory and setup, union
  with a power law for the solve) to L = 30 µm: 2N = 5.95·10⁶, model 823 GB, peak 600–720 GB,
  setup 27–53 min, solve 2.2–5.6 h on the 12-core machine (3.2–7.1 h total with the timing
  matvecs and the power summary); fits the 1 TB server, not a 256 GB node: docs/01's "≈ 2.2 M"
  counts the top face only (open 600 × 600 grid: 2N ≈ 2.16 M), the closing box now adds
  85–175 % instead of the few percent ADR 0006 expected (λ₁/5 cells and the mandatory Si fine
  band of the 2026-10-10 amendment, the 5.65 µm Si depth, the uniform-box recommendation of
  ADR 0008), and at the 234 nm leaf the model needs ≈ 138 kB per unknown, which puts even the
  top face alone near 256 GB. `--estimate-only` table for L = 20–32 µm (uniform and graded
  box), showing that the octree root quantisation sets the leaf (234 nm at 30 µm, 125 nm at 32 µm: L = 32 µm with
  q = 0.8 needs less memory than 30 µm). Benchmark options (`specklebem_mlfmm_scaling`):
  `--sigma` (series key `sigma_nm` in the ROW line and in `mlfmm_scaling_fit.py`, default 50 nm),
  `--history` (per-iteration residual and timestamp), `--far-field <deg>` (R = P_refl / P_inc and
  A = P_abs / P_inc after the solve; the options are validated before the assembly, and a
  post-processing error keeps the solve: ROW line without R / A, `post_error=1`, exit 4);
  `mlfmm_scaling_fit.py` also groups by the `machine` annotation. **Server run kit**
  `scripts/server_run_kit/` (Linux build incl. offline dependencies, thread / NUMA settings
  (physical cores, `OMP_PLACES=cores`), run list with memory and wall-time estimates (setup,
  timing matvecs, solve, post, total): smoke test, L = 20 µm, L = 30 µm with the graded q = 0.8
  alternative, a server Ag series L = 4 / 7 / 11 / 17.5 µm whose own γ_solve closes the Ag
  scaling DoD by measurement on one machine, and L = 32 µm; what to send back;
  `run_large_area.sh`: estimate, memory check incl. the smallest cgroup limit on the path to the
  root (v2) or the hierarchical limit (v1), with `--force-memory`, logged solve with VmHWM
  sampling and signal forwarding, summary row, `--matvecs`, series via `--L a,b,c`;
  smoke-tested locally with dash and bash at L = 1 µm). `benchmarks/si_large_area_extrapolation.py`,
  `server_runs/` ignored.
- tests / docs (WP22b3): MLFMM scaling DoD runs with the CGS2 GMRES, `--beam angular-spectrum`
  and exclusive machine time (`benchmarks/results/mlfmm_scaling.md`, rows block "WP22b3").
  **Si (uniform 50 nm box): γ_solve = 1.09 ± 0.04 over 2N = 5.9·10⁴ … 1.02·10⁶** (7 solved
  sizes; L = 10 µm: 68 iterations, 954 s, 74.8 GB): Phase 4 criterion (≤ 1.5) met by
  measurement. **Ag: γ_solve = 1.40 ± 0.14 measured over 2N = 7·10⁴ … 4.2·10⁵**; over the full
  window projected (1.40–1.51), not measured; the Ag part of the DoD remains open by measurement.
  The converged L = 17.5 µm solve (~5 h, ~100 GB) is scheduled for the 1 TB server run kit
  (WP22d). Setup / matvec / memory measured to 1.02·10⁶ (γ 0.99 / 0.99 / 1.06); the solves at
  7.6·10⁵ and 1.02·10⁶ exceed the 2-hour run limit (projected 2.6–2.9 h and 3.3–5.1 h from runs
  capped at 2 100 / 1 450 iterations, assuming the iterations do not keep growing at the local
  rate of L = 9 → 11 µm, γ_it 0.80). GMRES histories and the projection script are committed
  (`benchmarks/results/mlfmm_scaling_ag_histories.txt`, `benchmarks/mlfmm_scaling_projection.py`).
  CGS2 shortens the Ag solves 1.2–1.7 × against the WP22b1 MGS runs. Local leaf rule q = 0.8 at
  Si L = 4 / 8 µm (100 nm box): peak memory −34 % / −26 %, setup +32 % / +20 %, same iterations
  and matvec; measured, default unchanged at q = 1 (coordinator decision 2026-10-11: the uniform
  box is the recommended Si configuration, the local rule the opt-in memory option for graded
  boxes). WP22b1 rows kept as the older series.
- solver (simulation) / python (WP-B2): angle-aware beam-waist check (ADR 0006 amendment
  2026-10-11): `check_beam_waist(L, w0, BeamWaistOptions{.incidence_angle, .allow_wide})`
  enforces w₀ ≤ L cos θ_in / 4 (|θ_in| < π/2 validated; the exception states the rule and the
  numbers; `allow_wide` logs one SBEM_WARN). **API change:** the third parameter is now the
  options struct (`check_beam_waist(L, w0, true)` no longer compiles; write
  `{.allow_wide = true}`), so a `bool` cannot silently become an angle. Python:
  `sb.check_beam_waist(patch_length, waist, *, incidence_angle=0.0, allow_wide=False)`
  (both keyword-only). Callers: `fresnel_flat` passes its angle; its recorded 45° cases
  (w₀ = L/5 > L cos 45°/4, measured edge loss 0.06–0.1 %) keep their parameters via
  `Case::allow_wide_beam` / `--allow-wide-beam`; `mlfmm_scaling` (normal incidence) unchanged
  in behaviour.
- mlfmm / tests / python (WP21L): local leaf rule of the ADR 0008 amendment 2026-10-11, opt-in.
  `MlfmmParams::leaf_radius_quantile` (default 1.0 = the WP21 rule with bitwise identical results,
  until WP22b3 measures the local rule at large L; values < 1 only with `accuracy_digits` <= 3,
  otherwise `std::invalid_argument` / Python ValueError, `kLocalLeafRuleMaxDigits`) sets the leaf
  from the radius quantile r_q (`support_radius_quantile`, `support_radii`); `make_octree` gives every basis function a home level (`Octree` home-level
  constructor, `home_level`, `elevated_positions` / `elevated_first` / `elevated_count`,
  `active_elements`, `home_level_counts`): the leaf if r ≤ ratio(d₀) a_leaf, else the finest level
  with r ≤ ratio(d₀) a_l / 2 (`kElevatedSupportRatioFactor`; the amendment's leaf ratio gave O(1e-2)
  errors on a Si box: translators amplify interpolated child fields when the level order exceeds
  k r_min); home level < 2 raises `TruncationOrderError` (`mesh_or_leaf_size`). Far operator:
  elevated patterns sampled at the home level, aggregated after the children's interpolation and
  received after the downward pass; per-level radii, block checks, extents, truncated / exact pairs
  over the functions active on the level (`FarLevelInfo::home_functions`, `support_radius`,
  `elevated_pattern_bytes`, `truncated_basis_pairs`); `leaf_sampling(tree, k, d0, radius)`.
  Near field: a pair is near iff the ancestors on the coarser home level touch (`near_pattern`,
  `estimate_near_bytes`); `RadiationPatterns` stores leaf-level functions only (`slot`,
  `num_stored`). Python `mlfmm=dict(leaf_radius_quantile=...)`; `specklebem_mlfmm_scaling
  --leaf-radius-quantile` with the elevated functions in the estimate. Tests
  (`tests/unit/test_mlfmm_local_leaf.cpp`): home levels, near pattern vs brute force, partition of
  every basis pair into near / expansion / exact / truncated, quantile-1 identity, elevated
  columns / rows vs exact entries (3.6e-4 / 8.0e-5); validation-large graded box 1 × 1 × 4 µm vs
  dense: d₀ = 3 Si 3.4e-4, Ag 2.5e-4 (global rule 2.1e-4 / 2.0e-4, d₀ = 5 global 4.5e-6 / 5.2e-6).
  The local rule at d₀ = 5 missed 1e-5 (5.6e-5 / 6.4e-5) through the λ/2 leaves of a_min(5), not
  the elevated functions (7e-7 / 1e-6), hence the d₀ ≤ 3 restriction (unit test of the rejection).
  Elevated ratio ρ/2 (`kElevatedSupportRatioFactor`): with ρ the Si graded box gave 5.0e-2, with
  ρ/2 1.5e-4. Si L = 1 µm solve with q = 0.8: peak 10.3 → 6.3 GB, same 154 iterations; estimates
  Si L = 4 / 8 µm 55 → 35 / 84 → 60 GB with q = 0.8 (q = 0.99 elevates nothing there: coarse cells
  are 7.7–10.5 % of the functions; `benchmarks/results/mlfmm_scaling.md`).
- tests / tooling (WP-A1): investigation of the Ag absorbed power at oblique incidence (WP22c:
  local A = 0.0073 vs Fresnel 0.0127 at 45° s). `tests/support/ag_absorption_support.hpp`: absorbed
  power per surface part (any Dunavant degree), extinction integral P_ext = ½ Re ∮ (E_inc·J* +
  M·H_inc*) dS, analytic interface fields/currents of the beam on the infinite interface and their
  RWG interpolant, top-face metrics (correlation of J and M with the analytic currents), strip
  profiles, exact-row checks of the compressed operator (`op::assemble_sparse`). Study
  `benchmarks/ag_absorption.cpp` (`specklebem_ag_absorption`: the fresnel_flat case plus the
  diagnostics; `--allow-wide-beam`, `--refine` iterative refinement, `--save/--load-currents`,
  `--row-check`, `--profile`, `--spectrum` (raw and relative to the reference at +k₀ sin θ,
  `--spectrum-window rect|hann`), `--strip-check`; options of the other mode throw); tests
  `tests/validation/test_ag_absorption.cpp` (4 cases). `fresnel_flat::run` takes an optional
  inspection callback (called after `peak_rss` is sampled). Record
  `benchmarks/results/ag_absorption.md`: the formula, the quadrature and the box are correct; the
  far-field/extinction absorption (P_ext − P_sca)/P_inc of the same currents equals Fresnel
  (+0.2 %; A_ext is a difference of terms ≈ 160× larger than A, so its far-field quadrature must
  be good to ~10⁻⁵); the local ½ Re ∮ (n̂ × M)·J* is low because the L = 4 µm solution carries a
  spurious standing-wave component (±k₀ sin θ, of order 1 %) that shifts the phase of M against
  J by 4–6 mrad (Re Z_s/|Z_s| = 1.6 %, so 1 mrad ≈ 6 % of A). P_ext, P_refl and P_fwd depend only
  on the field radiated into R1, and currents that are the trace of a source-free vacuum field
  inside the box radiate nothing, so A_ext and R are exactly blind to such a component, while
  the local product's cross term with the Ag interior field is weighted by |ε′ − 1| ≈ 11: leading
  hypothesis a (nearly) non-radiating near-null mode of the discrete PMCHWT-family system at high
  contrast (null space of the exterior operator). Not solver error (unchanged at true residual
  10⁻⁶); h = 50 nm limits the local product to ~1–3 % even without the mode (interpolant −2.7 %,
  L = 3 µm −2.1 %), not to −43 %. The row and strip checks (exact-row residual 2.9·10⁻⁵, MLFMM
  error 4.4·10⁻⁵ without mirrored content) show that a near-null mode exists but do not decide
  between intrinsic excitation (discretisation error) and MLFMM-error excitation (gain needed
  only a few hundred, ~250–700): origin open, follow-up experiments E1–E5 in the record. Use P_ext − P_sca for
  Ag absorption. No library change.
- tests / tooling (WP22c, former WP12): flat-interface limit vs Fresnel (docs/05 row "Flat box,
  tapered beam, 0° and 45°", < 1 % on |r|²) — met for Ag and Si, 0° and 45°, p and s.
  `tests/support/fresnel_flat_support.hpp`: plane-wave Fresnel coefficients (exp(+jωt)), the
  beam reflectance R_beam of the rigorous angular-spectrum beam on an infinite interface (local
  s/p split per plane wave, power weights of `AngularSpectrumBeam::power()`), the incident flux
  through z = 0 (power() check, edge loss), the flat closed box from `rough_surface_box_params`,
  MLFMM solve through `Simulation`, reflected power from the far field (Gauss–Legendre in θ),
  absorbed power from the currents. Study `benchmarks/fresnel_flat.cpp`
  (`specklebem_fresnel_flat`, one case per process, `--estimate-only`, memory guard, `--compression
  dense` and `--d0` cross-checks); fast reference tests `tests/validation/test_flat_interface_fresnel.cpp`
  (4 cases); memory-guarded `tests/validation_large/test_flat_interface_fresnel_large.cpp` (Ag L =
  4 µm 0° and 45° p/s, Si L = 6 µm 0° and 45° p/s). Record `benchmarks/results/fresnel_flat.md`:
  errors vs R_beam Ag −0.52 % (0°, w₀ = 1 µm) / −0.07 % / −0.09 % (45° p/s, w₀ = 0.8 µm), Si
  +0.11 % / +0.10 % (0° p/s) / −0.14 % / −0.06 % (45° p/s); vs plane-wave Fresnel ≤ 0.59 % (Si
  45° p, of which −0.45 % is the beam's angular spread). At 45° the waist must be ≤ L/5 (L/4
  loses 0.6 % past the patch). The Ag 0° deficit at GMRES tolerance 1e-4 is solver error (−0.011 %
  and a closed energy balance at 1e-5; unchanged by MLFMM d₀ = 5 and the dense operator): Ag
  flat-surface solves need tolerance 1e-5, which the validation-large Ag cases use.
- solver (WP-G1): parallel classical Gram-Schmidt with re-orthogonalisation (CGS2) in the GMRES
  Arnoldi step, now the default (`GmresParams::orthogonalization = Orthogonalization::CGS2`;
  `Orthogonalization::MGS` keeps the previous serial modified Gram-Schmidt with the 0.7
  criterion, never chosen automatically). Always two passes (h = h₁ + h₂; the 0.7 criterion fired
  in 45/46 and 52/53 MGS steps of the dense Ag/Si sphere tests, so selective re-orthogonalisation
  saves nothing for these spectra); the pass-1 update is fused with the pass-2 projection (three
  sweeps over V per step). The Krylov basis is stored in contiguous column-major panels of 64
  columns, the last one sized to the remainder, so a filled basis takes exactly 16 (m + 1) n
  bytes (grows without copying); V^H w and V h are OpenMP products over fixed 4096-row blocks
  (`schedule(dynamic, 1)`) with thread-independent block partial sums added in block order, so
  results are bitwise reproducible and independent of the thread count (they differ from MGS by
  rounding; iteration counts may differ by ±1–2). Hand-written kernels instead of BLAS GEMV: a
  threaded BLAS may split the reductions by thread count, and nesting it inside the row-block
  loop would oversubscribe. `GmresResult` gains `orthogonalization_seconds`,
  `reorthogonalizations` and, with `GmresParams::keep_basis` (test hook), `basis`. Python:
  `gmres['orthogonalization']` ("cgs2" / "mgs"), read-only `SolveResult.orthogonalization_seconds`
  and `SolveResult.reorthogonalizations`. Benchmark `benchmarks/gmres_orthogonalization.cpp`
  (n = 2·10⁵, 1 000 iterations, 5900X): CGS2 with 24 threads ≈ 0.35 ms per basis vector and step;
  expected orthogonalisation speed-up over MGS on an idle machine ≈ 2–4× (microbenchmark
  1.6–3×), measured 3.7–4.7× in partly loaded 1 000-iteration runs; under heavy contention the
  gain is unreliable (0.65–1.5× in a review spot check). The real solve gain is to be confirmed
  with exclusive machine time (WP22b3); record `benchmarks/results/gmres_orthogonalization.md`.
- geometry / simulation (solver) / python (WP-B1): closing-box defaults of the ADR 0006
  amendment 2026-10-10. Geometry: `RoughSurfaceParams::exterior_wavelength` (λ₁) caps the
  automatic coarse spacing at the largest 2^M h_b ≤ λ₁/5 with the actual grid spacing
  (`geometry::box_spacing_for_exterior_wavelength`; trailing optional `exterior_wavelength`
  parameter of `make_mesh_from_height_map`, `detail::box_grading`, `detail::rough_box_arrays`);
  without it a graded box from the old automatic rule logs one SBEM_WARN (mesh unchanged), and
  an explicit coarse spacing above λ₁/5 warns when λ₁ is known. `geometry::kNoBandCoarseningDepth`
  (= 2 top-face spacings, the grading contract). Simulation helpers (`simulation.hpp`):
  `exterior_wavelength` (λ₀/|n₁|, conservative for lossy backgrounds; metallic backgrounds with
  Re(ε_r) ≤ 0 are rejected), `default_box_mesh_size` (nominal; 500 nm vacuum, h = 50 nm →
  100 nm, M = 1), `RoughBoxParams` / `rough_surface_box_params` (depth, λ₁, `box_mesh_size`
  unset so the generator applies the λ₁/5 cap, and the mandatory fine band 3δ + 3σ for weakly
  absorbing objects, 3δ > 2h: Si yes, Ag no at 500 nm / 50 nm; uniform box with SBEM_WARN and
  `uniform_fallback = true` if the band would reach the bottom) and `check_beam_waist`
  (w₀ ≤ L/4, `std::invalid_argument` unless `allow_wide`). L = 10 µm, h = 50 nm, seed 1: Ag
  (depth 2 µm) M = 1, closing box 47.5 % of the top face (old rule 7.19 %); Si (5.65 µm, 73
  fine rows) 195.5 % (old rule with band 153.4 %, uniform 326 %). Python: the helpers,
  `RoughBoxParams` (`kwargs()`, `uniform_fallback`), `default_box_depth` /
  `default_box_fine_depth` and the `exterior_wavelength` keyword of the rough-surface
  functions and `RoughSurface`; example `02_rough_surface_speckle.py` uses the box defaults
  and w₀ = L/4.
- excitation / python (WP-E1): rigorous Gaussian beam `excitation::AngularSpectrumBeam`
  (`sb.AngularSpectrumBeam`), the beam ADR 0006 (amendment item 4) requires for quantitative
  rough-surface results: a finite sum of exact propagating plane waves around
  `k̂0 = R_y(θ_in) ẑ` with spectrum `exp(−k_t² w0²/4)`, polarisation `ê0 − (k̂·ê0) k̂` (the WP-V1
  study construction, generalised to oblique incidence, s polarisation, a focus and a lossless
  background), `E(focus)·ê0 = 1 V/m`, evanescent part omitted (definition in docs/06).
  Gauss–Legendre (polar) × trapezoid (azimuth) grid, refined per direction until a 1.5× finer
  grid changes E by < `tolerance` (1e-10) on probe points of the ball `|r − focus| ≤ R`
  (default 4 w0); fixed orders optional (a fixed grid that misses `tolerance` is kept with a
  warning); `max_plane_waves` caps the check grids of both modes, orders are capped at 10⁶
  (`std::invalid_argument` for fixed, `std::runtime_error` for automatic grids past the caps).
  Fields are controlled only in that ball: `Excitation` gains `controlled_radius()` /
  `controlled_center()` (+∞ / origin by default; the beam returns R and its focus), the
  `Simulation` constructor rejects meshes with a vertex outside the ball
  (`std::invalid_argument` / `ValueError` naming the required `region_radius`), and an
  evaluation beyond 1.2 R logs one warning per beam (near fields at distant points; aliasing
  ghosts reach 1e-4 at 2 R and 0.1 at 4 R for w0 = 2.5 µm, R = 4 w0). For
  `|θ_in| + α_max > π/2` the outer components travel towards −z (logged, docs/06). Queries:
  plane-wave count, orders, achieved change, `power()` (Parseval), the plane waves themselves.
  New virtual `Excitation::fields(r)` (E and H; default the two calls, bitwise unchanged for
  `PlaneWave` / `GaussianBeam`), overridden by the beam with one pass and used by the RHS
  assembly and `post::total_field`; `sb.Excitation.fields`, `.controlled_radius`,
  `.controlled_center` in Python. Plane-wave grid immutable after construction, branch-free
  sincos with four partial sums: indicatively ~7 ns per plane wave and point (timed on a shared
  machine); ≈ 1 300–1 600 waves at R = 4 w0 (≈ (R/w0)² scaling; ≈ 3 600 at R ≈ 8 w0).
  Tests: each wave transverse with `H = k̂ × E / η` (div E and the curl
  equation of the sum at round-off), finite-difference Maxwell residual 9e-11 (paraxial beam:
  1.3e-2), focus normalisation, agreement with the study beam to 1.2e-14, paraxial limit
  (w0 = 5λ: max |E − E_paraxial| = 0.37 (λ/(π w0))², ratio 0.2496 at 10λ), 45° p/s central
  direction, polarisation and Gouy phase gradient `−k (1 − f²/2)`, rotation covariance, waist-plane
  power = `power()` = `(1 + f²/2) π w0²/(4η)`, argument validation, order / plane-wave caps,
  fixed-grid and outside-the-ball warnings (once), `fields()` through the base class, the
  `Simulation` region check (C++ and Python). `GaussianBeam`'s warning and
  header now point to it; validation helpers shared via `src/excitation/excitation_detail.hpp`.
- tooling / docs (WP-V1): box validity study `benchmarks/box_validity.cpp`
  (`specklebem_box_validity`, `SPECKLEBEM_BUILD_BENCHMARKS`): far field |F|² of one rough-box
  configuration (material, L with a shared cropped height map, waist, depth factor, coarse spacing,
  fine band) in the reflection and forward hemispheres and the xz/yz cuts, dense ICTF + GMRES
  (tol 1e-6), with a rigorous angular-spectrum Gaussian beam (or the paraxial one), `.npy` output,
  per-part far-field shares (top, walls, bottom, rim, foot) and `--compare` (hemisphere L2 of |F|²,
  docs/05 ε_rr); unconverged runs are neither written nor compared (exit status 1) unless
  `--allow-unconverged`, `--eps-imag` runs are marked DIAGNOSTIC and store ε_r. Record
  `benchmarks/results/box_validity.md`: at dense sizes (L ≤ 2.4 µm) depth ×2 (Ag 0.8–3.5 %,
  uniform box 1.1 %; Si fine band 0.13 %, ε_rr 0.04 %) and L ×1.2 (4–6 %) fail the docs/05
  tolerances; the patch edge is lit at the percent level (w₀ = L/3: 0.6–0.9 % of the rigorous
  beam power crosses z = 0 outside the patch) and the depth sensitivity follows the rim
  illumination (3× lower at w₀ = L/4), with surface plasmons carrying about half of it (lossy-Ag
  diagnostic). 400 nm coarse cells are 1.5–1.9 % from the uniform box; 100 nm cells 0.17 %
  (L = 1.5 µm), 0.59 % (L = 1 µm) and 1.54 % (L = 1 µm, depth ×2), so λ₁/5 is necessary but not
  shown to suffice for the depth check (depth ×2 with 100 nm: 1.76 % at L = 1 µm, 3.49 % at
  L = 2 µm). Si needs the fine band (0.44–0.88 % without; fine band + 400 nm cells 0.37 % from the
  uniform box). Proposed ADR 0006 changes: coarse spacing ≤ λ₁/5 (to be re-checked against a
  uniform box at w₀ ≤ L/4), w₀ ≤ L/4, rigorous beam; the w₀/L (and L) at which the depth check can
  pass is open (an earlier z_R ≥ 8 × depth, L ≳ 6–11 µm, estimate is withdrawn).
- tests/benchmarks (WP22b1, Phase 4 DoD): MLFMM scaling study on Gaussian rough boxes (sigma =
  50 nm, Lc = 500 nm, h = 50 nm, 100 nm box cells, Si fine band, paraxial beam w0 = L/4, d0 = 3,
  `recommend` formulation, full GMRES tol 1e-3). **Phase 4 scaling DoD partially demonstrated,
  still open** (N of the DoD read as the number of unknowns 2N). Si: time per solve gamma = 0.73
  +- 0.07 over 2N = 4.3e4 ... 5.2e5 (~0.88 with noise-reduced times: the clean first L = 1 um run,
  111 s; proxy iterations x (near + far) matvec), limited by memory with the 100 nm graded box
  (79 GB at 5.2e5, measured before d00317a). Ag: gamma = 1.2 +- 0.2 over 5 solved sizes 2N = 4.3e4
  ... 2.9e5 (1.4 from 7e4; the last interval alone 2.2, noise-sensitive; L = 3 um rerun on the
  merged tree: 1 004 iterations, 325 s quiet, 1 145 s under a parallel validation run), setup and
  matvec to 2N = 1.02e6 (gamma ~0.9). Not shown over 5e4 ... 1e6; remaining: the Ag solves at 4.2e5
  ... 1.02e6 after WP-G1 (fast GMRES orthogonalisation; the serial MGS is 42-66 % of the Ag solve)
  and Si at 2N >= 6e5 via the uniform box (`--box-mesh-size 50e-9`, estimates: Si L = 10 um 2N =
  1.02e6 ~87 GB model / ~73 GB near + far). Analysis: the global r_max of the box cells raises the
  leaves to 219-406 nm (the uncapped automatic 200-400 nm box rule to lambda/0.7 ... lambda/0.35,
  near field up to 307 GB), which sets the Si memory limit with 100 nm box cells; proposals: local
  leaf rule, on-the-fly / single-precision Si leaf patterns, CGS2 orthogonalisation (WP-G1) and a
  stronger Ag preconditioner. Executable `specklebem_mlfmm_scaling` (one case per process, box from
  `rough_surface_box_params` with `--box-mesh-size <m>|auto|uncapped`, `check_beam_waist` unless
  `--allow-wide-beam`, `--beam paraxial|angular-spectrum`, `--estimate-only` with a peak model that
  includes the far tables (far memory within 0.1 % of all 17 recorded cases; 1.5 near + 1.1 far + 2
  exact + 1 GB, conservative by +46 ... +94 % on the merged tree at small sizes), `--no-solve`,
  available-memory guard), fit script `benchmarks/mlfmm_scaling_fit.py` (series grouping, repeat
  minima, converged-only solve fits, slope standard errors, DoD window), smoke test
  `[mlfmm_scaling]`, record `benchmarks/results/mlfmm_scaling.md` with the raw rows.
- mlfmm (WP21f, WP21 review follow-ups): the automatic exact-part budget is capped at the dense
  matrix size 16 (2N)^2 (`min(max(2 x near, 1 GiB), dense)`: on small problems the 1 GiB floor
  allowed exact parts larger than dense); budget messages print the rule with one-decimal numbers
  instead of `std::to_string` ("2.000000"); `FarLevelInfo::search_run`, and levels after the
  first fallback log "search not run" instead of "not achievable, error 0.00e+00" /
  "(error 0, L = 0)"; `estimate_near_bytes()` moved from `far_operator.hpp` to `near_field.hpp`.
  Tests: each `assemble_region_sparse` region alone (full pattern, Ag/ICTF and n = 1.5/PMCHWT)
  against the dense matrix with the other region's weights zeroed, which sees the per-region
  jump terms the summed test cannot (a flipped sign: error 2 for ICTF, 3e-3 ... 7e-3 for PMCHWT;
  the summed test still passes); "search not run" on a 4-level n = 0.1 - 20j tree; budget cap.
- tests/benchmarks (WP22a, Phase 4 DoD): Ag sphere d = 4 um with MLFMM vs Mie on the paper's
  393 216-unknown mesh (octahedron-based sphere, 8 4^7 triangles, mean edge lambda/16.5; built by
  `make_octasphere` in `tests/support/ag_sphere_mlfmm_support.hpp`), d0 = 3, ICTF + left Jacobi,
  GMRES tol 1e-3: eps_rr = 0.023 % (xz) / 0.016 % (yz) (criterion <= 0.5 %, paper 0.26 % /
  0.37 %), 442 iterations (paper 424), 15.6 min, near field 12.2 GB, Ag exact part 6.8 GB (the
  automatic budget did not fire), peak RSS 33.8 GB; at tol 1e-4 (722 iterations) 0.0035 % /
  0.0025 % (the tolerance dominates eps_rr at 1e-3). Study executable
  `specklebem_ag_sphere_mlfmm` (one case per process, `--estimate-only`), validation-large cases
  (d = 1 um icosphere n = 5 at tol 1e-5: 0.019 % / 0.013 %, below the dense lambda/13 values; the
  4 um case guarded on the available memory), `system_memory::available_memory_bytes()`, record
  `benchmarks/results/ag_sphere_4um_mlfmm.md` (ramp-up d = 1 / 2 / 4 um, tolerance and mesh
  sensitivity, scaling at fixed h).

- mlfmm (WP21 review): exact far part stored as the region's (L_i, K_i) per basis pair
  (`op::RegionSparseOperator`, `op::assemble_region_sparse`; 40 instead of 96 bytes per pair,
  weights e_i, h_i, m_i applied in `apply()`): Ag icosphere R = 0.5 um, d0 = 3: 408 -> 170 MB
  (d0 = 5: 239 -> 100 MB); 2N = 61 440 Ag icosphere: 687 MB (four-entry storage 1.73 GB).
  Pre-assembly estimate (sum n_A n_B over the exact box pairs, logged; per-basis-pair count when
  the bound exceeds the budget) and budget guard `MlfmmParams::max_exact_far_bytes` (0 =
  automatic: max(2 x the near-field bytes from the octree, 1 GiB); `estimate_near_bytes()`)
  throwing `TruncationOrderError` with the new cause `exact_part_too_large` before any
  allocation; `exact_info()`, `exact_budget()`. The exact fallback is allowed only if
  x*(d0) / alpha <= 4 box edges of the first level without expansion (x* the root of
  (1 + x) e^-x = 10^-(d0+1), `truncation_decay_exponent()`: 11.76 / 16.69 for d0 = 3 / 5),
  replacing alpha a >= 1 (which let n = 1.5 - 0.6j make the exact part denser than the dense
  matrix). Leaf rule ratio `max_support_ratio(d0)`: r_max / a <= 0.6 for d0 <= 3, 0.3 for
  d0 > 3 (also the threshold of the cause mesh_or_leaf_size): 50-70 nm meshes return to lambda/4
  leaves at d0 = 3. Block check: per box the bases nearest to the partner box; "not run"
  instead of -1 in logs and exceptions. Python key `mlfmm.max_exact_far_bytes`; Simulation
  advice for the new cause; `Simulation::report()` lists automatic_leaf_size and the budget.
  Tests: RegionSparseOperator vs dense (jump terms, sparse pattern, errors), multi-level
  complementarity (3-sphere 4-level tree in every build, 4-level icosphere in release: near +
  exact = dense to 1e-12, pair counts = N^2), causes lossy_region / digits /
  exact_part_too_large, budget, x*(d0); validation-large Ag icosphere 2N = 61 440 exact rows
  (1.1e-4, setup 20 s, apply 0.23 s, peak RSS 4.9 GB).
- mlfmm (WP21): region policy of ADR 0008 §6 in `MlfmmFarOperator`, per region and level from the
  leaf up: expansion if the order search is achievable (lossy regions additionally: block check
  <= 10^-d0 or <= 2 x the lossless analogue's), otherwise per box pair a documented truncation
  when the decay bound (1 + alpha d) e^{-alpha d} (d = distance of the support bounding boxes;
  the WP-P2 bound of |G| and |grad G| relative to their undamped bounds) is <= 10^-(d0+1), else an
  exact region-masked sparse correction (`exact_part(i)`, `op::assemble_sparse` with the other
  region's weights zeroed; refined per basis pair with the same bound); `FarDecision`,
  `FarLevelInfo` (decision, search / block / reference errors, bounds, pair counts),
  `describe()` and `SBEM_INFO` report every decision; `TruncationOrderError` (cause: mesh or
  leaf size, lossy region, digits) when a region has neither expansion nor enough decay;
  `MlfmmParams::exact_far_regions` forces the exact fallback (diagnostics);
  `basis_patterns()` (patterns of any bases about any centre, for the block check on every
  level). Leaf rule `MlfmmParams::automatic_leaf_size` / `leaf_rule_params()`: leaf edge >=
  max(min_box_size_lambda, lambda/4 (d0 <= 3) or lambda/2 (d0 > 3), r_max / ratio); Python key
  `automatic_leaf_size`. `mlfmm::kMaxOctreeLevels` replaces the literal 21. Tests:
  `tests/unit/test_mlfmm_policy.cpp` (lossless expansion, very lossy truncation, Ag exact
  fallback with per-region complementarity to 1e-12 when forced exact, leaf rule, pool failure
  injection, basis patterns), Ag icosphere / rough box matvec vs dense and Ag GMRES through
  `Simulation` (validation-large), 2N = 61 440 icosphere exact-rows case, memory guards.

### Removed
- python (WP25): PML-route refusals use one rule: a valid call in the wrong state raises RuntimeError
  (`bistatic_rcs`, `field` changed from ValueError), invalid arguments raise ValueError.
- io / core (WP23): the C++ `io::open_hdf5` stub, the CMake option `SPECKLEBEM_ENABLE_HDF5` and
  `has_hdf5()` (C++ and Python); HDF5 is written by the Python package only, the C++ library keeps its `.npy` writer.

### Fixed
- operator (WP22a-f, WP22a review follow-ups): `SparseOperator(Matrix&& Z)` takes the near-field
  matrix by rvalue reference and swaps it in; the former by-value parameter still copied it
  (Eigen 3.4.0). Peak working set of the Ag sphere d = 2 um, octa n = 6 (2N = 98 304, near
  3.07 GB): 8.47 -> 5.54 GB, i.e. 2.75 -> 1.80 x the near field. Test: the operator owns the
  caller's arrays (same pointers). tests: `available_memory_bytes()` reads `MemAvailable` from
  `/proc/meminfo` on Linux (fallback: free pages), 0 on macOS (the guarded cases SKIP), with a
  unit test; `make_octasphere`/`edge_stats` moved to the light `tests/support/octasphere.hpp`;
  `system_memory.cpp` is the static library `specklebem_test_support` (`tests/support/`), linked
  by the test executables and `specklebem_ag_sphere_mlfmm` (also without the tests).
- operator (WP22a): `SparseOperator` swaps the matrix in instead of `Z_(std::move(Z))`, which
  copied it (Eigen 3.4.0's `SparseMatrix` has no move constructor): one transient copy of the
  MLFMM near field less (peak RSS of the 393 216-unknown Ag sphere: -12 GB).
- mlfmm (WP21, WP20b review): the apply-workspace pool builds a workspace completely before
  counting it and reserves pool capacity in `take()`, so a failed allocation leaves the pool
  consistent and returning a workspace cannot allocate; `describe()` states the workspace count
  consistently with `memory_bytes()` (at least one counted); `Simulation` wraps only
  `TruncationOrderError`, with advice per cause.
- mlfmm (WP20b): `mlfmm::MlfmmOperator` = Z_near + Z_far (`compression/mlfmm/mlfmm_operator.hpp`):
  owns the octree (leaf floor in wavelengths of R1), the far operator (built first, so that
  unusable configurations fail before the near assembly) and the exact near field
  (`assemble_near`); accessors `near_operator()`, `far_operator()`; `describe()` with octree,
  near nnz / memory / time, far per-level data and totals; `memory_bytes()` includes the
  interpolator tables and the pooled apply workspaces. `MlfmmStrategy::build` returns it.
  `MlfmmFarOperator`: a mutex-guarded pool of reusable apply workspaces (one block of
  uninitialised storage; each pass zeroes the box segments it accumulates into, by the owning
  thread) instead of per-call allocation and zero-fill: apply of a 2N = 61 440 icosphere
  (1 GB of fields) 1.52 s -> 0.66 s; `workspace_bytes()`, `workspaces()`; explicit
  `std::invalid_argument` when r_max >= leaf edge (far pairs could share a triangle, whose K jump
  term Z_far omits). `SphereInterpolator::memory_bytes()`. `Simulation`: compression "mlfmm"
  (GMRES only; Jacobi diagonal still from `assemble_diagonal`, bitwise equal to the near field's
  diagonal), `SimulationConfig::mlfmm` checked at construction, no dense size limit, a failed
  order search (e.g. Ag interior) is rethrown with the remedies; Python `compression="mlfmm"`
  and `mlfmm=dict(accuracy_digits=, max_elements_per_leaf=, min_box_size_lambda=)`. Tests:
  `tests/unit/test_mlfmm_operator.cpp` (near/far complementarity per column, Jacobi diagonal,
  two `std::thread`s vs serial applies, reused workspaces on 4 levels, guard, describe/memory),
  Simulation and Python cases, `tests/validation/test_mlfmm_simulation.cpp` (GMRES MLFMM vs
  dense: currents and RCS), `tests/validation_large/test_mlfmm_vs_dense_large.cpp` (matvec vs
  dense, 2N = 1.5e4 ... 2.5e4, icospheres and rough boxes, n = 1.5 and Si, PMCHWT and ICTF:
  d₀ = 3 with λ/4 leaves 4.3e-5 ... 2.1e-4, d₀ = 5 with λ/2 leaves (r_max/a ≈ 0.29) 3.7e-6 ...
  6.7e-6 and with λ leaves 5.5e-7; 2N = 8.8e4 against 128 exact rows 2.9e-4; Simulation GMRES
  on a 2N = 7680 rough box); slow `[mlfmm_apply_timing]`.
- mlfmm (WP20a): `mlfmm::MlfmmFarOperator` (`compression/mlfmm/far_operator.hpp`), the
  multilevel FMM far part Z_far of the full 2N × 2N system (both regions, all four blocks with the
  formulation weights a_i/η_i, b_i η_i, b_i/η_i; all leaf pairs that are neither the same leaf nor
  adjacent). Per region two aggregated vector fields (Σ x_J V, Σ x_M V), K via k̂ × at reception;
  leaf aggregation, upward pass (Lagrange interpolation with odd pole parity, shift
  e^{+jk k̂·(c_child − c_parent)}), translation with weighted order-truncated translators per
  (level, region, integer offset), downward pass (shift e^{−jk k̂·(c_child − c_parent)}, then
  anterpolation of the weighted field), unweighted leaf reception through the antipodal map.
  Orders per region and level from `search_truncation_order` with the enlarged diagonal (leaf:
  `leaf_sampling`, oversampled to p − 1); all searches run before any setup, and a region/level
  without an achievable order throws `std::runtime_error` (lossy policy: WP21). OpenMP over boxes,
  per-call buffers (concurrent `apply` safe), bitwise deterministic for any thread count;
  `describe()` reports boxes, orders, directions, translators, field memory and setup time per
  level. Tests (`tests/unit/test_mlfmm.cpp`): far-only and full-matvec error vs the dense matrix
  (4-level icosphere and rough box, release), equality with the single-level `far_block` sums at
  3 levels, complex symmetry of S Z_far (interpolation/anterpolation adjointness) and thread
  determinism, error cases; slow sweep `[mlfmm_far_sweep]` (errors vs leaf size, d₀, interior).
- mlfmm (WP19b): `mlfmm::RadiationPatterns` (`compression/mlfmm/patterns.hpp`): the
  (θ̂, φ̂) components of the RWG radiation patterns V_n(k̂) = ∫ (I − k̂k̂) f_n e^{+jk k̂·(r' − c)}
  of one region relative to the leaf-box centres, for every basis (Morton order) and direction,
  stored contiguously as (basis, direction, component); W_n = k̂ × V_n implicit
  (W_θ = −V_φ, W_φ = V_θ), receiving patterns R_m(k̂) = V_m(−k̂) through the antipodal index map
  (θ → π − θ, φ → φ + π; R_θ = V_θ(q'), R_φ = −V_φ(q')), no conjugation for complex k.
  Per-triangle Dunavant degree from |k| h (`pattern_quadrature_degree`, calibrated envelope
  3 (|k|h/4)^{d+1}/(d+1)!), OpenMP over leaf boxes. `far_block`: single-level FMM L and K^PV
  blocks between two interaction-list leaves, L = (ωμk/16π²) Σ w T R·V, K = (k²/16π²) Σ w T R·W
  (derivation in `src/compression/mlfmm/patterns.cpp`); `max_support_radius` and
  `leaf_sampling` (order search with the enlarged diagonal √3 a + 2 r_max, L_leaf =
  max(L, p − 1)); `translator(k, r, sampling, order)` overload (truncation below the sampling
  order). Tests: quadrature convergence (|k| h ≤ 3, real and Si-like k), antipodal map vs direct
  evaluation (1e-13), single-level FMM vs dense `element_blocks` on icospheres and a rough box
  (vacuum, n = 1.5, Si; d0 = 3 at a ≥ 0.75 λ; flipped K sign gives error 2), invalid input; slow
  sweep (`[patterns_sweep]`) with the error table up to d0 = 5. Finding: at a = λ/2 the nearest
  pairs exceed 10^-d0 (1.3e-3 at d0 = 3) although the statistical order search reports the
  target achievable; d0 = 5 needs a ≳ 1.5 λ.
- operator / mlfmm (WP19a): near field Z_near of the MLFMM. `op::SparseOperator` (row-major
  CSR with `Index` indices; `apply` OpenMP over rows, bitwise identical for any thread count,
  x and y may alias; `describe`, `memory_bytes`, `nonzeros`). `op::BasisPattern` (basis-pair
  pattern with row groups sharing one column list) and `op::assemble_sparse(Problem, pattern)`:
  the exact Galerkin entries of all four blocks (both regions, formulation weights, jump terms
  of coincident triangles) of every pattern pair, from the dense assembler's colour schedule
  (dynamic largest-first, shared helpers `test_schedule` / `for_each_test_triangle`) and pair
  blocks, visiting only the source triangles of the pattern pairs; every stored entry is bitwise
  equal to the `DenseStrategy` entry (same contributions, same order). `mlfmm::near_pattern`
  (one group per leaf: the leaf and its near list, ADR 0008 §1; checks that the octree belongs
  to the space) and `mlfmm::assemble_near`; the octree-to-pattern step lives in mlfmm because
  mlfmm sits above the operator layer. Tests (`[near_field]`): stored entries bitwise equal to
  the dense matrix and the stored pattern equal to the near pairs from the leaf `ijk` on an
  icosphere and a rough box (Si, Ag, n = 1.5; PMCHWT, ICTF and a single-region system with
  non-cancelling jump terms), `apply` against the masked dense product (1e-13) and across thread
  counts, `DenseStrategy` bitwise equal to a serial copy of the pre-WP19a algorithm, input
  errors.
- mlfmm (WP17): `mlfmm::Octree` over the RWG edge midpoints: root cube from the padded mesh
  vertex bounding box, anchored at its lower corner (flat or thin geometry stays one box layer
  thick), uniform depth (all leaves on the finest level; the first level on which every box
  holds <= `max_elements_per_leaf`, bounded by `max_levels` <= 21 and the floor
  `(1 - kMinBoxSizeTolerance)`·`min_box_size_lambda`·λ with `kMinBoxSizeTolerance = 1e-2`, so
  the 4 µm sphere at 500 nm, root 8 λ, reaches λ/4 leaves on 6 levels), Morton-ordered boxes
  per level with integer coordinates `Box::ijk`, contiguous element ranges (`permutation()` /
  `inverse_permutation()`), `find_box(level, ijk)`, near lists (same-level neighbours, self
  excluded) and interaction lists (<= 189, levels >= 2), `summary()`. Tests check the
  permutation, range nesting, Morton order, root box, box geometry and leaf criteria on
  icospheres, a thin rough box and flat plates (exactly flat and with ±1e-15 m jitter: one box
  layer, identical leaves), the near and interaction lists against an independent pairwise
  classification from `ijk`, list symmetry and completeness over all leaf pairs (each pair near
  or translated on exactly one level), the λ/4 floor (4 µm sphere regression) and `max_levels`
  bounds, input errors, and the build time (122,880 basis functions in ~0.03 s, release).
- solver / tooling (WP15, Phase 3 DoD): rough-box helpers `default_box_depth(object, λ)` =
  max(10 intensity absorption lengths δ/2, 2 µm) (`kMinBoxDepth`; Si 5.65 µm, Ag 2 µm) and
  `default_box_fine_depth(object, λ, σ)` = 3δ + 3σ in `simulation.hpp` (ADR 0006; a lossless
  dielectric object throws `std::invalid_argument`), with `[simulation]` unit tests. Study
  executable `benchmarks/formulation_convergence.cpp` (`SPECKLEBEM_BUILD_BENCHMARKS`): PMCHWT,
  ICTF and MCTF without and with left Jacobi on Si and Ag Gaussian rough surfaces (σ = 50 nm,
  Lc = 500 nm, h = 50 nm, L = 3.2 µm, 2N ≈ 3·10⁴, dense, full GMRES, tol 1e-3); record
  `benchmarks/results/formulation_convergence.md` and histories
  `benchmarks/results/formulation_convergence_histories.csv`. Si: ICTF 219, MCTF 254, PMCHWT 315,
  Jacobi 272 iterations; Ag: ICTF 622, MCTF 651, PMCHWT 886, Jacobi 1069; the three
  Jacobi-preconditioned formulations iterate identically (≤ 5e-13, issue #15). Fu et al.'s
  "PMCHWT does not converge" and "Jacobi best for Ag" are not reproduced at this size.
- mlfmm (WP18): plane-wave machinery (`compression/mlfmm/plane_wave.hpp`, `interpolation.hpp`):
  `truncation_order` (ADR 0008 §2, Re k bandwidth), `SphereSampling` (Gauss-Legendre x uniform
  phi, contiguous SoA directions / theta_hat / phi_hat, weights summing to 4 pi),
  `spherical_hankel2` (h_l^(2) of complex argument, Re z > 0, Im z <= 0, upward recurrence,
  measured <= 1.4e-14 relative for l <= 120, |z| in [0.05, 80]), `legendre_p`, the diagonal
  translator `translator` (exp(+jwt), prefactor -jk/4pi verified by the addition theorem),
  `expansion_error` (worst-case corner-to-corner check over the nearest interaction-list
  offsets, building block of the ADR 0008 §6 lossy-region policy; Ag-like k gives >> 1) and
  `SphereInterpolator` (separable local Lagrange, polar reflection with even/odd parity for
  scalar / theta-phi components, anterpolation = exact transpose). Measured: random points
  meet 10^-d0 for box edges >= lambda/2 (d0 = 3, 5; Si-like k from lambda/4), lambda/4 boxes
  lose up to one digit; the corner worst case stays above 10^-d0 at the ADR order (1e-3 ...
  9e-2); 0.1 x 10^-d0 interpolation needs p ~ 12-14 (d0 = 3) / ~ 20-22 (d0 = 5), not p = 6.
- mlfmm (WP18 review, ADR 0008 amendments): `interpolation_order(d0)` (measured table: 14 for
  d0 <= 3, 22 for d0 <= 5, throws above 5) and `leaf_sampling_order(L, p)` = max(L, p - 1);
  `SphereInterpolator` has no default order any more and accepts any p <= 2 n_theta_src
  (stencils over the pole-reflected nodes); the weight convention of the disaggregation (apply
  the parent weights before anterpolating) is documented. `expansion_error` gains the seeded
  statistical mode (`ExpansionCheck::random`, `pairs`, `seed`; platform-independent points) and
  `search_truncation_order` implements the per-level order search (smallest L >= formula
  meeting 10^-d0, breakdown detection). Measured: real k needs formula + 1 only at lambda/4 for
  d0 = 5; lambda/4 at d0 = 7 is not achievable (minimum 4e-7); Ag-like k reaches d0 = 3 for
  boxes <= 0.25 lambda_0 at L = 9 ... 22 and breaks down from 0.5 lambda_0. `spherical_hankel2`
  raises `std::underflow_error` when e^{-jz} underflows (instead of returning zeros), the
  translator raises `std::overflow_error` for non-finite coefficients or values. New tests:
  complex-z Hankel check up to |z| = 80 against Miller j_l / upward y_l with their Wronskian,
  lambda/4 interpolation with the oversampled leaf, order search, and a `slow` sweep
  (`mlfmm: plane-wave accuracy sweep (slow)`) printing the measured tables.
- kernels / operator (WP-P2): dense assembly performance on rough boxes. Profile benchmark
  `benchmarks/dense_assembly_profile.cpp` (per-class element_blocks timings, near/far degree and
  touching-rule statistics, schedule makespan, same-process before/after comparison) and record
  `benchmarks/results/dense_assembly_profile.md`. Root causes: OpenMP static schedule (one
  thread got all coarse box rows, ×2.5 / ×4.5 the ideal time), degree cap 19 for every object-
  region pair with a 400 nm box cell, C-library sincos/exp in the near/far loop. Fixes: dynamic
  largest-first schedule (bitwise identical matrix); `OperatorOptions::decay_aware_target`
  (default on; WP7c part 4: lossy regions bound the near/far error by target × the undamped
  magnitude bound of the block, ADR 0004; lossless regions bitwise unchanged);
  `OperatorOptions::fast_plain_kernel` (default on; branch-free vectorised sin/cos/exp in
  `kernels/fast_math.hpp`, ≤ 2 ulp (measured 1), blocks within 2.7e-15; operators.cpp built with
  `-fno-math-errno -fno-trapping-math`); `kernels::plain_rule_degree` (diagnostics); Python
  `kernels=dict(decay_aware_target=..., fast_plain_kernel=...)`. WP15 systems: Si 1513–1535 s → 242 s, Ag 1056–1293 s
  → 35 s (24 threads); the Si box stays ~5–10× the N²-scaled sphere figure because of its coarse
  box cells (|k_Si| h ≈ 30, see the record).
- python (WP14b3): calls on one `Simulation` from several Python threads are serialised by a
  per-object timed mutex taken with the GIL released (re-entrant calls from a `solve()` callback
  raise `RuntimeError`; from inside a callback another busy `Simulation` raises "Simulation
  busy" instead of waiting, so two solves calling into each other cannot deadlock; a main
  thread waiting for the lock checks for Ctrl-C every 50 ms); GMRES solves are interruptible
  with Ctrl-C (`PyErr_CheckSignals()` per iteration with a callback, every 50 ms on the main
  thread without one); `exterior=None` now means `excitation.background` (was vacuum). Tests
  for concurrent `solve()` (alternating tolerances) against `field()` / `bistatic_rcs()` /
  `currents` / `report()` with exact comparison to the two possible results, a concurrent first
  `solve()` on an unassembled Simulation, cross-Simulation callbacks, re-entrancy, interruption
  of a solve and of a lock wait via `_thread.interrupt_main()`, a dielectric background and the
  `SolveResult.x` keep-alive. `op::LinearOperator::apply` documents that it must be safe to
  call concurrently. CI sets `SPECKLEBEM_REQUIRE_SCIPY=1`, which makes the SciPy test fail
  instead of skip without SciPy.
- python (WP14b2): `Simulation` (keyword mapping onto `SimulationConfig`: formulation,
  preconditioner, solver, compression, `gmres=dict(...)`, `kernels=dict(...)`; unknown strings
  and keys raise `ValueError`), `assemble()` / `solve(tol, max_iter, restart, side, callback)`
  with the GIL released and Python exceptions from the callback propagated, `SolveResult`,
  `report()`, resolved `formulation` / `preconditioner`, `currents` and `rhs()` (copies),
  `operator()` as `LinearOperator` (`matvec`, `@`, `as_scipy()`; keeps the Simulation alive),
  `field`, `far_field`, `bistatic_rcs`; free functions `plane_grid`, `cylinder_grid`,
  `intensity`, `polarized_intensity`, `differential_reflection_coefficient`; mesh file I/O
  (`read_mesh` / `write_mesh` and the STL / OBJ / Gmsh variants) and `open_npy_directory` /
  `ResultWriter` with `str` / `os.PathLike` UTF-8 paths. Tests `tests/python/test_simulation.py`
  (GMRES vs direct, Mie n = 1.5 at icosphere n = 2 with eps_rr < 2 %, callbacks, keep-alive,
  SciPy GMRES on `as_scipy()`, GIL release) and `test_io.py` (round trips under a non-ASCII
  directory). CI installs SciPy (apt `python3-scipy`, MSYS2 `python-scipy`).
- python (WP14b1f): `TriangleMesh.quality()` returns `MeshQuality` (with `__repr__`) and
  `quality_report()` the C++ text report, as in C++; NumPy-style docstrings with units and
  conventions on every bound class, function and method; `ValueError` for non-finite points in
  field evaluation (`electric_field`, `magnetic_field`, Mie fields) and non-finite angles in
  `bistatic_rcs`. Tests for input layouts (int32/uint32, Fortran order, strided views, nested
  lists, empty mesh), keep-alive of `Mie.a_n`/`b_n` and docstring units.
- kernels (WP7c): fold-adaptive outer rule for shared-edge and shared-vertex pairs
  (`OperatorOptions::fold_adaptive`, default on): the test triangle is cut into Duffy pieces at
  the singular directions through the shared vertex (projected source edges of folds below
  90 degrees, obtuse apex angles), with the points of level l + 2 / l + 3 on the pieces of
  doubly obtuse shared edges (both triangles obtuse at the same vertex) and of far vertices
  projecting within 0.2 |AB| of a shared vertex; touching-pair error <= 8.3e-8 for dihedral
  angles 30 to 179 degrees with regular, skewed, obtuse, doubly obtuse and near-vertex
  triangles (WP7b: up to 4.5e-5 skewed / 4.2e-5 near-vertex / 8.7e-5 obtuse at 30 to 45
  degrees; not covered: near-vertex shapes at folds of 90 to ~95 degrees keep the WP7b table,
  <= 1.5e-7). Cost against the WP7b rule (time per call, folds below 90 degrees): mean x1.9
  (shared edges, at most x3.3 for doubly obtuse pairs) and x1.6 (shared vertices, at most
  x3.2); hard bound 48 pieces x 16 x 16 outer points per ordering. Folds >= 90 degrees whose
  pieces need no split (all Mie icosphere pairs) bitwise unchanged.
  `kernels::touching_rule_info` reports the deterministic outer point counts. Slow dihedral
  sweep and right-hand-side degree study registered as the ctest entries
  `kernels: fold sweep (slow)` and `assembler: rhs degree study (slow)` (label `slow`,
  excluded from CI and the standard runs). The log-Gauss tables are checked against all 2n
  moments.
- operator (WP7c): `OperatorOptions::quad_degree_rhs` (default 8, positive-interior), the
  Dunavant degree of `op::assemble_rhs` (before: `quad_degree_near` = 19); relative error
  <= 7.5e-14 against degree 20 for plane waves and Gaussian beams on lambda/10 meshes.
- python (WP14b1): bindings for `TriangleMesh` (read-only zero-copy `vertices`/`triangles`/`edges`
  views kept alive by the mesh, `quality_report()` as `MeshQuality`), `make_icosphere`,
  `make_sphere`, `HeightMap`, `generate_gaussian_height_map`, `make_rough_surface_mesh`,
  `make_mesh_from_height_map` (keyword arguments incl. `box_depth`, `box_mesh_size`,
  `box_fine_depth`), the Python class `RoughSurface`, `DispersiveMaterial`, `field_decay_length`,
  `PlaneWave` / `GaussianBeam` (`shared_ptr` holders, vectorised `electric_field` /
  `magnetic_field`) and `Mie` (broadcast `bistatic_rcs`, `scattered_E`, cross sections); GIL
  released in mesh generation and field loops; invalid input raises `ValueError`. C++:
  `geometry::MeshQuality`, `TriangleMesh::quality()` and `geometry::to_string(MeshQuality)`
  (`quality_report()` unchanged). The module now builds with the project warning set (GCC's
  `-Wnull-dereference`/`-Wmaybe-uninitialized` false positives in pybind11/Eigen headers are
  disabled per source). Tests `tests/python/test_geometry.py`, `test_physics.py`.
- core (WP-P1, ADR 0007): file paths are UTF-8 strings on every platform.
  `core::path_from_utf8` (validating, via `std::u8string`; malformed UTF-8 or a NUL byte throws
  `std::invalid_argument`), `core::path_to_utf8` (via `path::u8string()`) and
  `core::find_invalid_utf8` in `core/path.hpp`. `geometry::io` mesh readers/writers and
  `io::open_npy_directory` open files only through these helpers and name paths in UTF-8 in
  error messages, so non-ASCII paths work on Windows (previously interpreted in the ANSI code
  page). Tests: `tests/unit/test_path.cpp`, and mesh (STL/OBJ/Gmsh) and `.npy` directory round
  trips under a directory named `sbem_Jürgen_µm_路径_🙂` in `test_mesh_io.cpp` /
  `test_result_writer.cpp`.
- solver (WP14a): `Simulation` driver (`src/simulation.cpp`, pimpl) for the dense strategy. Owns
  mesh, RWG space, formulation, excitation, `op::Problem` (ω of the excitation), operator,
  right-hand side, Jacobi diagonal (built lazily, only for a Jacobi-preconditioned GMRES solve)
  and solution; automatic formulation / preconditioner via `formulation::recommend`
  (`SimulationConfig::formulation` and `diagonal_preconditioner` are `std::optional`);
  `SolverKind::Gmres` (left/right, Jacobi or none; `solve(GmresParams, cb)` overrides
  `config.gmres` for one call, a non-converged solve keeps its solution and logs a warning) or
  `SolverKind::Direct` (dense LU, mapped onto `GmresResult` with `residual_history` =
  {LU residual}); `compression` defaults to `"dense"` (other strategies throw until Phase 4/8);
  constructor checks (closed and outward-oriented mesh, 2N ≤ `op::kMaxDenseUnknowns` (now public),
  wavelength and background vs the excitation, implemented formulation (JMCFIE rejected), kernel
  options, GMRES parameters); idempotent `assemble()`; `report()` with mesh size, choices, memory,
  timings, iterations and residuals; accessors `problem()`, `formulation_kind()`,
  `diagonal_preconditioner()`, `num_unknowns()`, `system_operator()`, `rhs()`. Unit tests
  `tests/unit/test_simulation.cpp`; validation `tests/validation/test_simulation_mie.cpp` (WP11
  dielectric sphere through the driver: direct reproduces the WP11 ε_rr and stays below 1 % vs
  Mie, GMRES/ICTF within 1e-4).
- io (WP16): `io::open_npy_directory`, a dependency-free `ResultWriter` that stores vectors and
  matrices as NumPy `.npy` files (format 1.0, little-endian, C order, `<f8`/`<c16`), meshes as
  `<group>/vertices.npy` (`<f8`) and `<group>/triangles.npy` (`<i8`, 0-based) and attributes in
  one `attributes.json` (rewritten on every call, shortest round-trip Reals, strict UTF-8);
  validated names (`[A-Za-z0-9_.-]`, `/` for groups, Windows device names such as `CON` or
  `com1.x` rejected on all platforms); every file is written to `<file>~tmp` and renamed over the
  target, so targets are always complete; `open_hdf5` throws until HDF5 support lands. Unit tests
  `tests/unit/test_result_writer.cpp` (independent NPY reader, bitwise data); NumPy round trip
  `tests/python/test_result_writer.py` via the helper `specklebem_npy_fixture` (CI sets
  `SPECKLEBEM_NPY_FIXTURE`, so a missing helper fails instead of skipping).
- geometry (WP2c): rough-surface box with optional decay-aware fine band (`box_fine_depth`) and a
  per-column stitched relaxation band under rough rims (vertical wall edges ≤ 2 h_g, M = 3 kept for
  σ ≤ 250 nm, Lc ≥ 100 nm; box 7–14 % of the top face at L = 10 µm); `material::field_decay_length`.
- kernels (WP7b): graded outer quadrature for touching triangle pairs (generalised Gauss-log rules on
  Duffy sub-triangles, `OperatorOptions::outer_grading_levels`, default 4) and k-aware near/far degree
  selection (`target_accuracy`, default 1e-5, within [`quad_degree_far`, `quad_degree_near`] = [3, 19]);
  touching-pair error 2e-4 → 3e-9, class-boundary error 4e-3 → 4e-7; Mie benchmark re-measured on the
  24-core Windows machine (`benchmarks/results/mie_sphere_dense.md`).
- solver (WP13): `solver::gmres`, complex GMRES (Saad & Schultz; MGS Arnoldi with one
  re-orthogonalisation pass, Givens rotations, happy-breakdown and singular-Krylov handling),
  full or restarted (`GmresParams::restart`), left or right preconditioning
  (`PreconditionerSide`, default left: the stopping criterion is the preconditioned residual,
  invariant under row scaling with Jacobi, issue #15), `GmresResult::true_relative_residual`;
  `DiagonalPreconditioner` (validated, stores 1/d). Unit tests `tests/unit/test_gmres.cpp`
  (LU agreement, termination in ≤ n steps, breakdown, restart, row-scaling invariance, right
  preconditioning, errors, dense BEM sphere systems for Si/ICTF and Ag/PMCHWT + Jacobi);
  validation-large case `tests/validation_large/test_gmres_vs_lu_large.cpp` (Si rough-surface
  box, 2N = 19 968, GMRES vs LU).
- tooling (WP-W1): native Windows build with MSYS2: presets `win-release` (UCRT64 GCC, OpenMP,
  OpenBLAS) and `win-debug` (CLANG64 Clang/libc++, ASan + UBSan, `-Werror`), a Windows MSYS2
  UCRT64 CI job, `.gitattributes` (LF checkouts everywhere), Windows memory guard of the dense
  validation tests (`tests/support/system_memory.cpp`). CMake ≥ 3.25 (FetchContent `SYSTEM`);
  fetched Eigen/spdlog/Catch2/pybind11 are system includes; fetched spdlog 1.14.1 → 1.15.3 and
  Catch2 3.7.1 → 3.16.1 (Clang 22 compatibility; the fetched Catch2 runs only in the Windows job,
  Linux CI uses the system Catch2 3.4). CI also runs on pushes to `wp/**` branches.
- geometry (WP1, #1): `TriangleMesh` edge topology (interior edges with plus/minus triangles,
  boundary-edge count, non-manifold and duplicate-triangle rejection), orientation check and
  repair with per-component outward orientation, closedness, signed volume, bounding box,
  quality report; `make_icosphere` / `make_sphere`; unit tests `tests/unit/test_mesh.cpp`.
- reference (WP5, #6): `MieSolution`, Bohren–Huffman Mie series adapted to exp(+jωt) (conjugated
  coefficients and fields), near-field n_max criterion, scattered/internal near fields,
  bistatic RCS, scattering and extinction cross sections; unit tests against the BH sample case,
  the Rayleigh limit and surface boundary conditions; validation tests labelled `validation`.
- basis (WP3b, #4): `RwgSpace`, Rao–Wilton–Glisson basis functions on interior edges with O(1)
  triangle → basis lookup, values, divergences and signed supports; unit tests for normal
  continuity, flux and divergence identities, open meshes and `flip_normals`.
- kernels (WP3a, #3): Dunavant triangle rules of degree 1–20 (re-solved to double precision,
  documented degrees with negative weights or exterior points, `triangle_rule_is_positive_interior`)
  and Gauss–Legendre rules of any order; exactness tests up to degree 20 and n = 100.
- geometry (WP2, #2): Gaussian rough-surface height maps (Eqs. 8–9, FFT and direct convolution,
  exact σ normalisation, Lc as 1/e radius of the autocorrelation, portable RNG), `HeightMap`
  statistics, closed box mesh with rough top (`make_rough_surface_mesh`,
  `make_mesh_from_height_map`); 20-seed statistics test, seed-42 regression reference in
  `tests/data/`, closedness/orientation/timing tests.
- tooling: `.clang-format` puts `<unsupported/Eigen/...>` into the third-party include block;
  `docs/compiler_notes/` documents a GCC 13.3 -O3 loop-vectoriser store-permutation bug that the
  box mesher works around.
- geometry (WP6, #7): mesh import/export for binary and ASCII STL (exact vertex welding),
  OBJ and Gmsh v4.1 ASCII with bit-exact text output, atomic writes, line-ending/BOM tolerance,
  fixtures and adversarial-input regression tests.
- kernels (WP4, #5): analytic static integrals over a flat triangle (∫1/R, ∫(r'−r)/R, ∫R,
  ∫(r'−r)R, ∫∇'(1/R) with solid-angle normal part and in-plane principal value, stable log
  branches, boundary finite-part flag) and `classify` proximity classes; tests against composite
  Dunavant, graded Duffy references, solid-angle jump and invariances.
- excitation (WP8, #9): `PlaneWave` and paraxial `GaussianBeam` (q-parameter form with first-order
  longitudinal components, θ_in rotation about y, p/s polarisation; Maxwell residual (λ/πw₀)²/2);
  solver (WP8): `solve_direct` with column equilibration, iterative refinement and
  `DirectSolveInfo` diagnostics (Eigen LU, BLAS-accelerated when available).
- post (WP10, #11): Stratton–Chu near-field evaluation in R1 and R2 (region by generalised winding
  number), `total_field`, `far_field`, `bistatic_rcs`, grids, polarised intensities and relative DRC;
  the representation signs are verified against exact Mie surface currents projected onto RWG
  (unit test n = 2 → 3, validation test n = 3 → 4, O(h²) convergence).
- kernels (WP7, #8): `element_blocks`, Galerkin L and K blocks for a triangle pair in mixed-potential
  form with singularity subtraction of the 1/R and R terms (ADR 0004), proximity-class quadrature,
  symmetrised touching pairs, series remainder near R = 0, `jump_block` for the ±½ n̂ × f term,
  `validate(OperatorOptions)`; tests against brute-force and polar references, symmetry, static
  limit; hidden `[.slow]` sweeps; allocation test in a separate executable.
- operator (WP9, #10): `DenseStrategy::build` (2N × 2N block system of docs/03 with formulation
  weights and the effective jump terms K₁ = K^PV − ½ I, K₂ = K^PV + ½ I; deterministic OpenMP by
  triangle colouring), `assemble_rhs`, `assemble_diagonal` (bitwise equal to the diagonal),
  `Problem::omega` and `op::validate`; first dense Mie solves: ε_rr = 0.28 % (n = 1.5) and 0.36 % (Ag)
  on an icosphere of 1 280 triangles; tests for symmetry, per-region jump signs, exact-current residuals.
- tests (WP11, #12): Mie validation of the dense solver in both scattering planes with refinement
  (ε_rr 0.89 % → 0.27 % → 0.074 % for n = 1.5, 1.95 % → 0.35 % → 0.078 % for Ag on icospheres
  n = 2/3/4), power balance (optical theorem, far-field integral, surface-current absorption), new
  `validation-large` ctest label (release-only, memory-guarded), results in
  `benchmarks/results/mie_sphere_dense.md`.
- tooling: `.github/workflows/ci.yml` had an invalid flow-mapping `env` since the scaffold so CI
  never ran; fixed. CI runs `-L '^validation$'`; `validation-large` is nightly/manual.
- post (WP9a): `SurfaceSolution::omega` removed; the frequency comes only from `op::Problem::omega`
  (checked against an attached excitation to 1e-12 relative); post-processing tests use
  `excitation::PlaneWave`. `validation-large` tests run serially (RUN_SERIAL).
- geometry (WP2b, #14): graded side walls and bottom plate of the rough-surface box
  (`RoughSurfaceParams::box_mesh_size`, automatic coarse spacing min(depth/2, L/8, 10 h), conforming
  2:1 transition strips, decay-based validity contract, rough-rim guard); reference case L = 10 µm,
  50 nm, depth 2 µm: closing box 7.2 % of the top face instead of 180 % (224 000 → 85 750
  triangles); uniform fallback bit-identical to the previous mesh.
- build: fetched Eigen is a system include and no longer pollutes the CMake package registry.
- tooling: coordinator/worker/reviewer agent definitions, `CLAUDE.md`, `docs/backlog.md`.
- Project scaffold: CMake build with presets, dependency resolution, warnings, CI (GCC/Clang, Release/Debug+sanitizers).
- Public header layout for all planned layers (geometry, basis, kernels, formulation, operator, compression, solver, postprocessing, reference, backend, io).
- Implemented: core types and constants, logging, timer, reference materials (Ag/Si at 500 nm), dispersive material interpolation, PMCHWT/ICTF/MCTF weights and material-based recommendation, scalar Green's function and gradient, dense and sum operators, CPU backend.
- pybind11 module with materials and formulation enums; Python package skeleton.
- Unit tests (Catch2) and Python tests (pytest).
- Documentation: project plan, architecture, SIE and MLFMM theory, validation, conventions, coding guidelines, GPU strategy, compression survey, Python API, ADRs.

### Changed
- solver (WP-LU1): `solve_direct` raises `std::invalid_argument` (Python `ValueError`) for a matrix with non-finite
  entries instead of `std::runtime_error`; singular matrices still raise `std::runtime_error`.
- core / python / tests (WP-B3, ADR 0006 amendment 2026-10-12b): **uniform closing box by
  default.** `rough_surface_box_params(object, background, wavelength, sigma, mesh_size,
  grading = BoxGradingMode::Uniform)` returns `box_mesh_size = mesh_size` (box cells = top-face
  cells, M = 0); `BoxGradingMode::Graded` keeps the λ₁/5 box of the 2026-10-10 amendment
  (`box_mesh_size` unset, generator cap, uniform fallback). The fine band is set for weakly
  absorbing objects in both modes (with the uniform box it does not change the mesh; dropped
  without a warning if it would reach the bottom). `RoughBoxParams` gains `grading` and
  `beam_waist_fraction`. **`default_box_mesh_size` is renamed `graded_box_mesh_size`** (same
  values; removed under the old name in C++ and Python so that no caller silently reads the
  graded value as the default). New `is_weakly_absorbing(object, wavelength, mesh_size)` (3δ > 2h,
  the fine-band criterion), `is_plasmonic(object)` (Re ε_r < 0), `beam_waist_fraction(object,
  wavelength, mesh_size)` (1/6 weakly absorbing, e.g. Si; 1/4 otherwise, e.g. Ag) and
  `kWeakAbsorberBeamWaistFraction`. `BeamWaistOptions::max_fraction` (last member, so that a positional
  `{theta_in, allow_wide}` keeps its meaning; default
  `kMaxBeamWaistFraction` = 1/4, must lie in (0, 1/4]); the error and warning name the fraction
  ("L cos(theta_in) / 6"). `rough_surface_box_params` logs one SBEM_WARN for a plasmonic object
  with sigma > 0 (no quantitative rough-surface results until WP-V3; flat patches, sigma = 0, stay
  silent, no exception). Python mirrors all of it (`grading="uniform"|"graded"` or
  `sb.BoxGradingMode`, keyword-only; `check_beam_waist(..., max_fraction=...)`;
  `sb.MAX_BEAM_WAIST_FRACTION`, `sb.WEAK_ABSORBER_BEAM_WAIST_FRACTION`). Callers: the Fresnel flat
  support (`fresnel_flat_support.hpp`, WP22c/WP-A1 records) pins `Graded` and checks the waist with
  the material fraction (the recorded Si validation-large cases at L/4 now pass `allow_wide`); the
  MLFMM scaling support pins `Graded` (its coarse spacing stays explicit) and its `waist_factor`
  is optional, defaulting to 1 / `beam_waist_fraction` (Si L/6, Ag L/4; the recorded Si rows used
  L/4: `--waist-factor 4 --allow-wide-beam`); example 02 uses the Si fraction. The WP22d server
  run kit keeps L/4 for every run (coordinator decision: a scaling measurement comparable with
  the proxy and the WP22b3 series): `run_large_area.sh --waist-factor` (default 4) is always
  forwarded, with `--allow-wide-beam` for Si below a factor of 6; the recorded Si reproduce
  commands in `benchmarks/results/` name the waist explicitly.
- kernels (WP7d): the fold-adaptive rule classifies right angles at shared vertices (rough
  top faces, box rim) with a relative tolerance of 1e-6 instead of exact sign / distance tests,
  so exact and +-1e-9-perturbed right angles select the same rule (that of acute angles);
  shared-vertex pairs with a steep source triangle that has a vertex less than 11.5 degrees
  above the test plane (box rim) use the adaptive piece (rim K error 1.2e-7 -> 2e-10); with the
  ray partition and S within 0.3 |AB| of a shared vertex the pieces of the other apex get the
  points of level l + 3 (right angles at A, 75-degree fold: 3.8e-7 -> 4.7e-8). Icosphere
  blocks are bitwise unchanged. Tests: right-angle shapes in the fold sweep and fast rule /
  accuracy checks, top-face and rim pairs of a rough box against the reference, the near-B
  deviation pinned, the 16 x 16 point cap at grading levels 1 to 6. Doc fixes (ADR 0004).
- material (WP14b1f): `DispersiveMaterial` throws `std::invalid_argument` for non-finite,
  non-positive or not strictly increasing wavelengths, non-finite indices and indices with
  `Im(n) > 0` (the message says that exp(+jwt) needs `n - jk` and that `n + ik` optics data must
  be conjugated); `at_wavelength` throws for a non-finite wavelength and now accepts the table
  endpoints. Python: these raise `ValueError`.
- python (WP14b1f): `TriangleMesh.quality_report()` returns the text report (`str`) instead of a
  `MeshQuality`; use `quality()` for the numbers. `quality()`/`quality_report()` no longer
  release the GIL.
