# Python API (target)

The Python package is the primary user interface. It is intentionally small: build a mesh, define illumination and materials, run, post-process. Everything returns NumPy arrays; results can be dumped to HDF5.

```python
import numpy as np
import specklebem as sb

# --- geometry --------------------------------------------------------------
mesh = sb.make_sphere(radius=2e-6, target_edge_length=500e-9 / 27)
surf = sb.RoughSurface(L=30e-6, sigma=100e-9, Lc=1e-6, mesh_size=50e-9, seed=42)
mesh = surf.mesh                                  # sb.TriangleMesh
hm   = surf.height_map                            # (n, n) ndarray, metres
mesh = sb.read_mesh("part.stl")                   # general closed meshes (Phase 8)
mesh.vertices, mesh.triangles                     # zero-copy views
mesh.quality_report()

# --- materials -------------------------------------------------------------
ag = sb.silver_500nm()                            # eps_r = -9.794 - 0.313j  (exp(+jwt))
si = sb.Material(eps_r=18.478 - 0.606j)
au = sb.DispersiveMaterial.from_refractiveindex_info("Au_JohnsonChristy.csv").at(500e-9)

# --- excitation ------------------------------------------------------------
pw   = sb.PlaneWave(wavelength=500e-9, direction=[0, 0, 1], polarization=[1, 0, 0])
beam = sb.GaussianBeam(wavelength=500e-9, waist=5e-6, polarization="p", incidence_angle=np.deg2rad(5))
beam = sb.AngularSpectrumBeam(wavelength=500e-9, waist=5e-6, polarization="p",
                              incidence_angle=np.deg2rad(5), region_radius=20e-6)  # rigorous

# --- simulation ------------------------------------------------------------
sim = sb.Simulation(mesh, beam, object=si, exterior=sb.vacuum(),
                    formulation="ICTF",           # "PMCHWT" | "ICTF" | "MCTF" | "auto"
                    preconditioner="none",        # "none" | "diagonal" | "block" | "hlu"
                    compression="mlfmm",          # "dense" | "mlfmm" | "aca" | "hmatrix"
                    mlfmm=dict(accuracy_digits=3, max_elements_per_leaf=100),
                    backend="cpu")                # "cpu" | "cuda"
sim.assemble()
res = sim.solve(tol=1e-3, max_iter=2000, restart=None)
res.iterations, res.residual_history, res.wall_seconds
sim.report()                                      # levels, memory, timings

# The operator is also available for SciPy:
Z = sim.operator()                                # sb.LinearOperator with .matvec / .shape
import scipy.sparse.linalg as spla
spla.gmres(Z.as_scipy(), sim.rhs(), rtol=1e-3)

# --- fields and observables -----------------------------------------------
pts = sb.plane_grid(origin=[0, 0, -1e-3], u="x", v="y", size=300e-6, n=301)
E, H = sim.field(pts, kind="scattered")           # (n_pts, 3) complex
I = sb.intensity(E)
rcs  = sim.bistatic_rcs(plane="xz", angles=np.linspace(0, np.pi, 181))
drc  = sim.drc(radius=0.5, theta=np.linspace(-50, 50, 201), y=np.linspace(-0.25, 0.25, 500))
co, cross = sb.polarized_intensity(E, co_pol=[1, 0, 0])

# --- speckle statistics ----------------------------------------------------
sb.speckle_contrast(I)
sb.intensity_pdf(I, bins=64)
sb.correlation_coefficient(I_0deg, I_5deg)
mu_A = sb.field_correlation_from_intensity(I.reshape(301, 301))
sb.speckle_size(mu_A, pixel_size=1e-6)

# --- reference & I/O -------------------------------------------------------
mie = sb.Mie(radius=2e-6, wavelength=500e-9, material=ag)
mie.bistatic_rcs(theta, phi); mie.scattered_E(pts)
sim.save("run.h5")                                # mesh, currents, config, history
sb.load_fields("run.h5")
```

## Bound so far (WP14b1) and differences from the target

Geometry, materials, excitations and the Mie reference are bound; `Simulation`, operators,
post-processing and I/O followed in WP14b2 (next section).
Actual signatures (all lengths in metres, angles in radians):

```python
sb.TriangleMesh(vertices, triangles)    # (V,3) real, (F,3) integer; copied; ValueError if invalid
mesh.vertices, mesh.triangles, mesh.edges   # read-only zero-copy views (base = the mesh)
mesh.num_vertices, mesh.num_triangles, mesh.num_edges, mesh.num_boundary_edges, mesh.num_components
mesh.is_closed(), mesh.signed_volume(), mesh.bounding_box(), mesh.flip_normals()
mesh.quality()                          # sb.MeshQuality: attributes; str(q) == mesh.quality_report()
mesh.quality_report()                   # str: the C++ multi-line text report
sb.make_icosphere(radius, subdivisions, center=(0, 0, 0))
sb.make_sphere(radius, target_edge_length, center=(0, 0, 0))
sb.generate_gaussian_height_map(*, L, sigma, Lc, mesh_size, seed=0, use_fft=True)  # sb.HeightMap
sb.HeightMap(z, dx, dy)                 # z: (n_x, n_y) heights at (x_i, y_j); .z (copy), .rms()
sb.make_mesh_from_height_map(height_map, *, box_depth=None, box_mesh_size=None, box_fine_depth=None,
                             exterior_wavelength=None)
sb.make_rough_surface_mesh(*, L, sigma, Lc, mesh_size, seed=0, box_depth=None,
                           box_mesh_size=None, box_fine_depth=None, exterior_wavelength=None,
                           use_fft=True)
sb.RoughSurface(L, sigma, Lc, mesh_size, seed=0, *, box_depth=None, box_mesh_size=None,
                box_fine_depth=None, exterior_wavelength=None, use_fft=True)
                                        # .mesh (lazy), .height_map, .heights, .dx, .dy
sb.rough_surface_box_params(object, background, wavelength, sigma, mesh_size)  # sb.RoughBoxParams
p.box_depth, p.box_fine_depth (None: no band), p.box_mesh_size (None: automatic, capped),
p.exterior_wavelength, p.uniform_fallback, p.kwargs()
sb.default_box_depth(object, wavelength), sb.default_box_fine_depth(object, wavelength, sigma)
sb.default_box_mesh_size(background, wavelength, mesh_size), sb.exterior_wavelength(background, wavelength)
sb.check_beam_waist(patch_length, waist, allow_wide=False)   # ValueError if waist > L / 4
sb.field_decay_length(material, wavelength) / sb.field_decay_length(eps_r, wavelength)
sb.DispersiveMaterial(wavelengths, refractive_indices)   # n - jk; .at(wl) == .at_wavelength(wl)
sb.PlaneWave(wavelength, direction, polarization, background=sb.vacuum())
sb.GaussianBeam(wavelength, waist, polarization="p", incidence_angle=0.0, focus=(0, 0, 0),
                background=sb.vacuum())
sb.AngularSpectrumBeam(wavelength, waist, polarization="p", incidence_angle=0.0, focus=(0, 0, 0),
                       background=sb.vacuum(), *, tolerance=1e-10, region_radius=None,
                       polar_order=0, azimuth_order=0, max_plane_waves=2_000_000)
beam.num_plane_waves, beam.polar_order, beam.azimuth_order, beam.grid_change,
beam.max_polar_angle, beam.region_radius, beam.power, beam.waist, beam.tolerance
exc.electric_field(points), exc.magnetic_field(points), exc.omega, exc.wavelength, exc.background
exc.fields(points)                      # (E, H); one pass for AngularSpectrumBeam
exc.controlled_radius, exc.controlled_center   # inf / origin except AngularSpectrumBeam
sb.Mie(radius, wavelength, material, exterior=sb.vacuum(), n_max=0)
mie.bistatic_rcs(theta, phi)            # NumPy broadcasting; a float for scalar input
mie.scattered_E(points), mie.scattered_H(points), mie.internal_E(points)
mie.scattering_cross_section(), mie.extinction_cross_section(), mie.a_n, mie.b_n
```

- Defaults of the rough-surface keywords are those of `geometry::RoughSurfaceParams`
  (`L` = `edge_length_L`, `sigma` = `rms_roughness`, `Lc` = `correlation_length`); `seed=0`
  draws a non-deterministic seed. `RoughSurface` validates the box keywords when `.mesh` is
  first built.
- Closing box (ADR 0006 with the 2026-10-10 amendment): `rough_surface_box_params` chooses the
  box from the materials and is the recommended way to build a rough patch for a simulation:
  `sb.RoughSurface(L, sigma, Lc, h, seed, **sb.rough_surface_box_params(obj, bg, wl, sigma, h).kwargs())`.
  It sets the depth (`default_box_depth`), `exterior_wavelength` (lambda_1 = wavelength /
  |n_1|, conservative for a lossy background; a metallic background, Re(eps_r) <= 0, raises
  `ValueError`) and, for weakly absorbing objects, the mandatory fine band 3 delta + 3 sigma.
  "Weakly absorbing" is 3 delta > 2 h: it mirrors the grading contract (the walls without a
  band start to coarsen 2 top-face spacings below a smooth rim); Si yes, Ag no at 500 nm and
  h = 50 nm. `box_mesh_size` stays `None`: with `exterior_wavelength` the generator caps its
  automatic coarse spacing at the largest 2^M h <= lambda_1 / 5 with the actual grid spacing
  h = L / round(L / mesh_size) (100 nm at 500 nm in vacuum with h = 50 nm), so the coarse
  spacing never exceeds lambda_1 / 5 and no warning is logged. `default_box_mesh_size` gives
  the nominal value for `mesh_size`; do not pass it as `box_mesh_size` (for L not a multiple
  of h it can exceed lambda_1 / 5 slightly, which warns). If the band would reach the bottom
  plate (very rough surfaces), the uniform box is chosen: `box_mesh_size = mesh_size`,
  `uniform_fallback = True`, logged as a warning. Without `exterior_wavelength` and
  `box_mesh_size` the old automatic rule (min(depth / 2, L / 8, 10 h)) is used with a warning
  (not validated under illumination). lambda_1 / 5 is necessary, not shown sufficient (WP-V2).
- Beam waist: rough-surface drivers must call `check_beam_waist(L, waist)` (w0 <= L / 4,
  ADR 0006 amendment); `Simulation` cannot determine the patch size from a mesh.
  `allow_wide=True` accepts a wider waist with a warning.
- `PlaneWave.direction` is `k_hat` (normalised by the constructor), `polarization` the complex
  amplitude `e0` [V/m] (transverse). `GaussianBeam.waist` is `Params::waist_radius` (1/e^2
  intensity radius); `polarization` is `"p"`, `"s"` (any case) or `sb.Polarization.P/S`.
- `AngularSpectrumBeam` (WP-E1, `excitation::AngularSpectrumBeam`) is the rigorous Gaussian
  beam required for quantitative rough-surface results (ADR 0006 amendment item 4): a finite
  sum of exact propagating plane waves with spectrum `exp(-k_t^2 w0^2 / 4)` around
  `R_y(theta_in) z_hat` (definition in docs/06, "Incident beams"), `E(focus) . e0 = 1 V/m`.
  Same positional arguments as `GaussianBeam`; the keyword-only quadrature controls map to
  `AngularSpectrumBeam::Params` (`region_radius=None` = 4 w0). Fields are controlled only in
  the ball `|r - focus| <= region_radius` (`controlled_radius`): choose it to cover the mesh
  and any near-field observation points. `Simulation` raises `ValueError` when a mesh vertex
  lies outside the ball, and an evaluation beyond 1.2 `region_radius` logs one warning per beam.
  Construction releases the GIL and raises `RuntimeError` when the automatic grid does not
  converge within `max_plane_waves`, `ValueError` for orders above 10^6 or a fixed grid whose
  check grid exceeds `max_plane_waves`; a fixed grid that misses `tolerance` is kept with a
  warning. Cost: one sincos per plane wave and point (indicatively ~7 ns per wave for
  `electric_field` in a release build, timed on a shared machine; `fields` gives E and H for
  little more than the price of one).
- `Mie.material` is `MieParams::sphere`, `exterior` is `MieParams::medium` (lossless).
- Inputs: any real NumPy layout or nested list is accepted (int32/uint32 connectivity,
  Fortran order, strided views); it is converted to a C-contiguous float64 / int64 copy.
  `TriangleMesh(np.zeros((0, 3)), np.zeros((0, 3), int))` is a valid empty mesh
  (`bounding_box()` raises `RuntimeError`). The constructor re-orients the triangles (closed
  components face outward), so `mesh.triangles` may differ from the input; the views reflect
  a later `flip_normals()`.
- `points` is an `(n, 3)` real array (or one `(3,)` point); fields are `(n, 3)` (or `(3,)`)
  complex128. Non-finite points (field evaluation) and angles (`bistatic_rcs`) raise
  `ValueError` before any work is done.
- `DispersiveMaterial` takes n in the exp(+jwt) form `n - jk`: wavelengths must be finite,
  positive and strictly increasing, and an index with `Im(n) > 0` (unconjugated `n + ik`
  optics data) raises `ValueError` with a conjugation hint. The table endpoints are inclusive.
  The refractiveindex.info loader (`from_refractiveindex_info`, conjugating `n + ik`) is not
  implemented yet.
- Every bound class, function and method has a NumPy-style docstring with units
  (`help(sb.Mie)`).
- Exceptions: `std::invalid_argument`, `std::domain_error` (e.g. a Mie point on the wrong side
  of the sphere) and `std::out_of_range` (wavelength outside a `DispersiveMaterial` table) raise
  `ValueError`; `std::runtime_error` and `std::logic_error` raise `RuntimeError`.
- The GIL is released during mesh construction and generation, height-map generation and
  field / RCS evaluation loops; not in `quality()` / `quality_report()` (cheap, and they must
  not race with `flip_normals()`).

## Bound in WP14b2: Simulation, operator, post-processing, I/O

```python
sim = sb.Simulation(mesh, excitation, *, object, exterior=None, wavelength=None,
                    formulation="auto", preconditioner="auto", solver="gmres",
                    compression="dense", gmres=None, kernels=None, mlfmm=None)
sim.assemble()                          # idempotent; GIL released
res = sim.solve(tol=None, max_iter=None, restart=None, side=None, callback=None)  # SolveResult
res.iterations, res.converged, res.residual_history, res.true_relative_residual,
res.wall_seconds, res.x                 # residual_history: float64 copy; x: read-only view
sim.report()                            # str
sim.formulation, sim.preconditioner     # resolved: sb.Formulation, "diagonal" | "none"
sim.solver, sim.wavelength, sim.num_unknowns
sim.currents                            # [J; M] copy; RuntimeError before solve()
Z = sim.operator()                      # sb.LinearOperator: .shape, .dtype, .matvec(x), Z @ x,
Z.as_scipy()                            #   .describe(), .memory_bytes; scipy LinearOperator
sim.rhs()                               # copy; operator()/rhs(): RuntimeError before assemble()
E, H = sim.field(points, kind="scattered", quad_degree=6, min_distance_factor=1e-3)
F = sim.far_field(directions, quad_degree=6)      # (n, 3) [V], E_s ~ F exp(-jkr) / r
sigma = sim.bistatic_rcs(angles, plane="xz")       # "xz" | "yz" | (3,) plane normal; [m^2]
sb.plane_grid(origin, u, v, nu, nv)                # (nu * nv, 3), row i * nv + j
sb.cylinder_grid(radius, theta_min, theta_max, n_theta, y_min, y_max, n_y)
sb.intensity(E); co, cross = sb.polarized_intensity(E, co_pol)
sb.differential_reflection_coefficient(E_cyl, n_theta, n_y)
mesh = sb.read_mesh(path); sb.write_mesh(mesh, path)  # .stl / .obj / .msh by extension
sb.read_stl / read_obj / read_gmsh(path); sb.write_stl(mesh, path, ascii=False)
sb.write_obj(mesh, path); sb.write_gmsh(mesh, path)
with sb.open_npy_directory(path) as w:  # sb.ResultWriter
    w.write_vector(name, v); w.write_matrix(name, m); w.write_mesh(group, mesh)
    w.write_attribute(name, "text" or 1.0)
```

- Mapping onto `SimulationConfig`: `object` / `exterior` (None = `excitation.background`, the
  only value the C++ constructor accepts; WP14b3) / `wavelength`
  (None = `excitation.wavelength`); `formulation` "auto" leaves `formulation` empty (the
  recommendation), "PMCHWT" / "ICTF" / "MCTF" (any case) or an `sb.Formulation` set it ("JMCFIE"
  raises `ValueError`, not implemented); `preconditioner` "auto" / "none" / "diagonal" ->
  `diagonal_preconditioner` empty / false / true; `solver` "gmres" / "direct" -> `SolverKind`;
  `compression` is passed through ("dense" or "mlfmm" (WP20b, GMRES only); "aca" / "hmatrix"
  raise `ValueError`); `mlfmm` keys (WP20b, WP21): `accuracy_digits` (d0, default 3; 5 is
  verified for leaves >= lambda/2 with r_max / a <= 0.3, ADR 0008), `max_elements_per_leaf`
  (100), `min_box_size_lambda` (0.25, leaf floor in exterior wavelengths), `automatic_leaf_size`
  (True: the leaf rule of ADR 0008 raises the floor to max(min_box_size_lambda, a_min(d0),
  r_max / ratio(d0)) with a_min = lambda/4 for d0 <= 3 and lambda/2 for d0 > 3, ratio = 0.6 for
  d0 <= 3 and 0.3 for d0 > 3 (`mlfmm::max_support_ratio`), r_max the largest RWG support radius,
  so `accuracy_digits=5` needs no manual leaf choice; the user value acts as a lower bound; False
  uses `min_box_size_lambda` as given, for experiments), `leaf_radius_quantile` (1.0 = the
  global rule; < 1: local leaf rule of ADR 0008, WP21L: r_max in the leaf rule is replaced by this
  quantile r_q of the support radii, and the larger functions live on a coarser octree level,
  r <= ratio(d0) a / 2 there; opt-in until measured at large L (WP22b3), the WP22b1 Si boxes need
  ~0.8 to elevate their coarse box cells; only with `accuracy_digits` <= 3, a value < 1 with
  d0 > 3 raises ValueError because the lambda/2 leaves it allows miss 1e-5), `max_exact_far_bytes` (int, 0 = automatic: max(2 x the
  near-field bytes, 1 GiB), at most 16 (2N)^2; WP21 review, WP21f).
  Lossy interiors such as silver use the ADR 0008 §6 policy (WP21): no expansion where it is inaccurate, far pairs truncated by
  the decay bound (1 + alpha d) e^{-alpha d} <= 10^-(d0+1) or evaluated exactly (40 bytes per
  basis pair; allowed only if the exact pairs stay within 4 box edges, x*(d0) / alpha <= 4 a,
  and their estimated size, checked before allocation, fits `max_exact_far_bytes`);
  `operator().describe()` lists the decision per region and level. `assemble()` raises
  `RuntimeError` when a region has neither a usable expansion nor enough decay, or when the
  exact part exceeds the budget (the message names the region and the remedy: finer mesh /
  larger leaves, fewer digits, a larger `max_exact_far_bytes`, or "dense"). `gmres` keys: `tol`
  (`tolerance`),
  `max_iter`, `restart` (None = 0 = full GMRES), `side` ("left" / "right"), `verbose`. `kernels`
  keys are the `kernels::OperatorOptions` member names (`quad_degree_far`, `quad_degree_near`,
  `quad_degree_sing`, `outer_grading_levels`, `near_distance_factor`,
  `symmetrize_touching_above_kh`, `target_accuracy`, `quad_degree_rhs`, `fold_adaptive`,
  `decay_aware_target`, `fast_plain_kernel`; WP-P2: `decay_aware_target=False` restores the
  relative near/far target in lossy regions, `fast_plain_kernel=False` the C-library kernel
  arithmetic, so both together with the default schedule reproduce the pre-WP-P2 matrix
  bitwise).
  Unknown strings, keys or value types raise `ValueError`. The mesh is copied, the excitation
  shared (`shared_ptr`).
- `solve()` overrides apply to this call only (C++ `solve(GmresParams, cb)`; `restart=0` = full
  GMRES). The GIL is released during `assemble()`, `solve()`, `matvec`, field / far-field / RCS
  evaluation and file I/O. `callback(iteration, residual)` runs with the GIL re-acquired, once
  per GMRES iteration (iteration 1, 2, ...; not called by the direct solver); an exception it
  raises aborts the solve and reaches the caller unchanged (the previous solution is kept).
- Ctrl-C (WP14b3): a GMRES solve calls `PyErr_CheckSignals()` with the GIL re-acquired, every
  iteration when a callback is given (before it), otherwise only on the main thread (the only
  one that handles signals) and at most every 50 ms, because re-acquiring the GIL can wait up to
  `sys.getswitchinterval()` while another thread runs Python code. `KeyboardInterrupt` then
  takes the callback-exception path. Assembly and the direct solver are not interruptible.
  Tested with `_thread.interrupt_main()` from a timer thread (the aborted solve stores no
  solution) and with a callback raising `KeyboardInterrupt`; a real console Ctrl-C was not.
- Lifetimes: `currents` and `rhs()` are copies (a view could change under a later `solve()`);
  `SolveResult.x` is a read-only view that keeps its result alive (tested after `del res`);
  `operator()` keeps the Simulation alive.
- Threads (WP14b3): calls on one Simulation from several Python threads are serialised by a
  per-object `std::timed_mutex` held by the binding type (`PySimulation`, derived from the
  unchanged C++ `Simulation`). Limits: the calls run one after another, not in parallel (use one
  Simulation per thread for parallel work); from inside a `solve()` callback a busy Simulation
  raises `RuntimeError` instead of waiting; a main thread waiting for the lock stays
  interruptible with Ctrl-C. Design: every locking wrapper (`assemble`, `solve`, `report`,
  `currents`, `operator`, `rhs`, `field`, `far_field`, `bistatic_rcs`) releases the GIL first
  and then waits for the mutex, so no thread waits for it while holding the GIL; the holder may
  re-acquire the GIL (callback, signal check). Waiting is bounded three ways:
  - a locking call from inside a `solve()` callback on the same Simulation would wait for
    itself and raises `RuntimeError` at once (owner-thread check; a recursive mutex was
    rejected because a nested `solve()` would modify the state the outer GMRES is using);
  - a thread that already holds some Simulation's lock (a thread-local counter > 0: inside a
    `solve()` callback, or a finalizer run there) only `try_lock()`s another Simulation and
    raises `RuntimeError` ("Simulation busy") if it is held, because two solves calling into
    each other's Simulation from their callbacks would otherwise deadlock; a call into an idle
    Simulation works;
  - the main thread waits in 50 ms slices (`try_lock_for`) and between them re-acquires the
    GIL for `PyErr_CheckSignals()` (then releases it again), so `KeyboardInterrupt` arrives
    while another thread runs a long solve. Other threads do not handle signals and just wait.

  Properties fixed at construction (`formulation`, `preconditioner`, `solver`, `wavelength`,
  `num_unknowns`) do not lock; `LinearOperator.matvec` does not lock (the operator is immutable
  once assembled, and `op::LinearOperator::apply` must be safe to call concurrently).
- Python tests against the `win-debug` (ASan) module do not run yet: CLANG64 `python.exe` is
  not instrumented, the ASan runtime is loaded late with the extension, and its interceptors
  then reject CRT memory allocated before it started (`_wputenv_s` during import): by default
  with "attempting to call malloc_usable_size() for pointer which is not owned", with
  `ASAN_OPTIONS=check_malloc_usable_size=0` with "attempting free on address which was not
  malloc()-ed" (false positives, not SpeckleBem findings). It would need an instrumented or
  ASan-preloading Python.
- `LinearOperator.matvec(x)` accepts finite real or complex `(n,)` or `(n, 1)` input and
  returns `(n,)` complex128. `as_scipy()` imports SciPy lazily (`ImportError` without it).
- Deviations from the target sketch above: `field` has no `region` argument (the region of every
  point is found from the winding number; `kind="total"` adds the incident field in R1);
  `bistatic_rcs(angles, plane=...)` takes the angles first; `plane_grid` / `cylinder_grid` follow
  the C++ signatures (origin and full extent vectors, point counts); there is no `sim.drc` (use
  `cylinder_grid` + `field` + `differential_reflection_coefficient`), no `mlfmm` / `backend`
  keywords yet, and `sim.save` / HDF5 are not available (`open_npy_directory` instead).
- Paths are `str` or `os.PathLike` (not bytes), passed to C++ as UTF-8 (ADR 0007).

## Rules

- SI units; no implicit scaling.
- Arrays are `float64` / `complex128`, C-contiguous; `(n, 3)` for points and vector fields.
- Long-running calls release the GIL and report progress through a callback (`sim.solve(callback=lambda it, res: ...)`).
- Nothing in Python computes physics; it only orchestrates and converts.
- Logging is controlled with `sb.set_log_level("info")`.

## Implementation notes

- pybind11 with `pybind11/eigen.h`; `Eigen::Ref<const MatrixXr>` for zero-copy input, return by value for outputs.
- scikit-build-core builds the extension from the top-level CMake (`python/pyproject.toml`).
- Wheels: later via cibuildwheel; CUDA wheels are not planned (build from source).
