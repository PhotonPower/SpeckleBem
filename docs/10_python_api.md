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
