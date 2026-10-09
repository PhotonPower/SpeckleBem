# Conventions

These are binding for all code, tests and documentation. Deviations are bugs.

## Units
SI throughout the C++ core: metres, seconds, volts/metre, amperes/metre. The Python layer accepts the same (no implicit µm/nm). Helper constants `constants::c0, eps0, mu0, eta0` live in `core/types.hpp`.

## Time convention
`exp(+jωt)` as in the reference paper and most RF/CEM literature.
Consequences:
- Outgoing waves: `exp(−jkR)`; Green's function `G = exp(−jkR)/(4πR)`.
- Passive media: `Im(ε_r) ≤ 0`, `Im(n) ≤ 0`, `Im(k) ≤ 0`. Ag at 500 nm: `ε_r = −9.794 − j0.313`; Si: `18.478 − j0.606`.
- Data from optics sources (refractiveindex.info uses `n + ik`, i.e. `exp(−iωt)`) must be conjugated on import (`DispersiveMaterial` rejects `Im(n) > 0` with a conjugation hint; the planned refractiveindex.info loader conjugates).
- Maxwell curl equations: `∇×E = −jωμH`, `∇×H = jωεE`.

## Regions and normals
- `R1` = exterior / background (light is incident here). `R2` = object (sphere, substrate).
- Surface normal `n̂` points **out of R2 into R1**.
- Equivalent currents: `J = n̂ × H`, `M = −n̂ × E` on the R1 side; `J ≡ J1 = −J2`, `M ≡ M1 = −M2`.
- Unknown vector ordering: `x = [J; M]`, each of length N (number of interior edges), total 2N.

## Coordinates and illumination
- Rough surfaces: mean surface at `z = 0`, patch centred at the origin, object occupies `z > ξ(x,y)` (below the interface when "above" means `−z`). Light arrives from `z < 0` and propagates in `+z`, beam waist at `z = 0`. Observation planes "above" the surface therefore have `z < 0` (as in the paper: `z = −4.5 µm`, `z = −1 mm`).
- Spheres: centred at the origin; validation plane wave propagates in `+z`, E along `x` (p-polarisation in the xz-plane).
- Polarisation: **p** – E in the plane of incidence (xz); **s** – E along `y`. Co-/cross-polarised intensities are taken with respect to the incident E direction.
- Angles: incidence angle `θ_in` is a rotation about `y`; scattering angle `θ_s` is measured from `−z` in the xz-plane (rough surfaces, reflection geometry).
- Spheres and the Mie reference use ordinary spherical coordinates: polar angle `θ` from `+z` (forward scattering `θ = 0`, backscattering `θ = π`), azimuth `φ` from `+x`. In the xz-plane `θ_s = π − θ`. `reference::MieSolution::bistatic_rcs(theta, phi)` and the sphere validation tests follow this spherical convention.

## Discretisation
- Flat triangles, consistently oriented (counter-clockwise seen from R1). RWG basis per interior edge; Galerkin testing.
- Mesh sizes: `λ/10` for non-resonant rough surfaces, `λ/27` for the resonant Ag sphere (paper values; refine until converged).
- Iterative stopping criterion: relative residual `‖b − Zx‖/‖b‖ ≤ 1e-3` (paper), tests may use tighter values.

## Numerical precision
`double` / `std::complex<double>` on CPU. Single precision is allowed in GPU kernels only when a test demonstrates the result stays within the accuracy target of docs/05_validation.md.

## Naming
Namespaces follow directories: `specklebem::geometry`, `::basis`, `::kernels`, `::formulation`, `::op`, `::mlfmm`, `::solver`, `::post`, `::reference`, `::backend`, `::io`. Files are `snake_case.hpp/.cpp`; types `PascalCase`; functions and variables `snake_case`; private members end with `_`.
