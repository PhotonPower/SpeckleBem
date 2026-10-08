# Validation strategy

Every fast or approximate component is tested against a slower exact one. The chain of oracles is:

```
analytic (Mie, Fresnel, static integrals)
   └─► dense SIE (direct LU)
          └─► dense SIE (GMRES)
                 └─► MLFMM / ACA / H-matrix (GMRES)
                        └─► GPU backend
```

## Error metrics

- **Bistatic RCS error** (Eq. 7 of the paper): `ε_rr = sqrt(Σ (E_ref − E_sie)² / M) / max|E_ref|` over `M` angles.
- **Field error** along a line: relative error per point (Fig. 1b style) and its median.
- **Operator error**: `‖Z_fast x − Z_dense x‖ / ‖Z_dense x‖` over 10 random `x`.
- **Power balance** for lossy objects: `P_ext = P_sca + P_abs` within tolerance (independent of any reference solution).
- **Reciprocity**: scattering amplitude `S(k̂_s, k̂_i) = S(−k̂_i, −k̂_s)` for swapped directions.

## Acceptance table

| Test | Reference | Metric | Tolerance | Phase |
|------|-----------|--------|-----------|-------|
| Static integrals on separated triangles | Dunavant-20 | abs/rel | 1e-12 | 1 |
| Static integrals on coincident triangles | nested-quadrature extrapolation | rel | 1e-9 | 1 |
| Height-map statistics | prescribed σ, Lc | rel (20 seeds) | 2 % / 5 % | 1 |
| Dielectric sphere d = 1 µm, n = 1.5, λ/10 | Mie | ε_rr | < 1 % | 2 |
| Ag sphere d = 1 µm, λ/20, PMCHWT + LU | Mie | ε_rr | < 1 % | 2 |
| Mesh refinement λ/8 → λ/16 → λ/32 | Mie | ε_rr monotone decrease | — | 2 |
| Flat box, tapered beam, 0° and 45° | Fresnel r_p, r_s | rel on |r|² | < 1 % | 2 |
| GMRES vs LU, 2·10⁴ unknowns | dense LU | rel | < tol | 3 |
| MLFMM matvec, Si & Ag, 2·10⁴–10⁵ unknowns | dense matvec | rel | < 1e-3 (3 digits), < 1e-5 (5 digits) | 4 |
| Ag sphere d = 4 µm, λ/27, MLFMM | Mie | ε_rr normal/parallel planes | ≤ 0.5 % | 4 |
| Power balance, Ag and Si surfaces | — | rel | < 1 % | 4 |
| Reciprocity, rough surface | — | rel | < 1 % | 4 |
| Near field Ag sphere, xz-plane | Mie | rel per point | < 0.1 (lit), < 1 (shadow) | 5 |
| Rigorous beam Maxwell check | — | ‖∇×E + jωμH‖ / ‖ωμH‖ | < 1e-8 | 5 |
| Speckle PDF at z = −1 mm | Eq. 10 | KS test | p > 0.05 | 6 |
| Contrast, γ₁₂ vs σ | monotone decrease | — | — | 6 |
| Speckle size | λz/D | rel | < 10 % | 6 |
| GPU vs CPU, all kernels | CPU | rel | < 1e-4 (double), documented for single | 7 |
| ACA / H-matrix matvec | dense | rel | < ε set | 8 |
| Imported Gmsh closed mesh | power balance | rel | < 1 % | 8 |

## Rough-surface specific checks

- **Box-depth sensitivity**: scattered far field changes by < 0.1 % when the closing box depth is doubled (Si and Ag). Documents that the artificial boundary is inert.
- **Edge-effect sensitivity**: far field changes by < 0.5 % when L is increased by 20 % at fixed beam waist.
- **Flat-surface limit**: σ → 0 recovers the specular Fresnel reflection of the beam.
- **Regression**: height map for `seed = 42`, L = 10 µm is stored in `tests/data`; a change in the generator that alters it must be deliberate.

## Test organisation

- `tests/unit` – fast (< 1 s each), run on every push.
- `tests/validation` – minutes, labelled `validation`, run on every push for the small cases; the 4 µm sphere and 30 µm surfaces are nightly / manual (`ctest -L validation-large`).
- `tests/python` – API smoke tests and the statistics tests on small precomputed fields.
- Reference data: `tests/data/README.md`.
