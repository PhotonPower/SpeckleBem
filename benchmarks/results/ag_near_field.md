# Near field of the Ag sphere in the xz-plane vs Mie (WP24, Phase 5)

Phase 5 acceptance test of `docs/01_project_plan.md` ("Near field of the Ag sphere in the
xz-plane matches Mie with the error profile of Fig. 1(b): relative error < 0.1 along x, < 1 in
the shadow") and the `docs/05_validation.md` row "Near field Ag sphere, xz-plane | Mie | rel per
point | < 0.1 (lit), < 1 (shadow)".

**Verdict: met.** Acceptance case: the paper's d = 4 µm mesh (2N = 393 216) at GMRES tolerance
1e-4 (the `validation-large` test). Outside the near-surface band (|r| − a ≥ h/2 = 15 nm = λ/33)
the largest relative error per point over grid and shells is **0.012 in the lit zone**
(criterion < 0.1, margin ≈ 8×; on the x-axis 0.0012, in the forward region beside the shadow
0.0040) and **0.006 in the geometric shadow** (criterion < 1); the 95th percentiles are
0.0012 / 0.0018 and the medians 0.0003 / 0.0007. At the paper's GMRES tolerance 1e-3 (kept as
evidence) the criterion holds as well, but only with a 1.14× margin: maxima 0.088 lit (x-axis
0.024, beside the shadow 0.022) and 0.022 shadow, p95 0.009 / 0.012 (grid alone: maxima
0.061 / 0.017). The lit maximum sits in a standing-wave minimum in front of the lit pole
(|E_Mie| = 0.19 V/m at 232 nm from the surface) and is GMRES-tolerance error, not
discretisation error (0.088 → 0.012 from tol 1e-3 to 1e-4). The d = 1 µm companion (λ/26.5,
tol 1e-5) has maxima 0.0050 / 0.0049 (0.38 % / 0.46 % on the grid alone). The criterion also
holds on the CI-sized dense icosphere n = 3 (λ/6.6): 0.050 / 0.086.

## Setup

- The WP22a case (`tests/support/ag_sphere_mlfmm_support.hpp`, record
  `ag_sphere_4um_mlfmm.md`): sphere centred at the origin in vacuum, λ = 500 nm, Ag
  ε_r = −9.794 − 0.313j (`material::silver_500nm`), plane wave k̂ = +z, E along x, 1 V/m;
  `Simulation` with compression `"mlfmm"`, d₀ = 3, automatic leaf rule and exact-part budget,
  ICTF + left Jacobi (`formulation::recommend`), full GMRES.
  - d = 4 µm: octahedron-based sphere n = 7, 131 072 triangles, 2N = 393 216, mean edge
    h = 30.25 nm = λ/16.5 (the paper's mesh, docs/05 footnote 1), GMRES tol 1e-4 (acceptance,
    the `validation-large` test) and 1e-3 (the paper's tolerance, evidence and sensitivity).
  - d = 1 µm: icosphere n = 5, 20 480 triangles, 2N = 61 440, h = 18.88 nm = λ/26.5, GMRES tol
    1e-5 (as the WP22a validation-large case) and 1e-3 (sensitivity); icosphere n = 4 (λ/13.2,
    tol 1e-5) and n = 3 (λ/6.6, tol 1e-6) for the mesh trend.
- Field: `post::total_field` (incident + scattered field from the RWG currents in R1, the R2
  representation inside), Dunavant degree 6 per source triangle (the `FieldOptions` default) and
  degree 12 for observation points within one mean edge of the surface (the only near-surface
  setting `FieldOptions` offers is the degree; see the profile below for why 12 suffices).
  Reference: `reference::MieSolution` (n_max = 28 for d = 1 µm, 59 for d = 4 µm), total field
  x̂ e^{−jk₁z} + E_s outside, internal field inside.
- Metric per point: e(r) = |E_sim(r) − E_Mie(r)| / |E_Mie(r)| (complex 3-vectors, Euclidean
  norm): the denominator is the local Mie field, not |E_inc| (docs/05 "rel per point", the style
  of the paper's Fig. 1(b); coordinator decision in the WP24 review). In the lit zone |E_Mie|
  reaches ≈ 2.2 V/m for |E_inc| = 1 V/m (the runs do not print the maximum; the standing wave in
  front of the almost perfectly reflecting Ag, |r| = 0.991 at normal incidence, gives up to
  1 + |r| ≈ 2, and the near-surface hot spots at the equator reach 2.18 V/m on the 1 µm grid, see
  the table along x), so the lit maximum e bounds the error relative to |E_inc| by ≈ 2.2 e:
  ≈ 0.03 for the acceptance maximum 0.0124 at tol 1e-4, but 0.0876 × 2.2 ≈ 0.19 at tol 1e-3,
  which is why the acceptance case runs at tol 1e-4. Also recorded: the magnitude error
  ||E_sim| − |E_Mie|| / |E_Mie| and the absolute error |E_sim − E_Mie| in units of
  |E_inc| = 1 V/m.
- Observation points (all in the plane y = 0):
  - **grid** x, z ∈ [−1.5 d, 1.5 d] (the sphere and its surroundings, as Fig. 1(b)), odd number
    of points per axis so that the axes x = 0 and z = 0 are grid lines: d = 4 µm 243 × 243 at
    49.6 nm (λ/10.1), d = 1 µm 123 × 123 at 24.6 nm (λ/20.3); grid points within λ/100 of the
    surface are not evaluated (the intended sizes for the requested spacings λ/10 and λ/20 are
    241 and 121: in floating point 2·half / spacing is 240.00000000000003 and
    120.00000000000001, so the ceil in `grid_points` adds one interval and the odd-size rule a
    second point; kept so that the recorded runs stay reproducible, the unit test pins 243 / 123);
  - **shells**: circles of radius a + d around the sphere, 360 points each (1°), at 25
    log-spaced distances d = λ/100 … λ outside and 9 distances λ/100 … λ/5 inside (12 240
    points; the near-surface profile).
- Executable: `specklebem_ag_near_field` (`benchmarks/ag_near_field.cpp`,
  `-DSPECKLEBEM_BUILD_BENCHMARKS=ON`), one case per process, e.g.
  `specklebem_ag_near_field --mesh octa --n 7 --d 4e-6 --tol 1e-3 --spacing 5e-8
  --line-degrees 6,12,19 --near-degree 12 --near-band-h 1 --d-min-h 0.25,0.5,1,2 --out DIR`
  (`--out` writes the currents and all grid/shell fields and errors as `.npy`; `--load DIR`
  re-evaluates stored currents without solving). Shared code:
  `tests/support/ag_near_field_support.hpp`. Tests: `tests/validation_large/
  test_ag_near_field_large.cpp` (d = 1 µm and d = 4 µm, `validation-large`, memory-guarded) and
  `tests/validation/test_ag_near_field.cpp` (icosphere n = 3, dense, CI release).
- Machine: Windows 11, 12 cores / 24 threads, 128 GB, MSYS2 UCRT64 GCC, OpenBLAS, `win-release`,
  2026-10-14. Another agent's test runs and this WP's debug build overlapped with some runs, so
  the timings are indicative.

## Zones

With d = |r| − a the distance from the sphere surface. The faceted meshes deviate from the
sphere by their sagitta h²/8a per edge: 0.06 nm (4 µm), 0.09 nm (1 µm, n = 5), 0.36 nm (n = 4)
and ≈ 1.4 nm on the CI mesh n = 3 (≈ 1.9 nm by the facet-centre depth h²/6a), all below the
5 nm = λ/100 evaluation cut-off and below d_min (n = 2, 5.6 nm, fails the region check).

| Zone | Definition | Physics | Criterion |
|---|---|---|---|
| band | \|d\| < d_min = h/2 | near-surface exclusion (below) | none |
| **shadow** | d ≥ d_min, z > 0 and \|x\| < a (the geometric shadow) | incident and forward-scattered waves cancel behind the sphere, small denominators | e < 1 |
| **lit** | every other point with d ≥ d_min ("lit / outside"): z ≤ 0, the x-axis z = 0 ("along x" of the DoD) and the forward region beside the sphere (z > 0, \|x\| ≥ a) | incident + reflected / scattered wave, \|E\| = O(1) (min 0.23 V/m on the 4 µm grid for z ≤ 0, ≥ 0.33 V/m for z > 0) | e < 0.1 |
| interior | d ≤ −d_min | inside Ag the field amplitude decays to 1/e within λ/(2π\|Im n\|) = 25.4 nm (the intensity penetration depth λ/(4π\|Im n\|) is 12.7 nm; \|E_Mie\| < 1e-9 V/m at 0.5 µm depth), so e is meaningless there | none: \|E_sim − E_Mie\| / max\|E_inc\| reported as a diagnostic |

The lit points on the x-axis (|z| < 1e-9 a, which absorbs the rounding z ≈ 1e-22 m of the shell
points at ψ = 90°) and those beside the shadow (z > 0, |x| ≥ a) are reported as subsets. Before
the WP24 review the shadow zone was the whole half-space z > 0; the stricter rule above moves the
forward region beside the sphere under the 0.1 bound. Every case re-evaluated under it passes:
the forward region beside the shadow has maxima 0.0040 (4 µm, tol 1e-4), 0.022 (4 µm, tol 1e-3),
0.0016 (1 µm, n = 5) and 0.028 (CI mesh n = 3), of the size of the other lit errors (see below).

## Near-surface profile and the choice of d_min

Maximum of e over the front (z ≤ 0) and back (z > 0) halves of each shell (a diagnostic split
of the profile, `PROFILE` lines; not the acceptance zones, whose shadow is only the part
|x| < a of the back half), Dunavant degree 6 / 19 (degree 12 agrees with 19 within 1 % of e at
every |d| ≥ 0.2 h on all three meshes; inside, the absolute error is listed instead):

| d / h | d [nm] (4 µm) | 4 µm, tol 1e-3: front (deg 6 / 19) | back (6 / 19) | 4 µm, tol 1e-4 §: front / back (deg 6) | d [nm] (1 µm) | 1 µm n = 5, tol 1e-5: front (6 / 19) | back (6 / 19) |
|---:|---:|---|---|---|---:|---|---|
| 0.17 / 0.26 | 5.0 | 0.069 / 0.067 | 0.062 / 0.055 | 0.031 / 0.064 | 5.0 | 0.0108 / 0.0096 | 0.024 / 0.024 |
| 0.24 / 0.39 | 7.3 | 0.058 / 0.058 | 0.039 / 0.040 | 0.016 / 0.029 | 7.3 | 0.0062 / 0.0061 | 0.010 / 0.0099 |
| 0.36 / 0.57 | 10.8 | 0.051 / 0.051 | 0.029 / 0.029 | 0.0083 / 0.013 | 10.8 | 0.0050 / 0.0050 | 0.0049 / 0.0049 |
| 0.52 / 0.84 | 15.8 | 0.045 / 0.045 | 0.022 / 0.022 | 0.0050 / 0.0062 | 15.8 | 0.0041 / 0.0041 | 0.0037 / 0.0037 |
| 0.93 / 1.49 | 28.1 | 0.036 / 0.036 | 0.018 / 0.018 | 0.0037 / 0.0034 | 28.1 | 0.0031 / 0.0031 | 0.0044 / 0.0044 |
| 1.65 / 2.65 | 50 | 0.026 | 0.017 | 0.0029 / 0.0031 | 50 | 0.0022 | 0.0045 |
| 3.6 / 5.7 | 108 | 0.018 | 0.016 | 0.0021 / 0.0032 | 108 | 0.0014 | 0.0027 |
| 7.7 / 12.3 | 232 | **0.088** | 0.020 | 0.012 / 0.0029 | 232 | 0.0029 | 0.0015 |
| 16.5 / 26.5 | 500 | 0.019 | 0.012 | 0.0029 / 0.0017 | 500 | 0.0013 | 0.0009 |
| −0.17 / −0.26 (abs) | −5.0 | 0.023 / 0.021 | 0.044 / 0.030 | 0.018 / 0.028 | −5.0 | 0.0059 / 0.0045 | 0.0093 / 0.0073 |
| −0.51 / −0.81 (abs) | −15.4 | 0.012 / 0.012 | 0.012 / 0.012 | 0.0018 / 0.0049 | −15.4 | 0.0016 / 0.0016 | 0.0021 / 0.0021 |
| −3.3 / −5.3 (abs) | −100 | 3e-4 | 2e-4 | 4e-5 / 5e-5 | −100 | 6e-5 | 7e-5 |

(The first column gives d/h for the 4 µm and the 1 µm mesh; the 1 µm icosphere n = 4, λ/13.2,
shows the same structure in units of h: degree 6 and 19 agree to ≤ 2 % of e from d ≈ 0.5 h on.)

§ From the `specklebem_ag_near_field --tol 1e-4` benchmark run made before the WP24 review, not
from the ctest run under "Test runs"; the profile splits each shell at z = 0 into front and back
halves and is therefore independent of the zone rule (its 232 nm front value 0.012 is the lit
maximum 0.0124 of the acceptance test).

Reading:

1. **Quadrature.** The default degree 6 differs from degree 19 by ≤ 3 % of e for d ≥ 0.47 h
   and ≤ 1 % from 0.57 h on (4 µm: ≤ 1 % from 0.24 h); below, the nearly singular integrand of
   the closest triangles is under-resolved (degree 6 back-half error at 5 nm: 0.044 vs 0.030
   inside, 0.062 vs 0.055 outside). Degree 12 agrees with degree 19 within 1 % of e for |d| ≥ 0.2 h (it differs only at
   0.13 h inside the λ/13 mesh: 0.035 vs 0.024), so the grid uses degree 12 within one edge of the
   surface; the profile shows that this matters only below d_min.
2. **Discretisation.** With converged quadrature the error still rises steeply below ≈ h/2
   (1 µm: 0.0037 at h → 0.0055 at 0.47 h → 0.0096 at 0.26 h; 4 µm, tol 1e-4: 0.0034 at 0.93 h →
   0.0050 at 0.52 h → 0.031 at 0.17 h): the flat facets and the piecewise linear RWG currents
   deviate from the smooth sphere on the scale h, and the field of these deviations decays
   roughly like exp(−2π d/h). At tolerance 1e-3 on the 4 µm mesh the solver error (≈ 0.02–0.05
   near the surface) masks most of this rise.
3. **Choice.** d_min = **h/2** (= 15.1 nm = λ/33 on the 4 µm mesh, 9.4 nm = λ/53 on the 1 µm
   mesh; `kDMinPerEdge = 0.5`): beyond it the default quadrature is converged and the maximum e
   over a circle is within ≈ 1.5× its value at d = h. The choice is not what decides the verdict:
   on the 4 µm and 1 µm n = 5 meshes the criterion holds for every exclusion distance tried,
   including d_min = 0 (no exclusion beyond the λ/100 = 5 nm evaluation cut-off, i.e. down to
   the innermost shells at ±5 nm).
   Union of grid and shells, current zones, from the `d_min scan (union)` lines of the logs
   (lit max / shadow max, excluded points in parentheses):

   | d_min / h | 4 µm, tol 1e-3: d_min [nm] | lit / shadow (excluded) | 4 µm, tol 1e-4 | 1 µm n = 5, tol 1e-5: d_min [nm] | lit / shadow (excluded) |
   |---:|---:|---|---|---:|---|
   | 0 | 0 | 0.0876 / 0.0554 (0) | 0.0262 / 0.0517 (0) | 0 | 0.0096 / 0.0237 (0) |
   | 0.25 | 7.6 | 0.0876 / 0.0337 (1 832) | 0.0124 / 0.0218 (1 832) | 4.7 | 0.0096 / 0.0237 (0) |
   | **0.5** | 15.1 | 0.0876 / 0.0223 (3 336) | 0.0124 / 0.0062 (3 336) | 9.4 | 0.0050 / 0.0049 (2 228) |
   | 1 | 30.3 | 0.0876 / 0.0175 (5 636) | 0.0124 / 0.0033 (5 636) | 18.9 | 0.0038 / 0.0046 (4 124) |
   | 2 | 60.5 | 0.0876 / 0.0173 (7 752) | 0.0124 / 0.0032 (7 752) | 37.8 | 0.0029 / 0.0046 (6 460) |

   (The 4 µm lit maximum at tol 1e-3 lies at d = 232 nm and does not depend on d_min; the
   1 µm entry at 0.25 h = 4.7 nm excludes nothing, as no point lies between 4.7 nm and the
   5 nm cut-off; the large shadow values below h/2 are the near-surface rise of the profile.)
   On the coarse CI mesh n = 3 (λ/6.6, h/2 = 37.7 nm) the exclusion does matter: d_min = 0
   gives lit 0.166 / shadow 0.535, h/4 0.071 / 0.116, h/2 0.050 / 0.086, h 0.050 / 0.066,
   2h 0.050 / 0.041 (CI run of 2026-10-15).

## Results per zone (d_min = h/2)

Union of grid and shells (the acceptance statistic of the tests, points with |d| ≥ d_min) and,
below each case, grid and shells alone. e is dimensionless; "interior" is
|E_sim − E_Mie| / |E_inc|. Zones as defined above (shadow = geometric shadow |x| < a, z > 0).

| Case | Points | Zone | count | max | p99 | p95 | median | max at (x, z) [nm] |
|---|---|---|---:|---:|---:|---:|---:|---|
| **4 µm, octa n = 7, tol 1e-4 (acceptance)** | grid + shells | lit | 50 744 | **0.0124** | 0.0021 | 0.0012 | 2.7e-4 | (233, −2220), d = 232 nm |
| | grid + shells | lit, z = 0 | 200 | 0.0012 | 0.0012 | 9.3e-4 | 3.3e-4 | (2281, 0) |
| | grid + shells | lit beside the shadow | 20 296 | 0.0040 | 0.0011 | 6.9e-4 | 1.3e-4 | (2011, 141) |
| | grid + shells | shadow | 9 952 | **0.0062** | 0.0029 | 0.0018 | 7.5e-4 | (281, 1996), d = 15.8 nm |
| | grid + shells | interior (abs) | 7 201 | 0.0049 | 0.0014 | 7.9e-4 | 3e-9 | (0, 1985) |
| | grid | lit / shadow | 46 611 / 7 245 | 0.0078 / 0.0033 | 0.0014 / 0.0018 | 8.7e-4 / 0.0013 | 2.5e-4 / 6.9e-4 | (−149, −2231) / (0, 2033) |
| | shells | lit / shadow | 4 133 / 2 707 | 0.0124 / 0.0062 | 0.0040 / 0.0036 | 0.0027 / 0.0026 | 0.0010 / 0.0012 | |
| 4 µm, octa n = 7, tol 1e-3 (paper's tolerance) | grid + shells | lit | 50 744 | **0.0876** | 0.0162 | 0.0090 | 0.0020 | (233, −2220), d = 232 nm |
| | grid + shells | lit, z = 0 | 200 | 0.0242 | 0.0242 | 0.0195 | 0.0030 | (−2281, 0) |
| | grid + shells | lit beside the shadow | 20 296 | 0.0220 | 0.0125 | 0.0063 | 0.0016 | (−2281, 40) |
| | grid + shells | shadow | 9 952 | 0.0223 | 0.0153 | 0.0122 | 0.0048 | (−281, 1996), d = 15.8 nm |
| | grid + shells | interior (abs) | 7 201 | 0.0120 | 0.0050 | 0.0025 | 8e-9 | (0, 1985) |
| | grid | lit / shadow | 46 611 / 7 245 | 0.0608 / 0.0168 | 0.0107 / 0.0125 | 0.0055 / 0.0081 | 0.0019 / 0.0042 | (−149, −2231) / (−942, 1785) |
| | grid | interior (abs) | 5 041 | 0.0112 | 0.0017 | 3.5e-4 | 7e-14 | (0, −1983) |
| | shells | lit / shadow | 4 133 / 2 707 | 0.0876 / 0.0223 | 0.0356 / 0.0172 | 0.0207 / 0.0143 | 0.0084 / 0.0079 | |
| **1 µm, ico n = 5, tol 1e-5** | grid + shells | lit | 17 047 | 0.0050 | 0.0033 | 0.0018 | 6.4e-4 | (124, −496) |
| | grid + shells | lit, z = 0 | 124 | 0.0010 | 0.0010 | 0.0010 | 7.4e-4 | (732, 0) |
| | grid + shells | lit beside the shadow | 6 317 | 0.0016 | 0.0012 | 0.0010 | 4.8e-4 | (−501, 97) |
| | grid + shells | shadow | 4 289 | 0.0049 | 0.0033 | 0.0025 | 0.0013 | (9, 511) |
| | grid + shells | interior (abs) | 3 765 | 0.0027 | 0.0021 | 0.0017 | 3.3e-4 | (199, 447) |
| | grid | lit / shadow | 11 931 / 1 845 | 0.0038 / 0.0046 | 0.0018 / 0.0025 | 0.0012 / 0.0019 | 5.3e-4 / 8.7e-4 | (148, −492) / (0, 541) |
| | shells | lit / shadow | 5 116 / 2 444 | 0.0050 / 0.0049 | 0.0040 / 0.0038 | 0.0028 / 0.0026 | 0.0011 / 0.0016 | |
| 1 µm, ico n = 5, tol 1e-3 ‡ | grid | lit / shadow | | 0.038 / 0.026 | 0.019 / 0.009 | 0.014 / 0.007 | 0.006 / 0.003 | |
| | shells | lit / shadow | | 0.050 / 0.026 | | | | |
| 1 µm, ico n = 4 (λ/13.2), tol 1e-5 ‡ | grid | lit / shadow | | 0.013 / 0.021 | 0.008 / 0.009 | 0.006 / 0.006 | 0.002 / 0.002 | |
| 1 µm, ico n = 3 (λ/6.6), dense, tol 1e-6 (CI test) | grid + shells | lit / shadow | 15 485 / 3 195 | 0.050 / 0.086 | 0.035 / 0.059 | 0.026 / 0.046 | 0.011 / 0.025 | (148, −713) / (0, 550) |
| | grid + shells | lit, z = 0 / beside the shadow | 108 / 6 091 | 0.020 / 0.028 | | | | |

‡ Sensitivity and mesh-trend runs evaluated before the WP24 review with the half-space zones
(lit z ≤ 0, shadow z > 0) and not re-evaluated; the shadow column covers the whole back
half-space, i.e. it contains the forward region beside the sphere that is now lit.

Magnitude error ||E_sim| − |E_Mie|| / |E_Mie| (grid + shells, d_min = h/2): 4 µm tol 1e-4 max
0.0051 lit / 0.0048 shadow, tol 1e-3 0.044 / 0.017; 1 µm 0.0036 / 0.0036.

Observations:

- The error is set by the GMRES tolerance on the 4 µm case (tol 1e-3 → 1e-4 reduces the grid
  maxima 7.8× lit and 5.1× in the shadow, the union maxima 7.0× and 3.6×, as ε_rr in WP22a:
  0.023 % → 0.0035 %) and on the 1 µm case (tol 1e-3 → 1e-5: grid lit max 0.038 ‡ → 0.0038,
  10×). The 1 µm grid lit maximum falls with the mesh: λ/6.6 0.050, λ/13.2 0.013, λ/26.5 0.0038.
  The three values are not from one series: λ/6.6 is the CI case at tol 1e-6 (not 1e-5) under the
  current zones (its grid and union lit maxima coincide), λ/13.2 a tol 1e-5 run with the
  half-space zones ‡ (its lit zone is z ≤ 0 only), and λ/26.5 the tol 1e-5 acceptance run under
  the current zones. The λ/6.6 and λ/26.5 maxima lie at z < 0, so they are the same under either
  zone rule.
- The lit maximum is always in a standing-wave minimum of the incident and reflected waves in
  front of the lit hemisphere (4 µm: on the axis |E_Mie| = 0.31 V/m at d = 231 nm and 0.23 V/m at
  d = 479 nm; at tol 1e-3 the shell point with e = 0.088 has |E_Mie| = 0.19 V/m and an absolute
  error of only 0.017 V/m). The largest absolute errors (4 µm, tol 1e-3: 0.047 V/m in the then
  lit half-space z ≤ 0 ‡) lie at the near-surface hot spots where |E| ≈ 2 V/m.
- The shadow is not deep for these sizes in the xz-plane: |E_Mie| ≥ 0.33 V/m on the 4 µm grid
  (≥ 0.51 V/m at 1 µm), so the shadow errors are of the same size as the lit ones and two orders
  below the docs/05 bound of 1.
- Interior: |E_sim − E_Mie| ≤ 0.012 V/m at d ≤ −h/2 (4 µm, tol 1e-3; 0.005 V/m at tol 1e-4),
  falling with depth as the field itself (3e-4 at 100 nm); deep inside both fields are
  ~1e-30 V/m.

## Error along x (z = 0) and along z (x = 0)

d = 4 µm, tol 1e-3, grid (decimated; |E| in V/m):

| x [µm] (z = 0) | d [nm] | \|E_Mie\| | \|E_sim\| | e |
|---:|---:|---:|---:|---:|
| ±6.000 | 4000 | 1.050 | 1.050 | 0.0018 |
| ±5.008 | 3008 | 1.081 | 1.081 | 0.0021 |
| ±4.016 | 2017 | 1.123 | 1.120 | 0.0025 |
| ±3.025 | 1025 | 0.813 | 0.816 | 0.0055 |
| ±2.479 | 479 | 0.951 | 0.950 | 0.0075 |
| ±2.033 | 33 | 1.674 | 1.665 | 0.0103 |
| ±1.984 | −16.5 (interior) | 0.308 | 0.304 | (abs 0.0044) |

| z [µm] (x = 0) | d [nm] | zone | \|E_Mie\| | \|E_sim\| | e |
|---:|---:|---|---:|---:|---:|
| −6.000 | 4000 | lit | 0.875 | 0.877 | 0.0039 |
| −5.008 | 3008 | lit | 0.746 | 0.751 | 0.0067 |
| −4.016 | 2017 | lit | 0.879 | 0.876 | 0.0062 |
| −3.025 | 1025 | lit | 0.955 | 0.958 | 0.0032 |
| −2.479 | 479 | lit | 0.231 | 0.234 | 0.0269 |
| −2.231 | 231 | lit | 0.307 | 0.300 | 0.0270 |
| −2.033 | 33 | lit | 1.278 | 1.284 | 0.0093 |
| −1.984 | −16.5 | interior | 0.324 | 0.316 | (abs 0.011) |
| +1.984 | −16.5 | interior | 0.817 | 0.811 | (abs 0.011) |
| +2.033 | 33 | shadow | 1.186 | 1.186 | 0.0158 |
| +2.479 | 479 | shadow | 0.864 | 0.861 | 0.0041 |
| +3.025 | 1025 | shadow | 1.064 | 1.061 | 0.0030 |
| +4.016 | 2017 | shadow | 1.088 | 1.083 | 0.0052 |
| +5.008 | 3008 | shadow | 1.052 | 1.046 | 0.0051 |
| +6.000 | 4000 | shadow | 1.019 | 1.015 | 0.0042 |

d = 1 µm, ico n = 5, tol 1e-5 (grid):

| position | d [nm] | \|E_Mie\| | e |
|---|---:|---:|---:|
| x = ±1.500 µm, z = 0 | 1000 | 1.018 | 2.8e-4 |
| x = ±1.008 µm, z = 0 | 508 | 1.158 | 4.2e-4 |
| x = ±0.762 µm, z = 0 | 262 | 0.650 | 1.0e-3 |
| x = ±0.516 µm, z = 0 | 16 | 2.177 | 7.7e-4 |
| z = −1.500 µm, x = 0 (lit) | 1000 | 0.768 | 9.3e-4 |
| z = −0.738 µm (lit) | 238 | 0.470 | 1.3e-3 |
| z = −0.516 µm (lit) | 16 | 0.885 | 3.3e-3 |
| z = +0.516 µm (shadow) | 16 | 0.711 | 3.5e-3 |
| z = +0.762 µm (shadow) | 262 | 0.984 | 1.4e-3 |
| z = +1.500 µm (shadow) | 1000 | 1.130 | 6.0e-4 |

## Timings and memory

| Case | Assembly [s] | GMRES it / [s] | Shells (12 240 points), degree 6 / 12 / 19 [s] | Grid points / [s] (ms per point) | Peak RSS [GB] |
|---|---:|---:|---|---|---:|
| 4 µm, tol 1e-3 | 78 | 442 / 430 | 66 / 192 / 540 | 58 993 / 423 (7.2) | 24.8 |
| 4 µm, tol 1e-4 § | 153† | 722 / 1112† | 70 / – / – | 58 993 / 404 (6.9) | 26.5 |
| 1 µm n = 5, tol 1e-5 | 14 | 388 / 104 | 9.3 / 25 / 59 | 15 089 / 11.9 (0.79) | 7.6 |
| 1 µm n = 3, dense, tol 1e-6 (CI test) | 1.7 | 186 / 11† | together 27 329 points in ~4 s | | |

† overlapped with a debug build or test run.
§ The pre-review benchmark run (see the profile above); timings do not depend on the zone rule.
The ctest acceptance run of the same case is listed under "Test runs" (assembly 174 s, GMRES
1136 s, field 775 s for grid and shells).

The direct field evaluation costs O(N_points × N_triangles × 12) (degree 6): 4.3–4.6 ns per
(point × source sample) on 24 threads for the 4 µm mesh (1.57 M samples per point, including
the winding-number region test): the full 4 µm grid took 404 s (tol 1e-4 run, 4.3 ns) and 423 s
(tol 1e-3 run, 4.6 ns) of wall time on 24 threads, ≈ 2.7–2.8 thread-hours ≈ 1.4 core-hours on the
12 cores. Whether this is cheaper than the solve depends on the tolerance: in the tol 1e-4
acceptance test the field evaluation of grid and shells took 775 s against 1136 s of GMRES, in
the tol 1e-3 benchmark run 801 s against 584 s. No grid reduction was needed; the evaluation
was not accelerated by the MLFMM (the WP25 decision applies: direct evaluation is affordable at
these sizes).

## Test runs

Statistics over the union of grid and shells (d_min = h/2, zones as above: shadow = geometric
shadow, lit = every other exterior point; criterion unchanged), `win-release` build, 2026-10-14.
The binaries were built from the sources of commit 4ff95f1 (its d_min = 0 scan entry, 0 nm
with no excluded point, appears in the logs); the later commits of the branch change only this
record, docs/05 and comments.

`ctest --preset win-release -L validation-large -R "Ag near field"` (both cases passed, run
serially):

| Case | Points (excluded) | GMRES it (true residual) | lit max / p99 / p95 / median | lit on z = 0 max | lit beside the shadow max | shadow max / p99 / p95 / median | interior abs max | Test time |
|---|---|---|---|---:|---:|---|---:|---:|
| d = 1 µm, ico n = 5, tol 1e-5 | 27 329 (2 228) | 388 (1.2e-5) | 0.0050 / 0.0033 / 0.0018 / 0.00064 | 0.0010 | 0.0016 | 0.0049 / 0.0033 / 0.0025 / 0.0013 | 0.0027 | 229 s |
| **d = 4 µm, octa n = 7, tol 1e-4** | 71 233 (3 336) | 722 (1.04e-4) | **0.0124** / 0.0021 / 0.0012 / 0.00027 | 0.0012 | 0.0040 | **0.0062** / 0.0029 / 0.0018 / 0.00075 | 0.0049 | 2113 s |

The 1 µm test spent 28 s in the assembly, 149 s in GMRES and 48 s in the field evaluation
(peak RSS 7.6 GB); the 4 µm test 174 s, 1136 s and 775 s (peak RSS 26.5 GB). Margins against
the lit bound 0.1: 20× (1 µm) and 8.0× (4 µm).

Evidence at the paper's tolerance 1e-3 (the acceptance case before the review): the benchmark
`specklebem_ag_near_field --mesh octa --n 7 --d 4e-6 --tol 1e-3` with the same check and the
same build, 2026-10-14: GMRES 442 iterations (true residual 1.04e-3); 71 233 points (3 336
excluded); lit max **0.0876** / p99 0.0162 / p95 0.0090 / median 0.0020 at (233, −2220) nm;
lit on z = 0 0.0242; lit beside the shadow 0.0220 (at (−2281, 40) nm, just above the x-axis);
shadow max 0.0223 / 0.0153 / 0.0122 / 0.0048 at (−281, 1996) nm; interior abs 0.0120. It
passes with a margin of 1.14× (lit) and 4.5× for the forward region beside the shadow that the
stricter zone rule added to the 0.1 bound. Assembly 139 s, GMRES 584 s, field evaluation 801 s.

`tests/validation/test_ag_near_field.cpp` (icosphere n = 3, dense, tol 1e-6, 186 GMRES
iterations; CI release, 15–23 s alone, 140 s under `ctest -j 8` with the dense validation tests
oversubscribing OpenMP; 2026-10-15): lit max 0.050 (p95 0.026; x-axis 0.020, beside the shadow
0.028), shadow max 0.086 (p95 0.046) at (0, 550) nm in the geometric shadow right behind the
sphere. Verified in the review round by
`ctest --preset win-release -L '^validation$' -R "near field" -V` (WARN output: 27 329 points,
6 460 excluded at d_min = 37.7 nm; lit 15 485 points, max 0.0502, p95 0.0255 at (148, −713) nm;
lit on z = 0 0.0202; lit beside the shadow 0.0282; shadow 3 195 points, max 0.0860, p95 0.0456
at (0, 550) nm; interior abs 0.0131; 15.2 s).
