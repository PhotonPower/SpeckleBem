# Absolute DRC and far-field evaluation cost (WP25)

Acceptance (`docs/backlog.md` WP25): "DRC normalised against Fresnel on a flat surface (< 1 %);
far-field / grid evaluation time reported". Test
`tests/validation_large/test_drc_fresnel_large.cpp` (label `validation-large`, release only):

```bash
OMP_NUM_THREADS=4 ctest --preset win-release -L validation-large -R "DRC / Fresnel" -V
```

Hardware: Windows 11, 12 cores, 128 GB, win-release (UCRT64 GCC, OpenBLAS), `OMP_NUM_THREADS=4`,
measured 2026-10-14 while another worker ran a dense benchmark on the same machine (the timings
are upper bounds for an idle machine).

## Case

Flat Si open surface on the PML route (the WP-PML3a smoke geometry): patch 1.5 µm, taper 0.5 µm,
gap 0.5 µm, PML 1 µm (σ_max 4, order 2), h = 125 nm (3 872 triangles, 2N = 11 440); rigorous
angular-spectrum beam at normal incidence, w₀ = 0.5 µm, p; ICTF (automatic), dense, direct LU.
Built with the public `Simulation(OpenSurfaceMesh, PmlTermination, ...)` constructor; the DRC
is `Simulation::drc` (absolute, per steradian, P_inc = `Simulation::incident_power()` =
`AngularSpectrumBeam::power()`, the PML route's current window applied automatically).

## Result: DRC integral vs Fresnel

| Quantity | Value |
|----------|-------|
| P_inc | 5.49404·10⁻¹⁶ W |
| R = Σ w D on the 1° reflection hemisphere (`hemisphere_grid(-1, 90, 180)`) | 0.388091 |
| same on the 0.5° grid (180 × 360) | 0.388091 |
| Fresnel beam reflectance R_beam (`fresnel_flat::beam_reflectance`) | 0.388176 |
| plane-wave Fresnel \|r(0°)\|² | 0.387718 |
| **relative error R / R_beam − 1** | **−0.022 %** (criterion < 1 %: met) |
| D at the specular direction (−z) | 2.564 sr⁻¹ |
| assembly / LU | 1 560 s / 62 s (true residual 2·10⁻¹⁶) |

R equals `Simulation::reflected_power(90, 180) / incident_power()` to rounding and the support-code
integration of WP-PML3a (`fresnel_flat::hemisphere_power`) bitwise, and it reproduces the
WP-PML3a smoke result (−0.022 %) with the library functions.

## Evaluation time

The far field is a direct radiation integral, O(N_dir × N_src) with N_src = triangles × 12
(Dunavant degree 6), parallel over directions (OpenMP).

| Grid | Directions | Source points | Time (4 threads) |
|------|-----------:|--------------:|-----------------:|
| 1° hemisphere, windowed (PML route, physical part only (the window tapers the gap and drops the PML)) | 16 200 | (window drops the PML) | 3.74 s |
| 0.5° hemisphere (180 × 360), windowed | 64 800 | (as above) | 11.5 s |
| 1° hemisphere, every triangle, no window (box-route cost) | 16 200 | 46 464 | 9.03 s |

Cost per (direction × source point): **12.0 ns** on 4 threads of the loaded machine (from the
unwindowed row; a problem copy without the stretch, for timing only).

**Extrapolation (linear in the number of triangles; not measured):** a surface with 2N ≈ 10⁶
has ≈ 3.4·10⁵ triangles (2N / triangles = 2.95 here), i.e. 4.1·10⁶ source points: the 1° grid
(16 200 directions) then takes ≈ 790 s on 4 threads (≈ 4–5 min on 12 cores if the direction loop
scales), the 0.5° grid ≈ 3 200 s on 4 threads (≈ 15–20 min on 12 cores). The far-field / DRC
evaluation stays a small fraction of a 2N ≈ 10⁶ MLFMM solve (tens of minutes to hours,
`benchmarks/results/si_large_area.md`), so no acceleration is implemented here. Follow-up if Phase 6 needs many
dense angular grids (frequency or angle sweeps, ≥ 10⁵ directions per solve): evaluate the far
field through the MLFMM radiation patterns (aggregation to the root level and interpolation to
the requested directions, O(N + N_dir) per level instead of O(N · N_dir)). Near-field speckle
planes (`Simulation.field` on 300 × 300 points) are a different evaluation and were not timed
here.
