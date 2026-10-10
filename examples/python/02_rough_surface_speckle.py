"""Phase 6 target example: Si rough surface, 30 x 30 um, sigma = 100 nm, Lc = 1 um.

    import specklebem as sb
    L, sigma, Lc, h, wl = 30e-6, 100e-9, 1e-6, 50e-9, 500e-9
    si, vac = sb.silicon_500nm(), sb.vacuum()
    # Closing box from the materials (ADR 0006, amendment 2026-10-10): depth, coarse spacing
    # <= lambda_1 / 5 and the mandatory fine band 3 delta + 3 sigma for Si.
    box = sb.rough_surface_box_params(si, vac, wl, sigma, h)
    surf = sb.RoughSurface(L=L, sigma=sigma, Lc=Lc, mesh_size=h, seed=42, **box.kwargs())
    waist = L / 4                    # 7.5 um: w0 <= L / 4 (ADR 0006 amendment)
    sb.check_beam_waist(L, waist)    # raises ValueError for a wider waist
    beam = sb.GaussianBeam(wavelength=wl, waist=waist, polarization="p")
    sim  = sb.Simulation(surf.mesh, beam, object=si, formulation="ICTF")
    sim.solve(tol=1e-3)
    E = sim.field(sb.plane_grid(origin=[0, 0, -1e-3], u="x", v="y", size=300e-6, n=301))
    I = sb.intensity(E)
    print("contrast", sb.speckle_contrast(I))
    mu_A = sb.field_correlation_from_intensity(I)

GaussianBeam is the paraxial beam: it is fine for qualitative work, but it is 2.8 % off in
reflectance at w0 = 0.67 um (ADR 0006 amendment, decision 4). Quantitative rough-surface
results use the rigorous angular-spectrum Gaussian beam (AngularSpectrumBeam, WP-E1, landing
separately) in its place.

The API sketch above is the specification for docs/10_python_api.md. The runnable part below
only evaluates the box defaults and the waist rule (the 30 um patch needs the MLFMM of
Phase 6).
"""

import specklebem as sb

L, sigma, h, wl = 30e-6, 100e-9, 50e-9, 500e-9
box = sb.rough_surface_box_params(sb.silicon_500nm(), sb.vacuum(), wl, sigma, h)
sb.check_beam_waist(L, L / 4)
print(sb.__version__, box)
