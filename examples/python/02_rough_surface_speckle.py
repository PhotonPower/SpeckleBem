"""Phase 6 target example: Si rough surface, 30 x 30 um, sigma = 100 nm, Lc = 1 um.

    surf = sb.RoughSurface(L=30e-6, sigma=100e-9, Lc=1e-6, mesh_size=50e-9, seed=42)
    beam = sb.GaussianBeam(wavelength=500e-9, waist=10e-6, polarization="p")
    sim  = sb.Simulation(surf.mesh, beam, object=sb.silicon_500nm(), formulation="ICTF")
    sim.solve(tol=1e-3)
    E = sim.field(sb.plane_grid(origin=[0, 0, -1e-3], u="x", v="y", size=300e-6, n=301))
    I = sb.intensity(E)
    print("contrast", sb.speckle_contrast(I))
    mu_A = sb.field_correlation_from_intensity(I)
"""
