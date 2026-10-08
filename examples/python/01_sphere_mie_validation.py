"""Phase 2 target example: Ag sphere (d = 4 um) vs Mie theory at 500 nm.

    import specklebem as sb
    mesh = sb.make_sphere(radius=2e-6, target_edge_length=500e-9 / 27)
    exc  = sb.PlaneWave(wavelength=500e-9, direction=[0, 0, 1], polarization=[1, 0, 0])
    sim  = sb.Simulation(mesh, exc, object=sb.silver_500nm(), formulation="ICTF",
                         diagonal_preconditioner=True, compression="mlfmm")
    res  = sim.solve(tol=1e-3)
    theta = np.linspace(0, np.pi, 181)
    rcs_bem = sim.bistatic_rcs(plane="xz", angles=theta)
    rcs_mie = sb.Mie(radius=2e-6, wavelength=500e-9, material=sb.silver_500nm()).bistatic_rcs(theta, 0)
    # expected: relative error (Eq. 7 of the paper) below 0.5 %

The API sketch above is the specification for docs/10_python_api.md.
"""
import specklebem as sb

print(sb.__version__, sb.silver_500nm().n)
