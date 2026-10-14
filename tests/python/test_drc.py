"""Python mirror of the absolute far-field observables (WP25): SphereGrid, hemisphere_grid,
sphere_grid, GaussianBeam.power, Simulation.incident_power / drc / radiated_power /
reflected_power / scattered_power and the absolute cylinder DRC.

Driver cases use the cheap sphere of test_simulation.py (radius 0.5 um, icosphere n = 1,
lambda = 1 um, cheap quadrature, direct LU) and the tiny PML surface of test_pml.py. Physics
(Mie cross section, Fresnel reflectance) is in the C++ validation tests.
"""

import math

import numpy as np
import pytest

import specklebem as sb

RADIUS = 0.5e-6
LAMBDA = 1e-6
CHEAP = dict(
    quad_degree_far=1,
    quad_degree_near=2,
    quad_degree_sing=2,
    outer_grading_levels=0,
    target_accuracy=0.0,
)
N15 = sb.Material(eps_r=2.25)
ETA0 = 376.730313668  # Ohm, constants::eta0 of the C++ core


def sphere_sim(excitation):
    return sb.Simulation(
        sb.make_icosphere(RADIUS, 1), excitation, object=N15, solver="direct", kernels=CHEAP
    )


@pytest.fixture(scope="module")
def beam_sim():
    beam = sb.AngularSpectrumBeam(LAMBDA, 1e-6, "p", region_radius=1e-6)
    sim = sphere_sim(beam)
    sim.solve()
    return sim, beam


def test_grids():
    g = sb.hemisphere_grid(-1, 90, 180)
    assert isinstance(g, sb.SphereGrid) and len(g) == 90 * 180
    assert g.directions.shape == (16200, 3) and g.weights.shape == (16200,)
    assert g.theta.shape == (16200,)
    assert np.all(g.directions[:, 2] < 0)
    np.testing.assert_allclose(np.linalg.norm(g.directions, axis=1), 1.0, rtol=1e-15)
    assert g.weights.sum() == pytest.approx(2 * math.pi, rel=1e-14)
    assert not g.weights.flags.writeable  # read-only view into the grid
    s = sb.sphere_grid(24, 48)
    assert s.weights.sum() == pytest.approx(4 * math.pi, rel=1e-14)
    # int cos^2(theta) dOmega = 4 pi / 3 over the sphere
    assert np.dot(s.weights, s.directions[:, 2] ** 2) == pytest.approx(4 * math.pi / 3, 1e-13)
    assert "SphereGrid(" in repr(s)
    for bad in ((0.5, 4, 8), (-1, 0, 8), (-1, 4, 3)):
        with pytest.raises(ValueError):
            sb.hemisphere_grid(*bad)
    with pytest.raises(ValueError):
        sb.sphere_grid(4, 2)
    assert "[sr]" in sb.hemisphere_grid.__doc__ and "[sr]" in sb.SphereGrid.weights.__doc__


def test_gaussian_beam_power():
    gb = sb.GaussianBeam(500e-9, 2e-6)
    assert gb.power == pytest.approx(math.pi * 4e-12 / (4 * ETA0), rel=1e-9)
    assert "[W]" in sb.GaussianBeam.power.__doc__
    sim = sphere_sim(sb.GaussianBeam(LAMBDA, 1.5e-6))
    assert sim.incident_power() == sb.GaussianBeam(LAMBDA, 1.5e-6).power


def test_beam_drc_and_powers(beam_sim):
    sim, beam = beam_sim
    assert sim.incident_power() == beam.power
    g = sb.hemisphere_grid(-1, 6, 12)
    d = sim.drc(g.directions)
    assert d.shape == (72,) and np.all(d >= 0) and d.max() > 0
    p_refl = sim.reflected_power(6, 12)
    assert np.dot(g.weights, d) == pytest.approx(p_refl / sim.incident_power(), rel=1e-13)
    assert sim.radiated_power(g.directions, g.weights) == p_refl
    assert sim.reflectance(6, 12) == p_refl / sim.incident_power()
    with pytest.raises(ValueError):
        sim.reflectance(6, 3)
    # box route: far_field accepts the forward hemisphere (k_z > 0) too
    assert np.all(np.isfinite(sim.far_field([0.0, 0.0, 1.0])))
    # one (3,) direction -> float; drc = |F|^2 / (2 eta0 P_inc)
    d0 = sim.drc([0.0, 0.0, -2.0])
    assert isinstance(d0, float)
    F = sim.far_field([0.0, 0.0, -1.0])
    assert d0 == pytest.approx(np.vdot(F, F).real / (2 * ETA0 * beam.power), rel=1e-12)
    co, cross = sim.drc(g.directions, co_pol=[1, 0, 0])
    np.testing.assert_allclose(co + cross, d, rtol=1e-12, atol=1e-12 * d.max())
    co0, cross0 = sim.drc([0.0, 0.0, -1.0], co_pol=[1, 0, 0])
    assert isinstance(co0, float) and cross0 < 1e-3 * co0
    # the quadrature degree is passed through
    assert sim.reflected_power(6, 12, quad_degree=4) != p_refl
    full = sim.scattered_power(24, 48, quad_degree=4)
    halves = sim.reflected_power(12, 48, quad_degree=4)
    h_up = sb.hemisphere_grid(1, 12, 48)
    halves += sim.radiated_power(h_up.directions, h_up.weights, quad_degree=4)
    assert full == pytest.approx(halves, rel=1e-9)
    with pytest.raises(ValueError):
        sim.radiated_power(g.directions, g.weights[:5])
    with pytest.raises(ValueError):
        sim.radiated_power(g.directions, -g.weights)
    with pytest.raises(ValueError):
        sim.drc([[0.0, 0.0, np.nan]])
    with pytest.raises(ValueError):
        sim.drc(g.directions, co_pol=[0, 0, 0])
    with pytest.raises(ValueError):
        sim.reflected_power(0, 12)
    assert "dimensionless" in sb.Simulation.reflectance.__doc__
    for name in ("drc", "radiated_power", "reflected_power", "scattered_power"):
        assert "[W]" in getattr(sb.Simulation, name).__doc__ or "[1/sr]" in getattr(
            sb.Simulation, name
        ).__doc__


def test_plane_wave_has_no_incident_power():
    sim = sphere_sim(sb.PlaneWave(LAMBDA, [0, 0, 1], [1, 0, 0]))
    with pytest.raises(RuntimeError, match="plane wave"):
        sim.incident_power()
    with pytest.raises(RuntimeError):
        sim.reflected_power(4, 8)  # before solve()
    sim.solve()
    with pytest.raises(RuntimeError, match="plane wave"):
        sim.drc([0.0, 0.0, -1.0])
    with pytest.raises(RuntimeError, match="plane wave"):
        sim.reflectance(4, 8)
    # Scattering cross section from the power (the cheap n = 1 sphere is only indicative).
    c_sca = sim.scattered_power(32, 64) * 2 * ETA0  # |E0| = 1 V/m
    assert 0 < c_sca < 10 * math.pi * RADIUS**2


def test_pml_route_drc():
    h = 100e-9
    cells = 4
    hm = sb.HeightMap(np.zeros((cells + 1, cells + 1)), h, h)
    om = sb.make_open_surface_mesh(
        hm,
        sb.OpenSurfaceParams(
            patch_size=0.2e-6, taper_width=0.1e-6, gap_width=0.1e-6, pml_width=0.1e-6
        ),
    )
    beam = sb.AngularSpectrumBeam(500e-9, 0.4e-6, "p", region_radius=0.6e-6)
    sim = sb.Simulation.with_pml(
        om,
        beam,
        pml=sb.PmlTermination(sigma_max=1.0, dark_band_tolerance=0.9),
        object=N15,
        solver="direct",
        kernels=dict(
            outer_grading_levels=0,
            target_accuracy=0.0,
            quad_degree_near=5,
            stretched_touching_points=4,
        ),
    )
    sim.solve()
    g = sb.hemisphere_grid(-1, 6, 12)
    d = sim.drc(g.directions)  # windowed automatically
    assert np.all(np.isfinite(d)) and d.max() > 0
    assert np.dot(g.weights, d) == pytest.approx(sim.reflected_power(6, 12) / beam.power, 1e-13)
    assert sim.reflectance(6, 12) == sim.reflected_power(6, 12) / beam.power
    # PML-route exception rule: invalid direction -> ValueError (drc, radiated_power and
    # far_field alike); valid call in the wrong state -> RuntimeError.
    with pytest.raises(ValueError, match="k_z > 0"):
        sim.drc([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="k_z > 0"):
        sim.far_field([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="k_z > 0"):
        sim.radiated_power([[0.0, 0.0, 1.0]], [1.0])
    with pytest.raises(RuntimeError, match="PML route"):
        sim.scattered_power(4, 8)
    with pytest.raises(RuntimeError, match="PML route"):
        sim.bistatic_rcs([0.0, 0.1])
    with pytest.raises(RuntimeError, match="PML route"):
        sim.field([0.0, 0.0, -1e-6])


def test_cylinder_drc_absolute():
    rng = np.random.default_rng(25)
    n_theta, n_y = 3, 4
    E = rng.standard_normal((n_theta * n_y, 3)) + 1j * rng.standard_normal((n_theta * n_y, 3))
    rel = sb.differential_reflection_coefficient(E, n_theta, n_y)
    np.testing.assert_allclose(
        rel, (np.abs(E) ** 2).sum(axis=1).reshape(n_theta, n_y).mean(axis=1), rtol=1e-14
    )
    r, p = 1e-3, 2e-9
    d = sb.differential_reflection_coefficient(E, n_theta, n_y, radius=r, incident_power=p)
    np.testing.assert_allclose(d, rel * r**2 / (2 * ETA0 * p), rtol=1e-9)
    glass = sb.Material(eps_r=2.25)
    dg = sb.differential_reflection_coefficient(
        E, n_theta, n_y, radius=r, incident_power=p, background=glass
    )
    np.testing.assert_allclose(dg, 1.5 * d, rtol=1e-13)
    with pytest.raises(ValueError, match="both"):
        sb.differential_reflection_coefficient(E, n_theta, n_y, radius=r)
    with pytest.raises(ValueError):
        sb.differential_reflection_coefficient(E, n_theta, n_y, background=glass)
    with pytest.raises(ValueError):
        sb.differential_reflection_coefficient(E, n_theta, n_y, radius=r, incident_power=0.0)
    with pytest.raises(ValueError, match="lossless"):
        sb.differential_reflection_coefficient(
            E, n_theta, n_y, radius=r, incident_power=p, background=sb.Material(eps_r=2 - 0.1j)
        )
    assert "[1/sr]" in sb.differential_reflection_coefficient.__doc__
