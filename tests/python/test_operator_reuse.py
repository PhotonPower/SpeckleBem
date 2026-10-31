"""Operator reuse across excitations (WP-KR1, ADR 0013 decision 1): Simulation.set_excitation
and solve(initial_guess="previous").

The PML case is the small rough surface of test_pml.py (compression "mlfmm", the automatic ILUT
preconditioner, relaxed dark band); the box case the cheap sphere of test_simulation.py. Every
swap is compared with a fresh Simulation built with the new excitation (bitwise: the same
deterministic assembly, preconditioner and GMRES).
"""

import os

import numpy as np
import pytest

import specklebem as sb
from specklebem import io as sio

LAMBDA = 500e-9
H = 100e-9
CHEAP = dict(
    outer_grading_levels=0, target_accuracy=0.0, quad_degree_near=5, stretched_touching_points=4
)


def rough_mesh():
    params = sb.OpenSurfaceParams(
        patch_size=0.4e-6, taper_width=0.2e-6, gap_width=0.2e-6, pml_width=0.2e-6
    )
    hm = sb.generate_gaussian_height_map(L=0.8e-6, sigma=20e-9, Lc=200e-9, mesh_size=H, seed=29)
    return sb.make_open_surface_mesh(hm, params)


def beam(pol="p", theta_deg=0.0, region_radius=0.9e-6, wavelength=LAMBDA):
    return sb.AngularSpectrumBeam(
        wavelength,
        0.4e-6,
        pol,
        incidence_angle=np.radians(theta_deg),
        region_radius=region_radius,
    )


def make_pml(exc):
    return sb.Simulation.with_pml(
        rough_mesh(),
        exc,
        pml=sb.PmlTermination(sigma_max=1.0, correlation_length=200e-9, dark_band_tolerance=0.9),
        object=sb.Material(eps_r=2.25),
        compression="mlfmm",
        kernels=CHEAP,
        gmres=dict(tol=1e-9, max_iter=2000, verbose=False),
    )


DIRS = np.array([[0.0, 0.0, -1.0], [0.4, 0.0, -0.9], [0.0, 0.5, -0.8]])


def test_set_excitation_pml_route_equals_fresh_simulation():
    b0s, b20p = beam("s"), beam("p", 20.0)
    sim = make_pml(b0s)
    assert sim.preconditioner == "ilut"
    assert sim.solve().converged
    x1 = sim.currents
    rhs1 = sim.rhs()
    sim.set_excitation(b20p)
    assert sim.excitation is b20p
    assert sim.assembled and not sim.solved
    assert sim.last_result is None
    for call in (lambda: sim.currents, lambda: sim.far_field(DIRS), lambda: sim.drc(DIRS)):
        with pytest.raises(RuntimeError):
            call()
    assert sim.incident_power() == b20p.power
    fresh = make_pml(b20p)
    fresh.assemble()
    np.testing.assert_array_equal(sim.rhs(), fresh.rhs())
    assert sim.dark_band_ratio == fresh.dark_band_ratio
    r, f = sim.solve(), fresh.solve()
    assert r.converged and r.iterations == f.iterations
    np.testing.assert_array_equal(sim.currents, fresh.currents)
    np.testing.assert_array_equal(sim.far_field(DIRS), fresh.far_field(DIRS))
    np.testing.assert_array_equal(sim.drc(DIRS), fresh.drc(DIRS))
    assert sim.reflected_power(12, 24) == fresh.reflected_power(12, 24)

    # Back to the first excitation, started from the previous currents (of 20 deg p: an
    # uncorrelated guess, its residual is about 1.2 here, so it does not help; it must still
    # converge to the same currents).
    sim.set_excitation(b0s)
    np.testing.assert_array_equal(sim.rhs(), rhs1)
    p = sim.solve(initial_guess="previous")
    assert p.converged and p.residual_history[0] > 0.1
    assert np.linalg.norm(sim.currents - x1) <= 1e-7 * np.linalg.norm(x1)
    assert sim.last_result.iterations == p.iterations


def test_set_excitation_refusals_and_initial_guess_errors():
    sim = make_pml(beam("s"))
    with pytest.raises(RuntimeError, match="previous solve"):
        sim.solve(initial_guess="previous")
    with pytest.raises(ValueError, match="initial_guess"):
        sim.solve(initial_guess="last")
    with pytest.raises(ValueError):
        sim.set_excitation(None)
    with pytest.raises(ValueError, match="wavelength"):
        sim.set_excitation(beam(wavelength=600e-9))
    with pytest.raises(ValueError, match="unmasked part of the mesh"):
        sim.set_excitation(beam(region_radius=0.5e-6))
    with pytest.raises(ValueError, match="no dark band"):
        sim.set_excitation(sb.PlaneWave(LAMBDA, [0, 0, 1], [1, 0, 0]))
    # Before assemble(): only replaced, as if constructed with it.
    b = beam("p", 20.0)
    sim.set_excitation(b)
    assert sim.excitation is b and not sim.assembled


def test_set_excitation_box_route():
    cheap = dict(
        quad_degree_far=1,
        quad_degree_near=2,
        quad_degree_sing=2,
        outer_grading_levels=0,
        target_accuracy=0.0,
    )
    kw = dict(object=sb.Material(eps_r=2.25), kernels=cheap, gmres=dict(tol=1e-10, verbose=False))
    w1 = sb.PlaneWave(1e-6, [0, 0, 1], [1, 0, 0])
    w2 = sb.PlaneWave(1e-6, [0.6, 0, 0.8], [0, 1, 0])
    sim = sb.Simulation(sb.make_icosphere(0.5e-6, 1), w1, **kw)
    sim.solve()
    sim.set_excitation(w2)
    fresh = sb.Simulation(sb.make_icosphere(0.5e-6, 1), w2, **kw)
    sim.solve()
    fresh.solve()
    np.testing.assert_array_equal(sim.currents, fresh.currents)
    th = np.linspace(0, np.pi, 7)
    np.testing.assert_array_equal(sim.bistatic_rcs(th), fresh.bistatic_rcs(th))
    direct = sb.Simulation(sb.make_icosphere(0.5e-6, 1), w1, solver="direct", **kw)
    direct.solve()
    with pytest.raises(ValueError, match="GMRES"):
        direct.solve(initial_guess="previous")


@pytest.fixture(scope="module")
def h5py():
    if os.environ.get("SPECKLEBEM_REQUIRE_H5PY", "0") not in ("", "0"):
        import h5py as mod  # CI: a missing h5py is a failure, not a skip

        return mod
    return pytest.importorskip("h5py")


class _SwapAfterSnapshot:
    """A Simulation proxy whose _save_snapshot() is followed at once by set_excitation(other):
    the deterministic stand-in for a set_excitation() in another thread between the snapshot
    and the attribute reads of save_hdf5."""

    def __init__(self, sim, other):
        self._sim, self._other = sim, other

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def _save_snapshot(self):
        snap = self._sim._save_snapshot()
        self._sim.set_excitation(self._other)
        return snap


def test_hdf5_snapshot_holds_the_saved_excitation(h5py, tmp_path):
    """save_hdf5 takes the excitation and the dark-band ratio from the save snapshot (one lock
    with the currents and the right-hand side), not from separate reads afterwards (W3 of the
    WP-KR1 review)."""
    b0s, b20p = beam("s"), beam("p", 20.0)
    sim = make_pml(b0s)
    assert sim.solve().converged
    ratio0 = sim.dark_band_ratio
    sim.set_excitation(b20p)
    assert sim.solve().converged
    ratio20 = sim.dark_band_ratio
    assert ratio20 != ratio0
    x20, rhs20 = sim.currents, sim.rhs()

    snap = sim._save_snapshot()
    assert snap["excitation"] is b20p and snap["dark_band_ratio"] == ratio20
    np.testing.assert_array_equal(snap["currents"], x20)

    # A swap to 0 deg s right after the snapshot: the file still describes the 20 deg p solve.
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "swap.h5", _SwapAfterSnapshot(sim, b0s)))
    assert sim.excitation is b0s and not sim.solved and sim.dark_band_ratio == ratio0
    np.testing.assert_array_equal(r.currents, x20)
    np.testing.assert_array_equal(r.rhs, rhs20)
    x = r.excitation
    assert x["type"] == "AngularSpectrumBeam" and x["polarization"] == "P"
    assert x["incidence_angle"] == b20p.incidence_angle and x["power"] == b20p.power
    assert r.config["pml"]["dark_band_ratio"] == ratio20
