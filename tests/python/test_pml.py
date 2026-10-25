"""Python mirror of the open-surface PML route (WP-PML3a, ADR 0009): make_open_surface_mesh,
PmlTermination, CurrentWindow and Simulation.with_pml.

The surfaces are small (h = 100 nm, 8 x 8 or 16 x 16 cells) and have no dark band at 500 nm,
so the structural cases relax the dark-band tolerance; the tiny solve uses a low-index object
and sigma_max = 1 to keep the stretched quadrature cheap (as tests/unit/test_pml_route.cpp).
"""

import numpy as np
import pytest

import specklebem as sb

LAMBDA = 500e-9
H = 100e-9
RELAXED = dict(dark_band_tolerance=0.9)


def flat_map(edge, h=H):
    cells = int(round(edge / h))
    return sb.HeightMap(np.zeros((cells + 1, cells + 1)), edge / cells, edge / cells)


def open_mesh(patch=0.4e-6, taper=0.2e-6, gap=0.2e-6, pml=0.2e-6):
    params = sb.OpenSurfaceParams(
        patch_size=patch, taper_width=taper, gap_width=gap, pml_width=pml
    )
    return sb.make_open_surface_mesh(flat_map(patch + 2 * taper), params)


def beam(waist=0.4e-6, region_radius=0.9e-6):
    return sb.AngularSpectrumBeam(LAMBDA, waist, "p", region_radius=region_radius)


def test_open_surface_mesh_zones_and_half_widths():
    om = open_mesh()
    assert om.mesh.num_triangles == 2 * 16 * 16
    zones = om.zones
    assert zones.shape == (om.mesh.num_triangles,)
    assert set(np.unique(zones)) == {0, 1, 2, 3}
    assert int(sb.SurfaceZone.Pml) == 3
    assert om.patch_half == pytest.approx(0.2e-6)
    assert om.taper_end == pytest.approx(0.4e-6)
    assert om.pml_start == pytest.approx(0.6e-6)
    assert om.outer_half == pytest.approx(0.8e-6)
    assert "OpenSurfaceMesh" in repr(om)
    with pytest.raises(ValueError):
        sb.make_open_surface_mesh(
            flat_map(0.8e-6),
            sb.OpenSurfaceParams(patch_size=0.4e-6, taper_width=0.2e-6, pml_width=0.0),
        )


def test_pml_termination_and_window():
    t = sb.PmlTermination()
    assert t.sigma_max == 4.0 and t.order == 2.0
    assert t.correlation_length == 0.0 and t.dark_band_tolerance == 1e-4
    t = sb.PmlTermination(sigma_max=3.0, order=1.0)
    assert t.sigma_max == 3.0 and "PmlTermination" in repr(t)
    w = sb.CurrentWindow(1.0, 3.0)
    assert w(0.5) == 1.0 and w(3.0) == 0.0
    assert w(2.0) == pytest.approx(0.5, abs=1e-15)
    with pytest.raises(ValueError):
        sb.CurrentWindow(2.0, 1.0)


def test_with_pml_construction_and_properties():
    om = open_mesh()
    sim = sb.Simulation.with_pml(
        om, beam(), pml=sb.PmlTermination(**RELAXED), object=sb.silicon_500nm()
    )
    assert sim.has_pml
    w = sim.far_field_window
    assert w.inner_half == pytest.approx(om.taper_end)
    assert w.outer_half == pytest.approx(om.pml_start)
    assert 0.0 < sim.dark_band_ratio < 0.9
    assert sim.zone_half_widths[2] == pytest.approx(om.pml_start)
    assert sim.num_unknowns == 2 * om.mesh.num_edges
    assert "termination:    PML" in sim.report()

    box = sb.Simulation(
        sb.make_icosphere(0.3e-6, 1),
        sb.PlaneWave(LAMBDA, [0, 0, 1], [1, 0, 0]),
        object=sb.silicon_500nm(),
    )
    assert not box.has_pml
    assert box.far_field_window is None
    with pytest.raises(RuntimeError):
        box.dark_band_ratio


def test_with_pml_refusals():
    si = sb.silicon_500nm()
    relaxed = sb.PmlTermination(**RELAXED)
    with pytest.raises(ValueError, match="no dark band"):
        sb.Simulation.with_pml(open_mesh(), beam(), object=si)
    with pytest.raises(ValueError, match="dense"):  # "mlfmm" (WP-PML5a) needs GMRES
        sb.Simulation.with_pml(
            open_mesh(), beam(), pml=relaxed, object=si, compression="mlfmm", solver="direct"
        )
    with pytest.raises(ValueError, match="gap"):
        sb.Simulation.with_pml(open_mesh(gap=0.0), beam(), pml=relaxed, object=si)
    with pytest.raises(ValueError, match="unmasked part"):
        sb.Simulation.with_pml(
            open_mesh(), beam(region_radius=0.8e-6), pml=relaxed, object=si
        )
    with pytest.raises(ValueError, match="sigma_max"):
        sb.Simulation.with_pml(
            open_mesh(), beam(), pml=sb.PmlTermination(sigma_max=0.0, **RELAXED), object=si
        )


def test_with_pml_tiny_solve():
    om = open_mesh(patch=0.2e-6, taper=0.1e-6, gap=0.1e-6, pml=0.1e-6)
    sim = sb.Simulation.with_pml(
        om,
        beam(region_radius=0.6e-6),
        pml=sb.PmlTermination(sigma_max=1.0, **RELAXED),
        object=sb.Material(eps_r=2.25),
        solver="direct",
        kernels=dict(
            outer_grading_levels=0,
            target_accuracy=0.0,
            quad_degree_near=5,
            stretched_touching_points=4,
        ),
    )
    res = sim.solve()
    assert res.converged
    x = sim.currents
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) > 0
    b = sim.rhs()
    assert np.count_nonzero(b == 0) > 0  # masked PML test functions
    F = sim.far_field(np.array([[0.0, 0.0, -1.0], [0.5, 0.0, -0.8]]))
    assert F.shape == (2, 3) and np.all(np.isfinite(F)) and np.linalg.norm(F) > 0
    F2 = sim.far_field([0.0, 0.0, -1.0], window=sim.far_field_window)
    np.testing.assert_array_equal(F2, F[0])
    with pytest.raises(ValueError, match="beyond the PML stretch start"):
        sim.far_field([0.0, 0.0, -1.0], window=sb.CurrentWindow(om.taper_end, om.outer_half))
    # PML-route exception rule (WP25 review): valid call, wrong state -> RuntimeError; an
    # invalid direction (k_z > 0: R2 on this route) -> ValueError.
    with pytest.raises(RuntimeError, match="not available on the PML route"):
        sim.field([0.0, 0.0, -1e-6])
    with pytest.raises(RuntimeError, match="not available on the PML route"):
        sim.bistatic_rcs([0.0, 0.1])
    with pytest.raises(ValueError, match="k_z > 0"):
        sim.far_field([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="k_z > 0"):
        sim.far_field([[0.0, 0.0, -1.0], [0.6, 0.0, 0.8]], window=sim.far_field_window)
    assert np.all(np.isfinite(sim.far_field([1.0, 0.0, 0.0])))  # grazing k_z = 0 is accepted


def test_pml_termination_huygens_options():
    """WP-PML6 (ADR 0010): far-field source and Huygens-plane options of PmlTermination."""
    t = sb.PmlTermination()
    assert t.far_field_source == "currents"
    assert t.huygens_z_h is None and t.huygens_spacing is None and t.huygens_window is None
    w = sb.CurrentWindow(0.3e-6, 0.6e-6)
    t = sb.PmlTermination(
        far_field_source="Huygens", huygens_z_h=0.4e-6, huygens_spacing=1e-7, huygens_window=w
    )
    assert t.far_field_source == "huygens" and "far_field_source=huygens" in repr(t)
    assert t.huygens_z_h == 0.4e-6 and t.huygens_spacing == 1e-7
    assert t.huygens_window.outer_half == 0.6e-6
    t.far_field_source = "currents"
    t.huygens_z_h = None
    assert t.far_field_source == "currents" and t.huygens_z_h is None
    with pytest.raises(ValueError, match="currents"):
        sb.PmlTermination(far_field_source="plane")
    with pytest.raises(ValueError, match="huygens.spacing"):
        sb.Simulation.with_pml(
            open_mesh(),
            beam(),
            pml=sb.PmlTermination(huygens_spacing=0.3e-6, **RELAXED),
            object=sb.silicon_500nm(),
        )
    sim = sb.Simulation.with_pml(
        open_mesh(),
        beam(),
        pml=sb.PmlTermination(far_field_source="huygens", huygens_z_h=0.4e-6, **RELAXED),
        object=sb.silicon_500nm(),
    )
    assert sim.far_field_source == "huygens"
    assert sim.pml_termination.huygens_z_h == 0.4e-6
    assert "far field:      Huygens plane" in sim.report()
    box = sb.Simulation(
        sb.make_icosphere(0.3e-6, 1),
        sb.PlaneWave(LAMBDA, [0, 0, 1], [1, 0, 0]),
        object=sb.silicon_500nm(),
    )
    assert box.far_field_source == "currents"


def test_with_pml_tiny_solve_huygens_far_field():
    """Simulation.far_field(source="huygens") on the PML route: the termination's default
    source, per-call z_h / spacing / plane window, and the refusals."""
    om = open_mesh(patch=0.2e-6, taper=0.1e-6, gap=0.1e-6, pml=0.1e-6)
    sim = sb.Simulation.with_pml(
        om,
        beam(region_radius=0.6e-6),
        pml=sb.PmlTermination(sigma_max=1.0, far_field_source="huygens", **RELAXED),
        object=sb.Material(eps_r=2.25),
        solver="direct",
        kernels=dict(
            outer_grading_levels=0,
            target_accuracy=0.0,
            quad_degree_near=5,
            stretched_touching_points=4,
        ),
    )
    assert sim.solve().converged
    dirs = np.array([[0.0, 0.0, -1.0], [0.5, 0.0, -0.8]])
    F = sim.far_field(dirs)  # the termination's source: Huygens
    np.testing.assert_array_equal(sim.far_field(dirs, source="huygens"), F)
    Fc = sim.far_field(dirs, source="currents")
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(Fc))
    assert np.linalg.norm(F - Fc) > 0  # a different (equivalent) source
    F2 = sim.far_field(dirs, source="huygens", z_h=0.3e-6, spacing=1e-7)
    assert np.all(np.isfinite(F2)) and not np.array_equal(F2, F)
    F3 = sim.far_field(dirs, window=sb.CurrentWindow(0.1e-6, 0.25e-6), source="huygens")
    assert not np.array_equal(F3, F)
    D = sim.drc(dirs)  # the observables follow the termination's source
    ratio = D / np.sum(np.abs(F) ** 2, axis=1)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    Dc = sim.drc(dirs[:1])
    assert Dc != pytest.approx(np.sum(np.abs(Fc[0]) ** 2) * ratio[0], rel=1e-9)
    with pytest.raises(ValueError, match="Huygens source only"):
        sim.far_field(dirs, source="currents", z_h=0.3e-6)
    with pytest.raises(ValueError, match="currents"):
        sim.far_field(dirs, source="plane")
    with pytest.raises(ValueError, match="k_z > 0"):
        sim.far_field([0.0, 0.0, 1.0], source="huygens")
    with pytest.raises(ValueError, match="beyond the PML stretch start"):
        sim.far_field(dirs, window=sb.CurrentWindow(0.1e-6, 0.35e-6), source="huygens")


def test_with_pml_mlfmm_solve():
    """compression="mlfmm" on the PML route (WP-PML5a: MLFMM on the real functions + the exact
    PML ring) solves and agrees with the dense GMRES solve of the same system."""
    cheap = dict(
        outer_grading_levels=0, target_accuracy=0.0, quad_degree_near=5, stretched_touching_points=4
    )
    common = dict(
        pml=sb.PmlTermination(sigma_max=1.0, **RELAXED),
        object=sb.Material(eps_r=2.25),
        kernels=cheap,
        gmres=dict(tol=1e-8, max_iter=2000),
    )
    fast = sb.Simulation.with_pml(open_mesh(), beam(), compression="mlfmm", **common)
    res = fast.solve()
    assert res.converged
    report = fast.report()
    assert "PML ring:" in report and "stretched" in report
    dense = sb.Simulation.with_pml(open_mesh(), beam(), **common)
    assert dense.solve().converged
    x, xd = fast.currents, dense.currents
    assert np.all(np.isfinite(x))
    # d0 = 3 MLFMM on the real functions (3 octree levels on this mesh), exact ring.
    assert np.linalg.norm(x - xd) < 5e-2 * np.linalg.norm(xd)
    b = fast.rhs()
    assert np.count_nonzero(b == 0) > 0  # masked test rows stay in the system with rhs 0


def test_with_pml_mlfmm_ring_keys():
    """WP-PML5b: the PML ring keys of mlfmm= reach the C++ operator (ACA far ring by default, the
    exact far ring of WP-PML5a with pml_ring_use_aca=False); both solve to the same currents."""
    cheap = dict(
        outer_grading_levels=0, target_accuracy=0.0, quad_degree_near=5, stretched_touching_points=4
    )
    common = dict(
        pml=sb.PmlTermination(sigma_max=1.0, **RELAXED),
        object=sb.Material(eps_r=2.25),
        kernels=cheap,
        gmres=dict(tol=1e-8, max_iter=2000),
        compression="mlfmm",
    )
    keys = dict(pml_ring_eta=20.0, pml_ring_max_rank=1000, pml_ring_aca_epsilon=1e-6)
    aca = sb.Simulation.with_pml(open_mesh(), beam(), mlfmm=keys, **common)
    exact = sb.Simulation.with_pml(
        open_mesh(), beam(), mlfmm=dict(pml_ring_use_aca=False), **common
    )
    assert aca.solve().converged and exact.solve().converged
    rep = aca.report()
    assert "ring far ACA:" in rep and "pml_ring: use_aca true" in rep
    assert "eta 20" in rep
    assert "ring far ACA:" not in exact.report() and "pairs, exact)" in exact.report()
    cfg = aca.config["mlfmm"]
    assert cfg["pml_ring_eta"] == 20.0 and cfg["pml_ring_max_rank"] == 1000
    assert cfg["pml_ring_use_aca"] is True and cfg["pml_ring_row_clusters"] is True
    assert cfg["pml_ring_aca_epsilon"] == 1e-6 and cfg["pml_ring_max_far_bytes"] == 0
    x, xe = aca.currents, exact.currents
    assert np.linalg.norm(x - xe) < 1e-4 * np.linalg.norm(xe)
    with pytest.raises(ValueError, match="pml_ring"):
        sb.Simulation.with_pml(open_mesh(), beam(), mlfmm=dict(pml_ring_eta=0.0), **common)
    with pytest.raises(ValueError, match="unknown key"):
        sb.Simulation.with_pml(open_mesh(), beam(), mlfmm=dict(pml_ring_bogus=1), **common)
