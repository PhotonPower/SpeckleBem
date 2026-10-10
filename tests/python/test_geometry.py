"""Python bindings of geometry/: TriangleMesh, spheres and rough surfaces (WP14b1)."""

import gc

import numpy as np
import pytest

import specklebem as sb

TETRA_V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# Counter-clockwise seen from outside: normals point out of the object.
TETRA_T = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)


def test_mesh_round_trip_and_views():
    mesh = sb.TriangleMesh(TETRA_V, TETRA_T)
    assert (mesh.num_vertices, mesh.num_triangles, mesh.num_edges) == (4, 4, 6)
    v, t = mesh.vertices, mesh.triangles
    np.testing.assert_array_equal(v, TETRA_V)
    np.testing.assert_array_equal(t, TETRA_T)
    assert v.dtype == np.float64 and t.dtype == np.int64
    assert v.shape == (4, 3) and t.shape == (4, 3) and mesh.edges.shape == (6, 2)
    assert v.flags.c_contiguous and t.flags.c_contiguous
    assert not v.flags.writeable and not t.flags.writeable
    with pytest.raises(ValueError):
        v[0, 0] = 1.0
    # Zero copy: the same buffer on every access; the input was copied.
    assert np.shares_memory(mesh.vertices, v) and not np.shares_memory(v, TETRA_V)
    assert mesh.is_closed()
    assert mesh.signed_volume() == pytest.approx(1.0 / 6.0, rel=1e-14)
    lo, hi = mesh.bounding_box()
    np.testing.assert_array_equal(lo, [0, 0, 0])
    np.testing.assert_array_equal(hi, [1, 1, 1])


def test_view_keeps_mesh_alive():
    mesh = sb.make_icosphere(2.0, 1)
    v = mesh.vertices
    expected = v.copy()
    del mesh
    gc.collect()
    np.testing.assert_array_equal(v, expected)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 2.0, rtol=1e-14)


def test_icosphere_counts_and_quality():
    for n in range(3):
        mesh = sb.make_icosphere(1e-6, n)
        nv, nt, ne = mesh.num_vertices, mesh.num_triangles, mesh.num_edges
        assert (nv, nt, ne) == (10 * 4**n + 2, 20 * 4**n, 30 * 4**n)
        assert nv - ne + nt == 2
        assert mesh.signed_volume() > 0
        q = mesh.quality()
        assert isinstance(q, sb.MeshQuality) and repr(q).startswith("<MeshQuality V=")
        assert q.euler_characteristic == 2 and q.closed and q.consistently_oriented
        assert q.num_triangles == nt and q.num_boundary_edges == 0
        assert 1.0 <= q.max_aspect_ratio < 1.5
        lo, hi = q.min_edge_length * (1 - 1e-12), q.max_edge_length * (1 + 1e-12)
        assert lo <= q.mean_edge_length <= hi
        assert "Euler characteristic    : 2" in str(q)
        assert mesh.quality_report() == str(q)
    sphere = sb.make_sphere(radius=1e-6, target_edge_length=0.3e-6, center=[1e-6, 0, 0])
    assert sphere.quality().mean_edge_length <= 0.3e-6
    np.testing.assert_allclose(sphere.vertices.mean(axis=0), [1e-6, 0, 0], atol=1e-12)


def _layouts(v, t):
    """The same mesh as int32, Fortran-order, strided and nested-list input."""
    v_wide = np.zeros((v.shape[0], 6))
    v_wide[:, ::2] = v
    t_wide = np.zeros((t.shape[0] * 2, 3), dtype=np.int64)
    t_wide[::2] = t
    return {
        "int32": (v, t.astype(np.int32)),
        "uint32": (v, t.astype(np.uint32)),
        "fortran": (np.asfortranarray(v), np.asfortranarray(t)),
        "strided": (v_wide[:, ::2], t_wide[::2]),
        "lists": (v.tolist(), t.tolist()),
    }


@pytest.mark.parametrize("layout", ["int32", "uint32", "fortran", "strided", "lists"])
@pytest.mark.parametrize("name", ["tetra", "icosphere"])
def test_mesh_input_layouts(name, layout):
    if name == "tetra":
        ref = sb.TriangleMesh(TETRA_V, TETRA_T)
    else:
        ref = sb.make_icosphere(1e-6, 1)
    v, t = _layouts(np.array(ref.vertices), np.array(ref.triangles))[layout]
    assert layout == "lists" or not (v.flags.c_contiguous and t.dtype == np.int64)
    mesh = sb.TriangleMesh(v, t)
    np.testing.assert_array_equal(mesh.vertices, ref.vertices)
    np.testing.assert_array_equal(mesh.triangles, ref.triangles)
    np.testing.assert_array_equal(mesh.edges, ref.edges)


def test_empty_mesh():
    mesh = sb.TriangleMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert (mesh.num_vertices, mesh.num_triangles, mesh.num_edges) == (0, 0, 0)
    assert not mesh.is_closed()
    with pytest.raises(RuntimeError):
        mesh.bounding_box()


def test_docstrings_name_units_and_conventions():
    assert "[m]" in sb.make_icosphere.__doc__ and "radius" in sb.make_icosphere.__doc__
    assert "re-orients" in sb.TriangleMesh.__doc__
    assert "flip_normals()" in sb.TriangleMesh.__doc__
    assert "seed" in sb.generate_gaussian_height_map.__doc__
    assert "only logged" in sb.RoughSurface.__doc__
    assert "[m]" in sb.RoughSurface.dx.__doc__
    objs = (sb.TriangleMesh.quality, sb.MeshQuality, sb.HeightMap, sb.make_rough_surface_mesh)
    for obj in objs:
        assert obj.__doc__ and "[m" in obj.__doc__


def test_flip_normals():
    mesh = sb.make_icosphere(1.0, 2)
    vol = mesh.signed_volume()
    t = mesh.triangles
    first = t[0].copy()
    mesh.flip_normals()
    assert mesh.signed_volume() == pytest.approx(-vol, rel=1e-12)
    assert sorted(t[0]) == sorted(first) and not np.array_equal(t[0], first)


@pytest.mark.parametrize(
    "vertices, triangles",
    [
        (TETRA_V, np.array([[0, 1, 4]])),  # index out of range
        (TETRA_V, np.array([[0, 1, 1]])),  # repeated vertex
        (TETRA_V[:, :2], TETRA_T),  # (n, 2) vertices
        (TETRA_V, TETRA_T[:, :2]),  # (F, 2) triangles
        (TETRA_V, TETRA_T.astype(np.float64)),  # float connectivity
        (TETRA_V.astype(np.complex128), TETRA_T),  # complex coordinates
        (np.where(TETRA_V == 1.0, np.nan, TETRA_V), TETRA_T),  # non-finite
    ],
)
def test_mesh_invalid_input(vertices, triangles):
    with pytest.raises(ValueError):
        sb.TriangleMesh(vertices, triangles)


def test_sphere_invalid_input():
    with pytest.raises(ValueError):
        sb.make_icosphere(-1.0, 1)
    with pytest.raises(ValueError):
        sb.make_sphere(1.0, 0.0)


ROUGH = dict(L=2e-6, sigma=50e-9, Lc=300e-9, mesh_size=50e-9, seed=7)


def test_height_map():
    hm = sb.generate_gaussian_height_map(**ROUGH)
    z = hm.z
    n = round(ROUGH["L"] / ROUGH["mesh_size"]) + 1
    assert z.shape == (n, n) and z.dtype == np.float64 and z.flags.c_contiguous
    assert hm.dx == pytest.approx(ROUGH["L"] / (n - 1), rel=1e-14) and hm.dy == hm.dx
    # Metres: sigma of the map equals the C++ value and is near the 50 nm target.
    rms = np.sqrt(np.mean((z - z.mean()) ** 2))
    assert rms == pytest.approx(hm.rms(), rel=1e-12)
    assert rms == pytest.approx(ROUGH["sigma"], rel=0.25)
    # Fixed seed: reproducible; FFT and direct convolution agree.
    np.testing.assert_array_equal(sb.generate_gaussian_height_map(**ROUGH).z, z)
    direct = sb.generate_gaussian_height_map(**ROUGH, use_fft=False).z
    np.testing.assert_allclose(direct, z, rtol=0, atol=1e-10 * ROUGH["sigma"])
    # HeightMap from user data (e.g. AFM), (n_x, n_y) with n_x != n_y.
    user = sb.HeightMap(np.zeros((5, 3)), dx=1e-7, dy=2e-7)
    np.testing.assert_array_equal(user.z, np.zeros((5, 3)))


def test_rough_surface_mesh():
    surf = sb.RoughSurface(**ROUGH, box_depth=1e-6, box_fine_depth=0.4e-6)
    hm = surf.height_map
    assert hm.shape == (41, 41) and not hm.flags.writeable
    assert surf.dx == surf.heights.dx
    mesh = surf.mesh
    assert mesh is surf.mesh
    assert mesh.is_closed() and mesh.signed_volume() > 0
    lo, hi = mesh.bounding_box()
    assert hi[2] == pytest.approx(1e-6) and lo[2] == pytest.approx(hm.min())
    # The free functions give the same mesh.
    direct = sb.make_rough_surface_mesh(**ROUGH, box_depth=1e-6, box_fine_depth=0.4e-6)
    np.testing.assert_array_equal(direct.vertices, mesh.vertices)
    np.testing.assert_array_equal(direct.triangles, mesh.triangles)
    flat = sb.make_mesh_from_height_map(sb.HeightMap(np.zeros((5, 5)), 1e-7, 1e-7))
    assert flat.is_closed() and flat.signed_volume() == pytest.approx(16e-14 * 2e-6)


@pytest.mark.parametrize(
    "bad",
    [
        dict(sigma=-1e-9),
        dict(Lc=10e-9),  # under-resolved
        dict(box_depth=-1e-6),
        dict(box_mesh_size=0.0),
        dict(box_depth=1e-6, box_fine_depth=2e-6),  # fine band below the bottom plate
    ],
)
def test_rough_surface_invalid(bad):
    with pytest.raises(ValueError):
        sb.make_rough_surface_mesh(**{**ROUGH, **bad})


def test_rough_surface_invalid_exterior_wavelength():
    for bad in (0.0, -500e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sb.make_rough_surface_mesh(**ROUGH, exterior_wavelength=bad)


def test_box_defaults_adr0006_amendment():
    """Closing-box defaults of the ADR 0006 amendment 2026-10-10 (WP-B1)."""
    lam = 500e-9
    si, ag, vac = sb.silicon_500nm(), sb.silver_500nm(), sb.vacuum()
    assert sb.exterior_wavelength(vac, lam) == lam
    assert sb.exterior_wavelength(sb.Material(eps_r=2.25), lam) == pytest.approx(lam / 1.5)
    # Largest 2^M h <= lambda_1 / 5: 100 nm for 50 nm, uniform (h) for 60 nm.
    assert sb.default_box_mesh_size(vac, lam, 50e-9) == 2 * 50e-9
    assert sb.default_box_mesh_size(background=vac, wavelength=lam, mesh_size=60e-9) == 60e-9
    with pytest.raises(ValueError):
        sb.default_box_mesh_size(vac, lam, 0.0)
    # Si: mandatory fine band 3 delta + 3 sigma; Ag: none.
    p_si = sb.rough_surface_box_params(si, vac, lam, 50e-9, 50e-9)
    assert p_si.box_depth == sb.default_box_depth(si, lam)
    assert p_si.box_fine_depth == sb.default_box_fine_depth(si, lam, 50e-9)
    assert p_si.box_fine_depth == pytest.approx(3.54e-6, abs=0.01e-6)
    assert p_si.box_mesh_size == 100e-9 and p_si.exterior_wavelength == lam
    p_ag = sb.rough_surface_box_params(
        object=ag, background=vac, wavelength=lam, sigma=50e-9, mesh_size=50e-9
    )
    assert p_ag.box_depth == 2e-6 and p_ag.box_fine_depth is None
    assert p_ag.kwargs() == dict(
        box_depth=2e-6, box_mesh_size=100e-9, box_fine_depth=None, exterior_wavelength=lam
    )
    assert "RoughBoxParams" in repr(p_ag)
    with pytest.raises(ValueError):
        sb.rough_surface_box_params(sb.Material(eps_r=2.25), vac, lam, 50e-9, 50e-9)
    # The keywords build the mesh; the automatic rule with exterior_wavelength gives the same.
    surf = sb.RoughSurface(**ROUGH, **p_ag.kwargs())
    auto = sb.make_rough_surface_mesh(**ROUGH, box_depth=2e-6, exterior_wavelength=lam)
    np.testing.assert_array_equal(auto.triangles, surf.mesh.triangles)
    old = sb.make_rough_surface_mesh(**ROUGH, box_depth=2e-6)  # 250 nm cells, M = 2
    assert old.num_triangles < surf.mesh.num_triangles
    hm = surf.heights
    from_map = sb.make_mesh_from_height_map(hm, box_depth=2e-6, exterior_wavelength=lam)
    np.testing.assert_array_equal(from_map.triangles, surf.mesh.triangles)


def test_check_beam_waist():
    sb.check_beam_waist(10e-6, 2.5e-6)
    with pytest.raises(ValueError, match="L / 4"):
        sb.check_beam_waist(10e-6, 10e-6 / 3)
    sb.check_beam_waist(patch_length=10e-6, waist=10e-6 / 3, allow_wide=True)
    with pytest.raises(ValueError):
        sb.check_beam_waist(0.0, 1e-6)
