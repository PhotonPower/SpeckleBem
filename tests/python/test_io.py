"""Python bindings of the mesh file I/O and the .npy result writer (WP14b2).

Paths go to C++ as UTF-8 (ADR 0007); every test works under a non-ASCII directory.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import specklebem as sb


@pytest.fixture
def udir(tmp_path):
    d = tmp_path / "sbem_Jürgen_µm_路径_🙂"
    d.mkdir()
    return d


@pytest.mark.parametrize("ext", ["obj", "msh", "stl"])
def test_mesh_round_trip(udir, ext):
    mesh = sb.make_icosphere(0.5e-6, 1, center=(1e-7, 0, 0))
    path = udir / f"kugel_ä.{ext}"
    sb.write_mesh(mesh, path)  # os.PathLike
    back = sb.read_mesh(str(path))  # str
    assert back.num_triangles == 80 and back.is_closed()
    if ext == "stl":  # binary STL stores float32 and renumbers the welded vertices
        np.testing.assert_allclose(corners(back), corners(mesh), rtol=0, atol=1e-13)
    else:
        np.testing.assert_array_equal(back.triangles, mesh.triangles)
        np.testing.assert_array_equal(back.vertices, mesh.vertices)


def corners(mesh):
    return mesh.vertices[mesh.triangles]


def test_format_specific_mesh_io(udir):
    mesh = sb.make_icosphere(1.0, 1)
    for write, read, name in (
        (lambda m, p: sb.write_stl(m, p, ascii=True), sb.read_stl, "a.stl"),
        (sb.write_obj, sb.read_obj, "b.obj"),
        (sb.write_gmsh, sb.read_gmsh, "c.msh"),
    ):
        write(mesh, udir / name)
        np.testing.assert_array_equal(corners(read(udir / name)), corners(mesh))
    assert (udir / "a.stl").read_bytes().startswith(b"solid")


def test_mesh_io_errors(udir):
    with pytest.raises(RuntimeError, match="fehlt"):
        sb.read_mesh(udir / "fehlt.obj")
    with pytest.raises(ValueError):
        sb.read_mesh(udir / "x.ply")
    with pytest.raises(ValueError):
        sb.write_mesh(sb.make_icosphere(1.0, 0), bytes(udir / "x.obj"))
    with pytest.raises(TypeError):
        sb.read_mesh(42)


def test_npy_directory_round_trip(udir):
    out = udir / "ergebnis"
    mesh = sb.make_icosphere(0.5e-6, 1)
    v = np.arange(4) - 0.5j * np.arange(4)
    mr = np.arange(6.0).reshape(2, 3)
    mc = (mr + 1j).T  # Fortran-ordered view
    with sb.open_npy_directory(out) as w:
        assert isinstance(w, sb.ResultWriter)
        w.write_vector("currents", v)
        w.write_vector("real", np.array([1, 2, 3]))
        w.write_matrix("m/real", mr)
        w.write_matrix("m/complex", mc)
        w.write_mesh("mesh", mesh)
        w.write_attribute("name", "Kugel µm")
        w.write_attribute("wavelength", 500e-9)
        for bad in ("../x", "a b", ""):
            with pytest.raises(ValueError):
                w.write_vector(bad, v)
        with pytest.raises(ValueError):
            w.write_matrix("m1", v)
        with pytest.raises(ValueError):
            w.write_attribute("inf", float("inf"))
    load = lambda name: np.load(out / name, allow_pickle=False)  # noqa: E731
    np.testing.assert_array_equal(load("currents.npy"), v)
    r = load("real.npy")
    assert r.dtype == np.complex128 and np.array_equal(r, [1, 2, 3])
    assert load("m/real.npy").dtype == np.float64
    np.testing.assert_array_equal(load("m/real.npy"), mr)
    np.testing.assert_array_equal(load("m/complex.npy"), mc)
    np.testing.assert_array_equal(load("mesh/vertices.npy"), mesh.vertices)
    np.testing.assert_array_equal(load("mesh/triangles.npy"), mesh.triangles)
    attrs = json.loads(Path(out / "attributes.json").read_text(encoding="utf-8"))
    assert attrs == {"name": "Kugel µm", "wavelength": 500e-9}
