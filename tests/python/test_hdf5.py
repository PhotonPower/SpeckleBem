"""HDF5 result files on the Python side (specklebem.io, WP23; lead decision 2026-10-13).

Lossless round trips (bitwise array equality, equal attributes) of a solved sphere (dense,
direct and GMRES), of the open-surface PML route, of fields, far fields (nested attributes
included) and the solver history, the conversion of a C++ ``.npy`` directory (fixture
``npy_dir`` of conftest.py), the one-lock snapshot of the Simulation state, refusals, concurrent
writers and path handling. The tests need h5py and skip without it, unless
``SPECKLEBEM_REQUIRE_H5PY`` is set (as in CI): then a missing h5py fails. ``test_without_h5py``
runs in either case.
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import specklebem as sb
from specklebem import io as sio

RADIUS = 0.5e-6
LAMBDA_FAST = 1e-6
CHEAP = dict(
    quad_degree_far=1,
    quad_degree_near=2,
    quad_degree_sing=2,
    outer_grading_levels=0,
    target_accuracy=0.0,
)
N15 = sb.Material(eps_r=2.25)  # lossless n = 1.5


@pytest.fixture(scope="module")
def h5py():
    if os.environ.get("SPECKLEBEM_REQUIRE_H5PY", "0") not in ("", "0"):
        import h5py as mod  # CI: a missing h5py is a failure, not a skip

        return mod
    return pytest.importorskip("h5py")


def make_sim(**kw):
    kw.setdefault("kernels", CHEAP)
    kw.setdefault("gmres", dict(verbose=False))
    pw = sb.PlaneWave(LAMBDA_FAST, [0, 0, 1], [1, 0.5j, 0])
    return sb.Simulation(sb.make_icosphere(RADIUS, 1), pw, object=N15, **kw)


@pytest.fixture(scope="module")
def direct():
    sim = make_sim(formulation="PMCHWT", solver="direct")
    res = sim.solve()
    return sim, res


def assert_bitwise(a, b):
    """Same dtype, shape and bytes (distinguishes -0.0 from 0.0 and NaN payloads)."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def sphere_fields(sim):
    pts = np.array([[0.0, 0.0, -2e-6], [1.5e-6, 0.3e-6, 0.2e-6], [0.0, 0.1e-6, 3e-6]])
    E, H = sim.field(pts)
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.6, 0.8], [1.0, 0.0, 0.0]])
    return pts, E, H, dirs, sim.far_field(dirs)


def test_sphere_round_trip_lossless(direct, h5py, tmp_path):
    sim, res = direct
    pts, E, H, dirs, F = sphere_fields(sim)
    meta = {
        "operator": "Jürgen µm 路径",
        "seed": 42,
        "angle": 0.1,
        "eps": 2.25 - 0.5j,
        "flag": True,
        "grid": np.arange(6.0).reshape(2, 3),
        "labels": ["a", "é"],
        "study": {"name": "sphere", "level": 2},
    }
    path = tmp_path / "sphère_µm_路径.h5"  # non-ASCII file name (ADR 0007 spirit)
    out = sim.save(
        path,
        fields={
            "scattered": (pts, E, H),
            "e_only": {
                "points": pts,
                "E": E,
                "attrs": {"k": 1, "sub": {"x": 2, "deep": {"y": "z"}}, "w": 0.5},
            },
        },
        far_fields={"forward": (dirs, F)},
        metadata=meta,
    )
    assert out == path.absolute() and path.is_file()
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temporary file left

    r = sio.load_hdf5(path)
    # Root attributes.
    assert r.attrs["format"] == "specklebem-hdf5" and r.attrs["format_version"] == 1
    assert r.attrs["library_version"] == sb.__version__
    assert r.attrs["units"] == "SI" and r.attrs["time_convention"] == "exp(+jwt)"
    created = datetime.datetime.fromisoformat(r.attrs["created_utc"])
    assert created.utcoffset() == datetime.timedelta(0)
    assert r.attrs["material_convention"] == (
        "eps_r, mu_r relative; passive media Im <= 0 (exp(+jwt))"
    )
    # Mesh, currents, right-hand side: bitwise.
    mesh = sim.mesh
    assert_bitwise(r.mesh["vertices"], mesh.vertices)
    assert_bitwise(r.mesh["triangles"], mesh.triangles)
    assert_bitwise(r.mesh["edges"], mesh.edges)
    assert "zones" not in r.mesh
    assert_bitwise(r.currents, sim.currents)
    assert_bitwise(r.rhs, sim.rhs())
    assert r.N == mesh.num_edges
    assert_bitwise(np.concatenate([r.J, r.M]), sim.currents)
    # Fields and far fields; nested attrs of an entry come back as nested dicts.
    for key, ref in (("points", pts), ("E", E), ("H", H)):
        assert_bitwise(r.fields["scattered"][key], ref)
    assert "H" not in r.fields["e_only"]
    assert r.fields["e_only"]["attrs"] == {"k": 1, "sub": {"x": 2, "deep": {"y": "z"}}, "w": 0.5}
    assert list(r.fields["e_only"]["attrs"]) == ["k", "w", "sub"]  # attributes, then groups
    assert r.fields["scattered"]["attrs"] == {}
    assert_bitwise(r.far_fields["forward"]["directions"], dirs)
    assert_bitwise(r.far_fields["forward"]["F"], F)
    # Solver statistics and history (direct: 0 iterations, one residual).
    s = r.solver
    assert s["formulation"] == "PMCHWT" and s["solver"] == "direct"
    assert s["preconditioner"] == sim.preconditioner
    assert s["iterations"] == 0 and s["converged"] is True
    assert s["true_residual"] == res.true_relative_residual
    assert s["residual"] == res.residual_history[-1]
    assert s["wall_seconds"] == res.wall_seconds
    np.testing.assert_array_equal(r.history["residual"], res.residual_history)
    np.testing.assert_array_equal(r.history["iteration"], [0])
    # Configuration: every value of sim.config, materials as complex.
    cfg = sim.config
    for key in (
        "wavelength",
        "formulation",
        "formulation_automatic",
        "preconditioner",
        "preconditioner_automatic",
        "solver",
        "compression",
    ):
        assert r.config[key] == cfg[key], key
    for key in ("gmres", "kernels", "mlfmm"):
        assert r.config[key] == cfg[key], key
    assert r.config["termination"] == "box" and "pml" not in r.config
    assert r.config["num_unknowns"] == sim.num_unknowns
    assert r.config["object"] == {"eps_r": 2.25 + 0j, "mu_r": 1 + 0j}
    assert r.config["exterior"] == {"eps_r": 1 + 0j, "mu_r": 1 + 0j}
    # Excitation.
    x = r.excitation
    assert x["type"] == "PlaneWave" and x["wavelength"] == LAMBDA_FAST
    assert x["omega"] == sim.excitation.omega
    np.testing.assert_array_equal(x["direction"], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(x["polarization"], [1, 0.5j, 0])
    assert x["background"] == {"eps_r": 1 + 0j, "mu_r": 1 + 0j}
    # Metadata.
    m = r.metadata
    assert m["operator"] == meta["operator"] and m["seed"] == 42 and m["angle"] == 0.1
    assert m["eps"] == meta["eps"] and m["flag"] is True and m["labels"] == ["a", "é"]
    np.testing.assert_array_equal(m["grid"], meta["grid"])
    assert m["study"] == {"name": "sphere", "level": 2}
    assert list(m)[:4] == ["operator", "seed", "angle", "eps"]  # creation order kept

    # Stored types as seen by plain h5py (self-describing without SpeckleBem).
    with h5py.File(path, "r") as f:
        assert f["currents"].dtype == np.complex128 and f["currents"].attrs["N"] == r.N
        assert f["mesh/vertices"].dtype == np.float64
        assert f["mesh/triangles"].dtype == np.int64
        assert f["fields/scattered/E"].dtype == np.complex128
        assert f["config/object"].attrs["eps_r"] == 2.25
        assert "x = [J; M]" in f["currents"].attrs["layout"]
        assert "far_field" not in f and "far_fields/forward/F" in f
        assert "k1 the exterior (R1) wavenumber" in f["far_fields/forward/F"].attrs["description"]
        assert "T+" in f["mesh/edges"].attrs["description"]
        assert "a -> b" in f["mesh/edges"].attrs["description"]
        assert f["config/mlfmm"].attrs["max_exact_far_bytes"].dtype == np.uint64
        assert f["config/mlfmm"].attrs["max_exact_far_bytes"] == cfg["mlfmm"]["max_exact_far_bytes"]
        ring_budget = f["config/mlfmm"].attrs["pml_ring_max_far_bytes"]
        assert ring_budget.dtype == np.uint64
        assert ring_budget == cfg["mlfmm"]["pml_ring_max_far_bytes"]
        assert isinstance(f["fields/e_only/sub/deep"], h5py.Group)


def test_gmres_history_and_per_call_parameters(h5py, tmp_path):
    sim = make_sim(formulation="ICTF", preconditioner="none")
    assert sim.last_result is None and sim.last_solve_params is None
    rec = sio.HistoryRecorder()
    res = sim.solve(tol=1e-8, callback=rec)
    assert res.converged and res.iterations > 3
    assert sim.last_solve_params["tol"] == 1e-8
    np.testing.assert_array_equal(sim.last_result.residual_history, res.residual_history)

    # Default: residual history of the last solve, iterations 0..k, no times.
    r = sio.load_hdf5(sim.save(tmp_path / "default.h5"))
    np.testing.assert_array_equal(r.history["residual"], res.residual_history)
    np.testing.assert_array_equal(r.history["iteration"], np.arange(res.iterations + 1))
    assert "time" not in r.history
    s = r.solver
    assert s["tolerance"] == 1e-8  # the per-call override, not the gmres dict's 1e-3
    assert s["max_iter"] == 2000 and s["restart"] == 0 and s["side"] == "left"
    assert s["orthogonalization"] == "cgs2" and s["solver"] == "gmres"
    assert s["iterations"] == res.iterations and s["converged"] is True
    assert s["residual"] == res.residual_history[-1]
    assert s["true_residual"] == res.true_relative_residual
    assert s["reorthogonalizations"] == res.reorthogonalizations

    # Recorder: iterations 1..k with wall times.
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "recorder.h5", sim, history=rec))
    np.testing.assert_array_equal(r.history["iteration"], np.arange(1, res.iterations + 1))
    np.testing.assert_array_equal(r.history["residual"], rec.residual)
    np.testing.assert_array_equal(r.history["residual"], res.residual_history[1:])
    t = r.history["time"]
    assert t.shape == (res.iterations,) and np.all(np.diff(t) >= 0) and t[0] >= 0
    with h5py.File(tmp_path / "recorder.h5", "r") as f:
        assert f["solver/history/time"].attrs["units"] == "s"

    # A mapping and an explicit SolveResult are accepted as well.
    hist = {"residual": [1.0, 0.5, 0.25], "time": [0.0, 0.1, 0.2]}
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "map.h5", sim, history=hist))
    np.testing.assert_array_equal(r.history["iteration"], [0, 1, 2])
    np.testing.assert_array_equal(r.history["time"], hist["time"])
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "res.h5", sim, result=res))
    assert r.solver["iterations"] == res.iterations
    assert r.solver["tolerance"] == 1e-8  # res is the last solve: its parameters are known

    # An explicit result that is not the last solve: its statistics, but no parameters.
    res2 = sim.solve(tol=1e-4)
    assert res2.iterations < res.iterations
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "older.h5", sim, result=res))
    assert r.solver["iterations"] == res.iterations
    assert r.solver["true_residual"] == res.true_relative_residual
    for key in ("tolerance", "max_iter", "restart", "side", "orthogonalization"):
        assert key not in r.solver, key
    np.testing.assert_array_equal(r.history["residual"], res.residual_history)
    assert_bitwise(r.currents, sim.currents)  # the currents are those of the last solve
    r = sio.load_hdf5(sim.save(tmp_path / "latest.h5"))
    assert r.solver["tolerance"] == 1e-4 and r.solver["iterations"] == res2.iterations
    with pytest.raises(ValueError, match="history"):
        sio.save_hdf5(tmp_path / "bad.h5", sim, history={"residual": [1.0], "time": [0, 1]})
    with pytest.raises(ValueError, match="unknown keys"):
        sio.save_hdf5(tmp_path / "bad.h5", sim, history={"residual": [1.0], "x": [1]})
    assert not (tmp_path / "bad.h5").exists()


def test_unsolved_simulation(h5py, tmp_path):
    sim = make_sim()
    assert not sim.assembled and not sim.solved
    snap = sim._save_snapshot()
    assert snap == {"result": None, "params": None, "currents": None, "rhs": None}
    r = sio.load_hdf5(sim.save(tmp_path / "unsolved.h5"))
    assert r.currents is None and r.rhs is None and r.history == {}
    assert "iterations" not in r.solver and r.solver["formulation"] == sim.formulation.name
    assert "tolerance" not in r.solver  # no solve: the configured values are in /config/gmres
    assert r.config["gmres"]["tol"] == sim.config["gmres"]["tol"]
    assert_bitwise(r.mesh["vertices"], sim.mesh.vertices)
    with pytest.raises(ValueError, match="no currents"):
        _ = r.N
    sim.assemble()
    assert sim.assembled and not sim.solved
    r = sio.load_hdf5(sim.save(tmp_path / "assembled.h5"))
    assert r.currents is None
    assert_bitwise(r.rhs, sim.rhs())
    res = sim.solve()
    assert sim.solved
    snap = sim._save_snapshot()
    assert snap["result"].iterations == res.iterations and snap["params"] == sim.last_solve_params
    assert_bitwise(snap["currents"], sim.currents)
    assert_bitwise(snap["rhs"], sim.rhs())


def test_save_errors_are_not_swallowed(h5py, tmp_path):
    """A RuntimeError of the Simulation (here: a re-entrant call from the solve callback)
    propagates instead of silently writing a file without currents."""
    sim = make_sim(formulation="ICTF", preconditioner="none")
    errors = []

    def callback(iteration, residual):
        if iteration == 1:
            try:
                sim.save(tmp_path / "inside.h5")
            except RuntimeError as e:
                errors.append(str(e))

    sim.solve(callback=callback)
    assert len(errors) == 1 and "re-entrant" in errors[0]
    assert list(tmp_path.iterdir()) == []


def test_pml_round_trip(h5py, tmp_path):
    h = 100e-9
    edge = 0.4e-6
    cells = int(round(edge / h))
    hm = sb.HeightMap(np.zeros((cells + 1, cells + 1)), edge / cells, edge / cells)
    om = sb.make_open_surface_mesh(
        hm,
        sb.OpenSurfaceParams(
            patch_size=0.2e-6, taper_width=0.1e-6, gap_width=0.1e-6, pml_width=0.1e-6
        ),
    )
    beam = sb.AngularSpectrumBeam(500e-9, 0.4e-6, "p", region_radius=0.6e-6)
    term = sb.PmlTermination(sigma_max=1.0, dark_band_tolerance=0.9)
    sim = sb.Simulation.with_pml(
        om,
        beam,
        pml=term,
        object=N15,
        solver="direct",
        kernels=dict(
            outer_grading_levels=0,
            target_accuracy=0.0,
            quad_degree_near=5,
            stretched_touching_points=4,
        ),
    )
    res = sim.solve()
    dirs = np.array([[0.0, 0.0, -1.0], [0.5, 0.0, -0.8]])
    F = sim.far_field(dirs)
    r = sio.load_hdf5(sim.save(tmp_path / "pml.h5", far_fields={"windowed": (dirs, F)}))

    assert_bitwise(r.mesh["zones"], np.asarray(sim.zones, dtype=np.int64))
    np.testing.assert_array_equal(r.mesh["zones"], om.zones)
    assert_bitwise(r.mesh["vertices"], om.mesh.vertices)
    assert_bitwise(r.mesh["triangles"], sim.mesh.triangles)
    assert_bitwise(r.currents, sim.currents)
    assert_bitwise(r.rhs, sim.rhs())
    assert_bitwise(r.far_fields["windowed"]["F"], F)
    assert r.solver["true_residual"] == res.true_relative_residual
    c = r.config
    assert c["termination"] == "pml" and c["kernels"]["stretched_touching_points"] == 4
    p = c["pml"]
    assert p["sigma_max"] == 1.0 and p["order"] == 2.0 and p["correlation_length"] == 0.0
    assert p["dark_band_tolerance"] == 0.9 and p["dark_band_ratio"] == sim.dark_band_ratio
    assert (p["patch_half"], p["taper_end"], p["pml_start"], p["outer_half"]) == (
        sim.zone_half_widths
    )
    w = sim.far_field_window
    assert (p["window_inner_half"], p["window_outer_half"]) == (
        w.inner_half,
        w.outer_half,
    )
    x = r.excitation
    assert x["type"] == "AngularSpectrumBeam" and x["polarization"] == "P"
    assert x["waist"] == 0.4e-6 and x["incidence_angle"] == 0.0
    assert x["region_radius"] == 0.6e-6 and x["controlled_radius"] == 0.6e-6
    assert x["num_plane_waves"] == beam.num_plane_waves and x["power"] == beam.power
    np.testing.assert_array_equal(x["focus"], [0.0, 0.0, 0.0])


def test_gaussian_beam_attrs_and_mesh_only(h5py, tmp_path):
    gb = sb.GaussianBeam(LAMBDA_FAST, 2e-6, "s", incidence_angle=0.2, focus=(0, 0, 1e-7))
    sim = sb.Simulation(sb.make_icosphere(RADIUS, 1), gb, object=N15, kernels=CHEAP)
    x = sio.load_hdf5(sim.save(tmp_path / "gb.h5")).excitation
    assert x["type"] == "GaussianBeam" and x["polarization"] == "S"
    assert x["waist"] == 2e-6 and x["incidence_angle"] == 0.2
    np.testing.assert_array_equal(x["focus"], [0, 0, 1e-7])
    assert x["controlled_radius"] == np.inf

    # Without a Simulation: an explicit mesh and fields only.
    mesh = sb.make_icosphere(RADIUS, 0)
    pts = np.zeros((2, 3))
    E = np.ones((2, 3), complex)
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "mesh.h5", mesh=mesh, fields={"f": (pts, E)}))
    np.testing.assert_array_equal(r.mesh["triangles"], mesh.triangles)
    np.testing.assert_array_equal(r.fields["f"]["E"], E)
    assert r.config == {} and r.solver == {} and r.currents is None
    r = sio.load_hdf5(sio.save_hdf5(tmp_path / "tuple.h5", mesh=(mesh.vertices, mesh.triangles)))
    assert "edges" not in r.mesh
    np.testing.assert_array_equal(r.mesh["vertices"], mesh.vertices)


def test_input_errors(h5py, direct, tmp_path):
    sim, _ = direct
    target = tmp_path / "x.h5"
    with pytest.raises(ValueError, match="either sim or mesh"):
        sio.save_hdf5(target, sim, mesh=sim.mesh)
    with pytest.raises(ValueError, match="without '/'"):
        sio.save_hdf5(target, sim, far_fields={"a/b": (np.zeros((1, 3)), np.zeros((1, 3)))})
    with pytest.raises(ValueError, match="row counts"):
        sio.save_hdf5(target, sim, fields={"f": (np.zeros((2, 3)), np.zeros((1, 3)))})
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        sio.save_hdf5(target, sim, far_fields={"f": (np.zeros((2, 2)), np.zeros((2, 2)))})
    with pytest.raises(TypeError, match="NoneType"):
        sio.save_hdf5(target, sim, metadata={"x": None})
    # Complex values for real datasets are refused, never silently made real.
    pts = np.zeros((2, 3), complex)
    pts[0, 0] = 1j
    with pytest.raises(ValueError, match="imaginary parts"):
        sio.save_hdf5(target, sim, fields={"f": (pts, np.zeros((2, 3), complex))})
    with pytest.raises(ValueError, match="imaginary parts"):
        sio.save_hdf5(target, sim, far_fields={"f": (pts, np.zeros((2, 3), complex))})
    mesh = sb.make_icosphere(RADIUS, 0)
    with pytest.raises(ValueError, match="mesh/vertices.*imaginary parts"):
        sio.save_hdf5(target, mesh=(mesh.vertices + 0j, mesh.triangles))
    with pytest.raises(TypeError):
        sio.save_hdf5(b"bytes.h5", sim)
    assert list(tmp_path.iterdir()) == []  # failed writes leave nothing behind


def test_str_subclasses(h5py, tmp_path):
    """np.str_ values (e.g. from a NumPy string array) are stored as plain str."""
    names = np.array(["alpha", "béta"])
    meta = {"name": names[1], "list": [names[0], "x"], "arr": names, "nested": {"s": names[0]}}
    r = sio.load_hdf5(
        sio.save_hdf5(tmp_path / "str.h5", mesh=sb.make_icosphere(RADIUS, 0), metadata=meta)
    )
    assert r.metadata == {
        "name": "béta",
        "list": ["alpha", "x"],
        "arr": ["alpha", "béta"],
        "nested": {"s": "alpha"},
    }
    assert type(r.metadata["name"]) is str


def test_concurrent_writers(h5py, direct, tmp_path):
    """Threads writing the same target use distinct temporary files (tempfile.mkstemp)."""
    sim, _ = direct
    path = tmp_path / "shared.h5"
    errors = []

    def write(run):
        try:
            sim.save(path, metadata={"run": run}, overwrite=True)
        except BaseException as e:  # reported below
            errors.append(e)

    threads = [threading.Thread(target=write, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    r = sio.load_hdf5(path)
    assert r.metadata["run"] in range(4)
    assert_bitwise(r.currents, sim.currents)
    assert [p.name for p in tmp_path.iterdir()] == ["shared.h5"]  # no temporary file left
    if os.name != "nt":  # the usual new-file mode, not mkstemp's 0o600
        mask = os.umask(0)
        os.umask(mask)
        assert path.stat().st_mode & 0o777 == 0o666 & ~mask


def test_overwrite_protection(h5py, direct, tmp_path):
    sim, _ = direct
    path = tmp_path / "run.h5"
    sim.save(path, metadata={"run": 1})
    with pytest.raises(FileExistsError, match="overwrite=True"):
        sim.save(path, metadata={"run": 2})
    assert sio.load_hdf5(path).metadata == {"run": 1}  # untouched
    sim.save(str(path), metadata={"run": 2}, overwrite=True)
    assert sio.load_hdf5(path).metadata == {"run": 2}
    with pytest.raises(IsADirectoryError):
        sim.save(tmp_path, overwrite=True)
    assert [p.name for p in tmp_path.iterdir()] == ["run.h5"]


def test_refuses_foreign_or_newer_files(h5py, tmp_path):
    path = tmp_path / "foreign.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "other"
        f.attrs["format_version"] = 1
    with pytest.raises(ValueError, match="not a SpeckleBem HDF5 file"):
        sio.load_hdf5(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1.0])
    with pytest.raises(ValueError, match="not a SpeckleBem HDF5 file"):
        sio.load_hdf5(path)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "specklebem-hdf5"
        f.attrs["format_version"] = sio.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        sio.load_hdf5(path)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "specklebem-hdf5"
        f.attrs["format_version"] = "1"
    with pytest.raises(ValueError, match="invalid format_version"):
        sio.load_hdf5(path)


def test_npy_to_hdf5_lossless(h5py, npy_dir, tmp_path):
    out = tmp_path / "converted_µm.h5"
    assert sio.npy_to_hdf5(npy_dir, out) == out.absolute()
    files = sorted(npy_dir.rglob("*.npy"))
    assert len(files) == 6  # vector, empty, matrix_real, group/matrix_complex, mesh/{v, t}
    with h5py.File(out, "r") as f:
        assert f.attrs["format"] == "specklebem-hdf5" and f.attrs["source"] == "npy"
        for file in files:
            name = file.relative_to(npy_dir).with_suffix("").as_posix()
            ref = np.load(file, allow_pickle=False)
            ds = f[name]
            assert ds.dtype == ref.dtype and ds.shape == ref.shape, name
            np.testing.assert_array_equal(ds[()], ref)
    with open(npy_dir / "attributes.json", encoding="utf-8") as fh:
        attrs = json.load(fh)
    r = sio.load_hdf5(out)
    assert r.metadata == attrs and list(r.metadata) == list(attrs)  # values and order
    np.testing.assert_array_equal(r.mesh["vertices"], np.load(npy_dir / "mesh" / "vertices.npy"))
    np.testing.assert_array_equal(
        r.datasets["/group/matrix_complex"],
        np.load(npy_dir / "group" / "matrix_complex.npy"),
    )
    assert r.datasets["/empty"].shape == (0,)
    with pytest.raises(FileExistsError):
        sio.npy_to_hdf5(npy_dir, out)

    # Leftover temporary files are skipped with a warning; unknown files are refused.
    copy = tmp_path / "copy"
    shutil.copytree(npy_dir, copy)
    (copy / "vector.npy~tmp").write_bytes(b"partial")
    with pytest.warns(UserWarning, match="temporary"):
        r = sio.load_hdf5(sio.npy_to_hdf5(copy, tmp_path / "copy.h5"))
    assert "/vector" in r.datasets and len(r.datasets) == 6
    (copy / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="unexpected file"):
        sio.npy_to_hdf5(copy, tmp_path / "copy2.h5")
    with pytest.raises(NotADirectoryError):
        sio.npy_to_hdf5(copy / "notes.txt", tmp_path / "copy3.h5")
    assert not (tmp_path / "copy2.h5").exists()


def test_without_h5py(tmp_path):
    """Importing specklebem never needs h5py; the I/O functions raise ImportError with a hint."""
    code = (
        "import sys\n"
        "sys.modules['h5py'] = None\n"  # makes 'import h5py' raise ImportError
        "import specklebem as sb\n"
        "assert hasattr(sb.Simulation, 'save')\n"
        "for call in (lambda: sb.io.load_hdf5('x.h5'), lambda: sb.io.save_hdf5('x.h5'),\n"
        "             lambda: sb.io.npy_to_hdf5('.', 'x.h5')):\n"
        "    try:\n"
        "        call()\n"
        "    except ImportError as e:\n"
        "        assert 'pip install specklebem[io]' in str(e), e\n"
        "    else:\n"
        "        raise AssertionError('no ImportError')\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
    assert list(Path(tmp_path).iterdir()) == []
