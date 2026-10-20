"""HDF5 result files of SpeckleBem (WP23), written and read on the Python side with h5py.

Lead decision 2026-10-13: the C++ library has no HDF5 dependency and writes ``.npy``
directories (``open_npy_directory``); HDF5 is produced here. h5py is optional: importing
``specklebem`` never needs it, and the functions of this module raise ``ImportError`` with an
install hint (``pip install specklebem[io]``) when it is missing.

File layout (format ``specklebem-hdf5``, ``format_version`` 1; SI units, exp(+jwt)):

====================================  =========================================================
``/`` attrs                           format, format_version, library_version, units ("SI"),
                                      time_convention ("exp(+jwt)"), normal_convention,
                                      material_convention, created_utc (ISO 8601)
``/mesh/vertices``                    (V, 3) float64 [m]
``/mesh/triangles``                   (F, 3) int64, 0-based, counter-clockwise seen from R1
``/mesh/edges``                       (N, 2) int64 interior edges (a, b), a < b; edge n carries
                                      RWG basis n, whose plus triangle T+ holds the edge as
                                      a -> b counter-clockwise (T-: b -> a)
``/mesh/zones``                       (F,) int64 SurfaceZone per triangle (PML route only)
``/currents``                         (2N,) complex128 x = [J; M]; attrs N, layout, units
``/rhs``                              (2N,) complex128 right-hand side (if assembled)
``/fields/<name>/points``, ``E``,     (n, 3) float64 [m], (n, 3) complex128 [V/m], [A/m];
``H``                                 attrs of the entry (nested dicts are sub-groups)
``/far_fields/<name>/directions``,    (n, 3) float64, (n, 3) complex128 [V]
``F``                                 (E_s ~ F exp(-j k1 r) / r, k1 the exterior wavenumber)
``/solver`` attrs                     formulation, preconditioner, solver, tolerance, max_iter,
                                      restart, side, orthogonalization, iterations, converged,
                                      residual, true_residual, wall_seconds, ...
``/solver/history/iteration``,        (k,) int64, (k,) float64, (k,) float64 [s] (if recorded)
``residual``, ``time``
``/config`` attrs and sub-groups      wavelength, formulation, preconditioner, solver,
                                      compression, termination ("box" / "pml"), num_unknowns;
                                      ``object`` / ``exterior`` (eps_r, mu_r complex),
                                      ``gmres``, ``kernels``, ``mlfmm``, ``pml`` (PML route)
``/excitation`` attrs                 type, wavelength, omega, background eps_r / mu_r and the
                                      type's parameters (direction, polarization, waist, ...)
``/metadata`` attrs and sub-groups    user metadata (``metadata=``; ``attributes.json`` of a
                                      converted ``.npy`` directory)
====================================  =========================================================

Strings are UTF-8, complex attributes are HDF5 compound (r, i) types (read back by h5py as
complex). Groups and attributes keep their creation order; a mapping read back lists its
attributes first, then its sub-groups (HDF5 keeps the two apart).

Writes go to a unique temporary file in the target's directory (``tempfile.mkstemp``) that is
renamed over the target when complete (``os.replace``), so concurrent writers never share a
temporary file and a target is never left half-written.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "specklebem-hdf5"
"""Value of the root attribute ``format``."""
FORMAT_VERSION = 1
"""Layout version written by this module; files with a larger version are refused."""

_INSTALL_HINT = (
    "h5py is required for SpeckleBem HDF5 files: pip install specklebem[io] "
    "(or install h5py, e.g. python3-h5py / mingw-w64-ucrt-x86_64-python-h5py)"
)
_ZONE_NAMES = "0 Patch, 1 Taper, 2 Gap, 3 Pml (SurfaceZone)"
_EDGES_DESCRIPTION = (
    "interior edges (a, b), a < b; edge n carries RWG basis n. Its plus triangle T+ is the "
    "adjacent triangle in which the edge appears as a -> b in the counter-clockwise vertex "
    "order (v0 v1 v2 v0), the minus triangle T- the one in which it appears as b -> a"
)
_FAR_FIELD_DESCRIPTION = (
    "far-field amplitude F [V]: E_s(r) ~ F exp(-j k1 r) / r for r -> infinity along the "
    "direction, k1 the exterior (R1) wavenumber; time convention exp(+jwt)"
)
_MATERIAL_CONVENTION = "eps_r, mu_r relative; passive media Im <= 0 (exp(+jwt))"


def _default_file_mode() -> int:
    """0o666 minus the process umask: the mode of a newly created file (read once at import)."""
    if os.name == "nt":
        return 0o666
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


_NEW_FILE_MODE = _default_file_mode()

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "HistoryRecorder",
    "Result",
    "load_hdf5",
    "npy_to_hdf5",
    "save_hdf5",
]


def _h5py() -> Any:
    """The h5py module; ImportError with the install hint when it is missing."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ImportError(_INSTALL_HINT) from exc
    return h5py


class HistoryRecorder:
    """GMRES progress callback that records iteration, residual and elapsed time.

    Pass it as ``sim.solve(callback=recorder)`` and then to ``save_hdf5(..., history=recorder)``
    to store per-iteration wall times in ``/solver/history/time``. Times are seconds since the
    recorder was created or last ``reset()``; iterations start at 1 (the callback is not called
    for the initial residual, nor by the direct solver).
    """

    def __init__(self) -> None:
        self.iteration: list[int] = []
        self.residual: list[float] = []
        self.time: list[float] = []
        self._start = time.perf_counter()

    def reset(self) -> None:
        """Clear the records and restart the clock (call it right before ``solve``)."""
        self.iteration.clear()
        self.residual.clear()
        self.time.clear()
        self._start = time.perf_counter()

    def __call__(self, iteration: int, residual: float) -> None:
        self.iteration.append(int(iteration))
        self.residual.append(float(residual))
        self.time.append(time.perf_counter() - self._start)


@dataclass
class Result:
    """Content of a SpeckleBem HDF5 file (``load_hdf5``); arrays are NumPy copies.

    Attributes are Python scalars (``float``, ``int``, ``complex``, ``bool``, ``str``) or
    NumPy arrays; groups with attributes and sub-groups are nested dicts.
    """

    attrs: dict[str, Any] = field(default_factory=dict)
    """Root attributes (format, format_version, library_version, units, ...)."""
    mesh: dict[str, np.ndarray] = field(default_factory=dict)
    """``vertices``, ``triangles`` and, if present, ``edges`` and ``zones``."""
    currents: np.ndarray | None = None
    """x = [J; M] (2N,) complex128, or None."""
    rhs: np.ndarray | None = None
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    """name -> {"points", "E", "H" (if stored), "attrs"}."""
    far_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    """name -> {"directions", "F", "attrs"}."""
    solver: dict[str, Any] = field(default_factory=dict)
    """Attributes of ``/solver``."""
    history: dict[str, np.ndarray] = field(default_factory=dict)
    """``iteration``, ``residual`` and, if recorded, ``time``."""
    config: dict[str, Any] = field(default_factory=dict)
    excitation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, np.ndarray] = field(default_factory=dict)
    """Every dataset of the file by its absolute path (e.g. ``"/mesh/vertices"``)."""

    @property
    def N(self) -> int:
        """Number of RWG functions N (half the length of ``currents``)."""
        if self.currents is None:
            raise ValueError("Result: the file holds no currents")
        return int(self.currents.shape[0] // 2)

    @property
    def J(self) -> np.ndarray:
        """Electric surface current coefficients J [A/m] (first half of ``currents``)."""
        return self.currents[: self.N]  # type: ignore[index]

    @property
    def M(self) -> np.ndarray:
        """Magnetic surface current coefficients M [V/m] (second half of ``currents``)."""
        return self.currents[self.N :]  # type: ignore[index]


# --------------------------------------------------------------------------------------------
# Writing helpers
# --------------------------------------------------------------------------------------------


def _fspath(path: str | os.PathLike[str], what: str = "path") -> Path:
    p = os.fspath(path)
    if not isinstance(p, str):
        raise TypeError(f"{what}: expected a str or os.PathLike of str (bytes not supported)")
    return Path(p)


def _check_name(name: object, what: str) -> str:
    if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
        raise ValueError(f"{what}: names must be non-empty str without '/' (got {name!r})")
    return name


def _attr_value(value: Any, key: str) -> Any:
    """Value storable as an HDF5 attribute (lossless); TypeError otherwise."""
    if isinstance(value, (bool, np.bool_)):
        return np.bool_(value)
    if isinstance(value, str):  # also str subclasses such as np.str_ (not storable as such)
        return str(value)
    if isinstance(value, (int, float, complex, np.generic)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.dtype.kind in "biufc":
            return arr
        if arr.dtype.kind == "U" and arr.ndim == 1:
            return [str(s) for s in arr]
    raise TypeError(
        f"attribute {key!r}: cannot store {type(value).__name__} in HDF5 (use str, bool, int, "
        "float, complex, a numeric array, a list of str or a dict for a sub-group)"
    )


def _write_attrs(h5py: Any, group: Any, values: Mapping[str, Any]) -> None:
    """Attributes from a mapping; dict values become sub-groups (recursively)."""
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"attribute names must be non-empty str (got {key!r})")
        if isinstance(value, Mapping):
            _write_attrs(
                h5py, group.create_group(_check_name(key, "group"), track_order=True), value
            )
            continue
        v = _attr_value(value, key)
        if isinstance(v, list):  # list of str
            group.attrs.create(key, v, dtype=h5py.string_dtype("utf-8"))
        else:
            group.attrs[key] = v


def _dataset(
    group: Any, name: str, data: Any, dtype: str, ndim: int | None = None, cols: int | None = None
) -> Any:
    target = np.dtype(dtype)
    src = np.asarray(data)
    if np.iscomplexobj(src) and target.kind != "c":
        raise ValueError(
            f"{group.name.rstrip('/')}/{name}: complex values cannot be stored as real {target} "
            "(the imaginary parts would be lost)"
        )
    arr = np.ascontiguousarray(src, dtype=target)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{group.name}/{name}: expected {ndim} dimensions, got shape {arr.shape}")
    if cols is not None and arr.shape[-1] != cols:
        raise ValueError(f"{group.name}/{name}: expected shape (n, {cols}), got {arr.shape}")
    return group.create_dataset(name, data=arr)


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if hasattr(mesh, "vertices") and hasattr(mesh, "triangles"):
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.edges)
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        return np.asarray(mesh[0]), np.asarray(mesh[1]), None
    raise TypeError("mesh: expected a TriangleMesh or a (vertices, triangles) tuple")


def _write_mesh(group: Any, mesh: Any, zones: np.ndarray | None) -> None:
    v, t, e = _mesh_arrays(mesh)
    if t.dtype.kind not in "iu":
        raise ValueError("mesh: triangles must be integers")
    _dataset(group, "vertices", v, "<f8", 2, 3).attrs["units"] = "m"
    tri = _dataset(group, "triangles", t, "<i8", 2, 3)
    tri.attrs["description"] = "0-based vertex indices, counter-clockwise seen from R1"
    if e is not None:
        ed = _dataset(group, "edges", e, "<i8", 2, 2)
        ed.attrs["description"] = _EDGES_DESCRIPTION
    if zones is not None:
        _dataset(group, "zones", zones, "<i8", 1).attrs["values"] = _ZONE_NAMES


def _material_attrs(m: Any) -> dict[str, complex]:
    return {"eps_r": complex(m.eps_r), "mu_r": complex(m.mu_r)}


def _excitation_attrs(exc: Any) -> dict[str, Any]:
    from . import _specklebem as core

    a: dict[str, Any] = {
        "type": type(exc).__name__,
        "wavelength": float(exc.wavelength),
        "omega": float(exc.omega),
        "background": _material_attrs(exc.background),
        "controlled_radius": float(exc.controlled_radius),
        "controlled_center": np.asarray(exc.controlled_center, dtype=float),
    }
    if isinstance(exc, core.PlaneWave):
        a["direction"] = np.asarray(exc.direction, dtype=float)
        a["polarization"] = np.asarray(exc.polarization, dtype=complex)
    elif isinstance(exc, (core.GaussianBeam, core.AngularSpectrumBeam)):
        a["waist"] = float(exc.waist)
        a["polarization"] = exc.polarization.name
        a["incidence_angle"] = float(exc.incidence_angle)
        a["focus"] = np.asarray(exc.focus, dtype=float)
        if isinstance(exc, core.AngularSpectrumBeam):
            for key in ("tolerance", "region_radius", "grid_change", "max_polar_angle", "power"):
                a[key] = float(getattr(exc, key))
            for key in ("num_plane_waves", "polar_order", "azimuth_order"):
                a[key] = int(getattr(exc, key))
    return a


def _config_attrs(sim: Any) -> dict[str, Any]:
    cfg = dict(sim.config)
    nested = {k: cfg.pop(k) for k in ("gmres", "kernels", "mlfmm")}
    a: dict[str, Any] = dict(cfg)
    a["termination"] = "pml" if sim.has_pml else "box"
    a["num_unknowns"] = int(sim.num_unknowns)
    a["object"] = _material_attrs(sim.object)
    a["exterior"] = _material_attrs(sim.exterior)
    a.update(nested)
    a["mlfmm"] = dict(a["mlfmm"])
    for key in ("max_exact_far_bytes", "pml_ring_max_far_bytes"):  # size_t
        a["mlfmm"][key] = np.uint64(a["mlfmm"][key])
    if sim.has_pml:
        t = sim.pml_termination
        half = sim.zone_half_widths
        w = sim.far_field_window
        a["pml"] = {
            "sigma_max": float(t.sigma_max),
            "order": float(t.order),
            "correlation_length": float(t.correlation_length),
            "dark_band_tolerance": float(t.dark_band_tolerance),
            "patch_half": float(half[0]),
            "taper_end": float(half[1]),
            "pml_start": float(half[2]),
            "outer_half": float(half[3]),
            "dark_band_ratio": float(sim.dark_band_ratio),
            "window_inner_half": float(w.inner_half),
            "window_outer_half": float(w.outer_half),
        }
    return a


def _solver_attrs(sim: Any, result: Any, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """params: the GMRES parameters of the stored result (None = unknown: not written)."""
    a: dict[str, Any] = {}
    if sim is not None:
        cfg = sim.config
        a["formulation"] = cfg["formulation"]
        a["preconditioner"] = cfg["preconditioner"]
        a["solver"] = cfg["solver"]
    if params is not None:
        a["tolerance"] = float(params["tol"])
        for key in ("max_iter", "restart"):
            a[key] = int(params[key])
        for key in ("side", "orthogonalization"):
            a[key] = str(params[key])
    if result is not None:
        hist = np.asarray(result.residual_history, dtype=float)
        a["iterations"] = int(result.iterations)
        a["converged"] = bool(result.converged)
        a["residual"] = float(hist[-1]) if hist.size else float("nan")
        a["true_residual"] = float(result.true_relative_residual)
        a["wall_seconds"] = float(result.wall_seconds)
        a["orthogonalization_seconds"] = float(result.orthogonalization_seconds)
        a["reorthogonalizations"] = int(result.reorthogonalizations)
    return a


def _same_result(a: Any, b: Any) -> bool:
    """True if the SolveResults a and b describe the same solve (equal statistics and history)."""
    return (
        int(a.iterations) == int(b.iterations)
        and bool(a.converged) == bool(b.converged)
        and float(a.wall_seconds) == float(b.wall_seconds)
        and float(a.true_relative_residual) == float(b.true_relative_residual)
        and int(a.reorthogonalizations) == int(b.reorthogonalizations)
        and np.array_equal(np.asarray(a.residual_history), np.asarray(b.residual_history))
    )


def _history_arrays(history: Any, result: Any) -> dict[str, np.ndarray] | None:
    if history is None:
        if result is None:
            return None
        res = np.asarray(result.residual_history, dtype=np.float64)
        return {"iteration": np.arange(res.size, dtype=np.int64), "residual": res}
    if isinstance(history, HistoryRecorder):
        history = {
            "iteration": history.iteration,
            "residual": history.residual,
            "time": history.time,
        }
    elif hasattr(history, "residual_history"):  # a SolveResult
        return _history_arrays(None, history)
    if not isinstance(history, Mapping) or "residual" not in history:
        raise TypeError(
            "history: expected a HistoryRecorder, a SolveResult or a mapping with 'residual' "
            "(and optionally 'iteration', 'time')"
        )
    unknown = set(history) - {"iteration", "residual", "time"}
    if unknown:
        raise ValueError(f"history: unknown keys {sorted(unknown)}")
    res = np.asarray(history["residual"], dtype=np.float64)
    out = {
        "iteration": np.asarray(history.get("iteration", np.arange(res.size)), dtype=np.int64),
        "residual": res,
    }
    if history.get("time") is not None:
        out["time"] = np.asarray(history["time"], dtype=np.float64)
    for key, arr in out.items():
        if arr.shape != res.shape or arr.ndim != 1:
            raise ValueError(f"history['{key}']: expected shape {res.shape}, got {arr.shape}")
    return out


def _field_entry(name: str, entry: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        d = dict(entry)
    elif isinstance(entry, (tuple, list)) and len(entry) in (len(keys) - 1, len(keys)):
        d = dict(zip(keys, entry, strict=False))
    else:
        raise TypeError(f"{name}: expected a tuple {keys} or a mapping with these keys")
    unknown = set(d) - set(keys) - {"attrs"}
    if unknown or keys[0] not in d or keys[1] not in d:
        raise ValueError(f"{name}: expected keys {keys} (last optional) and 'attrs'")
    return d


def _write_fields(
    h5py: Any, root: Any, fields: Mapping[str, Any] | None, far_fields: Mapping[str, Any] | None
) -> None:
    for group_name, entries, keys, units in (
        ("fields", fields, ("points", "E", "H"), ("m", "V/m", "A/m")),
        ("far_fields", far_fields, ("directions", "F"), ("1", "V")),
    ):
        if not entries:
            continue
        if not isinstance(entries, Mapping):
            raise TypeError(f"{group_name}: expected a mapping name -> data")
        top = root.create_group(group_name, track_order=True)
        for name, entry in entries.items():
            d = _field_entry(f"{group_name}[{name!r}]", entry, keys)
            g = top.create_group(_check_name(name, group_name), track_order=True)
            n = None
            for key, unit in zip(keys, units, strict=True):
                if d.get(key) is None:
                    continue
                dtype = "<f8" if key in ("points", "directions") else "<c16"
                ds = _dataset(g, key, d[key], dtype, 2, 3)
                ds.attrs["units"] = unit
                if key == "F":
                    ds.attrs["description"] = _FAR_FIELD_DESCRIPTION
                if n is not None and ds.shape[0] != n:
                    raise ValueError(f"{group_name}[{name!r}]: row counts of the arrays differ")
                n = ds.shape[0]
            if d.get("attrs"):
                _write_attrs(h5py, g, d["attrs"])


def _write_file(path: Path, overwrite: bool, fill: Any) -> Path:
    """Writes via a unique temporary file in the same directory, renamed over path when complete.

    The existence check of ``overwrite=False`` is best effort: a file created at path by
    another process or thread between the check and the final rename is replaced.
    """
    h5py = _h5py()
    path = path.absolute()
    if path.is_dir():
        raise IsADirectoryError(f"{path}: is a directory")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path}: exists (pass overwrite=True to replace it)")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        os.chmod(tmp, _NEW_FILE_MODE)  # mkstemp creates 0o600; use the usual new-file mode
        with h5py.File(tmp, "w", track_order=True) as f:
            f.attrs["format"] = FORMAT
            f.attrs["format_version"] = np.int64(FORMAT_VERSION)
            fill(h5py, f)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _root_attrs(f: Any) -> None:
    from . import __version__

    f.attrs["library_version"] = __version__
    f.attrs["units"] = "SI"
    f.attrs["time_convention"] = "exp(+jwt)"
    f.attrs["normal_convention"] = "normals point out of R2 (object) into R1 (exterior)"
    f.attrs["material_convention"] = _MATERIAL_CONVENTION
    f.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def save_hdf5(
    path: str | os.PathLike[str],
    sim: Any = None,
    *,
    result: Any = None,
    history: Any = None,
    fields: Mapping[str, Any] | None = None,
    far_fields: Mapping[str, Any] | None = None,
    mesh: Any = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a self-describing SpeckleBem HDF5 file (layout in the module docstring).

    Parameters
    ----------
    path : str or os.PathLike
        Output file (UTF-8 / non-ASCII names work). Written to a temporary file in the same
        directory and renamed when complete, so an existing file is never left half-written.
    sim : Simulation, optional
        Source of ``/mesh`` (the mesh actually used, with its interior edges and, on the PML
        route, the zones), ``/currents`` (if solved), ``/rhs`` (if assembled), ``/config``,
        ``/excitation`` and the ``/solver`` attributes. The last solve, its parameters, the
        currents and the right-hand side are copied under one lock acquisition, so a solve
        running in another thread cannot mix two solves in one file (``save_hdf5`` waits for
        it). Without ``sim`` only the explicitly passed parts are written.
    result : SolveResult, optional
        Solve whose statistics and residual history are stored; None = ``sim.last_result``
        (the last successful ``sim.solve()``, whose effective GMRES parameters
        ``sim.last_solve_params`` give tolerance, max_iter, ... in ``/solver``). An explicit
        result gets these parameters only if it is that last solve (equal statistics and
        residual history); for any other result, and without a solve, tolerance, max_iter,
        restart, side and orthogonalization are omitted from ``/solver`` (the configured
        defaults are in ``/config/gmres``). ``/currents`` are always those of the last solve.
    history : HistoryRecorder, SolveResult or mapping, optional
        Per-iteration history for ``/solver/history``: a ``HistoryRecorder`` used as the solve
        callback (with times), or a mapping with ``residual`` and optional ``iteration``,
        ``time`` (1-D, equal lengths). None = the residual history of ``result`` with
        iterations 0, 1, ..., k (no times).
    fields : mapping, optional
        ``name -> (points, E, H)`` or ``name -> {"points", "E", "H", "attrs"}`` (H optional;
        (n, 3) arrays), e.g. ``{"scattered": (pts, *sim.field(pts))}``; ``attrs`` as for
        ``metadata``. Complex ``points`` raise ValueError (never silently made real).
    far_fields : mapping, optional
        ``name -> (directions, F)`` or ``{"directions", "F", "attrs"}`` ((n, 3) arrays),
        stored in ``/far_fields``.
    mesh : TriangleMesh or (vertices, triangles), optional
        Mesh to store when ``sim`` is None (ValueError together with ``sim``).
    metadata : mapping, optional
        User attributes of ``/metadata`` (str, bool, int, float, complex, numeric arrays,
        lists of str; nested mappings become sub-groups).
    overwrite : bool
        Replace an existing file (FileExistsError otherwise). The check is best effort: a
        file created at ``path`` by someone else while this one is written is replaced.

    Returns
    -------
    pathlib.Path
        The absolute path written.

    Raises
    ------
    ImportError
        Without h5py (install hint ``pip install specklebem[io]``).
    FileExistsError, IsADirectoryError, TypeError, ValueError
        For an existing file without ``overwrite``, a directory, unsupported values or
        inconsistent shapes.
    """
    _h5py()  # fail early without h5py
    p = _fspath(path)
    if sim is not None and mesh is not None:
        raise ValueError("save_hdf5: pass either sim or mesh, not both (sim provides the mesh)")
    params = None
    currents = rhs = None
    if sim is not None:
        # One lock acquisition: last solve, its parameters, currents and rhs are consistent.
        snap = sim._save_snapshot()
        currents, rhs = snap["currents"], snap["rhs"]
        last = snap["result"]
        if result is None:
            result, params = last, snap["params"]
        elif last is not None and _same_result(result, last):
            params = snap["params"]
    hist = _history_arrays(history, result)

    def fill(h5py: Any, f: Any) -> None:
        _root_attrs(f)
        if sim is not None:
            _write_mesh(f.create_group("mesh", track_order=True), sim.mesh, sim.zones)
        elif mesh is not None:
            _write_mesh(f.create_group("mesh", track_order=True), mesh, None)
        if currents is not None:
            ds = _dataset(f, "currents", currents, "<c16", 1)
            ds.attrs["N"] = np.int64(ds.shape[0] // 2)
            ds.attrs["layout"] = "x = [J; M], each block of length N (interior edges)"
            ds.attrs["units"] = "J: A/m, M: V/m"
        if rhs is not None:
            _dataset(f, "rhs", rhs, "<c16", 1)
        _write_fields(h5py, f, fields, far_fields)
        solver = _solver_attrs(sim, result, params)
        if solver or hist is not None:
            g = f.create_group("solver", track_order=True)
            _write_attrs(h5py, g, solver)
            if hist is not None:
                hg = g.create_group("history", track_order=True)
                for key, arr in hist.items():
                    hg.create_dataset(key, data=arr)
                if "time" in hist:
                    hg["time"].attrs["units"] = "s"
        if sim is not None:
            _write_attrs(h5py, f.create_group("config", track_order=True), _config_attrs(sim))
            exc = sim.excitation
            g = f.create_group("excitation", track_order=True)
            _write_attrs(h5py, g, _excitation_attrs(exc))
        if metadata:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata: expected a mapping")
            _write_attrs(h5py, f.create_group("metadata", track_order=True), metadata)

    return _write_file(p, overwrite, fill)


def _py(value: Any) -> Any:
    """HDF5 attribute value -> Python scalar / str / ndarray."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O" or value.dtype.kind == "S":
            return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in value.ravel()]
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_group(h5py: Any, g: Any) -> dict[str, Any]:
    """Attributes of g plus sub-groups as nested dicts (datasets are in Result.datasets)."""
    out: dict[str, Any] = {k: _py(v) for k, v in g.attrs.items()}
    for name, item in g.items():
        if isinstance(item, h5py.Group):
            out[name] = _read_group(h5py, item)
    return out


def load_hdf5(path: str | os.PathLike[str]) -> Result:
    """Read a SpeckleBem HDF5 file written by ``save_hdf5`` or ``npy_to_hdf5``.

    Raises
    ------
    ImportError
        Without h5py.
    ValueError
        If the root attribute ``format`` is missing or not ``"specklebem-hdf5"``, or
        ``format_version`` is newer than ``FORMAT_VERSION`` (written by a newer SpeckleBem).
    """
    h5py = _h5py()
    p = _fspath(path)
    r = Result()
    with h5py.File(p, "r") as f:
        fmt = _py(f.attrs.get("format"))
        if fmt != FORMAT:
            raise ValueError(
                f"{p}: not a SpeckleBem HDF5 file (format = {fmt!r}, expected {FORMAT!r})"
            )
        version = _py(f.attrs.get("format_version"))
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"{p}: invalid format_version {version!r}")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"{p}: format_version {version} is newer than the supported "
                f"{FORMAT_VERSION}; update SpeckleBem to read it"
            )
        r.attrs = {k: _py(v) for k, v in f.attrs.items()}

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                r.datasets["/" + name] = obj[()]

        f.visititems(visit)
        d = r.datasets
        r.mesh = {
            k.split("/")[-1]: v
            for k, v in d.items()
            if k.count("/") == 2 and k.startswith("/mesh/")
        }
        r.currents = d.get("/currents")
        r.rhs = d.get("/rhs")
        for group_name, target in (("fields", r.fields), ("far_fields", r.far_fields)):
            if group_name in f:
                for name, g in f[group_name].items():
                    entry: dict[str, Any] = {
                        k: v[()] for k, v in g.items() if isinstance(v, h5py.Dataset)
                    }
                    entry["attrs"] = _read_group(h5py, g)  # attrs and nested sub-groups
                    target[name] = entry
        if "solver" in f:
            r.solver = {k: _py(v) for k, v in f["solver"].attrs.items()}
            if "history" in f["solver"]:
                r.history = {k: v[()] for k, v in f["solver/history"].items()}
        for name in ("config", "excitation", "metadata"):
            if name in f:
                setattr(r, name, _read_group(h5py, f[name]))
    return r


def npy_to_hdf5(
    npy_dir: str | os.PathLike[str], h5_path: str | os.PathLike[str], *, overwrite: bool = False
) -> Path:
    """Convert a ``.npy`` result directory of the C++ writer (``open_npy_directory``) losslessly.

    Every ``<dir>/<a>/<b>.npy`` becomes the dataset ``/<a>/<b>`` with the same dtype, shape and
    values (sub-directories are groups, so a mesh group gives ``/mesh/vertices`` and
    ``/mesh/triangles``); the entries of ``attributes.json`` become the attributes of
    ``/metadata`` in their order. The root carries ``format``, ``format_version``, the other
    root attributes of ``save_hdf5`` and ``source = "npy"``. Leftover temporary files
    (``*~tmp`` of an interrupted C++ write) are skipped with a warning; any other file raises
    ValueError (nothing is dropped silently).

    Returns
    -------
    pathlib.Path
        The absolute path written.
    """
    _h5py()  # fail early without h5py
    src = _fspath(npy_dir, "npy_dir")
    if not src.is_dir():
        raise NotADirectoryError(f"{src}: not a directory")
    arrays: list[tuple[str, Path]] = []
    attrs: dict[str, Any] = {}
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        rel = Path(dirpath).relative_to(src)
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            if fn.endswith("~tmp"):
                warnings.warn(
                    f"npy_to_hdf5: skipping incomplete temporary file {full}", stacklevel=2
                )
            elif fn == "attributes.json" and rel == Path("."):
                with open(full, encoding="utf-8") as fh:
                    attrs = json.load(fh)
                if not isinstance(attrs, dict):
                    raise ValueError(f"{full}: expected a JSON object")
            elif fn.endswith(".npy") and len(fn) > 4:
                arrays.append(("/".join((*rel.parts, fn[:-4])), full))
            else:
                raise ValueError(
                    f"npy_to_hdf5: unexpected file {full} (not written by open_npy_directory)"
                )
    for key, value in attrs.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"attributes.json: unsupported value of {key!r}: {value!r}")

    def fill(h5py: Any, f: Any) -> None:
        _root_attrs(f)
        f.attrs["source"] = "npy"
        for name, file in arrays:
            arr = np.load(file, allow_pickle=False)
            parts = name.split("/")
            g = f
            for part in parts[:-1]:
                g = g[part] if part in g else g.create_group(part, track_order=True)
            g.create_dataset(parts[-1], data=arr)
        if attrs:
            meta = (
                f["metadata"] if "metadata" in f else f.create_group("metadata", track_order=True)
            )
            for key, value in attrs.items():
                meta.attrs[key] = value

    return _write_file(_fspath(h5_path, "h5_path"), overwrite, fill)


def _simulation_save(self: Any, path: str | os.PathLike[str], **kwargs: Any) -> Path:
    """Write this Simulation to an HDF5 file: ``specklebem.io.save_hdf5(path, self, **kwargs)``.

    Stores the mesh, currents, right-hand side, configuration, excitation and the last solve
    (statistics, residual history); keywords as for ``save_hdf5`` (``fields``, ``far_fields``,
    ``history``, ``metadata``, ``overwrite``, ...). Needs h5py (``pip install specklebem[io]``).
    """
    return save_hdf5(path, self, **kwargs)
