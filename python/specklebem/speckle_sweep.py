"""Phase 6 speckle sweep on rough surfaces (WP-S2): observation plane, one case, ROW records,
analysis of the saved patterns.

Reusable pieces of ``benchmarks/speckle_sweep.py`` (one case per process),
``benchmarks/speckle_sweep_stats.py`` (statistics of the saved patterns) and of the
validation-large test of the Phase 6 definition of done (WP-S2b). The speckle descriptors
themselves are ``specklebem`` / ``post::speckle_statistics`` and the harness
``specklebem.speckle`` (WP-S1); this module only prepares the patterns for them (section
"Analysis of saved patterns": co-polarised intensity, resampling to direction cosines,
leave-one-out ensemble envelope, centred window).

Geometry (docs/06): mean plane z = 0, object (Si) in z > xi(x, y), vacuum R1 below, rigorous
angular-spectrum Gaussian beam travelling in +z with its waist at z = 0; the observation plane
lies at z < 0 (reflection side). SI units throughout.

Speckle size and plane sampling. For a Gaussian illumination with the intensity footprint
exp(-2 r^2 / w0^2) the far-zone field correlation is the Fourier transform of the footprint
(van Cittert-Zernike), |mu_A(dx)|^2 = exp(-(pi w0 dx / (lambda z))^2) on axis, so the full width
at half maximum of |mu_A|^2 is 2 sqrt(ln 2) / pi * lambda z / w0 = 1.06 lambda z / D with
D = 2 w0 the illuminated diameter. The pattern is stationary in direction cosines (u, v), not in
plane coordinates: along an axis of the plane x = z tan(theta), u = sin(theta), so a speckle is
dx = (lambda z / D) / cos^3(theta) wide at the polar angle theta, and a plane that reaches theta_max
along its axes holds about 2 sin(theta_max) D / lambda speckles per axis (at most 2 D / lambda,
for an infinite plane), independent of its size beyond that.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: FWHM of |mu_A|^2 in units of lambda z / w0 for a Gaussian footprint exp(-2 r^2 / w0^2).
GAUSSIAN_FWHM_FACTOR = 2.0 * math.sqrt(math.log(2.0)) / math.pi


def _require_positive(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return v


def paraxial_speckle_size(wavelength: float, z: float, waist: float) -> float:
    """lambda |z| / D [m] with D = 2 w0, the on-axis speckle size of docs/05 row 44."""
    wl = _require_positive(wavelength, "wavelength")
    w0 = _require_positive(waist, "waist")
    if not (math.isfinite(z) and z != 0.0):
        raise ValueError(f"z must be finite and non-zero, got {z!r}")
    return wl * abs(z) / (2.0 * w0)


def gaussian_speckle_fwhm(wavelength: float, z: float, waist: float) -> float:
    """On-axis FWHM [m] of |mu_A|^2 for the footprint exp(-2 r^2 / w0^2): 1.06 lambda |z| / D."""
    return 2.0 * GAUSSIAN_FWHM_FACTOR * paraxial_speckle_size(wavelength, z, waist)


@dataclass(frozen=True)
class PlaneSampling:
    """Square observation plane centred on the z-axis at height z (< 0), n x n points.

    ``speckle_size`` is lambda |z| / D on axis, ``samples_per_speckle`` its ratio to the grid
    spacing, ``max_angle`` the polar angle [rad] at the midpoints of the plane edges,
    ``speckles_per_axis`` = 2 sin(max_angle) D / lambda (the count along an axis, with the
    1 / cos^3 growth of the speckles off axis) and ``paraxial_speckles_per_axis`` = size /
    speckle_size (the count if the speckles kept their on-axis size).
    """

    wavelength: float
    z: float
    waist: float
    size: float
    n: int
    spacing: float
    speckle_size: float
    speckle_fwhm: float
    samples_per_speckle: float
    max_angle: float
    speckles_per_axis: float
    paraxial_speckles_per_axis: float
    max_speckles_per_axis: float

    def points(self) -> np.ndarray:
        """(n * n, 3) points, row i * n + j at (x_i, y_j, z), x_i = -size / 2 + i * spacing."""
        from . import _specklebem as _core

        h = 0.5 * self.size
        return _core.plane_grid(
            origin=[-h, -h, self.z], u=[self.size, 0.0, 0.0], v=[0.0, self.size, 0.0],
            nu=self.n, nv=self.n,
        )

    def coordinates(self) -> np.ndarray:
        """x_i = y_i [m] of the grid lines (length n)."""
        return -0.5 * self.size + self.spacing * np.arange(self.n)


def plane_sampling(wavelength: float, z: float, waist: float, size: float, n: int) -> PlaneSampling:
    """Describe the n x n plane of edge ``size`` [m] at height z for the beam waist w0 [m]."""
    wl = _require_positive(wavelength, "wavelength")
    w0 = _require_positive(waist, "waist")
    s = _require_positive(size, "size")
    if int(n) != n or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}")
    n = int(n)
    if not (math.isfinite(z) and z < 0.0):
        raise ValueError(f"z must be finite and < 0 (reflection side, docs/06), got {z!r}")
    s0 = paraxial_speckle_size(wl, z, w0)
    spacing = s / (n - 1)
    max_angle = math.atan(0.5 * s / abs(z))
    d_over_wl = 2.0 * w0 / wl
    return PlaneSampling(
        wavelength=wl, z=float(z), waist=w0, size=s, n=n, spacing=spacing, speckle_size=s0,
        speckle_fwhm=gaussian_speckle_fwhm(wl, z, w0), samples_per_speckle=s0 / spacing,
        max_angle=max_angle, speckles_per_axis=2.0 * math.sin(max_angle) * d_over_wl,
        paraxial_speckles_per_axis=s / s0, max_speckles_per_axis=2.0 * d_over_wl,
    )


def choose_plane(
    wavelength: float,
    z: float,
    waist: float,
    *,
    max_angle: float = math.atan(2.0),
    samples_per_speckle: float = 4.0,
) -> PlaneSampling:
    """Plane reaching ``max_angle`` [rad] along its axes, sampled with at least
    ``samples_per_speckle`` points per on-axis speckle (lambda |z| / D); n is odd so that the
    centre sample lies on the z-axis."""
    if not (0.0 < max_angle < 0.5 * math.pi):
        raise ValueError(f"max_angle must lie in (0, pi/2), got {max_angle!r}")
    _require_positive(samples_per_speckle, "samples_per_speckle")
    s0 = paraxial_speckle_size(wavelength, z, waist)
    size = 2.0 * abs(z) * math.tan(max_angle)
    n = math.ceil(size / (s0 / samples_per_speckle) * (1.0 - 1e-12)) + 1
    if n % 2 == 0:
        n += 1
    return plane_sampling(wavelength, z, waist, size, n)


# --------------------------------------------------------------------------------------------
# ROW records (one line per case, like the C++ benchmark rows)

#: Key order of the ROW line; further keys follow in insertion order.
ROW_KEYS = (
    "material", "L_um", "h_nm", "sigma_nm", "Lc_um", "seed", "waist_um", "pol", "theta_deg",
    "z_mm", "plane_mm", "plane_n", "tol", "form", "precond", "d0", "lq", "N2", "triangles",
    "rmax_nm", "leaf_nm", "levels", "mesh_s", "setup_s", "it", "conv", "res", "true_res",
    "solve_s", "plane_s", "far_s", "peak_gb", "R", "contrast_global",
)


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    s = str(v)
    if not s or any(c.isspace() for c in s) or "=" in s:
        raise ValueError(f"ROW value must be a non-empty token without spaces or '=': {s!r}")
    return s


def format_row(record: Mapping[str, Any], prefix: str = "ROW") -> str:
    """``ROW k=v ...`` with the keys of ROW_KEYS first (those present), then the rest."""
    keys = [k for k in ROW_KEYS if k in record] + [k for k in record if k not in ROW_KEYS]
    parts = [prefix]
    for k in keys:
        if not k or any(c.isspace() for c in k) or "=" in k:
            raise ValueError(f"invalid ROW key {k!r}")
        parts.append(f"{k}={_format_value(record[k])}")
    return " ".join(parts)


def _parse_value(s: str) -> Any:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse_row(line: str, prefix: str = "ROW") -> dict[str, Any]:
    """Inverse of format_row: ints and floats are converted, other values stay strings."""
    tokens = line.split()
    if not tokens or tokens[0] != prefix:
        raise ValueError(f"not a {prefix} line: {line[:60]!r}")
    out: dict[str, Any] = {}
    for tok in tokens[1:]:
        key, sep, value = tok.partition("=")
        if not sep or not key:
            raise ValueError(f"malformed token {tok!r} in {prefix} line")
        if key in out:
            raise ValueError(f"duplicate key {key!r} in {prefix} line")
        out[key] = _parse_value(value)
    return out


def read_rows(lines: Iterable[str], prefix: str = "ROW") -> list[dict[str, Any]]:
    """All ``prefix`` lines of an iterable of lines (comments and other lines are skipped)."""
    return [parse_row(ln, prefix) for ln in lines if ln.split()[:1] == [prefix]]


# --------------------------------------------------------------------------------------------
# Memory


def peak_memory_bytes() -> int | None:
    """Peak working set (Windows) or maximum resident set (POSIX) of this process [bytes]."""
    if os.name == "nt":

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        c = _Counters()
        c.cb = ctypes.sizeof(c)
        try:
            k32 = ctypes.WinDLL("kernel32")
            proc = k32.GetCurrentProcess
            proc.restype = ctypes.c_void_p
            get = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            get.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
            if not get(proc(), ctypes.byref(c), c.cb):
                return None
        except (OSError, AttributeError):
            return None
        return int(c.PeakWorkingSetSize)
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024


# --------------------------------------------------------------------------------------------
# Mesh helpers


def max_support_radius(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Largest RWG support radius [m] of a closed mesh (mlfmm::max_support_radius): per edge
    max(|a - m|, |p - m|) over the edge midpoint m, its end a and the free vertices p of the two
    triangles. Sets the MLFMM leaf floor r_max / 0.6 (d0 <= 3, ADR 0008)."""
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(triangles, dtype=np.int64)
    r = 0.0
    for k in range(3):
        a, b, p = v[t[:, k]], v[t[:, (k + 1) % 3]], v[t[:, (k + 2) % 3]]
        mid = 0.5 * (a + b)
        half = 0.5 * np.linalg.norm(b - a, axis=1)
        free = np.linalg.norm(p - mid, axis=1)
        r = max(r, float(np.max(np.maximum(half, free))))
    return r


# --------------------------------------------------------------------------------------------
# One case


@dataclass
class CaseParams:
    """One sweep case (lengths in metres, angles in degrees). waist None = L cos(theta) *
    rough_surface_box_params(...).beam_waist_fraction (L / 6 for Si at normal incidence, ADR
    0006 amendment 2026-10-12b; the footprint is 1 / cos(theta) longer). theta_deg is the
    incidence angle of the beam (R_y(theta) z_hat, docs/06); the tag gets ``_t<theta>`` when
    it is non-zero, so normal-incidence tags are unchanged."""

    L: float
    sigma: float
    seed: int
    h: float = 50e-9
    Lc: float = 1e-6
    wavelength: float = 500e-9
    waist: float | None = None
    polarization: str = "p"
    solver_tol: float = 1e-4
    max_iter: int | None = None
    formulation: str = "auto"
    preconditioner: str = "auto"
    mlfmm: dict[str, Any] = field(default_factory=dict)
    region_radius_factor: float = 1.01
    material: str = "si"
    theta_deg: float = 0.0

    def tag(self) -> str:
        t = f"_t{self.theta_deg:g}" if self.theta_deg != 0.0 else ""
        return (f"{self.material}_L{self.L * 1e6:g}um_s{self.sigma * 1e9:g}nm_"
                f"Lc{self.Lc * 1e6:g}um_seed{self.seed}{t}")

    def default_waist(self, beam_waist_fraction: float) -> float:
        """waist if given, else L cos(theta) * beam_waist_fraction."""
        if self.waist is not None:
            return self.waist
        return self.L * math.cos(math.radians(self.theta_deg)) * beam_waist_fraction


def _material(name: str) -> Any:
    from . import _specklebem as _core

    if name == "si":
        return _core.silicon_500nm()
    if name == "ag":
        return _core.silver_500nm()
    raise ValueError(f"material must be 'si' or 'ag', got {name!r}")


def vacuum_wave_impedance(vac: Any, omega: float) -> float:
    """``vac.wave_impedance(omega)`` (the binding returns a complex number) as a float.

    Raises ValueError when the value has a non-zero imaginary part: the background (R1) of the
    sweep is vacuum, lossless, and its impedance enters the power normalisation as a real
    number."""
    z = complex(vac.wave_impedance(omega))
    if z.imag != 0:
        raise ValueError(f"vacuum wave impedance must be real, got {z!r}")
    return z.real


def build_simulation(p: CaseParams) -> tuple[Any, dict[str, Any]]:
    """Box-route Simulation of one case (ADR 0006 defaults, MLFMM, rigorous beam at 0 deg).

    Returns (sim, info); info holds the box parameters, the waist, omega, the background wave
    impedance ``eta0`` (``vacuum().wave_impedance``), r_max, mesh sizes and the mesh generation
    time."""
    from . import _specklebem as _core
    from .rough_surface import RoughSurface

    if not math.isclose(p.wavelength, 500e-9, rel_tol=1e-12):
        raise ValueError("the bundled materials are tabulated at 500 nm")
    obj, vac = _material(p.material), _core.vacuum()
    t0 = time.perf_counter()
    box = _core.rough_surface_box_params(obj, vac, p.wavelength, p.sigma, p.h)
    surf = RoughSurface(L=p.L, sigma=p.sigma, Lc=p.Lc, mesh_size=p.h, seed=p.seed,
                        **box.kwargs())
    mesh = surf.mesh
    mesh_s = time.perf_counter() - t0
    waist = p.default_waist(box.beam_waist_fraction)
    theta = math.radians(p.theta_deg)
    _core.check_beam_waist(p.L, waist, incidence_angle=theta,
                           max_fraction=box.beam_waist_fraction)
    verts = np.asarray(mesh.vertices)
    region = p.region_radius_factor * float(np.max(np.linalg.norm(verts, axis=1)))
    beam = _core.AngularSpectrumBeam(wavelength=p.wavelength, waist=waist,
                                     polarization=p.polarization, incidence_angle=theta,
                                     region_radius=region)
    sim = _core.Simulation(mesh, beam, object=obj, formulation=p.formulation,
                           preconditioner=p.preconditioner, compression="mlfmm",
                           mlfmm=dict(p.mlfmm) or None)
    info = dict(
        box_depth=box.box_depth, box_mesh_size=box.box_mesh_size,
        box_fine_depth=box.box_fine_depth, beam_waist_fraction=box.beam_waist_fraction,
        waist=waist, region_radius=region, num_plane_waves=beam.num_plane_waves,
        omega=float(beam.omega), eta0=vacuum_wave_impedance(vac, float(beam.omega)),
        triangles=mesh.num_triangles, rmax=max_support_radius(verts, mesh.triangles),
        height_rms=float(surf.heights.rms()), mesh_s=mesh_s,
    )
    return sim, info


def _parse_report(report: str) -> dict[str, Any]:
    """Leaf edge and level count from Simulation.report() (best effort, None when absent)."""
    import re

    num = r"([0-9.]+(?:e[-+]?[0-9]+)?)"
    out: dict[str, Any] = {"leaf_nm": None, "levels": None}
    m = re.search(rf"leaves of edge {num} lambda1 \(lambda1 = {num} m\)", report)
    if m:
        out["leaf_nm"] = float(m.group(1)) * float(m.group(2)) * 1e9
    m = re.search(r"octree (\d+) levels", report)
    if m:
        out["levels"] = int(m.group(1))
    return out


def run_case(
    p: CaseParams,
    plane: PlaneSampling,
    out_dir: str | os.PathLike[str],
    *,
    far_grid: tuple[int, int] | None = (90, 180),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build, assemble, solve, evaluate the scattered field on the plane and save everything.

    Files in out_dir (created): ``intensity.npy`` (n, n) |E_s|^2 [V^2/m^2] with [i, j] at
    (x_i, y_j); ``E.npy`` (n * n, 3) complex scattered E [V/m] in plane_grid order;
    ``incident_paraxial_intensity.npy`` (n, n) |E_inc|^2 of the paraxial GaussianBeam of the same
    waist (a note only: the rigorous beam is not controlled at |z| = 1 mm);
    ``currents.npy`` [J; M]; ``residual_history.npy``; with far_grid also ``far_dirs.npy``,
    ``far_weights.npy``, ``far_F.npy`` (1 deg reflection hemisphere by default);
    ``meta.json`` (the returned record plus the report) and ``row.txt`` (the ROW line).
    """
    from . import _specklebem as _core

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = dict(
        material=p.material, L_um=p.L * 1e6, h_nm=p.h * 1e9, sigma_nm=p.sigma * 1e9,
        Lc_um=p.Lc * 1e6, seed=p.seed, pol=p.polarization, theta_deg=p.theta_deg,
        z_mm=plane.z * 1e3, plane_mm=plane.size * 1e3, plane_n=plane.n, tol=p.solver_tol,
    )
    sim, info = build_simulation(p)
    if not math.isclose(info["waist"], plane.waist, rel_tol=1e-12):
        raise ValueError(
            f"plane sampling was computed for w0 = {plane.waist} m, the case uses "
            f"{info['waist']} m")
    rec.update(waist_um=info["waist"] * 1e6, form=str(sim.formulation).split(".")[-1].lower(),
               precond=sim.preconditioner, d0=p.mlfmm.get("accuracy_digits", 3),
               lq=p.mlfmm.get("leaf_radius_quantile", 1.0), N2=sim.num_unknowns,
               triangles=info["triangles"], rmax_nm=info["rmax"] * 1e9, mesh_s=info["mesh_s"])
    log(f"case {p.tag()}: 2N = {sim.num_unknowns}, triangles {info['triangles']}, r_max "
        f"{info['rmax'] * 1e9:.2f} nm, waist {info['waist']:.4g} m, box depth "
        f"{info['box_depth']:.4g} m, plane waves {info['num_plane_waves']}")

    t0 = time.perf_counter()
    sim.assemble()
    rec["setup_s"] = time.perf_counter() - t0
    rec["peak_setup_gb"] = (peak_memory_bytes() or 0) / 1e9
    report_setup = sim.report()
    rec.update({k: v for k, v in _parse_report(report_setup).items() if v is not None})
    log(f"assembled in {rec['setup_s']:.1f} s, peak {rec['peak_setup_gb']:.2f} GB")

    t0 = time.perf_counter()
    res = sim.solve(tol=p.solver_tol, max_iter=p.max_iter)
    rec["solve_s"] = time.perf_counter() - t0
    hist = np.asarray(res.residual_history)
    rec.update(it=res.iterations, conv=bool(res.converged),
               res=float(hist[-1]) if hist.size else float("nan"),
               true_res=float(res.true_relative_residual))
    log(f"solved: {res.iterations} it, converged {res.converged}, true residual "
        f"{res.true_relative_residual:.3e}, {rec['solve_s']:.1f} s")
    np.save(out / "residual_history.npy", hist)
    np.save(out / "currents.npy", np.asarray(sim.currents))

    pts = plane.points()
    t0 = time.perf_counter()
    E, _ = sim.field(pts, kind="scattered")
    rec["plane_s"] = time.perf_counter() - t0
    E = np.asarray(E)
    intensity = np.asarray(_core.intensity(E)).reshape(plane.n, plane.n)
    np.save(out / "E.npy", E)
    np.save(out / "intensity.npy", intensity)
    inc = _core.GaussianBeam(wavelength=p.wavelength, waist=info["waist"],
                             polarization=p.polarization,
                             incidence_angle=math.radians(p.theta_deg))
    i_inc = np.asarray(_core.intensity(inc.electric_field(pts))).reshape(plane.n, plane.n)
    np.save(out / "incident_paraxial_intensity.npy", i_inc)
    rec["contrast_global"] = float(np.std(intensity) / np.mean(intensity))
    rec["I_mean"] = float(np.mean(intensity))
    rec["I_inc_max"] = float(np.max(i_inc))
    log(f"plane field ({plane.n} x {plane.n}) in {rec['plane_s']:.1f} s")

    if far_grid is not None:
        g = _core.hemisphere_grid(-1, int(far_grid[0]), int(far_grid[1]))
        t0 = time.perf_counter()
        F = np.asarray(sim.far_field(g.directions))
        rec["far_s"] = time.perf_counter() - t0
        w = np.asarray(g.weights)
        p_refl = float(w @ np.sum(np.abs(F) ** 2, axis=1)) / (2.0 * info["eta0"])
        p_inc = float(sim.incident_power())
        rec["R"] = p_refl / p_inc
        np.save(out / "far_dirs.npy", np.asarray(g.directions))
        np.save(out / "far_weights.npy", w)
        np.save(out / "far_F.npy", F)
        log(f"far field ({len(w)} directions) in {rec['far_s']:.1f} s, R = {rec['R']:.5f}")

    rec["peak_gb"] = (peak_memory_bytes() or 0) / 1e9
    meta = dict(record=rec, params=asdict(p), info=info, plane=asdict(plane),
                report=sim.report(), report_setup=report_setup)
    with open(out / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, default=str)
    row = format_row(rec)
    with open(out / "row.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(row + "\n")
    return rec


# --------------------------------------------------------------------------------------------
# Analysis of saved patterns (WP-S2). The descriptors (contrast, KS, mu_A, speckle size,
# correlation) are the WP-S1 functions; the steps below only make the plane patterns
# stationary so that those functions apply as documented (docs/10, "Bound in WP-S1").

#: Lead rule of WP-S2b: the negative-exponential gate is asserted only below this coherent
#: (specular) power fraction exp(-(2 k sigma)^2).
FULLY_DEVELOPED_COHERENT_FRACTION = 1e-2


def coherent_fraction(sigma: float, wavelength: float) -> float:
    """exp(-(2 k sigma)^2), k = 2 pi / wavelength (R1): the specular (coherent) share of the
    reflected power at normal incidence for Gaussian heights (Rayleigh / Debye-Waller factor).
    """
    s = float(sigma)
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")
    k = 2.0 * math.pi / _require_positive(wavelength, "wavelength")
    return math.exp(-((2.0 * k * s) ** 2))


def fully_developed(sigma: float, wavelength: float) -> bool:
    """True if coherent_fraction(sigma, wavelength) < FULLY_DEVELOPED_COHERENT_FRACTION."""
    return coherent_fraction(sigma, wavelength) < FULLY_DEVELOPED_COHERENT_FRACTION


def copol_intensity(E: np.ndarray, n: int,
                    co_pol: Iterable[complex] = (1.0, 0.0, 0.0)) -> np.ndarray:
    """Co-polarised intensity |E . co_pol*|^2 of plane_grid samples E (n * n, 3) as (n, n)
    (``polarized_intensity``; row i * n + j -> [i, j]). docs/05 row 42 tests this quantity."""
    from . import _specklebem as _core

    co, _ = _core.polarized_intensity(np.asarray(E), co_pol=list(co_pol))
    co = np.asarray(co, dtype=float)
    if co.size != n * n:
        raise ValueError(f"expected {n * n} samples, got {co.size}")
    return co.reshape(n, n)


def direction_cosine_axis(u_max: float, du: float) -> np.ndarray:
    """Symmetric grid du * (-m ... m), m = floor(u_max / du), of direction cosines."""
    _require_positive(u_max, "u_max")
    _require_positive(du, "du")
    m = int(math.floor(u_max / du + 1e-9))
    if m < 1:
        raise ValueError("direction_cosine_axis: u_max must be at least du")
    return du * np.arange(-m, m + 1)


def plane_to_direction_cosines(
    intensity: np.ndarray, x0: float, dx: float, z: float, u: np.ndarray, *,
    radiant: bool = True,
) -> np.ndarray:
    """Resample a plane pattern onto direction cosines (bilinear).

    ``intensity[i, j]`` lies at (x0 + i dx, x0 + j dx, z); the result ``[a, b]`` is the pattern
    in the direction (u[a], u[b], -sqrt(1 - u^2 - v^2)) from the origin, i.e. at the plane point
    x = |z| u / w, y = |z| v / w, w = sqrt(1 - u^2 - v^2). Far-zone speckle is stationary in
    (u, v) (width c lambda / D at every angle); on the plane it widens as 1 / cos^3(theta).
    ``radiant`` multiplies by (r / |z|)^2 = 1 / w^2, so that the result is proportional to the
    radiant intensity |F|^2 (the 1 / r^2 fall-off of the plane removed).
    """
    grid = np.asarray(intensity, dtype=float)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] < 2:
        raise ValueError("plane_to_direction_cosines: expected a square (n, n) pattern")
    _require_positive(dx, "dx")
    if not (math.isfinite(z) and z != 0.0):
        raise ValueError(f"z must be finite and non-zero, got {z!r}")
    u = np.asarray(u, dtype=float)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    w2 = 1.0 - uu**2 - vv**2
    if np.min(w2) <= 0.0:
        raise ValueError("plane_to_direction_cosines: (u, v) grid leaves the unit disc")
    w = np.sqrt(w2)
    xs, ys = abs(z) * uu / w, abs(z) * vv / w
    x_end = x0 + dx * (grid.shape[0] - 1)
    tol = 1e-9 * dx
    if min(xs.min(), ys.min()) < x0 - tol or max(xs.max(), ys.max()) > x_end + tol:
        raise ValueError("plane_to_direction_cosines: the (u, v) grid reaches beyond the plane")
    fi, fj = (xs - x0) / dx, (ys - x0) / dx
    i = np.clip(np.floor(fi).astype(int), 0, grid.shape[0] - 2)
    j = np.clip(np.floor(fj).astype(int), 0, grid.shape[1] - 2)
    ti, tj = fi - i, fj - j
    out = ((1 - ti) * (1 - tj) * grid[i, j] + ti * (1 - tj) * grid[i + 1, j]
           + (1 - ti) * tj * grid[i, j + 1] + ti * tj * grid[i + 1, j + 1])
    return out / w2 if radiant else out


def mirror_images(a: np.ndarray) -> list[np.ndarray]:
    """[a, a mirrored in axis 0, in axis 1, in both] (grids symmetric about their centre).

    At normal incidence with E along x the ensemble-mean co-polarised intensity is even in u
    and in v (the height statistics are mirror invariant; the mirror x -> -x flips E_x, which
    leaves |E_x|^2 unchanged), so the mirror images are further samples of the same envelope.
    """
    a = np.asarray(a, dtype=float)
    return [a, a[::-1, :], a[:, ::-1], a[::-1, ::-1]]


def gaussian_smooth(field: np.ndarray, radius_px: float) -> np.ndarray:
    """Normalised smoothing with exp(-r^2 / radius^2) (zero padded; a constant is preserved up
    to the edges). radius 0 returns a copy."""
    f = np.asarray(field, dtype=float)
    if f.ndim != 2:
        raise ValueError("gaussian_smooth: expected a 2-D array")
    if not (math.isfinite(radius_px) and radius_px >= 0.0):
        raise ValueError("gaussian_smooth: radius must be finite and >= 0")
    if radius_px == 0.0:
        return f.copy()
    n0, n1 = f.shape
    k0 = np.fft.fftfreq(2 * n0)[:, None]
    k1 = np.fft.rfftfreq(2 * n1)[None, :]
    kern = np.exp(-((math.pi * radius_px) ** 2) * (k0**2 + k1**2))  # FT of exp(-r^2 / a^2)
    pad = np.zeros((2 * n0, 2 * n1))
    pad[:n0, :n1] = f
    num = np.fft.irfft2(np.fft.rfft2(pad) * kern, s=pad.shape)[:n0, :n1]
    pad[:] = 0.0
    pad[:n0, :n1] = 1.0
    den = np.fft.irfft2(np.fft.rfft2(pad) * kern, s=pad.shape)[:n0, :n1]
    return num / den


def leave_one_out_envelopes(patterns: Iterable[np.ndarray],
                            radius_px: float) -> list[np.ndarray]:
    """Envelope (local mean intensity) for each pattern, estimated without the pattern itself.

    For pattern i: the mean of the four mirror images of every other pattern and of the three
    non-trivial mirror images of pattern i, smoothed with ``gaussian_smooth(radius_px)``. A
    pattern divided by an envelope that contains it is biased towards its own mean (synthetic
    speckle with the sweep's window sizes: self-including envelope contrast -10 to -14 %, KS
    level up to 12 %; leave-one-out contrast -5 to +2 %, see benchmarks/results/speckle_sweep.md).

    The exclusion is not exact on the mirror axes: on the line u = 0 (axis 0 of a grid symmetric
    about its centre) the mirror image in axis 0 coincides with the pattern itself, on v = 0 the
    mirror image in axis 1 does, and at the centre all three do, so there the envelope contains
    the pattern's own values (weight 1/7 or 3/7 of the unsmoothed mean for two patterns; one
    grid line per axis, before smoothing).
    """
    pats = [np.asarray(p, dtype=float) for p in patterns]
    if not pats or any(p.ndim != 2 or p.shape != pats[0].shape for p in pats):
        raise ValueError("leave_one_out_envelopes: need 2-D patterns of one shape")
    out = []
    for i, p in enumerate(pats):
        imgs = [m for j, q in enumerate(pats) if j != i for m in mirror_images(q)]
        imgs += mirror_images(p)[1:]
        out.append(gaussian_smooth(np.mean(imgs, axis=0), radius_px))
    return out


def centred_window(envelope: np.ndarray, threshold: float) -> int:
    """Largest h such that the centred (2h + 1)^2 square of ``envelope`` (odd square grid) has
    every value >= threshold * max(envelope); -1 if even the centre pixel fails."""
    e = np.asarray(envelope, dtype=float)
    if e.ndim != 2 or e.shape[0] != e.shape[1] or e.shape[0] % 2 == 0:
        raise ValueError("centred_window: expected an odd square grid")
    if not (0.0 < threshold < 1.0):
        raise ValueError("centred_window: threshold must lie in (0, 1)")
    c = e.shape[0] // 2
    level = threshold * float(np.max(e))
    h = -1
    while h + 1 <= c:
        k = h + 1
        if np.min(e[c - k:c + k + 1, c - k:c + k + 1]) < level:
            break
        h = k
    return h


def centre_crop(a: np.ndarray, half: int) -> np.ndarray:
    """Centred (2 half + 1)^2 square of an odd square grid."""
    a = np.asarray(a)
    c = a.shape[0] // 2
    if half < 0 or half > c:
        raise ValueError("centre_crop: half width out of range")
    return a[c - half:c + half + 1, c - half:c + half + 1]


def decimate_count(shape: tuple[int, ...], stride: int) -> int:
    """Number of samples ``speckle.decimate`` keeps on a 2-D grid of this shape."""
    return (-(-int(shape[0]) // stride)) * (-(-int(shape[1]) // stride))


def decimated_samples(windows: Iterable[np.ndarray], stride: int,
                      offset: int = 0) -> list[np.ndarray]:
    """``speckle.decimate`` of each window divided by its own mean (the KS input of
    ``speckle.ensemble_statistics``: pooled I / <I>_pattern). ``offset`` (0 <= offset < stride)
    shifts the decimation grid by that many pixels along both axes (``w[offset::stride,
    offset::stride]``; 0 is the harness's grid), for sensitivity studies of the gate."""
    from .speckle import decimate

    if int(offset) != offset or not 0 <= offset < stride:
        raise ValueError(f"decimated_samples: offset must be an integer in [0, stride), got "
                         f"{offset!r}")
    out = []
    for w in windows:
        d = decimate(np.asarray(w)[int(offset):, int(offset):], stride).ravel()
        if not d.mean() > 0:
            raise ValueError("decimated_samples: a decimated window has zero mean")
        out.append(d / d.mean())
    return out


def pooled_ks(samples: Iterable[np.ndarray]) -> Any:
    """``ks_test_exponential`` of the concatenated normalised samples (needs >= 50)."""
    from . import _specklebem as _core

    return _core.ks_test_exponential(np.concatenate([np.ravel(s) for s in samples]))


def synthetic_far_zone_speckle(
    rng: np.random.Generator | None,
    plane: PlaneSampling,
    *,
    envelope_rms: float | None = None,
    samples_per_speckle: int = 16,
    n_fft: int = 256,
    aperture: np.ndarray | None = None,
) -> np.ndarray:
    """Synthetic plane pattern for testing and calibrating ``analyse_ensemble``.

    Fully developed far-zone speckle of a random-phase Gaussian spot |a|^2 = exp(-2 r^2 / w0^2)
    (w0 = plane.waist): the field in direction cosines is the FFT of the aperture field, so
    |mu_A|^2 is exactly the FT of the footprint (the WP-S1 Gaussian convention, FWHM
    1.060041 lambda / D in u, D = 2 w0), sampled with ``samples_per_speckle`` points per
    lambda / D. It is multiplied by the envelope exp(-(u^2 + v^2) / (2 envelope_rms^2)) (None:
    flat) and mapped to the plane as radiant intensity times w^2 = (|z| / r)^2 (bilinear), the
    form of the scattered co-polarised intensity of the sweep. Returns (n, n).

    ``aperture`` (n_fft, n_fft) complex replaces the random phase screen exp(2 pi j U) drawn from
    ``rng`` (``rng`` is then unused and may be None): e.g. ``circular_gaussian_field`` and
    ``correlated_field`` for patterns that share part of their random field (the sweep reuses
    one height map per seed at every sigma).
    """
    du, xx, yy = _aperture_grid(plane, samples_per_speckle, n_fft)
    if aperture is None:
        if rng is None:
            raise ValueError("synthetic_far_zone_speckle: need rng or aperture")
        phase = np.exp(2j * math.pi * rng.random((n_fft, n_fft)))
    else:
        phase = np.asarray(aperture, dtype=complex)
        if phase.shape != (n_fft, n_fft):
            raise ValueError(f"synthetic_far_zone_speckle: aperture must be ({n_fft}, {n_fft})")
    i_uv = np.abs(np.fft.fftshift(np.fft.fft2(_footprint(xx, yy, plane.waist) * phase))) ** 2
    if envelope_rms is not None:
        u = du * (np.arange(n_fft) - n_fft // 2)
        uu, vv = np.meshgrid(u, u, indexing="ij")
        i_uv = i_uv * np.exp(-(uu**2 + vv**2) / (2.0 * _require_positive(envelope_rms,
                                                                          "envelope_rms") ** 2))
    return _far_zone_to_plane(i_uv, du, plane)


def circular_gaussian_field(rng: np.random.Generator, n_fft: int = 256) -> np.ndarray:
    """White circular complex Gaussian field (n_fft, n_fft) with E|a|^2 = 1 (an aperture for
    ``synthetic_far_zone_speckle``: its far zone is exact circular Gaussian speckle)."""
    n = int(n_fft)
    if n != n_fft or n < 2:
        raise ValueError(f"circular_gaussian_field: n_fft must be an integer >= 2, got {n_fft!r}")
    return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(2.0)


def correlated_field(rng: np.random.Generator, field: np.ndarray, mu: float) -> np.ndarray:
    """mu a + sqrt(1 - mu^2) b with b a fresh ``circular_gaussian_field``: a field of the same
    statistics whose correlation with ``field`` is mu (0 <= mu <= 1). The far-zone speckle
    intensities of the two then correlate with |mu|^2 (Gaussian moment theorem), the analogue
    of the sweep's sigma steps on one scaled height map (measured rho_env 0.05-0.54)."""
    m = float(mu)
    if not (math.isfinite(m) and 0.0 <= m <= 1.0):
        raise ValueError(f"correlated_field: mu must lie in [0, 1], got {mu!r}")
    a = np.asarray(field, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("correlated_field: expected a square 2-D field")
    return m * a + math.sqrt(1.0 - m * m) * circular_gaussian_field(rng, a.shape[0])


def _aperture_grid(plane: PlaneSampling, samples_per_speckle: int,
                   n_fft: int) -> tuple[float, np.ndarray, np.ndarray]:
    """du of the far-zone FFT grid and the aperture coordinates (pitch lambda / (n_fft du))."""
    du = (plane.wavelength / (2.0 * plane.waist)) / samples_per_speckle
    pitch = plane.wavelength / (n_fft * du)
    x = pitch * (np.arange(n_fft) - n_fft // 2)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return du, xx, yy


def _footprint(xx: np.ndarray, yy: np.ndarray, w0: float) -> np.ndarray:
    """Field footprint exp(-r^2 / w0^2) (intensity exp(-2 r^2 / w0^2))."""
    return np.exp(-(xx**2 + yy**2) / w0**2)


def unit_height_screen(
    rng: np.random.Generator,
    plane: PlaneSampling,
    correlation_length: float,
    *,
    samples_per_speckle: int = 16,
    n_fft: int = 256,
) -> np.ndarray:
    """Gaussian random heights of unit rms with correlation exp(-r^2 / Lc^2) on the aperture grid
    of ``phase_screen_far_zone_speckle`` (periodic, pitch lambda / (n_fft du)); scale by sigma
    for a surface of rms sigma (one seed, every sigma: the sweep's convention)."""
    lc = _require_positive(correlation_length, "correlation_length")
    du, xx, _ = _aperture_grid(plane, samples_per_speckle, n_fft)
    pitch = float(xx[1, 0] - xx[0, 0])
    if lc < 3.0 * pitch or n_fft * pitch < 10.0 * lc:
        raise ValueError("unit_height_screen: correlation length not resolved by the grid")
    kx = 2.0 * math.pi * np.fft.fftfreq(n_fft, d=pitch)
    k2 = kx[:, None] ** 2 + kx[None, :] ** 2
    h = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n_fft, n_fft)))
                             * np.exp(-k2 * lc**2 / 8.0)))  # spectrum ~ exp(-k^2 Lc^2 / 4)
    h -= h.mean()
    return h / h.std()


def phase_screen_far_zone_speckle(
    unit_heights: np.ndarray,
    sigma: float,
    plane: PlaneSampling,
    *,
    samples_per_speckle: int = 16,
) -> np.ndarray:
    """Kirchhoff (thin phase screen) surrogate of the sweep: the far zone of the footprint
    exp(-r^2 / w0^2) times exp(-2 j k sigma h(x, y)) at normal incidence (scalar, no Fresnel or
    obliquity factors; the phase is 2 k h instead of k h (1 + cos theta_s), adequate for the
    small slopes sqrt(2) sigma / Lc <~ 0.35 of the sweep), mapped to the plane like
    ``synthetic_far_zone_speckle``. Contains the coherent (specular) part exp(-(2 k sigma)^2)
    and the finite number of illuminated correlation cells, so it gives the configuration's
    expected statistics (contrast, KS rejection rate, seed and sigma-step correlations)."""
    h = np.asarray(unit_heights, dtype=float)
    n_fft = h.shape[0]
    if h.ndim != 2 or h.shape[1] != n_fft:
        raise ValueError("phase_screen_far_zone_speckle: expected a square height grid")
    du, xx, yy = _aperture_grid(plane, samples_per_speckle, n_fft)
    k = 2.0 * math.pi / plane.wavelength
    field = _footprint(xx, yy, plane.waist) * np.exp(-2j * k * float(sigma) * h)
    i_uv = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return _far_zone_to_plane(i_uv, du, plane)


def _far_zone_to_plane(i_uv: np.ndarray, du: float, plane: PlaneSampling) -> np.ndarray:
    """Bilinear lookup of a centred (u, v) FFT grid at the plane points, times (|z| / r)^2."""
    n_fft = i_uv.shape[0]
    c = plane.coordinates()
    X, Y = np.meshgrid(c, c, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + plane.z**2)
    fi, fj = X / r / du + n_fft // 2, Y / r / du + n_fft // 2
    if min(fi.min(), fj.min()) < 0 or max(fi.max(), fj.max()) >= n_fft - 1:
        raise ValueError("synthetic_far_zone_speckle: n_fft too small for the plane")
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    ti, tj = fi - i0, fj - j0
    val = ((1 - ti) * (1 - tj) * i_uv[i0, j0] + ti * (1 - tj) * i_uv[i0 + 1, j0]
           + (1 - ti) * tj * i_uv[i0, j0 + 1] + ti * tj * i_uv[i0 + 1, j0 + 1])
    return val * (plane.z / r) ** 2


def analyse_ensemble(
    plane_patterns: Iterable[np.ndarray],
    plane: PlaneSampling,
    *,
    u_max: float = 0.66,
    samples_per_speckle: float = 8.0,
    envelope_radius: float = 0.5,
    threshold: float = 0.05,
    ks_spacing: float = 2.0,
) -> dict[str, Any]:
    """Speckle statistics of the patterns of one sigma (the seeds) with the WP-S1 harness.

    Steps: (1) the co-polarised plane patterns are resampled to direction cosines as radiant
    intensity (``plane_to_direction_cosines``, |u|, |v| <= u_max, du = lambda / D /
    samples_per_speckle, D = 2 w0); (2) each is divided by its leave-one-out envelope
    (``leave_one_out_envelopes``, smoothing radius ``envelope_radius`` speckle sizes; speckle
    size in u = ``speckle_size_expected(.., "gaussian")`` / |z|); (3) the centred window where
    the smoothed mean envelope (all mirror images) stays >= ``threshold`` x its maximum is cut
    out, the same for every pattern; (4) ``speckle.ensemble_statistics`` runs on the windows
    with pixel size du |z| (plane units on axis, so its speckle size compares directly with
    c lambda |z| / D), with the KS stride for ``ks_spacing`` speckle sizes when that gives >= 50
    pooled samples, else for one speckle size, else stride 1 (KS then invalid).

    Returns a dict: ``u``, ``du``, ``pixel_size``, ``speckle_u`` (c lambda / D),
    ``contrast_plane`` (``speckle_contrast`` of each raw plane pattern: the docs/05 row 43
    quantity), ``rho_plane`` ({(i, j): ``correlation_coefficient`` of raw plane patterns i < j}),
    ``radiant`` and ``normalised`` (full (u, v) grids), ``envelope`` (smoothed mean of all
    mirror images),``window_half`` [px], ``window_u`` (half width in u), ``window_speckles`` (window
    edge in speckle sizes), ``windows``, ``stats`` (the harness dict), ``ks_spacing_used``
    (``ks_spacing``, 1.0 or None = invalid) and ``ks_samples`` (decimated I / <I> per pattern at
    ``ks_spacing``, for pooling over sigma; ``ks_samples_one`` the same at one speckle size).
    """
    from . import _specklebem as _core
    from .speckle import ensemble_statistics, independent_stride

    pats = [np.asarray(p, dtype=float) for p in plane_patterns]
    if not pats or any(p.shape != (plane.n, plane.n) for p in pats):
        raise ValueError("analyse_ensemble: patterns must be (n, n) arrays of the plane")
    wl, zz, D = plane.wavelength, abs(plane.z), 2.0 * plane.waist
    size_plane = float(_core.speckle_size_expected(wl, zz, D, "gaussian"))
    speckle_u = size_plane / zz
    du = (wl / D) / _require_positive(samples_per_speckle, "samples_per_speckle")
    u = direction_cosine_axis(u_max, du)
    x0 = -0.5 * plane.size
    radiant = [plane_to_direction_cosines(p, x0, plane.spacing, plane.z, u) for p in pats]
    radius_px = envelope_radius * speckle_u / du
    envs = leave_one_out_envelopes(radiant, radius_px)
    normalised = [r / e for r, e in zip(radiant, envs)]
    mean_env = gaussian_smooth(np.mean([m for r in radiant for m in mirror_images(r)], axis=0),
                               radius_px)
    half = centred_window(mean_env, threshold)
    if half < 1:
        raise ValueError("analyse_ensemble: no window above the envelope threshold")
    windows = [centre_crop(q, half) for q in normalised]
    pixel = du * zz

    stride_req = independent_stride(size_plane, pixel, ks_spacing)
    stride_one = independent_stride(size_plane, pixel, 1.0)
    if sum(decimate_count(w.shape, stride_req) for w in windows) >= 50:
        stride, used = stride_req, ks_spacing
    elif sum(decimate_count(w.shape, stride_one) for w in windows) >= 50:
        stride, used = stride_one, 1.0
    else:
        stride, used = 1, None
    stats = ensemble_statistics(windows, pixel, wl, zz, D, shape="gaussian", ks_stride=stride)
    rho = {(i, j): float(_core.correlation_coefficient(pats[i].ravel(), pats[j].ravel()))
           for i in range(len(pats)) for j in range(i + 1, len(pats))}
    return dict(
        u=u, du=du, pixel_size=pixel, speckle_u=speckle_u,
        contrast_plane=np.array([float(_core.speckle_contrast(p.ravel())) for p in pats]),
        rho_plane=rho, radiant=radiant, normalised=normalised, envelope=mean_env,
        window_half=half, window_u=half * du, window_speckles=(2 * half + 1) * du / speckle_u,
        windows=windows, stats=stats, ks_spacing_used=used,
        ks_samples=decimated_samples(windows, stride_req),
        ks_samples_one=decimated_samples(windows, stride_one),
    )
