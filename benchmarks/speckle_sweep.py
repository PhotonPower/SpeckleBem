"""Phase 6 speckle sweep (WP-S2): one rough surface per process.

Builds the Si rough patch on the box route (ADR 0006 defaults: uniform box, fine band, depth,
w0 = L / 6), solves it with compression "mlfmm" and GMRES (the route's default preconditioner),
evaluates the scattered field on the observation plane at z (< 0, vacuum side) and saves the
intensity pattern plus metadata in OUT/<case tag>/ (``specklebem.speckle_sweep.run_case``). The
ROW line is printed and, with --rows, appended to a file.

--theta-deg and --pol take comma-separated lists (WP-KR3): their Cartesian product, angles
outer and polarisations inner in the given order, is a sequence of excitations of the same
surface, solved on one assembled operator (``specklebem.speckle_sweep.run_sequence``: one mesh,
one assembly and preconditioner setup, then per excitation a right-hand side swap and a solve),
one case directory and one ROW line per excitation. With one angle and one polarisation the run
is the single case of run_case, unchanged. --waist applies to every case; by default each case
gets L cos(theta) / 6 (option b1's footprint rule), and the plane is chosen per case.

Example (the WP-S2 configuration, record benchmarks/results/speckle_sweep.md)::

    /c/msys64/ucrt64/bin/python benchmarks/speckle_sweep.py \\
        --pythonpath build/win-release/python --L 8e-6 --sigma 50e-9 --seed 1 \\
        --leaf-quantile 0.99 --plane-size 4e-3 --plane-n 161 \\
        --out results/local_runs/wp_s2 --rows results/local_runs/wp_s2/rows.txt

The 0 deg / 30 deg pair of one surface in one process: add ``--theta-deg 0,30``.

All lengths in metres (SI). --threads sets OMP_NUM_THREADS before the library is loaded;
--pythonpath puts a build-tree package (``<build>/python``) first on sys.path.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


def _csv_list(convert):
    """argparse type: comma-separated values, each converted, at least one, no duplicates."""

    def parse(text: str) -> list:
        items = [t.strip() for t in text.split(",")]
        if any(not t for t in items):
            raise argparse.ArgumentTypeError(f"empty entry in {text!r}")
        try:
            values = [convert(t) for t in items]
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"invalid value in {text!r}: {e}") from e
        if len(set(values)) != len(values):
            raise argparse.ArgumentTypeError(f"duplicate entry in {text!r}")
        return values

    return parse


def _polarization(text: str) -> str:
    if text.lower() not in ("p", "s"):
        raise ValueError(f"polarisation must be p or s, got {text!r}")
    return text.lower()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--L", type=float, default=8e-6, help="patch edge [m]")
    ap.add_argument("--h", type=float, default=50e-9, help="mesh size [m]")
    ap.add_argument("--sigma", type=float, required=True, help="rms roughness [m]")
    ap.add_argument("--Lc", type=float, default=1e-6, help="correlation length (1/e) [m]")
    ap.add_argument("--seed", type=int, required=True, help="height-map seed (> 0)")
    ap.add_argument("--z", type=float, default=-1e-3, help="observation plane height [m] (< 0)")
    ap.add_argument(
        "--plane-size",
        type=float,
        default=None,
        help="plane edge [m] (default: reaching atan(2) = 63.4 deg along the axes)",
    )
    ap.add_argument(
        "--plane-n",
        type=int,
        default=None,
        help="points per plane axis (default: >= 8 per on-axis speckle, odd)",
    )
    ap.add_argument(
        "--waist",
        type=float,
        default=None,
        help="beam waist [m] for every case (default L cos(theta) / 6 for Si)",
    )
    ap.add_argument(
        "--theta-deg",
        type=_csv_list(float),
        default=[0.0],
        help="incidence angle(s) [deg], comma-separated (R_y(theta) z_hat; tag "
        "suffix _t<theta>); several values: one assembly for all",
    )
    ap.add_argument(
        "--pol",
        type=_csv_list(_polarization),
        default=["p"],
        help="polarisation(s) p and/or s, comma-separated (tag suffix _pols for s)",
    )
    ap.add_argument("--solver-tol", type=float, default=1e-4, help="GMRES tolerance")
    ap.add_argument("--max-iter", type=int, default=None, help="GMRES iteration cap")
    ap.add_argument(
        "--leaf-quantile",
        type=float,
        default=None,
        help="mlfmm leaf_radius_quantile (default: library default 1.0)",
    )
    ap.add_argument(
        "--far-grid",
        default="90,180",
        help="n_theta,n_phi of the reflection-hemisphere far field (R); 0 = skip",
    )
    ap.add_argument("--out", required=True, help="output directory (one subdirectory per case)")
    ap.add_argument("--rows", default=None, help="append the ROW line(s) to this file")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip each case whose directory already holds row.txt",
    )
    ap.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS")
    ap.add_argument(
        "--pythonpath",
        default=None,
        help="directory put first on sys.path (e.g. build/win-release/python)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if args.seed <= 0:
        raise SystemExit("--seed must be > 0 (seed 0 draws a random map)")
    if args.pythonpath:
        sys.path.insert(0, str(Path(args.pythonpath).resolve()))

    import specklebem as sb
    from specklebem import speckle_sweep as ss

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    mlfmm = {}
    if args.leaf_quantile is not None:
        mlfmm["leaf_radius_quantile"] = args.leaf_quantile
    cases = [
        ss.CaseParams(
            L=args.L,
            sigma=args.sigma,
            seed=args.seed,
            h=args.h,
            Lc=args.Lc,
            waist=args.waist,
            solver_tol=args.solver_tol,
            max_iter=args.max_iter,
            mlfmm=mlfmm,
            theta_deg=theta,
            polarization=pol,
        )
        for theta in args.theta_deg
        for pol in args.pol
    ]
    p = cases[0]
    box = sb.rough_surface_box_params(sb.silicon_500nm(), sb.vacuum(), p.wavelength, p.sigma, p.h)
    if (args.plane_size is None) != (args.plane_n is None):
        raise SystemExit("give both --plane-size and --plane-n, or neither")
    planes = []
    for c in cases:
        waist = c.default_waist(box.beam_waist_fraction)
        if args.plane_size is None:
            plane = ss.choose_plane(c.wavelength, args.z, waist, samples_per_speckle=8.0)
        else:
            plane = ss.plane_sampling(c.wavelength, args.z, waist, args.plane_size, args.plane_n)
        prefix = f"{c.tag()}: " if len(cases) > 1 else ""
        log(
            f"{prefix}plane: {plane.n} x {plane.n} points over {plane.size:.4g} m at z = "
            f"{plane.z:g} m, spacing {plane.spacing:.4g} m, on-axis speckle lambda z / D = "
            f"{plane.speckle_size:.4g} m ({plane.samples_per_speckle:.2f} samples), edge angle "
            f"{math.degrees(plane.max_angle):.1f} deg, ~{plane.speckles_per_axis:.1f} speckles "
            f"per axis (at most {plane.max_speckles_per_axis:.1f}; paraxial count "
            f"{plane.paraxial_speckles_per_axis:.1f})"
        )
        planes.append(plane)
    far = None
    if args.far_grid != "0":
        nt, nph = (int(s) for s in args.far_grid.split(","))
        far = (nt, nph)
    case_dirs = [Path(args.out) / c.tag() for c in cases]
    existing = [args.skip_existing and (d / "row.txt").is_file() for d in case_dirs]
    if len(cases) == 1:
        if existing[0]:
            log(f"{case_dirs[0].name}: row.txt exists, skipped")
            print((case_dirs[0] / "row.txt").read_text(encoding="utf-8").strip(), flush=True)
            return 0
        log(f"threads: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(default)')}")
        rows = [ss.format_row(ss.run_case(p, planes[0], case_dirs[0], far_grid=far, log=log))]
        print(rows[0], flush=True)
    else:
        if not all(existing):
            log(f"threads: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(default)')}")
        recs = ss.run_sequence(
            cases, planes, args.out, far_grid=far, skip_existing=args.skip_existing, log=log
        )
        rows = []
        for d, skipped, rec in zip(case_dirs, existing, recs):
            if skipped:  # printed as stored, not appended to --rows again
                print((d / "row.txt").read_text(encoding="utf-8").strip(), flush=True)
            else:
                rows.append(ss.format_row(rec))
                print(rows[-1], flush=True)
    if args.rows and rows:
        with open(args.rows, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(row + "\n" for row in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
