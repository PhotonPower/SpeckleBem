"""Tables and scaling exponents of the MLFMM scaling study (WP22b1).

Reads the ``ROW`` (and, unless ``--rows-only``, ``XROW``) lines that ``specklebem_mlfmm_scaling``
prints (from log files or a rows file) and fits ``log y = gamma log 2N + c`` by least squares for
the setup time, the time per matvec (and its near / far parts), the GMRES iterations, the solve
time, the time per iteration and the memory. The exponent does not depend on whether N or 2N is
used.

Grouping: one series per (material, box option, fine band, restart, h, waist factor, beam, d0,
tol, leaf radius quantile lq); rows written before these fields existed get the study defaults
(h = 50 nm, w0 = L/4, paraxial beam, d0 = 3, tol = 1e-3, lq = 1: the global leaf rule). Repeated runs of one size (same series and 2N) are merged by
taking, per quantity, the minimum over the runs that measured it (timings on a shared machine
carry load noise that only ever adds time; memory and iterations are deterministic).

Filters: iterations, solve time and time per iteration enter only from solved, converged rows
(``solved=1 conv=1``); setup, matvec and memory from every row. A row may carry
``exclude=<key>[,<key>...]`` to drop named quantities from the fits (documented outliers, e.g. a
load-inflated matvec median); the table marks them with ``*``.

Fits are reported over the full range of each series and over the DoD window 2N >= 5e4
(``--dod-min``), with the standard error of the slope (``gamma +- se``; none for two points).
The proxy ``it x (near + far)`` is the solve time without the orthogonalisation and without the
load noise of the solve itself.

Usage: python benchmarks/mlfmm_scaling_fit.py results/*.log [--min-n2 0] [--max-n2 1e12]
           [--dod-min 5e4] [--rows-only]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

# (key, name, needs a converged solve)
QUANTITIES = [
    ("setup_s", "setup", False),
    ("near_s", "near setup", False),
    ("far_s", "far setup", False),
    ("matvec_s", "matvec", False),
    ("near_mv_s", "near matvec", False),
    ("far_mv_s", "far matvec", False),
    ("nf_mv_s", "near + far matvec", False),
    ("it", "iterations", True),
    ("solve_s", "solve", True),
    ("s_per_it", "solve time per iteration", True),
    ("proxy_s", "proxy it x (near + far matvec)", True),
    ("near_gb", "near memory", False),
    ("far_gb", "far memory", False),
    ("peak_gb", "peak memory", False),
]
SOLVE_KEYS = {k for k, _, s in QUANTITIES if s}
DEFAULTS = {
    "restart": "0",
    "solved": "1",
    "h_nm": "50",
    "waist": "4",
    "beam": "paraxial",
    "d0": "3",
    "tol": "0.001",
    "fine": "0",
    "lq": "1",  # leaf radius quantile; rows before WP21L used the global leaf rule
}
GROUP_KEYS = ("material", "box", "fine", "restart", "h_nm", "waist", "beam", "d0", "tol", "lq")


def parse_rows(paths: list[str], rows_only: bool) -> list[dict[str, str]]:
    rows = []
    prefixes = ("ROW ",) if rows_only else ("ROW ", "XROW ")
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith(prefixes):
                    continue
                fields = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                for k, v in DEFAULTS.items():
                    fields.setdefault(k, v)
                # Normalise numeric group fields ("4" and "4.0" are the same series).
                for k in ("h_nm", "waist", "d0", "tol", "lq"):
                    fields[k] = f"{float(fields[k]):g}"
                it = int(fields.get("it", "0"))
                solve = float(fields.get("solve_s", "0"))
                near_mv = float(fields.get("near_mv_s", "0"))
                far_mv = float(fields.get("far_mv_s", "0"))
                fields["s_per_it"] = str(solve / it if it > 0 else 0.0)
                fields["nf_mv_s"] = str(near_mv + far_mv)
                fields["proxy_s"] = str(it * (near_mv + far_mv))
                rows.append(fields)
    return rows


def usable(r: dict[str, str], key: str) -> bool:
    """Whether row r contributes quantity key to the fits."""
    if key in r.get("exclude", "").split(","):
        return False
    if key in SOLVE_KEYS and not (r.get("solved") == "1" and r.get("conv") == "1"):
        return False
    return float(r.get(key, "0")) > 0


def merge_repeats(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """One entry per 2N: per quantity the minimum over the rows that measured it."""
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_n[int(r["N2"])].append(r)
    merged = []
    for n2 in sorted(by_n):
        group = by_n[n2]
        m: dict[str, object] = {"N2": n2, "runs": len(group), "base": group[0], "values": {}}
        for key, _, _ in QUANTITIES:
            vals = [float(r[key]) for r in group if usable(r, key)]
            if vals:
                m["values"][key] = min(vals)  # type: ignore[index]
        merged.append(m)
    return merged


def fit(xs: list[float], ys: list[float]) -> tuple[float, float, float | None]:
    """Least-squares slope, intercept and slope standard error of log y vs log x."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx == 0.0:
        raise ValueError("all sizes equal: no slope")
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    slope = sxy / sxx
    icpt = my - slope * mx
    if n < 3:
        return slope, icpt, None
    rss = sum((b - (icpt + slope * a)) ** 2 for a, b in zip(lx, ly))
    return slope, icpt, math.sqrt(rss / (n - 2) / sxx)


def fmt(v: dict[str, object], key: str, spec: str) -> str:
    vals = v["values"]  # type: ignore[assignment]
    if key not in vals:  # type: ignore[operator]
        return "-"
    return format(vals[key], spec)  # type: ignore[index]


def report_fits(merged: list[dict[str, object]], lo: float, hi: float, title: str) -> None:
    sel = [m for m in merged if lo <= m["N2"] <= hi]  # type: ignore[operator]
    lines = []
    for key, name, _ in QUANTITIES:
        pts = [(m["N2"], m["values"][key]) for m in sel if key in m["values"]]  # type: ignore
        if len(pts) < 2:
            continue
        try:
            g, _, se = fit([float(p[0]) for p in pts], [float(p[1]) for p in pts])
        except ValueError:
            continue
        err = f" +- {se:.2f}" if se is not None else " (2 points, no error)"
        lines.append(
            f"- {name}: gamma = {g:.2f}{err} (2N = {pts[0][0]} ... {pts[-1][0]}, "
            f"{len(pts)} sizes)"
        )
    if lines:
        print(f"\n{title}:\n")
        print("\n".join(lines))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-n2", type=float, default=0.0)
    ap.add_argument("--max-n2", type=float, default=1e12)
    ap.add_argument("--dod-min", type=float, default=5e4, help="lower 2N of the DoD window")
    ap.add_argument("--rows-only", action="store_true", help="ignore XROW lines")
    args = ap.parse_args(argv)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in parse_rows(args.files, args.rows_only):
        groups[tuple(r[k] for k in GROUP_KEYS)].append(r)
    for key, rows in sorted(groups.items()):
        desc = ", ".join(f"{k} {v}" for k, v in zip(GROUP_KEYS, key))
        merged = merge_repeats(rows)
        print(f"\n## {desc}\n")
        print(
            "| L [um] | 2N | runs | levels | leaf [nm] | setup [s] | near [s] | far [s] "
            "| matvec [s] | near mv [s] | far mv [s] | it | conv | solve [s] | s/it "
            "| near [GB] | far [GB] | peak [GB] |"
        )
        print("|" + "---:|" * 18)
        for m in merged:
            b = m["base"]  # type: ignore[assignment]
            excluded = set().union(*(set(r.get("exclude", "").split(",")) for r in rows
                                     if int(r["N2"]) == m["N2"]))
            mv = fmt(m, "matvec_s", ".3f") + ("*" if "matvec_s" in excluded else "")
            conv = "yes" if any(r.get("conv") == "1" for r in rows if int(r["N2"]) == m["N2"]) \
                else "-"
            print(
                f"| {float(b['L_um']):g} | {m['N2']} | {m['runs']} | {b['levels']} "
                f"| {float(b['leaf_nm']):.0f} | {fmt(m, 'setup_s', '.1f')} "
                f"| {fmt(m, 'near_s', '.1f')} | {fmt(m, 'far_s', '.1f')} | {mv} "
                f"| {fmt(m, 'near_mv_s', '.3f')} | {fmt(m, 'far_mv_s', '.3f')} "
                f"| {fmt(m, 'it', '.0f')} | {conv} | {fmt(m, 'solve_s', '.0f')} "
                f"| {fmt(m, 's_per_it', '.2f')} | {fmt(m, 'near_gb', '.2f')} "
                f"| {fmt(m, 'far_gb', '.2f')} | {fmt(m, 'peak_gb', '.1f')} |"
            )
        lo = max(args.min_n2, 0.0)
        report_fits(merged, lo, args.max_n2,
                    "Exponents gamma +- standard error (least squares in log-log, full range)")
        report_fits(merged, max(lo, args.dod_min), args.max_n2,
                    f"Exponents over the DoD window 2N >= {args.dod_min:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
