"""Tables and scaling exponents of the MLFMM scaling study (WP22b1).

Reads the ``ROW`` lines that ``specklebem_mlfmm_scaling`` prints (from log files or a rows file),
groups them by material and box option, and fits ``log y = gamma log N + c`` by least squares
for the setup time, the time per matvec (and its near / far parts), the GMRES iterations, the
solve time and the memory. The exponent does not depend on whether N or 2N is used.

Usage: python benchmarks/mlfmm_scaling_fit.py results/*.log [--min-n2 0] [--max-n2 1e9]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

QUANTITIES = [
    ("setup_s", "setup"),
    ("near_s", "near setup"),
    ("far_s", "far setup"),
    ("matvec_s", "matvec"),
    ("near_mv_s", "near matvec"),
    ("far_mv_s", "far matvec"),
    ("it", "iterations"),
    ("solve_s", "solve"),
    ("near_gb", "near memory"),
    ("far_gb", "far memory"),
    ("peak_gb", "peak memory"),
]


def parse_rows(paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("ROW "):
                    fields = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                    rows.append(fields)
    return rows


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept of log y vs log x."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    slope = sxy / sxx
    return slope, my - slope * mx


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-n2", type=float, default=0.0)
    ap.add_argument("--max-n2", type=float, default=1e12)
    args = ap.parse_args(argv)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in parse_rows(args.files):
        groups[(r["material"], r["box"])].append(r)
    for (material, box), rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["N2"]))
        print(f"\n## {material}, box {box}\n")
        print(
            "| L [um] | 2N | levels | leaf [nm] | setup [s] | near [s] | far [s] | matvec [s] "
            "| near mv [s] | far mv [s] | it | solve [s] | true res | near [GB] | far [GB] "
            "| peak [GB] |"
        )
        print("|" + "---:|" * 16)
        for r in rows:
            print(
                f"| {float(r['L_um']):g} | {int(r['N2'])} | {r['levels']} "
                f"| {float(r['leaf_nm']):.0f} | {float(r['setup_s']):.1f} "
                f"| {float(r['near_s']):.1f} | {float(r['far_s']):.1f} "
                f"| {float(r['matvec_s']):.3f} | {float(r['near_mv_s']):.3f} "
                f"| {float(r['far_mv_s']):.3f} | {r['it']} | {float(r['solve_s']):.0f} "
                f"| {float(r['true_res']):.2e} | {float(r['near_gb']):.2f} "
                f"| {float(r['far_gb']):.2f} | {float(r['peak_gb']):.1f} |"
            )
        sel = [r for r in rows if args.min_n2 <= int(r["N2"]) <= args.max_n2]
        if len(sel) < 2:
            continue
        n_lo, n_hi = int(sel[0]["N2"]), int(sel[-1]["N2"])
        print(f"\nExponents gamma (least squares in log-log, 2N = {n_lo} ... {n_hi}, "
              f"{len(sel)} sizes):\n")
        for key, name in QUANTITIES:
            pts = [(int(r["N2"]), float(r[key])) for r in sel if float(r.get(key, "0")) > 0]
            if len(pts) < 2:
                continue
            g, _ = fit([p[0] for p in pts], [p[1] for p in pts])
            print(f"- {name}: gamma = {g:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
