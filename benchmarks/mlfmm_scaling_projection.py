"""Projection of the capped WP22b3 Ag solves (record benchmarks/results/mlfmm_scaling.md).

Reads the GMRES histories of ``benchmarks/results/mlfmm_scaling_ag_histories.txt`` (k, wall time
t, monitored residual every 50 iterations, per run) and reproduces the figures of the record that
rest on them:

1. per run the time fit t(k) = a + b k + c k^2 (least squares over all samples), with b against
   the measured matvec and c / 2N (the CGS2 orthogonalisation, which grows as k^2 2N);
2. the iteration bracket of a run truncated at a cap: the residual over the last ``--window``
   iterations is extrapolated to ``--tol`` log-linearly (log r = s k + q: the low end) and as a
   power law (log r = p log k + q: the high end); the time follows from the run's t(k) fit
   (truncated at the same cap);
3. the back-test of the bracket on the converged runs (L = 11 um capped at 1400 / 1600 / 1800,
   L = 9 um at 1000 / 1200) and the projections of the capped L = 15 and 17.5 um runs, with
   their Krylov basis (k + 1) 2N 16 B;
4. the solve-time exponents with the projected points (low / high bracket), over the full range
   and over the DoD window 2N >= 5e4, and the iterations that gamma = 2.0 would allow at
   2N = 1.02e6;
5. the orthogonalisation share (solve - it x matvec) / solve and the local slopes between
   neighbouring converged sizes.

Usage: python benchmarks/mlfmm_scaling_projection.py
           [benchmarks/results/mlfmm_scaling_ag_histories.txt] [--window 300] [--tol 1e-3]
Python 3 with numpy only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

DEFAULT_FILE = Path(__file__).resolve().parent / "results" / "mlfmm_scaling_ag_histories.txt"
BACK_TESTS = [(11.0, [1400, 1600, 1800]), (9.0, [1000, 1200])]
CAPPED = [15.0, 17.5]
DOD_MIN = 5e4


def read_runs(path: Path) -> list[dict]:
    runs: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("run "):
            meta = dict(kv.split("=", 1) for kv in line.split()[1:])
            runs.append(
                {
                    "L": float(meta["L_um"]),
                    "n2": int(meta["N2"]),
                    "matvec": float(meta["matvec_s"]),
                    "it": int(meta["it"]),
                    "conv": meta["conv"] == "1",
                    "solve": float(meta["solve_s"]),
                    "orth": None if meta["orth_s"] == "na" else float(meta["orth_s"]),
                    "samples": [],
                }
            )
            continue
        k, t, r = line.split()
        runs[-1]["samples"].append((int(k), float(t), float(r)))
    for run in runs:
        s = np.array(run["samples"], dtype=float)
        run["k"], run["t"], run["r"] = s[:, 0], s[:, 1], s[:, 2]
    return runs


def time_fit(k: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    """Least-squares t(k) = a + b k + c k^2; returns (a, b, c)."""
    c, b, a = np.polyfit(k, t, 2)
    return float(a), float(b), float(c)


def bracket(run: dict, cap: int, window: int, tol: float) -> dict:
    """Iteration and time bracket of a run truncated after iteration ``cap``."""
    sel = run["k"] <= cap
    k, t, r = run["k"][sel], run["t"][sel], run["r"][sel]
    a, b, c = time_fit(k, t)
    w = k >= k[-1] - window
    s, q = np.polyfit(k[w], np.log(r[w]), 1)
    k_low = (math.log(tol) - q) / s
    p, qp = np.polyfit(np.log(k[w]), np.log(r[w]), 1)
    k_high = math.exp((math.log(tol) - qp) / p)

    def T(x: float) -> float:
        return a + b * x + c * x * x

    return {
        "cap": int(k[-1]),
        "k_low": k_low,
        "k_high": k_high,
        "t_low": T(k_low),
        "t_high": T(k_high),
        "p": p,
        "abc": (a, b, c),
    }


def slope(x: list[float], y: list[float]) -> tuple[float, float]:
    """Least-squares slope of log y over log x with its standard error."""
    lx, ly = np.log(np.array(x)), np.log(np.array(y))
    g, c0 = np.polyfit(lx, ly, 1)
    n = len(lx)
    if n < 3:
        return float(g), float("nan")
    res = ly - (g * lx + c0)
    se = math.sqrt(float(res @ res) / (n - 2) / float(((lx - lx.mean()) ** 2).sum()))
    return float(g), se


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", nargs="?", default=str(DEFAULT_FILE))
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    runs = read_runs(Path(args.file))
    by_l = {run["L"]: run for run in runs}

    print("1. Time fits t(k) = a + b k + c k^2 over all logged samples")
    print(f"{'L [um]':>7} {'2N':>9} {'a [s]':>8} {'b [s]':>8} {'matvec':>7} {'b/mv-1':>7} "
          f"{'c/2N [ns]':>9} {'24*2N/c [GB/s]':>14} {'rms [s]':>8}")
    for run in runs:
        a, b, c = time_fit(run["k"], run["t"])
        rms = float(np.sqrt(np.mean((run["t"] - (a + b * run["k"] + c * run["k"] ** 2)) ** 2)))
        print(f"{run['L']:7g} {run['n2']:9d} {a:8.1f} {b:8.4f} {run['matvec']:7.4f} "
              f"{b / run['matvec'] - 1:+7.1%} {c / run['n2'] * 1e9:9.3f} "
              f"{24 * run['n2'] / c / 1e9:14.1f} {rms:8.2f}")

    print("\n2. Back-test of the bracket on converged runs (window "
          f"{args.window} iterations, tol {args.tol:g})")
    for L, caps in BACK_TESTS:
        run = by_l[L]
        for cap in caps:
            br = bracket(run, cap, args.window, args.tol)
            over = br["k_low"] - run["it"]
            lo_ok = "ok" if over <= 0 else f"low overshoots by {over:.0f}"
            hi_ok = "ok" if br["k_high"] >= run["it"] else "high undershoots"
            print(f"  L = {L:g} um cap {cap}: it {br['k_low']:.0f} - {br['k_high']:.0f} "
                  f"(true {run['it']}; {lo_ok}, {hi_ok}), time {br['t_low']:.0f} - "
                  f"{br['t_high']:.0f} s (true {run['solve']:.0f} s), power p = {br['p']:.2f}")

    print("\n3. Projections of the capped runs")
    proj: dict[float, dict] = {}
    for L in CAPPED:
        run = by_l[L]
        br = bracket(run, run["it"], args.window, args.tol)
        proj[L] = br
        kry = [(k + 1) * run["n2"] * 16 / 1e9 for k in (br["k_low"], br["k_high"])]
        a, b, c = br["abc"]
        print(f"  L = {L:g} um (2N = {run['n2']}, cap {run['it']}): it {br['k_low']:.0f} - "
              f"{br['k_high']:.0f}, time {br['t_low']:.0f} - {br['t_high']:.0f} s "
              f"({br['t_low'] / 3600:.1f} - {br['t_high'] / 3600:.1f} h), Krylov basis "
              f"{kry[0]:.1f} - {kry[1]:.1f} GB; t(k) = {a:.1f} + {b:.4f} k + {c:.4e} k^2")

    print("\n4. Solve-time exponents")
    conv = [run for run in runs if run["conv"]]
    for label, lo in (("full range", 0.0), ("DoD window 2N >= 5e4", DOD_MIN)):
        xs = [run["n2"] for run in conv if run["n2"] >= lo]
        ys = [run["solve"] for run in conv if run["n2"] >= lo]
        g, se = slope(xs, ys)
        xp = xs + [by_l[L]["n2"] for L in CAPPED]
        gl, sel_ = slope(xp, ys + [proj[L]["t_low"] for L in CAPPED])
        gh, seh = slope(xp, ys + [proj[L]["t_high"] for L in CAPPED])
        print(f"  {label}: measured {g:.2f} +- {se:.2f} ({len(xs)} sizes); with projections "
              f"low {gl:.2f} +- {sel_:.2f} / high {gh:.2f} +- {seh:.2f}")
    ref = min((run for run in conv if run["n2"] >= DOD_MIN), key=lambda r: r["n2"])
    big = by_l[CAPPED[-1]]
    t_allowed = ref["solve"] * (big["n2"] / ref["n2"]) ** 2
    a, b, c = time_fit(big["k"], big["t"])
    k_allowed = (-b + math.sqrt(b * b - 4 * c * (a - t_allowed))) / (2 * c)
    print(f"  gamma = 2.0 from 2N = {ref['n2']} ({ref['solve']:.0f} s) allows {t_allowed:.0f} s at "
          f"2N = {big['n2']}: {k_allowed:.0f} iterations with its t(k), "
          f"{k_allowed / proj[CAPPED[-1]]['k_high']:.1f} x the high extrapolation")

    print("\n5. Orthogonalisation share and local slopes (converged runs)")
    for run in conv:
        share = (run["solve"] - run["it"] * run["matvec"]) / run["solve"]
        rep = f", reported CGS2 time {run['orth'] / run['solve']:.0%}" if run["orth"] else ""
        print(f"  L = {run['L']:g} um: (solve - it x matvec) / solve = {share:.0%}{rep}")
    for r0, r1 in zip(conv, conv[1:]):
        lx = math.log(r1["n2"] / r0["n2"])
        print(f"  L = {r0['L']:g} -> {r1['L']:g} um: local gamma_solve "
              f"{math.log(r1['solve'] / r0['solve']) / lx:.2f}, gamma_it "
              f"{math.log(r1['it'] / r0['it']) / lx:.2f}")


if __name__ == "__main__":
    main()
