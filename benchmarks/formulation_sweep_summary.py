#!/usr/bin/env python3
"""Summary of the WP22b2 MLFMM formulation sweep (benchmarks/results/formulation_convergence.md).

Reads the ``HISTORY`` lines that ``specklebem_mlfmm_scaling --history-file`` appends (one per
solve: case keys, ``form=``, ``jacobi=``, ``it=``, ``conv=``, ``stride=``, ``res=r_0,r_s,...``)
and, optionally, the ``ROW`` lines of the same runs (``--rows``: times, 2N, memory, the true
residual), and prints

* the iteration table per material x box x L x formulation (iterations, converged, final
  monitored residual, true residual of the formulation's own unpreconditioned system, solve time
  and s/it from the ROW lines when given),
* residual snapshots at selected iterations,
* the Jacobi identity check: max_k |r_k - r_k^ref| / r_k^ref between the Jacobi runs of one
  material, box, L and waist (reference: the PMCHWT + Jacobi run), from full-precision records
  (``--history-full-file``, stride 1, 17 digits) when given with ``--full``,
* for capped runs (conv=0) the residual reached and the decrease over the last 500 iterations.

A run is identified by (material, box, L, waist factor, formulation, Jacobi). Records written
before the ``waist=`` key existed (the 40 original WP22b2 HISTORY lines) count as waist 4, the
waist of all those runs. Two records with the same key are an error (a re-run must replace the
old line, not shadow it silently). The output is plain ASCII.

Usage:
    python benchmarks/formulation_sweep_summary.py \\
        benchmarks/results/formulation_convergence_mlfmm_histories.txt \\
        [--rows benchmarks/results/formulation_convergence_mlfmm_rows.txt] \\
        [--full benchmarks/results/formulation_convergence_mlfmm_histories_full.txt]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

SNAPSHOTS = (10, 50, 100, 200, 500, 1000, 2000, 3000, 4000)
ORDER = {("pmchwt", "0"): 0, ("ictf", "0"): 1, ("mctf", "0"): 2, ("pmchwt", "1"): 3,
         ("ictf", "1"): 4, ("mctf", "1"): 5}
DEFAULT_WAIST = "4"  # records without waist= (all original WP22b2 runs used w0 = L/4)


def parse_kv(line: str, prefix: str) -> dict[str, str]:
    out = {}
    for tok in line[len(prefix):].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def read(path: str, prefix: str) -> list[dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                rows.append(parse_kv(line.strip(), prefix))
    return rows


def box_name(r: dict[str, str]) -> str:
    try:
        return f"{float(r['box']):.0f} nm"
    except ValueError:
        return r["box"]


def waist(r: dict[str, str]) -> str:
    return f"{float(r.get('waist', DEFAULT_WAIST)):g}"


def key(r: dict[str, str]) -> tuple:
    return (r["material"], box_name(r), float(r["L_um"]), waist(r), r["form"], r["jacobi"])


def index(records: list[dict[str, str]], what: str) -> dict[tuple, dict[str, str]]:
    """Records by key(); raises ValueError on a duplicate key instead of keeping the last one."""
    out: dict[tuple, dict[str, str]] = {}
    for r in records:
        k = key(r)
        if k in out:
            raise ValueError(f"duplicate {what} record for {label(k)}")
        out[k] = r
    return out


def history_indices(it: int, stride: int) -> list[int]:
    """Iterations of the stored residuals: 0, stride, 2 stride, ... and always `it` (as the C++
    mlfmm_scaling::history_indices)."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    ks = list(range(0, it + 1, stride))
    if ks[-1] != it:
        ks.append(it)
    return ks


def history(r: dict[str, str]) -> tuple[list[int], list[float]]:
    res = [float(x) for x in r["res"].split(",")]
    ks = history_indices(int(r["it"]), int(r["stride"]))
    if len(ks) != len(res):
        raise ValueError(f"history length mismatch in {label(key(r))}: {len(res)} values, "
                         f"{len(ks)} expected")
    return ks, res


def value_at(ks: list[int], res: list[float], k: int) -> float | None:
    """Residual at iteration k (exact sample or log-linear interpolation), None beyond the end."""
    if k > ks[-1]:
        return None
    for i, kk in enumerate(ks):
        if kk == k:
            return res[i]
        if kk > k:
            a, b = ks[i - 1], kk
            t = (k - a) / (b - a)
            return math.exp((1 - t) * math.log(res[i - 1]) + t * math.log(res[i]))
    return None


def form_name(k: tuple) -> str:
    return k[4].upper() + (" + Jacobi" if k[5] == "1" else "")


def label(k: tuple) -> str:
    w = "" if k[3] == DEFAULT_WAIST else f" w0 = L/{k[3]}"
    return f"{k[0]} {k[1]} L = {k[2]:g} um{w} {form_name(k)}"


def sort_key(k: tuple) -> tuple:
    return (k[0], k[1], k[2], k[3], ORDER.get((k[4], k[5]), 9))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("histories")
    ap.add_argument("--rows", help="file(s) with the ROW lines of the same runs", nargs="*")
    ap.add_argument("--full", help="full-precision HISTORY records (Jacobi identity check)")
    a = ap.parse_args(argv)

    hist = index(read(a.histories, "HISTORY "), "HISTORY")
    row_records = []
    for p in a.rows or []:
        row_records += [r for r in read(p, "ROW ") if "form" in r]
    rows = index(row_records, "ROW")

    print("| Material | Box | L [um] | 2N | Formulation | Iterations | Converged | Residual "
          "| True res. | Solve [s] | s/it | Setup [s] | Peak [GB] |")
    print("|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for k in sorted(hist, key=sort_key):
        r = hist[k]
        ks, res = history(r)
        w = rows.get(k, {})
        solve = float(w["solve_s"]) if w else float("nan")
        it = int(r["it"])
        true_res = float(w.get("true_res", "nan"))
        print(f"| {k[0]} | {k[1]} | {k[2]:g} | {r['N2']} | {form_name(k)} | {it} | "
              f"{'yes' if r['conv'] == '1' else '**no (cap)**'} | {res[-1]:.3g} | "
              f"{true_res:.3g} | {solve:.0f} | {solve / max(it, 1):.3f} | "
              f"{float(w.get('setup_s', 'nan')):.0f} | {float(w.get('peak_gb', 'nan')):.1f} |")

    print("\nMonitored residual at selected iterations:\n")
    print("| Run | " + " | ".join(str(s) for s in SNAPSHOTS) + " |")
    print("|---|" + "---:|" * len(SNAPSHOTS))
    for k in sorted(hist, key=sort_key):
        ks, res = history(hist[k])
        cells = []
        for s in SNAPSHOTS:
            v = value_at(ks, res, s)
            cells.append("-" if v is None else f"{v:.2e}")
        print(f"| {label(k)} | " + " | ".join(cells) + " |")

    capped = [k for k in sorted(hist, key=sort_key) if hist[k]["conv"] != "1"]
    if capped:
        print("\nCapped runs (residual reached, decrease factor over the last 500 iterations):\n")
        for k in capped:
            ks, res = history(hist[k])
            back = value_at(ks, res, max(0, ks[-1] - 500))
            print(f"- {label(k)}: {res[-1]:.3e} after {ks[-1]} iterations, "
                  f"x{res[-1] / back:.3f} over the last 500")

    if a.full:
        full = defaultdict(dict)
        for k, r in index(read(a.full, "HISTORY "), "full-precision HISTORY").items():
            if r["jacobi"] == "1":
                full[k[:4]][r["form"]] = [float(x) for x in r["res"].split(",")]
        print("\nJacobi identity check (max_k |r_k - r_k^PMCHWT| / r_k^PMCHWT):\n")
        for g, forms in sorted(full.items()):
            ref = forms.get("pmchwt")
            if ref is None or len(forms) < 2:
                continue
            w = "" if g[3] == DEFAULT_WAIST else f" w0 = L/{g[3]}"
            for f, h in sorted(forms.items()):
                if f == "pmchwt":
                    continue
                n = min(len(h), len(ref))
                d = max(abs(h[i] - ref[i]) / ref[i] for i in range(n))
                print(f"- {g[0]} {g[1]} L = {g[2]:g} um{w}: {f.upper()} vs PMCHWT {d:.2e} "
                      f"(iterations {len(h) - 1} / {len(ref) - 1})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
