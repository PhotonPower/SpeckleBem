"""Extrapolation of the WP22d Si large-area proxy (record benchmarks/results/si_large_area.md).

Reads the ``ROW`` lines of ``benchmarks/results/mlfmm_scaling_rows.txt`` (the WP22b3 Si
uniform-box series, sigma = 50 nm, and the WP22d rows with ``series=wp22d``, sigma = 100 nm) and
the ``EST`` lines of ``benchmarks/results/si_large_area_estimates.txt`` (``--estimate-only``
output for the targets), and extrapolates the proxy (the largest WP22d row) to every target with
two methods:

A. Power law in 2N: Q_target = Q_proxy (2N_target / 2N_proxy)^gamma for the setup, the matvec,
   the solve time and the peak memory; gamma bounds = the union of gamma +- 2 standard errors of
   the sigma = 50 nm uniform-box series and of the sigma = 100 nm rows (when >= 3 sizes; with the
   3 WP22d rows that fit has 1 degree of freedom, so its standard errors are themselves
   uncertain). Blind to
   the leaf size (the octree quantisation), which the series covers only between 125 and 188 nm,
   and to the box (uniform box only).
B. Component model: the target's estimated near-field and far-operator memory (the estimator
   reproduced the measured near / far memory of every row to < 1 %) times the measured cost per
   GB of the rows: far matvec s/GB from the proxy (low) to 1.4 x the proxy (high: one more octree
   level raised it 1.3-1.4 x between 6 and 7 levels), near matvec and setup s/GB over all rows;
   iterations 0.9 ... 1.3 x the proxy, times the graded / uniform iteration ratio of the WP22b3
   rows at equal L (1.39-1.46) for a graded-box target; solve = it x matvec x (solve / (it x
   matvec)) of the rows (the CGS2 orthogonalisation share); peak memory = (peak - Krylov) / model
   of the rows times the target's model plus its Krylov basis.

The record quotes method B for the memory and the setup (method A cannot see the growth of the
near field and the Si patterns with the leaf: at L >= 25 um it falls below the 0.72 x model floor
that every measured row satisfied) and the union of A and B for the matvec and the solve.

Every figure is an extrapolation for the development machine (Ryzen 9 5900X, 12 cores / 24
threads, 2-channel DDR4), not a measurement.

Usage: python benchmarks/si_large_area_extrapolation.py
           [rows file] [estimates file]
Python 3, standard library only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS = ROOT / "results" / "mlfmm_scaling_rows.txt"
ESTS = ROOT / "results" / "si_large_area_estimates.txt"


def parse(path: Path, prefix: str) -> list[dict[str, str]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                out.append(dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv))
    return out


def fl(r: dict[str, str], k: str) -> float:
    return float(r[k])


def slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    g = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    c = my - g * mx
    rss = sum((b - (c + g * a)) ** 2 for a, b in zip(lx, ly))
    return g, math.sqrt(rss / (n - 2) / sxx) if n > 2 else 0.0


def hours(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3570 else f"{s / 60:.0f} min"


def span(r: tuple[float, float]) -> str:
    """'7–9 min', '1.0–2.2 h' or '34 min–1.1 h' for a (low, high) pair of seconds."""
    lo, hi = hours(r[0]), hours(r[1])
    if lo.split()[1] == hi.split()[1]:
        return f"{lo.split()[0]}–{hi}"
    return f"{lo}–{hi}"


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # en dashes in the Markdown table
    rows_path = Path(argv[0]) if argv else ROWS
    ests_path = Path(argv[1]) if len(argv) > 1 else ESTS
    rows = parse(rows_path, "ROW ")
    uniform = [
        r for r in rows
        if r["material"] == "si" and r["box"].startswith("50")
        and r.get("beam") == "angular_spectrum" and r.get("conv") == "1"
    ]
    series = [r for r in uniform if float(r.get("sigma_nm", "50")) == 50.0]
    wp22d = [r for r in uniform if r.get("series") == "wp22d"]
    if len(series) < 3 or not wp22d:
        print("need the WP22b3 uniform series and at least one WP22d row", file=sys.stderr)
        return 1
    proxy = max(wp22d, key=lambda r: fl(r, "N2"))
    pts = series + wp22d

    print(f"series: {len(series)} sigma = 50 nm rows, {len(wp22d)} WP22d rows; proxy L = "
          f"{proxy['L_um']} um, 2N = {proxy['N2']}, sigma {proxy['sigma_nm']} nm")
    # Exponent bounds: gamma +- 2 se of the sigma = 50 nm series, widened to the sigma = 100 nm
    # rows' gamma +- 2 se when they have >= 3 sizes.
    gam: dict[str, tuple[float, float]] = {}
    for k in ("setup_s", "matvec_s", "solve_s", "peak_gb", "it"):
        g, se = slope([fl(r, "N2") for r in series], [fl(r, k) for r in series])
        lo, hi = g - 2 * se, g + 2 * se
        msg = f"  gamma {k}: sigma 50 nm {g:.2f} +- {se:.2f}"
        if len(wp22d) >= 3:
            g2, se2 = slope([fl(r, "N2") for r in wp22d], [fl(r, k) for r in wp22d])
            lo, hi = min(lo, g2 - 2 * se2), max(hi, g2 + 2 * se2)
            msg += (f", sigma 100 nm {g2:.2f} +- {se2:.2f} ({len(wp22d)} points, "
                    f"{len(wp22d) - 2} dof)")
        gam[k] = (lo, hi)
        print(f"{msg} -> bounds {lo:.2f} ... {hi:.2f}")

    ratio_mem = [(fl(r, "peak_gb") - fl(r, "krylov_gb")) / fl(r, "peak_est_gb") for r in pts]
    near_mv = [fl(r, "near_mv_s") / fl(r, "near_gb") for r in pts]
    near_set = [fl(r, "near_s") / fl(r, "near_gb") for r in pts]
    far_set = [fl(r, "far_s") / fl(r, "far_gb") for r in pts]
    solve_over = [fl(r, "solve_s") / (fl(r, "it") * fl(r, "matvec_s")) for r in pts]
    far_mv_proxy = fl(proxy, "far_mv_s") / fl(proxy, "far_gb")
    print(f"  (peak - Krylov) / model {min(ratio_mem):.2f} ... {max(ratio_mem):.2f}; far matvec "
          f"{far_mv_proxy:.3f} s/GB (proxy); near matvec {min(near_mv):.3f} ... "
          f"{max(near_mv):.3f} s/GB; near setup {min(near_set):.1f} ... {max(near_set):.1f} s/GB; "
          f"far setup {min(far_set):.2f} ... {max(far_set):.2f} s/GB; solve / (it matvec) "
          f"{min(solve_over):.2f} ... {max(solve_over):.2f}")

    # Graded (lambda1/5 = 100 nm) vs uniform box: iterations at equal L (WP22b3, sigma = 50 nm).
    graded = [
        r for r in rows
        if r["material"] == "si" and r["box"].startswith("100")
        and r.get("beam") == "angular_spectrum" and r.get("conv") == "1"
        and float(r.get("sigma_nm", "50")) == 50.0
    ]
    it_uniform = {r["L_um"]: fl(r, "it") for r in series}
    graded_ratio = [fl(r, "it") / it_uniform[r["L_um"]] for r in graded if r["L_um"] in it_uniform]
    if not graded_ratio:
        print("need WP22b3 graded-box rows at the uniform-box sizes", file=sys.stderr)
        return 1
    print(f"  graded / uniform iterations at equal L (sigma 50 nm): {min(graded_ratio):.2f} ... "
          f"{max(graded_ratio):.2f}")

    it_p, n2_p = fl(proxy, "it"), fl(proxy, "N2")
    print("\n| target (σ = 100 nm) | 2N | leaf [nm] | model [GB] | peak A [GB] | peak B [GB] | "
          "setup A / B | matvec A / B [s] | it (B) | solve A | solve B |")
    print("|---|---:|---:|---:|---:|---:|---|---|---:|---|---|")
    for t in parse(ests_path, "EST "):
        if (t["material"] != "si" or float(t["L_um"]) <= float(proxy["L_um"])
                or float(t["sigma_nm"]) != float(proxy["sigma_nm"])):
            continue
        n2 = fl(t, "N2")
        x = n2 / n2_p
        uniform_box = float(t["box"]) == float(proxy["box"])

        def law(k: str) -> tuple[float, float]:
            lo, hi = gam[k]
            return fl(proxy, k) * x**lo, fl(proxy, k) * x**hi

        far_gb = fl(t, "pattern_gb") + fl(t, "far_table_gb")
        near_gb = fl(t, "near_gb")
        model = fl(t, "peak_setup_est_gb")
        it_lo, it_hi = 0.9 * it_p, 1.3 * it_p
        if not uniform_box:
            it_lo, it_hi = it_lo * min(graded_ratio), it_hi * max(graded_ratio)
        peak_b = (min(ratio_mem) * model + (it_lo + 1) * n2 * 16e-9,
                  max(ratio_mem) * model + (it_hi + 1) * n2 * 16e-9)
        mv_b = (far_gb * far_mv_proxy + near_gb * min(near_mv),
                far_gb * 1.4 * far_mv_proxy + near_gb * max(near_mv))
        solve_b = (it_lo * mv_b[0] * min(solve_over), it_hi * mv_b[1] * max(solve_over))
        setup_b = (near_gb * min(near_set) + far_gb * min(far_set),
                   near_gb * max(near_set) + far_gb * max(far_set))
        box = "uniform 50 nm" if uniform_box else f"graded {float(t['box']):g} nm"
        name = f"L = {float(t['L_um']):g} µm, {box}, q = {float(t['lq']):g}"
        if uniform_box:
            pa, sa, ma, va = law("peak_gb"), law("setup_s"), law("matvec_s"), law("solve_s")
            a = (f"{pa[0]:.0f}–{pa[1]:.0f}", span(sa), f"{ma[0]:.0f}–{ma[1]:.0f}", span(va))
        else:  # the power law of the uniform-box series does not apply to another box
            a = ("—", "—", "—", "—")
        n2s = f"{int(n2):,}".replace(",", " ")
        models = f"{model:,.0f}".replace(",", " ")
        print(f"| {name} | {n2s} | {float(t['leaf_nm']):.0f} | {models} | {a[0]} "
              f"| {peak_b[0]:.0f}–{peak_b[1]:.0f} | {a[1]} / "
              f"{span(setup_b)} | {a[2]} / "
              f"{mv_b[0]:.0f}–{mv_b[1]:.0f} | {it_lo:.0f}–{it_hi:.0f} | "
              f"{a[3]} | {span(solve_b)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
