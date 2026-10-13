"""Compare two full GMRES residual histories (WP-V3 experiment (v), record
benchmarks/results/ag_box_mechanism.md).

Reads HISTORY records (stride 1, written by specklebem_mlfmm_scaling --history-full-file) and
compares the record selected by --ref (e.g. the WP22b2 d0 = 3 run) with the one selected by
--test (the d0 = 5 rerun): residual ratio test/ref at every common iteration, the iteration
counts, and the plateaus of each history (windows of 100 iterations in which the residual drops
by less than 3 %, the WP22b2 staircase criterion), merged into intervals.

Usage:
  python benchmarks/ag_box_mechanism_histories.py REF_FILE TEST_FILE \
      --ref "material=ag L_um=3.2 box=50.000000" --ref-extra "d0=3 " "form=pmchwt jacobi=0" \
      --test-extra "d0=5 " "form=pmchwt jacobi=0"
"""

from __future__ import annotations

import argparse
import sys


def records(path: str, needles: list[str]) -> list[list[float]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("HISTORY") and all(n in line for n in needles):
                if "stride=1 " not in line:
                    raise SystemExit(f"{path}: record is not stride 1")
                out.append([float(x) for x in line.split("res=")[1].split(",")])
    return out


def plateaus(res: list[float], window: int = 100, drop: float = 0.03) -> list[tuple[int, int]]:
    flat = [k for k in range(len(res) - window) if res[k + window] > (1.0 - drop) * res[k]]
    merged: list[tuple[int, int]] = []
    for k in flat:
        if merged and k <= merged[-1][1]:
            merged[-1] = (merged[-1][0], k + window)
        else:
            merged.append((k, k + window))
    return merged


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref_file")
    p.add_argument("test_file")
    p.add_argument("--ref", required=True, help="substring common to both records")
    p.add_argument("--ref-extra", nargs="*", default=[])
    p.add_argument("--test-extra", nargs="*", default=[])
    a = p.parse_args(argv)
    ref = records(a.ref_file, [a.ref, *a.ref_extra])
    test = records(a.test_file, [a.ref, *a.test_extra])
    if len(ref) != 1 or len(test) != 1:
        raise SystemExit(f"need exactly one record each, found {len(ref)} / {len(test)}")
    r, t = ref[0], test[0]
    n = min(len(r), len(t))
    ratios = [t[k] / r[k] for k in range(1, n)]
    print(f"iterations ref {len(r) - 1} test {len(t) - 1} (common {n - 1})")
    print(f"residual ratio test/ref over the common iterations: min {min(ratios):.5f} "
          f"max {max(ratios):.5f} (largest |ratio - 1| at it "
          f"{1 + max(range(len(ratios)), key=lambda i: abs(ratios[i] - 1.0))})")
    for k in (50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000):
        if k < n:
            print(f"it {k:5d}: ref {r[k]:.4e} test {t[k]:.4e} ratio {t[k] / r[k]:.5f}")
    for name, h in (("ref", r), ("test", t)):
        pl = plateaus(h)
        desc = ", ".join(f"{a0}-{b0} ({h[a0]:.2e} -> {h[b0]:.2e})" for a0, b0 in pl)
        print(f"plateaus {name} (< 3 % drop per 100 it): {sum(b0 - a0 for a0, b0 in pl)} it in "
              f"{len(pl)} intervals: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
