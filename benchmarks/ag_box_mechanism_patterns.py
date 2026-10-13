"""Far-field difference patterns of stored box-study runs (WP-V3, record
benchmarks/results/ag_box_mechanism.md).

For pairs (A, B) of run directories written by specklebem_box_validity (complex cut amplitudes
cut_xz_F.npy / cut_yz_F.npy, and refl_F.npy where present), forms the difference dF = F_A - F_B
and prints its norm relative to |F_B| and the normalised complex correlation
|<dF_1, dF_2>| / (|dF_1| |dF_2|) between the differences of two pairs. A depth effect of the
continuous problem gives the same difference pattern in two discretisations of the box
(correlation near 1); numerical noise (GMRES tolerance, MLFMM error) gives uncorrelated
differences.

Usage:
  python benchmarks/ag_box_mechanism_patterns.py --pair A1 B1 --pair A2 B2 [--pair ...]
Every pair is compared with every other one (cuts xz and yz separately; the reflection
hemisphere, weighted by sqrt(sin theta), when all four runs have refl_F.npy).
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np


def load(run: str, name: str) -> np.ndarray | None:
    path = os.path.join(run, name + ".npy")
    return np.load(path) if os.path.exists(path) else None


def hemisphere_weights(rows: int) -> np.ndarray:
    theta = (np.arange(90) + 0.5) * math.pi / 180.0
    w = np.repeat(np.sqrt(np.sin(theta)), 120)
    if w.size != rows:
        raise SystemExit("refl_F.npy does not have the 90 x 120 hemisphere rows")
    return w[:, None]


def hemi_l2(t: np.ndarray, r: np.ndarray) -> float:
    """docs/05 metric of box_validity_support.hpp: sqrt(sum w (t - r)^2 / sum w r^2),
    w = sin theta."""
    w = np.sin((np.arange(t.shape[0]) + 0.5) * math.pi / 180.0)[:, None]
    return float(np.sqrt(np.sum(w * (t - r) ** 2) / np.sum(w * r**2)))


def averages(groups: list[list[str]]) -> None:
    """Intensity averages over groups of runs (e.g. several box depths) and the docs/05 metric
    between the averages of every two groups and between single runs of the groups."""
    means = []
    for g in groups:
        stack = [np.load(os.path.join(run, "refl_intensity.npy")) for run in g[1:]]
        means.append((g[0], np.mean(stack, axis=0), stack))
    for (n1, m1, s1), (n2, m2, s2) in itertools.combinations(means, 2):
        singles = [hemi_l2(x, y) for x in s1 for y in s2]
        print(f"average | {n1} ({len(s1)} runs) vs {n2} ({len(s2)} runs) | l2_refl of the averages "
              f"{hemi_l2(m1, m2):.4e} | single-run pairs: median {np.median(singles):.4e}, "
              f"max {max(singles):.4e}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    p.add_argument("--average", nargs="+", action="append", default=[], metavar="NAME DIR",
                   help="a named group of runs whose refl_intensity is averaged")
    a = p.parse_args(argv)
    if a.average:
        averages(a.average)
    if not a.pair:
        return 0
    names = [
        f"{os.path.basename(x.rstrip('/'))}-{os.path.basename(y.rstrip('/'))}" for x, y in a.pair
    ]
    for what in ("cut_xz_F", "cut_yz_F", "refl_F"):
        diffs, ok = [], True
        for x, y in a.pair:
            fx, fy = load(x, what), load(y, what)
            if fx is None or fy is None:
                ok = False
                break
            w = hemisphere_weights(fx.shape[0]) if what == "refl_F" else 1.0
            diffs.append(((fx - fy) * w, np.linalg.norm(fy * w)))
        if not ok:
            print(f"{what}: not available for every run, skipped")
            continue
        for n, (d, ref) in zip(names, diffs):
            print(f"{what} | {n} | |dF|/|F_B| {np.linalg.norm(d) / ref:.4e}")
        for (n1, (d1, _)), (n2, (d2, _)) in itertools.combinations(zip(names, diffs), 2):
            c = abs(np.vdot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2))
            print(f"{what} | corr({n1}, {n2}) {c:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
