"""Calibration of the WP-S2 speckle analysis on synthetic far-zone speckle (no solves).

Repeats the analysis of ``benchmarks/speckle_sweep_stats.py``
(``specklebem.speckle_sweep.analyse_ensemble`` per sigma, two seeds) on synthetic patterns of the
sweep's plane, Gaussian spot and sampling, with fixed seeds:

- ``--model gaussian``: exact fully developed (circular Gaussian) speckle with a Gaussian halo of
  rms ``s`` in u (``synthetic_far_zone_speckle``, one entry of ``--halos`` per "sigma"): the null
  distribution of the normalised contrast, the speckle-size error, the KS false-rejection rates
  (per halo with the harness stride, pooled at 2 and at 1 speckle size) and the seed-to-seed
  correlation of independent patterns. With ``--map-correlation mu`` the entries are not
  independent but reuse the two seeds' random fields as the sweep reuses its two height maps at
  every sigma: each seed's aperture is a circular Gaussian field that is carried from one entry
  to the next as mu a + sqrt(1 - mu^2) b (``correlated_field``; intensity correlation mu^2
  between adjacent entries, mu = 1: the same field at every entry). This is the null of the
  sigma-pooled gate with the sweep's dependence (sweep: sigma-step rho_env 0.05-0.54).
- ``--model phase-screen``: the Kirchhoff thin-phase-screen surrogate of the sweep
  (``phase_screen_far_zone_speckle``, Gaussian heights with correlation exp(-r^2 / Lc^2), the
  same unit map per seed scaled by sigma): coherent part and finite number of illuminated
  correlation cells included, so the expected values of every number in the record (plane
  contrast for row 43, its monotonicity, KS rejection rates, seed and sigma-step correlations).

An ensemble whose speckle size cannot be measured (``speckle_size`` raises: |mu_A|^2 stays above
one half inside the window, e.g. one dominant glint lobe) is counted as a failure and left out
of that rep's statistics. Sample sizes n are reported as the mean and the range over the
repetitions (they vary with the window, which follows the realised envelope).

Usage::

    python benchmarks/speckle_sweep_calibration.py --pythonpath build/win-release/python \\
        --model gaussian --halos 0.20,0.28,0.35,0.45 --reps 300
    python benchmarks/speckle_sweep_calibration.py --pythonpath build/win-release/python \\
        --model gaussian --map-correlation 0.62 --reps 300
    python benchmarks/speckle_sweep_calibration.py --pythonpath build/win-release/python \\
        --model phase-screen --sigmas 100,150,200,250 --reps 200
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _n_range(values: list[int]) -> str:
    """'mean [min-max]' of sample sizes over the repetitions ('–' if none)."""
    if not values:
        return "–"
    lo, hi = min(values), max(values)
    mean = sum(values) / len(values)
    return f"{mean:.0f}" if lo == hi else f"{mean:.0f} [{lo}–{hi}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pythonpath", default=None)
    ap.add_argument("--model", choices=["gaussian", "phase-screen"], default="gaussian")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--halos", default="0.20,0.28,0.35,0.45",
                    help="gaussian model: halo rms widths in u (comma separated)")
    ap.add_argument("--sigmas", default="100,150,200,250",
                    help="phase-screen model: rms heights [nm]")
    ap.add_argument("--lc", type=float, default=1e-6, help="correlation length [m] (1/e)")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--envelope-radius", type=float, default=0.5)
    ap.add_argument("--waist", type=float, default=8e-6 / 6, help="beam waist w0 [m] (L / 6)")
    ap.add_argument("--plane-n", type=int, default=161, help="points per axis of the 4 mm plane")
    ap.add_argument("--n-fft", type=int, default=256, help="far-zone FFT size of the generators")
    ap.add_argument("--map-correlation", type=float, default=None,
                    help="gaussian model: field correlation mu between adjacent entries of one "
                         "seed (default: independent draws)")
    args = ap.parse_args(argv)
    if args.map_correlation is not None:
        if args.model != "gaussian":
            ap.error("--map-correlation applies to --model gaussian only")
        if not 0.0 <= args.map_correlation <= 1.0:
            ap.error("--map-correlation must lie in [0, 1]")
    if args.pythonpath:
        sys.path.insert(0, str(Path(args.pythonpath).resolve()))
    import numpy as np

    import specklebem as sb
    from specklebem import speckle
    from specklebem import speckle_sweep as ss

    plane = ss.plane_sampling(500e-9, -1e-3, args.waist, 4e-3, args.plane_n)
    wl = plane.wavelength
    phase_screen = args.model == "phase-screen"
    keys = [float(s) for s in (args.sigmas if phase_screen else args.halos).split(",")]
    rng = np.random.default_rng(args.seed)
    correlated = args.map_correlation is not None
    acc = {s: dict(c=[], cp=[], size=[], ks_rej=0, ks_valid=0, rho_p=[], rho_e=[], half=[],
                   n2=[], n1=[], spacing=None, fail=0) for s in keys}
    steps = {(a, b): dict(rp=[], re=[]) for a, b in zip(keys, keys[1:])}
    gate = dict(rej2=0, rej1=0, n2=[], n1=[], valid2=0, valid1=0)
    mono = dict(c=0, rho=0, valid=0)
    for _ in range(args.reps):
        pooled2, pooled1, res_by = [], [], {}
        if phase_screen:
            screens = [ss.unit_height_screen(rng, plane, args.lc, n_fft=args.n_fft)
                       for _ in range(2)]
        elif correlated:
            fields = [ss.circular_gaussian_field(rng, args.n_fft) for _ in range(2)]
        for e, s in enumerate(keys):
            if phase_screen:
                pats = [ss.phase_screen_far_zone_speckle(h, s * 1e-9, plane) for h in screens]
                fd = ss.fully_developed(s * 1e-9, wl)
            elif correlated:
                if e > 0:
                    fields = [ss.correlated_field(rng, a, args.map_correlation) for a in fields]
                pats = [ss.synthetic_far_zone_speckle(None, plane, envelope_rms=s,
                                                      n_fft=args.n_fft, aperture=a)
                        for a in fields]
                fd = True
            else:
                pats = [ss.synthetic_far_zone_speckle(rng, plane, envelope_rms=s,
                                                      n_fft=args.n_fft) for _ in range(2)]
                fd = True
            a = acc[s]
            try:
                res = ss.analyse_ensemble(pats, plane, envelope_radius=args.envelope_radius)
            except RuntimeError:
                a["fail"] += 1
                res_by[s] = None
                continue
            res_by[s] = dict(res=res, pats=pats)
            st = res["stats"]
            a["half"].append(res["window_u"])
            a["n2"].append(sum(x.size for x in res["ks_samples"]))
            a["n1"].append(sum(x.size for x in res["ks_samples_one"]))
            a["c"].extend(st["contrast"].tolist())
            a["cp"].append(float(np.mean(res["contrast_plane"])))
            a["size"].append(st["speckle_size_rel_error"])
            if res["ks_spacing_used"]:
                a["ks_valid"] += 1
                a["ks_rej"] += int(st["ks_rejects_5pct"])
                a["spacing"] = res["ks_spacing_used"]
            a["rho_p"].append(res["rho_plane"][(0, 1)])
            a["rho_e"].append(float(sb.correlation_coefficient(res["windows"][0].ravel(),
                                                               res["windows"][1].ravel())))
            if fd:
                pooled2.extend(res["ks_samples"])
                pooled1.extend(res["ks_samples_one"])
        if phase_screen or correlated:
            for (s1, s2), d in steps.items():
                if res_by[s1] is None or res_by[s2] is None:
                    continue
                r1, r2 = res_by[s1]["res"], res_by[s2]["res"]
                half = min(r1["window_half"], r2["window_half"])
                for i in range(2):
                    d["rp"].append(float(sb.correlation_coefficient(
                        res_by[s1]["pats"][i].ravel(), res_by[s2]["pats"][i].ravel())))
                    d["re"].append(float(sb.correlation_coefficient(
                        ss.centre_crop(r1["normalised"][i], half).ravel(),
                        ss.centre_crop(r2["normalised"][i], half).ravel())))
        for name, pool in (("2", pooled2), ("1", pooled1)):
            n = sum(x.size for x in pool)
            if n >= 50:
                gate["n" + name].append(n)
                gate["valid" + name] += 1
                gate["rej" + name] += int(ss.pooled_ks(pool).rejects(0.05))
        if all(res_by[s] is not None for s in keys):
            mono["valid"] += 1
            mono["c"] += int(speckle.monotone_decreasing([acc[s]["cp"][-1] for s in keys]))
            mono["rho"] += int(speckle.monotone_decreasing([acc[s]["rho_e"][-1] for s in keys]))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    label = "σ [nm]" if phase_screen else "halo rms (u)"
    print(f"model {args.model}, reps {args.reps}, envelope radius {args.envelope_radius} "
          f"speckles, seed {args.seed}, w0 {args.waist:g} m, plane n {args.plane_n}"
          + (f", Lc {args.lc:g} m" if phase_screen else "")
          + (f", map correlation mu {args.map_correlation:g} (intensity "
             f"{args.map_correlation ** 2:.3f})" if correlated else ""))
    print(f"| {label} | failures | window ±u | KS harness: spacing, rejection rate "
          "| n (2 / 1 speckle), mean [range] | C co-pol plane: seed mean ± std | C env ± std "
          "| size err ± std | ρ seeds plane | ρ seeds env |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in keys:
        a = acc[s]
        if not a["c"]:
            print(f"| {s:g} | {a['fail']} | – | – | – | – | – | – | – | – |")
            continue
        c, cp, z = np.array(a["c"]), np.array(a["cp"]), np.array(a["size"])
        rp, re_ = np.array(a["rho_p"]), np.array(a["rho_e"])
        lev = (f"{a['spacing']:g}, {a['ks_rej'] / a['ks_valid']:.3f}" if a["ks_valid"]
               else "–")
        print(f"| {s:g} | {a['fail']} | {np.mean(a['half']):.3f} | {lev} "
              f"| {_n_range(a['n2'])} / {_n_range(a['n1'])} "
              f"| {cp.mean():.2f} ± {cp.std():.2f} | {c.mean():.3f} ± {c.std():.3f} "
              f"| {z.mean():+.3f} ± {z.std():.3f} | {rp.mean():.3f} ± {rp.std():.3f} "
              f"| {re_.mean():+.3f} ± {re_.std():.3f} |")
    for name in ("2", "1"):
        if gate["valid" + name]:
            print(f"pooled KS over the fully developed entries at {name} speckle size(s): "
                  f"n {_n_range(gate['n' + name])}, rejection rate "
                  f"{gate['rej' + name] / gate['valid' + name]:.3f} ({gate['valid' + name]} reps)")
    if mono["valid"]:
        print(f"monotone decreasing over the entries ({mono['valid']} complete reps): plane "
              f"contrast (seed means) {mono['c'] / mono['valid']:.3f}, ρ seeds env "
              f"{mono['rho'] / mono['valid']:.3f}")
    if phase_screen or correlated:
        k = 2 * math.pi / wl
        if phase_screen:
            print("| σ step [nm] | ρ plane ± std | ρ env ± std | exp(−(2kΔσ)²) |")
        else:
            print("| halo step (u) | ρ plane ± std | ρ env ± std | μ² |")
        print("|---|---|---|---|")
        for (s1, s2), d in steps.items():
            if not d["rp"]:
                continue
            rp, re_ = np.array(d["rp"]), np.array(d["re"])
            expected = (math.exp(-((2 * k * (s2 - s1) * 1e-9) ** 2)) if phase_screen
                        else args.map_correlation ** 2)
            print(f"| {s1:g} → {s2:g} | {rp.mean():.3f} ± {rp.std():.3f} "
                  f"| {re_.mean():+.3f} ± {re_.std():.3f} | {expected:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
