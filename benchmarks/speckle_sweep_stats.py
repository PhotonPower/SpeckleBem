"""Speckle statistics of the WP-S2 sweep with the WP-S1 harness (docs/05 rows 42-44).

Reads the case directories written by ``benchmarks/speckle_sweep.py`` (``E.npy``,
``meta.json``), groups them by sigma and evaluates every case with
``specklebem.speckle_sweep.analyse_ensemble`` (co-polarised intensity |E_x|^2, resampled to
direction cosines, leave-one-out envelope normalisation, centred window, then
``specklebem.speckle.ensemble_statistics``). Which number gates which row:

- row 42 (PDF; docs/05, decision c of 2026-10-30): per fully developed sigma (coherent fraction
  exp(-(2 k sigma)^2) < 1e-2, WP-S2b lead rule), ``ks_test_exponential`` on the decimated
  (``--ks-spacing``, default 2 speckle sizes) I / <I> of the windows pooled over the seeds,
  evaluable when n >= 50, decision ``KsResult.rejects(0.05)`` (Stephens), plain Kolmogorov p
  alongside: JSON ``sigmas[i].ks_spacing2`` / ``ks_spacing2_n``, summarised under
  ``row42_per_sigma`` and printed as the "docs/05 row 42 per sigma" lines (``--markdown``:
  stdout; otherwise stderr, so that the JSON on stdout stays valid). The same samples pooled
  over all fully developed sigma as well (``sigma_pooled_ks``) are supplementary, not the
  docs/05 gate (the maps are the same at every sigma); so is the 1-speckle pool
  (``pooled_ks_one``).
- row 43 (contrast vs sigma): ``speckle_contrast`` of the raw co-polarised plane pattern (whole
  4 x 4 mm plane), seed mean per sigma, ``speckle.monotone_decreasing``. gamma_12: see the
  record (open definition); the seed-to-seed and sigma-step ``correlation_coefficient`` values
  are printed.
- row 44 (speckle size): ``field_correlation_from_intensity`` + ``speckle_size`` on the
  normalised windows (pixel du |z|, plane units on axis), mean over u, v and seeds, against
  ``speckle_size_expected(lambda, |z|, 2 w0, "gaussian")``; judged (``size_within_10pct``) for
  the fully developed sigma only, the others are reported.

``--sensitivity`` adds the row-42 sensitivity study: the sigma-pooled pool (supplementary, not the
docs/05 gate) recomputed with one analysis parameter changed at a time (``SENSITIVITY_VARIANTS``:
envelope radius, window threshold, (u, v) samples per speckle, decimation offset, KS spacing, the
smallest fully developed sigma left out, u_max), as a Markdown table and under the JSON key
``sensitivity``.

Usage::

    python benchmarks/speckle_sweep_stats.py <OUT>/cases --pythonpath build/win-release/python \\
        [--markdown] [--sensitivity] [--json summary.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", help="directory holding the case subdirectories")
    ap.add_argument("--pythonpath", default=None, help="directory put first on sys.path")
    ap.add_argument("--u-max", type=float, default=0.66, help="|u|, |v| limit of the (u, v) grid")
    ap.add_argument("--samples", type=float, default=8.0, help="(u, v) samples per lambda / D")
    ap.add_argument("--envelope-radius", type=float, default=0.5,
                    help="envelope smoothing radius (1/e) in speckle sizes")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="window: mean envelope >= threshold x its maximum")
    ap.add_argument("--ks-spacing", type=float, default=2.0, help="KS decimation [speckle sizes]")
    ap.add_argument("--markdown", action="store_true", help="print Markdown tables")
    ap.add_argument("--sensitivity", action="store_true",
                    help="sigma-pooled row-42 pool (supplementary, not the docs/05 gate) under "
                    "one-at-a-time changes of the analysis parameters")
    ap.add_argument("--json", default=None, help="write a JSON summary to this file")
    return ap


def _ks_dict(r) -> dict:
    return dict(n=int(r.n), statistic=float(r.statistic), p=float(r.p_value),
                modified=float(r.modified_statistic), rejects=bool(r.rejects(0.05)))


def load_cases(cases_dir: Path) -> list[dict]:
    """Finished case directories (``meta.json`` and ``E.npy``) with their co-polarised
    intensity, sorted by directory name."""
    import numpy as np

    from specklebem import speckle_sweep as ss

    cases = []
    for d in sorted(cases_dir.iterdir()):
        if not ((d / "meta.json").is_file() and (d / "E.npy").is_file()):
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        pl = meta["plane"]
        plane = ss.plane_sampling(pl["wavelength"], pl["z"], pl["waist"], pl["size"], pl["n"])
        cases.append(dict(name=d.name, rec=meta["record"], plane=plane,
                          I=ss.copol_intensity(np.load(d / "E.npy"), plane.n)))
    if not cases:
        raise SystemExit("no finished cases")
    return cases


def analyse(cases_dir: Path, args, cases: list[dict] | None = None) -> dict:
    import numpy as np

    import specklebem as sb
    from specklebem import speckle
    from specklebem import speckle_sweep as ss

    if cases is None:
        cases = load_cases(cases_dir)
    plane = cases[0]["plane"]
    wl = plane.wavelength
    k = 2 * math.pi / wl
    sigmas = sorted({c["rec"]["sigma_nm"] for c in cases})
    out = dict(cases=[], sigmas=[], corr_seed=[], corr_step=[])
    per_sigma = {}
    for sg in sigmas:
        group = sorted((c for c in cases if c["rec"]["sigma_nm"] == sg),
                       key=lambda c: c["rec"]["seed"])
        res = ss.analyse_ensemble([c["I"] for c in group], plane, u_max=args.u_max,
                                  samples_per_speckle=args.samples,
                                  envelope_radius=args.envelope_radius,
                                  threshold=args.threshold, ks_spacing=args.ks_spacing)
        per_sigma[sg] = (group, res)
        st = res["stats"]
        expected = st["speckle_size_expected"]
        for i, c in enumerate(group):
            r = c["rec"]
            ks_i = st["ks_per_pattern"][i]
            out["cases"].append(dict(
                sigma_nm=sg, seed=r["seed"], N2=r["N2"], it=r["it"], setup_s=r["setup_s"],
                solve_s=r["solve_s"], plane_s=r["plane_s"], far_s=r.get("far_s"),
                peak_gb=r["peak_gb"], R=r.get("R"), contrast_total=r["contrast_global"],
                contrast_plane=float(res["contrast_plane"][i]),
                contrast_env=float(st["contrast"][i]),
                size_u=float(st["speckle_size"][i, 0]), size_v=float(st["speckle_size"][i, 1]),
                size_expected=expected,
                ks=_ks_dict(ks_i) if (ks_i is not None and res["ks_spacing_used"]) else None))
        ks2 = ss.pooled_ks(res["ks_samples"]) if sum(
            s.size for s in res["ks_samples"]) >= 50 else None
        cf = ss.coherent_fraction(sg * 1e-9, wl)
        rho_n = [float(sb.correlation_coefficient(res["windows"][i].ravel(),
                                                  res["windows"][j].ravel()))
                 for (i, j) in res["rho_plane"]]
        out["sigmas"].append(dict(
            sigma_nm=sg, n_seeds=len(group), coherent_fraction=cf,
            fully_developed=ss.fully_developed(sg * 1e-9, wl),
            window_u=res["window_u"], window_speckles=res["window_speckles"],
            contrast_plane_mean=float(np.mean(res["contrast_plane"])),
            contrast_plane_spread=float(np.ptp(res["contrast_plane"])),
            contrast_env_mean=st["mean_contrast"],
            ks_spacing_used=res["ks_spacing_used"],
            ks_harness=_ks_dict(st["ks"]) if res["ks_spacing_used"] else None,
            ks_spacing2=_ks_dict(ks2) if ks2 is not None else None,
            ks_spacing2_n=int(sum(s.size for s in res["ks_samples"])),
            size_mean=st["speckle_size_mean"], size_expected=st["speckle_size_expected"],
            size_rel_error=st["speckle_size_rel_error"],
            rho_seed_plane=[float(v) for v in res["rho_plane"].values()], rho_seed_env=rho_n))
        for ((i, j), v), v_env in zip(res["rho_plane"].items(), rho_n):
            out["corr_seed"].append(dict(sigma_nm=sg, seeds=(group[i]["rec"]["seed"],
                                                             group[j]["rec"]["seed"]),
                                         rho_plane=v, rho_env=v_env))
    # sigma steps (same seed, scaled map): Kirchhoff expectation exp(-(2 k dsigma)^2)
    for s1, s2 in zip(sigmas, sigmas[1:]):
        g1, r1 = per_sigma[s1]
        g2, r2 = per_sigma[s2]
        half = min(r1["window_half"], r2["window_half"])
        for i, c1 in enumerate(g1):
            for j, c2 in enumerate(g2):
                if c1["rec"]["seed"] != c2["rec"]["seed"]:
                    continue
                a = ss.centre_crop(r1["normalised"][i], half)
                b = ss.centre_crop(r2["normalised"][j], half)
                out["corr_step"].append(dict(
                    sigma_nm=(s1, s2), seed=c1["rec"]["seed"],
                    rho_plane=float(sb.correlation_coefficient(c1["I"].ravel(), c2["I"].ravel())),
                    rho_env=float(sb.correlation_coefficient(a.ravel(), b.ravel())),
                    expected=math.exp(-((2 * k * (s2 - s1) * 1e-9) ** 2))))
    # docs/05 row 42 (the gate): per fully developed sigma, pooled over the seeds, evaluable when
    # n >= 50 (``ks_spacing2`` above); summarised here, nothing recomputed
    out["row42_per_sigma"] = dict(spacing=args.ks_spacing, sigmas=[dict(
        sigma_nm=s["sigma_nm"], fully_developed=s["fully_developed"], n=s["ks_spacing2_n"],
        evaluable=s["fully_developed"] and s["ks_spacing2"] is not None,
        result=s["ks_spacing2"]) for s in out["sigmas"]])
    # supplementary, not the docs/05 gate: the same samples pooled over the fully developed
    # sigma as well (the maps are the same at every sigma: not independent)
    fd = [sg for sg in sigmas if ss.fully_developed(sg * 1e-9, wl)]
    samples = [s for sg in fd for s in per_sigma[sg][1]["ks_samples"]]
    pooled = None
    if samples and sum(s.size for s in samples) >= 50:
        pooled = _ks_dict(ss.pooled_ks(samples))
    out["sigma_pooled_ks"] = dict(sigmas=fd, spacing=args.ks_spacing, result=pooled)
    # supplementary (not the gate): the same pool at one speckle size (neighbour |mu_A|^2 =
    # 1/16; level 5-6 % on synthetic speckle, benchmarks/speckle_sweep_calibration.py)
    samples1 = [s for sg in fd for s in per_sigma[sg][1]["ks_samples_one"]]
    out["pooled_ks_one"] = (_ks_dict(ss.pooled_ks(samples1))
                            if samples1 and sum(s.size for s in samples1) >= 50 else None)
    # row 43: seed means ordered by sigma
    means = [s["contrast_plane_mean"] for s in out["sigmas"]]
    spread = max((s["contrast_plane_spread"] for s in out["sigmas"]), default=0.0)
    out["contrast_monotone"] = dict(
        means=means, strict=speckle.monotone_decreasing(means),
        within_spread=speckle.monotone_decreasing(means, spread), spread=spread,
        complete=all(s["n_seeds"] >= 2 for s in out["sigmas"]))
    # row 44: per fully developed sigma within 10 % (the others are reported, not judged)
    out["size_within_10pct"] = {s["sigma_nm"]: abs(s["size_rel_error"]) < 0.10
                                for s in out["sigmas"] if s["fully_developed"]}
    return out


#: Row-42 sensitivity study: (label, change) with one analysis parameter changed at a time
#: against the defaults of the parser. Keys: ``envelope_radius``, ``threshold``, ``samples``
#: ((u, v) samples per lambda / D), ``u_max``, ``ks_spacing`` (as the options), ``offset``
#: (decimation grid shifted by that many pixels, ``decimated_samples``), ``drop_lowest`` (the
#: smallest fully developed sigma left out of the pool).
SENSITIVITY_VARIANTS = (
    ("baseline", {}),
    ("envelope radius 0.35", dict(envelope_radius=0.35)),
    ("envelope radius 0.75", dict(envelope_radius=0.75)),
    ("envelope radius 1.0", dict(envelope_radius=1.0)),
    ("threshold 0.03", dict(threshold=0.03)),
    ("threshold 0.08", dict(threshold=0.08)),
    ("threshold 0.10", dict(threshold=0.10)),
    ("threshold 0.15", dict(threshold=0.15)),
    ("samples per speckle 6", dict(samples=6.0)),
    ("samples per speckle 7", dict(samples=7.0)),
    ("samples per speckle 10", dict(samples=10.0)),
    ("decimation offset 2 px", dict(offset=2)),
    ("decimation offset 4 px", dict(offset=4)),
    ("decimation offset 6 px", dict(offset=6)),
    ("decimation offset 8 px", dict(offset=8)),
    ("decimation offset 10 px", dict(offset=10)),
    ("KS spacing 1.5", dict(ks_spacing=1.5)),
    ("KS spacing 2.5", dict(ks_spacing=2.5)),
    ("smallest fully developed σ left out", dict(drop_lowest=True)),
    ("u_max 0.60", dict(u_max=0.60)),
    ("u_max 0.63", dict(u_max=0.63)),
)


def sensitivity(cases: list[dict], args, variants=None) -> dict:
    """The sigma-pooled row-42 pool (supplementary, not the docs/05 gate: pooled over the seeds
    and the fully developed sigma, decimated at ``ks_spacing`` speckle sizes) for each variant of
    ``variants`` (default: SENSITIVITY_VARIANTS).

    Returns dict(rows=[dict(label, change, sigmas, n, statistic, p, modified, rejects)],
    n_variants, n_rejecting); the counts are over the rows other than "baseline" with n >= 50
    (rows with fewer samples carry only ``n``)."""
    from specklebem import speckle
    from specklebem import speckle_sweep as ss

    if variants is None:
        variants = SENSITIVITY_VARIANTS
    plane = cases[0]["plane"]
    wl = plane.wavelength
    groups: dict[float, list[dict]] = {}
    for c in cases:
        groups.setdefault(c["rec"]["sigma_nm"], []).append(c)
    fd = [sg for sg in sorted(groups) if ss.fully_developed(sg * 1e-9, wl)]
    base = dict(envelope_radius=args.envelope_radius, threshold=args.threshold,
                samples=args.samples, u_max=args.u_max, ks_spacing=args.ks_spacing)
    cache: dict[tuple, dict] = {}
    rows = []
    for label, change in variants:
        unknown = set(change) - set(base) - {"offset", "drop_lowest"}
        if unknown:
            raise ValueError(f"sensitivity: unknown parameters {sorted(unknown)} in {label!r}")
        par = base | {k: v for k, v in change.items() if k in base}
        sigmas = fd[1:] if change.get("drop_lowest") else fd
        samples = []
        for sg in sigmas:
            key = (sg,) + tuple(par[k] for k in sorted(par))
            if key not in cache:
                group = sorted(groups[sg], key=lambda c: c["rec"]["seed"])
                cache[key] = ss.analyse_ensemble(
                    [c["I"] for c in group], plane, u_max=par["u_max"],
                    samples_per_speckle=par["samples"], envelope_radius=par["envelope_radius"],
                    threshold=par["threshold"], ks_spacing=par["ks_spacing"])
            res = cache[key]
            stride = speckle.independent_stride(res["stats"]["speckle_size_expected"],
                                                res["pixel_size"], par["ks_spacing"])
            samples += ss.decimated_samples(res["windows"], stride, change.get("offset", 0))
        n = int(sum(x.size for x in samples))
        ks = _ks_dict(ss.pooled_ks(samples)) if n >= 50 else dict(n=n)
        rows.append(dict(label=label, change=dict(change), sigmas=list(sigmas), **ks))
    others = [r for r in rows if r["label"] != "baseline" and "rejects" in r]
    return dict(rows=rows, n_variants=len(others),
                n_rejecting=sum(1 for r in others if r["rejects"]))


def print_sensitivity_markdown(sens: dict) -> None:
    print("| Variant | σ pooled [nm] | n | D | plain p | modified (crit. 1.094) | rejects |")
    print("|---|---|---|---|---|---|---|")
    for r in sens["rows"]:
        sg = ", ".join(f"{s:g}" for s in r["sigmas"])
        if "rejects" not in r:
            print(f"| {r['label']} | {sg} | {r['n']} (< 50) | – | – | – | – |")
            continue
        print(f"| {r['label']} | {sg} | {r['n']} | {r['statistic']:.4f} | {r['p']:.3f} "
              f"| {r['modified']:.3f} | {'**yes**' if r['rejects'] else 'no'} |")
    print()
    print(f"Variants rejecting: {sens['n_rejecting']} of {sens['n_variants']} (baseline not "
          "counted)")


def _f(v, fmt):
    return "–" if v is None else format(v, fmt)


def row42_per_sigma_lines(out: dict) -> list[str]:
    """The docs/05 row-42 gate per sigma as text: a header line, then one line per sigma with n,
    the modified statistic, the plain p and the decision where evaluable (fully developed,
    n >= 50), else why it is not evaluated."""
    g = out["row42_per_sigma"]
    lines = [f"docs/05 row 42 per σ (fully developed σ, pooled over the seeds, spacing "
             f"{g['spacing']:g} speckle sizes, n ≥ 50, Stephens rejects(0.05)):"]
    for s in g["sigmas"]:
        r = s["result"]
        if not s["fully_developed"]:
            verdict = "partially developed, not gated"
        elif r is None:
            verdict = "not evaluable (n < 50)"
        else:
            verdict = (f"modified = {r['modified']:.3f} (crit. 1.094), plain p = {r['p']:.3f}, "
                       + ("rejects" if r["rejects"] else "does not reject"))
        lines.append(f"σ = {s['sigma_nm']:g} nm: n = {s['n']}, {verdict}")
    return lines


def print_markdown(out: dict) -> None:
    print("| σ [nm] | seed | 2N | it | setup [s] | solve [s] | plane [s] | far [s] | peak [GB] "
          "| R | C total | C co-pol | C env | size u / v [µm] | KS D / p / mod (n) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in out["cases"]:
        ks = c["ks"]
        kss = (f"{ks['statistic']:.3f} / {ks['p']:.2f} / {ks['modified']:.2f} ({ks['n']})"
               if ks else "–")
        print(f"| {c['sigma_nm']:g} | {c['seed']} | {c['N2']} | {c['it']} | {c['setup_s']:.0f} "
              f"| {c['solve_s']:.0f} | {c['plane_s']:.0f} | {_f(c['far_s'], '.0f')} "
              f"| {c['peak_gb']:.1f} | {_f(c['R'], '.4f')} | {c['contrast_total']:.2f} "
              f"| {c['contrast_plane']:.2f} | {c['contrast_env']:.3f} "
              f"| {c['size_u'] * 1e6:.0f} / {c['size_v'] * 1e6:.0f} | {kss} |")
    print()
    print("| σ [nm] | coherent fraction | fully dev. | window ±u (speckles) | C co-pol mean "
          "(spread) | C env mean | KS harness: spacing, n, D, p, mod, rejects | KS 2 speckles: "
          "n, p, mod, rejects | size mean [µm] (rel. err.) | ρ seeds plane / env |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in out["sigmas"]:
        kh = s["ks_harness"]
        khs = (f"{s['ks_spacing_used']:g}, {kh['n']}, {kh['statistic']:.3f}, {kh['p']:.2f}, "
               f"{kh['modified']:.2f}, {'yes' if kh['rejects'] else 'no'}") if kh else "–"
        k2 = s["ks_spacing2"]
        k2s = (f"{k2['n']}, {k2['p']:.2f}, {k2['modified']:.2f}, "
               f"{'yes' if k2['rejects'] else 'no'}") if k2 else f"{s['ks_spacing2_n']} (< 50)"
        rp = ", ".join(f"{v:.3f}" for v in s["rho_seed_plane"])
        re_ = ", ".join(f"{v:.3f}" for v in s["rho_seed_env"])
        print(f"| {s['sigma_nm']:g} | {s['coherent_fraction']:.1e} "
              f"| {'yes' if s['fully_developed'] else 'no'} | {s['window_u']:.3f} "
              f"({s['window_speckles']:.1f}) | {s['contrast_plane_mean']:.2f} "
              f"({s['contrast_plane_spread']:.2f}) | {s['contrast_env_mean']:.3f} | {khs} | {k2s} "
              f"| {s['size_mean'] * 1e6:.1f} ({s['size_rel_error']:+.3f}) | {rp} / {re_} |")
    print()
    print("| σ step [nm] | seed | ρ plane | ρ env (common window) | exp(−(2kΔσ)²) |")
    print("|---|---|---|---|---|")
    for c in out["corr_step"]:
        print(f"| {c['sigma_nm'][0]:g} → {c['sigma_nm'][1]:g} | {c['seed']} "
              f"| {c['rho_plane']:.3f} | {c['rho_env']:.3f} | {c['expected']:.3f} |")
    print()
    head, *rows = row42_per_sigma_lines(out)
    print(head)
    for line in rows:
        print(f"- {line}")
    print()
    g = out["sigma_pooled_ks"]
    label = (f"σ-pooled (supplementary, not the docs/05 gate; σ = "
             f"{', '.join(f'{s:g}' for s in g['sigmas'])} nm, spacing {g['spacing']:g})")
    if g["result"]:
        r = g["result"]
        print(f"{label}: n = {r['n']}, D = {r['statistic']:.4f}, plain p = {r['p']:.3f}, "
              f"modified = {r['modified']:.3f} (crit. 1.094), rejects = {r['rejects']}")
    else:
        print(f"{label}: fewer than 50 samples")
    r1 = out["pooled_ks_one"]
    if r1:
        print(f"Supplementary, same pool at 1 speckle size: n = {r1['n']}, D = "
              f"{r1['statistic']:.4f}, plain p = {r1['p']:.3f}, modified = {r1['modified']:.3f}, "
              f"rejects = {r1['rejects']}")
    m = out["contrast_monotone"]
    print(f"Row 43 contrast seed means {', '.join(f'{v:.3f}' for v in m['means'])}: monotone "
          f"{m['strict']}, within spread {m['spread']:.2f}: {m['within_spread']}, all sigma "
          f"with two seeds: {m['complete']}")
    not_judged = [s["sigma_nm"] for s in out["sigmas"] if not s["fully_developed"]]
    print("Row 44 within 10 % (fully developed σ): "
          + (", ".join(f"{k:g}: {v}" for k, v in out["size_within_10pct"].items()) or "none")
          + (f"; reported, not judged (partially developed): "
             f"{', '.join(f'{v:g}' for v in not_judged)}" if not_judged else ""))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # sigma, mu in the tables and lines (cp1252 consoles)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if args.pythonpath:
        sys.path.insert(0, str(Path(args.pythonpath).resolve()))
    cases = load_cases(Path(args.cases))
    out = analyse(Path(args.cases), args, cases)
    if args.sensitivity:
        out["sensitivity"] = sensitivity(cases, args)
    if args.markdown:
        print_markdown(out)
        if args.sensitivity:
            print()
            print_sensitivity_markdown(out["sensitivity"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n",
                                   encoding="utf-8")
    if not args.markdown:
        # the gate lines go to stderr so that the JSON on stdout stays valid
        for line in row42_per_sigma_lines(out):
            print(line, file=sys.stderr)
    if not args.markdown and not args.json:
        print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
