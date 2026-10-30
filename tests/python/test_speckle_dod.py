"""Phase 6 definition of done (docs/01, docs/05 rows 42-44) on the saved speckle patterns.

Runs the committed WP-S2 analysis (``benchmarks/speckle_sweep_stats.py`` -> ``analyse`` ->
``specklebem.speckle_sweep.analyse_ensemble`` -> the WP-S1 harness) with the record's defaults
(parser defaults: u_max 0.66, 8 samples per lambda / D, envelope radius 0.5 speckle sizes, window
threshold 0.05, KS spacing 2 speckle sizes) on a directory of solved cases. No solve here.

Data: the directory ``$SBEM_SPECKLE_SWEEP_DIR`` (default ``<repo>/results/local_runs/wp_s2/cases``,
git-ignored) holding one subdirectory per case with ``E.npy`` and ``meta.json``
(``benchmarks/speckle_sweep.py``). Without finished cases in the default directory every data
test is skipped (CI has no data); if ``$SBEM_SPECKLE_SWEEP_DIR`` is set explicitly but holds no
finished case, the data tests fail (a mistyped path cannot pass silently; the policy of
``SPECKLEBEM_NPY_FIXTURE`` in conftest.py). Only normal-incidence cases are analysed (the oblique
gamma_12 cases have no analysis yet).

Two kinds of tests:

- **Record tests** (``record`` fixture) run only on the WP-S2 configuration (Si, L = 8 um,
  Lc = 1 um, h = 50 nm, 0 deg, sigma = 50 ... 250 nm x seeds 1-2, 161^2 plane of 4 mm;
  ``benchmarks/results/speckle_sweep.md``) and pin its numbers. Row 43 and row 44 (3 of the 4
  fully developed sigma) are asserted as met; sigma = 200 nm is a strict xfail against the 10 %
  bound. **Row 42 is not met with this configuration** (lead decision 2026-10-30): per sigma the
  seed-pooled decimated sample at 2 speckle sizes has n = 8 / 18 / 32 / 32 < 50, so the docs/05
  gate is not evaluable, and the patterns are measurably over-dispersed (the spot covers only
  (w0 / Lc)^2 = 1.8 correlation areas, pi w0^2 convention). The row-42 record test guards that
  *negative* result (sample sizes, the 1-speckle rejection, C_env > 1.1); it is not a pass of
  row 42. The sigma-pooled gate (n = 90) is supplementary and pinned only as a regression value.
- **Gate tests** run on any sweep directory (e.g. the server runs of the record's "Server recipe")
  with the docs/05 definitions: row 42 per sigma (pooled over the seeds, n >= 50 at 2 speckle
  sizes, ``KsResult.rejects(0.05)``) where evaluable, row 43 monotone, row 44 within 10 % per
  fully developed sigma (skipped on the record configuration, whose sigma = 200 nm entry is
  covered by the strict xfail above). Expected / inconclusive outcomes on the server series
  (not regressions; see the test docstrings and the record's "Server recipe"): a row-43
  failure on series (i) (Lc = 0.5 um, the plane contrast saturates), a single per-sigma row-42
  rejection on series (iii) (L = 25 um, two seeds), and in general one rejection among k
  evaluable sigma (family-wise rate 1 - 0.95^k on exact speckle).
"""

import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path

import pytest

from specklebem import speckle
from specklebem import speckle_sweep as ss

REPO = Path(__file__).resolve().parents[2]
ENV_DIR = "SBEM_SPECKLE_SWEEP_DIR"
WL = 500e-9

#: The WP-S2 record configuration (benchmarks/results/speckle_sweep.md, "Configuration").
RECORD_SIGMAS = (50.0, 100.0, 150.0, 200.0, 250.0)
RECORD_SEEDS = (1, 2)
RECORD = dict(material="si", L_um=8.0, Lc_um=1.0, h_nm=50.0, theta_deg=0.0, plane_n=161,
              plane_mm=4.0, z_mm=-1.0)

pytestmark = pytest.mark.validation_large


def sweep_dir() -> Path:
    env = os.environ.get(ENV_DIR)
    return Path(env) if env else REPO / "results" / "local_runs" / "wp_s2" / "cases"


def _stats_module():
    script = REPO / "benchmarks" / "speckle_sweep_stats.py"
    spec = importlib.util.spec_from_file_location("speckle_sweep_stats_dod", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finished(d: Path) -> bool:
    return d.is_dir() and any((c / "meta.json").is_file() and (c / "E.npy").is_file()
                              for c in d.iterdir() if c.is_dir())


@pytest.fixture(scope="module")
def sweep():
    """Analysis of the normal-incidence cases of the sweep directory (record defaults)."""
    d = sweep_dir()
    if not _finished(d):
        if os.environ.get(ENV_DIR):
            pytest.fail(f"{ENV_DIR}={os.environ[ENV_DIR]!r} holds no finished speckle case "
                        "(subdirectories with meta.json and E.npy written by "
                        "benchmarks/speckle_sweep.py)")
        pytest.skip(f"no solved speckle cases in {d} (set {ENV_DIR} to a directory of "
                    "benchmarks/speckle_sweep.py case directories)")
    mod = _stats_module()
    cases = [c for c in mod.load_cases(d) if float(c["rec"].get("theta_deg", 0.0)) == 0.0]
    if not cases:
        pytest.skip(f"no normal-incidence cases in {d}")
    keys = ("material", "L_um", "Lc_um", "h_nm", "plane_n", "plane_mm", "z_mm")
    configs = {tuple(c["rec"][k] for k in keys) for c in cases}
    if len(configs) != 1:
        pytest.fail(f"{d} mixes configurations {sorted(configs)} ({', '.join(keys)}): one "
                    "configuration per directory")
    args = mod.build_parser().parse_args([str(d)])
    out = mod.analyse(d, args, cases)
    by_sigma = {s["sigma_nm"]: s for s in out["sigmas"]}
    seeds = {sg: sorted(c["rec"]["seed"] for c in cases if c["rec"]["sigma_nm"] == sg)
             for sg in by_sigma}
    is_record = (all(cases[0]["rec"][k] == v for k, v in RECORD.items())
                 and tuple(sorted(by_sigma)) == RECORD_SIGMAS
                 and all(tuple(s) == RECORD_SEEDS for s in seeds.values()))
    return dict(dir=d, out=out, by_sigma=by_sigma, seeds=seeds, cases=cases,
                is_record=is_record)


@pytest.fixture(scope="module")
def record(sweep):
    if not sweep["is_record"]:
        pytest.skip("not the WP-S2 record configuration (L = 8 um, Lc = 1 um, 0 deg, "
                    "sigma = 50-250 nm x seeds 1-2): record values not applicable")
    return sweep


# --------------------------------------------------------------------------------------------
# Data-free: the fully-developed classification of the lead rule (docs/05 row 42)


def test_fully_developed_classification_of_the_sweep_sigmas():
    # exp(-(2 k sigma)^2) < 1e-2: sigma = 50 nm (0.21) partially developed, 100 nm (1.8e-3)
    # and above fully developed (lead decision 2026-10-30, c).
    assert not ss.fully_developed(50e-9, WL)
    assert ss.coherent_fraction(50e-9, WL) == pytest.approx(0.206, abs=5e-4)
    for sg in (100e-9, 150e-9, 200e-9, 250e-9):
        assert ss.fully_developed(sg, WL)
    assert ss.coherent_fraction(100e-9, WL) == pytest.approx(1.8e-3, abs=5e-5)
    assert ss.FULLY_DEVELOPED_COHERENT_FRACTION == 1e-2


# --------------------------------------------------------------------------------------------
# Record tests (WP-S2 configuration; numbers of benchmarks/results/speckle_sweep.md)


def test_record_classification_per_sigma(record):
    flags = {sg: s["fully_developed"] for sg, s in record["by_sigma"].items()}
    assert flags == {50.0: False, 100.0: True, 150.0: True, 200.0: True, 250.0: True}
    # row 44 judges the fully developed sigma only: sigma = 50 nm (+15.4 %, coherent fraction
    # 0.21) is reported, not judged against 10 %
    assert record["by_sigma"][50.0]["size_rel_error"] == pytest.approx(0.154, abs=5e-4)
    assert record["out"]["size_within_10pct"] == {100.0: True, 150.0: True, 200.0: False,
                                                  250.0: True}


def test_record_row43_contrast_decreases_monotonically(record):
    # Seed-mean co-polarised plane contrast (raw 4 x 4 mm plane), strictly decreasing over
    # sigma = 50 ... 250 nm, and each seed on its own (paired: the same two maps scaled).
    m = record["out"]["contrast_monotone"]
    assert m["means"] == pytest.approx([10.72, 5.65, 3.67, 2.66, 2.15], abs=5e-3)
    assert m["strict"] and m["complete"]
    assert speckle.monotone_decreasing(m["means"])
    expected = {1: [11.29, 6.58, 4.20, 3.12, 2.57], 2: [10.15, 4.71, 3.13, 2.20, 1.74]}
    for seed, values in expected.items():
        per_seed = [c["contrast_plane"] for sg in RECORD_SIGMAS
                    for c in record["out"]["cases"]
                    if c["sigma_nm"] == sg and c["seed"] == seed]
        assert per_seed == pytest.approx(values, abs=5e-3)
        assert speckle.monotone_decreasing(per_seed)


@pytest.mark.parametrize("sigma_nm, rel_error", [(100.0, 0.028), (150.0, 0.044), (250.0, 0.004)])
def test_record_row44_speckle_size_within_10_percent(record, sigma_nm, rel_error):
    # FWHM of |mu_A|^2 (mean over u, v and both seeds) vs speckle_size_expected(lambda, |z|,
    # 2 w0, "gaussian") = 198.8 um; docs/05: < 10 % per fully developed sigma.
    s = record["by_sigma"][sigma_nm]
    assert s["size_expected"] == pytest.approx(198.8e-6, rel=5e-4)
    assert s["size_rel_error"] == pytest.approx(rel_error, abs=5e-4)
    assert abs(s["size_rel_error"]) < 0.10


def test_record_row44_sigma200_recorded_value(record):
    # +11.4 % recorded (pinned to +-1 percentage point; the 10 % bound is the strict xfail below).
    s = record["by_sigma"][200.0]
    assert s["size_rel_error"] == pytest.approx(0.114, abs=0.01)
    assert s["size_mean"] == pytest.approx(221.5e-6, rel=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="two-seed scatter +-8-15 %: sigma = 200 nm is +11.4 % "
                   "(benchmarks/results/speckle_sweep.md, row 44 met for 3 of 4 sigma)")
def test_record_row44_sigma200_within_10_percent(record):
    assert abs(record["by_sigma"][200.0]["size_rel_error"]) < 0.10


def test_record_row42_not_evaluable_and_not_exponential(record):
    """Guards the recorded *negative* row-42 result; this is not a pass of row 42.

    docs/05 row 42 (lead decision 2026-10-30, c): per sigma, pooled over the seeds, n >= 50
    independent samples decimated at 2 speckle sizes, decision ``KsResult.rejects(0.05)``. With
    L = 8 um and two seeds the per-sigma sample has n = 8 / 18 / 32 / 32 for sigma = 100 / 150 /
    200 / 250 nm (8 at 50 nm): the gate is not evaluable (row 42 not met with this
    configuration, open until the server run). The physical finding is pinned as a regression
    guard: the 1-speckle pool over the fully developed sigma rejects (modified 1.73, p = 0.006)
    and the envelope-normalised contrast C_env > 1.1 for sigma >= 150 nm (over-dispersed: the
    spot covers (w0 / Lc)^2 = 1.8 correlation areas, pi w0^2 convention; 0.9 with the effective
    area pi w0^2 / 2).
    """
    by = record["by_sigma"]
    expected_n = {50.0: 8, 100.0: 8, 150.0: 18, 200.0: 32, 250.0: 32}
    for sg, n in expected_n.items():
        s = by[sg]
        assert s["fully_developed"] == (sg >= 100.0)
        assert s["ks_spacing2_n"] == n
        assert s["ks_spacing2_n"] < 50  # -> the docs/05 gate is not evaluable at this sigma
        assert s["ks_spacing2"] is None
        assert s["ks_spacing_used"] != 2.0  # the harness fell back to 1 speckle size or none
    per = {r["sigma_nm"]: r for r in record["out"]["row42_per_sigma"]["sigmas"]}
    assert not any(r["evaluable"] for r in per.values())
    assert {sg: r["n"] for sg, r in per.items()} == expected_n
    # physical finding (regression guard of the negative result)
    one = record["out"]["pooled_ks_one"]
    assert one["n"] == 278
    assert one["modified"] == pytest.approx(1.729, abs=5e-3)
    assert one["p"] == pytest.approx(0.006, abs=5e-4)
    assert one["rejects"]
    for sg, c_env in {150.0: 1.162, 200.0: 1.294, 250.0: 1.158}.items():
        assert by[sg]["contrast_env_mean"] == pytest.approx(c_env, abs=5e-4)
        assert by[sg]["contrast_env_mean"] > 1.1


def test_record_row42_sigma_pooled_regression_value_only(record):
    # Supplementary, never a pass and not the docs/05 gate: the sigma-pooled pool (sigma =
    # 100-250 nm, both seeds, 2 speckle sizes) reuses the same two scaled maps at every sigma
    # (not independent) and its decision flips under 12 of 20 neighbouring analysis choices. Only
    # its numbers are pinned.
    g = record["out"]["sigma_pooled_ks"]
    assert g["sigmas"] == [100.0, 150.0, 200.0, 250.0] and g["spacing"] == 2.0
    r = g["result"]
    assert r["n"] == 90
    assert r["modified"] == pytest.approx(1.086, abs=5e-4)
    assert r["statistic"] == pytest.approx(0.1131, abs=5e-5)
    assert r["p"] == pytest.approx(0.200, abs=5e-4)


def test_record_seed_correlation_supplementary(record):
    # gamma_12 is not testable at normal incidence (decision b); the seed-to-seed correlation of
    # the raw plane patterns is reported as supplementary only.
    rho = [record["by_sigma"][sg]["rho_seed_plane"][0] for sg in RECORD_SIGMAS]
    assert rho == pytest.approx([0.976, 0.854, 0.625, 0.465, 0.329], abs=5e-4)
    assert speckle.monotone_decreasing(rho)


# --------------------------------------------------------------------------------------------
# docs/05 gate tests on any sweep directory


def test_gate_row42_per_sigma_where_evaluable(sweep, record_property):
    """docs/05 row 42 per fully developed sigma: decimated I / <I> at 2 speckle sizes pooled over
    the seeds, n >= 50, ``KsResult.rejects(0.05)`` (Stephens) must be False.

    sigma with n < 50 are not evaluable (warning); without any evaluable sigma the test is
    skipped. Each evaluable sigma records n, the modified statistic and the plain p as test
    properties (``record_property``; JUnit XML).

    Reading a failure: every evaluable sigma is a 5 % test, so on exact speckle at least one of k
    evaluable sigma rejects with the family-wise probability 1 - 0.95^k (14 % for k = 3, 19 % for
    k = 4). On series (iii) of the record's "Server recipe" (L = 25 um, two seeds) the
    leave-one-out envelope from two seeds is itself detectable: the record's matched pipeline
    null (exact speckle, two seeds) rejects in ~17 % of its realisations (measured on the
    sigma-pooled sample, n ~ 820; the per-sigma level is not calibrated separately), so a single
    rejection there is an expected / inconclusive outcome, not a regression: it needs more seeds
    or a smoother envelope estimate before it decides row 42. The assertion is not weakened;
    the verdict is the lead's.
    """
    evaluable, not_evaluable = {}, {}
    for sg, s in sweep["by_sigma"].items():
        if not s["fully_developed"]:
            continue
        if s["ks_spacing2_n"] >= 50:
            evaluable[sg] = s["ks_spacing2"]
        else:
            not_evaluable[sg] = s["ks_spacing2_n"]
    if not_evaluable:
        warnings.warn("docs/05 row 42 not evaluable (n < 50 at 2 speckle sizes) at sigma = "
                      + ", ".join(f"{sg:g} nm (n = {n})" for sg, n in not_evaluable.items()),
                      stacklevel=1)
    if not evaluable:
        pytest.skip("row 42 not evaluable at any fully developed sigma: n = "
                    + ", ".join(f"{n}" for n in not_evaluable.values()) + " < 50")
    for sg, r in evaluable.items():
        record_property(f"row42_sigma{sg:g}nm_n", r["n"])
        record_property(f"row42_sigma{sg:g}nm_modified", round(r["modified"], 4))
        record_property(f"row42_sigma{sg:g}nm_plain_p", round(r["p"], 4))
    rejected = {sg: (r["n"], round(r["modified"], 3)) for sg, r in evaluable.items()
                if r["rejects"]}
    assert not rejected, f"exponential KS (Stephens) rejects at sigma: {rejected}"
    for r in evaluable.values():
        assert math.isfinite(r["p"])


def test_gate_row43_contrast_monotone(sweep):
    """docs/05 row 43: seed-mean co-polarised plane contrast strictly decreasing over sigma.

    Reading a failure: on series (i) of the record's "Server recipe" (Lc = 0.5 um at L = 8 um)
    the plane contrast is expected to saturate (phase-screen surrogate 2.00, 1.47, 1.32, 1.28
    for sigma = 100-250 nm, monotone in only 56 % of its realisations), so a failure there is an
    expected outcome, not a regression: row 43 rests on the Lc = 1 um record. On other
    directories a failure is a finding for the lead. The assertion is not weakened.
    """
    m = sweep["out"]["contrast_monotone"]
    if len(m["means"]) < 2:
        pytest.skip("row 43 needs at least two sigma")
    assert m["complete"], "row 43 needs two seeds per sigma"
    assert m["strict"], f"seed-mean contrast not strictly decreasing: {m['means']}"


def test_gate_row44_speckle_size(sweep):
    if sweep["is_record"]:
        pytest.skip("record configuration: covered by the record tests (sigma = 200 nm xfail)")
    errors = {sg: s["size_rel_error"] for sg, s in sweep["by_sigma"].items()
              if s["fully_developed"]}
    if not errors:
        pytest.skip("no fully developed sigma")
    outside = {sg: round(e, 4) for sg, e in errors.items() if not abs(e) < 0.10}
    assert not outside, f"speckle size error >= 10 % at sigma: {outside}"
