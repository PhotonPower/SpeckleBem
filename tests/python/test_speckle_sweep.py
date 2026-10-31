"""Unit tests of specklebem.speckle_sweep (WP-S2): plane sampling, ROW records, mesh helper,
analysis of saved patterns (on synthetic speckle), and the sequence of excitations on one
assembled operator (WP-KR3).

Pure functions with fixed inputs and seeds; the WP-KR3 section solves a tiny Si patch with
reduced quadrature (see its comment). The driver benchmarks/speckle_sweep.py and the statistics
script benchmarks/speckle_sweep_stats.py are imported by path.
"""

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import specklebem as sb
from specklebem import _specklebem as core
from specklebem import speckle
from specklebem import speckle_sweep as ss

WL = 500e-9
Z = -1e-3
W0 = 8e-6 / 6  # L / 6 at L = 8 um


def test_speckle_sizes_follow_lambda_z_over_d():
    s0 = ss.paraxial_speckle_size(WL, Z, W0)
    assert s0 == pytest.approx(WL * 1e-3 / (2 * W0), rel=1e-15)
    assert s0 == pytest.approx(187.5e-6, rel=1e-12)
    # FWHM of |mu_A|^2 = exp(-(pi w0 dx / (lambda z))^2): 2 sqrt(ln 2) / pi * lambda z / w0.
    fwhm = ss.gaussian_speckle_fwhm(WL, Z, W0)
    half = 0.5 * fwhm
    assert math.exp(-((math.pi * W0 * half / (WL * 1e-3)) ** 2)) == pytest.approx(0.5, rel=1e-12)
    assert fwhm / s0 == pytest.approx(4 * math.sqrt(math.log(2)) / math.pi, rel=1e-12)  # 1.06
    with pytest.raises(ValueError):
        ss.paraxial_speckle_size(WL, 0.0, W0)
    with pytest.raises(ValueError):
        ss.paraxial_speckle_size(-WL, Z, W0)


def test_plane_sampling_of_the_sweep_grid():
    p = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    assert p.spacing == pytest.approx(25e-6, rel=1e-12)
    assert p.samples_per_speckle == pytest.approx(7.5, rel=1e-12)
    assert p.max_angle == pytest.approx(math.atan(2.0), rel=1e-15)
    d_over_wl = 2 * W0 / WL  # 5.33
    assert p.speckles_per_axis == pytest.approx(2 * math.sin(math.atan(2.0)) * d_over_wl)
    assert p.speckles_per_axis == pytest.approx(9.54, abs=0.01)
    assert p.paraxial_speckles_per_axis == pytest.approx(4e-3 / 187.5e-6, rel=1e-12)
    assert p.max_speckles_per_axis == pytest.approx(2 * d_over_wl, rel=1e-15)
    assert p.coordinates()[0] == pytest.approx(-2e-3) and p.coordinates()[-1] == pytest.approx(2e-3)
    for bad in [dict(z=1e-3), dict(n=1), dict(n=10.5), dict(size=0.0), dict(waist=-1.0)]:
        kw = dict(wavelength=WL, z=Z, waist=W0, size=4e-3, n=161) | bad
        with pytest.raises(ValueError):
            ss.plane_sampling(**kw)


def test_choose_plane_meets_the_sampling_request():
    for samples in [4.0, 5.5, 8.0]:
        p = ss.choose_plane(WL, Z, W0, samples_per_speckle=samples)
        assert p.n % 2 == 1
        assert p.samples_per_speckle >= samples
        assert p.size == pytest.approx(2 * 1e-3 * 2.0, rel=1e-12)  # atan(2) default
    p = ss.choose_plane(WL, Z, W0, max_angle=math.radians(30), samples_per_speckle=4)
    assert p.max_angle == pytest.approx(math.radians(30), rel=1e-12)
    # 4 samples per 187.5 um over 2 tan(30 deg) mm = 1.1547 mm: ceil(24.63) + 1 = 26 -> odd 27.
    assert p.n == 27
    with pytest.raises(ValueError):
        ss.choose_plane(WL, Z, W0, max_angle=math.pi / 2)


def test_plane_points_follow_plane_grid_order():
    p = ss.plane_sampling(WL, Z, W0, 4e-3, 5)
    pts = np.asarray(p.points())
    assert pts.shape == (25, 3)
    np.testing.assert_allclose(pts[:, 2], Z)
    np.testing.assert_allclose(pts[0], [-2e-3, -2e-3, Z])
    np.testing.assert_allclose(pts[12], [0.0, 0.0, Z], atol=1e-18)  # centre on the axis
    np.testing.assert_allclose(pts[1], [-2e-3, -1e-3, Z])  # row i * n + j: j runs along y
    np.testing.assert_allclose(pts[5], [-1e-3, -2e-3, Z])


def test_row_round_trip_and_key_order():
    rec = dict(
        extra="x1",
        seed=2,
        L_um=8.0,
        sigma_nm=150.0,
        conv=True,
        res=9.87654321e-5,
        N2=741120,
        form="ictf",
    )
    line = ss.format_row(rec)
    assert line.startswith("ROW L_um=8 sigma_nm=150 seed=2 form=ictf N2=741120 conv=1 ")
    assert line.endswith(" extra=x1")
    back = ss.parse_row(line)
    assert back == dict(
        L_um=8,
        sigma_nm=150,
        seed=2,
        form="ictf",
        N2=741120,
        conv=1,
        res=pytest.approx(9.87654e-5, rel=1e-12),
        extra="x1",
    )
    assert isinstance(back["N2"], int) and isinstance(back["res"], float)
    assert ss.parse_row(ss.format_row(dict(a=0.1)), "ROW")["a"] == 0.1
    lines = ["# comment", "", line, "XROW a=1", "ROW a=2.5 b=s"]
    assert [r.get("a") for r in ss.read_rows(lines)] == [None, 2.5]
    assert ss.read_rows(lines, prefix="XROW") == [dict(a=1)]


@pytest.mark.parametrize("line", ["ROWX a=1", "XROW a=1", "", "ROW a", "ROW =1", "ROW a=1 a=2"])
def test_row_parse_rejects_malformed_lines(line):
    with pytest.raises(ValueError):
        ss.parse_row(line)


@pytest.mark.parametrize("rec", [dict(a="two words"), dict(a=""), dict(a="k=v"), {"b c": 1}])
def test_row_format_rejects_unparsable_tokens(rec):
    with pytest.raises(ValueError):
        ss.format_row(rec)


def test_max_support_radius_of_known_triangles():
    # Regular tetrahedron with edge a: r = height of a face from an edge midpoint = a sqrt(3)/2.
    a = 2.0
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / math.sqrt(2)
    t = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    assert ss.max_support_radius(v, t) == pytest.approx(a * math.sqrt(3) / 2, rel=1e-14)
    # Flat obtuse triangle: the far vertex seen from the midpoint of a short edge dominates
    # (7.5 > half the long edge, 5).
    v2 = np.array([[0, 0, 0], [10, 0, 0], [5, 0.1, 0]], float)
    assert ss.max_support_radius(v2, np.array([[0, 1, 2]])) == pytest.approx(
        math.hypot(7.5, 0.05), rel=1e-14
    )
    # The library icosphere: same rule on every edge, so r_max is at least half the longest edge.
    ico = sb.make_icosphere(1.0, 1)
    verts, tris = np.asarray(ico.vertices), np.asarray(ico.triangles)
    edges = np.concatenate([verts[tris[:, k]] - verts[tris[:, (k + 1) % 3]] for k in range(3)])
    assert ss.max_support_radius(verts, tris) > 0.5 * np.max(np.linalg.norm(edges, axis=1))


def test_case_tag_and_peak_memory():
    p = ss.CaseParams(L=8e-6, sigma=150e-9, seed=2)
    assert p.tag() == "si_L8um_s150nm_Lc1um_seed2"
    # s polarisation gets its own directory (WP-KR3); p tags are unchanged
    assert ss.CaseParams(L=8e-6, sigma=150e-9, seed=2, polarization="s").tag() == (
        "si_L8um_s150nm_Lc1um_seed2_pols"
    )
    assert ss.CaseParams(L=8e-6, sigma=150e-9, seed=2, polarization="P").tag() == p.tag()
    assert (
        ss.CaseParams(L=8e-6, sigma=150e-9, seed=2, polarization="s", theta_deg=30).tag()
        == "si_L8um_s150nm_Lc1um_seed2_t30_pols"
    )
    assert p.h == 50e-9 and p.solver_tol == 1e-4 and p.waist is None and p.theta_deg == 0.0
    assert p.default_waist(1 / 6) == pytest.approx(8e-6 / 6, rel=1e-15)
    q = ss.CaseParams(L=8e-6, sigma=150e-9, seed=1, theta_deg=30.0)
    assert q.tag() == "si_L8um_s150nm_Lc1um_seed1_t30"
    assert q.default_waist(1 / 6) == pytest.approx(8e-6 / 6 * math.cos(math.radians(30)))
    sb.check_beam_waist(
        8e-6, q.default_waist(1 / 6), incidence_angle=math.radians(30), max_fraction=1 / 6
    )  # the default passes the footprint rule
    assert (
        ss.CaseParams(L=8e-6, sigma=1e-7, seed=1, waist=1e-6, theta_deg=30).default_waist(1 / 6)
        == 1e-6
    )
    peak = ss.peak_memory_bytes()
    assert peak is None or peak > 1_000_000


def _driver():
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "speckle_sweep.py"
    spec = importlib.util.spec_from_file_location("speckle_sweep_driver", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_driver_parser():
    module = _driver()
    args = module.build_parser().parse_args(
        ["--sigma", "5e-8", "--seed", "1", "--out", "o", "--plane-size", "4e-3", "--plane-n", "161"]
    )
    assert args.L == 8e-6 and args.h == 50e-9 and args.Lc == 1e-6 and args.z == -1e-3
    assert args.solver_tol == 1e-4 and args.plane_n == 161 and args.far_grid == "90,180"
    assert args.theta_deg == [0.0] and args.pol == ["p"]  # lists since WP-KR3
    assert module.build_parser().parse_args(
        ["--sigma", "1e-7", "--seed", "1", "--out", "o", "--theta-deg", "30"]
    ).theta_deg == [30.0]
    args = module.build_parser().parse_args(
        ["--sigma", "1e-7", "--seed", "1", "--out", "o", "--theta-deg", "0, 30", "--pol", "S,p"]
    )
    assert args.theta_deg == [0.0, 30.0] and args.pol == ["s", "p"]
    for bad in (
        ["--theta-deg", "0,0"],
        ["--theta-deg", "0,"],
        ["--theta-deg", "x"],
        ["--pol", "x"],
        ["--pol", "p,P"],
    ):
        with pytest.raises(SystemExit):
            module.build_parser().parse_args(["--sigma", "1e-7", "--seed", "1", "--out", "o"] + bad)


@pytest.mark.parametrize("theta_deg, Lc", [(0.0, 0.5e-6), (30.0, 1e-6)])
def test_build_simulation_of_a_small_case_without_solve(theta_deg, Lc):
    # Case construction of the driver (mesh, beam, Simulation object; no assemble, no solve) on
    # a tiny patch, for the server recipe's variants (Lc = 0.5 um; 30 deg incidence).
    p = ss.CaseParams(L=1e-6, sigma=100e-9, seed=1, h=100e-9, Lc=Lc, theta_deg=theta_deg)
    sim, info = ss.build_simulation(p)
    assert isinstance(info["eta0"], float)
    assert info["eta0"] == pytest.approx(376.730313668, rel=1e-9)  # vacuum (R1)
    assert info["waist"] == pytest.approx(1e-6 * math.cos(math.radians(theta_deg)) / 6, rel=1e-12)
    assert sim.num_unknowns > 0 and info["triangles"] > 0 and info["rmax"] > 0
    assert p.tag() == (
        "si_L1um_s100nm_Lc0.5um_seed1" if theta_deg == 0.0 else "si_L1um_s100nm_Lc1um_seed1_t30"
    )


def test_report_parser_reads_leaf_and_levels():
    rep = (
        "  operator:       MlfmmOperator: 2N = 51840, d0 = 3, octree 5 levels, 610 leaves of "
        "edge 0.724812 lambda1 (lambda1 = 5e-07 m), built in 0.0034908 s; leaf floor 0.455605\n"
    )
    out = ss._parse_report(rep)
    assert out["levels"] == 5
    assert out["leaf_nm"] == pytest.approx(362.406, rel=1e-6)
    assert ss._parse_report("dense operator") == dict(leaf_nm=None, levels=None)


# --------------------------------------------------------------------------------------------
# Analysis of saved patterns


def test_synthetic_far_zone_speckle_statistics():
    plane = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    rng = np.random.default_rng(99)
    u = ss.direction_cosine_axis(0.66, WL / (2 * W0) / 8)
    contrasts, sizes = [], []
    for _ in range(6):
        flat = ss.synthetic_far_zone_speckle(rng, plane)
        assert flat.shape == (161, 161) and np.all(flat >= 0)
        # On the plane the radiant intensity is weighted with w^2 = cos^2(theta): contrast > 1.
        assert sb.speckle_contrast(flat.ravel()) > 1.05
        # Radiant intensity in (u, v) is stationary speckle: contrast ~1, WP-S1 size.
        rad = ss.plane_to_direction_cosines(flat, -2e-3, plane.spacing, Z, u)
        contrasts.append(sb.speckle_contrast(rad.ravel()))
        mu = sb.field_correlation_from_intensity(rad / rad.mean())
        sizes += [sb.speckle_size(mu, (u[1] - u[0]) * 1e-3, axis) for axis in (0, 1)]
    assert np.mean(contrasts) == pytest.approx(1.0, abs=0.1)
    expected = sb.speckle_size_expected(WL, 1e-3, 2 * W0, "gaussian")
    assert np.mean(sizes) == pytest.approx(expected, rel=0.12)  # ~40 speckles per pattern
    a = ss.synthetic_far_zone_speckle(np.random.default_rng(5), plane, envelope_rms=0.3)
    b = ss.synthetic_far_zone_speckle(np.random.default_rng(5), plane, envelope_rms=0.3)
    assert np.array_equal(a, b)  # fixed seed, reproducible


def test_coherent_fraction_and_the_fully_developed_rule():
    k = 2 * math.pi / WL
    assert ss.coherent_fraction(100e-9, WL) == pytest.approx(math.exp(-((2 * k * 1e-7) ** 2)))
    assert ss.coherent_fraction(50e-9, WL) == pytest.approx(0.2062, abs=1e-4)
    assert ss.coherent_fraction(100e-9, WL) == pytest.approx(1.806e-3, rel=1e-3)
    assert ss.coherent_fraction(0.0, WL) == 1.0
    assert not ss.fully_developed(50e-9, WL)
    assert all(ss.fully_developed(s, WL) for s in [100e-9, 150e-9, 200e-9, 250e-9])
    with pytest.raises(ValueError):
        ss.coherent_fraction(-1e-9, WL)


def test_direction_cosine_resampling():
    u = ss.direction_cosine_axis(0.6, 0.05)
    assert len(u) == 25 and u[12] == 0.0 and u[-1] == pytest.approx(0.6)
    np.testing.assert_allclose(u, -u[::-1], atol=1e-15)
    with pytest.raises(ValueError):
        ss.direction_cosine_axis(0.01, 0.05)
    # A plane field equal to x is resampled to |z| u / w, w = sqrt(1 - u^2 - v^2); radiant
    # intensity divides by w^2 (times (r / |z|)^2).
    n, size, z = 201, 4e-3, -1e-3
    x = -0.5 * size + size / (n - 1) * np.arange(n)
    grid = np.repeat(x[:, None], n, axis=1)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    w = np.sqrt(1 - uu**2 - vv**2)
    res = ss.plane_to_direction_cosines(grid, -0.5 * size, size / (n - 1), z, u, radiant=False)
    np.testing.assert_allclose(res, 1e-3 * uu / w, atol=1e-12)
    rad = ss.plane_to_direction_cosines(np.ones((n, n)), -0.5 * size, size / (n - 1), z, u)
    np.testing.assert_allclose(rad, 1 / w**2, rtol=1e-12)
    for umax in [0.9, 0.7]:  # leaves the unit disc / reaches beyond the plane
        with pytest.raises(ValueError):
            ss.plane_to_direction_cosines(
                grid, -0.5 * size, size / (n - 1), z, ss.direction_cosine_axis(umax, 0.05)
            )


def test_mirror_envelope_window_and_decimation_helpers():
    a = np.arange(9.0).reshape(3, 3)
    m = ss.mirror_images(a)
    assert len(m) == 4 and np.array_equal(m[3], a[::-1, ::-1]) and np.array_equal(m[1][0], a[2])
    np.testing.assert_allclose(ss.gaussian_smooth(np.full((20, 30), 2.5), 4.0), 2.5, rtol=1e-12)
    assert np.array_equal(ss.gaussian_smooth(a, 0.0), a)
    # Leave-one-out: pattern 0's envelope = (4 c1 + 3 c0) / 7 for constant patterns c0, c1.
    env = ss.leave_one_out_envelopes([np.full((5, 5), 1.0), np.full((5, 5), 8.0)], 1.5)
    np.testing.assert_allclose(env[0], (4 * 8 + 3 * 1) / 7, rtol=1e-12)
    np.testing.assert_allclose(env[1], (4 * 1 + 3 * 8) / 7, rtol=1e-12)
    env1 = ss.leave_one_out_envelopes([a], 0.0)[0]  # one pattern: its three mirror images
    np.testing.assert_allclose(env1, (a[::-1, :] + a[:, ::-1] + a[::-1, ::-1]) / 3)
    with pytest.raises(ValueError):
        ss.leave_one_out_envelopes([np.ones((3, 3)), np.ones((4, 4))], 1.0)
    # Window: exp(-r^2) on a pixel grid, threshold exp(-8) -> corners of the square at r^2 = 8.
    c = np.arange(-6, 7)
    e = np.exp(-(c[:, None] ** 2 + c[None, :] ** 2).astype(float))
    assert ss.centred_window(e, math.exp(-8.0) * 0.999) == 2
    assert ss.centred_window(e, 0.5) == 0
    assert ss.centred_window(np.ones((7, 7)), 0.5) == 3
    e2 = np.ones((5, 5))
    e2[2, 2] = 0.0
    assert ss.centred_window(e2, 0.5) == -1
    assert ss.centre_crop(np.arange(25).reshape(5, 5), 1).tolist() == [
        [6, 7, 8],
        [11, 12, 13],
        [16, 17, 18],
    ]
    for shape, stride in [((57, 57), 17), ((10, 7), 3), ((5, 5), 1)]:
        assert ss.decimate_count(shape, stride) == speckle.decimate(np.ones(shape), stride).size
    rng = np.random.default_rng(7)
    w = [rng.exponential(2.0, (20, 20)), rng.exponential(5.0, (20, 20))]
    d = ss.decimated_samples(w, 3)
    assert [x.size for x in d] == [49, 49] and all(x.mean() == pytest.approx(1.0) for x in d)
    assert ss.pooled_ks(d).n == 98
    np.testing.assert_array_equal(ss.decimated_samples(w, 3, 0)[1], d[1])
    # offset shifts the decimation grid: w[offset::stride, offset::stride] / its mean
    d2 = ss.decimated_samples(w, 3, 2)
    ref = w[0][2::3, 2::3].ravel()
    assert d2[0].size == 36 and np.allclose(d2[0], ref / ref.mean(), rtol=1e-14)
    for bad in (-1, 3, 1.5):
        with pytest.raises(ValueError):
            ss.decimated_samples(w, 3, bad)


def test_leave_one_out_envelope_contains_the_pattern_on_the_mirror_axes():
    # The documented exception: on u = 0 / v = 0 a mirror image of the pattern is the pattern.
    p = [np.zeros((5, 5)), np.zeros((5, 5))]
    p[0][2, 0] = 7.0  # on the line u = 0 (centre row of axis 0), off the centre
    env = ss.leave_one_out_envelopes(p, 0.0)[0]
    assert env[2, 0] == pytest.approx(7.0 / 7)  # 1 of the 7 images is pattern 0 itself
    q = [np.zeros((5, 5)), np.zeros((5, 5))]
    q[0][1, 3] = 7.0  # off the axes: its mirror images land elsewhere, the value is excluded
    env = ss.leave_one_out_envelopes(q, 0.0)[0]
    assert env[1, 3] == 0.0 and env[3, 3] == env[1, 1] == env[3, 1] == pytest.approx(1.0)
    c = [np.zeros((5, 5)), np.zeros((5, 5))]
    c[0][2, 2] = 7.0
    assert ss.leave_one_out_envelopes(c, 0.0)[0][2, 2] == pytest.approx(3.0)  # centre: 3/7


def test_correlated_aperture_fields_and_synthetic_speckle():
    rng = np.random.default_rng(2026)
    a = ss.circular_gaussian_field(rng, 128)
    assert a.shape == (128, 128) and np.mean(np.abs(a) ** 2) == pytest.approx(1.0, abs=0.03)
    assert abs(np.mean(a * a)) < 0.03  # circular: E[a^2] = 0
    assert np.array_equal(ss.correlated_field(rng, a, 1.0), a)
    for mu in (0.0, 0.6):
        b = ss.correlated_field(rng, a, mu)
        rho = np.vdot(a, b) / math.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
        assert rho.real == pytest.approx(mu, abs=0.03) and abs(rho.imag) < 0.03
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            ss.correlated_field(rng, a, bad)
    with pytest.raises(ValueError):
        ss.correlated_field(rng, np.ones(4), 0.5)
    with pytest.raises(ValueError):
        ss.circular_gaussian_field(rng, 1)
    plane = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    f0 = ss.circular_gaussian_field(np.random.default_rng(5), 256)
    p0 = ss.synthetic_far_zone_speckle(None, plane, envelope_rms=0.3, aperture=f0)
    assert np.array_equal(
        p0,
        ss.synthetic_far_zone_speckle(
            np.random.default_rng(1), plane, envelope_rms=0.3, aperture=f0
        ),
    )
    # Gaussian moment theorem: the intensity patterns of fields with correlation mu correlate
    # with mu^2 (radiant intensity in (u, v), flat envelope; ~1500 speckles in total).
    u = ss.direction_cosine_axis(0.66, WL / (2 * W0) / 8)
    rhos = []
    rng2 = np.random.default_rng(77)
    for _ in range(3):
        fa = ss.circular_gaussian_field(rng2, 256)
        fb = ss.correlated_field(rng2, fa, math.sqrt(0.5))
        ia, ib = (
            ss.plane_to_direction_cosines(
                ss.synthetic_far_zone_speckle(None, plane, aperture=f), -2e-3, plane.spacing, Z, u
            )
            for f in (fa, fb)
        )
        rhos.append(sb.correlation_coefficient(ia.ravel(), ib.ravel()))
    assert np.mean(rhos) == pytest.approx(0.5, abs=0.1)
    with pytest.raises(ValueError):
        ss.synthetic_far_zone_speckle(None, plane)
    with pytest.raises(ValueError):
        ss.synthetic_far_zone_speckle(None, plane, aperture=np.ones((8, 8)))


def test_analyse_ensemble_on_synthetic_far_zone_speckle():
    # Two seeds of exact Gaussian-spot speckle with a diffuse halo (rms 0.35 in u) on the
    # WP-S2 plane: after resampling, leave-one-out normalisation and windowing the harness
    # finds contrast ~1, the WP-S1 speckle size and an exponential PDF.
    plane = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    rng = np.random.default_rng(20261026)
    pats = [ss.synthetic_far_zone_speckle(rng, plane, envelope_rms=0.35) for _ in range(2)]
    res = ss.analyse_ensemble(pats, plane)
    st = res["stats"]
    assert res["du"] == pytest.approx(WL / (2 * W0) / 8)
    assert res["pixel_size"] == pytest.approx(res["du"] * 1e-3)
    assert st["speckle_size_expected"] == pytest.approx(1.060041 * WL * 1e-3 / (2 * W0), rel=1e-6)
    assert 0.2 < res["window_u"] <= 0.66 and res["window_speckles"] > 2.0
    assert res["windows"][0].shape == (2 * res["window_half"] + 1,) * 2
    assert st["mean_contrast"] == pytest.approx(1.0, abs=0.15)
    assert abs(st["speckle_size_rel_error"]) < 0.10
    assert res["ks_spacing_used"] in (2.0, 1.0)
    assert not st["ks_rejects_5pct"]
    assert all(c > 1.2 for c in res["contrast_plane"])  # raw plane pattern: halo + 1/r^2
    assert abs(res["rho_plane"][(0, 1)]) < 0.6
    assert [x.size for x in res["ks_samples"]] == [x.size for x in res["ks_samples"]][:1] * 2
    assert sum(x.size for x in res["ks_samples_one"]) > sum(x.size for x in res["ks_samples"])
    with pytest.raises(ValueError):
        ss.analyse_ensemble([np.ones((5, 5))], plane)


def test_phase_screen_surrogate():
    plane = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    h = ss.unit_height_screen(np.random.default_rng(3), plane, 1e-6)
    assert h.shape == (256, 256)
    assert h.mean() == pytest.approx(0.0, abs=1e-12) and h.std() == pytest.approx(1.0)
    # correlation exp(-r^2 / Lc^2): at a lag of Lc it is 1/e (periodic estimate, 43 Lc wide)
    pitch = WL / (256 * (WL / (2 * W0)) / 16)
    lag = round(1e-6 / pitch)
    c = np.mean(h * np.roll(h, lag, axis=0)) + np.mean(h * np.roll(h, lag, axis=1))
    assert 0.5 * c == pytest.approx(math.exp(-((lag * pitch / 1e-6) ** 2)), abs=0.08)
    with pytest.raises(ValueError):
        ss.unit_height_screen(np.random.default_rng(3), plane, 0.2e-6)  # not resolved
    # sigma = 0: the specular reflection of the spot (smooth, no speckle); large sigma: speckle.
    smooth = ss.phase_screen_far_zone_speckle(h, 0.0, plane)
    c0 = smooth[80, 80]
    assert np.unravel_index(np.argmax(smooth), smooth.shape) == (80, 80)
    assert smooth[80, 90] < c0 and smooth[80, 70] < c0
    rough = ss.phase_screen_far_zone_speckle(h, 250e-9, plane)
    assert rough.sum() == pytest.approx(smooth.sum(), rel=0.6)  # same power, spread out
    assert rough.max() < 0.2 * c0
    # The coherent spot of the smooth case carries ~exp(-(2 k sigma)^2) at small sigma.
    weak = ss.phase_screen_far_zone_speckle(h, 20e-9, plane)
    assert weak[80, 80] / c0 == pytest.approx(ss.coherent_fraction(20e-9, WL), abs=0.1)


def test_vacuum_wave_impedance_requires_a_real_value():
    # The binding returns the impedance as a complex number; the driver keeps its real part only
    # when the imaginary part is exactly zero (vacuum) and raises otherwise (no silent drop).
    class Medium:
        def __init__(self, z):
            self.z = z

        def wave_impedance(self, omega):
            assert omega == 3.0
            return self.z

    assert ss.vacuum_wave_impedance(Medium(complex(376.73, 0.0)), 3.0) == 376.73
    assert isinstance(ss.vacuum_wave_impedance(Medium(complex(376.73, 0.0)), 3.0), float)
    with pytest.raises(ValueError, match="vacuum wave impedance must be real"):
        ss.vacuum_wave_impedance(Medium(complex(376.73, -1e-9)), 3.0)
    vac = sb.vacuum()
    omega = 2 * math.pi * 299792458.0 / WL
    assert ss.vacuum_wave_impedance(vac, omega) == pytest.approx(376.730313668, rel=1e-9)


def _write_case(root, sigma_nm, seed, E, plane, **fields):
    """A finished case directory; ``fields`` (pol, theta_deg) go into the record and the tag."""
    suffix = "".join(
        f"_t{v:g}" if k == "theta_deg" and v != 0 else f"_pol{v}" if k == "pol" and v != "p" else ""
        for k, v in fields.items()
    )
    d = root / f"si_L8um_s{sigma_nm:g}nm_Lc1um_seed{seed}{suffix}"
    d.mkdir()
    np.save(d / "E.npy", E)
    rec = dict(
        **fields,
        sigma_nm=float(sigma_nm),
        seed=seed,
        N2=741120,
        it=100,
        setup_s=1.0,
        solve_s=2.0,
        plane_s=3.0,
        far_s=4.0,
        peak_gb=5.0,
        R=0.38,
        contrast_global=2.0,
    )
    from dataclasses import asdict

    (d / "meta.json").write_text(
        json.dumps(dict(record=rec, plane=asdict(plane))), encoding="utf-8"
    )


def test_stats_script_on_synthetic_case_directories(tmp_path, capsys):
    # Four "sigma" values x two seeds of fully developed speckle with shrinking halos written
    # as case directories (E along x): the script reads them, groups by sigma and reports the
    # three docs/05 rows with the WP-S1 harness.
    plane = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    rng = np.random.default_rng(424242)
    for sigma_nm, halo in [(100, 0.25), (150, 0.35), (200, 0.45), (250, 0.6)]:
        for seed in (1, 2):
            inten = ss.synthetic_far_zone_speckle(rng, plane, envelope_rms=halo)
            E = np.zeros((161 * 161, 3), dtype=complex)
            E[:, 0] = np.sqrt(inten.ravel()) * np.exp(2j * math.pi * rng.random(161 * 161))
            E[:, 2] = 0.1 * E[:, 0]  # not co-polarised: ignored
            _write_case(tmp_path, sigma_nm, seed, E, plane)
    (tmp_path / "unfinished").mkdir()
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "speckle_sweep_stats.py"
    spec = importlib.util.spec_from_file_location("speckle_sweep_stats", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out_json = tmp_path / "summary.json"
    assert module.main([str(tmp_path), "--markdown", "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert "docs/05 row 42 per σ" in text and "Row 43" in text and "Row 44" in text
    assert "σ-pooled (supplementary, not the docs/05 gate" in text
    assert "Row 42 gate" not in text
    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert [s["sigma_nm"] for s in out["sigmas"]] == [100, 150, 200, 250]
    assert len(out["cases"]) == 8 and len(out["corr_step"]) == 6
    assert all(s["fully_developed"] for s in out["sigmas"])
    gate = out["sigma_pooled_ks"]["result"]  # supplementary pool (not the docs/05 gate)
    assert gate["n"] >= 50 and not gate["rejects"]
    # the docs/05 gate per sigma: the summary repeats the per-sigma values, nothing recomputed
    per = out["row42_per_sigma"]
    assert per["spacing"] == 2.0
    assert [r["sigma_nm"] for r in per["sigmas"]] == [100, 150, 200, 250]
    for r, s in zip(per["sigmas"], out["sigmas"]):
        assert r["n"] == s["ks_spacing2_n"] and r["result"] == s["ks_spacing2"]
        assert r["evaluable"] == (s["ks_spacing2"] is not None)
        assert f"σ = {s['sigma_nm']:g} nm: n = {s['ks_spacing2_n']}, " in text
    assert len(out["size_within_10pct"]) == 4  # all four fully developed
    assert out["contrast_monotone"]["strict"] and out["contrast_monotone"]["complete"]
    # two seeds per entry: the size error scatters by 8-15 % (calibration), so only plumbing
    # is checked per entry; the mean over the entries is tighter
    assert all(abs(s["size_rel_error"]) < 0.3 for s in out["sigmas"])
    assert abs(np.mean([s["size_rel_error"] for s in out["sigmas"]])) < 0.1
    E0 = np.load(tmp_path / "si_L8um_s100nm_Lc1um_seed1" / "E.npy")
    assert out["cases"][0]["contrast_plane"] == pytest.approx(
        sb.speckle_contrast(np.abs(E0[:, 0]) ** 2), rel=1e-12
    )
    assert out["pooled_ks_one"]["n"] > gate["n"]

    # Row-42 sensitivity: the baseline row is the sigma-pooled pool; one change at a time (short
    # list).
    args = module.build_parser().parse_args([str(tmp_path)])
    cases = module.load_cases(tmp_path)
    variants = (
        ("baseline", {}),
        ("offset 3", dict(offset=3)),
        ("drop", dict(drop_lowest=True)),
        ("radius", dict(envelope_radius=0.75)),
    )
    sens = module.sensitivity(cases, args, variants)
    rows = {r["label"]: r for r in sens["rows"]}
    for key in ("n", "statistic", "modified", "rejects"):
        assert rows["baseline"][key] == gate[key]
    assert rows["drop"]["sigmas"] == [150, 200, 250] and rows["drop"]["n"] < gate["n"]
    assert rows["offset 3"]["n"] <= gate["n"] and rows["offset 3"]["statistic"] != gate["statistic"]
    assert sens["n_variants"] == sum("rejects" in r for r in sens["rows"]) - 1
    assert sens["n_rejecting"] == sum(r.get("rejects", False) for r in sens["rows"][1:])
    with pytest.raises(ValueError):
        module.sensitivity(cases, args, (("bad", dict(radius=1.0)),))
    # --sensitivity through main (the full list is minutes of work on real data: shortened)
    module.SENSITIVITY_VARIANTS = variants[:2]
    out_json2 = tmp_path / "summary_sens.json"
    assert (
        module.main([str(tmp_path), "--markdown", "--sensitivity", "--json", str(out_json2)]) == 0
    )
    text = capsys.readouterr().out
    assert "| offset 3 |" in text and "Variants rejecting:" in text
    sens2 = json.loads(out_json2.read_text(encoding="utf-8"))["sensitivity"]
    assert [r["label"] for r in sens2["rows"]] == ["baseline", "offset 3"]
    # JSON output (no --markdown): the per-sigma gate lines go to stderr, stdout stays JSON
    module.SENSITIVITY_VARIANTS = variants[:1]
    assert module.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["row42_per_sigma"]["spacing"] == 2.0
    assert captured.err.startswith("docs/05 row 42 per σ")
    assert captured.err.count(" nm: n = ") == 4

    # A partially developed sigma (the 100 nm patterns relabelled 50 nm, exp(-(2 k sigma)^2) =
    # 0.21): listed in the per-sigma gate lines as not gated, row 44 reports it but does not
    # judge it, and it stays out of the sigma-pooled pool.
    relabelled = [
        dict(c, rec=dict(c["rec"], sigma_nm=50.0) if c["rec"]["sigma_nm"] == 100 else c["rec"])
        for c in cases
    ]
    out50 = module.analyse(tmp_path, args, relabelled)
    assert [s["fully_developed"] for s in out50["sigmas"]] == [False, True, True, True]
    assert 50.0 not in out50["size_within_10pct"] and len(out50["size_within_10pct"]) == 3
    assert out50["sigma_pooled_ks"]["sigmas"] == [150, 200, 250]
    first = out50["row42_per_sigma"]["sigmas"][0]
    assert first["sigma_nm"] == 50.0 and not first["evaluable"]
    lines = module.row42_per_sigma_lines(out50)
    assert len(lines) == 5 and lines[1].endswith("partially developed, not gated")
    module.print_markdown(out50)
    text = capsys.readouterr().out
    assert "reported, not judged (partially developed): 50" in text
    assert "- σ = 50 nm: n = " in text


def _stats_script():
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "speckle_sweep_stats.py"
    spec = importlib.util.spec_from_file_location("speckle_sweep_stats", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stats_script_analyses_p_only_and_one_angle(tmp_path, capsys):
    # The statistics take |E_x|^2 as the co-polarised intensity, i.e. that of p: s cases of the
    # same surfaces (driver --pol p,s) must be left out and reported, never pooled, and a
    # directory mixing incidence angles (--theta-deg 0,30) is refused unless one is selected.
    plane = ss.plane_sampling(WL, Z, W0, 4e-3, 161)
    rng = np.random.default_rng(31415)

    def field(halo, axis):
        inten = ss.synthetic_far_zone_speckle(rng, plane, envelope_rms=halo)
        E = np.zeros((161 * 161, 3), dtype=complex)
        E[:, axis] = np.sqrt(inten.ravel()) * np.exp(2j * math.pi * rng.random(161 * 161))
        return E

    ref = tmp_path / "ref"  # p only, the records as the driver writes them
    mixed = tmp_path / "mixed"  # the same p cases plus s cases (E along y) and a 30 deg case
    ref.mkdir()
    mixed.mkdir()
    for sigma_nm, halo in [(150, 0.35), (250, 0.6)]:
        for seed in (1, 2):
            E = field(halo, 0)
            _write_case(ref, sigma_nm, seed, E, plane, pol="p", theta_deg=0.0)
            _write_case(mixed, sigma_nm, seed, E, plane, pol="p", theta_deg=0.0)
            _write_case(mixed, sigma_nm, seed, field(halo, 1), plane, pol="s", theta_deg=0.0)
    _write_case(mixed, 150, 1, field(0.35, 0), plane, pol="p", theta_deg=30.0)
    module = _stats_script()

    with pytest.raises(ValueError, match=r"mixes incidence angles \[0.0, 30.0\] deg"):
        module.scan_cases(mixed)
    with pytest.raises(ValueError, match="--theta-deg"):
        module.main([str(mixed)])
    cases, skipped = module.scan_cases(mixed, theta_deg=0.0)
    assert [c["name"] for c in cases] == [c["name"] for c in module.load_cases(ref)]
    assert all(c["rec"]["pol"] == "p" and c["rec"]["theta_deg"] == 0.0 for c in cases)
    assert sorted(s["name"] for s in skipped) == sorted(
        [f"si_L8um_s{sg}nm_Lc1um_seed{k}_pols" for sg in (150, 250) for k in (1, 2)]
        + ["si_L8um_s150nm_Lc1um_seed1_t30"]
    )
    assert sum(s["pol"] == "s" for s in skipped) == 4
    assert all(s["reason"].startswith("pol s") for s in skipped if s["pol"] == "s")
    t30 = next(s for s in skipped if s["theta_deg"] == 30.0)
    assert t30["reason"] == "theta_deg 30 (selected 0)"
    capsys.readouterr()

    # through main: the selected cases give the numbers of the p-only directory, the skipped
    # ones are reported on stderr and in the JSON
    assert module.main([str(ref), "--json", str(tmp_path / "ref.json")]) == 0
    assert "skipped" not in capsys.readouterr().err
    out_mixed = tmp_path / "mixed.json"
    assert module.main([str(mixed), "--theta-deg", "0", "--json", str(out_mixed)]) == 0
    err = capsys.readouterr().err
    assert "skipped 5 finished case(s), not analysed:" in err
    assert "  si_L8um_s250nm_Lc1um_seed2_pols: pol s (|E_x|^2 is the co-polarisation" in err
    a = json.loads((tmp_path / "ref.json").read_text(encoding="utf-8"))
    b = json.loads(out_mixed.read_text(encoding="utf-8"))
    assert a["skipped_cases"] == [] and len(b["skipped_cases"]) == 5
    for key in ("cases", "sigmas", "corr_seed", "corr_step", "row42_per_sigma"):
        assert a[key] == b[key], key
    assert [s["n_seeds"] for s in b["sigmas"]] == [2, 2]
    # the 30 deg case alone, and a directory without p cases: nothing to analyse
    t30_cases, t30_skipped = module.scan_cases(mixed, theta_deg=30.0)
    assert [c["name"] for c in t30_cases] == ["si_L8um_s150nm_Lc1um_seed1_t30"]
    assert len(t30_skipped) == 8
    only_s = tmp_path / "only_s"
    only_s.mkdir()
    _write_case(only_s, 150, 1, field(0.35, 1), plane, pol="s", theta_deg=0.0)
    s_cases, s_skipped = module.scan_cases(only_s)
    assert s_cases == [] and [s["pol"] for s in s_skipped] == ["s"]
    with pytest.raises(SystemExit, match=r"no finished p cases \(1 skipped\)"):
        module.load_cases(only_s)


def test_calibration_script_runs(capsys):
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "speckle_sweep_calibration.py"
    spec = importlib.util.spec_from_file_location("speckle_sweep_calibration", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.main(["--model", "gaussian", "--reps", "1", "--halos", "0.3,0.45"]) == 0
    assert module.main(["--model", "phase-screen", "--reps", "1", "--sigmas", "150,250"]) == 0
    text = capsys.readouterr().out
    assert "pooled KS" in text and "σ step" in text
    # correlated maps: the same two fields at every entry (mu = 1), sample sizes as mean [range]
    assert (
        module.main(
            ["--model", "gaussian", "--reps", "2", "--halos", "0.35,0.45", "--map-correlation", "1"]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "map correlation mu 1" in text and "| halo step (u) |" in text and "μ²" in text
    assert module._n_range([76, 90, 76]) == "81 [76–90]" and module._n_range([8, 8]) == "8"
    assert module._n_range([]) == "–"
    for bad in (
        ["--model", "phase-screen", "--map-correlation", "0.5"],
        ["--map-correlation", "1.5"],
    ):
        with pytest.raises(SystemExit):
            module.main(bad + ["--reps", "1"])


# --------------------------------------------------------------------------------------------
# Several excitations on one assembled operator (WP-KR3)
#
# A tiny Si patch (L = 1 um, h = 200 nm, 2N = 3780) with reduced quadrature and an octree of two
# levels (every interaction near, no far field): the full kernels and the automatic octree cost
# about 65 s of assembly and 20 s per solve here (the far operator of the 5.6 um deep box
# dominates), the reduced ones about 0.5 s and 1.5 s at tol 1e-2. The substitution wraps the
# Simulation constructor, so run_case and run_sequence get the same operator; the property under
# test (a sequence reproduces each case alone bitwise) does not depend on the quadrature.

KR3_KERNELS = dict(
    quad_degree_far=1,
    quad_degree_near=2,
    quad_degree_sing=2,
    outer_grading_levels=0,
    target_accuracy=0.0,
)
KR3_MLFMM = dict(automatic_leaf_size=False, min_box_size_lambda=4.0)
KR3_FILES = (
    "currents.npy",
    "intensity.npy",
    "E.npy",
    "far_F.npy",
    "residual_history.npy",
    "incident_paraxial_intensity.npy",
    "far_dirs.npy",
    "far_weights.npy",
)
KR3_FAR = (6, 12)
KR3_ARGS = [
    "--L",
    "1e-6",
    "--h",
    "2e-7",
    "--sigma",
    "1e-7",
    "--seed",
    "1",
    "--solver-tol",
    "1e-2",
    "--plane-size",
    "4e-3",
    "--plane-n",
    "9",
    "--far-grid",
    "6,12",
]
#: Record keys of run_case (frozen: run_case must keep its ROW line, WP-KR3).
RUN_CASE_KEYS = (
    "material",
    "L_um",
    "h_nm",
    "sigma_nm",
    "Lc_um",
    "seed",
    "pol",
    "theta_deg",
    "z_mm",
    "plane_mm",
    "plane_n",
    "tol",
    "waist_um",
    "form",
    "precond",
    "d0",
    "lq",
    "N2",
    "triangles",
    "rmax_nm",
    "mesh_s",
    "setup_s",
    "peak_setup_gb",
    "leaf_nm",
    "levels",
    "solve_s",
    "it",
    "conv",
    "res",
    "true_res",
    "plane_s",
    "contrast_global",
    "I_mean",
    "I_inc_max",
    "far_s",
    "R",
    "peak_gb",
)


def _kr3_simulation(real):
    # The reduced operator has no MLFMM far interactions (two levels, all near); the
    # determinism of the far path under set_excitation is covered bitwise by
    # tests/python/test_operator_reuse.py (WP-KR1: reused vs fresh Simulation on the mlfmm PML
    # route, 3 octree levels with an MLFMM far operator) and tests/unit/test_operator_reuse.cpp.
    def make(*args, **kw):
        kw["kernels"] = KR3_KERNELS
        kw["mlfmm"] = dict(kw.get("mlfmm") or {}, **KR3_MLFMM)
        return real(*args, **kw)

    return make


def _kr3_case(theta_deg=0.0, pol="p", **kw):
    return ss.CaseParams(
        L=1e-6,
        sigma=100e-9,
        seed=1,
        h=200e-9,
        theta_deg=theta_deg,
        polarization=pol,
        **(dict(solver_tol=1e-2) | kw),
    )


def _kr3_plane(p):
    box = sb.rough_surface_box_params(sb.silicon_500nm(), sb.vacuum(), p.wavelength, p.sigma, p.h)
    return ss.plane_sampling(p.wavelength, Z, p.default_waist(box.beam_waist_fraction), 4e-3, 9)


def _quiet(msg):
    pass


@pytest.fixture(scope="module")
def kr3(tmp_path_factory):
    """run_case of 0 deg p / s and 30 deg p / s alone, and the driver's sequence
    ``--theta-deg 0,30 --pol p,s`` (run_sequence: built and assembled with 0 deg p, then the
    0 deg s, 30 deg p and 30 deg s beams swapped in) of the same cases, with the reduced
    operator."""
    root = tmp_path_factory.mktemp("kr3")
    cases = [_kr3_case(0.0), _kr3_case(0.0, "s"), _kr3_case(30.0), _kr3_case(30.0, "s")]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "Simulation", _kr3_simulation(core.Simulation))
        single = [
            ss.run_case(c, _kr3_plane(c), root / "single" / c.tag(), far_grid=KR3_FAR, log=_quiet)
            for c in cases
        ]
        driver = _driver()
        rows = root / "rows.txt"
        out = root / "seq"
        code = driver.main(
            KR3_ARGS
            + ["--theta-deg", "0,30", "--pol", "p,s", "--out", str(out), "--rows", str(rows)]
        )
        assert code == 0
    yield dict(root=root, cases=cases, single=single, seq=out, rows=rows)


def test_run_sequence_reproduces_each_case_alone(kr3):
    lines = kr3["rows"].read_text(encoding="utf-8").splitlines()
    seq = ss.read_rows(lines)
    assert [(r["theta_deg"], r["pol"]) for r in seq] == [(0, "p"), (0, "s"), (30, "p"), (30, "s")]
    assert len(lines) == 4
    assert [c.tag().endswith("_pols") for c in kr3["cases"]] == [False, True, False, True]
    for c, ref, rec in zip(kr3["cases"], kr3["single"], seq):
        a, b = kr3["root"] / "single" / c.tag(), kr3["seq"] / c.tag()
        for f in KR3_FILES:
            assert np.array_equal(np.load(a / f), np.load(b / f)), (c.tag(), f)
        stored = json.loads((b / "meta.json").read_text(encoding="utf-8"))["record"]
        assert stored["R"] == ref["R"] and stored["it"] == ref["it"]
        assert stored["res"] == ref["res"] and stored["conv"] == ref["conv"]
        assert stored["waist_um"] == ref["waist_um"]
        # the row line carries 6 significant digits of the same values
        assert rec["it"] == ref["it"] and rec == ss.parse_row(
            (b / "row.txt").read_text(encoding="utf-8")
        )
    # the two angles got their own default waists (L cos(theta) / 6, option b1), both
    # polarisations of an angle the same one
    assert seq[2]["waist_um"] == pytest.approx(
        seq[0]["waist_um"] * math.cos(math.radians(30)), rel=1e-5
    )
    assert (seq[1]["waist_um"], seq[3]["waist_um"]) == (seq[0]["waist_um"], seq[2]["waist_um"])
    # the s beam was really swapped in: its fields differ from those of p at the same angle
    for k in (0, 2):
        p_dir, s_dir = (kr3["seq"] / kr3["cases"][k + i].tag() for i in (0, 1))
        for f in ("currents.npy", "E.npy", "far_F.npy"):
            assert not np.array_equal(np.load(p_dir / f), np.load(s_dir / f)), (k, f)


def test_run_sequence_records_reuse_and_keeps_the_run_case_keys(kr3):
    assert list(kr3["single"][0]) == list(RUN_CASE_KEYS)  # run_case unchanged
    metas = [
        json.loads((kr3["seq"] / c.tag() / "meta.json").read_text(encoding="utf-8"))
        for c in kr3["cases"]
    ]
    ref_meta = json.loads(
        (kr3["root"] / "single" / kr3["cases"][0].tag() / "meta.json").read_text(encoding="utf-8")
    )
    first, *later = (m["record"] for m in metas)
    assert list(first) == list(RUN_CASE_KEYS) + ["reused", "swap_s"]
    assert (first["reused"], first["swap_s"]) == (0, 0.0)
    assert first["mesh_s"] > 0.0 and first["setup_s"] > 0.0
    for rec in later:
        assert list(rec) == list(first)
        assert rec["reused"] == 1 and rec["swap_s"] > 0.0
        assert rec["setup_s"] == rec["swap_s"] and rec["mesh_s"] == 0.0
        assert rec["leaf_nm"] == first["leaf_nm"] and rec["levels"] == first["levels"] == 2
    for m in metas:
        assert list(m) == list(ref_meta)  # record, params, info, plane, report, report_setup
        assert list(m["info"]) == list(ref_meta["info"])
        assert m["info"]["rmax"] == ref_meta["info"]["rmax"]
        # the shared mesh time stays in info for every case (the record has it on case 0 only)
        assert m["info"]["mesh_s"] == first["mesh_s"]
    assert [(m["params"]["theta_deg"], m["params"]["polarization"]) for m in metas] == [
        (0.0, "p"),
        (0.0, "s"),
        (30.0, "p"),
        (30.0, "s"),
    ]
    row = (kr3["seq"] / kr3["cases"][1].tag() / "row.txt").read_text(encoding="utf-8")
    assert row.rstrip("\n").endswith(f" reused=1 swap_s={later[0]['swap_s']:.6g}")


def test_run_sequence_rejects_different_surfaces_and_bad_planes(tmp_path):
    base = _kr3_case()
    plane = _kr3_plane(base)
    for change, name in [
        (dict(sigma=50e-9), "sigma"),
        (dict(seed=2), "seed"),
        (dict(h=100e-9), "h"),
        (dict(material="ag"), "material"),
        (dict(mlfmm=dict(accuracy_digits=5)), "mlfmm"),
        (dict(preconditioner="jacobi"), "preconditioner"),
        (dict(region_radius_factor=1.1), "region_radius_factor"),
    ]:
        other = ss.CaseParams(**(asdict(_kr3_case(30.0)) | change))
        with pytest.raises(ValueError, match=f"operator field '{name}'"):
            ss.run_sequence([base, other], plane, tmp_path, log=_quiet)
    with pytest.raises(ValueError, match="duplicate case tags"):
        ss.run_sequence([base, _kr3_case(solver_tol=1e-3)], plane, tmp_path, log=_quiet)
    with pytest.raises(ValueError, match="at least one case"):
        ss.run_sequence([], plane, tmp_path, log=_quiet)
    with pytest.raises(ValueError, match="one per case"):
        ss.run_sequence([base, _kr3_case(30.0)], [plane], tmp_path, log=_quiet)
    # solver settings and the excitation may differ (checked without building anything)
    ss.check_sequence([base, _kr3_case(30.0, "s", solver_tol=1e-3, max_iter=5, waist=1e-7)])
    # one plane for cases of different default waists: refused before the assembly
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "Simulation", _refuse_simulation)
        with pytest.raises(ValueError, match="plane sampling was computed for"):
            ss.run_sequence([base, _kr3_case(30.0)], plane, tmp_path, log=_quiet)
    assert not any(tmp_path.iterdir())


def _refuse_simulation(*args, **kw):
    raise AssertionError("no Simulation may be built here")


class _FakeSimulation:
    """Records construction, assemble, set_excitation and the solver settings of every solve
    (``solves``: incidence angle, polarisation, tol, max_iter); solves nothing (writes zeros)."""

    log = []
    solves = []

    def __init__(self, mesh, beam, **kw):
        self.beam = beam
        self.formulation, self.preconditioner, self.num_unknowns = "Formulation.ICTF", "none", 6
        _FakeSimulation.log.append(("init", beam.incidence_angle))

    def assemble(self):
        _FakeSimulation.log.append(("assemble", self.beam.incidence_angle))

    def set_excitation(self, beam):
        self.beam = beam
        _FakeSimulation.log.append(("swap", beam.incidence_angle))

    def report(self):
        return "fake"

    def solve(self, tol, max_iter):
        _FakeSimulation.log.append(("solve", self.beam.incidence_angle))
        pol = str(self.beam.polarization).split(".")[-1].lower()  # Polarization.P -> "p"
        _FakeSimulation.solves.append((self.beam.incidence_angle, pol, tol, max_iter))

        class Res:
            iterations, converged, residual_history, true_relative_residual = 1, True, [0.5], 0.5

        return Res()

    @property
    def currents(self):
        return np.zeros(6, dtype=complex)

    def field(self, pts, kind):
        return np.ones((len(pts), 3), dtype=complex), None

    def far_field(self, dirs):
        return np.ones((len(dirs), 3), dtype=complex)

    def incident_power(self):
        return 1.0


def test_run_sequence_skip_existing_builds_with_the_first_missing_case(tmp_path):
    # per-case solver settings (they may differ within a sequence) reach each solve
    cases = [
        _kr3_case(0.0),
        _kr3_case(30.0, solver_tol=1e-3, max_iter=50),
        _kr3_case(30.0, "s", solver_tol=5e-3, max_iter=7),
    ]
    planes = [_kr3_plane(c) for c in cases]
    first = tmp_path / cases[0].tag()
    first.mkdir()
    stored = "ROW material=si theta_deg=0 it=7 R=0.25"
    (first / "row.txt").write_text(stored + "\n", encoding="utf-8")
    _FakeSimulation.log = []
    _FakeSimulation.solves = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "Simulation", _FakeSimulation)
        recs = ss.run_sequence(
            cases, planes, tmp_path, far_grid=KR3_FAR, skip_existing=True, log=_quiet
        )
    t30 = math.radians(30.0)
    # built with the 30 deg p beam, assembled once, one swap for 30 deg s, two solves
    assert _FakeSimulation.log == [
        ("init", t30),
        ("assemble", t30),
        ("solve", t30),
        ("swap", t30),
        ("solve", t30),
    ]
    assert _FakeSimulation.solves == [(t30, "p", 1e-3, 50), (t30, "s", 5e-3, 7)]
    assert [r["tol"] for r in recs[1:]] == [1e-3, 5e-3]
    assert recs[0] == ss.parse_row(stored)
    assert (first / "row.txt").read_text(encoding="utf-8") == stored + "\n"  # untouched
    assert sorted(p.name for p in first.iterdir()) == ["row.txt"]
    assert (recs[1]["reused"], recs[1]["swap_s"], recs[2]["reused"]) == (0, 0.0, 1)
    assert recs[1]["mesh_s"] > 0.0 and recs[2]["mesh_s"] == 0.0
    for c, rec in zip(cases[1:], recs[1:]):
        assert (tmp_path / c.tag() / "row.txt").is_file()
        assert (
            ss.parse_row((tmp_path / c.tag() / "row.txt").read_text(encoding="utf-8"))["pol"]
            == c.polarization
        )
    # everything present: nothing is built, the stored rows come back
    _FakeSimulation.log = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "Simulation", _refuse_simulation)
        again = ss.run_sequence(cases, planes, tmp_path, skip_existing=True, log=_quiet)
    assert again[0] == recs[0] and [r["it"] for r in again[1:]] == [1, 1]
    # without skip_existing every case is solved again (the first one assembles)
    _FakeSimulation.log = []
    _FakeSimulation.solves = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "Simulation", _FakeSimulation)
        recs = ss.run_sequence(cases[:2], planes[:2], tmp_path, far_grid=None, log=_quiet)
    assert [e[0] for e in _FakeSimulation.log] == ["init", "assemble", "solve", "swap", "solve"]
    assert [x[2:] for x in _FakeSimulation.solves] == [(1e-2, None), (1e-3, 50)]
    assert recs[0]["reused"] == 0 and "R" not in recs[0]


def test_driver_dispatches_lists_to_run_sequence(tmp_path, monkeypatch, capsys):
    # The Cartesian product (angles outer, polarisations inner), one plane per case with its own
    # waist; a single angle and polarisation still goes through run_case. No solve.
    module = _driver()
    calls = []

    def fake_sequence(cases, planes, out, *, far_grid, skip_existing, log):
        calls.append(("seq", cases, planes, far_grid, skip_existing))
        return [
            dict(theta_deg=c.theta_deg, pol=c.polarization, reused=int(k > 0))
            for k, c in enumerate(cases)
        ]

    def fake_case(p, plane, out, *, far_grid, log):
        calls.append(("case", p, plane, far_grid))
        return dict(theta_deg=p.theta_deg, pol=p.polarization)

    monkeypatch.setattr(ss, "run_sequence", fake_sequence)
    monkeypatch.setattr(ss, "run_case", fake_case)
    rows = tmp_path / "rows.txt"
    base = [
        "--sigma",
        "1e-7",
        "--seed",
        "1",
        "--L",
        "1e-6",
        "--out",
        str(tmp_path),
        "--rows",
        str(rows),
    ]
    assert module.main(base + ["--theta-deg", "0,30", "--pol", "p,s", "--skip-existing"]) == 0
    kind, cases, planes, far, skip = calls[-1]
    assert kind == "seq" and far == (90, 180) and skip
    assert [(c.theta_deg, c.polarization) for c in cases] == [
        (0, "p"),
        (0, "s"),
        (30, "p"),
        (30, "s"),
    ]
    waists = [pl.waist for pl in planes]
    assert waists[0] == waists[1] == pytest.approx(1e-6 / 6)
    assert waists[2] == waists[3] == pytest.approx(1e-6 * math.cos(math.radians(30)) / 6)
    out = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("ROW")]
    assert out == [
        ss.format_row(r)
        for r in fake_sequence(cases, planes, None, far_grid=None, skip_existing=False, log=None)
    ]
    assert rows.read_text(encoding="utf-8").splitlines() == out
    # --waist applies to every case
    module.main(base + ["--theta-deg", "0,30", "--waist", "1e-7"])
    assert all(c.waist == 1e-7 for c in calls[-1][1])
    assert all(pl.waist == 1e-7 for pl in calls[-1][2])
    # one value: run_case, as before
    module.main(base + ["--theta-deg", "30", "--far-grid", "0"])
    kind, p, plane, far = calls[-1]
    assert kind == "case" and p.theta_deg == 30.0 and p.polarization == "p" and far is None
    assert plane.waist == pytest.approx(1e-6 * math.cos(math.radians(30)) / 6)
