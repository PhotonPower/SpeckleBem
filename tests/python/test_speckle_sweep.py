"""Unit tests of specklebem.speckle_sweep (WP-S2): plane sampling, ROW records, mesh helper,
analysis of saved patterns (on synthetic speckle).

Pure functions with fixed inputs and seeds; no solve. The driver benchmarks/speckle_sweep.py
and the statistics script benchmarks/speckle_sweep_stats.py are imported by path.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import specklebem as sb
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
    rec = dict(extra="x1", seed=2, L_um=8.0, sigma_nm=150.0, conv=True, res=9.87654321e-5,
               N2=741120, form="ictf")
    line = ss.format_row(rec)
    assert line.startswith("ROW L_um=8 sigma_nm=150 seed=2 form=ictf N2=741120 conv=1 ")
    assert line.endswith(" extra=x1")
    back = ss.parse_row(line)
    assert back == dict(L_um=8, sigma_nm=150, seed=2, form="ictf", N2=741120, conv=1,
                        res=pytest.approx(9.87654e-5, rel=1e-12), extra="x1")
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
        math.hypot(7.5, 0.05), rel=1e-14)
    # The library icosphere: same rule on every edge, so r_max is at least half the longest edge.
    ico = sb.make_icosphere(1.0, 1)
    verts, tris = np.asarray(ico.vertices), np.asarray(ico.triangles)
    edges = np.concatenate([verts[tris[:, k]] - verts[tris[:, (k + 1) % 3]] for k in range(3)])
    assert ss.max_support_radius(verts, tris) > 0.5 * np.max(np.linalg.norm(edges, axis=1))


def test_case_tag_and_peak_memory():
    p = ss.CaseParams(L=8e-6, sigma=150e-9, seed=2)
    assert p.tag() == "si_L8um_s150nm_Lc1um_seed2"
    assert p.h == 50e-9 and p.solver_tol == 1e-4 and p.waist is None and p.theta_deg == 0.0
    assert p.default_waist(1 / 6) == pytest.approx(8e-6 / 6, rel=1e-15)
    q = ss.CaseParams(L=8e-6, sigma=150e-9, seed=1, theta_deg=30.0)
    assert q.tag() == "si_L8um_s150nm_Lc1um_seed1_t30"
    assert q.default_waist(1 / 6) == pytest.approx(8e-6 / 6 * math.cos(math.radians(30)))
    sb.check_beam_waist(8e-6, q.default_waist(1 / 6), incidence_angle=math.radians(30),
                        max_fraction=1 / 6)  # the default passes the footprint rule
    assert ss.CaseParams(L=8e-6, sigma=1e-7, seed=1, waist=1e-6,
                         theta_deg=30).default_waist(1 / 6) == 1e-6
    peak = ss.peak_memory_bytes()
    assert peak is None or peak > 1_000_000


def test_driver_parser():
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "speckle_sweep.py"
    spec = importlib.util.spec_from_file_location("speckle_sweep_driver", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        ["--sigma", "5e-8", "--seed", "1", "--out", "o", "--plane-size", "4e-3",
         "--plane-n", "161"])
    assert args.L == 8e-6 and args.h == 50e-9 and args.Lc == 1e-6 and args.z == -1e-3
    assert args.solver_tol == 1e-4 and args.plane_n == 161 and args.far_grid == "90,180"
    assert args.theta_deg == 0.0
    assert module.build_parser().parse_args(
        ["--sigma", "1e-7", "--seed", "1", "--out", "o", "--theta-deg", "30"]).theta_deg == 30.0


@pytest.mark.parametrize("theta_deg, Lc", [(0.0, 0.5e-6), (30.0, 1e-6)])
def test_build_simulation_of_a_small_case_without_solve(theta_deg, Lc):
    # Case construction of the driver (mesh, beam, Simulation object; no assemble, no solve) on
    # a tiny patch, for the server recipe's variants (Lc = 0.5 um; 30 deg incidence).
    p = ss.CaseParams(L=1e-6, sigma=100e-9, seed=1, h=100e-9, Lc=Lc, theta_deg=theta_deg)
    sim, info = ss.build_simulation(p)
    assert isinstance(info["eta0"], float)
    assert info["eta0"] == pytest.approx(376.730313668, rel=1e-9)  # vacuum (R1)
    assert info["waist"] == pytest.approx(1e-6 * math.cos(math.radians(theta_deg)) / 6,
                                          rel=1e-12)
    assert sim.num_unknowns > 0 and info["triangles"] > 0 and info["rmax"] > 0
    assert p.tag() == ("si_L1um_s100nm_Lc0.5um_seed1" if theta_deg == 0.0
                       else "si_L1um_s100nm_Lc1um_seed1_t30")


def test_report_parser_reads_leaf_and_levels():
    rep = ("  operator:       MlfmmOperator: 2N = 51840, d0 = 3, octree 5 levels, 610 leaves of "
           "edge 0.724812 lambda1 (lambda1 = 5e-07 m), built in 0.0034908 s; leaf floor 0.455605\n")
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
            ss.plane_to_direction_cosines(grid, -0.5 * size, size / (n - 1), z,
                                          ss.direction_cosine_axis(umax, 0.05))


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
    assert ss.centre_crop(np.arange(25).reshape(5, 5), 1).tolist() == [[6, 7, 8], [11, 12, 13],
                                                                       [16, 17, 18]]
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
    assert np.array_equal(p0, ss.synthetic_far_zone_speckle(np.random.default_rng(1), plane,
                                                            envelope_rms=0.3, aperture=f0))
    # Gaussian moment theorem: the intensity patterns of fields with correlation mu correlate
    # with mu^2 (radiant intensity in (u, v), flat envelope; ~1500 speckles in total).
    u = ss.direction_cosine_axis(0.66, WL / (2 * W0) / 8)
    rhos = []
    rng2 = np.random.default_rng(77)
    for _ in range(3):
        fa = ss.circular_gaussian_field(rng2, 256)
        fb = ss.correlated_field(rng2, fa, math.sqrt(0.5))
        ia, ib = (ss.plane_to_direction_cosines(
            ss.synthetic_far_zone_speckle(None, plane, aperture=f), -2e-3, plane.spacing, Z, u)
            for f in (fa, fb))
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
    assert st["speckle_size_expected"] == pytest.approx(1.060041 * WL * 1e-3 / (2 * W0),
                                                        rel=1e-6)
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


def _write_case(root, sigma_nm, seed, E, plane):
    d = root / f"si_L8um_s{sigma_nm:g}nm_Lc1um_seed{seed}"
    d.mkdir()
    np.save(d / "E.npy", E)
    rec = dict(sigma_nm=float(sigma_nm), seed=seed, N2=741120, it=100, setup_s=1.0,
               solve_s=2.0, plane_s=3.0, far_s=4.0, peak_gb=5.0, R=0.38, contrast_global=2.0)
    from dataclasses import asdict
    (d / "meta.json").write_text(json.dumps(dict(record=rec, plane=asdict(plane))),
                                 encoding="utf-8")


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
        sb.speckle_contrast(np.abs(E0[:, 0]) ** 2), rel=1e-12)
    assert out["pooled_ks_one"]["n"] > gate["n"]

    # Row-42 sensitivity: the baseline row is the sigma-pooled pool; one change at a time (short
    # list).
    args = module.build_parser().parse_args([str(tmp_path)])
    cases = module.load_cases(tmp_path)
    variants = (("baseline", {}), ("offset 3", dict(offset=3)),
                ("drop", dict(drop_lowest=True)), ("radius", dict(envelope_radius=0.75)))
    sens = module.sensitivity(cases, args, variants)
    rows = {r["label"]: r for r in sens["rows"]}
    for key in ("n", "statistic", "modified", "rejects"):
        assert rows["baseline"][key] == gate[key]
    assert rows["drop"]["sigmas"] == [150, 200, 250] and rows["drop"]["n"] < gate["n"]
    assert rows["offset 3"]["n"] <= gate["n"] and rows["offset 3"]["statistic"] != gate[
        "statistic"]
    assert sens["n_variants"] == sum("rejects" in r for r in sens["rows"]) - 1
    assert sens["n_rejecting"] == sum(r.get("rejects", False) for r in sens["rows"][1:])
    with pytest.raises(ValueError):
        module.sensitivity(cases, args, (("bad", dict(radius=1.0)),))
    # --sensitivity through main (the full list is minutes of work on real data: shortened)
    module.SENSITIVITY_VARIANTS = variants[:2]
    out_json2 = tmp_path / "summary_sens.json"
    assert module.main([str(tmp_path), "--markdown", "--sensitivity", "--json",
                        str(out_json2)]) == 0
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
    relabelled = [dict(c, rec=dict(c["rec"], sigma_nm=50.0) if c["rec"]["sigma_nm"] == 100
                       else c["rec"]) for c in cases]
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
    assert module.main(["--model", "gaussian", "--reps", "2", "--halos", "0.35,0.45",
                        "--map-correlation", "1"]) == 0
    text = capsys.readouterr().out
    assert "map correlation mu 1" in text and "| halo step (u) |" in text and "μ²" in text
    assert module._n_range([76, 90, 76]) == "81 [76–90]" and module._n_range([8, 8]) == "8"
    assert module._n_range([]) == "–"
    for bad in (["--model", "phase-screen", "--map-correlation", "0.5"],
                ["--map-correlation", "1.5"]):
        with pytest.raises(SystemExit):
            module.main(bad + ["--reps", "1"])
