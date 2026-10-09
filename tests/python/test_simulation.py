"""Python bindings of the Simulation driver, the system operator and post-processing (WP14b2).

Driver cases use the sphere of tests/unit/test_simulation.cpp: radius 0.5 um, icosphere n = 1
(2N = 240), lambda = 1 um and its cheap quadrature options, so that each solve is fast.
"""

import gc
import inspect
import threading
import time
import weakref

import numpy as np
import pytest

import specklebem as sb

RADIUS = 0.5e-6
LAMBDA_FAST = 1e-6
CHEAP = dict(
    quad_degree_far=1,
    quad_degree_near=2,
    quad_degree_sing=2,
    outer_grading_levels=0,
    target_accuracy=0.0,
)
N15 = sb.Material(eps_r=2.25)  # lossless n = 1.5


def make_sim(material=N15, subdivisions=1, **kw):
    kw.setdefault("kernels", CHEAP)
    kw.setdefault("gmres", dict(verbose=False))
    pw = sb.PlaneWave(LAMBDA_FAST, [0, 0, 1], [1, 0, 0])
    return sb.Simulation(sb.make_icosphere(RADIUS, subdivisions), pw, object=material, **kw)


@pytest.fixture(scope="module")
def direct():
    sim = make_sim(formulation="PMCHWT", solver="direct")
    sim.solve()
    return sim


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_gmres_equals_direct(direct):
    res = direct.solve()  # repeated solve is allowed; the direct result maps onto SolveResult
    assert res.iterations == 0 and res.converged and res.residual_history.shape == (1,)
    assert res.true_relative_residual < 1e-10
    for kw in (
        dict(formulation="ICTF", preconditioner="none"),
        dict(formulation="pmchwt", preconditioner="diagonal", gmres=dict(side="right")),
    ):
        sim = make_sim(**kw)
        res = sim.solve(tol=1e-10)
        assert res.converged and res.residual_history.shape == (res.iterations + 1,)
        assert rel(sim.currents, direct.currents) < 1e-7
        np.testing.assert_array_equal(res.x, sim.currents)
        th = np.linspace(0, np.pi, 19)
        np.testing.assert_allclose(sim.bistatic_rcs(th), direct.bistatic_rcs(th), rtol=1e-6)
    assert sim.num_unknowns == 240 and "converged" in sim.report()


def test_auto_selection_and_overrides():
    si, ag = make_sim(sb.silicon_500nm()), make_sim(sb.silver_500nm())
    assert (si.formulation, si.preconditioner) == (sb.Formulation.ICTF, "none")
    assert (ag.formulation, ag.preconditioner) == (sb.Formulation.ICTF, "diagonal")
    ag = make_sim(sb.silver_500nm(), formulation=sb.Formulation.MCTF, preconditioner="none")
    assert (ag.formulation, ag.preconditioner) == (sb.Formulation.MCTF, "none")
    ag = make_sim(sb.silver_500nm(), formulation="PMCHWT")
    assert (ag.formulation, ag.preconditioner) == (sb.Formulation.PMCHWT, "diagonal")
    assert ag.solver == "gmres" and ag.wavelength == LAMBDA_FAST
    for bad in (
        dict(formulation="EFIE"),
        dict(formulation="JMCFIE"),  # reserved, not implemented
        dict(preconditioner="block"),
        dict(solver="lu"),
        dict(compression="mlfmm"),
        dict(gmres=dict(tolerance=1e-3)),
        dict(gmres=dict(side="up")),
        dict(gmres=dict(tol=-1.0)),
        dict(kernels=dict(quad_order=3)),
        dict(kernels=dict(quad_degree_far=2.5)),
        dict(wavelength=500e-9),  # differs from the excitation's
        dict(exterior=sb.Material(2.0)),  # differs from the excitation's background
    ):
        with pytest.raises(ValueError):
            make_sim(**bad)
    with pytest.raises(ValueError):
        sb.Simulation(sb.make_icosphere(RADIUS, 1), None, object=N15)
    with pytest.raises(TypeError):  # object is keyword-only and required
        sb.Simulation(sb.make_icosphere(RADIUS, 1), sb.PlaneWave(1e-6, [0, 0, 1], [1, 0, 0]))


def test_callback_and_overrides():
    sim = make_sim(formulation="ICTF", preconditioner="none")
    calls = []
    res = sim.solve(tol=1e-6, callback=lambda it, r: calls.append((it, r)))
    assert [c[0] for c in calls] == list(range(1, res.iterations + 1))
    np.testing.assert_array_equal([c[1] for c in calls], res.residual_history[1:])
    assert sim.solve(tol=1e-2).iterations < res.iterations
    short = sim.solve(max_iter=3, restart=2, side="right")
    assert short.iterations == 3 and not short.converged  # not an error; stored anyway

    class Stop(Exception):
        pass

    err = Stop("stop at 4")

    def cb(it, r):
        if it == 4:
            raise err

    with pytest.raises(Stop, match="stop at 4") as excinfo:
        sim.solve(callback=cb)
    assert excinfo.value is err  # the same exception object propagates unchanged
    assert sim.solve().converged  # still usable after the aborted solve
    with pytest.raises(TypeError):
        sim.solve(callback=3)
    for bad in (dict(tol=0.0), dict(max_iter=0), dict(restart=-1), dict(side="middle")):
        with pytest.raises(ValueError):
            sim.solve(**bad)
    for bad_gmres in ({1: 2}, {b"tol": 1e-3}, dict(side=b"left")):
        with pytest.raises(ValueError):
            make_sim(gmres=bad_gmres)
    with pytest.raises(ValueError):
        make_sim(kernels={3: 1})


def test_gmres_solve_releases_gil():
    """A GMRES solve (pre-assembled) runs in a worker thread; this thread keeps ticking."""
    sim = make_sim(subdivisions=2, formulation="PMCHWT", preconditioner="none")
    sim.assemble()
    done = threading.Event()
    result = {}

    def work():
        result["res"] = sim.solve(tol=1e-13, max_iter=3000)
        done.set()

    t0 = time.perf_counter()
    worker = threading.Thread(target=work)
    worker.start()
    gap, last = 0.0, time.perf_counter()
    while not done.is_set():
        now = time.perf_counter()
        gap, last = max(gap, now - last), now
    worker.join()
    elapsed = time.perf_counter() - t0
    assert result["res"].iterations > 50  # long enough for a meaningful timing check
    assert gap < 0.25 * elapsed, (gap, elapsed)


def test_errors_before_assemble_and_solve():
    sim = make_sim()
    for f in (sim.operator, sim.rhs, lambda: sim.currents, lambda: sim.bistatic_rcs(0.0)):
        with pytest.raises(RuntimeError):
            f()
    sim.assemble()
    assert sim.rhs().shape == (240,) and sim.operator().shape == (240, 240)
    with pytest.raises(RuntimeError):
        sim.field([0.0, 0.0, 2e-6])


def test_operator_and_keep_alive():
    sim = make_sim(formulation="ICTF", preconditioner="none")
    sim.assemble()
    Z, b = sim.operator(), sim.rhs()
    assert Z.dtype == np.complex128 and Z.memory_bytes > 0 and "dense" in Z.describe()
    res = sim.solve(tol=1e-10)
    x = sim.currents
    alive = weakref.ref(sim)
    del sim
    gc.collect()
    assert alive() is not None  # kept alive by the operator
    n = Z.shape[1]
    Zd = np.column_stack([Z.matvec(e) for e in np.eye(n)])
    v = np.random.default_rng(3).standard_normal(n) + 1j
    np.testing.assert_allclose(Z @ v, Zd @ v, rtol=1e-12, atol=0)
    np.testing.assert_allclose(Z.matvec(v.reshape(n, 1)), Zd @ v, rtol=1e-12, atol=0)
    assert rel(Zd @ x, b) < 1e-9 and rel(res.x, x) == 0
    for bad in (np.ones(n - 1), np.ones((n, 2)), np.array(["a"] * n), np.full(n, np.nan)):
        with pytest.raises(ValueError):
            Z.matvec(bad)
    del Z
    gc.collect()
    assert alive() is None


def test_operator_as_scipy():
    """SciPy GMRES on as_scipy() (skipped without the optional dev dependency SciPy)."""
    spla = pytest.importorskip("scipy.sparse.linalg")
    sim = make_sim(formulation="ICTF", preconditioner="none")
    x, b = sim.solve(tol=1e-10).x, sim.rhs()
    A = sim.operator().as_scipy()
    n = sim.num_unknowns
    assert A.shape == (n, n)
    rtol = "rtol" if "rtol" in inspect.signature(spla.gmres).parameters else "tol"
    y, info = spla.gmres(A, b, restart=n, maxiter=n, atol=0.0, **{rtol: 1e-10})
    assert info == 0 and rel(y, x) < 1e-7


def test_fields_far_field_and_rcs(direct):
    pts = np.array([[0.0, 0.0, -2e-6], [1.5e-6, 0.5e-6, 0.2e-6], [0.0, 0.1e-6, 0.0]])
    Es, Hs = direct.field(pts)
    Et, Ht = direct.field(pts, kind="total")
    assert Es.shape == (3, 3) and Es.dtype == np.complex128 and Es.flags.c_contiguous
    pw = sb.PlaneWave(LAMBDA_FAST, [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(Et[:2], Es[:2] + pw.electric_field(pts[:2]), rtol=1e-13)
    np.testing.assert_allclose(Ht[:2], Hs[:2] + pw.magnetic_field(pts[:2]), rtol=1e-13)
    np.testing.assert_array_equal(Et[2], Es[2])  # inside R2 both are the total field
    np.testing.assert_array_equal(direct.field(pts[0])[0], Es[0])
    for bad in (dict(kind="incident"), dict(points=[[RADIUS, 0.0, 0.0]])):
        with pytest.raises(ValueError):
            direct.field(**{"points": pts, **bad})
    with pytest.raises(ValueError, match="finite"):
        direct.field([[0.0, np.nan, 0.0]])

    th = np.linspace(0, np.pi, 7)
    F = direct.far_field(np.column_stack([np.sin(th), 0 * th, np.cos(th)]))
    sigma = direct.bistatic_rcs(th, plane="xz")
    np.testing.assert_allclose(sigma, 4 * np.pi * np.sum(np.abs(F) ** 2, axis=1), rtol=1e-12)
    np.testing.assert_array_equal(direct.bistatic_rcs(th, plane=[0, 1, 0]), sigma)
    assert isinstance(direct.bistatic_rcs(0.5), float)
    yz = direct.bistatic_rcs(th, plane="yz")
    np.testing.assert_array_equal(yz, direct.bistatic_rcs(th, plane=np.array([-1.0, 0, 0])))
    for bad in (dict(plane="xy"), dict(plane=[0, 0, 1]), dict(angles=[np.inf])):
        with pytest.raises(ValueError):
            direct.bistatic_rcs(**{"angles": th, **bad})


def test_free_post_functions():
    g = sb.plane_grid([0, 0, -1e-3], [2e-6, 0, 0], [0, 1e-6, 0], 3, 2)
    assert g.shape == (6, 3)
    np.testing.assert_allclose(g[1 * 2 + 1], [1e-6, 1e-6, -1e-3])
    c = sb.cylinder_grid(1.0, -0.5, 0.5, 5, -0.1, 0.1, 3)
    assert c.shape == (15, 3)
    np.testing.assert_allclose(c[0], [np.sin(-0.5), -0.1, -np.cos(-0.5)], rtol=1e-14)
    E = np.random.default_rng(5).standard_normal((15, 3)) * (1 + 0.5j)
    I = sb.intensity(E)
    np.testing.assert_allclose(I, np.sum(np.abs(E) ** 2, axis=1), rtol=1e-14)
    assert sb.intensity(E[0]) == pytest.approx(I[0], rel=1e-14)
    co, cross = sb.polarized_intensity(E, co_pol=[2, 0, 0])
    np.testing.assert_allclose(co, np.abs(E[:, 0]) ** 2, rtol=1e-14)
    np.testing.assert_allclose(co + cross, I, rtol=1e-14)
    drc = sb.differential_reflection_coefficient(E, 5, 3)
    np.testing.assert_allclose(drc, I.reshape(5, 3).mean(axis=1), rtol=1e-14)
    with pytest.raises(ValueError):
        sb.intensity(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        sb.plane_grid([0, 0, 0], [1, 0, 0], [0, np.nan, 0], 2, 2)
    assert "[m]" in sb.Simulation.field.__doc__ and "[m^2]" in sb.Simulation.bistatic_rcs.__doc__


def test_mie_dielectric_sphere_n2_gil_released():
    """WP11 n = 1.5 case at icosphere n = 2 (default kernels, PMCHWT + LU) against Mie.

    Bound eps_rr < 2 % in the xz- and yz-planes, as tests/validation/test_mie_sphere_dense.cpp
    (measured 0.0089 / 0.0089). Assembly and solve run in a worker thread while this thread
    keeps ticking: the longest pause between ticks shows that the GIL is released.
    """
    lam = 500e-9
    pw = sb.PlaneWave(lam, [0, 0, 1], [1, 0, 0])
    mesh = sb.make_icosphere(RADIUS, 2)
    sim = sb.Simulation(mesh, pw, object=N15, formulation="PMCHWT", solver="direct")
    done = threading.Event()

    def work():
        sim.assemble()
        sim.solve()
        done.set()

    t0 = time.perf_counter()
    worker = threading.Thread(target=work)
    worker.start()
    gap, last = 0.0, time.perf_counter()
    while not done.is_set():
        now = time.perf_counter()
        gap, last = max(gap, now - last), now
    worker.join()
    elapsed = time.perf_counter() - t0
    assert sim.num_unknowns == 960
    assert gap < 0.25 * elapsed, (gap, elapsed)

    th = np.linspace(0, np.pi, 181)
    mie = sb.Mie(RADIUS, lam, N15)
    for plane, phi in (("xz", 0.0), ("yz", np.pi / 2)):
        ref = np.sqrt(mie.bistatic_rcs(th, phi))
        eps_rr = np.sqrt(np.mean((ref - np.sqrt(sim.bistatic_rcs(th, plane))) ** 2)) / ref.max()
        assert eps_rr < 0.02, (plane, eps_rr)
