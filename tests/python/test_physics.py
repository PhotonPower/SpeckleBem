"""Python bindings of materials, excitations and the Mie reference (WP14b1)."""

import numpy as np
import pytest

import specklebem as sb

ETA0 = 376.730313668
C0 = 299792458.0
LAMBDA = 500e-9


def test_reference_materials_convention():
    ag, si = sb.silver_500nm(), sb.silicon_500nm()
    assert ag.eps_r == pytest.approx(-9.794 - 0.313j, abs=1e-12)
    assert si.eps_r == pytest.approx(18.478 - 0.606j, abs=1e-12)
    for m in (ag, si):  # exp(+jwt): passive media have Im(eps_r) <= 0 and Im(n) <= 0
        assert m.eps_r.imag <= 0 and m.n.imag <= 0


def test_field_decay_length():
    si = sb.silicon_500nm()
    d_si = sb.field_decay_length(si, LAMBDA)
    assert d_si == pytest.approx(1.129e-6, rel=1e-3)
    assert d_si == pytest.approx(LAMBDA / (2 * np.pi * abs(np.sqrt(si.eps_r).imag)), rel=1e-12)
    assert sb.field_decay_length(si.eps_r, LAMBDA) == pytest.approx(d_si, rel=1e-14)
    assert sb.field_decay_length(eps_r=sb.silver_500nm().eps_r, wavelength=LAMBDA) == (
        pytest.approx(25.4e-9, rel=2e-3)
    )
    assert sb.field_decay_length(2.25, LAMBDA) == np.inf
    with pytest.raises(ValueError):
        sb.field_decay_length(si, -1.0)


def test_dispersive_material():
    lam = np.array([400e-9, 500e-9, 600e-9])
    n = np.array([1.5 - 0.1j, 2.0 - 0.2j, 2.5 + 0.0j])
    disp = sb.DispersiveMaterial(lam, n)
    n_450 = 0.5 * (n[0] + n[1])
    assert disp.at(450e-9).eps_r == pytest.approx(n_450**2, rel=1e-14)
    assert disp.at_wavelength(550e-9).eps_r == pytest.approx((0.5 * (n[1] + n[2])) ** 2)
    assert disp.at(450e-9).mu_r == 1
    with pytest.raises(ValueError):
        disp.at(700e-9)
    with pytest.raises(ValueError):
        sb.DispersiveMaterial(lam, n[:2])


def plane_wave_reference(k_hat, e0, pts):
    k = 2 * np.pi / LAMBDA
    k_hat = np.asarray(k_hat, float) / np.linalg.norm(k_hat)
    E = np.exp(-1j * k * pts @ k_hat)[:, None] * np.asarray(e0)[None, :]
    return E, np.cross(k_hat, E) / ETA0


def test_plane_wave():
    k_hat = [1.0, 0.0, 1.0]  # normalised by the constructor
    e0 = np.array([1.0, 0.0, -1.0]) / np.sqrt(2) + 0.5j * np.array([0.0, 1.0, 0.0])
    pw = sb.PlaneWave(wavelength=LAMBDA, direction=k_hat, polarization=e0)
    assert isinstance(pw, sb.Excitation)
    assert pw.wavelength == pytest.approx(LAMBDA, rel=1e-14)
    assert pw.omega == pytest.approx(2 * np.pi * C0 / LAMBDA, rel=1e-14)
    assert pw.background.eps_r == 1
    pts = np.random.default_rng(1).uniform(-2e-6, 2e-6, size=(7, 3))
    E, H = pw.electric_field(pts), pw.magnetic_field(pts)
    assert E.shape == (7, 3) and E.dtype == np.complex128 and E.flags.c_contiguous
    E_ref, H_ref = plane_wave_reference(k_hat, e0, pts)
    np.testing.assert_allclose(E, E_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H, H_ref, rtol=0, atol=1e-12 / ETA0)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), np.linalg.norm(e0), rtol=1e-14)
    # A single point gives a (3,) array; an empty set an empty (0, 3) array.
    np.testing.assert_allclose(pw.electric_field(pts[0]), E[0], rtol=1e-15)
    assert pw.electric_field(np.zeros((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        pw.electric_field(np.zeros((4, 2)))
    with pytest.raises(ValueError):  # not transverse
        sb.PlaneWave(LAMBDA, [0, 0, 1], [0, 0, 1])


@pytest.mark.parametrize("theta", [0.0, np.pi / 6])
@pytest.mark.parametrize("pol", ["p", "S", sb.Polarization.P])
def test_gaussian_beam_wide_waist_is_plane_wave(theta, pol):
    beam = sb.GaussianBeam(wavelength=LAMBDA, waist=1.0, polarization=pol, incidence_angle=theta)
    assert beam.waist == 1.0 and beam.incidence_angle == theta
    k_hat = [np.sin(theta), 0.0, np.cos(theta)]
    if beam.polarization == sb.Polarization.P:
        e_hat = [np.cos(theta), 0.0, -np.sin(theta)]
    else:
        e_hat = [0.0, 1.0, 0.0]
    pts = np.random.default_rng(2024).uniform(-1, 1, size=(20, 3))
    pts *= 9.99 * LAMBDA / np.maximum(1.0, np.linalg.norm(pts, axis=1))[:, None]
    E_ref, H_ref = plane_wave_reference(k_hat, e_hat, pts)
    assert np.linalg.norm(beam.electric_field(pts) - E_ref, axis=1).max() < 1e-6
    assert np.linalg.norm(beam.magnetic_field(pts) - H_ref, axis=1).max() * ETA0 < 1e-6


def test_gaussian_beam_invalid():
    with pytest.raises(ValueError):
        sb.GaussianBeam(wavelength=LAMBDA, waist=5e-6, polarization="x")
    with pytest.raises(ValueError):
        sb.GaussianBeam(wavelength=LAMBDA, waist=-1.0)
    with pytest.raises(ValueError):  # lossy background
        sb.GaussianBeam(LAMBDA, 5e-6, background=sb.silicon_500nm())


def test_mie_bohren_huffman():
    # BH appendix A: m = 1.55, radius 0.525 um, lambda 0.6328 um (x = 5.213), lossless.
    a, lam = 0.525e-6, 0.6328e-6
    mie = sb.Mie(radius=a, wavelength=lam, material=sb.Material(eps_r=1.55**2))
    geo = np.pi * a**2
    assert mie.scattering_cross_section() / geo == pytest.approx(3.10543, rel=1e-4)
    assert mie.extinction_cross_section() / geo == pytest.approx(3.10543, rel=1e-4)
    back = mie.bistatic_rcs(np.pi, 0.0)
    assert isinstance(back, float) and back / geo == pytest.approx(2.92534, rel=1e-4)
    assert mie.a_n.shape == (26,) and not mie.a_n.flags.writeable

    theta = np.linspace(0, np.pi, 181)
    rcs = mie.bistatic_rcs(theta, 0.0)
    assert rcs.shape == (181,) and rcs.dtype == np.float64 and np.all(rcs > 0)
    grid = mie.bistatic_rcs(theta[:4, None], np.array([0.0, 0.5, 1.0]))
    assert grid.shape == (4, 3)
    assert grid[1, 2] == pytest.approx(mie.bistatic_rcs(theta[1], 1.0), rel=1e-15)

    pts = np.array([[2e-6, 0, 0], [0, 0, -3e-6], [1e-6, 1e-6, 1e-6]])
    Es = mie.scattered_E(pts)
    assert Es.shape == (3, 3) and Es.dtype == np.complex128
    assert mie.scattered_H(pts).shape == (3, 3)
    assert mie.internal_E(np.zeros((1, 3))).shape == (1, 3)
    with pytest.raises(ValueError):
        mie.scattered_E([0.0, 0.0, 0.0])  # inside the sphere
    # Far field: 4 pi r^2 |E_s|^2 -> sigma (|E_inc| = 1 V/m), error O(1/(k r)).
    r, th, ph = 1e-2, 0.7, 0.3
    p = r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    far = 4 * np.pi * r**2 * np.sum(np.abs(mie.scattered_E(p)) ** 2)
    assert far == pytest.approx(mie.bistatic_rcs(th, ph), rel=1e-3)
