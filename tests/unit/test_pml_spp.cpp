// Unit tests of the validation 7b measurement (WP-PML4a, tests/support/pml_spp_support.hpp): the
// cylindrical Hankel expansion, the SPP wavenumber and field impedance, the fit of outgoing +
// reflected cylindrical SPP waves on synthetic data (exact, noisy with a fixed seed on a dense grid
// and at the real 7b sampling, with a mismatched background, with a reflected wave of the planar
// asymptotic form, and the phase rotation when the reflection plane is moved: the relation the
// geometric check of the record relies on), the sampling of the RWG currents along the x-axis and
// the in-place LU.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "pml_spp_support.hpp"

using namespace pml_spp;

namespace {

constexpr Real kK0 = 2.0 * constants::pi / 500e-9;

/// Synthetic J_x-like samples: out H1'(k rho) + in H1'(k (2 a - rho)) + background terms.
struct Synthetic {
    Complex k;
    Complex out = Complex(1.0, 0.3);
    Complex in = 0.0;
    Real plane = 2.0e-6;
    std::vector<std::pair<Real, Complex>> background;  ///< (power, amplitude)
    WaveForm form = WaveForm::H1Prime;
};

std::vector<Real> grid(Real rho_min, Real rho_max, Real step) {
    std::vector<Real> rho;
    for (Real r = rho_min; r <= rho_max * (1.0 + 1e-12); r += step) rho.push_back(r);
    return rho;
}

std::vector<Complex> evaluate(const Synthetic& s, const std::vector<Real>& rho) {
    const Complex j(0.0, 1.0);
    std::vector<Complex> y;
    for (const Real r : rho) {
        Complex v = s.out * wave(s.form, s.k * r) + s.in * wave(s.form, s.k * (2.0 * s.plane - r));
        for (const auto& [p, c] : s.background)
            v += c * std::exp(-j * (kK0 * r)) * std::pow(kK0 * r, -p);
        y.push_back(v);
    }
    return y;
}

FitOptions options(const Synthetic& s, Complex k_guess) {
    FitOptions o;
    o.form = s.form;
    o.plane = s.plane;
    o.k0 = kK0;
    o.k_guess = k_guess;
    return o;
}

Complex ag_k_spp() {
    return spp_wavenumber(kK0, Complex(1.0, 0.0), material::silver_500nm().eps_r);
}

}  // namespace

TEST_CASE("pml_spp: Hankel expansion matches tabulated Bessel values at z = 10", "[pml_spp]") {
    // J0(10), Y0(10), J1(10), Y1(10) (Abramowitz & Stegun table 9.1 / DLMF).
    const Complex h0 = hankel(2, 0, Complex(10.0, 0.0));
    const Complex h1 = hankel(2, 1, Complex(10.0, 0.0));
    CHECK(std::abs(h0 - Complex(-0.2459357644513483, -0.05567116728359939)) < 1e-9);
    CHECK(std::abs(h1 - Complex(0.04347274616886144, -0.24901542420695388)) < 1e-9);
    const Complex h0_1 = hankel(1, 0, Complex(10.0, 0.0));
    CHECK(std::abs(h0_1 - std::conj(h0)) < 1e-15);
}

TEST_CASE("pml_spp: Hankel expansion satisfies the Wronskian and the recurrence at complex z",
          "[pml_spp]") {
    for (const Complex z : {Complex(8.0, -0.02), Complex(13.3, -0.05), Complex(25.0, -0.4)}) {
        const Complex w = hankel(1, 0, z) * hankel(2, 1, z) - hankel(2, 0, z) * hankel(1, 1, z);
        const Complex expected = Complex(0.0, 4.0) / (constants::pi * z);
        CHECK(std::abs(w / expected - 1.0) < 1e-7);
        for (int nu = 1; nu <= 3; ++nu) {
            const Complex lhs = hankel(2, nu - 1, z) + hankel(2, nu + 1, z);
            const Complex rhs = 2.0 * static_cast<Real>(nu) / z * hankel(2, nu, z);
            CHECK(std::abs(lhs - rhs) < 1e-7 * std::abs(rhs));
        }
    }
    // H1' = H0 - H1 / z against a central difference of H1.
    const Complex z(12.0, -0.03);
    const Real d = 1e-4;
    const Complex fd = (hankel(2, 1, z + d) - hankel(2, 1, z - d)) / (2.0 * d);
    CHECK(std::abs(wave(WaveForm::H1Prime, z) - fd) < 1e-7);
}

TEST_CASE("pml_spp: Hankel expansion rejects arguments outside its domain", "[pml_spp]") {
    CHECK_THROWS_AS(hankel(2, 0, Complex(5.9, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS(hankel(2, 0, Complex(-10.0, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS(hankel(3, 0, Complex(10.0, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS(hankel(2, 5, Complex(10.0, 0.0)), std::invalid_argument);
    CHECK_THROWS_AS(hankel(2, 0, Complex(std::nan(""), 0.0)), std::invalid_argument);
}

TEST_CASE("pml_spp: SPP wavenumber of Ag at 500 nm", "[pml_spp]") {
    const Complex k = ag_k_spp();
    CHECK(std::abs(k / kK0 - Complex(1.05525, -0.001915)) < 2e-5);
    CHECK(k.imag() < 0.0);
    CHECK(std::abs(2.0 * constants::pi / k.real() - 473.8e-9) < 0.2e-9);
    CHECK(std::abs(1.0 / std::abs(k.imag()) - 41.5e-6) < 0.2e-6);
    CHECK_THROWS_AS(spp_wavenumber(0.0, 1.0, -2.0), std::invalid_argument);
    CHECK_THROWS_AS(spp_wavenumber(kK0, 1.0, -1.0), std::invalid_argument);
}

TEST_CASE("pml_spp: fit recovers r, k and the background from exact synthetic data", "[pml_spp]") {
    // Numerical dispersion offset of Re k (Im k is not fitted by default).
    const Complex k_true(ag_k_spp().real() * 1.012, ag_k_spp().imag());
    Synthetic s;
    s.k = k_true;
    s.in = s.out * std::polar(3e-3, 0.7);
    s.background = {{1.5, Complex(0.4, -0.2)}, {2.0, Complex(-0.3, 0.1)}};
    const std::vector<Real> rho = grid(0.6e-6, s.plane, 500e-9 / 6.0 / 2.0);
    const FitResult f = fit_spp(rho, evaluate(s, rho), options(s, ag_k_spp()));
    CHECK(std::abs(f.k / k_true - 1.0) < 1e-8);
    CHECK(std::abs(f.r() - s.in / s.out) < 1e-8);
    CHECK(f.residual < 1e-8);
    REQUIRE(f.background.size() == 3);
    CHECK(std::abs(f.background[1] - Complex(0.4, -0.2)) < 1e-6);
    CHECK(std::abs(f.background[0]) < 1e-6);
    CHECK(f.points == static_cast<Index>(rho.size()));
}

TEST_CASE("pml_spp: fit with noise (fixed seed) keeps |r| to the noise level", "[pml_spp]") {
    std::mt19937_64 gen(20261014);
    std::normal_distribution<Real> normal(0.0, 1.0);
    Synthetic s;
    s.k = Complex(ag_k_spp().real() * 0.993, ag_k_spp().imag());
    const std::vector<Real> rho = grid(0.6e-6, s.plane, 500e-9 / 6.0 / 2.0);
    for (const Real r_true : {0.0, 1e-3, 1e-2}) {
        s.in = s.out * std::polar(r_true, -1.2);
        s.background = {{1.5, Complex(0.5, 0.1)}};
        std::vector<Complex> y = evaluate(s, rho);
        // Relative noise 1e-4 of the local SPP amplitude.
        for (std::size_t i = 0; i < y.size(); ++i)
            y[i] += 1e-4 * std::abs(s.out * wave(s.form, s.k * rho[i])) *
                    Complex(normal(gen), normal(gen));
        const FitResult f = fit_spp(rho, y, options(s, ag_k_spp()));
        INFO("r_true " << r_true << ", r_fit " << f.r() << ", k/k_true " << f.k / s.k);
        CHECK(std::abs(f.r() - s.in / s.out) < 1e-4);
        // k is a nuisance parameter, nearly degenerate with the background columns.
        CHECK(std::abs(f.k / s.k - 1.0) < 2e-3);
    }
}

TEST_CASE("pml_spp: fit with noise at the real 7b sampling (fixed seed) separates 1e-2 from 0",
          "[pml_spp]") {
    // The sampling of the 7b fit (seven_b_case(): h = 100 nm, plane a = 1.5 um, backgrounds
    // {1, 1.5, 2}): M_y at rho = 0.65 ... 1.45 um (9 samples), J_x at 0.6 ... 1.5 um (10), for
    // five complex amplitudes and Re k. Data: the SPP with a dispersion offset of Re k (+-1.2 %,
    // the range of the fitted k/k_spp), one background power in the model (1.5) and one outside it
    // (1.25), relative noise 1e-4 of the local SPP amplitude, several draws per case. Few samples
    // amplify the noise about 10x against the dense grid above (|dr| up to ~1e-3 at r = 1e-2);
    // 2e-3 still separates r = 1e-2 from r = 0.
    std::mt19937_64 gen(20261015);
    std::normal_distribution<Real> normal(0.0, 1.0);
    // Six draws per case in optimised builds (72 fits, < 1 s); one under -O0 + ASan, where a fit
    // costs ~0.3 s (12 fits, the same tolerance).
#ifdef NDEBUG
    constexpr int kDraws = 6;
#else
    constexpr int kDraws = 1;
#endif
    const auto line = [](Real first, int count) {
        std::vector<Real> rho;
        for (int i = 0; i < count; ++i) rho.push_back(first + 100e-9 * static_cast<Real>(i));
        return rho;
    };
    const std::vector<std::pair<const char*, std::vector<Real>>> samplings = {
        {"My", line(0.65e-6, 9)}, {"Jx", line(0.6e-6, 10)}};
    const std::array<Real, 3> r_values = {0.0, 1e-3, 1e-2};
    std::array<std::vector<Real>, 3> errors;
    Real worst_zero = 0.0;  // largest |r_fit| for r = 0
    for (const auto& [name, rho] : samplings) {
        for (const Real disp : {0.988, 1.012}) {
            Synthetic s;
            s.plane = 1.5e-6;
            s.k = Complex(ag_k_spp().real() * disp, ag_k_spp().imag());
            s.background = {{1.5, Complex(0.5, 0.1)}, {1.25, Complex(0.3, -0.2)}};
            for (std::size_t ir = 0; ir < r_values.size(); ++ir) {
                const Real r_true = r_values[ir];
                s.in = s.out * std::polar(r_true, -1.2);
                for (int draw = 0; draw < kDraws; ++draw) {
                    std::vector<Complex> y = evaluate(s, rho);
                    for (std::size_t i = 0; i < y.size(); ++i)
                        y[i] += 1e-4 * std::abs(s.out * wave(s.form, s.k * rho[i])) *
                                Complex(normal(gen), normal(gen));
                    const FitResult f = fit_spp(rho, y, options(s, ag_k_spp()));
                    const Real dr = std::abs(f.r() - s.in / s.out);
                    INFO(name << " (" << rho.size() << " samples), k/k_spp " << disp << ", r_true "
                              << r_true << ", draw " << draw << ": r_fit " << f.r() << ", |dr| "
                              << dr);
                    CHECK(f.points == static_cast<Index>(rho.size()));
                    CHECK(dr < 2e-3);
                    errors[ir].push_back(dr);
                    if (ir == 0)
                        worst_zero = std::max(worst_zero, std::abs(f.r()));
                }
            }
        }
    }
    for (std::size_t ir = 0; ir < r_values.size(); ++ir) {
        std::vector<Real>& e = errors[ir];
        std::sort(e.begin(), e.end());
        // Measured: r = 0 / 1e-3 / 1e-2: max |dr| 1.3e-4 / 3.8e-4 / 9.6e-4, median 5e-5 / 7e-5 /
        // 3.1e-4 (24 fits each in optimised builds).
        UNSCOPED_INFO("7b sampling, noise 1e-4, r = " << r_values[ir] << ": " << e.size()
                                                      << " fits, max |dr| " << e.back()
                                                      << ", median " << e[e.size() / 2]);
    }
    // r = 1e-2 and r = 0 stay apart: |r_fit| < 2e-3 for r = 0 against > 8e-3 for r = 1e-2.
    CHECK(worst_zero < 2e-3);
}

TEST_CASE("pml_spp: fit recovers r of a reflected wave of the planar asymptotic form",
          "[pml_spp]") {
    // A reflected wave built independently of the model column w(k (2 a - rho)): the leading
    // large-argument form of H1' at the image distance, exp(-j k (2 a - rho)) / sqrt(2 a - rho),
    // scaled to the model column's value w(k a) at the plane. The two differ by the 1 / (k (2a -
    // rho)) corrections of the expansion (k a = 19.9 at a = 1.5 um), so r is recovered to that
    // asymptotic accuracy, not to round-off. Dense grid (lambda / 12), no noise, Re k fitted.
    Synthetic s;
    s.plane = 1.5e-6;
    s.k = ag_k_spp();
    s.background = {{1.5, Complex(0.4, -0.2)}};
    const Complex r_true = std::polar(1e-2, 0.9);
    const std::vector<Real> rho = grid(0.6e-6, s.plane, 500e-9 / 12.0);
    const Real a = s.plane;
    const Complex j(0.0, 1.0);
    const Complex at_plane = wave(s.form, s.k * a);
    std::vector<Complex> y = evaluate(s, rho);  // s.in = 0: outgoing + background only
    for (std::size_t i = 0; i < rho.size(); ++i)
        y[i] += s.out * r_true * at_plane * std::sqrt(a / (2.0 * a - rho[i])) *
                std::exp(-j * s.k * (a - rho[i]));
    const FitResult f = fit_spp(rho, y, options(s, ag_k_spp()));
    const Complex r_fit = f.r();
    // Tolerance O(1 / (k a)) = 0.05; measured |r_fit / r_true - 1| = 0.030.
    const Real accuracy = 1.0 / (s.k.real() * a);
    INFO("r_fit " << r_fit << ", r_true " << r_true << ", |r_fit / r_true - 1| "
                  << std::abs(r_fit / r_true - 1.0) << ", residual " << f.residual);
    CHECK(std::abs(std::abs(r_fit) / std::abs(r_true) - 1.0) < accuracy);
    CHECK(std::abs(std::arg(r_fit / r_true)) < accuracy);
}

TEST_CASE("pml_spp: SPP field impedance and the out(M_y) / out(J_x) diagnostic", "[pml_spp]") {
    // Ag at 500 nm: kappa = sqrt(k_spp^2 - k0^2) = 0.337 k0, |E_x / H_y| = 0.337 eta0 = 127.0 Ohm.
    const Complex k = ag_k_spp();
    const Real z = spp_impedance(kK0, k);
    CHECK(std::abs(z - 126.999) < 0.01);
    // The metal side gives the same value (tangential E and H continuous): kappa_1 / eps_1 =
    // -kappa_2 / eps_2 is the SPP dispersion relation, so this checks spp_wavenumber too.
    const Complex eps_ag = material::silver_500nm().eps_r;
    CHECK(std::abs(spp_impedance(kK0, k, eps_ag) / z - 1.0) < 1e-12);
    // Lossless eps_2 = -4: k_spp = k0 sqrt(4 / 3), kappa = k0 / sqrt(3).
    const Complex k4 = spp_wavenumber(kK0, Complex(1.0, 0.0), Complex(-4.0, 0.0));
    CHECK(std::abs(spp_impedance(kK0, k4) - constants::eta0 / std::sqrt(3.0)) < 1e-9);
    CHECK(std::abs(spp_impedance(kK0, k4, Complex(-4.0, 0.0)) - constants::eta0 / std::sqrt(3.0)) <
          1e-9);
    CHECK_THROWS_AS(spp_impedance(0.0, k), std::invalid_argument);
    CHECK_THROWS_AS(spp_impedance(kK0, k, Complex(0.0, 0.0)), std::invalid_argument);
    // out_impedance divides the outgoing amplitudes of one source and half-axis.
    std::vector<QuantityFit> fits(3);
    fits[0] = {"local", "Jx", 1, {}};
    fits[0].fit.out = Complex(0.0, 2e-3);
    fits[1] = {"local", "My", 1, {}};
    fits[1].fit.out = Complex(0.1, 0.0);
    fits[2] = {"local", "My", -1, {}};
    CHECK(std::abs(out_impedance(fits, "local", 1) - 50.0) < 1e-12);
    CHECK_THROWS_AS(out_impedance(fits, "local", -1), std::invalid_argument);
    CHECK_THROWS_AS(out_impedance(fits, "groove", 1), std::invalid_argument);
}

TEST_CASE("pml_spp: background of another power leaks little into r", "[pml_spp]") {
    // Data with a background exp(-j k0 rho) (k0 rho)^-1.25 (not in the default model {1, 1.5, 2})
    // as strong as the SPP at rho_min, sampled at h = lambda / 6 like J_x on the mesh: the outgoing
    // columns absorb it, the reflected column stays clean (leakage ~ 1 / (2 k window) of the
    // model mismatch).
    Synthetic s;
    s.k = ag_k_spp();
    s.in = s.out * std::polar(2e-3, 0.3);
    const std::vector<Real> rho = grid(0.6e-6, s.plane, 500e-9 / 6.0);
    const Real spp_at_min = std::abs(s.out * wave(s.form, s.k * rho.front()));
    s.background = {{1.25, spp_at_min * std::pow(kK0 * rho.front(), 1.25) * Complex(0.6, 0.8)}};
    const FitResult f = fit_spp(rho, evaluate(s, rho), options(s, ag_k_spp()));
    INFO("r_fit " << f.r() << ", residual " << f.residual << ", k/k_spp " << f.k / s.k);
    CHECK(std::abs(f.r() - s.in / s.out) < 2e-4);
}

TEST_CASE("pml_spp: moving the assumed reflection plane rotates r by 2 Re(k) delta", "[pml_spp]") {
    // The geometric check: data reflected at a, fitted with the plane a + delta, give the same
    // |r| and a phase advanced by 2 Re(k) delta (the standing wave referred to another plane).
    Synthetic s;
    s.k = ag_k_spp();
    s.in = s.out * std::polar(5e-2, 0.4);
    const std::vector<Real> rho = grid(0.6e-6, s.plane, 500e-9 / 6.0 / 2.0);
    const std::vector<Complex> y = evaluate(s, rho);
    FitOptions o = options(s, ag_k_spp());
    o.fit_k = false;
    const FitResult f0 = fit_spp(rho, y, o);
    const Real delta = 500e-9 / 6.0;
    o.plane = s.plane + delta;
    const FitResult f1 = fit_spp(rho, y, o);
    const Real rotation = std::arg(f1.r() / f0.r());
    const Real predicted = std::remainder(2.0 * s.k.real() * delta, 2.0 * constants::pi);
    CHECK(std::abs(rotation - predicted) < 0.02);
    CHECK(std::abs(std::abs(f1.r()) / std::abs(f0.r()) - 1.0) < 0.05);
}

TEST_CASE("pml_spp: fit rejects inconsistent input", "[pml_spp]") {
    Synthetic s;
    s.k = ag_k_spp();
    const std::vector<Real> rho = grid(0.6e-6, s.plane, 100e-9);
    const std::vector<Complex> y = evaluate(s, rho);
    FitOptions o = options(s, ag_k_spp());
    CHECK_THROWS_AS(fit_spp(rho, std::vector<Complex>(y.begin(), y.end() - 1), o),
                    std::invalid_argument);
    o.plane = 1.0e-6;  // samples beyond the plane
    CHECK_THROWS_AS(fit_spp(rho, y, o), std::invalid_argument);
    o = options(s, ag_k_spp());
    o.k0 = 0.0;
    CHECK_THROWS_AS(fit_spp(rho, y, o), std::invalid_argument);
    const std::vector<Real> few(rho.begin(), rho.begin() + 3);
    CHECK_THROWS_AS(
        fit_spp(few, std::vector<Complex>(y.begin(), y.begin() + 3), options(s, ag_k_spp())),
        std::invalid_argument);
}

namespace {

/// RWG coefficients of a field with the RT0 form J(r) = a + b (r - r0) (exact in the space):
/// the normal component at each edge midpoint, normal pointing out of T_n^+.
VectorXc rt0_coefficients(const basis::RwgSpace& space, const Vec3& a, Real b, int block) {
    const Index N = space.size();
    const Vertices& V = space.mesh().vertices();
    VectorXc x = VectorXc::Zero(2 * N);
    for (Index n = 0; n < N; ++n) {
        const std::array<Index, 2> e = edge_vertices(space, n);
        const Vec3 p = V.row(e[0]).transpose();
        const Vec3 q = V.row(e[1]).transpose();
        const Vec3 mid = 0.5 * (p + q);
        const Vec3 t = (q - p).normalized();
        const Vec3 fp = V.row(space.plus_free_vertex(n)).transpose();
        Vec3 m = (mid - fp) - (mid - fp).dot(t) * t;
        m.normalize();
        const Vec3 J = a + b * mid;
        x(static_cast<Index>(block) * N + n) = J.dot(m);
    }
    return x;
}

geometry::OpenSurfaceMesh small_flat_mesh() {
    SppCase c;
    c.patch = 0.5e-6;
    c.taper = 0.25e-6;
    c.pml_start = 0.75e-6;
    c.pml_width = 0.25e-6;
    c.mesh_size = 125e-9;
    return make_open_mesh(c, false);
}

}  // namespace

TEST_CASE("pml_spp: axis sampling reproduces RT0 fields exactly", "[pml_spp]") {
    const geometry::OpenSurfaceMesh om = small_flat_mesh();
    const basis::RwgSpace space(om.mesh);
    const Real h = 125e-9;
    // Constant field (J and M blocks differently).
    VectorXc x = rt0_coefficients(space, Vec3(0.7, -0.4, 0.0), 0.0, 0) +
                 rt0_coefficients(space, Vec3(-0.2, 1.3, 0.0), 0.0, 1);
    const AxisData d = axis_data(space, x, h);
    REQUIRE(d.jx.x.size() >= 10);
    REQUIRE(d.my.x.size() >= 10);
    REQUIRE(d.divj.x.size() >= 10);
    for (const Complex v : d.jx.v) CHECK(std::abs(v - 0.7) < 1e-12);
    for (const Complex v : d.my.v) CHECK(std::abs(v - 1.3) < 1e-12);
    // Triangles at the open edge miss the flux through their boundary edge (no RWG function).
    const Real inner = om.outer_half - h;
    for (std::size_t i = 0; i < d.divj.x.size(); ++i)
        if (std::abs(d.divj.x[i]) < inner)
            CHECK(std::abs(d.divj.v[i]) < 1e-12 / h);
    // Edges at one x appear once (the two at y = +-h/2 are averaged), sorted by x.
    for (std::size_t i = 1; i < d.jx.x.size(); ++i) CHECK(d.jx.x[i] - d.jx.x[i - 1] > 0.5 * h);
    // J = b r (in-plane): J_x = b x on the axis, div_s J = 2 b.
    const Real b = 3.0e6;
    x = rt0_coefficients(space, Vec3::Zero(), b, 0);
    const AxisSamples jx = edge_current_on_axis(space, x, 0, 0, h);
    for (std::size_t i = 0; i < jx.x.size(); ++i)
        CHECK(std::abs(jx.v[i] - b * jx.x[i]) < 1e-9 * b * h);
    const AxisSamples dv = divergence_on_axis(space, x, 0, h);
    for (std::size_t i = 0; i < dv.x.size(); ++i)
        if (std::abs(dv.x[i]) < inner)
            CHECK(std::abs(dv.v[i] - 2.0 * b) < 1e-9 * b);
    // Window selection on the negative half-axis.
    std::vector<Real> rho;
    std::vector<Complex> v;
    select_window(jx, -1, 0.2e-6, 0.6e-6, rho, v);
    REQUIRE(!rho.empty());
    for (std::size_t i = 0; i < rho.size(); ++i) {
        CHECK(rho[i] >= 0.2e-6);
        CHECK(rho[i] <= 0.6e-6 * (1.0 + 1e-9));
        CHECK(std::abs(v[i] + b * rho[i]) < 1e-9 * b * h);
    }
    CHECK_THROWS_AS(edge_current_on_axis(space, x, 2, 0, h), std::invalid_argument);
    CHECK_THROWS_AS(edge_current_on_axis(space, VectorXc::Zero(3), 0, 0, h), std::invalid_argument);
}

TEST_CASE("pml_spp: localized source keeps only the rows near the origin", "[pml_spp]") {
    const geometry::OpenSurfaceMesh om = small_flat_mesh();
    const basis::RwgSpace space(om.mesh);
    const Index N = space.size();
    VectorXc b(2 * N);
    for (Index i = 0; i < 2 * N; ++i) b(i) = Complex(static_cast<Real>(i) + 1.0, 0.5);
    const Real radius = 150e-9;
    const VectorXc e = localized_rhs(space, b, radius);
    Index kept = 0;
    for (Index n = 0; n < N; ++n) {
        const bool inside = edge_midpoint(space, n).norm() <= radius;
        CHECK(e(n) == (inside ? b(n) : Complex(0.0, 0.0)));
        CHECK(e(N + n) == (inside ? b(N + n) : Complex(0.0, 0.0)));
        kept += inside ? 1 : 0;
    }
    CHECK(kept > 0);
    CHECK_THROWS_AS(localized_rhs(space, b, 1e-9), std::invalid_argument);
}

TEST_CASE("pml_spp: grooved and flat meshes share the grid; size estimate", "[pml_spp]") {
    SppCase c;
    c.patch = 0.5e-6;
    c.taper = 0.25e-6;
    c.pml_start = 0.75e-6;
    c.pml_width = 0.25e-6;
    c.mesh_size = 125e-9;
    c.groove_radius = 100e-9;
    const geometry::OpenSurfaceMesh flat = make_open_mesh(c, false);
    const geometry::OpenSurfaceMesh grooved = make_open_mesh(c, true);
    REQUIRE(flat.mesh.vertices().rows() == grooved.mesh.vertices().rows());
    CHECK((flat.mesh.vertices().leftCols(2).array() == grooved.mesh.vertices().leftCols(2).array())
              .all());
    CHECK(grooved.mesh.vertices().col(2).maxCoeff() > 0.9 * c.groove_depth);
    CHECK(flat.mesh.vertices().col(2).cwiseAbs().maxCoeff() == 0.0);
    const Size s = size_of(c);
    CHECK(s.unknowns == 2 * flat.mesh.num_edges());
    CHECK(std::abs(s.half[2] - 0.75e-6) < 1e-12);
    CHECK(std::abs(s.half[3] - 1.0e-6) < 1e-12);
    c.pml_start = 0.875e-6;  // gap 0.375 um = 3 h: exact
    CHECK_NOTHROW(make_open_mesh(c, false));
    c.pml_start = 0.8e-6;  // gap 0.3 um rounds up to 0.375 um: refused
    CHECK_THROWS_AS(make_open_mesh(c, false), std::invalid_argument);
    c.pml_start = 0.5e-6;  // gap 0
    CHECK_THROWS_AS(make_open_mesh(c, false), std::invalid_argument);
}

TEST_CASE("pml_spp: in-place LU solves with Eigen and with LAPACK", "[pml_spp]") {
    // Fixed-seed complex matrix with a dominant diagonal; the Eigen path factorises the moved-in
    // storage through an Eigen::Ref (no second matrix), the LAPACK path (when built) with zgetrf.
    std::mt19937_64 gen(20261015);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    const Index n = 40;
    MatrixXc Z(n, n);
    for (Index i = 0; i < n; ++i)
        for (Index k = 0; k < n; ++k) Z(i, k) = Complex(u(gen), u(gen));
    Z.diagonal().array() += Complex(static_cast<Real>(n), 0.0);
    VectorXc b(n);
    for (Index i = 0; i < n; ++i) b(i) = Complex(u(gen), u(gen));
    for (const bool use_lapack : {false, true}) {
        MatrixXc copy = Z;
        const InPlaceLu lu(std::move(copy), use_lapack);
        if (!use_lapack)
            CHECK_FALSE(lu.lapack());
        const VectorXc x = lu.solve(b);
        INFO("lapack " << lu.lapack());
        CHECK((Z * x - b).norm() < 1e-12 * b.norm());
        CHECK_THROWS_AS(lu.solve(VectorXc::Zero(n + 1)), std::invalid_argument);
    }
    CHECK_THROWS_AS(InPlaceLu(MatrixXc::Zero(3, 4), false), std::invalid_argument);
}
