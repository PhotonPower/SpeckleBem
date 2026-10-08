#include "specklebem/core/logging.hpp"
#include "specklebem/excitation/excitation.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>

using namespace specklebem;
using namespace specklebem::excitation;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr Real kPi = constants::pi;
constexpr Complex kJ{0.0, 1.0};

using Field = std::function<Vec3c(const Vec3&)>;

/// Unconjugated cross product (Eigen's cross conjugates the result for complex scalars).
Vec3c ccross(const Vec3c& a, const Vec3c& b) {
    return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// Curl by central differences with one Richardson step (error O(h^4)).
Vec3c curl(const Field& f, const Vec3& r, Real h) {
    Eigen::Matrix<Complex, 3, 3> jac;  // jac(i, j) = d f_i / d x_j
    for (int j = 0; j < 3; ++j) {
        const auto central = [&](Real step) {
            const Vec3 dr = step * Vec3::Unit(j);
            return Vec3c((f(r + dr) - f(r - dr)) / (2 * step));
        };
        jac.col(j) = (4.0 * central(0.5 * h) - central(h)) / 3.0;
    }
    return {jac(2, 1) - jac(1, 2), jac(0, 2) - jac(2, 0), jac(1, 0) - jac(0, 1)};
}

Real omega_of(Real lambda) {
    return 2 * kPi * constants::c0 / lambda;
}

/// Relative residuals of both curl equations at r:
///   first  = |curl E + j w mu H| / |w mu H|,  second = |curl H - j w eps E| / |w eps E|.
std::pair<Real, Real> maxwell_residuals(const Excitation& ex, const Vec3& r, Real h) {
    const Real w = ex.omega();
    const Complex mu = constants::mu0 * ex.background().mu_r;
    const Complex eps = constants::eps0 * ex.background().eps_r;
    const Field e = [&](const Vec3& x) { return ex.electric_field(x); };
    const Field hf = [&](const Vec3& x) { return ex.magnetic_field(x); };
    const Vec3c E = ex.electric_field(r);
    const Vec3c H = ex.magnetic_field(r);
    const Vec3c res_e = curl(e, r, h) + kJ * w * mu * H;
    const Vec3c res_h = curl(hf, r, h) - kJ * w * eps * E;
    return {res_e.norm() / (w * std::abs(mu) * H.norm()),
            res_h.norm() / (w * std::abs(eps) * E.norm())};
}

Vec3 random_unit(std::mt19937_64& rng) {
    std::normal_distribution<Real> n(0.0, 1.0);
    Vec3 v(n(rng), n(rng), n(rng));
    return v.normalized();
}

/// Counts the warnings logged while it is alive (logger level lowered to warn if needed and
/// restored afterwards).
class WarningCounter {
public:
    WarningCounter()
        : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64)),
          saved_level_(spdlog::default_logger()->level()) {
        sink_->set_level(spdlog::level::warn);
        spdlog::default_logger()->set_level(std::min(saved_level_, spdlog::level::warn));
        spdlog::default_logger()->sinks().push_back(sink_);
    }
    ~WarningCounter() {
        auto& sinks = spdlog::default_logger()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
        spdlog::default_logger()->set_level(saved_level_);
    }
    WarningCounter(const WarningCounter&) = delete;
    WarningCounter& operator=(const WarningCounter&) = delete;
    [[nodiscard]] std::size_t count() const { return sink_->last_formatted().size(); }

private:
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
    spdlog::level::level_enum saved_level_;
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Plane wave
// ---------------------------------------------------------------------------------------------

TEST_CASE("excitation wavelength and frequency", "[excitation]") {
    const PlaneWave pw(500e-9, Vec3::UnitZ(), Vec3c(1, 0, 0));
    CHECK_THAT(pw.wavelength(), WithinRel(500e-9, 1e-14));
    CHECK_THAT(pw.omega(), WithinRel(omega_of(500e-9), 1e-14));
    // The wavelength is the vacuum wavelength also in a dense background.
    const material::Material glass{Complex(2.25, 0), Complex(1, 0)};
    const PlaneWave pg(633e-9, Vec3::UnitZ(), Vec3c(1, 0, 0), glass);
    CHECK_THAT(pg.wavelength(), WithinRel(633e-9, 1e-14));
}

TEST_CASE("plane wave: constant amplitude and outgoing phase along k_hat", "[excitation]") {
    const Real lambda = 500e-9;
    const material::Material glass{Complex(2.25, 0), Complex(1, 0)};
    const Vec3 k_hat = Vec3(1, -2, 2).normalized();
    const Vec3c e0 = Vec3c(Complex(2, 0), Complex(1, 0), Complex(0, 0));  // e0 . k_hat = 0
    const PlaneWave pw(lambda, Vec3(1, -2, 2), e0, glass);  // k_hat not normalised on input
    const Real k = 2 * kPi * 1.5 / lambda;

    std::mt19937_64 rng(808);
    std::uniform_real_distribution<Real> u(-5 * lambda, 5 * lambda);
    for (int i = 0; i < 20; ++i) {
        const Vec3 r(u(rng), u(rng), u(rng));
        const Vec3c E = pw.electric_field(r);
        CHECK_THAT(E.norm(), WithinRel(e0.norm(), 1e-13));
        // exp(+jwt): E = e0 exp(-j k k_hat . r), i.e. the phase decreases along k_hat.
        const Complex expected = std::exp(-kJ * k * k_hat.dot(r));
        CHECK(std::abs(E(0) / e0(0) - expected) < 1e-12);
        const Real dz = 0.1 * lambda;
        const Complex ratio = pw.electric_field(r + dz * k_hat)(0) / E(0);
        CHECK_THAT(std::arg(ratio), WithinAbs(-k * dz, 1e-10));
    }
}

TEST_CASE("plane wave: H = k_hat x E / eta", "[excitation]") {
    const material::Material glass{Complex(2.25, 0), Complex(1, 0)};
    const Vec3 k_hat = Vec3(0, 1, 1).normalized();
    const Vec3c e0 = Vec3c(Complex(1, 0), Complex(0, 0.5), Complex(0, -0.5));
    const PlaneWave pw(500e-9, k_hat, e0, glass);
    const Real eta = constants::eta0 / 1.5;
    const Vec3 r(1e-7, -3e-7, 2e-7);
    const Vec3c E = pw.electric_field(r);
    const Vec3c expected = ccross(k_hat.cast<Complex>(), E) / eta;
    CHECK((pw.magnetic_field(r) - expected).norm() < 1e-14 * expected.norm());
}

TEST_CASE("plane wave: docs/06 sphere illumination has H along +y/eta0", "[excitation]") {
    const PlaneWave pw(500e-9, Vec3::UnitZ(), Vec3c(1, 0, 0));
    const Vec3c H0 = pw.magnetic_field(Vec3::Zero());
    CHECK_THAT(H0(0).real(), WithinAbs(0.0, 1e-18));
    CHECK_THAT(H0(1).real(), WithinRel(1.0 / constants::eta0, 1e-12));
    CHECK_THAT(H0(1).imag(), WithinAbs(0.0, 1e-18));
    CHECK_THAT(H0(2).real(), WithinAbs(0.0, 1e-18));
    // Poynting vector Re(E x H*)/2 along +z.
    const Vec3c E0 = pw.electric_field(Vec3::Zero());
    const Vec3c S = ccross(E0, H0.conjugate());
    CHECK(S(2).real() > 0);
}

TEST_CASE("plane wave: finite-difference Maxwell equations", "[excitation]") {
    const Real lambda = 500e-9;
    const material::Material glass{Complex(2.25, 0), Complex(1, 0)};
    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<Real> u(-3 * lambda, 3 * lambda);
    for (const auto& bg : {material::vacuum(), glass}) {
        const Vec3 k_hat = random_unit(rng);
        // Elliptic polarisation built from two vectors perpendicular to k_hat.
        const Vec3 a = k_hat.unitOrthogonal();
        const Vec3 b = k_hat.cross(a);
        const Vec3c e0 = a.cast<Complex>() * Complex(0.7, 0.2) + b.cast<Complex>() * Complex(0, 1);
        const PlaneWave pw(lambda, k_hat, e0, bg);
        for (int i = 0; i < 10; ++i) {
            const Vec3 r(u(rng), u(rng), u(rng));
            const auto [res_e, res_h] = maxwell_residuals(pw, r, 1e-4 * lambda);
            CHECK(res_e < 1e-8);
            CHECK(res_h < 1e-8);
        }
    }
}

TEST_CASE("plane wave: circular polarisation has constant |E|", "[excitation]") {
    const Real lambda = 500e-9;
    const Vec3c e0 = Vec3c(Complex(1, 0), Complex(0, -1), Complex(0, 0)) / std::sqrt(2.0);
    const PlaneWave pw(lambda, Vec3::UnitZ(), e0);
    for (int i = 0; i < 16; ++i) {
        const Vec3 r(0.3 * lambda, -0.1 * lambda, 0.0625 * i * lambda);
        const Vec3c E = pw.electric_field(r);
        CHECK_THAT(E.norm(), WithinRel(1.0, 1e-13));
        // Circular: the real field rotates, |Re E| = |Im E| and Re E . Im E = 0.
        CHECK_THAT(E.real().norm(), WithinRel(E.imag().norm(), 1e-12));
        CHECK_THAT(E.real().dot(E.imag()), WithinAbs(0.0, 1e-13));
    }
}

TEST_CASE("plane wave: invalid inputs throw", "[excitation]") {
    const Vec3c ex(1, 0, 0);
    CHECK_THROWS_AS(PlaneWave(500e-9, Vec3::Zero(), ex), std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(500e-9, Vec3::UnitZ(), Vec3c::Zero()), std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(500e-9, Vec3::UnitZ(), Vec3c(1, 0, 1e-6)), std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(500e-9, Vec3::UnitX(), ex), std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(0.0, Vec3::UnitZ(), ex), std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(-500e-9, Vec3::UnitZ(), ex), std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(500e-9, Vec3::UnitZ(), ex, material::silver_500nm()),
                    std::invalid_argument);
    CHECK_THROWS_AS(PlaneWave(500e-9, Vec3::UnitZ(), ex, material::silicon_500nm()),
                    std::invalid_argument);
    CHECK_NOTHROW(PlaneWave(500e-9, Vec3::UnitZ(), Vec3c(1, 0, 1e-13)));
}

// ---------------------------------------------------------------------------------------------
// Gaussian beam
// ---------------------------------------------------------------------------------------------

namespace {

GaussianBeam::Params beam_params(Real w0, Real theta = 0.0, Polarization pol = Polarization::P) {
    GaussianBeam::Params p;
    p.wavelength = 500e-9;
    p.waist_radius = w0;
    p.focus = Vec3(0.2e-6, -0.1e-6, 0.3e-6);
    p.incidence_angle = theta;
    p.polarization = pol;
    return p;
}

/// Textbook transverse field (w, R, psi form) for comparison with the implementation.
Complex textbook_amplitude(Real k, Real w0, Real zeta, Real rho) {
    const Real z_r = 0.5 * k * w0 * w0;
    const Real w = w0 * std::sqrt(1 + (zeta / z_r) * (zeta / z_r));
    const Real inv_r = (zeta == 0) ? 0.0 : 1.0 / (zeta * (1 + (z_r / zeta) * (z_r / zeta)));
    const Real psi = std::atan(zeta / z_r);
    return (w0 / w) * std::exp(-rho * rho / (w * w)) *
           std::exp(-kJ * (k * zeta + 0.5 * k * rho * rho * inv_r - psi));
}

}  // namespace

TEST_CASE("Gaussian beam: on-axis amplitude w0/w and Gouy phase", "[excitation]") {
    const auto p = beam_params(3e-6);
    const GaussianBeam beam(p);
    const Real k = 2 * kPi / p.wavelength;
    const Real z_r = kPi * p.waist_radius * p.waist_radius / p.wavelength;
    CHECK_THAT(beam.wavelength(), WithinRel(p.wavelength, 1e-14));
    for (const Real t : {-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0}) {
        const Real zeta = t * z_r;
        const Vec3 r = p.focus + zeta * Vec3::UnitZ();
        const Vec3c E = beam.electric_field(r);
        const Real w = p.waist_radius * std::sqrt(1 + t * t);
        CHECK_THAT(std::abs(E(0)), WithinRel(p.waist_radius / w, 1e-12));
        CHECK(std::abs(E(1)) == 0.0);
        CHECK(std::abs(E(2)) < 1e-15);  // no longitudinal field on the axis
        // Phase relative to the carrier: + Gouy phase atan(zeta/z_R).
        const Complex envelope = E(0) * std::exp(kJ * k * zeta);
        CHECK_THAT(std::arg(envelope), WithinAbs(std::atan(t), 1e-9));
    }
    // Unit amplitude at the focus.
    const Vec3c E0 = beam.electric_field(p.focus);
    CHECK_THAT(E0(0).real(), WithinRel(1.0, 1e-15));
    CHECK_THAT(E0(0).imag(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("Gaussian beam: transverse field matches the w, R, psi form", "[excitation]") {
    for (const Real theta : {0.0, 0.4}) {
        const auto p = beam_params(2e-6, theta);
        const GaussianBeam beam(p);
        const Real k = 2 * kPi / p.wavelength;
        const Vec3 k_hat(std::sin(theta), 0, std::cos(theta));
        const Vec3 e_p(std::cos(theta), 0, -std::sin(theta));
        std::mt19937_64 rng(77);
        std::uniform_real_distribution<Real> u(-4e-6, 4e-6);
        std::uniform_real_distribution<Real> z(-60e-6, 60e-6);
        for (int i = 0; i < 20; ++i) {
            const Vec3 d = u(rng) * e_p + u(rng) * Vec3::UnitY() + z(rng) * k_hat;
            const Real zeta = k_hat.dot(d);
            const Real rho = (d - zeta * k_hat).norm();
            const Complex ref = textbook_amplitude(k, p.waist_radius, zeta, rho);
            const Vec3c E = beam.electric_field(p.focus + d);
            const Complex et = e_p.cast<Complex>().dot(E);
            CHECK(std::abs(et - ref) < 1e-10 * std::abs(ref));
        }
    }
}

TEST_CASE("Gaussian beam: 1/e^2 intensity radius at the waist equals w0", "[excitation]") {
    const auto p = beam_params(3e-6);
    const GaussianBeam beam(p);
    const Real i0 = std::norm(beam.electric_field(p.focus)(0));
    for (const Vec3& dir : {Vec3(Vec3::UnitX()), Vec3(Vec3::UnitY()), Vec3(1, 1, 0).normalized()}) {
        const Vec3c E = beam.electric_field(p.focus + p.waist_radius * dir);
        CHECK_THAT(std::norm(E(0)) / i0, WithinRel(std::exp(-2.0), 1e-12));
        // At the waist the transverse field has a flat phase.
        CHECK_THAT(std::arg(E(0)), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("Gaussian beam: wide beam equals the plane wave near the focus", "[excitation]") {
    const Real lambda = 500e-9;
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    for (const Real theta : {0.0, kPi / 6}) {
        for (const Polarization pol : {Polarization::P, Polarization::S}) {
            GaussianBeam::Params p = beam_params(1.0, theta, pol);
            p.focus = Vec3::Zero();
            const GaussianBeam beam(p);
            const Vec3 k_hat(std::sin(theta), 0, std::cos(theta));
            const Vec3 e_hat = pol == Polarization::P ? Vec3(std::cos(theta), 0, -std::sin(theta))
                                                      : Vec3(Vec3::UnitY());
            const PlaneWave pw(lambda, k_hat, e_hat.cast<Complex>());
            for (int i = 0; i < 20; ++i) {
                Vec3 r(u(rng), u(rng), u(rng));
                if (r.norm() > 1) {
                    r.normalize();
                }
                r *= 9.99 * lambda;  // |r| < 10 lambda
                const Vec3c dE = beam.electric_field(r) - pw.electric_field(r);
                const Vec3c dH = beam.magnetic_field(r) - pw.magnetic_field(r);
                CHECK(dE.norm() < 1e-6);
                CHECK(dH.norm() * constants::eta0 < 1e-6);
            }
        }
    }
}

TEST_CASE("Gaussian beam: incidence angle rotates k_hat and the p-polarisation", "[excitation]") {
    const Real theta = kPi / 6;  // 30 degrees
    const auto p = beam_params(3e-6, theta, Polarization::P);
    const GaussianBeam beam(p);
    const Real k = 2 * kPi / p.wavelength;
    const Vec3 k_hat(0.5, 0, std::sqrt(3.0) / 2);  // R_y(30 deg) z_hat
    for (const Real zeta : {-2e-6, 0.0, 1e-6, 5e-6}) {
        const Vec3 r = p.focus + zeta * k_hat;
        const Vec3c E = beam.electric_field(r);
        const Vec3c H = beam.magnetic_field(r);
        CHECK(std::abs(k_hat.cast<Complex>().dot(E)) < 1e-14 * E.norm());  // E perpendicular
        CHECK(std::abs(E(1)) == 0.0);                                      // E in the xz-plane
        // e_p = R_y(30 deg) x_hat = (cos, 0, -sin): E_z / E_x = -tan(30 deg).
        CHECK_THAT((E(2) / E(0)).real(), WithinRel(-std::tan(theta), 1e-12));
        // H along +y (k_hat x e_p = y_hat), Poynting vector along k_hat.
        CHECK(std::abs(H(0)) < 1e-14 * H.norm());
        CHECK(std::abs(H(2)) < 1e-14 * H.norm());
        const Vec3 S = ccross(E, H.conjugate()).real();
        CHECK_THAT(S.normalized().dot(k_hat), WithinRel(1.0, 1e-12));
    }
    // Carrier phase decreases along the rotated axis.
    const Real dz = 1e-8;
    const Complex ratio =
        beam.electric_field(p.focus + dz * k_hat)(0) / beam.electric_field(p.focus)(0);
    const Real z_r = 0.5 * k * p.waist_radius * p.waist_radius;
    CHECK_THAT(std::arg(ratio), WithinAbs(-k * dz + std::atan(dz / z_r), 1e-12));
    // In the plane of incidence (y = focus_y) the p-polarised beam has E_y = 0 exactly.
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<Real> u(-3e-6, 3e-6);
    for (int i = 0; i < 10; ++i) {
        const Vec3 r = p.focus + Vec3(u(rng), 0, u(rng));
        CHECK(std::abs(beam.electric_field(r)(1)) == 0.0);
        CHECK(std::abs(beam.magnetic_field(r)(0)) == 0.0);
        CHECK(std::abs(beam.magnetic_field(r)(2)) == 0.0);
    }
}

TEST_CASE("Gaussian beam: s-polarisation is along y", "[excitation]") {
    for (const Real theta : {0.0, kPi / 6, -0.3}) {
        const auto p = beam_params(3e-6, theta, Polarization::S);
        const GaussianBeam beam(p);
        const Vec3 k_hat(std::sin(theta), 0, std::cos(theta));
        // On the axis and anywhere in the plane of incidence: E || y_hat, H in the xz-plane.
        std::mt19937_64 rng(6);
        std::uniform_real_distribution<Real> u(-3e-6, 3e-6);
        for (int i = 0; i < 10; ++i) {
            const Vec3 r = p.focus + (i == 0 ? Vec3::Zero().eval() : Vec3(u(rng), 0, u(rng)));
            const Vec3c E = beam.electric_field(r);
            CHECK(std::abs(E(0)) == 0.0);
            CHECK(std::abs(E(2)) == 0.0);
            CHECK(std::abs(E(1)) > 0.0);
            CHECK(std::abs(beam.magnetic_field(r)(1)) == 0.0);
        }
        const Vec3c H = beam.magnetic_field(p.focus);
        const Vec3c expected = k_hat.cross(Vec3::UnitY()).cast<Complex>() / constants::eta0;
        CHECK((H - expected).norm() < 1e-14 * expected.norm());
    }
}

namespace {

/// Maximum Maxwell residuals over sample points inside the waist: rho <= w0 (both transverse
/// directions) and |zeta| <= z_R / 2, with positions scaled by w0 and z_R so that two waists
/// are compared at the same relative positions.
std::pair<Real, Real> max_beam_residuals(Real w0, Real theta, Polarization pol) {
    auto p = beam_params(w0, theta, pol);
    const GaussianBeam beam(p);
    const Real z_r = kPi * w0 * w0 / p.wavelength;
    const Vec3 k_hat(std::sin(theta), 0, std::cos(theta));
    const Vec3 e_p(std::cos(theta), 0, -std::sin(theta));
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    Real max_e = 0;
    Real max_h = 0;
    for (int i = 0; i < 12; ++i) {
        Real a = u(rng);
        Real b = u(rng);
        const Real s = std::max(1.0, std::hypot(a, b));
        a /= s;
        b /= s;
        if (i == 0) {
            a = 1;  // the edge of the waist along both transverse axes
            b = 0;
        } else if (i == 1) {
            a = 0;
            b = 1;
        }
        const Vec3 r = p.focus + w0 * (a * e_p + b * Vec3::UnitY()) + 0.5 * z_r * u(rng) * k_hat;
        const auto [re, rh] = maxwell_residuals(beam, r, 1e-4 * p.wavelength);
        max_e = std::max(max_e, re);
        max_h = std::max(max_h, rh);
    }
    return {max_e, max_h};
}

}  // namespace

TEST_CASE("Gaussian beam: paraxial Maxwell residual scales like (lambda/(pi w0))^2",
          "[excitation]") {
    // Measured (lambda = 500 nm, rho <= w0, |zeta| <= z_R/2, theta_in = 0 and 30 deg, p and s):
    // max |curl E + j w mu H| / |w mu H| = 1.40e-3 for w0 = 3 um and 3.52e-4 for w0 = 6 um
    // (ratio 0.250), i.e. (lambda/(pi w0))^2 / 2; the curl-H residual is the same. Without the
    // longitudinal components the residual would be first order, 2 rho/(k w0^2) = 0.053 at
    // rho = w0 for w0 = 3 um, and would only halve for w0 = 6 um.
    for (const Real theta : {0.0, kPi / 6}) {
        for (const Polarization pol : {Polarization::P, Polarization::S}) {
            const auto [e3, h3] = max_beam_residuals(3e-6, theta, pol);
            const auto [e6, h6] = max_beam_residuals(6e-6, theta, pol);
            INFO("theta = " << theta << ", residuals w0=3um: " << e3 << ", " << h3
                            << "; w0=6um: " << e6 << ", " << h6);
            const Real bound = std::pow(500e-9 / (kPi * 3e-6), 2);  // (lambda/(pi w0))^2
            CHECK(e3 < bound);
            CHECK(h3 < bound);
            CHECK(e6 / e3 > 0.2);
            CHECK(e6 / e3 < 0.3);
            CHECK(h6 / h3 > 0.2);
            CHECK(h6 / h3 < 0.3);
        }
    }
}

TEST_CASE("Gaussian beam: transverse-only fields would be first-order accurate", "[excitation]") {
    // Documents why the longitudinal components are included: the purely transverse paraxial
    // fields E_t = e_hat A, H_t = k_hat x E_t / eta miss (curl E_t)_zeta = -dA/dv (p-pol), a
    // relative error 2 rho/(k w0^2) = lambda/(pi w0) at rho = w0 that only halves when w0
    // doubles.
    const auto residual = [](Real w0) {
        const auto p = beam_params(w0);
        const GaussianBeam beam(p);
        const Vec3 k_hat = Vec3::UnitZ();
        const auto transverse = [&](const Vec3c& f) {
            return Vec3c(f - k_hat.cast<Complex>() * f(2));
        };
        const Field e = [&](const Vec3& x) { return transverse(beam.electric_field(x)); };
        const Vec3 r = p.focus + w0 * Vec3::UnitY();
        const Vec3c H = transverse(beam.magnetic_field(r));
        const Complex wmu = beam.omega() * constants::mu0;
        return (curl(e, r, 1e-4 * p.wavelength) + kJ * wmu * H).norm() / std::abs(wmu * H.norm());
    };
    const Real r3 = residual(3e-6);
    const Real r6 = residual(6e-6);
    CHECK_THAT(r3, WithinRel(500e-9 / (kPi * 3e-6), 0.03));
    CHECK_THAT(r6 / r3, WithinRel(0.5, 0.03));
}

TEST_CASE("Gaussian beam: one paraxial warning per constructed beam", "[excitation]") {
    const WarningCounter counter;
    const GaussianBeam a(beam_params(3e-6));
    const GaussianBeam b(beam_params(6e-6));
    (void)a.electric_field(Vec3::Zero());
    (void)b.magnetic_field(Vec3::Zero());
    CHECK(counter.count() == 2);
}

TEST_CASE("Gaussian beam: invalid inputs throw", "[excitation]") {
    auto p = beam_params(3e-6);
    p.waist_radius = 0;
    CHECK_THROWS_AS(GaussianBeam(p), std::invalid_argument);
    p.waist_radius = -1e-6;
    CHECK_THROWS_AS(GaussianBeam(p), std::invalid_argument);
    p = beam_params(3e-6);
    p.wavelength = 0;
    CHECK_THROWS_AS(GaussianBeam(p), std::invalid_argument);
    p.wavelength = -500e-9;
    CHECK_THROWS_AS(GaussianBeam(p), std::invalid_argument);
    p = beam_params(3e-6);
    CHECK_THROWS_AS(GaussianBeam(p, material::silver_500nm()), std::invalid_argument);
    p.focus.x() = std::nan("");
    CHECK_THROWS_AS(GaussianBeam(p), std::invalid_argument);
    // docs/06: the beam travels towards +z, |theta_in| < pi/2.
    for (const Real theta : {kPi / 2, -kPi / 2, 2.0, -3.0, std::nan("")}) {
        p = beam_params(3e-6, theta);
        CHECK_THROWS_AS(GaussianBeam(p), std::invalid_argument);
    }
    p = beam_params(3e-6, 1.5);
    CHECK_NOTHROW(GaussianBeam(p));
    p = beam_params(3e-6);
    const material::Material glass{Complex(2.25, 0), Complex(1, 0)};
    CHECK_NOTHROW(GaussianBeam(p, glass));
}

TEST_CASE("Gaussian beam: dense background shortens the Rayleigh range", "[excitation]") {
    const material::Material glass{Complex(2.25, 0), Complex(1, 0)};
    const auto p = beam_params(3e-6);
    const GaussianBeam beam(p, glass);
    const Real z_r = kPi * p.waist_radius * p.waist_radius * 1.5 / p.wavelength;
    const Vec3c E = beam.electric_field(p.focus + z_r * Vec3::UnitZ());
    CHECK_THAT(std::abs(E(0)), WithinRel(1 / std::sqrt(2.0), 1e-12));
    const Vec3c H = beam.magnetic_field(p.focus);
    CHECK_THAT(H(1).real(), WithinRel(1.5 / constants::eta0, 1e-12));
    const auto [re, rh] = maxwell_residuals(beam, p.focus + Vec3(1e-6, 1e-6, 2e-6), 5e-11);
    CHECK(re < 0.05);
    CHECK(rh < 0.05);
}
