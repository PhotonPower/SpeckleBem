#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/kernels/quadrature.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace specklebem;
using namespace specklebem::excitation;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr Real kPi = constants::pi;
constexpr Complex kJ{0.0, 1.0};
constexpr Real kLambda = 500e-9;

using Field = std::function<Vec3c(const Vec3&)>;
using Params = AngularSpectrumBeam::Params;

Params beam_params(Real w0, Real theta = 0.0, Polarization pol = Polarization::P) {
    Params p;
    p.wavelength = kLambda;
    p.waist_radius = w0;
    p.incidence_angle = theta;
    p.polarization = pol;
    return p;
}

/// Unconjugated cross product of complex vectors.
Vec3c ccross(const Vec3& a, const Vec3c& b) {
    return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// Unconjugated cross product of complex vectors (Eigen's cross conjugates complex results).
Vec3c cross_cc(const Vec3c& a, const Vec3c& b) {
    return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

/// Jacobian d f_i / d x_j by central differences with one Richardson step (error O(h^4)).
Eigen::Matrix<Complex, 3, 3> jacobian(const Field& f, const Vec3& r, Real h) {
    Eigen::Matrix<Complex, 3, 3> jac;
    for (int j = 0; j < 3; ++j) {
        const auto central = [&](Real step) {
            const Vec3 dr = step * Vec3::Unit(j);
            return Vec3c((f(r + dr) - f(r - dr)) / (2 * step));
        };
        jac.col(j) = (4.0 * central(0.5 * h) - central(h)) / 3.0;
    }
    return jac;
}

Vec3c curl_of(const Eigen::Matrix<Complex, 3, 3>& jac) {
    return {jac(2, 1) - jac(1, 2), jac(0, 2) - jac(2, 0), jac(1, 0) - jac(0, 1)};
}

/// Counts the warnings logged while it lives (as in test_excitation.cpp).
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
    WarningCounter(WarningCounter&&) = delete;
    WarningCounter& operator=(WarningCounter&&) = delete;
    [[nodiscard]] std::size_t count() const { return sink_->last_formatted().size(); }

private:
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
    spdlog::level::level_enum saved_level_;
};

/// R_y(theta) applied to a beam-frame vector.
Vec3 rotate_y(Real theta, const Vec3& v) {
    const Real c = std::cos(theta), s = std::sin(theta);
    return {c * v.x() + s * v.z(), v.y(), -s * v.x() + c * v.z()};
}
Vec3c rotate_yc(Real theta, const Vec3c& v) {
    const Real c = std::cos(theta), s = std::sin(theta);
    return {c * v.x() + s * v.z(), v.y(), -s * v.x() + c * v.z()};
}

/// The WP-V1 study beam (benchmarks/box_validity.cpp, normal incidence, x-polarised, vacuum,
/// focus at the origin) on a fixed grid: the oracle of the construction.
struct StudyBeam {
    std::vector<Vec3> k;  // k k_hat
    std::vector<Vec3> e;  // c_i e_i, normalised to E_x(0) = 1
    StudyBeam(Real w0, int n_alpha, int n_phi) {
        const Real kk = 2 * kPi / kLambda;
        const Real s = std::sqrt(4.0 * 39.0) / (kk * w0);
        const Real a_max = s >= 1.0 ? 0.5 * kPi : std::asin(s);
        const kernels::LineRule gl = kernels::gauss_legendre(n_alpha);
        const Real dphi = 2 * kPi / n_phi;
        Real norm = 0;
        for (int ia = 0; ia < n_alpha; ++ia) {
            const auto ua = static_cast<std::size_t>(ia);
            const Real a = 0.5 * a_max * (gl.nodes[ua] + 1.0);
            const Real wa = 0.5 * a_max * gl.weights[ua];
            const Real kt = kk * std::sin(a);
            const Real c = wa * dphi * std::exp(-0.25 * kt * kt * w0 * w0) * kk * kk * std::sin(a) *
                           std::cos(a);
            for (int ip = 0; ip < n_phi; ++ip) {
                const Real p = dphi * ip;
                const Vec3 kh(std::sin(a) * std::cos(p), std::sin(a) * std::sin(p), std::cos(a));
                k.push_back(kk * kh);
                e.push_back(c * (Vec3::UnitX() - kh.x() * kh));
                norm += c * (1.0 - kh.x() * kh.x());
            }
        }
        for (Vec3& v : e) v /= norm;
    }
    [[nodiscard]] Vec3c electric_field(const Vec3& r) const {
        Vec3c out = Vec3c::Zero();
        for (std::size_t i = 0; i < k.size(); ++i) {
            out += e[i].cast<Complex>() * std::exp(-kJ * k[i].dot(r));
        }
        return out;
    }
    [[nodiscard]] Vec3c magnetic_field(const Vec3& r) const {
        Vec3c out = Vec3c::Zero();
        for (std::size_t i = 0; i < k.size(); ++i) {
            out += (k[i].normalized().cross(e[i]) / constants::eta0).cast<Complex>() *
                   std::exp(-kJ * k[i].dot(r));
        }
        return out;
    }
};

}  // namespace

TEST_CASE("angular-spectrum beam: every plane wave is exact, so the sum solves Maxwell",
          "[excitation]") {
    // Oblique p beam in a dense background (n = 1.5): each wave is propagating, transverse and
    // carries H = k_hat x E / eta, hence div E = 0 and curl E = -j w mu H hold for the sum to
    // round-off. The analytic derivatives of the sum and the evaluation path (branch-free sincos
    // near the focus, std::sin/cos beyond k |r - focus| = 1e6) are checked against it.
    // Measured: evaluation vs direct sum <= 2.5e-16 V/m, |div E| <= 1e-17 k V/m,
    // |curl E + j w mu H| <= 3e-12 k V/m (the constants mismatch below).
    const material::Material glass{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    auto p = beam_params(1e-6, 0.5);
    p.focus = Vec3(0.3e-6, -0.2e-6, 0.1e-6);
    const AngularSpectrumBeam beam(p, glass);
    const Real k = beam.background().wavenumber(beam.omega()).real();
    const Real eta = beam.background().wave_impedance(beam.omega()).real();
    const Real wmu = beam.omega() * constants::mu0;
    const Vec3 k0(std::sin(0.5), 0, std::cos(0.5));
    const auto dirs = beam.plane_wave_directions();
    const auto amps = beam.plane_wave_amplitudes();
    REQUIRE(static_cast<Index>(dirs.size()) == beam.num_plane_waves());
    REQUIRE(beam.num_plane_waves() == Index{beam.polar_order()} * beam.azimuth_order());
    Real a_max = 0;
    for (const Vec3& a : amps) a_max = std::max(a_max, a.norm());
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        REQUIRE_THAT(dirs[i].norm(), WithinAbs(1.0, 1e-15));
        REQUIRE(dirs[i].dot(k0) > 0);
        REQUIRE(std::abs(dirs[i].dot(amps[i])) <= 1e-15 * a_max);
    }
    const Real far = 1.2e6 / k;  // slow evaluation path
    for (const Vec3& d :
         {Vec3(0, 0, 0), Vec3(1e-6, 0.5e-6, -2e-6), Vec3(-3e-6, 1e-6, 2e-6), Vec3(far, 0, 0)}) {
        const Vec3 r = p.focus + d;
        Vec3c e = Vec3c::Zero(), h = Vec3c::Zero(), res = Vec3c::Zero();
        Complex div = 0;
        for (std::size_t i = 0; i < dirs.size(); ++i) {
            const Complex ph = std::exp(-kJ * (k * dirs[i]).dot(d));
            const Vec3c ei = amps[i].cast<Complex>() * ph;
            const Vec3c hi = ccross(dirs[i], ei) / eta;
            e += ei;
            h += hi;
            div += -kJ * k * dirs[i].dot(amps[i]) * ph;            // div E_i
            res += -kJ * k * ccross(dirs[i], ei) + kJ * wmu * hi;  // curl E_i + j w mu H_i
        }
        const auto [ef, hf] = beam.fields(r);
        // Phase rounding grows like k |d| eps.
        const Real tol = 1e-13 * (1 + k * d.norm());
        CAPTURE(d.transpose(), (beam.electric_field(r) - e).norm(), std::abs(div), res.norm());
        CHECK((beam.electric_field(r) - e).norm() <= tol);
        CHECK((beam.magnetic_field(r) - h).norm() * eta <= tol);
        CHECK((ef - beam.electric_field(r)).norm() <= 1e-15);
        CHECK((hf - beam.magnetic_field(r)).norm() * eta <= 1e-15);
        CHECK(std::abs(div) <= 1e-14 * k);
        // The rounded CODATA constants of core/types.hpp give k eta = w mu only to
        // |eta0 / (mu0 c0) - 1| = 3e-11; that mismatch bounds the residual, not the beam.
        const Real constants_mismatch =
            std::abs(constants::eta0 / (constants::mu0 * constants::c0) - 1);
        CHECK(res.norm() <= (1e-14 + constants_mismatch) * k);
    }
}

TEST_CASE("angular-spectrum beam: finite-difference Maxwell residual at round-off level",
          "[excitation]") {
    // Measured (w0 = 1 um, theta_in = 0.4, h = lambda/400, O(h^4) Richardson): curl residuals
    // 7.1e-11 ... 8.9e-11, div E / (k |E|) <= 3.2e-11 (finite-difference error); the
    // paraxial GaussianBeam of the same waist has (lambda/(pi w0))^2 / 2 = 1.3e-2.
    for (const Polarization pol : {Polarization::P, Polarization::S}) {
        const auto p = beam_params(1e-6, 0.4, pol);
        const AngularSpectrumBeam beam(p);
        const Field e = [&](const Vec3& x) { return beam.electric_field(x); };
        const Field hf = [&](const Vec3& x) { return beam.magnetic_field(x); };
        const Real w = beam.omega();
        for (const Vec3& r : {Vec3(0, 0, 0), Vec3(1e-6, 0, 0), Vec3(0.3e-6, -0.8e-6, 1.5e-6),
                              Vec3(-1.2e-6, 0.4e-6, -2e-6)}) {
            const auto je = jacobian(e, r, kLambda / 400);
            const auto jh = jacobian(hf, r, kLambda / 400);
            const Vec3c E = beam.electric_field(r);
            const Vec3c H = beam.magnetic_field(r);
            const Real res_e = (curl_of(je) + kJ * w * constants::mu0 * H).norm() /
                               (w * constants::mu0 * H.norm());
            const Real res_h = (curl_of(jh) - kJ * w * constants::eps0 * E).norm() /
                               (w * constants::eps0 * E.norm());
            const Real div = std::abs(je.trace()) / (2 * kPi / kLambda * E.norm());
            CAPTURE(r.transpose(), res_e, res_h, div);
            CHECK(res_e < 1e-9);
            CHECK(res_h < 1e-9);
            CHECK(div < 1e-9);
        }
    }
}

TEST_CASE("angular-spectrum beam: E(focus) = e0_hat at 1 V/m", "[excitation]") {
    const material::Material glass{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    for (const Real theta : {0.0, kPi / 4, -0.3}) {
        for (const Polarization pol : {Polarization::P, Polarization::S}) {
            auto p = beam_params(0.8e-6, theta, pol);
            p.focus = Vec3(1e-6, 2e-6, -0.5e-6);
            const AngularSpectrumBeam beam(p, glass);
            const Vec3 e0 =
                pol == Polarization::P ? rotate_y(theta, Vec3::UnitX()) : Vec3(Vec3::UnitY());
            const Vec3c E = beam.electric_field(p.focus);
            CAPTURE(theta, E.transpose());
            CHECK((E - e0.cast<Complex>()).norm() < 1e-14);
            // eta H(focus) is along k0_hat x e0_hat with |eta H . (k0 x e0)| close to 1.
            const Vec3 h0 = rotate_y(theta, Vec3::UnitZ()).cross(e0);
            const Vec3c H = beam.magnetic_field(p.focus) *
                            beam.background().wave_impedance(beam.omega()).real();
            CHECK((H - h0.dot(H.real()) * h0.cast<Complex>()).norm() < 1e-14);
            CHECK(std::abs(H.dot(h0.cast<Complex>()).real() - 1.0) < 0.05);
        }
    }
}

TEST_CASE("angular-spectrum beam: reproduces the WP-V1 study construction", "[excitation]") {
    // Normal incidence, x-polarised, vacuum, w0 = 0.667 um (k w0 = 8.4: spectrum cut at k_t = k
    // in both). The study grid of 120 x 160 waves is converged to round-off on |r| <= 4 w0.
    const Real w0 = 6.6667e-7;
    const AngularSpectrumBeam beam(beam_params(w0));
    const StudyBeam study(w0, 120, 160);
    CHECK_THAT(beam.max_polar_angle(), WithinAbs(0.5 * kPi, 0.0));
    for (const Vec3& r :
         {Vec3(0, 0, 0), Vec3(0.5e-6, 0, 0), Vec3(0, 0.6e-6, 0.4e-6),
          Vec3(0.7e-6, -0.7e-6, -1.5e-6), Vec3(-1.2e-6, 0.3e-6, 2.0e-6), Vec3(0, 0, -2.6e-6)}) {
        CAPTURE(r.transpose());
        CHECK((beam.electric_field(r) - study.electric_field(r)).norm() < 1e-12);
        CHECK((beam.magnetic_field(r) - study.magnetic_field(r)).norm() * constants::eta0 < 1e-12);
    }
}

TEST_CASE("angular-spectrum beam: paraxial limit agrees with GaussianBeam to O(f^2)",
          "[excitation]") {
    // f = lambda/(pi w0) (divergence half-angle). Points rho <= 2 w0, |zeta| <= z_R/4,
    // theta_in = 0.2, p and s. Bound used: max |E - E_paraxial| <= f^2 / 2 V/m; measured for
    // w0 = 5 lambda (f^2 = 4.05e-3): 1.49e-3 (p) and 1.41e-3 (s), i.e. 0.35-0.37 f^2; w0 = 10
    // lambda gives 0.2496 of that (the O(f^2) scaling).
    const auto max_diff = [](Real w0, Polarization pol) {
        const Real theta = 0.2;
        const Real z_r = kPi * w0 * w0 / kLambda;
        auto p = beam_params(w0, theta, pol);
        p.region_radius = std::hypot(2 * w0, 0.25 * z_r) * 1.01;
        const AngularSpectrumBeam beam(p);
        GaussianBeam::Params gp;
        gp.wavelength = kLambda;
        gp.waist_radius = w0;
        gp.incidence_angle = theta;
        gp.polarization = pol;
        const GaussianBeam par(gp);
        Real dmax = 0;
        for (const Real zeta : {-0.25 * z_r, -0.125 * z_r, 0.0, 0.125 * z_r, 0.25 * z_r}) {
            for (const Real rho : {0.0, 0.5 * w0, w0, 1.5 * w0, 2 * w0}) {
                for (const Real phi : {0.0, 0.7, 1.9, 4.0}) {
                    const Vec3 r =
                        rotate_y(theta, Vec3(rho * std::cos(phi), rho * std::sin(phi), zeta));
                    dmax = std::max(dmax, (beam.electric_field(r) - par.electric_field(r)).norm());
                }
            }
        }
        return dmax;
    };
    for (const Polarization pol : {Polarization::P, Polarization::S}) {
        const Real f5 = kLambda / (kPi * 5 * kLambda);
        const Real d5 = max_diff(5 * kLambda, pol);
        const Real d10 = max_diff(10 * kLambda, pol);
        CAPTURE(d5, d10, f5 * f5);
        CHECK(d5 < 0.5 * f5 * f5);
        CHECK(d10 / d5 > 0.22);
        CHECK(d10 / d5 < 0.28);
    }
}

TEST_CASE("angular-spectrum beam: oblique 45 deg central direction and polarisation",
          "[excitation]") {
    const Real theta = kPi / 4;
    const Real w0 = 1.5e-6;
    const Vec3 k0 = rotate_y(theta, Vec3::UnitZ());
    const Real k = 2 * kPi / kLambda;
    const Real f = 2.0 / (k * w0);  // lambda/(pi w0)
    for (const Polarization pol : {Polarization::P, Polarization::S}) {
        const AngularSpectrumBeam beam(beam_params(w0, theta, pol));
        const AngularSpectrumBeam normal(beam_params(w0, 0.0, pol));
        const Vec3 e0 =
            pol == Polarization::P ? rotate_y(theta, Vec3::UnitX()) : Vec3(Vec3::UnitY());
        // E(focus) is transverse to k0 and along e0 (s: along y).
        const Vec3c E = beam.electric_field(Vec3::Zero());
        CHECK(std::abs(E.dot(k0.cast<Complex>())) < 1e-15);
        if (pol == Polarization::S) {
            CHECK(std::abs(E.x()) < 1e-15);
            CHECK(std::abs(E.z()) < 1e-15);
        }
        // Phase gradient of E . e0 at the focus: along -k0 with the Gouy correction,
        // d arg / d zeta = -k + 1/z_R = -k (1 - f^2/2); measured -k (1 - 0.500012 f^2) (f = 0.106).
        const Field fe = [&](const Vec3& x) { return beam.electric_field(x); };
        const auto jac = jacobian(fe, Vec3::Zero(), kLambda / 50);
        const Eigen::Matrix<Complex, 1, 3> grad = e0.cast<Complex>().transpose() * jac;
        const Complex fval = E.dot(e0.cast<Complex>());
        const Vec3 phase_grad = (grad / fval).imag().transpose();
        CAPTURE(phase_grad.transpose() / k, f);
        CHECK((phase_grad - phase_grad.dot(k0) * k0).norm() < 1e-8 * k);
        CHECK_THAT(-phase_grad.dot(k0) / k, WithinAbs(1.0 - 0.5 * f * f, 1e-3 * f * f));
        // Rotation covariance: E_45(R d) = R E_0(d) (the grid is built in the beam frame).
        CHECK(beam.num_plane_waves() == normal.num_plane_waves());
        for (const Vec3& d : {Vec3(1e-6, 0.5e-6, 2e-6), Vec3(-2e-6, 1e-6, -3e-6)}) {
            const Vec3c ref = rotate_yc(theta, normal.electric_field(d));
            CHECK((beam.electric_field(rotate_y(theta, d)) - ref).norm() < 1e-12);
        }
    }
}

TEST_CASE("angular-spectrum beam: power through the waist plane", "[excitation]") {
    // w0 = 5 lambda, theta_in = 0.3, background n = 1.5: 0.5 Re(E x H*) . k0_hat integrated over
    // the disc rho <= 4 w0 of the plane through the focus normal to k0_hat (Gauss-Legendre in rho
    // x trapezoid in phi) equals power() (Parseval) and the numerically integrated paraxial
    // GaussianBeam flux (pi w0^2/(4 eta)) to O(f^2), f = lambda_1/(pi w0). Leading order: the
    // projection and cos(alpha) give P / P_paraxial = <1 - s^2>_{A^2} / <1 - s^2/2>_A^2
    // = 1 + 2/(k w0)^2 = 1 + f^2/2 (s = k_t/k); measured 1 + 0.5006 f^2 (f^2 = 1.8e-3). The disc
    // integral matches power() and pi w0^2/(4 eta) to 1.7e-14.
    const material::Material glass{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    const Real w0 = 5 * kLambda;
    const Real theta = 0.3;
    const AngularSpectrumBeam beam(beam_params(w0, theta), glass);
    GaussianBeam::Params gp;
    gp.wavelength = kLambda;
    gp.waist_radius = w0;
    gp.incidence_angle = theta;
    const GaussianBeam par(gp, glass);
    const Vec3 k0 = rotate_y(theta, Vec3::UnitZ());
    const kernels::LineRule gl = kernels::gauss_legendre(40);
    const int n_phi = 16;
    const Real rho_max = 4 * w0;
    Real p_beam = 0, p_par = 0;
    for (std::size_t i = 0; i < gl.nodes.size(); ++i) {
        const Real rho = 0.5 * rho_max * (gl.nodes[i] + 1);
        const Real wr = 0.5 * rho_max * gl.weights[i] * rho * 2 * kPi / n_phi;
        for (int j = 0; j < n_phi; ++j) {
            const Real phi = 2 * kPi * j / n_phi;
            const Vec3 r = rotate_y(theta, Vec3(rho * std::cos(phi), rho * std::sin(phi), 0));
            const auto [e, h] = beam.fields(r);
            p_beam += 0.5 * wr * cross_cc(e, h.conjugate()).real().dot(k0);
            const Vec3c sp = cross_cc(par.electric_field(r), par.magnetic_field(r).conjugate());
            p_par += 0.5 * wr * sp.real().dot(k0);
        }
    }
    const Real eta = constants::eta0 / 1.5;
    const Real f = kLambda / (1.5 * kPi * w0);
    CAPTURE(p_beam, beam.power(), p_par, kPi * w0 * w0 / (4 * eta), f * f);
    CHECK_THAT(p_par, WithinRel(kPi * w0 * w0 / (4 * eta), 1e-12));
    CHECK_THAT(p_beam, WithinRel(beam.power(), 1e-12));
    CHECK_THAT((p_beam / p_par - 1) / (f * f), WithinAbs(0.5, 0.02));
}

TEST_CASE("angular-spectrum beam: quadrature controls and reporting", "[excitation]") {
    auto p = beam_params(1e-6);
    const AngularSpectrumBeam automatic(p);
    CHECK(automatic.grid_change() < p.tolerance);
    CHECK(automatic.region_radius() == 4e-6);
    CHECK(automatic.azimuth_order() % 4 == 0);
    p.polar_order = 10;
    p.azimuth_order = 13;
    const AngularSpectrumBeam fixed(p);
    CHECK(fixed.polar_order() == 10);
    CHECK(fixed.azimuth_order() == 16);
    CHECK(fixed.num_plane_waves() == 160);
    CHECK(fixed.grid_change() > 1e-3);  // far too coarse for R = 4 w0, reported not thrown
    // Normalisation holds on any grid.
    CHECK((fixed.electric_field(Vec3::Zero()) - Vec3c(1, 0, 0)).norm() < 1e-14);
}

TEST_CASE("angular-spectrum beam: invalid arguments throw", "[excitation]") {
    const auto make = [](const Params& p, material::Material bg = material::vacuum()) {
        return AngularSpectrumBeam(p, bg);
    };
    const Params ok = beam_params(1e-6);
    CHECK_THROWS_AS(make(ok, material::silicon_500nm()), std::invalid_argument);
    auto p = ok;
    p.waist_radius = 0;
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.waist_radius = -1e-6;
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.wavelength = 0;
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.focus = Vec3(0, std::nan(""), 0);
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.incidence_angle = kPi / 2;
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    for (const Real tol : {0.0, -1e-10, 0.1, std::nan("")}) {
        p = ok;
        p.tolerance = tol;
        CHECK_THROWS_AS(make(p), std::invalid_argument);
    }
    p = ok;
    p.region_radius = -1e-6;
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.polar_order = 10;  // azimuth_order missing
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.max_plane_waves = 0;
    CHECK_THROWS_AS(make(p), std::invalid_argument);
    p = ok;
    p.max_plane_waves = 2000;  // the automatic refinement needs more
    CHECK_THROWS_AS(make(p), std::runtime_error);
}

TEST_CASE("angular-spectrum beam: order and plane-wave caps", "[excitation]") {
    const Params ok = beam_params(1e-6);
    // Fixed orders above kMaxOrder are rejected before any grid is built.
    for (const auto& [na, np] : {std::pair{AngularSpectrumBeam::kMaxOrder + 1, 8},
                                 std::pair{8, AngularSpectrumBeam::kMaxOrder + 1},
                                 std::pair{std::numeric_limits<int>::max(), 8},
                                 std::pair{8, std::numeric_limits<int>::max()}}) {
        auto p = ok;
        p.polar_order = na;
        p.azimuth_order = np;
        CHECK_THROWS_AS(AngularSpectrumBeam(p), std::invalid_argument);
    }
    // max_plane_waves applies to the check grid of a fixed grid: 10 x 16 -> 23 x 32 = 736.
    auto p = ok;
    p.polar_order = 10;
    p.azimuth_order = 16;
    p.max_plane_waves = 735;
    CHECK_THROWS_AS(AngularSpectrumBeam(p), std::invalid_argument);
    p.max_plane_waves = 736;
    CHECK(AngularSpectrumBeam(p).num_plane_waves() == 160);
    // Automatic mode: a huge k R gives start orders above kMaxOrder (formerly an int overflow
    // for R = 1 km); rejected before any allocation.
    for (const Real radius : {1.0, 1e3, 1e12}) {
        p = ok;
        p.region_radius = radius;
        p.max_plane_waves = std::numeric_limits<Index>::max();
        CAPTURE(radius);
        CHECK_THROWS_AS(AngularSpectrumBeam(p), std::runtime_error);
    }
}

TEST_CASE("angular-spectrum beam: fixed grids that miss the tolerance warn", "[excitation]") {
    const AngularSpectrumBeam automatic(beam_params(1e-6));
    auto p = beam_params(1e-6);
    {
        // The automatic orders as a fixed grid: converged, no warning.
        p.polar_order = automatic.polar_order();
        p.azimuth_order = automatic.azimuth_order();
        const WarningCounter warnings;
        const AngularSpectrumBeam fixed(p);
        CHECK(fixed.grid_change() < p.tolerance);
        CHECK(warnings.count() == 0);
    }
    {
        p.polar_order = 10;
        p.azimuth_order = 16;
        const WarningCounter warnings;
        const AngularSpectrumBeam coarse(p);
        CHECK(coarse.grid_change() >= p.tolerance);
        CHECK(warnings.count() == 1);
    }
}

TEST_CASE("angular-spectrum beam: controlled region and one-time warning outside it",
          "[excitation]") {
    auto p = beam_params(1e-6, 0.3);
    p.focus = Vec3(0.2e-6, -0.1e-6, 0.4e-6);
    p.region_radius = 3e-6;
    const AngularSpectrumBeam beam(p);
    const Excitation& base = beam;
    CHECK(base.controlled_radius() == 3e-6);
    CHECK(base.controlled_center() == p.focus);
    // Sources defined everywhere: infinite controlled radius around the origin.
    const PlaneWave pw(kLambda, Vec3::UnitZ(), Vec3c(1, 0, 0));
    CHECK(std::isinf(pw.controlled_radius()));
    CHECK(pw.controlled_center() == Vec3::Zero());
    GaussianBeam::Params gp;
    gp.wavelength = kLambda;
    gp.focus = Vec3(1e-6, 0, 0);
    const GaussianBeam gb(gp);
    CHECK(std::isinf(gb.controlled_radius()));

    const Vec3 dir = Vec3(1, 2, -2).normalized();
    const WarningCounter warnings;
    // Inside 1.2 R: silent.
    (void)beam.electric_field(p.focus + 1.19 * 3e-6 * dir);
    (void)beam.fields(p.focus + 1.19 * 3e-6 * dir);
    CHECK(warnings.count() == 0);
    // Beyond 1.2 R: one warning, whichever evaluator is used and however often.
    (void)beam.electric_field(p.focus + 1.21 * 3e-6 * dir);
    (void)beam.magnetic_field(p.focus + 2 * 3e-6 * dir);
    (void)base.fields(p.focus + 4 * 3e-6 * dir);
    (void)beam.electric_field(p.focus + 4 * 3e-6 * dir);
    CHECK(warnings.count() == 1);
    // A copy has its own flag.
    const AngularSpectrumBeam copy = beam;  // NOLINT(performance-unnecessary-copy-initialization)
    (void)copy.electric_field(p.focus + 2 * 3e-6 * dir);
    (void)beam.electric_field(p.focus + 2 * 3e-6 * dir);
    CHECK(warnings.count() == 2);
}

TEST_CASE("excitation fields(): one call equals the two separate fields", "[excitation]") {
    // Default implementation (PlaneWave, GaussianBeam): bitwise the two calls; the
    // AngularSpectrumBeam override through the base class: the one-pass sum.
    const PlaneWave pw(kLambda, Vec3(0.3, 0, 1), Vec3c(1, 0, -0.3));
    GaussianBeam::Params gp;
    gp.wavelength = kLambda;
    gp.waist_radius = 1e-6;
    gp.incidence_angle = 0.2;
    const GaussianBeam gb(gp);
    const AngularSpectrumBeam asb(beam_params(1e-6, 0.2));
    for (const Vec3& r : {Vec3(0, 0, 0), Vec3(0.4e-6, -0.7e-6, 1.1e-6), Vec3(-2e-6, 1e-6, -1e-6)}) {
        for (const Excitation* e :
             {static_cast<const Excitation*>(&pw), static_cast<const Excitation*>(&gb)}) {
            const auto [ef, hf] = e->fields(r);
            CHECK(ef == e->electric_field(r));
            CHECK(hf == e->magnetic_field(r));
        }
        const Excitation& base = asb;
        const auto [ef, hf] = base.fields(r);
        CHECK((ef - asb.electric_field(r)).norm() <= 1e-15);
        CHECK((hf - asb.magnetic_field(r)).norm() * constants::eta0 <= 1e-15);
    }
}
