// Diagnostics of the absorbed power of the flat box (WP-A1,
// tests/support/ag_absorption_support.hpp; study benchmarks/ag_absorption.cpp, record
// benchmarks/results/ag_absorption.md): the analytic interface fields of the beam (sign conventions
// via the impedance of the transmitted wave, the absorbed flux against 1 - R_beam), the extinction
// integral against the optical theorem, and the consistency of the per-part absorbed power, the RWG
// interpolant and the top-face metrics.
#include "specklebem/geometry/sphere.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <random>

#include "ag_absorption_support.hpp"
#include "fresnel_flat_support.hpp"

using namespace fresnel_flat;
namespace aa = ag_absorption;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

excitation::AngularSpectrumBeam make_test_beam(Real w0, Real theta_deg,
                                               excitation::Polarization pol, Real region_radius) {
    excitation::AngularSpectrumBeam::Params p;
    p.wavelength = kLambda;
    p.waist_radius = w0;
    p.incidence_angle = theta_deg * kDeg;
    p.polarization = pol;
    p.region_radius = region_radius;
    return excitation::AngularSpectrumBeam(p, material::vacuum());
}

// Not inlined: GCC 16 -O3 otherwise speculatively devirtualises the excitation calls of the test
// body to the (final, inline) AngularSpectrumBeam members of this translation unit and warns
// (-Warray-bounds false positive) against the visible PlaneWave allocation.
[[gnu::noinline]] std::shared_ptr<excitation::Excitation> x_polarised_plane_wave(Real lambda) {
    return std::make_shared<excitation::PlaneWave>(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
}

}  // namespace

TEST_CASE("Interface fields of a wide beam: impedance of the transmitted wave at the focus",
          "[validation][ag_absorption]") {
    // A wide beam is locally the central plane wave: at the focus the tangential fields on the
    // R1 side equal those of the transmitted wave, E_x / H_y = eta0 kz2 / eps2 (TM, p) and
    // E_y / H_x = -eta0 / kz2 (TE, s), kz2 = sqrt(eps2 - sin^2 theta) with Im kz2 <= 0. This fixes
    // the signs of r_s, r_p and of the reflected fields (the absorbed flux alone does not).
    const Complex eps = material::silver_500nm().eps_r;
    const Real eta0 =
        material::vacuum().wave_impedance(2.0 * constants::pi * constants::c0 / kLambda).real();
    for (const Real theta : {0.0, 45.0}) {
        const Real st = std::sin(theta * kDeg);
        Complex kz2 = std::sqrt(eps - st * st);
        if (kz2.imag() > 0.0)
            kz2 = -kz2;
        for (const auto pol : {excitation::Polarization::P, excitation::Polarization::S}) {
            CAPTURE(theta, pol == excitation::Polarization::P);
            const auto beam = make_test_beam(8e-6, theta, pol, 1e-6);
            const aa::InterfaceFields f(beam, 1.0, eps);
            const auto [E, H] = f.fields(0.0, 0.0);
            if (pol == excitation::Polarization::P) {
                const Complex z = E.x() / H.y();
                CHECK(std::abs(z / (eta0 * kz2 / eps) - 1.0) < 2e-3);
                CHECK(std::abs(E.y()) < 1e-6 * std::abs(E.x()));
            } else {
                const Complex z = E.y() / H.x();
                CHECK(std::abs(z / (-eta0 / kz2) - 1.0) < 2e-3);
                CHECK(std::abs(E.x()) < 1e-6 * std::abs(E.y()));
            }
            // Normal component of the reference currents vanishes; M = n x ... is tangential.
            const auto [J, M] = f.currents(0.0, 0.0);
            CHECK(J.z() == Complex(0.0));
            CHECK(M.z() == Complex(0.0));
            // E_tan = n x M with n = -z.
            const Vec3c nxm = aa::cross_real(-Vec3::UnitZ(), M);
            CHECK(std::abs(nxm.x() - E.x()) < 1e-14 * std::abs(E.norm()));
            CHECK(std::abs(nxm.y() - E.y()) < 1e-14 * std::abs(E.norm()));
        }
    }
}

TEST_CASE("Interface fields: absorbed flux over the plane equals (1 - R_beam - away) P_inc",
          "[validation][ag_absorption]") {
    const Complex eps = material::silver_500nm().eps_r;
    for (const auto pol : {excitation::Polarization::P, excitation::Polarization::S}) {
        for (const Real theta : {0.0, 45.0}) {
            CAPTURE(theta, pol == excitation::Polarization::P);
            // w0 = 0.8 um: the cone stays clear of grazing incidence at 45 deg (with w0 = 0.4 um
            // the outer components near k_z = 0 land far along x, and the window converges like 1 /
            // X).
            const Real w0 = 0.8e-6;
            const Real region = 5.0 * w0 / std::cos(theta * kDeg);
            const auto beam = make_test_beam(w0, theta, pol, region);
            const aa::InterfaceFields f(beam, 1.0, eps);
            const BeamReflectance br = beam_reflectance(beam, 1.0, eps);
            const Real a_ref = 1.0 - br.reflectance - br.away_share;
            const Real a_num =
                aa::reference_absorbed_flux(f, region / std::sqrt(2.0), kLambda / 8.0) /
                beam.power();
            CHECK_THAT(a_num, WithinRel(a_ref, 1e-5));
            CHECK(a_ref > 0.005);
        }
    }
}

TEST_CASE("Extinction integral equals the optical theorem for arbitrary currents",
          "[validation][ag_absorption]") {
    // For a plane wave, 1/2 Re oint (E_inc . J* + M . H_inc*) dS is algebraically the optical
    // theorem -(4 pi / k) Im[F(k_inc) . e0*] |E0|^2 / (2 eta) applied to the same currents, so
    // random currents (fixed seed) must agree to rounding (both use Dunavant degree 6).
    const Real lambda = 500e-9;
    const std::shared_ptr<excitation::Excitation> wave = x_polarised_plane_wave(lambda);
    SimulationConfig cfg;
    cfg.wavelength = lambda;
    cfg.object = material::silver_500nm();
    const Simulation sim(geometry::make_icosphere(0.3e-6, 1), wave, cfg);
    const Index n2 = sim.num_unknowns();
    std::mt19937 rng(20261011);
    std::normal_distribution<Real> g(0.0, 1.0);
    VectorXc x(n2);
    for (Index i = 0; i < n2; ++i) x(i) = Complex(g(rng), g(rng)) * (i < n2 / 2 ? 1e-3 : 0.3);
    const post::SurfaceSolution s{&sim.problem(), x};
    const Real p_ext = aa::extinction_power(s, *wave, 6);
    Vertices fwd(1, 3);
    fwd.row(0) = Vec3::UnitZ().transpose();
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(s, fwd, F);
    const Real omega = 2.0 * constants::pi * constants::c0 / lambda;
    const Real k = material::vacuum().wavenumber(omega).real();
    const Real eta = material::vacuum().wave_impedance(omega).real();
    const Real c_ext = -4.0 * constants::pi / k * F(0, 0).imag();
    CHECK_THAT(p_ext, WithinRel(c_ext / (2.0 * eta), 1e-10));
    CHECK(std::abs(p_ext) > 0.0);
}

TEST_CASE("Absorbed power parts, RWG interpolant and top-face metrics are consistent",
          "[validation][ag_absorption]") {
    Case c;
    c.material = "ag";
    c.L = 0.5e-6;
    c.waist = 0.125e-6;
    c.theta_deg = 30.0;
    c.pol = excitation::Polarization::S;
    const Geometry geo = make_geometry(c);
    const auto beam = make_beam(c, geo.r_max);
    const Simulation sim(geo.mesh, beam, make_config(c));
    const Complex eps = material::silver_500nm().eps_r;
    const aa::InterfaceFields f(*beam, 1.0, eps);
    const VectorXc interp = aa::interpolate_reference(*sim.problem().space, f, geo.spacing);
    const post::SurfaceSolution s{&sim.problem(), interp};

    // Triangle classification: the top face has 2 cells^2 triangles.
    Index top = 0;
    for (Index t = 0; t < geo.mesh.num_triangles(); ++t)
        top += aa::part_of(geo.mesh, t, geo.spacing) == aa::Part::Top ? 1 : 0;
    CHECK(top == geo.top_triangles);

    // Degree 2 is exact for the quadratic integrand: any higher degree agrees.
    const aa::AbsorbedParts p2 = aa::absorbed_power_parts(s, geo.spacing, 2);
    const aa::AbsorbedParts p6 = aa::absorbed_power_parts(s, geo.spacing, 6);
    CHECK_THAT(p6.top, WithinRel(p2.top, 1e-11));
    CHECK_THAT(p6.walls, WithinAbs(p2.walls, 1e-11 * std::abs(p2.top)));
    CHECK_THAT(p6.bottom, WithinAbs(p2.bottom, 1e-11 * std::abs(p2.top)));
    CHECK(p2.bottom == 0.0);  // only top-face edges carry interpolated coefficients

    // The "simulated" currents are the interpolant here: sim and int metrics coincide, and the
    // interpolant is close to the reference at 10 points per wavelength.
    const aa::TopMetrics m =
        aa::top_metrics(s, f, interp, geo.spacing, [](const Vec3&) { return true; }, 4);
    CHECK_THAT(m.p_sim, WithinRel(m.p_int, 1e-12));
    CHECK_THAT(m.p_int, WithinRel(p2.top, 1e-11));
    CHECK_THAT(m.err_j, WithinRel(m.err_j_int, 1e-12));
    CHECK(m.err_j < 0.12);  // RT0 interpolation is first order: ~0.09 at k h = 0.63
    CHECK(m.err_m < 0.12);
    CHECK(std::abs(m.c_j - 1.0) < 0.01);
    CHECK(std::abs(m.c_m - 1.0) < 0.01);
    CHECK(m.p_ref > 0.0);
    // The surface reactance of Ag dominates: |M| / |J| ~ |Z_s| ~ 0.3 eta0.
    CHECK(m.m_over_j > 80.0);
    CHECK(m.m_over_j < 160.0);
}
