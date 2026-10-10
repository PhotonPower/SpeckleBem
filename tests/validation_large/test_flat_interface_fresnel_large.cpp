// Flat-interface limit vs Fresnel (WP22c; docs/01 "Flat interface limit", docs/05 row "Flat box,
// tapered beam, 0 deg and 45 deg | Fresnel r_p, r_s | rel on |r|^2 | < 1 %"; ctest label
// `validation-large`, release only, manual: `ctest --preset win-release -L validation-large -R
// "Flat interface"`). Flat patch (sigma = 0) of edge L closed by the ADR 0006 box
// (rough_surface_box_params: lambda_1 / 5 cells, Si fine band), rigorous angular-spectrum beam
// focused on the surface centre, Simulation with compression "mlfmm" (d0 = 3), formulation per
// formulation::recommend, full GMRES at tolerance 1e-4. The reflected power fraction R_sim (flux of
// the scattered far field through the reflection hemisphere over AngularSpectrumBeam::power()) is
// compared with the beam reflectance R_beam of an infinite interface (the sharp reference) and the
// plane-wave |r|^2 at the central angle (tests/support/fresnel_flat_support.hpp). Measured values,
// timings and memory: benchmarks/results/fresnel_flat.md. Each case SKIPs unless its estimated
// peak plus a 10 GB margin is available (shared machines).
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "fresnel_flat_support.hpp"

using namespace fresnel_flat;

namespace {

[[maybe_unused]] constexpr Real kMarginBytes = 10e9;

/// Runs the case unless the build is unoptimised or the estimated peak (setup model + full-GMRES
/// basis at expected_iterations) plus the margin is not available; checks the docs/05 criterion
/// against R_beam (and returns the result for further checks).
[[maybe_unused]] bool run_case(const Case& c, int expected_iterations, Result& out) {
#ifndef NDEBUG
    (void)c;
    (void)expected_iterations;
    (void)out;
    SKIP("validation-large cases run in optimised builds only");
    return false;
#else
    const Geometry geo = make_geometry(c);
    const auto beam = make_beam(c, geo.r_max);
    const Estimate e = estimate(geo, c, beam);
    const Real need = e.peak_setup_bytes + static_cast<Real>(expected_iterations) *
                                               static_cast<Real>(e.unknowns) *
                                               static_cast<Real>(sizeof(Complex));
    const Real avail = system_memory::available_memory_bytes();
    if (!(avail > need + kMarginBytes)) {
        SKIP(label(c) << ": estimated peak " << need / 1e9 << " GB + " << kMarginBytes / 1e9
                      << " GB margin, but only " << avail / 1e9 << " GB available");
        return false;
    }
    out = run(c, geo, beam, 0.0);
    WARN(label(c) << "\n" << summary(out));
    REQUIRE(out.converged);
    // Far-field quadrature and incident power are exact to far below the criterion.
    CHECK(std::abs(out.p_refl_grid / out.p_refl - 1.0) < 1e-6);
    CHECK(std::abs(out.ref.flux.square / out.ref.power - 1.0) < 1e-6);
    // docs/05: < 1 % on |r|^2 (here the reflected power fraction of the beam).
    CHECK(std::abs(out.error_beam()) < 0.01);
    return true;
#endif
}

Case make_case(const std::string& material, Real L, Real w0, Real theta_deg,
               excitation::Polarization pol) {
    Case c;
    c.material = material;
    c.L = L;
    c.waist = w0;
    c.theta_deg = theta_deg;
    c.pol = pol;
    return c;
}

}  // namespace

TEST_CASE("Flat interface: Ag, L = 4 um, w0 = 1 um, normal incidence vs Fresnel",
          "[validation-large][fresnel]") {
    Result r;
    if (!run_case(make_case("ag", 4e-6, 1e-6, 0.0, excitation::Polarization::P), 1300, r))
        return;
    CHECK(std::abs(r.error_fresnel()) < 0.01);
}

TEST_CASE("Flat interface: Si, L = 6 um, w0 = 1.5 um, normal incidence vs Fresnel",
          "[validation-large][fresnel]") {
    Result r;
    if (!run_case(make_case("si", 6e-6, 1.5e-6, 0.0, excitation::Polarization::P), 200, r))
        return;
    CHECK(std::abs(r.error_fresnel()) < 0.01);
}

TEST_CASE("Flat interface: Ag, L = 4 um, w0 = 0.8 um, 45 deg, p and s vs Fresnel",
          "[validation-large][fresnel]") {
    for (const auto pol : {excitation::Polarization::P, excitation::Polarization::S}) {
        Result r;
        if (!run_case(make_case("ag", 4e-6, 0.8e-6, 45.0, pol), 2000, r))
            return;
        CHECK(std::abs(r.error_fresnel()) < 0.01);
    }
}

TEST_CASE("Flat interface: Si, L = 6 um, w0 = 1.2 um, 45 deg, p and s vs Fresnel",
          "[validation-large][fresnel]") {
    for (const auto pol : {excitation::Polarization::P, excitation::Polarization::S}) {
        Result r;
        if (!run_case(make_case("si", 6e-6, 1.2e-6, 45.0, pol), 300, r))
            return;
        // R_beam differs from the plane-wave |r|^2 by -0.45 % (p) / +0.18 % (s) at this waist.
        CHECK(std::abs(r.error_fresnel()) < 0.01);
    }
}
