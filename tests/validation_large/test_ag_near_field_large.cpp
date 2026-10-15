// Near field of the Ag sphere in the xz-plane vs Mie (WP24; docs/05 row "Near field Ag sphere,
// xz-plane": relative error per point < 0.1 lit, < 1 shadow; docs/01 Phase 5 DoD). ctest label
// `validation-large`, release only, manual:
// `ctest --preset win-release -L validation-large -R "Ag near field"`.
// The WP22a case (tests/support/ag_sphere_mlfmm_support.hpp: sphere in vacuum, lambda = 500 nm,
// Ag eps_r = -9.794 - 0.313j, plane wave +z, E along x, Simulation with MLFMM d0 = 3, ICTF + left
// Jacobi, full GMRES) is solved and the total field post::total_field is compared with the Mie
// total field on the xz grid x, z in [-1.5 d, 1.5 d] and on circles around the sphere
// (tests/support/ag_near_field_support.hpp: metric e = |E_sim - E_Mie| / |E_Mie|; outside the
// near-surface band |r| - a < d_min = kDMinPerEdge x mean edge, chosen from the measured
// near-surface profile, "shadow" is the geometric shadow |x| < a, z > 0 and "lit" every other
// exterior point, including z <= 0, the x-axis and the forward region beside the sphere).
// Criterion exactly as docs/05: max e < 0.1 lit, < 1 shadow. Measured values, the profile and
// timings: benchmarks/results/ag_near_field.md.
//  * d = 1 um, icosphere n = 5 (2N = 61 440, h = lambda / 26.5), GMRES tol 1e-5 as the WP22a
//    test; about 4 min, peak ~8 GB (max e 0.005 lit and shadow).
//  * d = 4 um, octahedron-based sphere n = 7 (2N = 393 216, the paper's mesh, Phase 4 DoD case)
//    at GMRES tol 1e-4 (722 iterations); ~35 min, peak ~26.5 GB (max e 0.012 lit, 0.006 shadow).
//    The docs/05 criterion concerns the BEM field, the GMRES tolerance is a solver setting: at
//    the paper's tolerance 1e-3 the criterion is met as well (max e 0.088 lit, 0.022 shadow), but
//    only with a 1.14x margin that is set by the solver tolerance, not by the discretisation.
//    SKIPs unless the estimated peak plus a 10 GB margin is available (shared machines).
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "ag_near_field_support.hpp"

using namespace ag_near_field;
using ag_sphere_mlfmm::Case;

namespace {

/// Memory guard of test_ag_sphere_mlfmm_large.cpp (peak <= 3 x the near-field estimate + the
/// full-GMRES basis; the field evaluation adds < 0.1 GB).
[[maybe_unused]] constexpr Real kPeakPerNear = 3.0;
[[maybe_unused]] constexpr Real kMarginBytes = 10e9;

[[maybe_unused]] bool case_runs(const Case& c, int expected_iterations) {
#ifndef NDEBUG
    (void)c;
    (void)expected_iterations;
    SKIP("validation-large cases run in optimised builds only");
    return false;
#else
    const geometry::TriangleMesh mesh =
        ag_sphere_mlfmm::make_mesh(c.mesh, c.diameter, c.subdivisions);
    const ag_sphere_mlfmm::Estimate e = ag_sphere_mlfmm::estimate(mesh, c);
    const Real krylov = static_cast<Real>(expected_iterations) * static_cast<Real>(e.unknowns) *
                        static_cast<Real>(sizeof(Complex));
    const Real need = kPeakPerNear * static_cast<Real>(e.near_bytes) + krylov;
    const Real avail = system_memory::available_memory_bytes();
    if (!(avail > need + kMarginBytes)) {
        SKIP(ag_sphere_mlfmm::label(c)
             << ": estimated peak " << need / 1e9 << " GB + " << kMarginBytes / 1e9
             << " GB margin, but only " << avail / 1e9 << " GB available");
        return false;
    }
    WARN(ag_sphere_mlfmm::label(c) << ": " << ag_sphere_mlfmm::summary(e) << "; estimated peak "
                                   << need / 1e9 << " GB, " << avail / 1e9 << " GB available");
    return true;
#endif
}

/// Solves the case and checks the docs/05 criterion on the grid of spacing max_spacing.
[[maybe_unused]] void run_and_check(const Case& c, Real max_spacing) {
    Check chk;
    // run() calls the callback only after a converged solve; otherwise chk stays default and only
    // the solver summary is printed.
    const ag_sphere_mlfmm::Result r = ag_sphere_mlfmm::run(
        c, [&](const Simulation& sim) { chk = check(sim.solution(), c.diameter, max_spacing); });
    const std::string near =
        r.converged ? summary(chk) : std::string("near field not evaluated (GMRES not converged)");
    WARN(ag_sphere_mlfmm::label(c) << "\n" << ag_sphere_mlfmm::summary(r) << "\n" << near);
    REQUIRE(r.converged);
    REQUIRE(chk.zones.lit.count > 0);
    REQUIRE(chk.zones.shadow.count > 0);
    REQUIRE(chk.zones.lit_x_axis.count > 0);
    REQUIRE(chk.zones.lit_forward.count > 0);
    CHECK(chk.zones.lit.max < kLitTolerance);
    CHECK(chk.zones.shadow.max < kShadowTolerance);
}

}  // namespace

TEST_CASE("Ag near field: d = 1 um at icosphere n = 5 (lambda / 26.5) vs Mie in the xz-plane",
          "[validation-large][ag_near_field]") {
    Case c;
    c.mesh = "ico";
    c.subdivisions = 5;
    c.diameter = 1e-6;
    c.tolerance = 1e-5;
    if (!case_runs(c, 600))
        return;
    run_and_check(c, kLambda / 20.0);
}

TEST_CASE("Ag near field: d = 4 um at the paper's 393 216 unknowns vs Mie in the xz-plane",
          "[validation-large][ag_near_field]") {
    Case c;              // defaults: octa n = 7, d = 4 um, d0 = 3
    c.tolerance = 1e-4;  // 722 iterations (the paper's 1e-3: 442, see the file comment)
    if (!case_runs(c, 800))
        return;
    run_and_check(c, kLambda / 10.0);
}
