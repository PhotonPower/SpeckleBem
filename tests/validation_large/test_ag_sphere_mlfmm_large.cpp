// Ag sphere with MLFMM vs Mie (WP22a; ctest label `validation-large`, release only, manual:
// `ctest --preset win-release -L validation-large -R "Ag sphere MLFMM"`). Sphere in vacuum,
// lambda = 500 nm, Ag eps_r = -9.794 - 0.313j, plane wave +z, E along x; Simulation with
// compression "mlfmm" (d0 = 3, automatic leaf rule and exact-part budget), formulation per
// formulation::recommend (ICTF + left Jacobi), full GMRES; eps_rr of docs/05 in the xz- and
// yz-planes over 181 angles (tests/support/ag_sphere_mlfmm_support.hpp). Measured values, timings
// and memory: benchmarks/results/ag_sphere_4um_mlfmm.md.
//  * d = 1 um, icosphere n = 5 (2N = 61 440, h = lambda / 26.5): the "Ag sphere d = 1 um, lambda /
//    20" case of docs/05 (< 1 %), whose dense LU does not fit into 128 GB; GMRES tol 1e-5 so that
//    the solver error stays below the discretisation error, and eps_rr below the dense LU values at
//    icosphere n = 4 (lambda / 13: 0.0784 % / 0.0569 %, benchmarks/results/mie_sphere_dense.md).
//    About 3 min, peak ~12 GB.
//  * d = 4 um, octahedron-based sphere n = 7 (131 072 triangles, 2N = 393 216, the paper's mesh):
//    the Phase 4 DoD (docs/01, docs/05: eps_rr <= 0.5 % in both planes) at the paper's GMRES
//    tolerance 1e-3. 11-16 min, peak ~34 GB: SKIPs unless the estimated peak plus a 10 GB margin is
//    available (shared machines).
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ag_sphere_mlfmm_support.hpp"

using namespace ag_sphere_mlfmm;

namespace {

/// Peak working set of an MLFMM case estimated from the near-field estimate: measured on the WP22a
/// ramp-up (benchmarks/results/ag_sphere_4um_mlfmm.md: 2N = 393 216, 33.8 GB = 2.77 x the 12.2 GB
/// near field including a 2.8 GB Krylov basis), peak RSS <= kPeakPerNear x near field + the
/// full-GMRES basis.
[[maybe_unused]] constexpr Real kPeakPerNear = 3.0;
[[maybe_unused]] constexpr Real kMarginBytes = 10e9;

/// SKIPs (returns false) in unoptimised builds and when the case does not fit into the memory
/// available now.
[[maybe_unused]] bool case_runs(const Case& c, int expected_iterations) {
#ifndef NDEBUG
    (void)c;
    (void)expected_iterations;
    SKIP("validation-large cases run in optimised builds only");
    return false;
#else
    const geometry::TriangleMesh mesh = make_mesh(c.mesh, c.diameter, c.subdivisions);
    const Estimate e = estimate(mesh, c);
    const Real krylov = static_cast<Real>(expected_iterations) * static_cast<Real>(e.unknowns) *
                        static_cast<Real>(sizeof(Complex));
    const Real need = kPeakPerNear * static_cast<Real>(e.near_bytes) + krylov;
    const Real avail = system_memory::available_memory_bytes();
    if (!(avail > need + kMarginBytes)) {
        SKIP(label(c) << ": estimated peak " << need / 1e9 << " GB + " << kMarginBytes / 1e9
                      << " GB margin, but only " << avail / 1e9 << " GB available");
        return false;
    }
    WARN(label(c) << ": " << summary(e) << "; estimated peak " << need / 1e9 << " GB, "
                  << avail / 1e9 << " GB available");
    return true;
#endif
}

}  // namespace

TEST_CASE("Ag sphere MLFMM: d = 1 um at icosphere n = 5 (lambda / 26.5) vs Mie",
          "[validation-large][ag_sphere_mlfmm]") {
    Case c;
    c.mesh = "ico";
    c.subdivisions = 5;
    c.diameter = 1e-6;
    c.tolerance = 1e-5;
    if (!case_runs(c, 600))
        return;
    const Result r = run(c);
    WARN(label(c) << "\n" << summary(r));
    REQUIRE(r.converged);
    CHECK(r.eps_xz < 0.01);
    CHECK(r.eps_yz < 0.01);
    // Below the dense LU values at icosphere n = 4 (lambda / 13).
    CHECK(r.eps_xz < 7.84e-4);
    CHECK(r.eps_yz < 5.69e-4);
}

TEST_CASE("Ag sphere MLFMM: d = 4 um at the paper's 393 216 unknowns vs Mie (Phase 4 DoD)",
          "[validation-large][ag_sphere_mlfmm]") {
    Case c;  // defaults: octa n = 7, d = 4 um, d0 = 3, tol 1e-3
    if (!case_runs(c, 600))
        return;
    const Result r = run(c);
    WARN(label(c) << "\n" << summary(r));
    REQUIRE(r.converged);
    CHECK(r.eps_xz <= 0.005);
    CHECK(r.eps_yz <= 0.005);
}
