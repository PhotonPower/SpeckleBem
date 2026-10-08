// Large Mie validation cases of the dense SIE solver (WP11, ctest label `validation-large`,
// nightly / manual: `ctest --preset release -L validation-large`). Same sphere, illumination,
// formulation and metrics as tests/validation/test_mie_sphere_dense.cpp (radius 0.5 um,
// lambda = 500 nm, k = +z, E along x, PMCHWT + dense LU, eps_rr over 181 angles in the xz- and
// yz-planes, power balance of docs/05), on the finer icospheres of the Phase 2 DoD (docs/01,
// docs/05):
//  * dielectric n = 1.5 at icosphere n = 4 (5 120 triangles, 2N = 15 360, h ~ lambda / 13): the
//    "lambda / 10" DoD case on the finest feasible icosphere; eps_rr < 1 % and below n = 3;
//  * Ag at n = 4: eps_rr < 1 % and below n = 3; power balance |P_ext - P_sca - P_abs| / P_ext
//    < 1 % (2.3 % at n = 3, see test_mie_sphere_dense.cpp), C_ext and C_sca within 2 % of Mie,
//    and the error of the surface-current C_abs below its n = 3 value (measured -10 % at n = 4,
//    -140 % at n = 3: the absorbed power of Ag is ~1.6 % of P_ext, a small difference of large
//    reactive surface fields; the 2 % bound on C_abs is checked at n = 5);
//  * Ag at n = 5 (20 480 triangles, 2N = 61 440, h ~ lambda / 26): the "lambda / 20 PMCHWT + LU"
//    DoD case; eps_rr < 1 % and below the n = 4 values, full power balance including C_abs
//    within 2 %. Needs ~120 GB (Z plus the LU copy) and SKIPs below that.
// Every case SKIPs in unoptimised builds and when 2 * 16 (2N)^2 bytes exceed 60 % of the
// physical memory. Measured values and timings: benchmarks/results/mie_sphere_dense.md. No
// timing assertions (4-core machine without BLAS: an n = 4 case takes ~6 min, 7.7 GB).
#include <catch2/catch_test_macros.hpp>

#include "mie_dense_test_support.hpp"

using namespace mie_dense_test;

namespace {

/// SKIPs (returns false) in unoptimised builds and when the dense solve does not fit.
bool large_case_runs(int subdivisions) {
#ifndef NDEBUG
    (void)subdivisions;
    SKIP("validation-large cases run in optimised builds only");
    return false;
#else
    const Index unknowns = icosphere_unknowns(subdivisions);
    const Real need = dense_solve_bytes(unknowns);
    const Real phys = physical_memory_bytes();
    if (exceeds_memory(unknowns)) {
        SKIP("icosphere n = " << subdivisions << " (2N = " << unknowns << ") needs ~" << need / 1e9
                              << " GB for Z and its LU copy, more than 60 % of the " << phys / 1e9
                              << " GB of physical memory");
        return false;
    }
    WARN("icosphere n = " << subdivisions << ": 2N = " << unknowns << ", estimated " << need / 1e9
                          << " GB of " << phys / 1e9 << " GB physical memory");
    return true;
#endif
}

}  // namespace

TEST_CASE("Mie validation large: dielectric sphere n = 1.5 at icosphere n = 4 (dense PMCHWT + LU)",
          "[validation-large][mie_dense]") {
    if (!large_case_runs(4))
        return;
    const material::Material n15 = lossless_n15();
    const DenseMieResult r3 = solve_mie_dense(n15, 3);
    report("n = 1.5", 3, r3);
    const DenseMieResult r4 = solve_mie_dense(n15, 4, true);
    report("n = 1.5", 4, r4);
    report_power("n = 1.5", 4, r4.power, mie_cross_sections(n15));
    CHECK(r4.info.residual < 1e-10);
    CHECK(r4.eps_xz < 0.01);
    CHECK(r4.eps_yz < 0.01);
    CHECK(r4.eps_xz < r3.eps_xz);
    CHECK(r4.eps_yz < r3.eps_yz);
    CHECK(r4.power.defect() < 0.01);
}

TEST_CASE("Mie validation large: Ag sphere at icosphere n = 4 (dense PMCHWT + LU)",
          "[validation-large][mie_dense]") {
    if (!large_case_runs(4))
        return;
    const material::Material ag = material::silver_500nm();
    const MieCrossSections mie = mie_cross_sections(ag);
    const DenseMieResult r3 = solve_mie_dense(ag, 3, true);
    report("Ag", 3, r3);
    report_power("Ag", 3, r3.power, mie);
    const DenseMieResult r4 = solve_mie_dense(ag, 4, true);
    report("Ag", 4, r4);
    report_power("Ag", 4, r4.power, mie);
    CHECK(r4.info.residual < 1e-10);
    CHECK(r4.eps_xz < 0.01);
    CHECK(r4.eps_yz < 0.01);
    CHECK(r4.eps_xz < r3.eps_xz);
    CHECK(r4.eps_yz < r3.eps_yz);
    const PowerBalance& p = r4.power;
    CHECK(p.defect() < 0.01);
    CHECK(rel_error(p.c_ext, mie.c_ext) < 0.02);
    CHECK(rel_error(p.c_sca, mie.c_sca) < 0.02);
    CHECK(rel_error(p.c_ext - p.c_sca, mie.c_abs) < 0.02);
    CHECK(rel_error(p.c_abs, mie.c_abs) < rel_error(r3.power.c_abs, mie.c_abs));
}

TEST_CASE("Mie validation large: Ag sphere at icosphere n = 5 for lambda / 20 (dense PMCHWT + LU)",
          "[validation-large][mie_dense]") {
    if (!large_case_runs(5))
        return;
    // n = 4 values measured by the n = 4 case above (benchmarks/results/mie_sphere_dense.md;
    // deterministic, the assembly is bitwise thread-count independent), hard-coded so that this
    // case does not repeat the n = 4 solve.
    constexpr Real kEpsXzN4 = 7.84e-4;
    constexpr Real kEpsYzN4 = 5.69e-4;
    const material::Material ag = material::silver_500nm();
    const MieCrossSections mie = mie_cross_sections(ag);
    const DenseMieResult r5 = solve_mie_dense(ag, 5, true);
    report("Ag", 5, r5);
    report_power("Ag", 5, r5.power, mie);
    CHECK(r5.info.residual < 1e-10);
    CHECK(r5.eps_xz < 0.01);
    CHECK(r5.eps_yz < 0.01);
    CHECK(r5.eps_xz < kEpsXzN4);
    CHECK(r5.eps_yz < kEpsYzN4);
    const PowerBalance& p = r5.power;
    CHECK(p.defect() < 0.01);
    CHECK(rel_error(p.c_ext, mie.c_ext) < 0.02);
    CHECK(rel_error(p.c_sca, mie.c_sca) < 0.02);
    CHECK(rel_error(p.c_abs, mie.c_abs) < 0.02);
}
