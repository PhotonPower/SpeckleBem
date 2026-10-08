// Mie validation of the dense SIE solver (WP11, Phase 2 acceptance tests of docs/01 and
// docs/05): sphere of radius 0.5 um (d = 1 um) in vacuum, lambda = 500 nm, plane wave k = +z,
// E along x (docs/06), PMCHWT + solver::solve_direct (mie_dense_test_support.hpp).
//  * bistatic-RCS error eps_rr of docs/05 (E_ref = sqrt(sigma_Mie), E_sie = sqrt(sigma),
//    normalised by max E_ref) over 181 angles theta = 0..180 deg in the xz-plane (phi = 0) and in
//    the yz-plane (phi = pi / 2, bistatic_rcs plane normal -x), icosphere n = 2 and n = 3:
//    monotone decrease n = 2 -> 3 and eps_rr(n = 3) < 1 % for the dielectric sphere n = 1.5 and
//    for Ag (material::silver_500nm()). n = 3 is h ~ lambda / 6.6; the finer DoD meshes
//    (lambda / 10, lambda / 20) are the validation-large cases (tests/validation_large).
//  * power balance of docs/05 (P_ext from the optical theorem, P_sca from the far field over the
//    unit sphere, P_abs from the surface currents; mie_dense_test_support.hpp):
//    - Si (silicon_500nm), n = 3: |P_ext - P_sca - P_abs| / P_ext < 1 % and C_ext, C_sca, C_abs
//      each within 2 % of Mie;
//    - n = 1.5 (lossless), n = 3: |P_ext - P_sca - P_abs| / P_ext < 1 % (P_abs ~ 0);
//    - Ag, n = 2 -> 3: C_ext and C_sca within 2 % of Mie, the far-field absorption C_ext - C_sca
//      within 2 % of the Mie C_abs, and the balance defect decreasing with refinement. The
//      surface-current P_abs of Ag is only ~1.6 % of P_ext (weak loss, Im eps_r = -0.313), so it
//      is a small difference of large reactive surface fields and converges slowly: measured
//      defect 50 % (n = 2), 2.3 % (n = 3), 0.16 % (n = 4). The < 1 % balance of Ag is therefore
//      checked at n = 4 (validation-large); the defect at n = 3 is independent of the
//      quadrature degrees (checked with far/near/singular degrees up to 8/12/14) and is the RWG
//      discretisation of the surface fields, not an assembly error.
//  * reciprocity (docs/05) is not covered here: it needs a second excitation and solve and is a
//    Phase 4 acceptance row (rough surfaces).
// Unoptimised sanitizer build: the dense LU of 2N = 3840 takes ~25 min (Eigen without BLAS), so
// only the n = 2 solves run there (as in test_assembler_mie.cpp); the n = 3 parts are
// release-only (release: n = 3 assembly ~2 s, LU ~7 s on 4 cores).
#include <catch2/catch_test_macros.hpp>

#include "mie_dense_test_support.hpp"

using namespace mie_dense_test;

TEST_CASE("Mie validation: dielectric sphere n = 1.5 (dense PMCHWT + LU)",
          "[validation][mie_dense]") {
    // Measured: eps_rr xz / yz n = 2: 0.0089 / 0.0089 (sanitizer and release builds alike),
    // n = 3: 0.0027 / 0.0028; balance defect at n = 3: 0.2 %.
    const material::Material n15 = lossless_n15();
    const DenseMieResult r2 = solve_mie_dense(n15, 2);
    report("n = 1.5", 2, r2);
    CHECK(r2.info.residual < 1e-10);
    CHECK(r2.eps_xz < 0.02);
    CHECK(r2.eps_yz < 0.02);
#ifdef NDEBUG
    const DenseMieResult r3 = solve_mie_dense(n15, 3, true);
    report("n = 1.5", 3, r3);
    report_power("n = 1.5", 3, r3.power, mie_cross_sections(n15));
    CHECK(r3.info.residual < 1e-10);
    CHECK(r3.eps_xz < r2.eps_xz);
    CHECK(r3.eps_yz < r2.eps_yz);
    CHECK(r3.eps_xz < 0.01);
    CHECK(r3.eps_yz < 0.01);
    CHECK(r3.power.defect() < 0.01);
#else
    WARN("icosphere n = 3 runs in optimised builds only (LU of 2N = 3840 too slow under ASan)");
#endif
}

TEST_CASE("Mie validation: Ag sphere (dense PMCHWT + LU)", "[validation][mie_dense]") {
    // Measured: eps_rr xz / yz n = 2: 0.020 / 0.015, n = 3: 0.0035 / 0.0025; power at n = 3:
    // C_ext -0.89 %, C_sca -0.91 %, C_ext - C_sca vs Mie C_abs +0.3 %, surface C_abs -140 %
    // (balance defect 2.3 %, see the file comment); n = 2 defect 50 %.
    const material::Material ag = material::silver_500nm();
    const MieCrossSections mie = mie_cross_sections(ag);
    const DenseMieResult r2 = solve_mie_dense(ag, 2, true);
    report("Ag", 2, r2);
    report_power("Ag", 2, r2.power, mie);
    CHECK(r2.info.residual < 1e-10);
    CHECK(r2.eps_xz < 0.05);
    CHECK(r2.eps_yz < 0.05);
#ifdef NDEBUG
    const DenseMieResult r3 = solve_mie_dense(ag, 3, true);
    report("Ag", 3, r3);
    report_power("Ag", 3, r3.power, mie);
    CHECK(r3.info.residual < 1e-10);
    CHECK(r3.eps_xz < r2.eps_xz);
    CHECK(r3.eps_yz < r2.eps_yz);
    CHECK(r3.eps_xz < 0.01);
    CHECK(r3.eps_yz < 0.01);
    const PowerBalance& p = r3.power;
    CHECK(rel_error(p.c_ext, mie.c_ext) < 0.02);
    CHECK(rel_error(p.c_sca, mie.c_sca) < 0.02);
    CHECK(rel_error(p.c_ext - p.c_sca, mie.c_abs) < 0.02);
    CHECK(p.defect() < r2.power.defect());
    // Regression bound at n = 3 (measured 2.3 %); the docs/05 < 1 % criterion for Ag is checked
    // at n = 4 in the validation-large suite.
    CHECK(p.defect() < 0.03);
#else
    WARN("icosphere n = 3 runs in optimised builds only (LU of 2N = 3840 too slow under ASan)");
#endif
}

TEST_CASE("Mie validation: power balance of the Si sphere (dense PMCHWT + LU)",
          "[validation][mie_dense]") {
#ifdef NDEBUG
    // Measured at n = 3: defect 0.4 %, C_ext -1.0 %, C_sca -1.5 %, C_abs -1.2 % vs Mie.
    const material::Material si = material::silicon_500nm();
    const MieCrossSections mie = mie_cross_sections(si);
    const DenseMieResult r = solve_mie_dense(si, 3, true);
    report("Si", 3, r);
    report_power("Si", 3, r.power, mie);
    CHECK(r.info.residual < 1e-10);
    CHECK(r.power.defect() < 0.01);
    CHECK(rel_error(r.power.c_ext, mie.c_ext) < 0.02);
    CHECK(rel_error(r.power.c_sca, mie.c_sca) < 0.02);
    CHECK(rel_error(r.power.c_abs, mie.c_abs) < 0.02);
#else
    SKIP("dense LU of 2N = 3840 is too slow in the unoptimised build");
#endif
}
