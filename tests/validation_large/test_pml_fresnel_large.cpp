// Validation 7a of the open-surface PML route (WP-PML3b, ADR 0009 decision 7a; docs/05 "PML
// route", row 7a): flat Si and Ag open surfaces terminated by the complex-coordinate-stretching
// PML under the rigorous angular-spectrum beam at 0 and 45 degrees, s and p, must reproduce the
// beam reflectance R_beam of the same beam on the infinite interface to < 1 % (relative, on
// |r|^2). The configuration is pml_fresnel::seven_a_case() (benchmarks/results/pml_validation.md):
// dense operator, recommended formulation, LU (one factorisation per material and angle, both
// polarisations as right-hand sides). Label `validation-large` (release only, RUN_SERIAL; manual:
// `ctest --preset win-release -L validation-large -R "PML 7a"`); each case SKIPs when the matrix
// plus its LU copy does not fit into 60 % of the available memory.
//
// Acceptance for Si at 45 deg is the fifth case, the mesh-converged result
// (seven_a_si45_fine_case(), h = lambda / 6, smaller geometry with a relaxed dark band). The Si
// 45 deg case at h = lambda / 4 is not mesh-converged (lambda / 5 is off by +5.9 % for p; its
// -0.76 % is fortuitous), so its error is reported with WARN only; its residual and edge-decay
// checks stay. Run times (record): 0 deg ~5 min, 45 deg 27-45 min per material, fine Si case
// ~35 min, peak 68 GB, on a 12-core desktop.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "pml_fresnel_support.hpp"

using namespace pml_fresnel;

#ifdef NDEBUG
namespace {

/// Whether a case's error against R_beam is an acceptance check or only reported.
enum class Acceptance { Check, ReportOnly };

/// SKIPs the running test case when the dense matrix plus its LU copy does not fit into 60 % of
/// the available memory.
void skip_unless_fits(const PmlCase& c) {
    const Size s = size_of(c);
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > s.peak_bytes()))
        SKIP("dense 2N = " << s.unknowns << " needs " << s.peak_bytes() / 1e9
                           << " GB (matrix + LU copy), available " << avail / 1e9 << " GB");
}

void run_seven_a(const std::string& material, Real theta_deg, Acceptance acceptance) {
    const PmlCase c = seven_a_case(material, theta_deg);
    skip_unless_fits(c);
    const RunResult r = run(c);
    REQUIRE(r.pols.size() == 2);
    for (const PolResult& p : r.pols) {
        WARN(row_line(c, r, p));
        CHECK(p.residual < 1e-8);
        CHECK(p.decay.j < 1e-3);
        CHECK(p.decay.m < 1e-3);
        if (acceptance == Acceptance::Check) {
            // Validation 7a: < 1 % relative on |r|^2 (docs/05, ADR 0009 decision 7a; not loosened).
            CHECK(std::abs(p.error()) < 0.01);
        } else {
            WARN("not mesh-converged, reported only: R_sim / R_beam - 1 = " << 100.0 * p.error()
                                                                            << " %");
        }
    }
}

}  // namespace
#endif

TEST_CASE("PML 7a: flat Si, 0 deg, s and p against the Fresnel beam reflectance", "[pml_7a]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    run_seven_a("si", 0.0, Acceptance::Check);
#endif
}

TEST_CASE("PML 7a: flat Si, 45 deg, h = lambda / 4 (not mesh-converged, reported only)",
          "[pml_7a]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // Not an acceptance case: Si at 45 deg is not mesh-converged at h = lambda / 4 (lambda / 5
    // gave +5.9 % for p, lambda / 6 +0.18 %), so the -0.76 % / -0.41 % measured here is a
    // fortuitous pass. Acceptance for Si 45 deg is the h = lambda / 6 case below.
    run_seven_a("si", 45.0, Acceptance::ReportOnly);
#endif
}

TEST_CASE("PML 7a: flat Ag, 0 deg, s and p against the Fresnel beam reflectance", "[pml_7a]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    run_seven_a("ag", 0.0, Acceptance::Check);
#endif
}

TEST_CASE("PML 7a: flat Ag, 45 deg, s and p against the Fresnel beam reflectance", "[pml_7a]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    run_seven_a("ag", 45.0, Acceptance::Check);
#endif
}

TEST_CASE("PML 7a: flat Si, 45 deg, s and p at h = lambda / 6 (mesh-converged)", "[pml_7a]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // seven_a_si45_fine_case(), the Si 45 deg acceptance case: the result at h = lambda / 4 is
    // not mesh-converged (lambda / 5 is off by +5.9 % for p); at lambda / 6 the power balance
    // closes. Smaller geometry with the dark-band tolerance relaxed to 1e-2, because the
    // seven_a_case() geometry at lambda / 6 (2N = 104 016, 173 GB dense) is infeasible densely;
    // the dark-band level is an ADR 0009 decision 4 configuration check, not the 7a tolerance
    // (record: at lambda / 6 the 5.9e-3 and 2.6e-2 geometries agree within 0.05 % / 0.08 %).
    const PmlCase c = seven_a_si45_fine_case();
    skip_unless_fits(c);
    const RunResult r = run(c);
    REQUIRE(r.pols.size() == 2);
    for (const PolResult& p : r.pols) {
        WARN(row_line(c, r, p));
        CHECK(p.residual < 1e-8);
        CHECK(p.decay.j < 1e-3);
        CHECK(p.decay.m < 1e-3);
        // Resolution indicator: the power entering the real part closes the balance (measured
        // 0.21 % / 0.17 %; 6.2 % / 3.3 % at lambda / 4).
        CHECK(std::abs(1.0 - p.r_sim() - p.a_real()) < 0.01);
        // Validation 7a: < 1 % relative on |r|^2 (docs/05, ADR 0009 decision 7a; not loosened).
        CHECK(std::abs(p.error()) < 0.01);
    }
#endif
}
