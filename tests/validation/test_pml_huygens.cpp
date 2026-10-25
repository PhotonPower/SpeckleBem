// Validation 7f of the PML route's Huygens-plane far field at CI size (WP-PML6, ADR 0010; docs/05
// "PML route", row 7f): on flat Ag and flat Si in the 7a zones with the thin PML of the 7h CI case
// (pml_huygens::flat_ci_case(): patch 1.5 um, taper 0.5 um, gap 0.5 um, W_pml = lambda, sigma_max
// 2, w0 = lambda, h = lambda / 4, 2N = 7 632; dense LU) there is no bound wave and the reflected
// beam does not reach the plane window (from 2.5 w0), so the far field of the Huygens plane
// (defaults z_h = lambda, spacing lambda / 4, the current window, transition Delta = lambda) must
// agree with the windowed-current far field of the same solution: L2 metric (ADR 0006 box-check,
// pml_rough_ag::compare) <= 1e-3 and |dR| <= 1e-4; the sensitivities z_h x 1.5 and spacing / 1.5
// must meet the same criterion against the currents (WARN lines; record
// benchmarks/results/pml_huygens.md, section 7f). Criteria of docs/05, not loosened. The smaller
// 7h CI geometry (2N = 4 592: window from 1.5 w0) does not qualify for 7f (record). The 7a-size
// cases (2N = 11 440, ~7 min and 4.2 GB each) are in
// tests/validation_large/test_pml_huygens_large.cpp. Release only (the stretched dense assembly
// takes minutes under the sanitizers); 4-8 min per material on 12 cores.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "pml_huygens_support.hpp"

namespace {

[[maybe_unused]] void check_7f_ci(const std::string& material) {
    using namespace pml_huygens;
    const RoughCase c = flat_ci_case(material);
    const Solved s = solve(c, Solver::Dense);
    REQUIRE(s.unknowns == 7632);
    REQUIRE(s.converged);
    CHECK(s.residual < 1e-8);
    const SevenF r = evaluate_7f(s);
    for (const std::string& line : r.lines) WARN(line);
    // Defaults: plane at lambda below the (flat) surface, spacing lambda / 4, current window.
    CHECK(std::abs(r.plane.plane.z + kLambda) < 1e-15);
    CHECK(r.plane.plane.normal_z == -1.0);
    CHECK(r.plane.plane.window.inner_half == s.half[1]);
    CHECK(r.plane.plane.window.outer_half == s.half[2]);
    // Validation 7f (docs/05; not loosened).
    CHECK(r.m.l2 <= 1e-3);
    CHECK(std::abs(r.m.dR) <= 1e-4);
    CHECK(r.mz.l2 <= 1e-3);
    CHECK(std::abs(r.mz.dR) <= 1e-4);
    CHECK(r.ms.l2 <= 1e-3);
    CHECK(std::abs(r.ms.dR) <= 1e-4);
}

}  // namespace

TEST_CASE("pml_huygens 7f: flat Ag, Huygens plane vs windowed currents (CI size)",
          "[pml_huygens_7f]") {
#ifndef NDEBUG
    SKIP("dense stretched assembly of 2N = 7 632: release only");
#else
    check_7f_ci("ag");
#endif
}

TEST_CASE("pml_huygens 7f: flat Si, Huygens plane vs windowed currents (CI size)",
          "[pml_huygens_7f]") {
#ifndef NDEBUG
    SKIP("dense stretched assembly of 2N = 7 632: release only");
#else
    check_7f_ci("si");
#endif
}
