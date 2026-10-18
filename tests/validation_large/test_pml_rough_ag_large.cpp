// Validations 7c and 7d of the open-surface PML route on rough surfaces (WP-PML4b, ADR 0009
// decision 7; docs/05 "PML route", rows 7c and 7d): the far field of a rough spot in a polished
// surface with absorbing surroundings (pml_rough_ag::u_case(): sigma = 50 nm, Lc = 500 nm, seed 1,
// L = 3.75 um, h = lambda / 4, w0 = lambda, normal incidence, PML defaults) must change by < 0.1 %
// for W_pml x 1.5 and for sigma_max x 1.5 (7c, rough Ag), and by < 0.5 % for L x 1.2 at fixed
// waist (7d), in the docs/05 box-check metric (L2 of |F|^2 over the reflection hemisphere,
// solid-angle weights; benchmarks/results/pml_rough_ag.md). The tolerances are the docs/05 values,
// not loosened.
//
// 7d acceptance is the rough Si control (si_u_case() / si_patch_case(): the same geometry, height
// map, PML and beam with Si). On rough Ag the L x 1.2 change fails 0.5 % (2.57 %); the termination
// is not the cause (7c, flat control, Si control). The change comes from SPP currents in the gap:
// the far-field window truncates them (1.61 % on the same currents) and the added annulus changes
// them (2.42 % at the same absolute window, which still includes the window acting on the changed
// currents; the two are not separated). ADR 0010 / WP-PML6 (row 7g) decides whether a physical L
// dependence remains (record, section 7d). The Ag L x 1.2 change is therefore reported with WARN
// only, next to the documented value, plus a regression guard on that value (a reported value,
// not an acceptance); its residual, dark-band and edge-decay checks stay (lead decision 2026-10-18
// (ADR 0010); like the Si 45 deg lambda / 4 case of 7a).
//
// Label `validation-large` (release only, RUN_SERIAL; manual:
// `ctest --preset win-release -L validation-large -R "PML 7[cd]"`); each case SKIPs when its
// largest dense matrix does not fit into 60 % of the available memory (LU in place: the peak is one
// matrix, 8.3 GB for U, 13.6 GB for W_pml x 1.5, 12.1 GB for L x 1.2). Each case solves its U
// itself; run times per solve are in the record.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "pml_rough_ag_support.hpp"

using namespace pml_rough_ag;
using Catch::Approx;

#ifdef NDEBUG
namespace {

/// Whether a 7d comparison is an acceptance check or only reported.
enum class Acceptance { Check, ReportOnly };

/// L2 metric of Ag L x 1.2 vs U documented in benchmarks/results/pml_rough_ag.md (run l12 vs u).
constexpr Real kDocumentedAgL2 = 2.5675e-2;

void skip_unless_fits(const RoughCase& c) {
    const Size s = size_of(c);
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > s.dense_bytes))
        SKIP("dense 2N = " << s.unknowns << " needs " << s.dense_bytes / 1e9
                           << " GB (LU in place), available " << avail / 1e9 << " GB");
}

/// Solves a case and checks the solve itself (residual, dark band, decay at the truncation).
RunResult checked_run(const RoughCase& c) {
    const RunResult r = run(c);
    WARN(row_line(c, r));
    CHECK(r.residual < 1e-8);
    CHECK(r.est.dark_band <= c.dark_band_tolerance);
    CHECK(r.decay.j < 1e-3);
    CHECK(r.decay.m < 1e-3);
    return r;
}

/// 7d: base U against L x 1.2 at fixed waist, each with Simulation's default far-field window.
void seven_d(const RoughCase& base, const RoughCase& big, Acceptance acceptance) {
    skip_unless_fits(big);
    const RunResult u = checked_run(base);
    const RunResult r = checked_run(big);
    const Metrics m = compare(r.far.front(), u.far.front());
    WARN(compare_line(r.far.front(), u.far.front(), m));
    if (acceptance == Acceptance::Check) {
        // Validation 7d: < 0.5 % (docs/05, ADR 0009 decision 7d; not loosened).
        CHECK(m.l2 < 5e-3);
    } else {
        WARN("7d on rough " << base.material
                            << " reported only (SPP currents in the gap, record section 7d; ADR "
                               "0010 / WP-PML6): l2 = "
                            << 100.0 * m.l2 << " % (documented " << 100.0 * kDocumentedAgL2
                            << " %, tolerance 0.5 %)");
        // Regression guard on the documented value (not an acceptance): a change by more than
        // 10 % means the solve or the far field changed and the record must be revisited.
        CHECK(m.l2 == Approx(kDocumentedAgL2).epsilon(0.1));
    }
}

}  // namespace
#endif

TEST_CASE("PML 7c: rough Ag far field changes by < 0.1 % for W_pml x1.5 and sigma_max x1.5",
          "[pml_7c]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    skip_unless_fits(pml_width_case());
    const RunResult u = checked_run(u_case());
    for (const RoughCase& c : {pml_width_case(), sigma_case()}) {
        const RunResult r = checked_run(c);
        const Metrics m = compare(r.far.front(), u.far.front());
        WARN(compare_line(r.far.front(), u.far.front(), m));
        // Validation 7c: < 0.1 % (docs/05, ADR 0009 decision 7c; not loosened).
        CHECK(m.l2 < 1e-3);
    }
#endif
}

TEST_CASE("PML 7d: rough Si far field changes by < 0.5 % for L x1.2 at fixed waist", "[pml_7d]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    seven_d(si_u_case(), si_patch_case(), Acceptance::Check);
#endif
}

TEST_CASE("PML 7d: rough Ag far field for L x1.2 at fixed waist (SPP L-dependence, reported only)",
          "[pml_7d]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    seven_d(u_case(), patch_case(), Acceptance::ReportOnly);
#endif
}
