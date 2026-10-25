// Validations 7f (7a geometry) and 7g of the PML route's Huygens-plane far field (WP-PML6, ADR
// 0010; docs/05 "PML route", rows 7f and 7g; record benchmarks/results/pml_huygens.md):
//  * 7f on flat Ag and flat Si in the 7a geometry (pml_huygens::flat_case(): patch 1.5 um, taper /
//    gap 0.5 um, W_pml = 2 lambda, w0 = lambda, h = lambda / 4, 2N = 11 440; dense LU, ~7 min and
//    4.2 GB each): Huygens plane (defaults, z_h x 1.5, spacing / 1.5) vs the windowed currents of
//    the same solution, L2 metric <= 1e-3 and |dR| <= 1e-4 (checked; the CI-size 7f cases,
//    2N = 7 632, are tests/validation/test_pml_huygens.cpp). SKIPs when the matrix plus its LU
//    copy does not fit into 60 % of the available memory;
//  * 7g, rough Ag (the WP-PML4b U case, pml_rough_ag::u_case(), 2N = 22 816; its L x 1.2
//    partner and a wide-gap partner, pml_huygens::wide_gap_case()), fast path (compression
//    "mlfmm", d0 = 3, GMRES 1e-7): 7d with the Huygens far field (L x 1.2 at fixed waist, each
//    run with its own default plane window) < 0.5 %; z_h x 1.5 and the plane window x (1 +- 0.3)
//    (U's window scaled on the wide-gap run) < 0.1 % each (ADR 0006 box-check L2 metric);
//  * 7g, rough Si 7e repeated with the Huygens far field: PML route (fast path, Huygens plane at
//    its defaults) against the uniform box (pml_si_crosscheck::Case defaults) within the box's
//    own sensitivity (pml_si_crosscheck::kTolerances);
//  * 6b (WP-PML6b, ADR 0010 decision 2026-10-25 item 4), the gap >= 6 lambda rule on the
//    wide-gap pair of the second roughness seed (pml_huygens::wide_gap3_case(2) and
//    wide_gap3_patch_case(2): 3 / 2.625 um gap, pml_start 5.375 um, 2N = 62 016 each; fast
//    path; one case, because both solves are needed for 7d, ~1-2 h): in the cone theta <= 30
//    deg the window sensitivities (flank x 0.7 / x 1.3 of (taper_end, taper_end + 4 lambda),
//    windowed currents and Huygens plane, U and L x 1.2) <= 1e-3, and 7d (currents with the
//    default and the 4 lambda windows, Huygens with 4 lambda) >= 10 x the sensitivity of the
//    same source (pml_huygens::six_b_gates; acceptance of the WP, not loosened).
// Row 7g is NOT met (WP-PML6, record section "7g"): on rough Ag 7d with the Huygens plane is 2.78
// %, z_h x 1.5 1.05 %, window x 0.7 / x 1.3 1.75 / 0.66 %; on rough Si l2 and eps_rr pass but R
// moves by -5.3e-4 (tolerance 4.1e-5). The plane, bounded by pml_start, misses the radiation at
// grazing angles (theta 75-90 deg: -67 % of the band power at U), more so the farther it is from
// the surface, on Ag and Si alike. The far-field source of the PML route therefore stays the
// windowed currents (ADR 0010 decision rule); the 7g values are reported with WARN next to the
// documented values, plus a regression guard on each (a reported value, not an acceptance; as the
// 7d Ag case of test_pml_rough_ag_large.cpp); the Si l2 / eps_rr parts and the plane-only
// invariants stay checked. The tolerances of docs/05 are not loosened; the row is closed as
// "not met - physical" (lead decision 2026-10-25, ADR 0010; confirmed by the 6b case below).
// Label `validation-large` (release only, RUN_SERIAL; manual: `ctest
// --preset win-release -L validation-large -R "7f|7g|6b"`). Each case SKIPs when its estimated
// peak does not fit into 60 % of the available memory.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pml_huygens_support.hpp"
#include "pml_si_crosscheck_support.hpp"

#ifdef NDEBUG
using Catch::Approx;

namespace {

/// Documented 7g values (benchmarks/results/pml_huygens.md; L2 metric / dR), guarded to 10 %:
/// rough Ag L x 1.2 with the Huygens plane (own default windows), z_h x 1.5 on U, the plane window
/// x 0.7 / x 1.3 (wide-gap run), and dR of rough Si (Huygens plane vs the uniform box).
using specklebem::Real;
constexpr Real kDocumented7dAg = 2.7763e-2;
constexpr Real kDocumentedZh15Ag = 1.0536e-2;
constexpr Real kDocumentedW07Ag = 1.7488e-2;
constexpr Real kDocumentedW13Ag = 6.6421e-3;
constexpr Real kDocumented7eDR = -5.332e-4;

/// Validation 7f in the 7a geometry (docs/05; not loosened).
void check_7f(const std::string& material) {
    using namespace pml_huygens;
    const RoughCase c = flat_case(material);
    const pml_rough_ag::Size size = pml_rough_ag::size_of(c);
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > 2.0 * size.dense_bytes))
        SKIP("dense 2N = " << size.unknowns << " needs " << 2.0 * size.dense_bytes / 1e9
                           << " GB (matrix + LU copy), available " << avail / 1e9 << " GB");
    const Solved s = solve(c, Solver::Dense);
    REQUIRE(s.unknowns == 11440);
    REQUIRE(s.converged);
    CHECK(s.residual < 1e-8);
    const SevenF r = evaluate_7f(s);
    for (const std::string& line : r.lines) WARN(line);
    CHECK(std::abs(r.plane.plane.z + kLambda) < 1e-15);
    CHECK(r.plane.plane.normal_z == -1.0);
    CHECK(r.plane.plane.window.outer_half == s.half[2]);
    CHECK(r.m.l2 <= 1e-3);
    CHECK(std::abs(r.m.dR) <= 1e-4);
    CHECK(r.mz.l2 <= 1e-3);
    CHECK(std::abs(r.mz.dR) <= 1e-4);
    CHECK(r.ms.l2 <= 1e-3);
    CHECK(std::abs(r.ms.dR) <= 1e-4);
}

}  // namespace
#endif

TEST_CASE("PML 7f: flat Ag, Huygens plane vs windowed currents (7a geometry)", "[pml_7f]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    check_7f("ag");
#endif
}

TEST_CASE("PML 7f: flat Si, Huygens plane vs windowed currents (7a geometry)", "[pml_7f]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    check_7f("si");
#endif
}

TEST_CASE("PML 7g: Huygens far field on rough Ag (L x 1.2, z_h x 1.5, window x (1 +- 0.3))",
          "[pml_7g]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    using namespace pml_huygens;
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > 16e9))
        SKIP("the fast path of 2N = 34 000 needs ~10 GB; available " << avail / 1e9 << " GB");
    // U's window (taper_end, pml_start) as the zones compute it (2.375 / 2.875 um up to round-off;
    // the same window as benchmarks/pml_huygens.cpp, which produced the documented values).
    const pml_rough_ag::Size u_size = pml_rough_ag::size_of(pml_rough_ag::u_case());
    const post::CurrentWindow u_window{u_size.half[1], u_size.half[2]};
    CHECK(u_window.inner_half == Approx(2.375e-6).epsilon(1e-12));
    CHECK(u_window.outer_half == Approx(2.875e-6).epsilon(1e-12));
    FarField uh, uz, lh, w07, w10, w13;
    {
        const Solved s = solve(pml_rough_ag::u_case(), Solver::Fast);
        REQUIRE(s.converged);
        CHECK(s.sim->far_field_window()->inner_half == Approx(u_window.inner_half).epsilon(1e-12));
        CHECK(s.sim->far_field_window()->outer_half == Approx(u_window.outer_half).epsilon(1e-12));
        const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
        uh = radiate(pf, s, "u.H");
        WARN(row_line(s, currents_far_field(s, "u.C"), uh, pf));
        post::HuygensOptions z15;
        z15.z_h = 1.5 * kLambda;
        uz = radiate(plane_field(s, z15), s, "u.Hz15");
    }
    {
        const Solved s = solve(pml_rough_ag::patch_case(), Solver::Fast);
        REQUIRE(s.converged);
        const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
        lh = radiate(pf, s, "l12.H");
        WARN(row_line(s, currents_far_field(s, "l12.C"), lh, pf));
    }
    {
        const Solved s = solve(wide_gap_case(), Solver::Fast);
        REQUIRE(s.converged);
        post::HuygensOptions w;
        w.window = scaled(u_window, 1.3);
        const post::HuygensPlaneField pf = plane_field(s, w);
        w13 = radiate(pf, s, "ugap.Hw13");
        w10 = radiate(post::huygens_reweight(pf, u_window), s, "ugap.Hw10");
        w07 = radiate(post::huygens_reweight(pf, scaled(u_window, 0.7)), s, "ugap.Hw07");
        WARN(row_line(s, currents_far_field(s, "ugap.C", u_window), w10, pf));
    }
    const pml_rough_ag::Metrics m7d = pml_rough_ag::compare(lh, uh);
    const pml_rough_ag::Metrics mz = pml_rough_ag::compare(uz, uh);
    const pml_rough_ag::Metrics m07 = pml_rough_ag::compare(w07, w10);
    const pml_rough_ag::Metrics m13 = pml_rough_ag::compare(w13, w10);
    WARN(pml_rough_ag::compare_line(lh, uh, m7d));
    WARN(pml_rough_ag::compare_line(uz, uh, mz));
    WARN(pml_rough_ag::compare_line(w07, w10, m07));
    WARN(pml_rough_ag::compare_line(w13, w10, m13));
    // Validation 7g (docs/05: 7d < 0.5 %, sensitivities < 0.1 %; not loosened) is not met: the
    // values are reported, with a regression guard on each documented value (a change by more
    // than 10 % means the solve or the plane evaluation changed and the record must be revisited).
    WARN("7g on rough Ag reported only (record section 7g; ADR 0010): 7d "
         << 100.0 * m7d.l2 << " %, z_h x1.5 " << 100.0 * mz.l2 << " %, window x0.7 "
         << 100.0 * m07.l2 << " %, x1.3 " << 100.0 * m13.l2 << " % (tolerances 0.5 / 0.1 %)");
    CHECK(m7d.l2 == Approx(kDocumented7dAg).epsilon(0.1));
    CHECK(mz.l2 == Approx(kDocumentedZh15Ag).epsilon(0.1));
    CHECK(m07.l2 == Approx(kDocumentedW07Ag).epsilon(0.1));
    CHECK(m13.l2 == Approx(kDocumentedW13Ag).epsilon(0.1));
    // The plane window of the wide-gap run at U's window reproduces U (the gap is inert at a fixed
    // window, as for the currents in 7c): a plane-only invariant, checked.
    CHECK(pml_rough_ag::compare(w10, uh).l2 < 1e-3);
#endif
}

TEST_CASE("PML 6b: wide-gap rule on rough Ag, second seed (window x (1 +- 0.3), 7d, theta <= 30)",
          "[pml_6b]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    using namespace pml_huygens;
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > 10e9))
        SKIP("the fast path of 2N = 62 016 needs ~5 GB; available " << avail / 1e9 << " GB");
    constexpr std::uint64_t kSeed = kSeeds[1];  // the second surface (new evidence)
    const RoughCase uc = wide_gap3_case(kSeed);
    const RoughCase lc = wide_gap3_patch_case(kSeed);
    const Real u_taper_end = pml_rough_ag::size_of(uc).half[1];
    CHECK(u_taper_end == Approx(2.375e-6).epsilon(1e-12));
    // Every window of the study fits before the solves (an invalid one throws here).
    for (const RoughCase* c : {&uc, &lc}) {
        const pml_rough_ag::Size size = pml_rough_ag::size_of(*c);
        CHECK(size.half[2] == Approx(5.375e-6).epsilon(1e-12));
        REQUIRE_NOTHROW(six_b_windows({size.half[1], size.half[2]}, u_taper_end));
    }
    std::map<std::string, FarField> far;
    for (const RoughCase* c : {&uc, &lc}) {
        const Solved s = solve(*c, Solver::Fast);
        REQUIRE(s.unknowns == 62016);
        REQUIRE(s.converged);
        CHECK(s.residual < 1e-6);
        const SixBFields f = six_b_far_fields(s, u_taper_end);
        for (const std::string& line : f.lines) WARN(line);
        for (const FarField& x : f.far) far[x.tag] = x;
    }
    std::vector<Comparison> cmp = six_b_case_comparisons(uc.tag);
    for (const Comparison& k : six_b_case_comparisons(lc.tag)) cmp.push_back(k);
    for (const Comparison& k : six_b_pair_comparisons(uc.tag, lc.tag)) cmp.push_back(k);
    for (const Comparison& k : cmp)
        WARN(cone_line(k.label, far.at(k.test), far.at(k.ref),
                       cone_compare(far.at(k.test), far.at(k.ref))));
    // Acceptance (ADR 0010 decision 2026-10-25 item 4; not loosened): in the cone theta <= 30 deg
    // the window sensitivities of both sources <= 1e-3, and 7d >= 10 x them.
    const SixBGates g = six_b_gates(far, uc.tag, lc.tag);
    WARN(gate_line(kSeed, g));
    CHECK(g.sens_currents <= kSixBSensitivity);
    CHECK(g.sens_huygens <= kSixBSensitivity);
    CHECK(g.d7_currents >= kSixBRatio * g.sens_currents);
    CHECK(g.d7_currents_d4 >= kSixBRatio * g.sens_currents);
    CHECK(g.d7_huygens >= kSixBRatio * g.sens_huygens);
#endif
}

TEST_CASE("PML 7g: rough Si 7e with the Huygens far field (PML fast path vs uniform box)",
          "[pml_7g]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    using namespace pml_si_crosscheck;
    Case c;  // the 7e configuration
    c.pml_solver = PmlSolver::Fast;
    c.pml_tol = 1e-7;
    c.pml_far_field = FarFieldSource::Huygens;
    const geometry::HeightMap map = mesh_map(c);
    const BoxGeometry box = make_box(map, c);
    const BoxEstimate be = estimate_box(c, box);
    const Real avail = system_memory::available_memory_bytes() / 1e9;
    if (!(0.6 * avail > std::max(be.peak_gb(), 15.0)))
        SKIP("box route peak " << be.peak_gb() << " GB, PML fast path ~10 GB; available " << avail
                               << " GB");
    RouteResult b = run_box(c, map);
    b.peak_estimate_gb = be.peak_gb();
    WARN(row_line(c, b, "box_7e"));
    REQUIRE(b.converged);
    RouteResult p = run_pml(c, map);
    WARN(row_line(c, p, "pml_7e_huygens"));
    REQUIRE(p.converged);
    REQUIRE(p.ff_currents.has_value());
    CHECK(p.ff_check < 1e-4);
    const Metrics m = compare(p.ff, b.ff, p.reflectance(), b.reflectance());
    WARN(compare_line("pml_7e_huygens", "box_7e", m, kTolerances));
    const Metrics mc = compare(*p.ff_currents, b.ff, p.p_refl_currents / p.p_inc, b.reflectance());
    WARN(compare_line("pml_7e_currents", "box_7e", mc, kTolerances));
    // 7e repeated with the Huygens far field (row 7g; tolerances of 7e, not loosened): l2_refl and
    // eps_rr are met and checked; dR is not met (the plane misses the grazing power, record
    // section 7g): reported, with a regression guard on the documented value.
    CHECK(m.l2_refl <= kTolerances.l2_refl);
    CHECK(std::max(m.eps_xz, m.eps_yz) <= kTolerances.eps_rr);
    WARN("7e with the Huygens plane: dR = " << m.dR << " (documented " << kDocumented7eDR
                                            << ", tolerance " << kTolerances.dR << "; not met)");
    CHECK(m.dR == Approx(kDocumented7eDR).epsilon(0.1));
    // The windowed currents of the same solve reproduce the WP-PML4c 7e result.
    CHECK(mc.l2_refl <= kTolerances.l2_refl);
    CHECK(std::abs(mc.dR) <= kTolerances.dR);
#endif
}
