// Validation 7b of the open-surface PML route (WP-PML4a, ADR 0009 decision 7b; docs/05 "PML
// route", row 7b): an SPP launched on a flat Ag open surface must be reflected by the PML start
// with |r_spp| < 1e-2. Configuration pml_spp::seven_b_case() (benchmarks/results/pml_validation.md,
// "7b SPP absorption"): the PML defaults (sigma_max 4, order 2, W_pml = 2 lambda), a Gaussian
// groove under the rigorous beam at normal incidence, grooved minus flat solution, J_x and M_y
// fitted along the x-axis with an outgoing and a reflected cylindrical SPP wave.
//
// The second case checks the measurement itself (the geometric check of the record): a weak PML
// (sigma_max 0.25, W_pml = lambda) reflects measurably, and moving its start by h = 100 nm leaves
// the M_y reflection coefficient referred to the PML start unchanged, whereas referred to a fixed
// plane it would rotate by 2 Re(k_spp) h = 2.65 rad.
//
// Label `validation-large` (release only, RUN_SERIAL; manual:
// `ctest --preset win-release -L validation-large -R "PML 7b"`); each case SKIPs when its dense
// matrix does not fit into 60 % of the available memory (the LU is in place with LAPACK and with
// Eigen, pml_spp::InPlaceLu, so the peak is the one matrix). Run times (record): 7b
// case 2N = 14 800, two dense assemblies; the check case 2N = 9 440 and 10 416, one each.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "pml_spp_support.hpp"

using namespace pml_spp;

#ifdef NDEBUG
namespace {

void skip_unless_fits(const SppCase& c) {
    const Size s = size_of(c);
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > s.dense_bytes))
        SKIP("dense 2N = " << s.unknowns << " needs " << s.dense_bytes / 1e9
                           << " GB (LU in place), available " << avail / 1e9 << " GB");
}

}  // namespace
#endif

TEST_CASE("PML 7b: SPP on flat Ag reflected at the PML start below 1e-2", "[pml_7b]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const SppCase c = seven_b_case();
    skip_unless_fits(c);
    const RunResult r = run(c);
    WARN(row_line(c, r, "groove"));
    WARN(row_line(c, r, "local"));
    CHECK(r.residual < 1e-8);
    for (const QuantityFit& f : r.fits) {
        if (f.quantity == "divJ")
            continue;  // reported only (piecewise-constant divergence, noisier)
        INFO(f.source << " " << f.quantity << " side " << f.side << ": |r| " << std::abs(f.fit.r())
                      << ", fit residual " << f.fit.residual << ", k/k_spp " << f.fit.k / r.k_spp);
        // The fitted model describes the samples (measured 1.5e-4 J_x, 4.4e-5 M_y) and the
        // wavenumber is the SPP's.
        CHECK(f.fit.residual < 1e-3);
        CHECK(std::abs(f.fit.k.real() / r.k_spp.real() - 1.0) < 0.03);
        // Every fit of either launch is below the 7b limit as well.
        CHECK(std::abs(f.fit.r()) < 1e-2);
    }
    // The two launches measure the same reflection in the clean observable M_y (record: 3.40e-5
    // and 3.51e-5, phases 0.02 rad apart; |difference| ~ 1e-6): a deterministic property of the
    // PML, not fit noise. J_x of the local source is not SPP-clean (its |out(M_y) / out(J_x)| is
    // 30 Ohm against the SPP's 127 Ohm, the groove's 111 Ohm) and is not compared.
    for (const int side : {1, -1}) {
        const Complex m_groove = find_fit(r, "groove", "My", side).r();
        const Complex m_local = find_fit(r, "local", "My", side).r();
        INFO("side " << side << ": r(M_y) groove " << m_groove << ", local " << m_local
                     << "; |out My/Jx| groove " << out_impedance(r.fits, "groove", side)
                     << " Ohm, local " << out_impedance(r.fits, "local", side) << " Ohm, SPP "
                     << spp_impedance(r.k0, r.k_spp) << " Ohm");
        CHECK(std::abs(m_groove - m_local) < 2e-4);
    }
    // Validation 7b: |r_spp| < 1e-2 (docs/05, ADR 0009 decision 7b; not loosened).
    CHECK(r.r_spp < 1e-2);
#endif
}

TEST_CASE("PML 7b check: a weak PML's reflection is measured and referred to the PML start",
          "[pml_7b]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    SppCase c = seven_b_case();
    c.launch = Launch::Source;
    c.sigma_max = 0.25;
    c.pml_width = kLambda;
    SppCase moved = c;
    moved.pml_start = c.pml_start + c.mesh_size;
    skip_unless_fits(moved);
    const RunResult r0 = run(c);
    const RunResult r1 = run(moved);
    WARN(row_line(c, r0, "local"));
    WARN(row_line(moved, r1, "local"));
    const Complex m0 = find_fit(r0, "local", "My", 1).r();
    const Complex m1 = find_fit(r1, "local", "My", 1).r();
    INFO("M_y: r(a) = " << m0 << ", r(a + h) = " << m1);
    // A reflection well above the default PML's (1e-4 level) and the measurement floor.
    CHECK(std::abs(m0) > 5e-3);
    CHECK(std::abs(m1) > 5e-3);
    // Referred to the PML start the phase is unchanged (measured 1e-3 rad); referred to a fixed
    // plane it would rotate by 2 Re(k_spp) h.
    const Real fixed_plane_rotation = 2.0 * r0.k_spp.real() * c.mesh_size;
    REQUIRE(fixed_plane_rotation > 2.0);
    CHECK(std::abs(std::arg(m1 / m0)) < 0.2);
    CHECK(std::abs(std::abs(m1) / std::abs(m0) - 1.0) < 0.5);
#endif
}
