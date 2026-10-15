// Validation 7e of the open-surface PML route (WP-PML4c, ADR 0009 decision 7e; docs/05 "PML
// route", row 7e): the PML route against the validated uniform closing box (ADR 0006) for the
// same rough Si surface (seed 1, sigma = 50 nm, Lc = 500 nm), L = 4.5 um, w0 = 0.75 um = L / 6,
// h = lambda / 4 on both routes, normal incidence, E along x (pml_si_crosscheck::Case defaults,
// benchmarks/results/pml_si_crosscheck.md). The far fields must agree within the box's own
// sensitivity (pml_si_crosscheck::kTolerances: L2 of |F|^2 over the reflection hemisphere,
// eps_rr on the xz/yz cuts, reflectance R). Label `validation-large` (release only, RUN_SERIAL;
// manual: `ctest --preset win-release -L validation-large -R "7e"`); SKIPs when the PML route's
// matrix plus LU copy or the box route's MLFMM estimate does not fit into 60 % of the available
// memory. Run time (ctest, 2026-10-15, 12-core desktop shared with other dense runs): 69-100 min
// in total (box route 12-16 min, PML route 56-84 min, dominated by the dense assembly), peak
// 24.3 GB.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include "pml_si_crosscheck_support.hpp"

using namespace pml_si_crosscheck;

TEST_CASE("PML 7e: rough Si, PML route against the uniform box (L 4.5 um, w0 L/6, h lambda/4)",
          "[pml_7e]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Case c;  // the 7e configuration
    const geometry::HeightMap map = mesh_map(c);
    const geometry::OpenSurfaceMesh om = make_open_mesh(map, c);
    const BoxGeometry box = make_box(map, c);
    // The same surface on both routes: the patch heights are bitwise the box top face, and the
    // patch triangles are the box top-face triangles (oriented vertex triples).
    const HeightIdentity id = height_identity(box.top, om.mesh, box.mesh);
    INFO(identity_line(id, count_patch_triangles(om), box.top_triangles));
    REQUIRE(id.bitwise());
    const TriangleIdentity tri = patch_triangle_identity(box.top, om, box.mesh);
    INFO("patch triangles: open " << tri.open_patch << ", box top " << tri.box_top
                                  << ", vertex triples matched " << tri.matched);
    REQUIRE(tri.agree());
    const PmlEstimate pe = estimate_pml(c, om);
    const BoxEstimate be = estimate_box(c, box);
    const Real avail = system_memory::available_memory_bytes() / 1e9;
    if (!(0.6 * avail > std::max(pe.peak_gb, be.peak_gb())))
        SKIP("PML route peak " << pe.peak_gb << " GB (2N = " << pe.unknowns
                               << ", matrix + LU copy), box route " << be.peak_gb()
                               << " GB; available " << avail << " GB");

    RouteResult b = run_box(c, map);
    b.peak_estimate_gb = be.peak_gb();
    WARN(row_line(c, b, "box_7e"));
    REQUIRE(b.converged);
    CHECK(b.residual <= 1.2 * c.box_tol);

    RouteResult p = run_pml(c, map);
    p.peak_estimate_gb = pe.peak_gb;
    WARN(row_line(c, p, "pml_7e"));
    CHECK(p.residual < 1e-8);
    CHECK(p.dark_band <= c.dark_band_tolerance);
    CHECK(p.decay.j < 1e-3);
    CHECK(p.decay.m < 1e-3);
    CHECK(p.ff_check < 1e-4);
    CHECK(b.ff_check < 1e-4);

    // Validation 7e: agreement within the box's own sensitivity (record "Tolerance"; not loosened).
    const Metrics m = compare(p.ff, b.ff, p.reflectance(), b.reflectance());
    WARN(compare_line("pml_7e", "box_7e", m, kTolerances));
    CHECK(m.l2_refl <= kTolerances.l2_refl);
    CHECK(std::max(m.eps_xz, m.eps_yz) <= kTolerances.eps_rr);
    CHECK(std::abs(m.dR) <= kTolerances.dR);
#endif
}
