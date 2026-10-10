// Smoke test of the MLFMM scaling study's case code (WP22b1, benchmarks/mlfmm_scaling.cpp,
// tests/support/mlfmm_scaling_support.hpp) at tiny patch sizes: geometry, estimates and one
// complete Ag run (setup, timed matvecs, GMRES) through Simulation with compression "mlfmm".
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>

#include "mlfmm_scaling_support.hpp"

using namespace mlfmm_scaling;

TEST_CASE("mlfmm_scaling: geometry follows the WP-V1 box recommendations", "[mlfmm_scaling]") {
    Case si;
    si.material = "si";
    si.L = 0.6e-6;
    const Geometry gs = make_geometry(si);
    CHECK(gs.mesh.signed_volume() > 0.0);
    REQUIRE(gs.fine_depth.has_value());  // Si: fine band 3 delta + 3 sigma
    CHECK_THAT(*gs.fine_depth,
               Catch::Matchers::WithinRel(
                   default_box_fine_depth(material::silicon_500nm(), kLambda, kSigma), 1e-12));
    CHECK_THAT(gs.depth, Catch::Matchers::WithinRel(5.6455e-6, 1e-3));
    CHECK(gs.grading.target_spacing == kBoxMeshSize);
    CHECK(gs.top_triangles == 2 * 12 * 12);

    Case ag = si;
    ag.material = "ag";
    const Geometry ga = make_geometry(ag);
    CHECK_FALSE(ga.fine_depth.has_value());  // Ag: no fine band
    CHECK(ga.depth == kMinBoxDepth);
    CHECK(ga.grading.levels == 1);  // 100 nm = 2 x 50 nm

    ag.box_mesh_size.reset();  // automatic rule: min(depth / 2, L / 8, 10 h)
    CHECK(make_geometry(ag).grading.target_spacing < kBoxMeshSize);

    Case bad = si;
    bad.material = "gold";
    CHECK_THROWS_AS(make_geometry(bad), std::invalid_argument);
    bad = si;
    bad.seed = 0;
    CHECK_THROWS_AS(make_geometry(bad), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: estimate respects the ADR 0008 leaf rule", "[mlfmm_scaling]") {
    for (const std::string m : {"si", "ag"}) {
        Case c;
        c.material = m;
        c.L = 0.8e-6;
        const Geometry geo = make_geometry(c);
        const Estimate e = estimate(geo, c);
        CHECK(e.unknowns == 2 * geo.mesh.num_edges());
        CHECK(e.leaf_edge >= (1.0 - mlfmm::kMinBoxSizeTolerance) * 0.25 * kLambda);
        CHECK(e.max_support_radius / e.leaf_edge <= mlfmm::kLeafMaxSupportRatioD3 + 1e-12);
        CHECK(e.near_bytes > 0);
        CHECK(e.exact_budget >= e.near_bytes);
        CHECK(e.peak_setup_bytes > static_cast<Real>(e.near_bytes));
        if (m == "si")
            CHECK(e.exact_pairs == 0);  // Si: x* / alpha = 1.3 um, no exact fallback estimate
    }
}

TEST_CASE("mlfmm_scaling: exact-pair estimate counts pairs within the decay reach",
          "[mlfmm_scaling]") {
    // Two separated icosphere-free meshes are awkward; use a rough patch and compare the count
    // for a strongly decaying k with a larger reach: more pairs, and zero for no decay.
    Case c;
    c.material = "ag";
    c.L = 0.6e-6;
    const Geometry geo = make_geometry(c);
    const basis::RwgSpace space(geo.mesh);
    const Real k0 = 2.0 * constants::pi / kLambda;
    CHECK(estimate_exact_pairs(space, Complex(k0, 0.0), 3.0, 0) == 0);
    const Index tight = estimate_exact_pairs(space, Complex(k0, -100e6), 3.0, 0);
    const Index wide = estimate_exact_pairs(space, Complex(k0, -30e6), 3.0, 0);
    CHECK(tight >= space.size());  // every basis with itself (gap 0)
    CHECK(wide > tight);
    CHECK(estimate_exact_pairs(space, Complex(k0, -30e6), 3.0, wide) == 0);
}

TEST_CASE("mlfmm_scaling: one tiny Ag case runs end to end", "[mlfmm_scaling]") {
#ifndef NDEBUG
    SKIP("unoptimised build: the end-to-end MLFMM run is checked in release");
#else
    Case c;
    c.material = "ag";
    c.L = 0.6e-6;
    c.matvecs = 3;
    c.tolerance = 1e-1;  // smoke test: the study itself uses 1e-3
    c.max_iter = 200;
    const Geometry geo = make_geometry(c);
    const Result r = run(geo, c);
    CHECK(r.formulation == "ICTF");
    CHECK(r.jacobi);
    CHECK(r.converged);
    CHECK(r.true_residual < 0.2);
    CHECK(r.est.levels >= 3);
    CHECK(r.matvec_s > 0.0);
    CHECK(r.near_matvec_s > 0.0);
    CHECK(r.far_matvec_s > 0.0);
    CHECK(r.near_s > 0.0);
    CHECK(r.far_s > 0.0);
    CHECK(r.assembly_s >= r.near_s);
    CHECK(r.near_bytes > 0);
    CHECK(r.far_bytes > 0);
    CHECK(r.krylov_bytes > 0.0);
    CHECK(r.peak_rss >= r.peak_setup_rss);
    const std::string line = row(c, r);
    CHECK(line.rfind("ROW material=ag", 0) == 0);
    CHECK(line.find(" N2=" + std::to_string(r.est.unknowns) + " ") != std::string::npos);
#endif
}
