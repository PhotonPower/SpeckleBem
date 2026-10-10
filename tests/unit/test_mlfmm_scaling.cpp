// Smoke test of the MLFMM scaling study's case code (WP22b1, benchmarks/mlfmm_scaling.cpp,
// tests/support/mlfmm_scaling_support.hpp) at tiny patch sizes: geometry (box parameters from
// rough_surface_box_params, the waist rule), estimates, the describe() parser and one complete
// Ag run (setup, timed matvecs, GMRES) through Simulation with compression "mlfmm". The Si
// estimate (order searches up to L ~ 100) and the end-to-end run are checked in optimised builds
// only, so that every case stays below ~20 s under the sanitizers.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlfmm_scaling_support.hpp"

using namespace mlfmm_scaling;

TEST_CASE("mlfmm_scaling: geometry follows the ADR 0006 box parameters", "[mlfmm_scaling]") {
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
    CHECK(uses_fine_band(si));
    const RoughBoxParams bs = box_params(si);
    CHECK(bs.box_mesh_size == kBoxMeshSize);  // explicit override of the automatic rule
    CHECK_THAT(bs.exterior_wavelength, Catch::Matchers::WithinRel(kLambda, 1e-12));

    Case ag = si;
    ag.material = "ag";
    const Geometry ga = make_geometry(ag);
    CHECK_FALSE(ga.fine_depth.has_value());  // Ag: no fine band
    CHECK(ga.depth == kMinBoxDepth);
    CHECK(ga.grading.levels == 1);  // 100 nm = 2 x 50 nm
    CHECK_FALSE(uses_fine_band(ag));
    ag.fine_band = true;
    CHECK(uses_fine_band(ag));
    ag.fine_band.reset();

    // Automatic rule capped at lambda1 / 5 (rough_surface_box_params) and the uncapped rule of
    // the record's "auto" series.
    ag.box_mesh_size.reset();
    CHECK(make_geometry(ag).grading.target_spacing <= kBoxMeshSize * (1.0 + 1e-9));
    CHECK(box_option(ag) == "auto");
    ag.uncapped_box = true;
    CHECK(box_option(ag) == "uncapped");
    CHECK(make_geometry(ag).grading.target_spacing > 0.0);
    ag.box_mesh_size = kBoxMeshSize;  // uncapped needs the automatic rule
    CHECK_THROWS_AS(make_geometry(ag), std::invalid_argument);

    Case bad = si;
    bad.material = "gold";
    CHECK_THROWS_AS(make_geometry(bad), std::invalid_argument);
    bad = si;
    bad.seed = 0;
    CHECK_THROWS_AS(make_geometry(bad), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: the waist rule w0 <= L/4 is enforced", "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.waist_factor = 3.0;  // w0 = L/3 > L/4
    CHECK_THROWS_AS(make_geometry(c), std::invalid_argument);
    c.allow_wide_beam = true;  // deliberate edge-effect study: allowed (logged)
    CHECK_NOTHROW(make_geometry(c));
    c.allow_wide_beam = false;
    c.waist_factor = 4.0;
    CHECK_NOTHROW(make_geometry(c));
}

TEST_CASE("mlfmm_scaling: the angular-spectrum beam covers the mesh", "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.beam = Beam::angular_spectrum;
    const Geometry geo = make_geometry(c);
    const auto beam = make_beam(c, geo);
    const Real rmax = geo.mesh.vertices().rowwise().norm().maxCoeff();
    CHECK(beam->controlled_radius() >= rmax);
    CHECK(beam_name(c) == "angular_spectrum");
    // E(focus) . x = 1 V/m, E along x at normal incidence.
    const Vec3c e0 = beam->electric_field(Vec3::Zero());
    CHECK_THAT(e0.x().real(), Catch::Matchers::WithinAbs(1.0, 1e-8));
    CHECK(std::abs(e0.y()) < 1e-8);
    c.beam = Beam::paraxial;
    CHECK(std::isinf(make_beam(c, geo)->controlled_radius()));
}

TEST_CASE("mlfmm_scaling: estimate respects the ADR 0008 leaf rule", "[mlfmm_scaling]") {
    std::vector<std::string> materials = {"ag"};
#ifdef NDEBUG
    materials.emplace_back("si");  // the Si order searches (L up to ~110) are slow unoptimised
    const Real patch = 0.8e-6;
#else
    const Real patch = 0.4e-6;  // sanitizer build: smaller patch, same octree depth
#endif
    for (const std::string& m : materials) {
        Case c;
        c.material = m;
        c.L = patch;
        const Geometry geo = make_geometry(c);
        const Estimate e = estimate(geo, c);
        CHECK(e.unknowns == 2 * geo.mesh.num_edges());
        CHECK(e.leaf_edge >= (1.0 - mlfmm::kMinBoxSizeTolerance) * 0.25 * kLambda);
        CHECK(e.max_support_radius / e.leaf_edge <= mlfmm::kLeafMaxSupportRatioD3 + 1e-12);
        CHECK(e.near_bytes > 0);
        CHECK(e.exact_budget >= e.near_bytes);
        // R1 always expands at the leaf; the far tables come with the patterns.
        CHECK(e.leaf_order[0] > 0);
        CHECK(e.expansion_levels[0] >= 1);
        CHECK(e.pattern_bytes > 0);
        CHECK(e.far_table_bytes > 0);
        CHECK(e.peak_setup_bytes > kPeakNearFactor * static_cast<Real>(e.near_bytes) +
                                       static_cast<Real>(e.pattern_bytes + e.far_table_bytes));
        if (m == "si") {
            CHECK(e.exact_pairs == 0);  // Si: x* / alpha = 13 um, no exact fallback estimate
            CHECK(e.leaf_order[1] > e.leaf_order[0]);  // the Si interior expands (k2 = 4.3 k1)
        } else {
            CHECK(e.leaf_order[1] == 0);  // Ag interior: no expansion (exact / truncated)
            CHECK(e.expansion_levels[1] == 0);
        }
    }
}

TEST_CASE("mlfmm_scaling: exact-pair estimate counts pairs within the decay reach",
          "[mlfmm_scaling]") {
    // A small rough patch; a strongly decaying k (short reach x*/alpha) gives fewer pairs than a
    // weaker one, and no decay gives none.
    Case c;
    c.material = "ag";
    c.L = 0.3e-6;
    const Geometry geo = make_geometry(c);
    const basis::RwgSpace space(geo.mesh);
    const Real k0 = 2.0 * constants::pi / kLambda;
    CHECK(estimate_exact_pairs(space, Complex(k0, 0.0), 3.0, 0) == 0);
    const Index tight = estimate_exact_pairs(space, Complex(k0, -200e6), 3.0, 0);  // 59 nm
    const Index wide = estimate_exact_pairs(space, Complex(k0, -80e6), 3.0, 0);    // 147 nm
    CHECK(tight >= space.size());  // every basis with itself (gap 0)
    CHECK(wide > tight);
    CHECK(estimate_exact_pairs(space, Complex(k0, -80e6), 3.0, wide) == 0);
}

TEST_CASE("mlfmm_scaling: number_after parses describe() and refuses missing markers",
          "[mlfmm_scaling]") {
    const std::string text = "octree built in 0.25 s\n  far: 3.5 s setup\n  near assembled in x";
    CHECK(number_after(text, "built in ") == 0.25);
    CHECK(number_after(text, "\n  far: ") == 3.5);
    CHECK_THROWS_AS(number_after(text, "missing marker "), std::runtime_error);
    CHECK_THROWS_AS(number_after(text, "assembled in "), std::runtime_error);  // no number
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
    const Result r = run(geo, c, estimate(geo, c));
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
    // The far-memory model (patterns + tables) against the operator (block check not
    // modelled: the model may only be larger).
    CHECK(static_cast<Real>(r.est.pattern_bytes + r.est.far_table_bytes) >=
          0.9 * static_cast<Real>(r.far_bytes));
    const std::string line = row(c, r);
    CHECK(line.rfind("ROW material=ag", 0) == 0);
    CHECK(line.find(" N2=" + std::to_string(r.est.unknowns) + " ") != std::string::npos);
    CHECK(line.find(" beam=paraxial ") != std::string::npos);
#endif
}
