// Near field of the Ag sphere in the xz-plane vs Mie, CI-sized companion of the docs/05 row
// "Near field Ag sphere, xz-plane" (WP24; the acceptance cases d = 1 um at lambda / 26.5 and the
// paper's d = 4 um mesh are tests/validation_large/test_ag_near_field_large.cpp). Sphere of
// diameter 1 um in vacuum, lambda = 500 nm, Ag eps_r = -9.794 - 0.313j, plane wave +z, E along x,
// icosphere n = 3 (2N = 3840, h = lambda / 6.6), Simulation with the dense operator (ICTF + left
// Jacobi per formulation::recommend) and GMRES to 1e-6; total field on the xz grid x, z in
// [-1.5 um, 1.5 um] at spacing lambda / 20 and on circles around the sphere against Mie
// (tests/support/ag_near_field_support.hpp: metric, zones, d_min = h / 2). The docs/05 criterion
// (max e < 0.1 lit = every exterior point outside the geometric shadow, including the x-axis and
// the forward region beside the sphere; < 1 in the geometric shadow |x| < a, z > 0) holds already
// on this coarse mesh (measured: lit 0.050, shadow 0.086, benchmarks/results/ag_near_field.md), so
// it is checked unchanged.
// Release only: the dense assembly of 2N = 3840 takes minutes in the unoptimised sanitizer build.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ag_near_field_support.hpp"

using namespace ag_near_field;

TEST_CASE("Ag near field: d = 1 um at icosphere n = 3 (dense) vs Mie in the xz-plane",
          "[validation][ag_near_field]") {
#ifdef NDEBUG
    ag_sphere_mlfmm::Case c;
    c.mesh = "ico";
    c.subdivisions = 3;
    c.diameter = 1e-6;
    c.tolerance = 1e-6;
    SimulationConfig cfg = ag_sphere_mlfmm::make_config(c);
    cfg.compression = "dense";
    cfg.gmres.verbose = false;
    Simulation sim(ag_sphere_mlfmm::make_mesh(c.mesh, c.diameter, c.subdivisions),
                   ag_sphere_mlfmm::make_wave(), cfg);
    sim.assemble();
    const solver::GmresResult g = sim.solve();
    REQUIRE(g.converged);
    const Check chk = check(sim.solution(), c.diameter, kLambda / 20.0);
    WARN("GMRES " << g.iterations << " iterations\n" << summary(chk));
    REQUIRE(chk.zones.lit.count > 0);
    REQUIRE(chk.zones.shadow.count > 0);
    REQUIRE(chk.zones.lit_x_axis.count > 0);
    REQUIRE(chk.zones.lit_forward.count > 0);
    CHECK(chk.zones.lit.max < kLitTolerance);
    CHECK(chk.zones.shadow.max < kShadowTolerance);
#else
    SKIP("dense assembly of 2N = 3840 is too slow in the unoptimised build");
#endif
}
