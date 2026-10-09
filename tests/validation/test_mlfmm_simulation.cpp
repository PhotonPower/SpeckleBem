// GMRES through Simulation with compression "mlfmm" against the same solve on the dense operator
// (WP20b). The MLFMM matvec differs from the dense one by ~1e-4 (d0 = 3; tests/validation_large/
// test_mlfmm_vs_dense_large.cpp); the solutions agree to about that times the conditioning.
// Default kernel options. Measured 2026-10-10 (win-release): currents 4.6e-5, RCS 1.3e-6,
// GMRES 117 (dense) / 122 (MLFMM) iterations. The rough-box case (2N = 7680, ~1 min) is in
// tests/validation_large/test_mlfmm_vs_dense_large.cpp.
#include "specklebem/geometry/sphere.hpp"

#include <catch2/catch_test_macros.hpp>

#include "mlfmm_simulation_support.hpp"

using namespace mlfmm_simulation_test;

TEST_CASE("mlfmm simulation: Si sphere, ICTF, default MLFMM parameters", "[validation][mlfmm]") {
#ifndef NDEBUG
    SKIP("dense GMRES of 2N = 3840: release only");
#else
    // R = 0.5 um, icosphere n = 3 (2N = 3840); the defaults (100 elements per leaf, lambda / 4
    // floor) give 3 levels with lambda / 2 leaves.
    SimulationConfig cfg;
    cfg.object = material::silicon_500nm();
    const Comparison c =
        compare(geometry::make_icosphere(kLambda, 3), cfg, mlfmm::MlfmmParams{}, "Si sphere");
    CHECK(c.currents <= 1e-3);
    CHECK(c.rcs <= 1e-3);
#endif
}
