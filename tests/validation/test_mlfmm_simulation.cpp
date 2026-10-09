// GMRES through Simulation with compression "mlfmm" against the same solve on the dense operator
// (WP20b). The MLFMM matvec differs from the dense one by ~1e-4 (d0 = 3; tests/validation_large/
// test_mlfmm_vs_dense_large.cpp); the solutions agree to about that times the conditioning.
// lambda = 500 nm, vacuum exterior, default kernel options, GMRES tolerance 1e-6 for both.
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <utility>

using namespace specklebem;

namespace {

constexpr Real kLambda = 500e-9;

struct Comparison {
    Real currents = 0;  ///< |x_mlfmm - x_dense| / |x_dense|
    Real rcs = 0;       ///< max |sigma_mlfmm - sigma_dense| / max sigma_dense (xz plane)
};

/// Solves the same configuration with "dense" and "mlfmm" (params m).
Comparison compare(const geometry::TriangleMesh& mesh, SimulationConfig cfg,
                   const mlfmm::MlfmmParams& m, const std::string& name) {
    const auto wave =
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
    cfg.wavelength = kLambda;
    cfg.gmres.tolerance = 1e-6;
    cfg.gmres.verbose = false;
    Simulation dense(mesh, wave, cfg);
    cfg.compression = "mlfmm";
    cfg.mlfmm = m;
    Simulation fmm(mesh, wave, cfg);
    const solver::GmresResult rd = dense.solve();
    const solver::GmresResult rf = fmm.solve();
    REQUIRE(rd.converged);
    REQUIRE(rf.converged);
    const VectorXr theta = VectorXr::LinSpaced(91, 0.0, constants::pi);
    const VectorXr sd = post::bistatic_rcs(dense.solution(), Vec3::UnitY(), theta);
    const VectorXr sf = post::bistatic_rcs(fmm.solution(), Vec3::UnitY(), theta);
    Comparison c;
    c.currents = (fmm.solution().currents - dense.solution().currents).norm() /
                 dense.solution().currents.norm();
    c.rcs = (sf - sd).cwiseAbs().maxCoeff() / sd.maxCoeff();
    WARN(name << ": 2N = " << dense.num_unknowns() << ", GMRES " << rd.iterations << " (dense) / "
              << rf.iterations << " (MLFMM) iterations, " << rd.wall_seconds << " / "
              << rf.wall_seconds << " s; currents " << c.currents << ", RCS " << c.rcs << "\n"
              << fmm.report());
    return c;
}

}  // namespace

TEST_CASE("mlfmm simulation: Si sphere, ICTF, default MLFMM parameters", "[validation][mlfmm]") {
#ifndef NDEBUG
    SKIP("dense GMRES of 2N = 3840: release only");
#else
    // R = 0.5 um, icosphere n = 3 (2N = 3840); the defaults (100 elements per leaf, lambda / 4
    // floor) give lambda / 2 leaves.
    SimulationConfig cfg;
    cfg.object = material::silicon_500nm();
    const Comparison c =
        compare(geometry::make_icosphere(kLambda, 3), cfg, mlfmm::MlfmmParams{}, "Si sphere");
    CHECK(c.currents <= 1e-3);
    CHECK(c.rcs <= 1e-3);
#endif
}

TEST_CASE("mlfmm simulation: rough box n = 1.5, PMCHWT + Jacobi, lambda / 4 leaves",
          "[validation][mlfmm]") {
#ifndef NDEBUG
    SKIP("dense GMRES of 2N = 7680: release only");
#else
    // 1 um x 1 um x 0.3 um, mesh 50 nm (2N = 7680), 4 levels with lambda / 4 leaves.
    geometry::RoughSurfaceParams p;
    p.edge_length_L = 1e-6;
    p.rms_roughness = 30e-9;
    p.correlation_length = 250e-9;
    p.mesh_size = 50e-9;
    p.box_depth = 0.3e-6;
    p.box_mesh_size = p.mesh_size;
    p.seed = 7;
    SimulationConfig cfg;
    cfg.object = {Complex(2.25, 0.0), Complex(1.0, 0.0)};
    cfg.formulation = formulation::Kind::PMCHWT;
    cfg.diagonal_preconditioner = true;
    mlfmm::MlfmmParams m;
    m.octree.max_elements_per_leaf = 4;
    const Comparison c = compare(geometry::make_rough_surface_mesh(p), cfg, m, "rough box");
    CHECK(c.currents <= 1e-3);
    CHECK(c.rcs <= 1e-3);
#endif
}
