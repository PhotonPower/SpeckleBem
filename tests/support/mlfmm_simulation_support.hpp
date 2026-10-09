#pragma once
/// @file mlfmm_simulation_support.hpp
/// GMRES through Simulation with compression "mlfmm" against the same solve on the dense
/// operator (WP20b; tests/validation/test_mlfmm_simulation.cpp and the larger case in
/// tests/validation_large/test_mlfmm_vs_dense_large.cpp). lambda = 500 nm plane wave (+z, E along
/// x), vacuum exterior, GMRES tolerance 1e-6 for both solves.
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

namespace mlfmm_simulation_test {

using namespace specklebem;

inline constexpr Real kLambda = 500e-9;

struct Comparison {
    Real currents = 0;  ///< |x_mlfmm - x_dense| / |x_dense|
    Real rcs = 0;       ///< max |sigma_mlfmm - sigma_dense| / max sigma_dense (xz plane)
};

/// Solves the configuration with "dense" and with "mlfmm" (params m) and compares.
inline Comparison compare(const geometry::TriangleMesh& mesh, SimulationConfig cfg,
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

}  // namespace mlfmm_simulation_test
