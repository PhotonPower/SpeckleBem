// Total scattered power through the public API (WP25): Simulation::scattered_power over the full
// sphere (post::sphere_grid, Gauss-Legendre in theta x trapezoid in phi) divided by the plane-wave
// intensity |E_0|^2 / (2 eta_1) is the scattering cross section, compared with Mie.
// Case: the validated Si sphere of docs/05 "Power balance, Ag and Si spheres d = 1 um, PMCHWT +
// LU" (radius 0.5 um, silicon_500nm, lambda = 500 nm, ka = 2 pi, plane wave k = +z, E along x,
// icosphere n = 3 = lambda / 6.6, default kernel options). Tolerance: C_sca within 2 % of Mie,
// the bound of that case in tests/validation/test_mie_sphere_dense.cpp (measured there -1.5 %;
// here, through Simulation with the current default options, -1.17 %).
// Cross-check: the independent unit-sphere rule of mie_dense_test::power_balance (Gauss-Legendre
// in cos theta) on the same currents agrees to 1e-9 (both rules are converged; measured 4e-15).
// The sanitizer build runs the cross-check only, at n = 1 with cheap quadrature (the default
// options cost ~8 min at n = 2 there).
#include "specklebem/simulation.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <memory>

#include "mie_dense_test_support.hpp"

using namespace mie_dense_test;

namespace {

struct PowerResult {
    Real c_sca = 0;          ///< scattered_power / I_inc [m^2]
    Real c_sca_support = 0;  ///< power_balance (Gauss-Legendre in cos theta) [m^2]
    Real c_sca_mie = 0;
    Real post_s = 0;  ///< scattered_power evaluation time [s]
};

/// cheap: low fixed quadrature degrees (the WP7 rules of the driver unit tests), for the
/// sanitizer build, where the default options cost ~8 min at n = 2.
PowerResult run(int subdivisions, bool cheap = false) {
    const material::Material si = material::silicon_500nm();
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = si;
    cfg.formulation = formulation::Kind::PMCHWT;
    cfg.solver = SolverKind::Direct;
    if (cheap) {
        cfg.kernels.quad_degree_far = 1;
        cfg.kernels.quad_degree_near = 2;
        cfg.kernels.quad_degree_sing = 2;
        cfg.kernels.outer_grading_levels = 0;
        cfg.kernels.target_accuracy = 0.0;
    }
    Simulation sim(
        geometry::make_icosphere(kRadius, subdivisions),
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0)), cfg);
    const solver::GmresResult res = sim.solve();
    REQUIRE(res.true_relative_residual < 1e-10);
    const Real eta1 = constants::eta0;
    const Real i_inc = 1.0 / (2.0 * eta1);  // |E_0| = 1 V/m
    PowerResult r;
    const auto t0 = std::chrono::steady_clock::now();
    r.c_sca = sim.scattered_power(64, 128) / i_inc;
    r.post_s = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    r.c_sca_support = power_balance(sim.solution(), Vec3c(1.0, 0.0, 0.0), k_vacuum(), eta1).c_sca;
    r.c_sca_mie = mie_cross_sections(si).c_sca;
    WARN("Si sphere d = 1 um, icosphere n = "
         << subdivisions << " (2N = " << sim.num_unknowns() << "): C_sca = " << r.c_sca
         << " m^2 (scattered_power 64 x 128, " << r.post_s << " s), Mie " << r.c_sca_mie << ", rel "
         << r.c_sca / r.c_sca_mie - 1.0 << "; cos-theta rule " << r.c_sca_support << " (rel "
         << r.c_sca / r.c_sca_support - 1.0 << ")");
    return r;
}

}  // namespace

TEST_CASE("DRC / power: scattered power of the Si sphere against the Mie cross section",
          "[validation][drc_power]") {
#ifdef NDEBUG
    const PowerResult r = run(3);
    CHECK(rel_error(r.c_sca, r.c_sca_mie) < 0.02);
#else
    const PowerResult r = run(1, true);
    WARN("icosphere n = 3 (Mie bound) runs in optimised builds only");
#endif
    CHECK(rel_error(r.c_sca, r.c_sca_support) < 1e-9);
}
