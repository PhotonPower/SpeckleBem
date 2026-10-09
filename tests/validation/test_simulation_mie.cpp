// Mie validation through the Simulation driver (WP14a): the WP11 dielectric case (sphere of
// radius 0.5 um, n = 1.5, lambda = 500 nm, plane wave k = +z, E along x, default kernel options)
// at icosphere n = 3 (h ~ lambda / 6.6):
//  * direct LU with PMCHWT through the driver reproduces eps_rr of the low-level WP11 pipeline
//    (mie_dense_test_support.hpp, solve_mie_dense) to 1e-10 relative in the xz- and yz-planes
//    (same Problem, so the results are in fact bitwise equal);
//  * GMRES with ICTF, tol 1e-8, no preconditioner reaches the direct eps_rr to 1e-4 (absolute).
// The sanitizer build runs the same checks at n = 2 (the LU of 2N = 3840 is too slow there, as in
// test_mie_sphere_dense.cpp).
#include "specklebem/simulation.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

#include "mie_dense_test_support.hpp"

using namespace mie_dense_test;

namespace {

struct DriverResult {
    Real eps_xz = 0, eps_yz = 0;
    solver::GmresResult res;
};

DriverResult run_driver(int subdivisions, SolverKind solver_kind, formulation::Kind kind) {
    const material::Material n15 = lossless_n15();
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = n15;
    cfg.formulation = kind;
    cfg.diagonal_preconditioner = false;
    cfg.solver = solver_kind;
    cfg.gmres.tolerance = 1e-8;
    cfg.gmres.verbose = false;
    Simulation sim(
        geometry::make_icosphere(kRadius, subdivisions),
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0)), cfg);
    DriverResult out;
    out.res = sim.solve();
    const MieSolution mie(MieParams{kRadius, kLambda, n15, material::vacuum(), 0});
    out.eps_xz = rcs_eps_rr(sim.solution(), mie, kRcsAngles, Vec3::UnitY(), 0.0);
    out.eps_yz = rcs_eps_rr(sim.solution(), mie, kRcsAngles, -Vec3::UnitX(), 0.5 * kPi);
    WARN("Simulation driver, icosphere n = " << subdivisions << ":\n" << sim.report());
    return out;
}

}  // namespace

TEST_CASE("Simulation driver: Mie dielectric sphere n = 1.5 (direct and GMRES)",
          "[validation][simulation]") {
#ifdef NDEBUG
    const int n = 3;
#else
    const int n = 2;  // LU of 2N = 3840 too slow under ASan
#endif
    const DenseMieResult wp11 = solve_mie_dense(lossless_n15(), n);
    report("WP11 pipeline, n = 1.5", n, wp11);

    const DriverResult direct = run_driver(n, SolverKind::Direct, formulation::Kind::PMCHWT);
    CHECK(direct.res.converged);
    CHECK(direct.res.true_relative_residual < 1e-10);
    CHECK(rel_error(direct.eps_xz, wp11.eps_xz) < 1e-10);
    CHECK(rel_error(direct.eps_yz, wp11.eps_yz) < 1e-10);

    const DriverResult gm = run_driver(n, SolverKind::Gmres, formulation::Kind::ICTF);
    WARN("icosphere n = " << n << ": eps_rr direct xz / yz = " << direct.eps_xz << " / "
                          << direct.eps_yz << ", GMRES (ICTF, tol 1e-8) " << gm.eps_xz << " / "
                          << gm.eps_yz << " after " << gm.res.iterations
                          << " iterations (true residual " << gm.res.true_relative_residual << ", "
                          << gm.res.wall_seconds << " s); LU " << direct.res.wall_seconds << " s");
    CHECK(gm.res.converged);
    CHECK(gm.res.true_relative_residual < 1e-7);
    CHECK(std::abs(gm.eps_xz - direct.eps_xz) < 1e-4);
    CHECK(std::abs(gm.eps_yz - direct.eps_yz) < 1e-4);
}
