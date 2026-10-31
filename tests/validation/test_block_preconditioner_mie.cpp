// Mie sphere through Simulation with compression "mlfmm" and the near-field block-diagonal
// preconditioner (WP-PC1, ADR 0011 decision 3: "results unchanged within the solver tolerance").
// Ag sphere d = 1 um (vacuum, 500 nm, k = +z, E along x; ag_sphere_mlfmm_support.hpp), icosphere
// n = 3 (2N = 3840), default MLFMM parameters (d0 = 3), ICTF (automatic), GMRES tolerance 1e-6.
// The block-diagonal solves (octree leaves and one level up) must reproduce the Jacobi solve (the
// default for Ag) to the solver tolerance: bistatic RCS (xz plane, 1 degree) within 1e-5 of its
// maximum and eps_rr against Mie within 1e-5 (absolute) in both planes; the guard eps_rr < 1 %
// (docs/05 row "Ag sphere d = 1 um") holds for every preconditioner. Iteration counts are
// reported (WARN), not checked. Measured 2026-10-25 (win-release): Jacobi 210, leaf 490, one
// level up 266 iterations; RCS differences 4.8e-7 / 1.4e-6, eps_rr 3.481e-3 (xz) / 2.560e-3 (yz).
// The near-field Schur-complement preconditioner (WP-SC2, ADR 0012 decision 2; J and M blocks one
// level up) is gated with the same tolerances (results, not iterations: on the box route it may
// converge more slowly than Jacobi, as the blocks do). Measured 2026-10-31 (win-release): 261
// iterations, RCS difference 4.2e-7, eps_rr 3.481e-3 (xz) / 2.560e-3 (yz). The ILUT of the
// near-field blocks A and D (WP-SC3, ADR 0012 decision 6) is gated with the same tolerances.
// Measured 2026-10-31 (win-release) with the then default parameters fill 3, drop 1e-4, AMD: 63
// iterations, RCS difference 4.7e-7, eps_rr 3.481e-3 (xz) / 2.560e-3 (yz), setup 8.2 s, 88 MB.
// Since WP-SC4 (ADR 0012 decision 9) two ILUT sets run: the library default (fill 2, drop 1e-2,
// the default of the "mlfmm" PML route) and fill 3 / drop 1e-3 (the reference set of the
// decision-9 candidate check: the default would have become fill 2 / drop 1e-3 if it needed more
// than 1.5x the iterations of this set). Measured 2026-10-31 (win-release): default 68 iterations,
// RCS difference 1.3e-6, setup 1.1 s, 22 MB; fill 3 / drop 1e-3 63 iterations, 5.1e-7, 6.0 s, 69 MB
// (benchmarks/results/preconditioner.md, section WP-SC4).
#include "specklebem/simulation.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <optional>
#include <sstream>
#include <string>

#include "ag_sphere_mlfmm_support.hpp"

using namespace specklebem;

TEST_CASE("block preconditioner: Ag sphere with MLFMM reproduces the Jacobi solution",
          "[validation][block_preconditioner]") {
#ifndef NDEBUG
    SKIP("MLFMM GMRES solves of 2N = 3840: release only");
#else
    ag_sphere_mlfmm::Case c;
    c.mesh = "ico";
    c.subdivisions = 3;
    c.diameter = 1e-6;
    c.tolerance = 1e-6;
    const geometry::TriangleMesh mesh =
        ag_sphere_mlfmm::make_mesh(c.mesh, c.diameter, c.subdivisions);
    const reference::MieSolution mie(
        reference::MieParams{0.5 * c.diameter, ag_sphere_mlfmm::kLambda, material::silver_500nm(),
                             material::vacuum(), 0});
    const VectorXr theta = VectorXr::LinSpaced(ag_sphere_mlfmm::kRcsAngles, 0.0, constants::pi);
    struct Out {
        VectorXr rcs;
        Real eps_xz = 0, eps_yz = 0;
        int iterations = 0;
    };
    const auto run = [&](PreconditionerKind kind, int levels,
                         const std::optional<solver::IlutParams>& ilut) {
        SimulationConfig cfg = ag_sphere_mlfmm::make_config(c);
        cfg.preconditioner = kind;
        cfg.block_levels_above_leaf = levels;
        if (ilut)
            cfg.ilut = *ilut;
        cfg.gmres.verbose = false;
        Simulation sim(mesh, ag_sphere_mlfmm::make_wave(), cfg);
        const solver::GmresResult r = sim.solve();
        REQUIRE(r.converged);
        Out o;
        o.iterations = r.iterations;
        o.rcs = post::bistatic_rcs(sim.solution(), Vec3::UnitY(), theta);
        o.eps_xz = ag_sphere_mlfmm::eps_rr(sim.solution(), mie, ag_sphere_mlfmm::kRcsAngles,
                                           Vec3::UnitY(), 0.0);
        o.eps_yz = ag_sphere_mlfmm::eps_rr(sim.solution(), mie, ag_sphere_mlfmm::kRcsAngles,
                                           -Vec3::UnitX(), 0.5 * constants::pi);
        if (kind == PreconditionerKind::BlockDiagonal) {
            REQUIRE(sim.block_preconditioner() != nullptr);
            WARN("level offset " << levels << ": " << sim.block_preconditioner()->describe());
        }
        if (kind == PreconditionerKind::Schur) {
            REQUIRE(sim.schur_preconditioner() != nullptr);
            WARN("level offset " << levels << ": " << sim.schur_preconditioner()->describe());
        }
        if (kind == PreconditionerKind::IncompleteLu) {
            REQUIRE(sim.incomplete_lu_preconditioner() != nullptr);
            WARN(sim.incomplete_lu_preconditioner()->describe());
        }
        return o;
    };
    const Out jac = run(PreconditionerKind::Diagonal, 0, std::nullopt);
    CHECK(jac.eps_xz < 1e-2);
    CHECK(jac.eps_yz < 1e-2);
    struct Variant {
        PreconditionerKind kind;
        int levels;
        std::optional<solver::IlutParams> ilut;  ///< empty: the library default
    };
    solver::IlutParams fill3;
    fill3.fill_factor = 3.0;
    fill3.drop_tol = 1e-3;
    for (const Variant& v : {Variant{PreconditionerKind::BlockDiagonal, 0, std::nullopt},
                             Variant{PreconditionerKind::BlockDiagonal, 1, std::nullopt},
                             Variant{PreconditionerKind::Schur, 1, std::nullopt},
                             Variant{PreconditionerKind::IncompleteLu, 1, std::nullopt},
                             Variant{PreconditionerKind::IncompleteLu, 1, fill3}}) {
        const int levels = v.levels;
        const Out b = run(v.kind, levels, v.ilut);
        const Real drcs = (b.rcs - jac.rcs).cwiseAbs().maxCoeff() / jac.rcs.maxCoeff();
        std::ostringstream what;
        if (v.kind == PreconditionerKind::IncompleteLu) {
            const solver::IlutParams q = v.ilut.value_or(solver::IlutParams{});
            what << " (fill " << q.fill_factor << ", drop " << q.drop_tol
                 << (v.ilut ? ", explicit)" : ", library default)");
        } else {
            what << " (levels above the leaf " << levels << ")";
        }
        WARN("Ag sphere 2N = 3840, GMRES tol 1e-6: Jacobi "
             << jac.iterations << " iterations, " << to_string(v.kind) << what.str() << " "
             << b.iterations << "; RCS difference " << drcs << ", eps_rr xz " << b.eps_xz
             << " (Jacobi " << jac.eps_xz << "), yz " << b.eps_yz << " (" << jac.eps_yz << ")");
        CHECK(drcs <= 1e-5);
        CHECK(std::abs(b.eps_xz - jac.eps_xz) <= 1e-5);
        CHECK(std::abs(b.eps_yz - jac.eps_yz) <= 1e-5);
        CHECK(b.eps_xz < 1e-2);
        CHECK(b.eps_yz < 1e-2);
    }
#endif
}
