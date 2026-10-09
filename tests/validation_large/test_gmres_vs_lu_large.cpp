// GMRES vs dense LU on a ~2 * 10^4-unknown rough-surface system (WP13; docs/05 row "GMRES vs
// LU, 2 * 10^4 unknowns", Phase 3 DoD "GMRES reproduces the direct solution to residual
// tolerance on a 20 k-unknown problem"). ctest label `validation-large`, manual / nightly:
//   ctest --preset release -L validation-large -R gmres
//
// System: a closed Si box with a Gaussian rough top face (make_rough_surface_mesh, sigma =
// 50 nm, Lc = 500 nm, fixed seed), L = 1.6 um, mesh size 50 nm (lambda / 10), depth 0.5 um with
// the uniform box (walls and bottom at the top-face spacing): 2N = 19 968. The shallow box is
// not the ADR 0006 half-space truncation; it is a legitimate closed dielectric scatterer, and
// the case tests the linear algebra (GMRES against LU on a realistic rough-surface SIE
// system), not the scattering physics. Plane wave at normal incidence (k = +z from z < 0,
// E along x, lambda = 500 nm), ICTF, unpreconditioned full GMRES with tol 1e-6.
//
// Memory: Z (16 (2N)^2 bytes = 6.4 GB) plus the LU copy of solve_direct, 12.8 GB; the Krylov
// basis adds 16 (iterations + 1) 2N bytes. SKIPs in unoptimised builds and when Z plus the LU
// copy exceed 60 % of the physical memory.
//
// Measured 2026-10-09 (win-release, 24 cores, 128 GB, OpenBLAS; 182 s in total): assembly
// 10.4 s; full GMRES 330 iterations in 87 s, monitored = true residual 9.67e-7; LU 84.8 s,
// residual 2.7e-15, rcond 8.9e-8; |x_GMRES - x_LU| / |x_LU| = 5.3e-6; peak RSS 12.8 GB.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/solver/direct.hpp"
#include "specklebem/solver/gmres.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <memory>

#include "system_memory.hpp"

using namespace specklebem;

namespace {

[[maybe_unused]] double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

TEST_CASE("gmres vs LU large: Si rough-surface box, 2N ~ 2e4, ICTF unpreconditioned",
          "[validation-large][gmres]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = 1.6e-6;
    rp.rms_roughness = 50e-9;
    rp.correlation_length = 500e-9;
    rp.mesh_size = 50e-9;
    rp.seed = 13;
    rp.box_depth = 0.5e-6;
    rp.box_mesh_size = rp.mesh_size;  // uniform box (Si is weakly absorbing, ADR 0006)
    const geometry::TriangleMesh mesh = geometry::make_rough_surface_mesh(rp);
    const basis::RwgSpace space(mesh);
    const Index unknowns = 2 * space.size();

    const Real need = 2.0 * 16.0 * static_cast<Real>(unknowns) * static_cast<Real>(unknowns);
    const Real phys = system_memory::physical_memory_bytes();
    if (!(phys > 0.0) || need > 0.6 * phys) {
        SKIP("2N = " << unknowns << " needs ~" << need / 1e9 << " GB for Z and its LU copy, more "
                     << "than 60 % of the " << phys / 1e9 << " GB of physical memory");
    }
    WARN("rough-surface box: " << mesh.num_triangles() << " triangles, 2N = " << unknowns
                               << ", estimated " << need / 1e9 << " GB of " << phys / 1e9
                               << " GB physical memory");

    const excitation::PlaneWave wave(500e-9, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
    const std::unique_ptr<formulation::Formulation> form =
        formulation::make_formulation(formulation::Kind::ICTF);
    op::Problem problem;
    problem.space = &space;
    problem.exterior = material::vacuum();
    problem.object = material::silicon_500nm();
    problem.formulation = form.get();
    problem.excitation = &wave;
    problem.omega = wave.omega();

    auto t0 = std::chrono::steady_clock::now();
    const std::shared_ptr<op::LinearOperator> built = op::DenseStrategy().build(problem);
    const auto* Z = dynamic_cast<const op::DenseOperator*>(built.get());
    REQUIRE(Z != nullptr);
    const VectorXc b = op::assemble_rhs(problem);
    const double assembly_s = seconds_since(t0);

    solver::GmresParams gp;
    gp.tolerance = 1e-6;
    gp.max_iter = 5000;
    gp.restart = 0;
    gp.verbose = true;
    const solver::IdentityPreconditioner none;
    const solver::GmresResult g = solver::gmres(*Z, b, none, gp);

    t0 = std::chrono::steady_clock::now();
    solver::DirectSolveInfo info;
    const VectorXc x_lu = solver::solve_direct(*Z, b, &info);
    const double lu_s = seconds_since(t0);
    const Real err = (g.x - x_lu).norm() / x_lu.norm();

    WARN("2N = " << unknowns << ": assembly " << assembly_s << " s; GMRES " << g.iterations
                 << " iterations, " << g.wall_seconds << " s, monitored residual "
                 << g.residual_history.back() << ", true residual " << g.true_relative_residual
                 << "; LU " << lu_s << " s, residual " << info.residual << ", rcond " << info.rcond
                 << "; |x_GMRES - x_LU| / |x_LU| = " << err << "; peak RSS "
                 << system_memory::peak_rss_bytes() / 1e9 << " GB");
    CHECK(unknowns > 19000);
    CHECK(unknowns < 21000);
    CHECK(g.converged);
    CHECK(g.residual_history.back() <= gp.tolerance);
    // Unpreconditioned: the monitored residual is the true one (Arnoldi estimate).
    CHECK(g.true_relative_residual <= 1e-6);
    CHECK(err <= 1e-3);
#endif
}
