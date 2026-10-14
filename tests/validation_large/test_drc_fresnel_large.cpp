// Absolute DRC against Fresnel on a flat surface (WP25; docs/01 Phase 5 "DRC normalised against
// Fresnel on a flat surface (< 1 %)"; ctest label `validation-large`, release only, manual:
// `ctest --preset win-release -L validation-large -R "DRC / Fresnel"`).
//
// Case: the PML-route flat Si surface of the WP-PML3a smoke test (patch 1.5 um, taper 0.5 um, gap
// 0.5 um, PML 1 um, sigma_max 4, order 2, h = 125 nm; rigorous angular-spectrum beam at normal
// incidence, w0 = 0.5 um, p; dense operator, recommended formulation, direct LU; 2N = 11 440,
// ~4.3 GB), built with the public Simulation PML constructor. The DRC comes from
// Simulation::drc (absolute, per steradian, P_inc = Simulation::incident_power() =
// AngularSpectrumBeam::power(), current window of the PML route applied automatically) on the
// 1 deg reflection-hemisphere grid post::hemisphere_grid(-1, 90, 180); its integral sum_i w_i D_i
// is the reflectance and must match the Fresnel beam reflectance R_beam
// (fresnel_flat::beam_reflectance: every plane wave of the beam reflected with its own Fresnel
// coefficients) within 1 %. Checks also: the integral equals Simulation::reflected_power / P_inc
// and the support-code integration (fresnel_flat::hemisphere_power) bitwise.
// Measured (2026-10-14, 4 threads): R = 0.388091 vs R_beam 0.388176, -0.022 %; assembly 1560 s,
// LU 62 s, ~4.3 GB.
//
// Timing (benchmarks/results/drc_evaluation.md): the DRC on the 1 deg grid (16 200 directions)
// and on a 0.5 deg grid (180 x 360 = 64 800 directions), windowed (the PML route), and on the 1 deg
// grid without window over every triangle (the cost of a box-route surface of this size; the
// problem copy without stretch is for timing only, its values are not physical).
// Mesh spacing: SPECKLEBEM_DRC_FRESNEL_H_NM (default 125 nm). SKIPs when the dense matrix does not
// fit into 60 % of the available memory.
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/postprocessing/scattering.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

#include "fresnel_flat_support.hpp"

using namespace fresnel_flat;

namespace {

[[maybe_unused]] Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

TEST_CASE("DRC / Fresnel: absolute DRC of a flat Si surface (PML route) integrates to R_beam",
          "[drc_fresnel]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    using Clock = std::chrono::steady_clock;
    Real h = 125e-9;
    if (const char* env = std::getenv("SPECKLEBEM_DRC_FRESNEL_H_NM"))
        h = std::stod(env) * 1e-9;
    const Real patch = 1.5e-6, taper = 0.5e-6, gap = 0.5e-6, pml_w = 1.0e-6;
    const Real w0 = 0.5e-6;
    geometry::OpenSurfaceParams op;
    op.patch_size = patch;
    op.taper_width = taper;
    op.gap_width = gap;
    op.pml_width = pml_w;
    geometry::OpenSurfaceMesh om =
        geometry::make_open_surface_mesh(flat_height_map(patch + 2.0 * taper, h), op);
    const Real a2 = om.pml_start;
    const Index unknowns = 2 * om.mesh.num_edges();
    const Index triangles = om.mesh.num_triangles();
    const Real need = 16.0 * static_cast<Real>(unknowns) * static_cast<Real>(unknowns);
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > need))
        SKIP("dense 2N = " << unknowns << " needs " << need / 1e9 << " GB, available "
                           << avail / 1e9 << " GB");

    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = w0;
    bp.polarization = excitation::Polarization::P;
    bp.region_radius = 1.02 * std::sqrt(2.0) * a2;  // the unmasked part (corners of the gap)
    const auto beam = std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());

    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = material::silicon_500nm();
    cfg.compression = "dense";
    cfg.solver = SolverKind::Direct;
    const PmlTermination pml;  // sigma_max 4, order 2, dark band 1e-4
    const auto t0 = Clock::now();
    Simulation sim(std::move(om), pml, beam, cfg);
    sim.assemble();
    const Real assembly_s = seconds_since(t0);
    const auto t1 = Clock::now();
    const solver::GmresResult res = sim.solve();
    const Real solve_s = seconds_since(t1);
    REQUIRE(res.converged);

    // Absolute DRC on the 1 deg reflection hemisphere and its integral.
    const post::SphereGrid g1 = post::hemisphere_grid(-1.0, 90, 180);
    const auto t2 = Clock::now();
    const VectorXr d1 = sim.drc(g1.dirs);
    const Real drc_1deg_s = seconds_since(t2);
    const Real p_inc = sim.incident_power();
    const Real R = g1.weights.dot(d1);
    const BeamReflectance ref = beam_reflectance(*beam, 1.0, cfg.object.eps_r);
    const Real err = R / ref.reflectance - 1.0;

    // Consistency with the power functions and the WP22c/PML support integration.
    const Real p_refl = sim.reflected_power(90, 180);
    const HemispherePower hp = hemisphere_power(sim.solution(), g1, sim.far_field_options(6),
                                                constants::eta0, Vec3(0.0, 0.0, -1.0), 0.3);
    // Specular peak: D at the specular direction (the pole of the grid has no node; evaluate it).
    const VectorXr d_spec = sim.drc(Vertices(Vec3(0.0, 0.0, -1.0).transpose()));

    // Timing: 0.5 deg grid (windowed) and the 1 deg grid over every triangle without window.
    const post::SphereGrid g05 = post::hemisphere_grid(-1.0, 180, 360);
    const auto t3 = Clock::now();
    const VectorXr d05 = sim.drc(g05.dirs);
    const Real drc_05deg_s = seconds_since(t3);
    const Real R05 = g05.weights.dot(d05);
    op::Problem unstretched = sim.problem();
    unstretched.stretch = nullptr;
    const post::SurfaceSolution all{&unstretched, sim.solution().currents};
    const auto t4 = Clock::now();
    const VectorXr d_all = post::drc(all, g1.dirs, p_inc);
    const Real drc_all_s = seconds_since(t4);
    const auto source_points =
        static_cast<Real>(triangles) * static_cast<Real>(kernels::triangle_rule(6).weights.size());
    const Real ns_per_pair = 1e9 * drc_all_s / (static_cast<Real>(g1.dirs.rows()) * source_points);

    WARN("DRC / Fresnel (flat Si, PML route, h = "
         << h * 1e9 << " nm): 2N = " << unknowns << ", " << triangles << " triangles; assembly "
         << assembly_s << " s, LU " << solve_s << " s, residual " << res.true_relative_residual
         << "\n  P_inc " << p_inc << " W, R = int D dOmega (1 deg) " << R << ", 0.5 deg " << R05
         << ", R_beam " << ref.reflectance << ", Fresnel " << ref.fresnel_central << ", error "
         << 100.0 * err << " %\n  D(specular) " << d_spec(0) << " 1/sr, max D on the grid "
         << d1.maxCoeff() << "\n  timing (OMP threads as set): drc 1 deg (16 200 dirs, windowed) "
         << drc_1deg_s << " s, 0.5 deg (64 800 dirs, windowed) " << drc_05deg_s
         << " s, 1 deg over all triangles (no window) " << drc_all_s << " s = " << ns_per_pair
         << " ns per (direction x source point), " << source_points << " source points; D max "
         << d_all.maxCoeff() << "\n"
         << sim.report());
    CHECK(std::abs(err) < 0.01);
    CHECK(std::abs(R / (p_refl / p_inc) - 1.0) < 1e-13);
    CHECK(hp.total == p_refl);
    CHECK(std::abs(R05 / R - 1.0) < 1e-4);
#endif
}
