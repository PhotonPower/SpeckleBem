// PML route smoke solve (WP-PML3a, ADR 0009; ctest label `validation-large`, release only,
// manual: `ctest --preset win-release -L validation-large -R "PML smoke"`). A flat Si open
// surface (patch 1.5 um, taper 0.5 um, gap 0.5 um, PML 1 um = 2 lambda_1 at 500 nm, sigma_max 4,
// order 2) under the rigorous angular-spectrum beam at normal incidence (w0 = 0.5 um: dark band
// at the PML start), dense operator, recommended formulation, direct LU. Checks that it solves,
// that the currents have decayed at the truncation edge (< 1e-3 of their maximum) and that the
// windowed reflected power (far field of the real part, window across the gap) lies in a loose
// sanity band of the Fresnel beam reflectance. The < 1 % acceptance (validation 7a) is WP-PML3b;
// the numbers are printed (WARN). Mesh spacing: SPECKLEBEM_PML_SMOKE_H_NM (default 125 nm; the
// dense size grows as h^-4 in time). SKIPs when the dense matrix does not fit into 60 % of the
// available memory.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

#include "fresnel_flat_support.hpp"

using namespace fresnel_flat;

TEST_CASE("PML smoke: flat Si open surface with a PML against the Fresnel beam reflectance",
          "[pml_smoke]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    Real h = 125e-9;
    if (const char* env = std::getenv("SPECKLEBEM_PML_SMOKE_H_NM"))
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
    const Real a0 = om.patch_half, a1 = om.taper_end, a2 = om.pml_start, a3 = om.outer_half;
    const Index unknowns = 2 * om.mesh.num_edges();
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
    PmlTermination pml;  // sigma_max 4, order 2, dark band 1e-4
    const auto t0 = std::chrono::steady_clock::now();
    Simulation sim(std::move(om), pml, beam, cfg);
    sim.assemble();
    const auto t1 = std::chrono::steady_clock::now();
    const solver::GmresResult res = sim.solve();
    const auto t2 = std::chrono::steady_clock::now();
    REQUIRE(res.converged);

    // Current decay: |J|, |M| at the centroids of the outermost ring of triangles against the
    // maximum over all triangles.
    const post::SurfaceSolution& s = sim.solution();
    const basis::RwgSpace& space = *s.problem->space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    Real j_max = 0, m_max = 0, j_edge = 0, m_edge = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 c = mesh.centroid(t);
        const basis::RwgSpace::Support sup = space.support(t);
        Vec3c J = Vec3c::Zero(), M = Vec3c::Zero();
        for (int a = 0; a < sup.count; ++a) {
            const Vec3 f = space.value(sup.n[a], t, c);
            J += s.currents(sup.n[a]) * f.cast<Complex>();
            M += s.currents(N + sup.n[a]) * f.cast<Complex>();
        }
        j_max = std::max(j_max, J.norm());
        m_max = std::max(m_max, M.norm());
        if (std::max(std::abs(c.x()), std::abs(c.y())) > a3 - h) {
            j_edge = std::max(j_edge, J.norm());
            m_edge = std::max(m_edge, M.norm());
        }
    }

    // Windowed reflected power over the beam power.
    const Real eta1 = constants::eta0;
    const HemisphereGrid g = hemisphere_grid(-1.0, 1.0);
    const HemispherePower hp =
        hemisphere_power(s, g, sim.far_field_options(6), eta1, Vec3(0.0, 0.0, -1.0), 0.3);
    const HemisphereGrid g2 = hemisphere_grid(-1.0, 1.0, 2);
    const HemispherePower hp2 =
        hemisphere_power(s, g2, sim.far_field_options(8), eta1, Vec3(0.0, 0.0, -1.0), 0.3);
    const Real R = hp.total / beam->power();
    const BeamReflectance ref = beam_reflectance(*beam, 1.0, cfg.object.eps_r);
    const Real err = R / ref.reflectance - 1.0;
    const Real assembly_s = std::chrono::duration<double>(t1 - t0).count();
    const Real solve_s = std::chrono::duration<double>(t2 - t1).count();
    WARN("PML smoke (flat Si, h = "
         << h * 1e9 << " nm, half widths " << a0 << " / " << a1 << " / " << a2 << " / " << a3
         << " m): 2N = " << unknowns << ", dark band ratio " << sim.dark_band_ratio()
         << "\n  assembly " << assembly_s << " s, LU " << solve_s << " s, residual "
         << res.true_relative_residual << "\n  edge decay |J| " << j_edge / j_max << ", |M| "
         << m_edge / m_max << "\n  R_sim " << R
         << " (grid/2, degree 8: " << hp2.total / beam->power() << "), R_beam " << ref.reflectance
         << ", Fresnel " << ref.fresnel_central << ", error " << 100.0 * err << " %, cone share "
         << hp.cone / hp.total << ", grazing share " << hp.grazing / hp.total << "\n"
         << sim.report());
    CHECK(j_edge < 1e-3 * j_max);
    CHECK(m_edge < 1e-3 * m_max);
    // Loose sanity band only (validation 7a, < 1 %, is WP-PML3b).
    CHECK(std::abs(err) < 0.10);
#endif
}
