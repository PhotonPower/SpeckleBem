#pragma once
/// @file pml_huygens_support.hpp
/// Validations 7f and 7g of the PML route's Huygens-plane far field (ADR 0010, WP-PML6; docs/05
/// "PML route", rows 7f and 7g). Shared by the study executable benchmarks/pml_huygens.cpp, the
/// validation test tests/validation/test_pml_huygens.cpp (7f at CI size) and the validation-large
/// test tests/validation_large/test_pml_huygens_large.cpp (7f in the 7a geometry, 7g); no Catch2.
///
/// Cases are pml_rough_ag::RoughCase (vacuum over Ag or Si, rigorous beam at 0 deg, E along x):
///  * 7f: flat Ag and flat Si in the 7a geometry at normal incidence (pml_fresnel::seven_a_case:
///    patch 1.5 um, taper / gap 0.5 um, W_pml = 2 lambda, sigma_max 4, m 2, w0 = lambda, h =
///    lambda / 4; 2N = 11 440; flat_case()) and at CI size with a thinner PML (flat_ci_case(),
///    2N = 7 632): the Huygens far field must agree with the windowed-current far field (no
///    bound wave on a flat surface): L2 metric <= 1e-3, |dR| <= 1e-4;
///  * 7g: the rough Ag U case of WP-PML4b (pml_rough_ag::u_case(), 2N = 22 816), its L x 1.2
///    partner (patch_case()) and a wide-gap partner (gap 1.375 um, the same patch, taper, PML,
///    beam and map) on which the plane window can be scaled by 0.7 and 1.3 (the scaled U window
///    must end before pml_start); solved through the fast path (compression "mlfmm": MLFMM d0 = 3
///    plus the PML ring, GMRES tol 1e-7, the 7h configuration) or densely.
///
/// Far fields (pml_rough_ag::FarField: |F|^2 on post::hemisphere_grid(-1, 90, 180) and the xz /
/// yz cuts, R = P_refl / P_inc): the windowed currents (Simulation::far_field_options with
/// FarFieldSource::Currents) and the Huygens plane (post::huygens_plane_field once per
/// z_h / spacing, radiated on the grids; other windows of the same grid by post::huygens_reweight,
/// bitwise the direct evaluation). The metric is pml_rough_ag::compare (ADR 0006 box-check L2 of
/// |F|^2 over the hemisphere; eps_rr on the cuts; dR = R_t / R_r - 1).
#include "specklebem/postprocessing/huygens.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_rough_ag_support.hpp"

namespace pml_huygens {

using namespace specklebem;
using pml_rough_ag::FarField;
using pml_rough_ag::kLambda;
using pml_rough_ag::Metrics;
using pml_rough_ag::RoughCase;

/// 7f: flat Ag or Si in the 7a geometry at 0 deg (pml_fresnel::seven_a_case("ag"|"si", 0)).
[[nodiscard]] inline RoughCase flat_case(const std::string& material) {
    RoughCase c;
    c.tag = "flat_" + material;
    c.material = material;
    c.rms = 0.0;
    c.patch = 1.5e-6;
    c.map_patch = 1.5e-6;
    c.taper = 0.5e-6;
    c.gap = 0.5e-6;
    c.pml_width = 2.0 * kLambda;
    c.mesh_size = kLambda / 4.0;
    c.waist = kLambda;
    return c;
}

/// 7f at CI size: flat_case() (the 7a zones: patch 1.5 um, taper 0.5 um, gap 0.5 um, w0 = lambda,
/// h = lambda / 4, plane window (1.25, 1.75 um) = (2.5 w0, 3.5 w0), transition Delta = lambda)
/// with the thin PML of the 7h CI case (pml_ring::flat_ag_ci_case(): W_pml = lambda, sigma_max 2)
/// and the hemisphere 60 x 120; 2N = 7 632 instead of 11 440. The 7h CI geometry itself (patch
/// 1 um, taper 0.25 um: window (0.75, 1.25 um), starting at 1.5 w0) does not qualify for 7f: the
/// reflected beam crosses the plane window there and the plane's truncation error dominates (Ag
/// L2 8.1e-3, dR -5.4e-4; a 1 um patch with a 0.5 um taper, window from 2 w0: 2.1e-3; record
/// benchmarks/results/pml_huygens.md, section 7f).
[[nodiscard]] inline RoughCase flat_ci_case(const std::string& material) {
    RoughCase c = flat_case(material);
    c.tag = "flat_ci_" + material;
    c.correlation_length = 0.0;
    c.pml_width = kLambda;
    c.sigma_max = 2.0;
    c.n_theta = 60;
    c.n_phi = 120;
    return c;
}

/// 7g: U with the gap widened to 1.375 um (pml_start 3.75 um), so that the U window (taper_end
/// 2.375, pml_start 2.875 um) scaled by 1.3 (3.0875 / 3.7375 um) still ends before pml_start;
/// 7c showed the gap inert at a fixed current window (<= 4.4e-5).
[[nodiscard]] inline RoughCase wide_gap_case() {
    RoughCase c = pml_rough_ag::u_case();
    c.tag = "ugap";
    c.gap = 1.375e-6;
    return c;
}

/// Diagnostic (benchmarks/pml_huygens.cpp --cases ugap3): U with a 3 um gap (pml_start
/// 5.375 um), for nested plane windows of one plane (does the Huygens far field converge with
/// the plane width?).
[[nodiscard]] inline RoughCase wide_gap3_case() {
    RoughCase c = pml_rough_ag::u_case();
    c.tag = "ugap3";
    c.gap = 3.0e-6;
    return c;
}

/// Diagnostic: the 7d partner of wide_gap3_case() (L x 1.2 = 4.5 um, gap 2.625 um: the same
/// pml_start 5.375 um, so both share the plane window (2.375, 5.375 um)).
[[nodiscard]] inline RoughCase wide_gap3_patch_case() {
    RoughCase c = pml_rough_ag::patch_case();
    c.tag = "l12gap3";
    c.gap = 2.625e-6;
    return c;
}

/// Solver of a case.
enum class Solver {
    Dense,  ///< dense operator, LU (Simulation SolverKind::Direct)
    Fast    ///< compression "mlfmm" (MLFMM + PML ring), GMRES (the 7h configuration)
};

struct FastSettings {
    Real digits = 3.0;
    Real gmres_tolerance = 1e-7;
    int max_iter = 5000;
};

/// A solved case (the Simulation owns the problem the solution points into).
struct Solved {
    RoughCase c;
    std::shared_ptr<excitation::AngularSpectrumBeam> beam;
    std::unique_ptr<Simulation> sim;
    std::string solver;
    Index unknowns = 0;
    std::array<Real, 4> half{};
    int iterations = 0;
    bool converged = false;
    Real residual = 0;  ///< true relative residual
    Real assemble_s = 0, solve_s = 0;
    pml_rough_ag::ZoneCurrents gap;
    Real peak_rss = 0;
    std::string report;
    [[nodiscard]] post::SurfaceSolution solution() const { return sim->solution(); }
    [[nodiscard]] Real power() const { return beam->power(); }
};

[[nodiscard]] inline Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Builds and solves a case.
inline Solved solve(const RoughCase& c, Solver solver, const FastSettings& f = {}) {
    Solved r;
    r.c = c;
    const pml_rough_ag::Size size = pml_rough_ag::size_of(c);
    r.half = size.half;
    r.beam = pml_rough_ag::make_beam(c, size.half[2]);
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    if (solver == Solver::Fast) {
        cfg.compression = "mlfmm";
        cfg.solver = SolverKind::Gmres;
        cfg.mlfmm.accuracy_digits = f.digits;
        cfg.gmres.tolerance = f.gmres_tolerance;
        cfg.gmres.max_iter = f.max_iter;
        r.solver = "fast-gmres";
    } else {
        r.solver = "dense-lu";
    }
    r.sim = std::make_unique<Simulation>(pml_rough_ag::make_open_mesh(c),
                                         pml_rough_ag::make_termination(c), r.beam, cfg);
    r.unknowns = r.sim->num_unknowns();
    auto t0 = std::chrono::steady_clock::now();
    r.sim->assemble();
    r.assemble_s = seconds_since(t0);
    const solver::GmresResult g = r.sim->solve();
    r.solve_s = g.wall_seconds;
    r.iterations = g.iterations;
    r.converged = g.converged;
    r.residual = g.true_relative_residual;
    r.gap = pml_rough_ag::zone_currents(*r.sim, r.sim->solution().currents);
    r.report = r.sim->report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// The windowed-current far field of a solved case (default window or `window`).
[[nodiscard]] inline FarField currents_far_field(
    const Solved& s, const std::string& tag,
    const std::optional<post::CurrentWindow>& window = std::nullopt) {
    FarField f = pml_rough_ag::evaluate_far_field(
        *s.sim, s.solution(), s.c, window.value_or(*s.sim->far_field_window()), s.power());
    f.tag = tag;
    return f;
}

/// The Huygens plane field of a solved case (FieldOptions of the currents' quad degree; the plane
/// window defaults to the current window).
[[nodiscard]] inline post::HuygensPlaneField plane_field(const Solved& s,
                                                         const post::HuygensOptions& h) {
    post::FieldOptions fo = s.sim->far_field_options(s.c.ff_degree, FarFieldSource::Huygens);
    fo.huygens = h;
    return post::huygens_plane_field(s.solution(), fo);
}

/// Far field of a plane field on the case's grids.
[[nodiscard]] inline FarField radiate(const post::HuygensPlaneField& pf, const Solved& s,
                                      const std::string& tag) {
    FarField f;
    f.tag = tag;
    f.window = pf.plane.window;
    f.n_theta = s.c.n_theta;
    f.n_phi = s.c.n_phi;
    f.power = s.power();
    f.eta1 = material::vacuum().wave_impedance(s.sim->problem().omega).real();
    const post::SphereGrid g = post::hemisphere_grid(-1.0, s.c.n_theta, s.c.n_phi);
    box_validity::FieldMatrix F;
    post::huygens_radiate(pf, g.dirs, F);
    f.hemi = F.rowwise().squaredNorm();
    f.p_refl = g.weights.dot(f.hemi) / (2.0 * f.eta1);
    post::huygens_radiate(pf, box_validity::cut_directions(false), F);
    f.cut_xz = F.rowwise().squaredNorm();
    post::huygens_radiate(pf, box_validity::cut_directions(true), F);
    f.cut_yz = F.rowwise().squaredNorm();
    return f;
}

/// Window scaled by a factor (both half widths).
[[nodiscard]] inline post::CurrentWindow scaled(const post::CurrentWindow& w, Real factor) {
    return {factor * w.inner_half, factor * w.outer_half};
}

/// One machine-readable line of a solved case.
[[nodiscard]] inline std::string row_line(const Solved& s, const FarField& currents,
                                          const FarField& huygens,
                                          const post::HuygensPlaneField& pf) {
    char buf[1600];
    const RoughCase& c = s.c;
    const int len = std::snprintf(
        buf, sizeof(buf),
        "ROW | huygens | %s | %s | L %.4g | taper %.4g | gap %.4g | pml %.4g | h %.4g | rms %.3g "
        "| Lc %.3g | seed %llu | w0 %.4g | half %.4g/%.4g/%.4g/%.4g | 2N %lld | %s | its %d | "
        "converged %s | true res %.2e | gap rms J %.3e M %.3e | R currents %.7f | R huygens %.7f "
        "| plane z %.4g spacing %.4g window %.4g/%.4g points %lld | plane %.1f s | assemble %.1f "
        "s | solve %.1f s | peak %.2f GB",
        c.tag.c_str(), c.material.c_str(), c.patch, c.taper, c.gap, c.pml_width, c.mesh_size, c.rms,
        c.correlation_length, static_cast<unsigned long long>(c.seed), c.waist, s.half[0],
        s.half[1], s.half[2], s.half[3], static_cast<long long>(s.unknowns), s.solver.c_str(),
        s.iterations, s.converged ? "yes" : "no", s.residual, s.gap.gap_j, s.gap.gap_m,
        currents.reflectance(), huygens.reflectance(), pf.plane.z, pf.plane.spacing,
        pf.plane.window.inner_half, pf.plane.window.outer_half,
        static_cast<long long>(pf.plane.points.rows()), pf.seconds, s.assemble_s, s.solve_s,
        s.peak_rss / 1e9);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_huygens::row_line: formatting failed or line truncated");
    return buf;
}

/// The 7f comparisons of a solved flat case: the Huygens plane at its defaults, z_h x 1.5 and
/// spacing / 1.5 (lambda / 6) against the windowed currents of the same solution, and the two
/// sensitivities against the default plane. Validation 7f (docs/05) requires l2 <= 1e-3 and
/// |dR| <= 1e-4 for m, mz and ms.
struct SevenF {
    post::HuygensPlaneField plane;  ///< the default plane field
    FarField currents, huygens, z15, spacing;
    Metrics m;                       ///< Huygens (defaults) vs currents
    Metrics mz;                      ///< z_h x 1.5 vs currents
    Metrics ms;                      ///< spacing / 1.5 vs currents
    Metrics mzh;                     ///< z_h x 1.5 vs the default plane
    Metrics msh;                     ///< spacing / 1.5 vs the default plane
    std::vector<std::string> lines;  ///< ROW line and the five comparison lines
};

[[nodiscard]] inline SevenF evaluate_7f(const Solved& s) {
    SevenF r;
    r.currents = currents_far_field(s, s.c.tag + ".C");
    r.plane = plane_field(s, post::HuygensOptions{});
    r.huygens = radiate(r.plane, s, s.c.tag + ".H");
    post::HuygensOptions z15;
    z15.z_h = 1.5 * kLambda;
    r.z15 = radiate(plane_field(s, z15), s, s.c.tag + ".Hz15");
    post::HuygensOptions sp;
    sp.spacing = kLambda / 6.0;
    r.spacing = radiate(plane_field(s, sp), s, s.c.tag + ".Hsp");
    r.m = pml_rough_ag::compare(r.huygens, r.currents);
    r.mz = pml_rough_ag::compare(r.z15, r.currents);
    r.ms = pml_rough_ag::compare(r.spacing, r.currents);
    r.mzh = pml_rough_ag::compare(r.z15, r.huygens);
    r.msh = pml_rough_ag::compare(r.spacing, r.huygens);
    r.lines.push_back(row_line(s, r.currents, r.huygens, r.plane));
    r.lines.push_back(pml_rough_ag::compare_line(r.huygens, r.currents, r.m));
    r.lines.push_back(pml_rough_ag::compare_line(r.z15, r.currents, r.mz));
    r.lines.push_back(pml_rough_ag::compare_line(r.spacing, r.currents, r.ms));
    r.lines.push_back(pml_rough_ag::compare_line(r.z15, r.huygens, r.mzh));
    r.lines.push_back(pml_rough_ag::compare_line(r.spacing, r.huygens, r.msh));
    return r;
}

/// One line per Huygens evaluation (plane geometry, cost, R).
[[nodiscard]] inline std::string plane_line(const std::string& tag,
                                            const post::HuygensPlaneField& pf, const FarField& f) {
    char buf[400];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "PLANE | %s | z %.5g m | spacing %.5g m | window %.5g/%.5g m | points %lld | plane %.2f s "
        "| R %.7f",
        tag.c_str(), pf.plane.z, pf.plane.spacing, pf.plane.window.inner_half,
        pf.plane.window.outer_half, static_cast<long long>(pf.plane.points.rows()), pf.seconds,
        f.reflectance());
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_huygens::plane_line: formatting failed or line truncated");
    return buf;
}

}  // namespace pml_huygens
