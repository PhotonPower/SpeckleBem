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
///    plus the PML ring, GMRES tol 1e-7, the 7h configuration) or densely;
///  * 6b (WP-PML6b, ADR 0010 decision 2026-10-25 item 4): the wide-gap plane (U with a 3 um gap,
///    wide_gap3_case(seed), and its L x 1.2 partner wide_gap3_patch_case(seed), both pml_start
///    5.375 um, 2N = 62 016) for two roughness seeds: kSeeds = {1, 2} (seed 1 is the surface of
///    WP-PML4b / WP-PML6; seed 2 the second, independent surface of the same statistics). The
///    window sensitivity of both far-field sources (flank x 0.7 / x 1.3 of the window (taper_end,
///    taper_end + 4 lambda) at a fixed inner edge, six_b_windows), 7d with the windowed currents
///    and the Huygens plane, over the hemisphere and in the cone theta <= 30 deg (cone_compare);
///    the acceptance gates are six_b_gates (window sensitivities <= 1e-3 in the cone, 7d >= 10 x
///    them).
///
/// Far fields (pml_rough_ag::FarField: |F|^2 on post::hemisphere_grid(-1, 90, 180) and the xz /
/// yz cuts, R = P_refl / P_inc): the windowed currents (Simulation::far_field_options with
/// FarFieldSource::Currents) and the Huygens plane (post::huygens_plane_field once per
/// z_h / spacing, radiated on the grids; other windows of the same grid by post::huygens_reweight,
/// bitwise the direct evaluation). The metric is pml_rough_ag::compare (ADR 0006 box-check L2 of
/// |F|^2 over the hemisphere; eps_rr on the cuts; dR = R_t / R_r - 1).
#include "specklebem/postprocessing/huygens.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_ring_support.hpp"
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

/// 7f premise control (benchmarks/pml_huygens.cpp --study 7f --cases ci-control; not an
/// acceptance case): the 7h CI geometry (pml_ring::flat_ag_ci_case(): patch 1 um, taper 0.25 um,
/// gap 0.5 um, W_pml = lambda, sigma_max 2; plane window (0.75, 1.25 um) from 1.5 w0, 2N =
/// 4 592, dark-band tolerance 5e-2) with Ag or Si; the reflected beam crosses the plane window.
[[nodiscard]] inline RoughCase flat_ci7h_case(const std::string& material) {
    RoughCase c = pml_ring::flat_ag_ci_case();
    c.tag = "flat_ci7h_" + material;
    c.material = material;
    return c;
}

/// 7f premise control: flat_ci7h_case("ag") with a 0.5 um taper (plane window (1.0, 1.5 um) from
/// 2 w0, 2N = 6 016; the scout between the 7h CI geometry and flat_ci_case()).
[[nodiscard]] inline RoughCase flat_ci7h_taper_case() {
    RoughCase c = flat_ci7h_case("ag");
    c.tag = "flat_ci7h_t05_ag";
    c.taper = 0.5e-6;
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

/// Roughness seeds of the wide-gap plane (WP-PML6b): seed 1 is the surface of every earlier
/// rough-Ag PML result (WP-PML4b, WP-PML6), seed 2 the second surface (same sigma, Lc, map
/// size and crop; geometry::generate_gaussian_height_map with seed 2).
inline constexpr std::array<std::uint64_t, 2> kSeeds = {1, 2};

/// Tag suffix of a seed: "" for seed 1 (the tags of the WP-PML6 record), "_s<seed>" otherwise.
[[nodiscard]] inline std::string seed_suffix(std::uint64_t seed) {
    return seed == 1 ? std::string() : "_s" + std::to_string(seed);
}

/// U with a 3 um gap (pml_start 5.375 um, 2N = 62 016; benchmarks/pml_huygens.cpp --cases ugap3
/// and --study 6b): the wide-gap plane of the diagnostic "extended plane" (WP-PML6) and of
/// WP-PML6b; `seed` selects the roughness (kSeeds; tag "ugap3" + seed_suffix(seed)).
[[nodiscard]] inline RoughCase wide_gap3_case(std::uint64_t seed = 1) {
    RoughCase c = pml_rough_ag::u_case();
    c.tag = "ugap3" + seed_suffix(seed);
    c.gap = 3.0e-6;
    c.seed = seed;
    return c;
}

/// The 7d partner of wide_gap3_case(seed) (L x 1.2 = 4.5 um, gap 2.625 um: the same pml_start
/// 5.375 um, so both share the plane window (2.375, 5.375 um); the same height map on the common
/// area; tag "l12gap3" + seed_suffix(seed)).
[[nodiscard]] inline RoughCase wide_gap3_patch_case(std::uint64_t seed = 1) {
    RoughCase c = pml_rough_ag::patch_case();
    c.tag = "l12gap3" + seed_suffix(seed);
    c.gap = 2.625e-6;
    c.seed = seed;
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
    if (solver == Solver::Fast)  // the label carries the (automatic) preconditioner
        r.solver += std::string("+") + to_string(r.sim->preconditioner_kind());
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

/// Window with its flank (outer_half - inner_half) scaled by a factor at a fixed inner edge.
/// @throws std::invalid_argument for a factor <= 0 (or not finite) or a window with
///         outer_half <= inner_half.
[[nodiscard]] inline post::CurrentWindow flank_scaled(const post::CurrentWindow& w, Real factor) {
    if (!(std::isfinite(factor) && factor > 0.0))
        throw std::invalid_argument("pml_huygens::flank_scaled: factor must be > 0");
    if (!(w.outer_half > w.inner_half))
        throw std::invalid_argument("pml_huygens::flank_scaled: window has no flank");
    return {w.inner_half, w.inner_half + factor * (w.outer_half - w.inner_half)};
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

// ---------------------------------------------------------------------------------------------
// WP-PML6b: wide-gap confirmation (ADR 0010, decision 2026-10-25, items 3-4)
// ---------------------------------------------------------------------------------------------

/// Cone views of a far-field comparison, test t against reference r (I = |F|^2 on the hemisphere
/// grid with the solid-angle weights of pml_rough_ag::compare; theta from the specular pole):
///   all        = the L2 metric over the hemisphere (pml_rough_ag::Metrics::l2),
///   cone       = the same over the rows with theta <= the cone angle (30 deg: the cone in which
///                the gap >= 6 lambda rule makes rough-metal far fields quantitative),
///   off        = the same over theta >= the cone angle (Metrics::l2_off30 at 30 deg),
///   band_60_75 / band_75_90 = power of t in the band over that of r, minus 1,
///   dR         = R_t / R_r - 1.
struct ConeMetrics {
    Real all = 0;
    Real cone = 0;
    Real off = 0;
    Real band_60_75 = 0;
    Real band_75_90 = 0;
    Real dR = 0;
};

/// sqrt(sum w (t - r)^2 / sum w r^2) over the rows with lo <= theta <= hi [rad].
/// @throws std::invalid_argument for different sizes or a reference that is zero on the rows.
[[nodiscard]] inline Real band_l2(const VectorXr& t, const VectorXr& r, const VectorXr& w,
                                  const VectorXr& theta, Real lo, Real hi) {
    if (t.size() != r.size() || w.size() != r.size() || theta.size() != r.size())
        throw std::invalid_argument("pml_huygens::band_l2: sizes differ");
    Real num = 0, den = 0;
    for (Index i = 0; i < r.size(); ++i) {
        if (theta(i) < lo || theta(i) > hi)
            continue;
        num += w(i) * (t(i) - r(i)) * (t(i) - r(i));
        den += w(i) * r(i) * r(i);
    }
    if (!(den > 0.0))
        throw std::invalid_argument("pml_huygens::band_l2: reference is zero on the band");
    return std::sqrt(num / den);
}

/// sum w t / sum w r - 1 over the rows with lo <= theta < hi [rad].
/// @throws std::invalid_argument for different sizes or a reference without power in the band.
[[nodiscard]] inline Real band_power_change(const VectorXr& t, const VectorXr& r, const VectorXr& w,
                                            const VectorXr& theta, Real lo, Real hi) {
    if (t.size() != r.size() || w.size() != r.size() || theta.size() != r.size())
        throw std::invalid_argument("pml_huygens::band_power_change: sizes differ");
    Real pt = 0, pr = 0;
    for (Index i = 0; i < r.size(); ++i) {
        if (theta(i) < lo || theta(i) >= hi)
            continue;
        pt += w(i) * t(i);
        pr += w(i) * r(i);
    }
    if (!(pr > 0.0))
        throw std::invalid_argument("pml_huygens::band_power_change: no reference power");
    return pt / pr - 1.0;
}

/// Cone views of t against r (see ConeMetrics); cone_deg in degrees.
/// @throws std::invalid_argument if the grids differ.
[[nodiscard]] inline ConeMetrics cone_compare(const FarField& t, const FarField& r,
                                              Real cone_deg = 30.0) {
    if (t.n_theta != r.n_theta || t.n_phi != r.n_phi || t.hemi.size() != r.hemi.size())
        throw std::invalid_argument("pml_huygens::cone_compare: far-field grids differ");
    const post::SphereGrid g = post::hemisphere_grid(-1.0, r.n_theta, r.n_phi);
    constexpr Real kRadPerDeg = constants::pi / 180.0;
    const Real cone = cone_deg * kRadPerDeg;
    const Real top = 90.0 * kRadPerDeg;
    ConeMetrics m;
    m.all = band_l2(t.hemi, r.hemi, g.weights, g.theta, 0.0, top);
    m.cone = band_l2(t.hemi, r.hemi, g.weights, g.theta, 0.0, cone);
    m.off = band_l2(t.hemi, r.hemi, g.weights, g.theta, cone, top);
    m.band_60_75 =
        band_power_change(t.hemi, r.hemi, g.weights, g.theta, 60.0 * kRadPerDeg, 75.0 * kRadPerDeg);
    m.band_75_90 =
        band_power_change(t.hemi, r.hemi, g.weights, g.theta, 75.0 * kRadPerDeg, 2.0 * top);
    m.dR = t.reflectance() / r.reflectance() - 1.0;
    return m;
}

/// One machine-readable CONE line (the format of the WP-PML6 record's CONE lines).
[[nodiscard]] inline std::string cone_line(const std::string& label, const FarField& t,
                                           const FarField& r, const ConeMetrics& m) {
    char buf[600];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "CONE | %-44s | %-12s vs %-12s | all %.4e | <=30 %.4e | >=30 %.4e | band 60-75 %+.3f | "
        "band 75-90 %+.3f | dR %+.3e",
        label.c_str(), t.tag.c_str(), r.tag.c_str(), m.all, m.cone, m.off, m.band_60_75,
        m.band_75_90, m.dR);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_huygens::cone_line: formatting failed or line truncated");
    return buf;
}

/// Flank of the 6b reference window: Delta = 4 lambda from the taper end (the converged flank of
/// the diagnostic "extended plane": z_h x 1.5 and Huygens vs currents <= 4.2e-4 at theta <= 30
/// deg), and the window factors x (1 -+ 0.3) applied to that flank at a fixed inner edge.
inline constexpr Real kSixBDelta = 4.0 * kLambda;
inline constexpr std::array<Real, 2> kSixBFactors = {0.7, 1.3};

/// The windows of a 6b far-field evaluation of one wide-gap case:
///   def              the case's default current window (taper_end, pml_start),
///   def_f07          def with its flank x 0.7 (x 1.3 would end beyond pml_start),
///   d4               (taper_end, taper_end + kSixBDelta): the reference of the window study,
///   d4_f07 / d4_f13  d4 with its flank x 0.7 / x 1.3 (inner edge fixed at the taper end),
///   abs / abs_d4     U's absolute windows (u_taper_end, pml_start) and (u_taper_end,
///                    u_taper_end + kSixBDelta) on a case whose taper end differs from U's (the
///                    L x 1.2 partner).
/// The 7g study scaled both half widths of the window; on the wide-gap plane x 0.7 would move the
/// inner edge into the patch (1.66 < 1.875 um) and x 1.3 the outer edge beyond pml_start, so 6b
/// scales the flank.
struct SixBWindows {
    post::CurrentWindow def{};
    post::CurrentWindow def_f07{};
    post::CurrentWindow d4{};
    post::CurrentWindow d4_f07{};
    post::CurrentWindow d4_f13{};
    std::optional<post::CurrentWindow> abs;
    std::optional<post::CurrentWindow> abs_d4;
};

/// The 6b windows for the default window `def` (taper_end, pml_start) of a case and U's taper end.
/// @throws std::invalid_argument if a window would end beyond pml_start (def.outer_half) (checked
///         before the solve by the callers: an invalid window must not cost a 62 016-unknown
///         solve) or U's taper end does not lie inside the case's default window.
[[nodiscard]] inline SixBWindows six_b_windows(const post::CurrentWindow& def, Real u_taper_end) {
    if (!(def.outer_half > def.inner_half && def.inner_half > 0.0))
        throw std::invalid_argument("pml_huygens::six_b_windows: invalid default window");
    SixBWindows w;
    w.def = def;
    w.def_f07 = flank_scaled(def, kSixBFactors[0]);
    w.d4 = {def.inner_half, def.inner_half + kSixBDelta};
    w.d4_f07 = flank_scaled(w.d4, kSixBFactors[0]);
    w.d4_f13 = flank_scaled(w.d4, kSixBFactors[1]);
    if (std::abs(u_taper_end - def.inner_half) > 1e-9 * def.inner_half) {
        if (!(u_taper_end > 0.0 && u_taper_end < def.outer_half))
            throw std::invalid_argument(
                "pml_huygens::six_b_windows: U's taper end lies outside the case's window");
        w.abs = post::CurrentWindow{u_taper_end, def.outer_half};
        w.abs_d4 = post::CurrentWindow{u_taper_end, u_taper_end + kSixBDelta};
    }
    for (const post::CurrentWindow& x :
         {w.d4, w.d4_f07, w.d4_f13, w.def_f07, w.abs_d4.value_or(w.d4)}) {
        if (!(x.outer_half <= def.outer_half))
            throw std::invalid_argument(
                "pml_huygens::six_b_windows: a window ends beyond pml_start (the gap is too "
                "narrow for the 6b window study)");
    }
    return w;
}

/// The far fields of one solved wide-gap case for 6b (tags <case>.<suffix>):
///   currents  C (default window), Cf07, CD4, CD4f07, CD4f13 and, on the L x 1.2 partner, Cabs,
///             CabsD4 (SixBWindows);
///   Huygens   H (the default plane: z_h = lambda, spacing lambda / 4, window = the current
///             window), HD4, HD4f07, HD4f13 (the same plane, reweighted), Hz15D4 (z_h x 1.5 at
///             d4) and, on the partner, HabsD4.
/// `lines` holds the ROW line of the case and one PLANE line per Huygens far field.
struct SixBFields {
    std::vector<FarField> far;
    std::vector<std::string> lines;
};

[[nodiscard]] inline SixBFields six_b_far_fields(const Solved& s, Real u_taper_end) {
    const SixBWindows w = six_b_windows(*s.sim->far_field_window(), u_taper_end);
    const std::string& t = s.c.tag;
    SixBFields r;
    const auto add_c = [&](const std::string& sfx, const post::CurrentWindow& win) {
        r.far.push_back(currents_far_field(s, t + "." + sfx, win));
    };
    const auto add_h = [&](const std::string& sfx, const post::HuygensPlaneField& pf) {
        r.far.push_back(radiate(pf, s, t + "." + sfx));
        r.lines.push_back(plane_line(r.far.back().tag, pf, r.far.back()));
    };
    add_c("C", w.def);
    const post::HuygensPlaneField pf = plane_field(s, post::HuygensOptions{});
    add_h("H", pf);
    r.lines.insert(r.lines.begin(), row_line(s, r.far[0], r.far[1], pf));
    add_c("Cf07", w.def_f07);
    add_c("CD4", w.d4);
    add_c("CD4f07", w.d4_f07);
    add_c("CD4f13", w.d4_f13);
    add_h("HD4", post::huygens_reweight(pf, w.d4));
    add_h("HD4f07", post::huygens_reweight(pf, w.d4_f07));
    add_h("HD4f13", post::huygens_reweight(pf, w.d4_f13));
    post::HuygensOptions z15;
    z15.z_h = 1.5 * kLambda;
    add_h("Hz15D4", post::huygens_reweight(plane_field(s, z15), w.d4));
    if (w.abs) {
        add_c("Cabs", *w.abs);
        add_c("CabsD4", *w.abs_d4);
        add_h("HabsD4", post::huygens_reweight(pf, *w.abs_d4));
    }
    return r;
}

/// One far-field comparison of the 6b record: test against reference, by tag.
struct Comparison {
    std::string label;
    std::string test;
    std::string ref;
};

/// The comparisons of one wide-gap case `t` (window sensitivities, plane vs currents, z_h x 1.5).
[[nodiscard]] inline std::vector<Comparison> six_b_case_comparisons(const std::string& t) {
    return {
        {"6b currents flank x0.7 (Delta 4 lambda)", t + ".CD4f07", t + ".CD4"},
        {"6b currents flank x1.3 (Delta 4 lambda)", t + ".CD4f13", t + ".CD4"},
        {"6b huygens flank x0.7 (Delta 4 lambda)", t + ".HD4f07", t + ".HD4"},
        {"6b huygens flank x1.3 (Delta 4 lambda)", t + ".HD4f13", t + ".HD4"},
        {"6b currents default window flank x0.7", t + ".Cf07", t + ".C"},
        {"6b currents Delta 4 lambda vs default window", t + ".CD4", t + ".C"},
        {"6b huygens vs currents (Delta 4 lambda)", t + ".HD4", t + ".CD4"},
        {"6b huygens vs currents (default windows)", t + ".H", t + ".C"},
        {"6b z_h x1.5 (Delta 4 lambda)", t + ".Hz15D4", t + ".HD4"},
    };
}

/// The 7d comparisons of a wide-gap pair: L x 1.2 `l` against U `u` (same seed).
[[nodiscard]] inline std::vector<Comparison> six_b_pair_comparisons(const std::string& u,
                                                                    const std::string& l) {
    return {
        {"6b 7d currents, default windows", l + ".C", u + ".C"},
        {"6b 7d currents, U absolute window", l + ".Cabs", u + ".C"},
        {"6b 7d currents, Delta 4 lambda own windows", l + ".CD4", u + ".CD4"},
        {"6b 7d currents, Delta 4 lambda U window", l + ".CabsD4", u + ".CD4"},
        {"6b 7d huygens, Delta 4 lambda own windows", l + ".HD4", u + ".HD4"},
        {"6b 7d huygens, Delta 4 lambda U window", l + ".HabsD4", u + ".HD4"},
    };
}

/// Acceptance of WP-PML6b (ADR 0010 decision 2026-10-25 item 4) for one seed, cone theta <= 30
/// deg: the window sensitivities (flank x 0.7 / x 1.3 of d4, max over U and L x 1.2) of both
/// sources <= kSixBSensitivity, and 7d (currents with the default and the d4 windows, Huygens with
/// d4; own windows) >= kSixBRatio x the sensitivity of the same source.
inline constexpr Real kSixBSensitivity = 1e-3;
inline constexpr Real kSixBRatio = 10.0;

struct SixBGates {
    Real sens_currents = 0;
    Real sens_huygens = 0;
    Real d7_currents = 0;     ///< default windows
    Real d7_currents_d4 = 0;  ///< d4 windows
    Real d7_huygens = 0;      ///< d4 windows
    [[nodiscard]] bool sensitivity_met() const {
        return sens_currents <= kSixBSensitivity && sens_huygens <= kSixBSensitivity;
    }
    [[nodiscard]] bool ratio_met() const {
        return d7_currents >= kSixBRatio * sens_currents &&
               d7_currents_d4 >= kSixBRatio * sens_currents &&
               d7_huygens >= kSixBRatio * sens_huygens;
    }
};

/// The gates of the pair (u, l) from far fields by tag (six_b_far_fields of both cases).
/// @throws std::invalid_argument if a far field is missing.
[[nodiscard]] inline SixBGates six_b_gates(const std::map<std::string, FarField>& far,
                                           const std::string& u, const std::string& l) {
    const auto at = [&](const std::string& tag) -> const FarField& {
        const auto it = far.find(tag);
        if (it == far.end())
            throw std::invalid_argument("pml_huygens::six_b_gates: missing far field " + tag);
        return it->second;
    };
    const auto cone = [&](const std::string& t, const std::string& r) {
        return cone_compare(at(t), at(r)).cone;
    };
    SixBGates g;
    for (const std::string& t : {u, l}) {
        for (const char* f : {".CD4f07", ".CD4f13"})
            g.sens_currents = std::max(g.sens_currents, cone(t + f, t + ".CD4"));
        for (const char* f : {".HD4f07", ".HD4f13"})
            g.sens_huygens = std::max(g.sens_huygens, cone(t + f, t + ".HD4"));
    }
    g.d7_currents = cone(l + ".C", u + ".C");
    g.d7_currents_d4 = cone(l + ".CD4", u + ".CD4");
    g.d7_huygens = cone(l + ".HD4", u + ".HD4");
    return g;
}

/// One machine-readable GATE line of a seed.
[[nodiscard]] inline std::string gate_line(std::uint64_t seed, const SixBGates& g) {
    char buf[600];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "GATE | 6b | seed %llu | theta <= 30 deg | window sensitivity currents %.3e huygens %.3e "
        "(<= %.0e: %s) | 7d currents %.3e (Delta 4 lambda %.3e) huygens %.3e | ratio currents "
        "%.1f / %.1f huygens %.1f (>= %.0f: %s)",
        static_cast<unsigned long long>(seed), g.sens_currents, g.sens_huygens, kSixBSensitivity,
        g.sensitivity_met() ? "met" : "NOT MET", g.d7_currents, g.d7_currents_d4, g.d7_huygens,
        g.d7_currents / g.sens_currents, g.d7_currents_d4 / g.sens_currents,
        g.d7_huygens / g.sens_huygens, kSixBRatio, g.ratio_met() ? "met" : "NOT MET");
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_huygens::gate_line: formatting failed or line truncated");
    return buf;
}

}  // namespace pml_huygens
