#pragma once
/// @file pml_rough_ag_support.hpp
/// Validations 7c and 7d of the open-surface PML route on rough Ag (ADR 0009 decision 7, WP-PML4b;
/// docs/05 "PML route", rows 7c and 7d): the far field of a Gaussian rough Ag patch in a polished
/// (flat) Ag surface with absorbing surroundings must change by < 0.1 % when W_pml or sigma_max
/// is scaled by 1.5 (7c) and by < 0.5 % when the patch edge L grows by 20 % at fixed waist (7d).
/// Shared by the study executable benchmarks/pml_rough_ag.cpp, the validation-large test
/// tests/validation_large/test_pml_rough_ag_large.cpp and the unit test
/// tests/unit/test_pml_rough_ag.cpp; no Catch2.
///
/// Surface. A Gaussian height map (geometry::generate_gaussian_height_map, sigma, Lc, seed, grid
/// spacing h) is generated on the edge map_patch + 2 W_t and cropped centrally to patch + 2 W_t,
/// so that runs with different L (7d) share the surface on the common area (as the box study,
/// benchmarks/box_validity.cpp); geometry::make_open_surface_mesh tapers it to 0 across W_t >= Lc
/// and continues it flat through the gap and the PML. Vacuum R1 over Ag R2 (eps_r = -9.794 -
/// 0.313j at 500 nm), the object occupies z > xi, light comes from z < 0 travelling in +z. The 7d
/// control uses the same geometry and height map with Si (eps_r = 18.478 - 0.606j, no surface
/// plasmons; si_u_case(), si_patch_case()), which separates the termination from the SPP currents
/// in the gap of the rough Ag patch (benchmarks/results/pml_rough_ag.md; ADR 0010).
///
/// Excitation and solve. The rigorous angular-spectrum beam (focus at the origin, normal
/// incidence, E along x, waist w0), Simulation's PML constructor (its dark-band, taper and
/// controlled-region checks apply), dense operator, LU in place (pml_spp::InPlaceLu: LAPACK zgetrf
/// when available, else Eigen; peak = one matrix, 16 (2N)^2 bytes).
///
/// Far field. |F|^2 of the windowed currents (Simulation::far_field_options(): raised-cosine
/// window inner = taper_end, outer = pml_start, or an explicit window) on the reflection
/// hemisphere post::hemisphere_grid(-1, n_theta, n_phi) (Gauss-Legendre in theta, solid-angle
/// weights) and on the xz and yz cuts of box_validity::cut_directions(). The reflectance R =
/// P_refl / P_inc with P_refl = (1 / 2 eta_1) sum_i w_i |F_i|^2 (what Simulation::reflected_power
/// evaluates on this grid) and P_inc = AngularSpectrumBeam::power().
///
/// Metric (ADR 0006 amendment / docs/05 box checks; test T against reference R, I = |F|^2):
///   l2       = sqrt(sum w (I_T - I_R)^2 / sum w I_R^2) over the hemisphere grid, w the grid's
///              solid-angle weights (Gauss-Legendre weight x sin(theta) x dphi: the w = sin(theta)
///              metric of the box study on a different quadrature of the same integral),
///   l2_off30 = the same over the rows with theta >= 30 deg (off-specular part),
///   eps_xz/yz = docs/05 eps_rr = sqrt(mean (I_T - I_R)^2) / max I_R on the cuts,
///   dR       = R_T / R_R - 1.
#include "specklebem/core/path.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/postprocessing/scattering.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "box_validity_support.hpp"
#include "pml_spp_support.hpp"

namespace pml_rough_ag {

using namespace specklebem;
using fresnel_flat::kDeg;
using fresnel_flat::kLambda;

/// One run: surface, zones, PML, beam, kernels, far-field evaluation.
struct RoughCase {
    std::string tag = "u";
    Real patch = 3.75e-6;     ///< L [m]; L / 2 a multiple of h
    Real map_patch = 4.5e-6;  ///< L_gen >= L [m]: the map is generated on L_gen + 2 W_t
    Real taper = 0.5e-6;      ///< W_t >= Lc [m]
    Real gap = 0.5e-6;        ///< W_g [m] (the default far-field window spans it)
    Real pml_width = 1.0e-6;  ///< W_pml [m] (2 lambda)
    Real mesh_size = 125e-9;  ///< h [m] (lambda / 4, the Ag rule of 7a)
    Real sigma_max = 4.0;     ///< PmlTermination::sigma_max
    Real order = 2.0;         ///< PmlTermination::order
    Real dark_band_tolerance = 1e-4;
    Real target_accuracy = 1e-5;         ///< kernels::OperatorOptions::target_accuracy
    int stretched_touching_points = 10;  ///< kernels::OperatorOptions::stretched_touching_points
    Real waist = 0.5e-6;                 ///< w0 [m] (lambda: the normal-incidence rule)
    Real rms = 50e-9;                    ///< sigma [m]
    Real correlation_length = 500e-9;    ///< Lc [m]
    std::uint64_t seed = 1;
    std::string material = "ag";  ///< "ag" (silver_500nm) or "si" (silicon_500nm, 7d control)
    int n_theta = 90;             ///< hemisphere grid (Gauss-Legendre in theta)
    int n_phi = 180;              ///< azimuths (trapezoid)
    int ff_degree = 6;            ///< Dunavant degree of the far-field source quadrature
    /// Windows evaluated in addition to Simulation's default (taper_end, pml_start); each must
    /// end at or before pml_start.
    std::vector<post::CurrentWindow> extra_windows;
    /// Diagnostic only: Im(eps_r) of the object instead of the tabulated value (Ag: -0.313; a more
    /// lossy Ag shortens the SPP decay length, as the box study's loss diagnostic); not a
    /// validated material.
    std::optional<Real> eps_imag;
    bool use_lapack = true;
};

/// The unperturbed run U of 7c / 7d (benchmarks/results/pml_rough_ag.md): sigma = 50 nm, Lc =
/// 500 nm, seed 1 (map generated on 4.5 + 1 um and cropped), L = 3.75 um, W_t = W_g = 0.5 um,
/// W_pml = 2 lambda, sigma_max 4, m 2, h = lambda / 4, w0 = lambda (w0 = L / 7.5), dark band 1e-4.
[[nodiscard]] inline RoughCase u_case() {
    return RoughCase{};
}

/// 7c: W_pml x 1.5 (1.5 um).
[[nodiscard]] inline RoughCase pml_width_case() {
    RoughCase c = u_case();
    c.tag = "w15";
    c.pml_width = 1.5 * c.pml_width;
    return c;
}

/// 7c: sigma_max x 1.5 (6).
[[nodiscard]] inline RoughCase sigma_case() {
    RoughCase c = u_case();
    c.tag = "s15";
    c.sigma_max = 1.5 * c.sigma_max;
    return c;
}

/// 7d: L x 1.2 (4.5 um) at fixed waist, the same map (cropped less), every zone width fixed.
[[nodiscard]] inline RoughCase patch_case() {
    RoughCase c = u_case();
    c.tag = "l12";
    c.patch = 1.2 * c.patch;
    return c;
}

/// 7d control on rough Si (no SPPs): U with Si, h = lambda / 4 (7a: Si converges at lambda / 4 at
/// normal incidence), the same map, zones, PML and beam.
[[nodiscard]] inline RoughCase si_u_case() {
    RoughCase c = u_case();
    c.tag = "si_u";
    c.material = "si";
    return c;
}

/// 7d control on rough Si: L x 1.2 (4.5 um) at fixed waist.
[[nodiscard]] inline RoughCase si_patch_case() {
    RoughCase c = patch_case();
    c.tag = "si_l12";
    c.material = "si";
    return c;
}

/// Object material of a case. @throws std::invalid_argument for an unknown name.
[[nodiscard]] inline material::Material object_material(const std::string& name) {
    if (name == "ag")
        return material::silver_500nm();
    if (name == "si")
        return material::silicon_500nm();
    throw std::invalid_argument("pml_rough_ag: material must be \"ag\" or \"si\", got \"" + name +
                                "\"");
}

/// Central crop of a height map to the edge `edge` (same spacing).
/// @throws std::invalid_argument if the map is smaller, the edge is not a whole number of
///         cells, or the cell difference is odd (no central crop on the grid).
[[nodiscard]] inline geometry::HeightMap crop_map(const geometry::HeightMap& m, Real edge) {
    if (m.dx != m.dy)
        throw std::invalid_argument("pml_rough_ag::crop_map: dx != dy");
    const Real cells_f = edge / m.dx;
    const auto cells = static_cast<Index>(std::llround(cells_f));
    if (std::abs(cells_f - static_cast<Real>(cells)) > 1e-6)
        throw std::invalid_argument("pml_rough_ag::crop_map: edge is not a whole number of cells");
    const Index n = cells + 1;
    const Index diff = m.z.rows() - n;
    if (m.z.rows() != m.z.cols() || diff < 0 || diff % 2 != 0)
        throw std::invalid_argument(
            "pml_rough_ag::crop_map: the map must be square, at least as large and differ by "
            "an even number of cells");
    geometry::HeightMap out;
    out.dx = m.dx;
    out.dy = m.dy;
    out.z = m.z.block(diff / 2, diff / 2, n, n);
    return out;
}

/// The height map of a case: generated on map_patch + 2 W_t, cropped to patch + 2 W_t.
[[nodiscard]] inline geometry::HeightMap height_map(const RoughCase& c) {
    if (!(c.map_patch >= c.patch))
        throw std::invalid_argument("pml_rough_ag: map_patch must be >= patch");
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = c.map_patch + 2.0 * c.taper;
    rp.rms_roughness = c.rms;
    rp.correlation_length = c.correlation_length;
    rp.mesh_size = c.mesh_size;
    rp.seed = c.seed;
    if (c.rms == 0.0)  // flat control (same grid and zones)
        return fresnel_flat::flat_height_map(c.patch + 2.0 * c.taper, c.mesh_size);
    return crop_map(geometry::generate_gaussian_height_map(rp), c.patch + 2.0 * c.taper);
}

/// Open mesh of a case; the zones must be the requested ones (no grid rounding).
[[nodiscard]] inline geometry::OpenSurfaceMesh make_open_mesh(const RoughCase& c) {
    geometry::OpenSurfaceParams op;
    op.patch_size = c.patch;
    op.taper_width = c.taper;
    op.gap_width = c.gap;
    op.pml_width = c.pml_width;
    geometry::OpenSurfaceMesh m = geometry::make_open_surface_mesh(height_map(c), op);
    const Real tol = 1e-6 * c.mesh_size;
    if (std::abs(m.patch_half - 0.5 * c.patch) > tol ||
        std::abs(m.taper_end - m.patch_half - c.taper) > tol ||
        std::abs(m.pml_start - m.taper_end - c.gap) > tol ||
        std::abs(m.outer_half - m.pml_start - c.pml_width) > tol)
        throw std::invalid_argument(
            "pml_rough_ag: a zone is not on the mesh grid (h must divide L / 2, W_t, W_g and "
            "W_pml)");
    return m;
}

[[nodiscard]] inline std::shared_ptr<excitation::AngularSpectrumBeam> make_beam(const RoughCase& c,
                                                                                Real pml_start) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.waist;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = 0.0;
    bp.polarization = excitation::Polarization::P;  // E along x at normal incidence
    bp.region_radius = 1.02 * std::sqrt(2.0) * pml_start;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());
}

[[nodiscard]] inline SimulationConfig make_config(const RoughCase& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = object_material(c.material);
    if (c.eps_imag) {
        if (!(*c.eps_imag < 0.0))
            throw std::invalid_argument("pml_rough_ag: eps_imag must be < 0 (passive, exp(+jwt))");
        cfg.object.eps_r = Complex(cfg.object.eps_r.real(), *c.eps_imag);
    }
    cfg.compression = "dense";
    cfg.solver = SolverKind::Direct;
    cfg.kernels.target_accuracy = c.target_accuracy;
    cfg.kernels.stretched_touching_points = c.stretched_touching_points;
    return cfg;
}

[[nodiscard]] inline PmlTermination make_termination(const RoughCase& c) {
    PmlTermination p;
    p.sigma_max = c.sigma_max;
    p.order = c.order;
    p.correlation_length = c.correlation_length;
    p.dark_band_tolerance = c.dark_band_tolerance;
    return p;
}

/// Mesh size of a case without assembly.
struct Size {
    Index triangles = 0;
    Index unknowns = 0;          ///< 2N
    std::array<Real, 4> half{};  ///< a_0 .. a_3 [m]
    Real dense_bytes = 0;        ///< 16 (2N)^2: the peak with the in-place LU
};

[[nodiscard]] inline Size size_of(const RoughCase& c) {
    const geometry::OpenSurfaceMesh om = make_open_mesh(c);
    Size s;
    s.triangles = om.mesh.num_triangles();
    s.unknowns = 2 * om.mesh.num_edges();
    s.half = {om.patch_half, om.taper_end, om.pml_start, om.outer_half};
    s.dense_bytes = 16.0 * static_cast<Real>(s.unknowns) * static_cast<Real>(s.unknowns);
    return s;
}

/// Dark-band ratio and masked test functions of a case, measured by the PML constructor with
/// the tolerance relaxed (no assembly; the other checks apply).
struct Estimate {
    Size size;
    Real dark_band = 0;
    Index masked = 0;
    Real patch_rms = 0;  ///< rms of the cropped map [m]
};

[[nodiscard]] inline Estimate estimate(const RoughCase& c) {
    Estimate e;
    e.size = size_of(c);
    RoughCase relaxed = c;
    relaxed.dark_band_tolerance = 1e6;
    const Simulation sim(make_open_mesh(relaxed), make_termination(relaxed),
                         make_beam(relaxed, e.size.half[2]), make_config(relaxed));
    e.dark_band = sim.dark_band_ratio();
    const std::vector<std::uint8_t> mask =
        op::pml_masked_functions(*sim.problem().space, sim.stretch());
    e.masked = static_cast<Index>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
    e.patch_rms = height_map(c).rms();
    return e;
}

// ---------------------------------------------------------------------------------------------
// Far fields and the metric
// ---------------------------------------------------------------------------------------------

/// Far-field intensities of one run and one window.
struct FarField {
    std::string tag;
    post::CurrentWindow window{};
    int n_theta = 0;
    int n_phi = 0;
    VectorXr hemi;    ///< |F|^2 [V^2] per row of post::hemisphere_grid(-1, n_theta, n_phi)
    VectorXr cut_xz;  ///< |F|^2 on box_validity::cut_directions(false)
    VectorXr cut_yz;  ///< |F|^2 on box_validity::cut_directions(true)
    Real power = 0;   ///< P_inc [W]
    Real eta1 = 0;    ///< wave impedance of R1 [Ohm]
    Real p_refl = 0;  ///< (1 / 2 eta_1) sum w |F|^2 [W]
    [[nodiscard]] Real reflectance() const { return p_refl / power; }
};

/// Metrics of test t against reference r (see the file comment).
struct Metrics {
    Real l2 = 0;
    Real l2_off30 = 0;
    Real eps_xz = 0;
    Real eps_yz = 0;
    Real dR = 0;
};

/// sqrt(sum w (t - r)^2 / sum w r^2) over the rows with theta >= theta_min [rad].
/// @throws std::invalid_argument for different sizes or a reference that is zero on the rows.
[[nodiscard]] inline Real weighted_l2(const VectorXr& t, const VectorXr& r, const VectorXr& w,
                                      const VectorXr& theta, Real theta_min = 0.0) {
    if (t.size() != r.size() || w.size() != r.size() || theta.size() != r.size())
        throw std::invalid_argument("pml_rough_ag::weighted_l2: sizes differ");
    Real num = 0, den = 0;
    for (Index i = 0; i < r.size(); ++i) {
        if (theta(i) < theta_min)
            continue;
        num += w(i) * (t(i) - r(i)) * (t(i) - r(i));
        den += w(i) * r(i) * r(i);
    }
    if (!(den > 0.0))
        throw std::invalid_argument("pml_rough_ag::weighted_l2: reference is zero");
    return std::sqrt(num / den);
}

/// docs/05 eps_rr = sqrt(mean (t - r)^2) / max r.
/// @throws std::invalid_argument for different or zero sizes or max r <= 0.
[[nodiscard]] inline Real eps_rr(const VectorXr& t, const VectorXr& r) {
    if (t.size() != r.size() || r.size() == 0)
        throw std::invalid_argument("pml_rough_ag::eps_rr: sizes differ");
    if (!(r.maxCoeff() > 0.0))
        throw std::invalid_argument("pml_rough_ag::eps_rr: reference is zero");
    return std::sqrt((t - r).squaredNorm() / static_cast<Real>(r.size())) / r.maxCoeff();
}

/// All metrics of t against r on the grid both were evaluated on.
/// @throws std::invalid_argument if the grids differ.
[[nodiscard]] inline Metrics compare(const FarField& t, const FarField& r) {
    if (t.n_theta != r.n_theta || t.n_phi != r.n_phi || t.hemi.size() != r.hemi.size() ||
        t.cut_xz.size() != r.cut_xz.size() || t.cut_yz.size() != r.cut_yz.size())
        throw std::invalid_argument("pml_rough_ag::compare: far-field grids differ");
    const post::SphereGrid g = post::hemisphere_grid(-1.0, r.n_theta, r.n_phi);
    Metrics m;
    m.l2 = weighted_l2(t.hemi, r.hemi, g.weights, g.theta);
    m.l2_off30 = weighted_l2(t.hemi, r.hemi, g.weights, g.theta, 30.0 * kDeg);
    m.eps_xz = eps_rr(t.cut_xz, r.cut_xz);
    m.eps_yz = eps_rr(t.cut_yz, r.cut_yz);
    m.dR = t.reflectance() / r.reflectance() - 1.0;
    return m;
}

/// Writes a far field as text ("%.17g", reproducible bitwise by read_far_field()).
/// @throws std::runtime_error if the file cannot be written.
inline void write_far_field(const std::string& path, const FarField& f) {
    std::ofstream out(core::path_from_utf8(path), std::ios::binary);
    if (!out)
        throw std::runtime_error("pml_rough_ag: cannot write " + path);
    char buf[128];
    const auto num = [&](Real v) {
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return std::string(buf);
    };
    out << "pml_rough_ag_far_field 1\n";
    out << "tag " << (f.tag.empty() ? "-" : f.tag) << "\n";
    out << "window " << num(f.window.inner_half) << " " << num(f.window.outer_half) << "\n";
    out << "grid " << f.n_theta << " " << f.n_phi << "\n";
    out << "power " << num(f.power) << "\neta1 " << num(f.eta1) << "\np_refl " << num(f.p_refl)
        << "\n";
    const std::array<std::pair<const char*, const VectorXr*>, 3> blocks = {
        {{"hemi", &f.hemi}, {"cut_xz", &f.cut_xz}, {"cut_yz", &f.cut_yz}}};
    for (const auto& [name, v] : blocks) {
        out << name << " " << v->size() << "\n";
        for (Index i = 0; i < v->size(); ++i) out << num((*v)(i)) << "\n";
    }
    if (!out)
        throw std::runtime_error("pml_rough_ag: write to " + path + " failed");
}

/// Reads a file of write_far_field().
/// @throws std::runtime_error if the file cannot be opened or is malformed.
[[nodiscard]] inline FarField read_far_field(const std::string& path) {
    std::ifstream in(core::path_from_utf8(path), std::ios::binary);
    if (!in)
        throw std::runtime_error("pml_rough_ag: cannot open " + path);
    const auto fail = [&](const std::string& what) {
        return std::runtime_error("pml_rough_ag: malformed far-field file " + path + " (" + what +
                                  ")");
    };
    const auto expect = [&](const char* key) {
        std::string k;
        if (!(in >> k) || k != key)
            throw fail(key);
    };
    FarField f;
    int version = 0;
    expect("pml_rough_ag_far_field");
    if (!(in >> version) || version != 1)
        throw fail("version");
    expect("tag");
    in >> f.tag;
    expect("window");
    in >> f.window.inner_half >> f.window.outer_half;
    expect("grid");
    in >> f.n_theta >> f.n_phi;
    expect("power");
    in >> f.power;
    expect("eta1");
    in >> f.eta1;
    expect("p_refl");
    in >> f.p_refl;
    const std::array<std::pair<const char*, VectorXr*>, 3> blocks = {
        {{"hemi", &f.hemi}, {"cut_xz", &f.cut_xz}, {"cut_yz", &f.cut_yz}}};
    for (const auto& [name, v] : blocks) {
        expect(name);
        Index n = -1;
        if (!(in >> n) || n < 0)
            throw fail(name);
        v->resize(n);
        for (Index i = 0; i < n; ++i)
            if (!(in >> (*v)(i)))
                throw fail(name);
    }
    if (!in)
        throw fail("read");
    return f;
}

// ---------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------

/// Absolute DRC D = |F|^2 / (2 eta_1 P_inc) [1/sr] on the xz cut at theta [deg] (a cut node,
/// -89.5 .. 89.5 in 0.5 deg steps; negative theta: -x side).
/// @throws std::invalid_argument for theta not on the cut grid.
[[nodiscard]] inline Real drc_xz(const FarField& f, Real theta_deg) {
    const Real m = (theta_deg + 89.5) / box_validity::kCutStep;
    const auto i = static_cast<Index>(std::llround(m));
    if (std::abs(m - static_cast<Real>(i)) > 1e-9 || i < 0 || i >= f.cut_xz.size())
        throw std::invalid_argument("pml_rough_ag::drc_xz: theta not on the cut grid");
    return f.cut_xz(i) / (2.0 * f.eta1 * f.power);
}

/// Area-weighted rms of |J| and |M| (centroid values) over the triangles of one zone, relative
/// to the same over the Patch zone: the current level that the far-field window truncates in
/// the gap (surface waves on rough Ag; only the reflected beam tail on a flat surface).
struct ZoneCurrents {
    Real gap_j = 0;
    Real gap_m = 0;
};

[[nodiscard]] inline ZoneCurrents zone_currents(const Simulation& sim, const VectorXc& x) {
    const basis::RwgSpace& space = *sim.problem().space;
    const geometry::TriangleMesh& mesh = space.mesh();
    std::array<Real, 2> pj{}, pm{}, area{};
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const geometry::SurfaceZone z = sim.zones()[static_cast<std::size_t>(t)];
        const int k = z == geometry::SurfaceZone::Patch ? 0
                      : z == geometry::SurfaceZone::Gap ? 1
                                                        : -1;
        if (k < 0)
            continue;
        const Vec3 c = mesh.centroid(t);
        const Real a = mesh.area(t);
        pj[static_cast<std::size_t>(k)] += a * pml_spp::current_at(space, x, 0, t, c).squaredNorm();
        pm[static_cast<std::size_t>(k)] += a * pml_spp::current_at(space, x, 1, t, c).squaredNorm();
        area[static_cast<std::size_t>(k)] += a;
    }
    if (!(area[0] > 0.0 && area[1] > 0.0 && pj[0] > 0.0 && pm[0] > 0.0))
        throw std::logic_error("pml_rough_ag::zone_currents: empty patch or gap");
    return {std::sqrt(pj[1] / area[1] / (pj[0] / area[0])),
            std::sqrt(pm[1] / area[1] / (pm[0] / area[0]))};
}

/// Result of one run.
struct RunResult {
    Estimate est;
    std::string formulation;
    bool lapack = false;
    Real setup_s = 0;
    Real assembly_s = 0;  ///< operator + right-hand side
    Real lu_s = 0;
    Real solve_s = 0;
    Real post_s = 0;
    Real peak_rss = 0;
    Real residual = 0;  ///< |b - Z x| / |b| on 64 sampled rows
    pml_fresnel::EdgeDecay decay;
    ZoneCurrents gap;           ///< current level in the gap relative to the patch
    std::vector<FarField> far;  ///< default window first, then c.extra_windows
    std::string report;
};

/// Far field of a solution for one window.
[[nodiscard]] inline FarField evaluate_far_field(const Simulation& sim,
                                                 const post::SurfaceSolution& s, const RoughCase& c,
                                                 const post::CurrentWindow& window, Real power) {
    FarField f;
    f.tag = c.tag;
    f.window = window;
    f.n_theta = c.n_theta;
    f.n_phi = c.n_phi;
    f.power = power;
    f.eta1 = material::vacuum().wave_impedance(sim.problem().omega).real();
    post::FieldOptions fo = sim.far_field_options(c.ff_degree);
    fo.window = window;
    const post::SphereGrid g = post::hemisphere_grid(-1.0, c.n_theta, c.n_phi);
    box_validity::FieldMatrix F;
    post::far_field(s, g.dirs, F, fo);
    f.hemi = F.rowwise().squaredNorm();
    f.p_refl = g.weights.dot(f.hemi) / (2.0 * f.eta1);
    post::far_field(s, box_validity::cut_directions(false), F, fo);
    f.cut_xz = F.rowwise().squaredNorm();
    post::far_field(s, box_validity::cut_directions(true), F, fo);
    f.cut_yz = F.rowwise().squaredNorm();
    return f;
}

/// Builds, assembles, factorises and solves a case and evaluates its far fields.
inline RunResult run(const RoughCase& c) {
    using Clock = std::chrono::steady_clock;
    const auto secs = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<Real>(b - a).count();
    };
    RunResult r;
    const auto t0 = Clock::now();
    r.est = estimate(c);
    const auto beam = make_beam(c, r.est.size.half[2]);
    const Simulation sim(make_open_mesh(c), make_termination(c), beam, make_config(c));
    for (const post::CurrentWindow& w : c.extra_windows)
        if (!(w.outer_half <= r.est.size.half[2] && w.inner_half < w.outer_half))
            throw std::invalid_argument(
                "pml_rough_ag: an extra window must end at or before pml_start (strictly, as "
                "post::far_field checks it; checked before the assembly)");
    r.formulation = formulation::make_formulation(sim.formulation_kind())->name();
    const auto t1 = Clock::now();
    r.setup_s = secs(t0, t1);
    const VectorXc b = op::assemble_rhs(sim.problem());
    pml_spp::RunResult lu;
    std::vector<VectorXc> x = pml_spp::detail::solve_dense(sim, {b}, c.use_lapack, lu);
    r.assembly_s = secs(t1, Clock::now()) - lu.lu_s - lu.solve_s;
    r.lu_s = lu.lu_s;
    r.solve_s = lu.solve_s;
    r.residual = lu.residual;
    r.lapack = lu.lapack;
    const auto t4 = Clock::now();
    const post::SurfaceSolution s{&sim.problem(), x.front()};
    const Real power = beam->power();
    r.far.push_back(evaluate_far_field(sim, s, c, *sim.far_field_window(), power));
    for (const post::CurrentWindow& w : c.extra_windows)
        r.far.push_back(evaluate_far_field(sim, s, c, w, power));
    r.decay = pml_fresnel::edge_decay(s, r.est.size.half[3], c.mesh_size);
    r.gap = zone_currents(sim, x.front());
    r.post_s = secs(t4, Clock::now());
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// One machine-readable line per run (default window).
[[nodiscard]] inline std::string row_line(const RoughCase& c, const RunResult& r) {
    const FarField& f = r.far.front();
    char buf[2000];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "ROW | rough_ag | %s | %s | L %.4g | Lgen %.4g | taper %.4g | gap %.4g | pml %.4g | h %.4g "
        "| "
        "sigma %.3g | m %.3g | acc %.1e | stp %d | w0 %.4g | rms %.3g (map %.4g) | Lc %.3g | seed "
        "%llu | half %.4g/%.4g/%.4g/%.4g | window %.4g/%.4g | 2N %lld | masked %lld of %lld | "
        "dark %.3e (tol %.1e) | R %.7f | P_refl %.7e | P_inc %.7e | edge J %.2e M %.2e | gap rms J "
        "%.3e M %.3e | res "
        "%.1e | %s | setup %.1f s | assembly %.1f s | LU %.1f s | solve %.1f s | post %.1f s | "
        "peak %.2f GB | dense %.2f GB%s",
        c.tag.c_str(), c.material.c_str(), c.patch, c.map_patch, c.taper, c.gap, c.pml_width,
        c.mesh_size, c.sigma_max, c.order, c.target_accuracy, c.stretched_touching_points, c.waist,
        c.rms, r.est.patch_rms, c.correlation_length, static_cast<unsigned long long>(c.seed),
        r.est.size.half[0], r.est.size.half[1], r.est.size.half[2], r.est.size.half[3],
        f.window.inner_half, f.window.outer_half, static_cast<long long>(r.est.size.unknowns),
        static_cast<long long>(r.est.masked), static_cast<long long>(r.est.size.unknowns / 2),
        r.est.dark_band, c.dark_band_tolerance, f.reflectance(), f.p_refl, f.power, r.decay.j,
        r.decay.m, r.gap.gap_j, r.gap.gap_m, r.residual, r.lapack ? "lapack" : "eigen", r.setup_s,
        r.assembly_s, r.lu_s, r.solve_s, r.post_s, r.peak_rss / 1e9, r.est.size.dense_bytes / 1e9,
        c.eps_imag ? (" | DIAGNOSTIC Im eps_r " + std::to_string(*c.eps_imag)).c_str() : "");
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_rough_ag::row_line: formatting failed or line truncated");
    return buf;
}

/// One machine-readable comparison line.
[[nodiscard]] inline std::string compare_line(const FarField& t, const FarField& r,
                                              const Metrics& m) {
    char buf[600];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "COMPARE | %s (window %.4g/%.4g) vs %s (window %.4g/%.4g) | l2 %.4e | l2_off30 %.4e | "
        "eps_xz %.4e | eps_yz %.4e | dR %+.4e | R %.7f vs %.7f",
        t.tag.c_str(), t.window.inner_half, t.window.outer_half, r.tag.c_str(), r.window.inner_half,
        r.window.outer_half, m.l2, m.l2_off30, m.eps_xz, m.eps_yz, m.dR, t.reflectance(),
        r.reflectance());
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_rough_ag::compare_line: formatting failed or line truncated");
    return buf;
}

/// DRC of the plane of incidence (xz cut) at a few angles, one line.
[[nodiscard]] inline std::string drc_line(const FarField& f) {
    std::ostringstream os;
    os << "DRC | " << f.tag << " | xz [1/sr]";
    char buf[64];
    for (const Real th : {-60.0, -40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0, 60.0}) {
        std::snprintf(buf, sizeof(buf), " | %+.0f %.4e", th, drc_xz(f, th));
        os << buf;
    }
    return os.str();
}

}  // namespace pml_rough_ag
