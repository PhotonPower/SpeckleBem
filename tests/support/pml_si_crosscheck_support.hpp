#pragma once
/// @file pml_si_crosscheck_support.hpp
/// Validation 7e of the open-surface PML route (ADR 0009 decision 7e, WP-PML4c; docs/05 "PML
/// route" row 7e): the same Gaussian rough Si surface is solved on both routes,
///  * the PML route: open surface (patch, taper, gap, PML; geometry::make_open_surface_mesh),
///    complex-coordinate-stretching PML, dense operator, LU (LAPACK when available) or GMRES,
///    far field of the real part with the default window across the gap;
///  * the box route: the validated uniform closing box of ADR 0006 (amendment 2026-10-12b: box
///    cells = top-face cells, Si fine band, default depth 5.65 um), MLFMM d0 = 3 + full GMRES,
///    the WP-V2 settings of benchmarks/results/box_validity_mlfmm.md,
/// under one rigorous angular-spectrum beam (normal incidence, E along x, one controlled ball
/// for both meshes), and their far fields are compared with the ADR 0006 box-validity metric
/// (box_validity_support.hpp): the relative L2 difference of |F|^2 over the reflection
/// hemisphere (w = sin theta), eps_rr on the xz/yz cuts, and the reflectance R (WP25:
/// reflected power over the beam power). Shared by benchmarks/pml_si_crosscheck.cpp and the
/// tests tests/unit/test_pml_si_crosscheck.cpp and
/// tests/validation_large/test_pml_si_crosscheck_large.cpp; no Catch2.
///
/// Height map. One map is generated once (geometry::generate_gaussian_height_map, seed, sigma,
/// Lc) on map_size x map_size at the fine spacing gen_spacing and subsampled to the mesh
/// spacing h (an integer multiple of gen_spacing), so that meshes at h = lambda/4, lambda/6 and
/// lambda/10 sample the same surface. The PML route meshes the subsampled map directly (its
/// patch carries the full heights, the taper takes them to zero, gap and PML are flat); the box
/// route crops the same subsampled map centrally to L x L. height_identity() checks that the
/// patch heights of the open mesh are bitwise the box top-face heights on the common grid, and
/// patch_triangle_identity() that the patch triangles are the box top-face triangles.
#include "specklebem/core/path.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/io/result_writer.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ag_box_mechanism_support.hpp"
#include "box_validity_support.hpp"
#include "mlfmm_scaling_support.hpp"
#include "pml_fresnel_support.hpp"
#include "system_memory.hpp"

namespace pml_si_crosscheck {

using namespace specklebem;

inline constexpr Real kLambda = 500e-9;

/// Linear solver of the PML route.
enum class PmlSolver {
    Lu,     ///< dense LU of a copy of Z (solver::DenseLu: LAPACK zgetrf or Eigen)
    Gmres,  ///< full GMRES on the dense operator (no copy; Simulation::solve)
    /// WP-PML5b: the fast path, compression "mlfmm" on the PML route (MLFMM on the real
    /// functions + the PML ring with ACA far blocks, d0 = Case::digits), full GMRES
    Fast
};

/// One configuration of the cross-check (both routes; one route per run).
struct Case {
    // Surface (the same map on both routes).
    std::uint64_t seed = 1;          ///< generator seed (> 0, fixed)
    Real sigma = 50e-9;              ///< rms height [m]
    Real correlation_length = 5e-7;  ///< Lc [m]
    /// Edge of the generated map [m]; must cover the PML route's patch and taper
    /// (L + 2 W_t) and every box L of a study (default: PML L up to 5.5 um).
    Real map_size = 6.5e-6;
    Real gen_spacing = kLambda / 60.0;  ///< generation spacing [m]; h must be a multiple
    // Patch and beam.
    Real L = 4.5e-6;               ///< patch edge = box side [m]
    Real waist = 0.75e-6;          ///< w0 [m] (L / 6: ADR 0006 rule for sigma <= 50 nm)
    Real mesh_size = kLambda / 4;  ///< h [m], both routes
    /// Controlled ball of the beam [m], the same for every run of a study (box vertices up to
    /// ~6.9 um from the focus at L = 5.5 um; the PML route needs 1.02 sqrt(2) pml_start).
    Real beam_radius = 7.0e-6;
    // PML route (ADR 0009, the 7a rules).
    Real taper = 0.5e-6;      ///< W_t >= Lc [m]
    Real gap = 0.5e-6;        ///< W_g [m] (the far-field window spans it)
    Real pml_width = 1.0e-6;  ///< W_pml = 2 lambda [m]
    Real sigma_max = 4.0;
    Real order = 2.0;
    Real dark_band_tolerance = 1e-4;
    Real target_accuracy = 1e-5;         ///< kernels::OperatorOptions (both routes)
    int stretched_touching_points = 10;  ///< kernels::OperatorOptions
    PmlSolver pml_solver = PmlSolver::Lu;
    bool use_lapack = true;  ///< LU: LAPACK when available (else Eigen PartialPivLU)
    Real pml_tol = 1e-8;     ///< GMRES tolerance of the PML route
    int pml_max_iter = 2000;
    // Box route (ADR 0006, WP-V2 settings).
    Real depth_factor = 1.0;  ///< box depth = factor x default_box_depth (5.65 um)
    Real digits = 3.0;        ///< MLFMM d0
    Real box_tol = 1e-5;      ///< GMRES tolerance (Si noise floor 3.7e-5, WP-V2)
    int box_max_iter = 400;
    // Far field.
    int ff_degree = 10;  ///< Dunavant degree of the far-field source quadrature
    /// PML route only (WP-PML6): far-field source of the observables (windowed currents, the
    /// WP-PML4c record; or the Huygens plane at its defaults, ADR 0010).
    FarFieldSource pml_far_field = FarFieldSource::Currents;
};

/// Throws std::invalid_argument for an invalid case (before any meshing).
inline void validate(const Case& c) {
    const auto pos = [](Real v) { return std::isfinite(v) && v > 0.0; };
    if (c.seed == 0)
        throw std::invalid_argument("pml_si_crosscheck: seed must be fixed (> 0)");
    if (!pos(c.sigma) || !pos(c.correlation_length) || !pos(c.map_size) || !pos(c.gen_spacing) ||
        !pos(c.L) || !pos(c.waist) || !pos(c.mesh_size) || !pos(c.beam_radius) || !pos(c.taper) ||
        !pos(c.pml_width) || !pos(c.sigma_max) || !pos(c.dark_band_tolerance) ||
        !pos(c.target_accuracy) || !pos(c.pml_tol) || !pos(c.depth_factor) || !pos(c.digits) ||
        !pos(c.box_tol))
        throw std::invalid_argument("pml_si_crosscheck: sizes, tolerances and widths must be > 0");
    if (!(c.gap > 0.0) || !std::isfinite(c.gap) || !(c.order >= 0.0) || !std::isfinite(c.order))
        throw std::invalid_argument("pml_si_crosscheck: need gap > 0 and order >= 0");
    if (c.pml_max_iter < 1 || c.box_max_iter < 1 || c.ff_degree < 1 ||
        c.stretched_touching_points < 1)
        throw std::invalid_argument(
            "pml_si_crosscheck: max-iter, ff degree and touching points must be >= 1");
}

// ---------------------------------------------------------------------------------------------
// Height map.

/// The generated (fine) map of a case.
inline geometry::HeightMap generated_map(const Case& c) {
    validate(c);
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = c.map_size;
    rp.rms_roughness = c.sigma;
    rp.correlation_length = c.correlation_length;
    rp.mesh_size = c.gen_spacing;
    rp.seed = c.seed;
    return geometry::generate_gaussian_height_map(rp);
}

/// Integer k with k dx = h (relative 1e-9).
/// @throws std::invalid_argument if h is not an integer multiple (>= 1) of the map spacing.
inline Index subsample_factor(const geometry::HeightMap& fine, Real h) {
    if (!(fine.dx > 0.0) || !(h > 0.0))
        throw std::invalid_argument("subsample_factor: spacings must be > 0");
    const auto k = static_cast<Index>(std::llround(h / fine.dx));
    if (k < 1 || std::abs(static_cast<Real>(k) * fine.dx - h) > 1e-9 * h)
        throw std::invalid_argument(
            "subsample_factor: h is not an integer multiple of the map "
            "spacing (choose --gen-h = h / k)");
    return k;
}

/// Every k-th node of a square map: z_sub(i, j) = z(k i, k j), spacing k dx (the same centred
/// grid, x_i = -(n - 1) dx / 2 + i dx, so the nodes coincide with the fine ones).
/// @throws std::invalid_argument for k < 1, a non-square map, dx != dy or n - 1 not divisible
///         by k.
inline geometry::HeightMap subsample(const geometry::HeightMap& fine, Index k) {
    const Index n = fine.z.rows();
    if (k < 1 || fine.z.cols() != n || n < 2 || fine.dx != fine.dy || (n - 1) % k != 0)
        throw std::invalid_argument(
            "subsample: need a square map with dx = dy and n - 1 divisible by k >= 1");
    const Index m = (n - 1) / k + 1;
    geometry::HeightMap s;
    s.dx = static_cast<Real>(k) * fine.dx;
    s.dy = s.dx;
    s.z.resize(m, m);
    for (Index i = 0; i < m; ++i)
        for (Index j = 0; j < m; ++j) s.z(i, j) = fine.z(k * i, k * j);
    return s;
}

/// Central crop of a square map to an L x L patch on its own grid.
/// @throws std::invalid_argument unless L is a centred sub-grid (relative 1e-9).
inline geometry::HeightMap crop(const geometry::HeightMap& big, Real L) {
    const Index nb = big.z.rows();
    const Index ns = static_cast<Index>(std::llround(L / big.dx)) + 1;
    if (big.z.cols() != nb || ns < 2 || ns > nb || (nb - ns) % 2 != 0 ||
        std::abs(static_cast<Real>(ns - 1) * big.dx - L) > 1e-9 * L)
        throw std::invalid_argument("crop: L must be a centred sub-grid of the map");
    const Index m = (nb - ns) / 2;
    geometry::HeightMap h;
    h.dx = big.dx;
    h.dy = big.dy;
    h.z = big.z.block(m, m, ns, ns);
    return h;
}

/// The map both routes mesh: generated_map() subsampled to h.
inline geometry::HeightMap mesh_map(const Case& c) {
    const geometry::HeightMap fine = generated_map(c);
    return subsample(fine, subsample_factor(fine, c.mesh_size));
}

// ---------------------------------------------------------------------------------------------
// Meshes.

/// Open surface of the PML route on the mesh map (patch L, taper, gap, PML).
inline geometry::OpenSurfaceMesh make_open_mesh(const geometry::HeightMap& map, const Case& c) {
    geometry::OpenSurfaceParams op;
    op.patch_size = c.L;
    op.taper_width = c.taper;
    op.gap_width = c.gap;
    op.pml_width = c.pml_width;
    return geometry::make_open_surface_mesh(map, op);
}

/// Closing box of the box route.
struct BoxGeometry {
    geometry::HeightMap top;  ///< the cropped map (top face)
    RoughBoxParams params;    ///< rough_surface_box_params(Si, vacuum, uniform)
    Real depth = 0;
    geometry::detail::BoxGrading grading;
    geometry::TriangleMesh mesh;
    Index top_triangles = 0;
};

/// The uniform box of ADR 0006 (amendment 2026-10-12b) on the central L x L crop of the mesh map:
/// box cells = top-face cells, the Si fine band 3 delta + 3 sigma (inert in the uniform box),
/// depth = depth_factor x default_box_depth. Checks the waist rule (w0 <= L / 6 for Si).
inline BoxGeometry make_box(const geometry::HeightMap& map, const Case& c) {
    validate(c);
    BoxGeometry b;
    b.top = crop(map, c.L);
    const material::Material si = material::silicon_500nm();
    b.params = rough_surface_box_params(si, material::vacuum(), kLambda, c.sigma, c.mesh_size,
                                        BoxGradingMode::Uniform);
    check_beam_waist(c.L, c.waist, {.max_fraction = b.params.beam_waist_fraction});
    b.depth = c.depth_factor * b.params.box_depth;
    std::optional<Real> fine = b.params.box_fine_depth;
    if (fine && !(*fine < b.depth))
        fine.reset();
    b.grading = geometry::detail::box_grading(b.top, b.depth, b.params.box_mesh_size, fine,
                                              b.params.exterior_wavelength);
    b.mesh = geometry::make_mesh_from_height_map(b.top, b.depth, b.params.box_mesh_size, fine,
                                                 b.params.exterior_wavelength);
    b.params.box_fine_depth = fine;
    b.top_triangles = 2 * (b.top.z.rows() - 1) * (b.top.z.cols() - 1);
    return b;
}

/// Result of height_identity().
struct HeightIdentity {
    Index grid_nodes = 0;      ///< n^2 nodes of the cropped map (the box top face)
    Index open_nodes = 0;      ///< nodes of the common grid found in the open mesh
    Index box_nodes = 0;       ///< nodes found in the box mesh (top face: smallest z per node)
    Index open_vs_map = 0;     ///< open-mesh z != map z (bitwise)
    Index box_vs_map = 0;      ///< box top-face z != map z (bitwise)
    Index open_vs_box = 0;     ///< open-mesh z != box z (bitwise)
    Real max_xy_offset = 0.0;  ///< largest |x|, |y| difference open vs box vertex [m]
    [[nodiscard]] bool bitwise() const {
        return open_nodes == grid_nodes && box_nodes == grid_nodes && open_vs_map == 0 &&
               box_vs_map == 0 && open_vs_box == 0;
    }
};

namespace detail {

/// Throws std::invalid_argument unless `top` is a square map with at least 2 x 2 nodes and dx > 0.
inline void check_grid_map(const geometry::HeightMap& top, const char* what) {
    const Index n = top.z.rows();
    if (top.z.cols() != n || n < 2 || !(top.dx > 0.0))
        throw std::invalid_argument(std::string(what) + ": need a square map with dx > 0");
}

/// Node (i, j) of the centred n x n grid of `top` (x_i = -(n - 1) dx / 2 + i dx) at (x, y);
/// false unless (x, y) lies within 1e-6 dx of a node.
inline bool grid_node(const geometry::HeightMap& top, Real x, Real y, Index& i, Index& j) {
    const Index n = top.z.rows();
    const Real half = 0.5 * static_cast<Real>(n - 1) * top.dx;
    i = static_cast<Index>(std::llround((x + half) / top.dx));
    j = static_cast<Index>(std::llround((y + half) / top.dx));
    if (i < 0 || j < 0 || i >= n || j >= n)
        return false;
    return std::abs(static_cast<Real>(i) * top.dx - half - x) <= 1e-6 * top.dx &&
           std::abs(static_cast<Real>(j) * top.dx - half - y) <= 1e-6 * top.dx;
}

}  // namespace detail

/// Compares the patch heights of the open mesh with the box top face on the common grid (the
/// nodes of `top`, the cropped map): every node of the open mesh and the top-face node of the box
/// (the vertex of smallest z at that (x, y): walls and bottom lie below the rim, z > xi) must
/// carry exactly top.z(i, j). Vertices within 1e-6 dx of a node count as that node.
inline HeightIdentity height_identity(const geometry::HeightMap& top,
                                      const geometry::TriangleMesh& open,
                                      const geometry::TriangleMesh& box) {
    detail::check_grid_map(top, "height_identity");
    const Index n = top.z.rows();
    const auto node_of = [&](Real x, Real y, Index& i, Index& j) {
        return detail::grid_node(top, x, y, i, j);
    };
    const auto nn = static_cast<std::size_t>(n * n);
    constexpr Real kNone = std::numeric_limits<Real>::infinity();
    std::vector<Real> zo(nn, kNone), zb(nn, kNone);
    std::vector<Vec3> po(nn, Vec3::Zero()), pb(nn, Vec3::Zero());
    HeightIdentity r;
    r.grid_nodes = n * n;
    Index i = 0, j = 0;
    for (Index v = 0; v < open.num_vertices(); ++v) {
        const Vec3 p = open.vertices().row(v).transpose();
        if (!node_of(p.x(), p.y(), i, j))
            continue;
        const auto k = static_cast<std::size_t>(i * n + j);
        if (zo[k] != kNone)
            throw std::logic_error("height_identity: two open-mesh vertices on one node");
        zo[k] = p.z();
        po[k] = p;
    }
    for (Index v = 0; v < box.num_vertices(); ++v) {
        const Vec3 p = box.vertices().row(v).transpose();
        if (!node_of(p.x(), p.y(), i, j))
            continue;
        const auto k = static_cast<std::size_t>(i * n + j);
        if (p.z() < zb[k]) {
            zb[k] = p.z();
            pb[k] = p;
        }
    }
    for (Index a = 0; a < n; ++a) {
        for (Index b = 0; b < n; ++b) {
            const auto k = static_cast<std::size_t>(a * n + b);
            const Real zm = top.z(a, b);
            if (zo[k] != kNone) {
                ++r.open_nodes;
                if (zo[k] != zm)
                    ++r.open_vs_map;
            }
            if (zb[k] != kNone) {
                ++r.box_nodes;
                if (zb[k] != zm)
                    ++r.box_vs_map;
            }
            if (zo[k] != kNone && zb[k] != kNone) {
                if (zo[k] != zb[k])
                    ++r.open_vs_box;
                r.max_xy_offset = std::max({r.max_xy_offset, std::abs(po[k].x() - pb[k].x()),
                                            std::abs(po[k].y() - pb[k].y())});
            }
        }
    }
    return r;
}

/// Number of geometry::SurfaceZone::Patch triangles of the open mesh.
inline Index count_patch_triangles(const geometry::OpenSurfaceMesh& om) {
    return static_cast<Index>(
        std::count(om.zone.begin(), om.zone.end(), geometry::SurfaceZone::Patch));
}

/// Result of patch_triangle_identity().
struct TriangleIdentity {
    Index open_patch = 0;  ///< SurfaceZone::Patch triangles of the open mesh
    Index box_top = 0;     ///< box triangles whose three vertices are top-face vertices
    Index matched = 0;     ///< open patch triangles whose oriented node triple is a box top one
    [[nodiscard]] bool agree() const {
        return open_patch > 0 && open_patch == box_top && matched == open_patch;
    }
};

/// Compares the patch triangles of the open mesh with the box top-face triangles as oriented
/// vertex triples on the common grid of `top` (the cropped map). Each triangle becomes the cyclic
/// sequence of the grid nodes of its vertices, rotated so that the smallest node index comes
/// first (this keeps the orientation). A box triangle belongs to the top face when its three
/// vertices are the top-face vertices of their nodes (smallest z per node, as in
/// height_identity()). An open patch triangle with a vertex off the grid counts as unmatched.
inline TriangleIdentity patch_triangle_identity(const geometry::HeightMap& top,
                                                const geometry::OpenSurfaceMesh& om,
                                                const geometry::TriangleMesh& box) {
    detail::check_grid_map(top, "patch_triangle_identity");
    if (static_cast<Index>(om.zone.size()) != om.mesh.num_triangles())
        throw std::invalid_argument("patch_triangle_identity: one zone per open triangle needed");
    const Index n = top.z.rows();
    const auto nn = static_cast<std::size_t>(n * n);
    using Triple = std::array<Index, 3>;
    const auto cyclic = [](Index a, Index b, Index c) -> Triple {
        if (b < a && b < c)
            return {b, c, a};
        if (c < a && c < b)
            return {c, a, b};
        return {a, b, c};
    };
    Index i = 0, j = 0;
    // Box: the top-face vertex of every node, then the triangles made of three of them.
    std::vector<Index> top_vertex(nn, -1);
    std::vector<Real> zmin(nn, std::numeric_limits<Real>::infinity());
    for (Index v = 0; v < box.num_vertices(); ++v) {
        const Vec3 p = box.vertices().row(v).transpose();
        if (!detail::grid_node(top, p.x(), p.y(), i, j))
            continue;
        const auto k = static_cast<std::size_t>(i * n + j);
        if (p.z() < zmin[k]) {
            zmin[k] = p.z();
            top_vertex[k] = v;
        }
    }
    std::vector<Index> box_node(static_cast<std::size_t>(box.num_vertices()), -1);
    for (std::size_t k = 0; k < nn; ++k)
        if (top_vertex[k] >= 0)
            box_node[static_cast<std::size_t>(top_vertex[k])] = static_cast<Index>(k);
    std::vector<Triple> box_keys;
    const Triangles& bt = box.triangles();
    for (Index t = 0; t < bt.rows(); ++t) {
        const Index a = box_node[static_cast<std::size_t>(bt(t, 0))];
        const Index b = box_node[static_cast<std::size_t>(bt(t, 1))];
        const Index c = box_node[static_cast<std::size_t>(bt(t, 2))];
        if (a >= 0 && b >= 0 && c >= 0)
            box_keys.push_back(cyclic(a, b, c));
    }
    std::sort(box_keys.begin(), box_keys.end());
    TriangleIdentity r;
    r.box_top = static_cast<Index>(box_keys.size());
    // Open mesh: the node of every vertex, then the patch triangles.
    std::vector<Index> open_node(static_cast<std::size_t>(om.mesh.num_vertices()), -1);
    for (Index v = 0; v < om.mesh.num_vertices(); ++v) {
        const Vec3 p = om.mesh.vertices().row(v).transpose();
        if (detail::grid_node(top, p.x(), p.y(), i, j))
            open_node[static_cast<std::size_t>(v)] = i * n + j;
    }
    const Triangles& ot = om.mesh.triangles();
    for (Index t = 0; t < ot.rows(); ++t) {
        if (om.zone[static_cast<std::size_t>(t)] != geometry::SurfaceZone::Patch)
            continue;
        ++r.open_patch;
        const Index a = open_node[static_cast<std::size_t>(ot(t, 0))];
        const Index b = open_node[static_cast<std::size_t>(ot(t, 1))];
        const Index c = open_node[static_cast<std::size_t>(ot(t, 2))];
        if (a >= 0 && b >= 0 && c >= 0 &&
            std::binary_search(box_keys.begin(), box_keys.end(), cyclic(a, b, c)))
            ++r.matched;
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// Beam, configurations.

/// The rigorous beam of both routes: normal incidence from z < 0, E along x (Polarization::P at
/// 0 deg), focus at the origin, controlled ball beam_radius.
inline std::shared_ptr<excitation::AngularSpectrumBeam> make_beam(const Case& c) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.waist;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = 0.0;
    bp.polarization = excitation::Polarization::P;
    bp.region_radius = c.beam_radius;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());
}

inline PmlTermination make_termination(const Case& c) {
    PmlTermination p;
    p.sigma_max = c.sigma_max;
    p.order = c.order;
    p.correlation_length = c.correlation_length;
    p.dark_band_tolerance = c.dark_band_tolerance;
    p.far_field_source = c.pml_far_field;
    return p;
}

inline SimulationConfig pml_config(const Case& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = material::silicon_500nm();
    cfg.formulation = formulation::Kind::ICTF;
    cfg.compression = c.pml_solver == PmlSolver::Fast ? "mlfmm" : "dense";
    cfg.mlfmm.accuracy_digits = c.digits;
    cfg.solver = c.pml_solver == PmlSolver::Lu ? SolverKind::Direct : SolverKind::Gmres;
    cfg.gmres.tolerance = c.pml_tol;
    cfg.gmres.max_iter = c.pml_max_iter;
    cfg.gmres.restart = 0;
    cfg.kernels.target_accuracy = c.target_accuracy;
    cfg.kernels.stretched_touching_points = c.stretched_touching_points;
    return cfg;
}

/// Box route: MLFMM d0 = digits with the global leaf rule (leaf_radius_quantile 1), ICTF, Jacobi
/// per formulation::recommend (Si: none), full GMRES (WP-V2).
inline SimulationConfig box_config(const Case& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = material::silicon_500nm();
    cfg.formulation = formulation::Kind::ICTF;
    cfg.compression = "mlfmm";
    cfg.mlfmm.accuracy_digits = c.digits;
    cfg.mlfmm.leaf_radius_quantile = 1.0;
    cfg.solver = SolverKind::Gmres;
    cfg.gmres.tolerance = c.box_tol;
    cfg.gmres.max_iter = c.box_max_iter;
    cfg.gmres.restart = 0;
    cfg.kernels.target_accuracy = c.target_accuracy;
    return cfg;
}

// ---------------------------------------------------------------------------------------------
// Memory.

/// Memory [GB, 1e9 bytes] of the PML route: the dense matrix 16 (2N)^2, plus the LU copy (LU)
/// or the full-GMRES Krylov basis at pml_max_iter (GMRES).
struct PmlEstimate {
    Index triangles = 0;
    Index unknowns = 0;  ///< 2N
    std::array<Real, 4> half{};
    Real dense_gb = 0;
    Real peak_gb = 0;
};

inline PmlEstimate estimate_pml(const Case& c, const geometry::OpenSurfaceMesh& om) {
    PmlEstimate e;
    e.triangles = om.mesh.num_triangles();
    e.unknowns = 2 * om.mesh.num_edges();
    e.half = {om.patch_half, om.taper_end, om.pml_start, om.outer_half};
    e.dense_gb = 16.0 * static_cast<Real>(e.unknowns) * static_cast<Real>(e.unknowns) / 1e9;
    // The fast path (no dense matrix): at most the dense matrix (the 5a ring bound) + Krylov.
    e.peak_gb = c.pml_solver == PmlSolver::Lu
                    ? 2.0 * e.dense_gb
                    : e.dense_gb + box_validity::krylov_basis_gb(c.pml_max_iter, e.unknowns);
    return e;
}

/// Box route: the MLFMM setup peak model of the scaling study plus the Krylov basis.
struct BoxEstimate {
    mlfmm_scaling::Estimate mlfmm;
    Real setup_gb = 0;
    Real krylov_gb = 0;
    [[nodiscard]] Real peak_gb() const { return setup_gb + krylov_gb; }
};

inline BoxEstimate estimate_box(const Case& c, const BoxGeometry& b) {
    mlfmm_scaling::Case mc;
    mc.material = "si";
    mc.L = c.L;
    mc.sigma = c.sigma;
    mc.mesh_size = c.mesh_size;
    mc.seed = c.seed;
    mc.digits = c.digits;
    mc.leaf_radius_quantile = 1.0;
    mc.tolerance = c.box_tol;
    mc.max_iter = c.box_max_iter;
    const mlfmm_scaling::Geometry geo{b.mesh, b.depth, b.params.box_fine_depth, b.grading,
                                      b.top_triangles};
    BoxEstimate e;
    e.mlfmm = mlfmm_scaling::estimate(geo, mc);
    e.setup_gb = mlfmm_scaling::gb(e.mlfmm.peak_setup_bytes);
    e.krylov_gb = box_validity::krylov_basis_gb(c.box_max_iter, e.mlfmm.unknowns);
    return e;
}

/// Margin [GB] kept between a run's peak estimate and the available memory.
inline constexpr Real kMemoryMarginGb = 4.0;

/// Memory guard of the study: unless `force`, refuses (std::runtime_error) a run whose peak
/// estimate exceeds max_gb, or whose peak plus kMemoryMarginGb exceeds the available memory
/// (skipped when that is unknown, available_gb <= 0).
/// @throws std::invalid_argument for a negative or non-finite estimate, non-finite available
///         memory or max_gb <= 0 (also with force).
inline void check_memory(Real peak_gb, Real available_gb, Real max_gb, bool force) {
    if (!std::isfinite(peak_gb) || peak_gb < 0 || !std::isfinite(available_gb) ||
        !std::isfinite(max_gb) || !(max_gb > 0))
        throw std::invalid_argument("check_memory: need a finite peak >= 0 and max_gb > 0");
    if (force)
        return;
    if (peak_gb > max_gb)
        throw std::runtime_error("peak estimate " + std::to_string(peak_gb) +
                                 " GB exceeds --max-gb " + std::to_string(max_gb) +
                                 " (pass --force to run anyway)");
    if (available_gb > 0 && peak_gb + kMemoryMarginGb > available_gb)
        throw std::runtime_error("peak estimate " + std::to_string(peak_gb) +
                                 " GB + 4 GB exceeds the available memory " +
                                 std::to_string(available_gb) + " GB (wait or --force)");
}

// ---------------------------------------------------------------------------------------------
// Far field and metric.

/// Number of Gauss-Legendre polar nodes / azimuths of the reflectance grid (WP25 / 7a: 1 deg).
inline constexpr int kRTheta = 90;
inline constexpr int kRPhi = 180;

/// Reflection-hemisphere grid and the xz/yz cuts of the box-validity study
/// (box_validity_support.hpp); the forward hemisphere is not evaluated (it lies in R2 on the PML
/// route), so FarField::fwd stays empty.
inline box_validity::FarField far_field(const post::SurfaceSolution& s,
                                        const post::FieldOptions& fo) {
    using box_validity::intensity;
    box_validity::FieldMatrix F;
    box_validity::FarField ff;
    post::far_field(s, box_validity::hemisphere_directions(-1.0), F, fo);
    ff.refl = intensity(F, box_validity::kHemiTheta, box_validity::kHemiPhi);
    post::far_field(s, box_validity::cut_directions(false), F, fo);
    ff.cut_xz = intensity(F, box_validity::kCut, 1);
    post::far_field(s, box_validity::cut_directions(true), F, fo);
    ff.cut_yz = intensity(F, box_validity::kCut, 1);
    return ff;
}

/// The ADR 0006 box-validity metric of a test far field against a reference (relative values):
/// l2_refl (the verdict metric), l2_off30 (theta >= 30 deg), dP_refl (midpoint-grid power of the
/// reflection hemisphere), eps_xz / eps_yz (docs/05 eps_rr on the cuts), and dR = R_t / R_r - 1
/// (WP25 reflectance, Gauss-Legendre grid).
struct Metrics {
    Real l2_refl = 0, l2_off30 = 0, dP_refl = 0, eps_xz = 0, eps_yz = 0, dR = 0;
};

/// @throws std::invalid_argument for different grids, a zero reference or R_r <= 0.
inline Metrics compare(const box_validity::FarField& t, const box_validity::FarField& r, Real R_t,
                       Real R_r) {
    if (t.refl.rows() != r.refl.rows() || t.refl.cols() != r.refl.cols() ||
        t.cut_xz.size() != r.cut_xz.size() || t.cut_yz.size() != r.cut_yz.size())
        throw std::invalid_argument("pml_si_crosscheck::compare: far-field grids differ");
    if (!(R_r > 0.0) || !std::isfinite(R_t))
        throw std::invalid_argument("pml_si_crosscheck::compare: need R_r > 0 and finite R_t");
    Metrics m;
    m.l2_refl = box_validity::hemi_l2(t.refl, r.refl, 0.0);
    m.l2_off30 = box_validity::hemi_l2(t.refl, r.refl, box_validity::kOffSpecularTheta);
    m.dP_refl = box_validity::hemi_power(t.refl) / box_validity::hemi_power(r.refl) - 1.0;
    m.eps_xz = box_validity::eps_rr(t.cut_xz, r.cut_xz);
    m.eps_yz = box_validity::eps_rr(t.cut_yz, r.cut_yz);
    m.dR = R_t / R_r - 1.0;
    return m;
}

/// Tolerances of 7e: the box's own sensitivity (benchmarks/results/pml_si_crosscheck.md,
/// "Tolerance"): the largest change of the uniform box under L x 1.2 at w0 = L/5 .. L/6,
/// sigma = 50 nm (WP-V2 / WP-V2b records, seeds 1 and 2, h = 50 nm) and under the L partners of
/// this study at h = lambda / 4, for each quantity.
struct Tolerances {
    Real l2_refl = 0;  ///< on Metrics::l2_refl
    Real eps_rr = 0;   ///< on max(eps_xz, eps_yz)
    Real dR = 0;       ///< on |Metrics::dR|
};

/// Values fixed from the box runs before the PML route was solved (record, "Tolerance").
inline constexpr Tolerances kTolerances{2.29e-3, 5.8e-4, 4.1e-5};

/// Whether each metric is within its tolerance.
struct Verdict {
    bool l2 = false, eps = false, R = false;
    [[nodiscard]] bool all() const { return l2 && eps && R; }
};

inline Verdict verdict(const Metrics& m, const Tolerances& t) {
    Verdict v;
    v.l2 = m.l2_refl <= t.l2_refl;
    v.eps = std::max(m.eps_xz, m.eps_yz) <= t.eps_rr;
    v.R = std::abs(m.dR) <= t.dR;
    return v;
}

// ---------------------------------------------------------------------------------------------
// Runs.

/// Result of one route.
struct RouteResult {
    std::string route;  ///< "pml" or "box"
    Index triangles = 0;
    Index unknowns = 0;  ///< 2N
    // PML route.
    std::array<Real, 4> half{};  ///< a_0 .. a_3 [m]
    Index masked = 0;            ///< masked test functions (of N)
    Real dark_band = 0;          ///< Simulation::dark_band_ratio()
    pml_fresnel::EdgeDecay decay;
    // Box route.
    Real depth = 0;
    Index top_triangles = 0;
    Real peak_estimate_gb = 0;
    // Solve.
    std::string solver;
    int iterations = 0;
    bool converged = false;
    Real residual = 0;  ///< true relative residual |b - Z x| / |b|
    // Observables.
    Real p_inc = 0;   ///< Simulation::incident_power() [W]
    Real p_refl = 0;  ///< reflected power on the kRTheta x kRPhi grid [W]
    [[nodiscard]] Real reflectance() const { return p_refl / p_inc; }
    box_validity::FarField ff;
    /// PML route with Case::pml_far_field == Huygens (WP-PML6): the windowed-current far field
    /// of the same solution and its reflected power, for the Huygens-vs-currents comparison.
    std::optional<box_validity::FarField> ff_currents;
    Real p_refl_currents = 0;
    Real ff_check = 0;  ///< l2_refl of Dunavant degree 8 against ff_degree
    Index plane_waves = 0;
    // Costs.
    Real setup_s = 0, assembly_s = 0, solve_s = 0, post_s = 0;
    Real operator_gb = 0;
    Real peak_rss = 0;  ///< process peak after the run [bytes]
    std::string report;
};

namespace detail {

inline Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Far field, reflectance and quadrature check of a solved route (the reflectance as
/// Simulation::reflected_power() / incident_power(): post::radiated_power on the kRTheta x kRPhi
/// hemisphere grid with Simulation::far_field_options(ff_degree)).
inline void observe(const Simulation& sim, const post::SurfaceSolution& s, const Case& c,
                    RouteResult& r) {
    const post::FieldOptions fo = sim.far_field_options(c.ff_degree);
    r.ff = far_field(s, fo);
    post::FieldOptions fo8 = sim.far_field_options(8);
    box_validity::FieldMatrix F;
    post::far_field(s, box_validity::hemisphere_directions(-1.0), F, fo8);
    r.ff_check = box_validity::hemi_l2(
        box_validity::intensity(F, box_validity::kHemiTheta, box_validity::kHemiPhi), r.ff.refl,
        0.0);
    r.p_inc = sim.incident_power();
    r.p_refl = post::radiated_power(s, post::hemisphere_grid(-1.0, kRTheta, kRPhi), fo);
}

}  // namespace detail

/// Dark-band ratio and masked-function count of the PML route without assembly (the
/// Simulation constructor with the dark-band tolerance relaxed to 1).
struct PmlChecks {
    Real dark_band = 0;
    Index masked = 0;
    Index functions = 0;  ///< N
};

inline PmlChecks pml_checks(const Case& c, const geometry::HeightMap& map) {
    Case relaxed = c;
    relaxed.dark_band_tolerance = 1.0;
    const Simulation sim(make_open_mesh(map, relaxed), make_termination(relaxed), make_beam(c),
                         pml_config(relaxed));
    const std::vector<std::uint8_t> mask =
        op::pml_masked_functions(*sim.problem().space, sim.stretch());
    PmlChecks k;
    k.dark_band = sim.dark_band_ratio();
    k.masked = static_cast<Index>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
    k.functions = sim.problem().space->size();
    return k;
}

/// Solves the PML route on the mesh map.
inline RouteResult run_pml(const Case& c, const geometry::HeightMap& map) {
    validate(c);
    const auto t0 = std::chrono::steady_clock::now();
    RouteResult r;
    r.route = "pml";
    geometry::OpenSurfaceMesh om = make_open_mesh(map, c);
    r.half = {om.patch_half, om.taper_end, om.pml_start, om.outer_half};
    const auto beam = make_beam(c);
    r.plane_waves = beam->num_plane_waves();
    Simulation sim(std::move(om), make_termination(c), beam, pml_config(c));
    r.triangles = sim.mesh().num_triangles();
    r.unknowns = sim.num_unknowns();
    r.dark_band = sim.dark_band_ratio();
    const std::vector<std::uint8_t> mask =
        op::pml_masked_functions(*sim.problem().space, sim.stretch());
    r.masked = static_cast<Index>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
    r.setup_s = detail::seconds_since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    sim.assemble();
    r.assembly_s = detail::seconds_since(t1);
    r.operator_gb = static_cast<Real>(sim.system_operator()->memory_bytes()) / 1e9;
    const auto t2 = std::chrono::steady_clock::now();
    VectorXc x;
    if (c.pml_solver == PmlSolver::Lu) {
        const auto dense =
            std::dynamic_pointer_cast<const op::DenseOperator>(sim.system_operator());
        if (!dense)
            throw std::logic_error("pml_si_crosscheck: the PML route operator is not dense");
        const MatrixXc& Z = dense->matrix();
        const VectorXc& b = sim.rhs();
        {
            const solver::DenseLu lu(
                Z, c.use_lapack ? solver::LuBackend::Auto : solver::LuBackend::Eigen);
            r.solver = lu.backend() == solver::LuBackend::Lapack ? "lapack-lu" : "eigen-lu";
            x = lu.solve(b);
        }
        r.residual = (b - Z * x).norm() / b.norm();
        r.converged = true;
    } else {
        const solver::GmresResult g = sim.solve();
        // The solver label carries the preconditioner (automatic on the fast route: block-
        // diagonal since WP-PC1; on the dense route the recommendation).
        r.solver = std::string(c.pml_solver == PmlSolver::Fast ? "fast-gmres+" : "gmres+") +
                   to_string(sim.preconditioner_kind());
        r.iterations = g.iterations;
        r.converged = g.converged;
        r.residual = g.true_relative_residual;
        x = sim.solution().currents;
    }
    r.solve_s = detail::seconds_since(t2);
    const auto t3 = std::chrono::steady_clock::now();
    const post::SurfaceSolution s{&sim.problem(), x};
    detail::observe(sim, s, c, r);
    if (c.pml_far_field == FarFieldSource::Huygens) {
        const post::FieldOptions fc = sim.far_field_options(c.ff_degree, FarFieldSource::Currents);
        r.ff_currents = far_field(s, fc);
        r.p_refl_currents =
            post::radiated_power(s, post::hemisphere_grid(-1.0, kRTheta, kRPhi), fc);
    }
    r.decay = pml_fresnel::edge_decay(s, r.half[3], c.mesh_size);
    r.post_s = detail::seconds_since(t3);
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// Solves the box route on the central L x L crop of the mesh map.
inline RouteResult run_box(const Case& c, const geometry::HeightMap& map) {
    validate(c);
    const auto t0 = std::chrono::steady_clock::now();
    RouteResult r;
    r.route = "box";
    BoxGeometry b = make_box(map, c);
    r.depth = b.depth;
    r.top_triangles = b.top_triangles;
    const auto beam = make_beam(c);
    r.plane_waves = beam->num_plane_waves();
    Simulation sim(std::move(b.mesh), beam, box_config(c));
    r.triangles = sim.mesh().num_triangles();
    r.unknowns = sim.num_unknowns();
    r.setup_s = detail::seconds_since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    sim.assemble();
    r.assembly_s = detail::seconds_since(t1);
    r.operator_gb = static_cast<Real>(sim.system_operator()->memory_bytes()) / 1e9;
    const solver::GmresResult g = sim.solve();
    r.solver = "mlfmm-gmres";
    r.iterations = g.iterations;
    r.converged = g.converged;
    r.residual = g.true_relative_residual;
    r.solve_s = g.wall_seconds;
    const auto t3 = std::chrono::steady_clock::now();
    detail::observe(sim, sim.solution(), c, r);
    r.post_s = detail::seconds_since(t3);
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

// ---------------------------------------------------------------------------------------------
// Output lines and stored runs.

/// One machine-readable line per route.
inline std::string row_line(const Case& c, const RouteResult& r, const std::string& tag) {
    char buf[2400];
    int len = 0;
    if (r.route == "pml") {
        len = std::snprintf(
            buf, sizeof(buf),
            "ROW | %s | route pml | seed %llu | sigma %.4g | Lc %.4g | L %.4g | w0 %.4g | h %.6g | "
            "gen %.6g | map %.4g | taper %.4g | gap %.4g | pml %.4g | sigma_max %.3g | m %.3g | "
            "half %.4g/%.4g/%.4g/%.4g | 2N %lld | triangles %lld | masked %lld of %lld | dark "
            "%.3e | edge J %.2e M %.2e | %s | iterations %d | residual %.2e | R %.7f | P_refl "
            "%.6e | P_inc %.6e | ff check %.2e | waves %lld | setup %.1f s | assembly %.1f s | "
            "solve %.1f s | post %.1f s | operator %.2f GB | peak %.2f GB%s",
            tag.c_str(), static_cast<unsigned long long>(c.seed), c.sigma, c.correlation_length,
            c.L, c.waist, c.mesh_size, c.gen_spacing, c.map_size, c.taper, c.gap, c.pml_width,
            c.sigma_max, c.order, r.half[0], r.half[1], r.half[2], r.half[3],
            static_cast<long long>(r.unknowns), static_cast<long long>(r.triangles),
            static_cast<long long>(r.masked), static_cast<long long>(r.unknowns / 2), r.dark_band,
            r.decay.j, r.decay.m, r.solver.c_str(), r.iterations, r.residual, r.reflectance(),
            r.p_refl, r.p_inc, r.ff_check, static_cast<long long>(r.plane_waves), r.setup_s,
            r.assembly_s, r.solve_s, r.post_s, r.operator_gb, r.peak_rss / 1e9,
            c.pml_far_field == FarFieldSource::Huygens ? " | far field Huygens plane (WP-PML6)"
                                                       : "");
    } else {
        len = std::snprintf(
            buf, sizeof(buf),
            "ROW | %s | route box | seed %llu | sigma %.4g | Lc %.4g | L %.4g | w0 %.4g | h %.6g | "
            "gen %.6g | map %.4g | depth %.4g | uniform box | 2N %lld | triangles %lld (top %lld) "
            "| "
            "d0 %g | %s | iterations %d | converged %s | true residual %.2e | R %.7f | P_refl "
            "%.6e | P_inc %.6e | ff check %.2e | waves %lld | setup %.1f s | assembly %.1f s | "
            "solve %.1f s | post %.1f s | operator %.2f GB | peak %.2f GB | peak estimate %.2f GB",
            tag.c_str(), static_cast<unsigned long long>(c.seed), c.sigma, c.correlation_length,
            c.L, c.waist, c.mesh_size, c.gen_spacing, c.map_size, r.depth,
            static_cast<long long>(r.unknowns), static_cast<long long>(r.triangles),
            static_cast<long long>(r.top_triangles), c.digits, r.solver.c_str(), r.iterations,
            r.converged ? "yes" : "no", r.residual, r.reflectance(), r.p_refl, r.p_inc, r.ff_check,
            static_cast<long long>(r.plane_waves), r.setup_s, r.assembly_s, r.solve_s, r.post_s,
            r.operator_gb, r.peak_rss / 1e9, r.peak_estimate_gb);
    }
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_si_crosscheck::row_line: formatting failed or truncated");
    return buf;
}

/// Identity line of height_identity().
inline std::string identity_line(const HeightIdentity& h, Index open_patch_triangles,
                                 Index box_top_triangles) {
    char buf[600];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "IDENTITY | grid nodes %lld | open %lld, box %lld | z mismatches open-map %lld, box-map "
        "%lld, open-box %lld | max xy offset %.3e m | patch triangles open %lld, box top %lld | "
        "bitwise %s",
        static_cast<long long>(h.grid_nodes), static_cast<long long>(h.open_nodes),
        static_cast<long long>(h.box_nodes), static_cast<long long>(h.open_vs_map),
        static_cast<long long>(h.box_vs_map), static_cast<long long>(h.open_vs_box),
        h.max_xy_offset, static_cast<long long>(open_patch_triangles),
        static_cast<long long>(box_top_triangles), h.bitwise() ? "yes" : "NO");
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_si_crosscheck::identity_line: formatting failed");
    return buf;
}

/// One COMPARE line (test against reference) with the 7e tolerances and verdicts.
inline std::string compare_line(const std::string& test, const std::string& ref, const Metrics& m,
                                const Tolerances& t) {
    const Verdict v = verdict(m, t);
    char buf[900];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "COMPARE | test %s | reference %s | l2_refl %.4f %% | l2_off30 %.4f %% | eps_rr xz %.4f "
        "%% yz %.4f %% | dP_refl %+.3e | dR %+.3e | tolerances l2 %.4f %%, eps %.4f %%, R %.1e | "
        "l2 %s | eps %s | R %s",
        test.c_str(), ref.c_str(), 100.0 * m.l2_refl, 100.0 * m.l2_off30, 100.0 * m.eps_xz,
        100.0 * m.eps_yz, m.dP_refl, m.dR, 100.0 * t.l2_refl, 100.0 * t.eps_rr, t.dR,
        v.l2 ? "pass" : "FAIL", v.eps ? "pass" : "FAIL", v.R ? "pass" : "FAIL");
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_si_crosscheck::compare_line: formatting failed");
    return buf;
}

/// Attributes that must agree between two stored runs of one comparison (the surface and the
/// beam); route, L and h may differ.
inline const std::vector<std::string>& shared_keys() {
    static const std::vector<std::string> k = {
        "seed", "sigma", "correlation_length", "map_size", "gen_spacing", "waist", "beam_radius"};
    return k;
}

/// Writes a solved route into dir (io::open_npy_directory): the intensities and attributes.
inline void store(const std::string& dir, const Case& c, const RouteResult& r,
                  const std::string& tag) {
    auto w = io::open_npy_directory(dir);
    w->write_matrix("refl_intensity", r.ff.refl);
    w->write_matrix("cut_xz_intensity", r.ff.cut_xz);
    w->write_matrix("cut_yz_intensity", r.ff.cut_yz);
    w->write_attribute("tag", tag);
    w->write_attribute("route", r.route);
    if (r.route == "pml")
        w->write_attribute("far_field_source",
                           c.pml_far_field == FarFieldSource::Huygens ? "huygens" : "currents");
    w->write_attribute("seed", static_cast<Real>(c.seed));
    w->write_attribute("sigma", c.sigma);
    w->write_attribute("correlation_length", c.correlation_length);
    w->write_attribute("map_size", c.map_size);
    w->write_attribute("gen_spacing", c.gen_spacing);
    w->write_attribute("waist", c.waist);
    w->write_attribute("beam_radius", c.beam_radius);
    w->write_attribute("L", c.L);
    w->write_attribute("mesh_size", c.mesh_size);
    w->write_attribute("unknowns", static_cast<Real>(r.unknowns));
    w->write_attribute("converged", r.converged ? "yes" : "no");
    w->write_attribute("residual", r.residual);
    w->write_attribute("reflectance", r.reflectance());
    w->write_attribute("p_refl", r.p_refl);
    w->write_attribute("p_inc", r.p_inc);
}

/// A stored route (store()).
struct StoredRun {
    box_validity::FarField ff;
    Real reflectance = 0;
    std::string tag;
    ag_box_mechanism::Attributes attrs;
};

inline StoredRun load(const std::string& dir) {
    const std::filesystem::path d = core::path_from_utf8(dir);
    StoredRun s;
    s.attrs = ag_box_mechanism::read_attributes(d / "attributes.json");
    std::optional<std::string> converged, reflectance;
    for (const auto& [key, value] : s.attrs) {
        if (key == "converged")
            converged = value;
        else if (key == "reflectance")
            reflectance = value;
        else if (key == "tag")
            s.tag = value;
    }
    if (converged.value_or("no") != "yes" || !reflectance)
        throw std::runtime_error("stored run " + dir + " is not a converged cross-check run");
    s.reflectance = std::stod(*reflectance);
    s.ff.refl = ag_box_mechanism::read_npy<Real>(d / "refl_intensity.npy");
    s.ff.cut_xz = ag_box_mechanism::read_npy<Real>(d / "cut_xz_intensity.npy");
    s.ff.cut_yz = ag_box_mechanism::read_npy<Real>(d / "cut_yz_intensity.npy");
    if (s.tag.empty())
        s.tag = dir;
    return s;
}

/// Value of a stored attribute (empty if missing).
inline std::string attribute(const StoredRun& s, const std::string& key) {
    for (const auto& [k, v] : s.attrs)
        if (k == key)
            return v;
    return {};
}

/// Compares two stored runs after checking shared_keys().
/// @throws std::runtime_error if a shared attribute differs or is missing.
inline Metrics compare_stored(const StoredRun& test, const StoredRun& ref) {
    for (const std::string& k : shared_keys()) {
        const std::string a = attribute(test, k), b = attribute(ref, k);
        if (a.empty() || a != b)
            throw std::runtime_error("stored runs differ in '" + k + "' (" + a + " vs " + b + ")");
    }
    return compare(test.ff, ref.ff, test.reflectance, ref.reflectance);
}

}  // namespace pml_si_crosscheck
