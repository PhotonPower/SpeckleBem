#pragma once
/// @file mlfmm_scaling_support.hpp
/// One case of the MLFMM scaling study (WP22b1, Phase 4 definition of done in docs/01: "scaling
/// exponent gamma (time per solve vs N) <= 1.5 for Si rough surfaces, <= 2.0 for Ag, over N =
/// 5e4 ... 1e6"). Shared by the study executable benchmarks/mlfmm_scaling.cpp and the smoke test
/// tests/unit/test_mlfmm_scaling.cpp, so it must not depend on Catch2.
///
/// Setup (docs/06, ADR 0006, ADR 0008): Gaussian rough surface (sigma = 50 nm, Lc = 500 nm, mesh
/// 50 nm, fixed seed) on an L x L patch, closed by the graded box of ADR 0006 with depth
/// default_box_depth (Si 5.65 um, Ag 2 um), coarse spacing box_mesh_size (default 100 nm =
/// lambda1 / 5, recommendation 1 of benchmarks/results/box_validity.md on the WP-V1 branch; unset:
/// the automatic rule of geometry::RoughSurfaceParams) and, for Si, the fine band
/// default_box_fine_depth = 3 delta + 3 sigma (recommendation 2). Object Si (eps_r = 18.478 -
/// 0.606j) or Ag (-9.794 - 0.313j) in vacuum, lambda = 500 nm; paraxial Gaussian beam
/// (excitation::GaussianBeam, normal incidence, travelling +z, focus at the origin, E along x,
/// waist w0 = L / 4). Simulation with compression "mlfmm" (d0 = 3, automatic leaf rule and
/// exact-part budget), formulation per formulation::recommend (Si: ICTF, Ag: ICTF + left Jacobi),
/// full GMRES with tolerance 1e-3.
///
/// Per case: 2N, octree levels and leaf edge, setup time (Simulation::assemble) split into the
/// octree, the far setup (order searches, block checks, translators, the Ag exact part) and the
/// near field, the time per matvec (median of several applies of the MLFMM operator, and of its
/// near and far parts separately), GMRES iterations and solve time, memory per component and the
/// peak working set after the setup and at the end.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/simulation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "system_memory.hpp"

namespace mlfmm_scaling {

using namespace specklebem;

inline constexpr Real kLambda = 500e-9;
inline constexpr Real kSigma = 50e-9;
inline constexpr Real kCorrelationLength = 500e-9;
inline constexpr Real kMeshSize = 50e-9;
/// Default coarse spacing of the closing box: lambda1 / 5 (WP-V1 recommendation 1).
inline constexpr Real kBoxMeshSize = 100e-9;

/// One case of the study.
struct Case {
    std::string material = "si";  ///< "si" or "ag"
    Real L = 4e-6;                ///< patch edge [m]
    Real mesh_size = kMeshSize;   ///< top-face spacing [m]
    std::uint64_t seed = 1;
    /// Coarse spacing of the closing box [m]; empty: automatic rule (RoughSurfaceParams).
    std::optional<Real> box_mesh_size = kBoxMeshSize;
    /// ADR 0006 fine band 3 delta + 3 sigma; empty: Si yes, Ag no.
    std::optional<bool> fine_band;
    Real waist_factor = 4.0;  ///< w0 = L / waist_factor
    Real digits = 3.0;        ///< accuracy_digits d0
    Real tolerance = 1e-3;
    int max_iter = 3000;
    int restart = 0;                      ///< 0: full GMRES
    int matvecs = 5;                      ///< timed applies per operator (median)
    bool solve = true;                    ///< false: setup and matvec timings only (no GMRES)
    std::size_t max_exact_far_bytes = 0;  ///< 0: automatic budget
    std::optional<formulation::Kind> formulation;
    std::optional<bool> jacobi;
};

inline material::Material object_material(const std::string& name) {
    if (name == "si")
        return material::silicon_500nm();
    if (name == "ag")
        return material::silver_500nm();
    throw std::invalid_argument("mlfmm_scaling: material must be \"si\" or \"ag\", got \"" + name +
                                "\"");
}

inline bool uses_fine_band(const Case& c) {
    return c.fine_band.value_or(c.material == "si");
}

/// Rough box mesh of a case and its grading data.
struct Geometry {
    geometry::TriangleMesh mesh;
    Real depth = 0;
    std::optional<Real> fine_depth;
    geometry::detail::BoxGrading grading;
    Index top_triangles = 0;
};

inline Geometry make_geometry(const Case& c) {
    if (!(c.L > 0.0) || !(c.mesh_size > 0.0) || !(c.waist_factor > 0.0) || c.seed == 0)
        throw std::invalid_argument(
            "mlfmm_scaling: need L > 0, mesh_size > 0, waist_factor > 0 "
            "and a fixed seed > 0");
    const material::Material object = object_material(c.material);
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = c.L;
    rp.rms_roughness = kSigma;
    rp.correlation_length = kCorrelationLength;
    rp.mesh_size = c.mesh_size;
    rp.seed = c.seed;
    const geometry::HeightMap hm = geometry::generate_gaussian_height_map(rp);
    const Real depth = default_box_depth(object, kLambda);
    std::optional<Real> fine;
    if (uses_fine_band(c))
        fine = default_box_fine_depth(object, kLambda, kSigma);
    geometry::detail::BoxGrading g =
        geometry::detail::box_grading(hm, depth, c.box_mesh_size, fine);
    geometry::TriangleMesh mesh =
        geometry::make_mesh_from_height_map(hm, depth, c.box_mesh_size, fine);
    const Index top = 2 * (hm.z.rows() - 1) * (hm.z.cols() - 1);
    return Geometry{std::move(mesh), depth, fine, std::move(g), top};
}

inline std::shared_ptr<excitation::GaussianBeam> make_beam(const Case& c) {
    excitation::GaussianBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.L / c.waist_factor;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = 0.0;
    bp.polarization = excitation::Polarization::P;  // E along x at normal incidence
    return std::make_shared<excitation::GaussianBeam>(bp);
}

inline SimulationConfig make_config(const Case& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = object_material(c.material);
    cfg.compression = "mlfmm";
    cfg.mlfmm.accuracy_digits = c.digits;
    cfg.mlfmm.max_exact_far_bytes = c.max_exact_far_bytes;
    cfg.formulation = c.formulation;
    cfg.diagonal_preconditioner = c.jacobi;
    cfg.gmres.tolerance = c.tolerance;
    cfg.gmres.max_iter = c.max_iter;
    cfg.gmres.restart = c.restart;
    cfg.gmres.verbose = true;
    return cfg;
}

/// Estimated number of ordered basis pairs of the exact far part of region R2 (ADR 0008 §6
/// fallback of a lossy interior): pairs whose support bounding boxes are closer than x*(d0) /
/// alpha (alpha = -Im k2, the decay bound delta > 10^-(d0+1) of far_operator.hpp), minus the near
/// pairs (estimate_near_bytes / 96 bytes). Assumes, as measured for Ag (WP21, WP22a), that the
/// expansion fails on the levels that hold these pairs; 0 for a region without decay strong
/// enough for the fallback (x* / alpha > 2 um: Si). A geometric count without assembly, O(N
/// candidates per basis).
inline Index estimate_exact_pairs(const basis::RwgSpace& space, Complex k2, Real digits,
                                  Index near_pairs) {
    const Real alpha = -k2.imag();
    if (!(alpha > 0.0))
        return 0;
    const Real reach = mlfmm::truncation_decay_exponent(digits) / alpha;
    if (reach > 2e-6)
        return 0;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index n = space.size();
    std::vector<Vec3> lo(static_cast<std::size_t>(n)), hi(static_cast<std::size_t>(n));
    Real extent = 0.0;
    for (Index b = 0; b < n; ++b) {
        const std::array<Index, 4> v = {mesh.edges()(b, 0), mesh.edges()(b, 1),
                                        space.plus_free_vertex(b), space.minus_free_vertex(b)};
        Vec3 l = mesh.vertices().row(v[0]).transpose();
        Vec3 h = l;
        for (const Index vi : v) {
            const Vec3 p = mesh.vertices().row(vi).transpose();
            l = l.cwiseMin(p);
            h = h.cwiseMax(p);
        }
        lo[static_cast<std::size_t>(b)] = l;
        hi[static_cast<std::size_t>(b)] = h;
        extent = std::max(extent, (h - l).maxCoeff());
    }
    // Uniform grid over the lower bbox corners with cell >= reach + extent: partners lie in the
    // 27 neighbouring cells.
    const Real cell = reach + extent;
    Vec3 origin = lo[0];
    for (const Vec3& l : lo) origin = origin.cwiseMin(l);
    const auto key = [&](const Vec3& p, int dx, int dy, int dz) {
        const auto ix = static_cast<std::int64_t>(std::floor((p.x() - origin.x()) / cell)) + dx;
        const auto iy = static_cast<std::int64_t>(std::floor((p.y() - origin.y()) / cell)) + dy;
        const auto iz = static_cast<std::int64_t>(std::floor((p.z() - origin.z()) / cell)) + dz;
        return (ix * 1048576 + iy) * 1048576 + iz;
    };
    std::unordered_map<std::int64_t, std::vector<Index>> grid;
    for (Index b = 0; b < n; ++b) grid[key(lo[static_cast<std::size_t>(b)], 0, 0, 0)].push_back(b);
    Index within = 0;  // ordered pairs (including m = n)
    for (Index b = 0; b < n; ++b) {
        const Vec3& lb = lo[static_cast<std::size_t>(b)];
        const Vec3& hb = hi[static_cast<std::size_t>(b)];
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto it = grid.find(key(lb, dx, dy, dz));
                    if (it == grid.end())
                        continue;
                    for (const Index o : it->second) {
                        const Vec3& lo2 = lo[static_cast<std::size_t>(o)];
                        const Vec3& hi2 = hi[static_cast<std::size_t>(o)];
                        const Vec3 gap = (lo2 - hb).cwiseMax(lb - hi2).cwiseMax(Vec3::Zero());
                        if (gap.norm() < reach)
                            ++within;
                    }
                }
    }
    return std::max<Index>(0, within - near_pairs);
}

/// Mesh, octree and memory estimates of a case, without assembly.
struct Estimate {
    Index triangles = 0;
    Index top_triangles = 0;
    Index unknowns = 0;    ///< 2N
    Index box_levels = 0;  ///< M of the graded box
    Index fine_rows = 0;
    Real depth = 0;
    Real fine_depth = 0;          ///< 0: no fine band
    Real max_support_radius = 0;  ///< r_max [m]
    int levels = 0;               ///< octree levels incl. the root (after the leaf rule)
    Real leaf_edge = 0;           ///< [m]
    Index leaves = 0;
    std::size_t near_bytes = 0;   ///< mlfmm::estimate_near_bytes
    Index exact_pairs = 0;        ///< estimate_exact_pairs (ordered basis pairs of R2)
    std::size_t exact_bytes = 0;  ///< exact_pairs x 40 bytes
    std::size_t exact_budget = 0;
    /// Leaf sampling order per region from mlfmm::leaf_sampling (0: region without a usable
    /// leaf expansion, e.g. the Ag interior).
    std::array<int, 2> leaf_order{};
    /// Leaf radiation patterns of both regions: N x 2 (L + 1)^2 directions x 2 components x 16
    /// bytes per expanded region (the dominant part of the far memory).
    std::size_t pattern_bytes = 0;
    /// Peak working set model of the setup: 2 near + 1.1 patterns + 2 exact + 1 GB (calibrated on
    /// the first runs of the study: the sparse near-field assembly holds about twice its
    /// result), without the Krylov basis.
    Real peak_setup_bytes = 0;
};

/// Result of one solved case.
struct Result {
    Estimate est;
    std::string formulation;
    bool jacobi = false;
    int iterations = 0;
    bool converged = false;
    Real final_residual = 0;  ///< monitored (preconditioned) residual
    Real true_residual = 0;   ///< |b - Z x| / |b|
    Real assembly_s = 0;      ///< Simulation::assemble (octree, far, near, rhs, Jacobi diagonal)
    Real tree_s = 0, far_s = 0, near_s = 0;  ///< parts of the operator construction
    Real exact_s = 0;                        ///< exact parts (included in far_s)
    Real matvec_s = 0;                       ///< median MlfmmOperator::apply
    Real near_matvec_s = 0, far_matvec_s = 0;
    Real solve_s = 0;  ///< GMRES wall time
    std::size_t near_bytes = 0;
    std::size_t near_nnz = 0;
    std::array<std::size_t, 2> exact_bytes{};  ///< per region (R1, R2)
    std::array<Index, 2> exact_pairs{};
    std::size_t far_bytes = 0;  ///< far operator without the exact parts (tables, workspaces)
    std::size_t operator_bytes = 0;
    Real krylov_bytes = 0;    ///< (iterations + 1) 2N 16 bytes (full GMRES)
    Real peak_setup_rss = 0;  ///< peak working set after Simulation::assemble [bytes]
    Real peak_rss = 0;        ///< at the end [bytes]
    std::string describe;     ///< MlfmmOperator::describe()
    std::string report;       ///< Simulation::report()
};

/// Estimate for a built geometry (no assembly; seconds).
inline Estimate estimate(const Geometry& geo, const Case& c) {
    const Simulation sim(geo.mesh, make_beam(c), make_config(c));
    const basis::RwgSpace& space = *sim.problem().space;
    Estimate e;
    e.triangles = geo.mesh.num_triangles();
    e.top_triangles = geo.top_triangles;
    e.unknowns = sim.num_unknowns();
    e.box_levels = geo.grading.levels;
    e.fine_rows = geo.grading.fine_rows;
    e.depth = geo.depth;
    e.fine_depth = geo.fine_depth.value_or(0.0);
    e.max_support_radius = mlfmm::max_support_radius(space);
    const mlfmm::OctreeParams op = mlfmm::leaf_rule_params(space, kLambda, sim.config().mlfmm);
    const mlfmm::Octree tree(space, kLambda, op);
    e.levels = tree.levels();
    e.leaf_edge = tree.box_size(tree.leaf_level());
    e.leaves = static_cast<Index>(tree.boxes_at_level(tree.leaf_level()).size());
    e.near_bytes = mlfmm::estimate_near_bytes(tree);
    const auto near_pairs = static_cast<Index>(e.near_bytes / 96);
    const Complex k2 =
        sim.config().object.wavenumber(2.0 * constants::pi * constants::c0 / kLambda);
    e.exact_pairs = estimate_exact_pairs(space, k2, c.digits, near_pairs);
    e.exact_bytes = static_cast<std::size_t>(e.exact_pairs) * 40;
    e.exact_budget =
        c.max_exact_far_bytes > 0
            ? c.max_exact_far_bytes
            : std::min(std::max(static_cast<std::size_t>(mlfmm::kExactFarNearFactor *
                                                         static_cast<Real>(e.near_bytes)),
                                mlfmm::kExactFarMinBytes),
                       std::size_t{16} * static_cast<std::size_t>(e.unknowns) *
                           static_cast<std::size_t>(e.unknowns));
    const Real omega = 2.0 * constants::pi * constants::c0 / kLambda;
    const std::array<Complex, 2> k = {sim.config().exterior.wavenumber(omega), k2};
    for (std::size_t i = 0; i < 2; ++i) {
        // A region whose leaf order search fails (the Ag interior: exact / truncated far part)
        // has no leaf patterns. (Si with lambda/0.7 leaves has alpha D = 1.5 and still expands.)
        try {
            const mlfmm::LeafSampling ls = mlfmm::leaf_sampling(space, tree, k[i], c.digits);
            if (!ls.search.achievable)
                continue;
            e.leaf_order[i] = ls.sampling_order;
            const auto dirs =
                static_cast<std::size_t>(2 * (ls.sampling_order + 1) * (ls.sampling_order + 1));
            e.pattern_bytes += static_cast<std::size_t>(e.unknowns / 2) * dirs * 2 * 16;
        } catch (const std::exception&) {
            // underflow / invalid order: no usable expansion, no patterns
        }
    }
    e.peak_setup_bytes = 2.0 * static_cast<Real>(e.near_bytes) +
                         1.1 * static_cast<Real>(e.pattern_bytes) +
                         2.0 * static_cast<Real>(e.exact_bytes) + 1e9;
    return e;
}

/// Number following `marker` in `text` (0 if absent).
inline Real number_after(const std::string& text, const std::string& marker) {
    const auto pos = text.find(marker);
    if (pos == std::string::npos)
        return 0.0;
    std::istringstream is(text.substr(pos + marker.size()));
    Real v = 0.0;
    is >> v;
    return is ? v : 0.0;
}

/// Wall time of one call of f [s].
inline Real seconds_of(const std::function<void()>& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Median of a non-empty sample.
inline Real median(std::vector<Real> t) {
    if (t.empty())
        throw std::invalid_argument("mlfmm_scaling::median: empty sample");
    std::sort(t.begin(), t.end());
    const std::size_t m = t.size() / 2;
    return t.size() % 2 == 1 ? t[m] : 0.5 * (t[m - 1] + t[m]);
}

/// Assembles and solves the case through Simulation (compression "mlfmm"), timing the matvec.
inline Result run(const Geometry& geo, const Case& c) {
    using Clock = std::chrono::steady_clock;
    Result r;
    r.est = estimate(geo, c);
    Simulation sim(geo.mesh, make_beam(c), make_config(c));
    r.formulation = formulation::make_formulation(sim.formulation_kind())->name();
    r.jacobi = sim.diagonal_preconditioner();
    {
        const auto t0 = Clock::now();
        sim.assemble();
        r.assembly_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    }
    r.peak_setup_rss = system_memory::peak_rss_bytes();
    const auto op = std::dynamic_pointer_cast<const mlfmm::MlfmmOperator>(sim.system_operator());
    if (op == nullptr)
        throw std::logic_error("mlfmm_scaling::run: the system operator is not an MLFMM");
    r.describe = op->describe();
    r.tree_s = number_after(r.describe, "built in ");
    r.near_s = number_after(r.describe, "assembled in ");
    r.far_s = number_after(r.describe, "\n  far: ");
    r.near_bytes = op->near_operator().memory_bytes();
    r.near_nnz = static_cast<std::size_t>(op->near_operator().nonzeros());
    std::size_t exact_total = 0;
    for (int region = 0; region < 2; ++region) {
        const mlfmm::ExactPartInfo& xi = op->far_operator().exact_info(region);
        const auto i = static_cast<std::size_t>(region);
        r.exact_bytes[i] = xi.bytes;
        r.exact_pairs[i] = xi.pairs;
        r.exact_s += xi.seconds;
        exact_total += xi.bytes;
    }
    {
        // Timed applies on a fixed random vector.
        std::mt19937_64 rng(12345);
        std::normal_distribution<Real> nd;
        VectorXc x(sim.num_unknowns());
        for (Index i = 0; i < x.size(); ++i) x(i) = Complex(nd(rng), nd(rng));
        VectorXc y(x.size());
        // Interleaved (full, near, far per round, after one untimed round): transients of the
        // machine (other processes, memory released after the setup) hit all three alike.
        const std::array<std::function<void()>, 3> calls = {
            [&] { op->apply(x, y); }, [&] { op->near_operator().apply(x, y); },
            [&] { op->far_operator().apply(x, y); }};
        std::array<std::vector<Real>, 3> t;
        for (int round = 0; round <= std::max(1, c.matvecs); ++round)
            for (std::size_t k = 0; k < 3; ++k) {
                const Real s = seconds_of(calls[k]);
                if (round > 0)
                    t[k].push_back(s);
            }
        r.matvec_s = median(t[0]);
        r.near_matvec_s = median(t[1]);
        r.far_matvec_s = median(t[2]);
    }
    if (c.solve) {
        const solver::GmresResult g = sim.solve();
        r.iterations = g.iterations;
        r.converged = g.converged;
        r.final_residual = g.residual_history.empty() ? 0.0 : g.residual_history.back();
        r.true_residual = g.true_relative_residual;
        r.solve_s = g.wall_seconds;
        const int basis = c.restart > 0 ? std::min(g.iterations, c.restart) : g.iterations;
        r.krylov_bytes = static_cast<Real>(basis + 1) * static_cast<Real>(sim.num_unknowns()) *
                         static_cast<Real>(sizeof(Complex));
        r.report = sim.report();
    }
    r.far_bytes = op->far_operator().memory_bytes() - exact_total;
    r.operator_bytes = op->memory_bytes();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// Case label, e.g. "si L = 4 um, box 100 nm, fine band".
inline std::string label(const Case& c) {
    std::ostringstream os;
    os << c.material << " L = " << c.L * 1e6 << " um, h = " << c.mesh_size * 1e9 << " nm, box "
       << (c.box_mesh_size
               ? std::to_string(static_cast<int>(std::lround(*c.box_mesh_size * 1e9))) + " nm"
               : std::string("auto"))
       << (uses_fine_band(c) ? ", fine band" : ", no fine band") << ", w0 = L/" << c.waist_factor
       << ", seed " << c.seed << ", d0 = " << c.digits << ", tol = " << c.tolerance;
    return os.str();
}

inline Real gb(Real bytes) {
    return bytes / 1e9;
}
inline Real gb(std::size_t bytes) {
    return static_cast<Real>(bytes) / 1e9;
}

/// Multi-line summary of an estimate.
inline std::string summary(const Estimate& e) {
    std::ostringstream os;
    os << "triangles " << e.triangles << " (top " << e.top_triangles << ", box "
       << e.triangles - e.top_triangles << " = "
       << 100.0 * static_cast<Real>(e.triangles - e.top_triangles) /
              static_cast<Real>(e.top_triangles)
       << " %), box M = " << e.box_levels << ", fine rows " << e.fine_rows << ", depth "
       << e.depth * 1e6 << " um, fine band " << e.fine_depth * 1e6 << " um; 2N = " << e.unknowns
       << "; r_max " << e.max_support_radius * 1e9 << " nm; octree " << e.levels
       << " levels, leaf edge " << e.leaf_edge * 1e9 << " nm (lambda / " << kLambda / e.leaf_edge
       << "), " << e.leaves << " leaves; near estimate " << gb(e.near_bytes)
       << " GB, exact R2 estimate " << gb(e.exact_bytes) << " GB (" << e.exact_pairs
       << " pairs), budget " << gb(e.exact_budget) << " GB; leaf orders " << e.leaf_order[0]
       << " / " << e.leaf_order[1] << ", leaf patterns " << gb(e.pattern_bytes)
       << " GB; setup peak estimate " << gb(e.peak_setup_bytes) << " GB";
    return os.str();
}

/// Multi-line summary of a result.
inline std::string summary(const Result& r) {
    std::ostringstream os;
    os << summary(r.est) << "\n"
       << "formulation " << r.formulation << (r.jacobi ? " + left Jacobi" : "") << "; GMRES "
       << r.iterations << " iterations, converged " << (r.converged ? "yes" : "no")
       << ", monitored residual " << r.final_residual << ", true residual " << r.true_residual
       << "\nsetup " << r.assembly_s << " s (octree " << r.tree_s << " s, far " << r.far_s
       << " s incl. exact parts " << r.exact_s << " s, near " << r.near_s << " s); matvec "
       << r.matvec_s << " s (near " << r.near_matvec_s << " s, far " << r.far_matvec_s
       << " s); solve " << r.solve_s << " s (" << r.solve_s / std::max(1, r.iterations)
       << " s/it)\nmemory: near " << gb(r.near_bytes) << " GB (nnz " << r.near_nnz << ", "
       << static_cast<Real>(r.near_nnz) / static_cast<Real>(r.est.unknowns)
       << " per row), exact R1 " << gb(r.exact_bytes[0]) << " GB, exact R2 " << gb(r.exact_bytes[1])
       << " GB (" << r.exact_pairs[1] << " pairs), far " << gb(r.far_bytes) << " GB, operator "
       << gb(r.operator_bytes) << " GB, Krylov " << gb(r.krylov_bytes) << " GB, peak after setup "
       << gb(r.peak_setup_rss) << " GB, peak " << gb(r.peak_rss) << " GB";
    return os.str();
}

/// One machine-readable line (key=value, SI seconds, GB), parsed by
/// benchmarks/mlfmm_scaling_fit.py.
inline std::string row(const Case& c, const Result& r) {
    std::ostringstream os;
    os.precision(6);
    os << "ROW material=" << c.material << " L_um=" << c.L * 1e6 << " box="
       << (c.box_mesh_size ? std::to_string(*c.box_mesh_size * 1e9) : std::string("auto"))
       << " fine=" << (uses_fine_band(c) ? 1 : 0) << " restart=" << c.restart
       << " solved=" << (c.solve ? 1 : 0) << " N2=" << r.est.unknowns << " levels=" << r.est.levels
       << " leaf_nm=" << r.est.leaf_edge * 1e9 << " rmax_nm=" << r.est.max_support_radius * 1e9
       << " setup_s=" << r.assembly_s << " tree_s=" << r.tree_s << " far_s=" << r.far_s
       << " exact_s=" << r.exact_s << " near_s=" << r.near_s << " matvec_s=" << r.matvec_s
       << " near_mv_s=" << r.near_matvec_s << " far_mv_s=" << r.far_matvec_s
       << " it=" << r.iterations << " conv=" << (r.converged ? 1 : 0) << " res=" << r.final_residual
       << " true_res=" << r.true_residual << " solve_s=" << r.solve_s
       << " near_gb=" << gb(r.near_bytes) << " exact_gb=" << gb(r.exact_bytes[0] + r.exact_bytes[1])
       << " exact_est_gb=" << gb(r.est.exact_bytes) << " pattern_est_gb=" << gb(r.est.pattern_bytes)
       << " peak_est_gb=" << gb(r.est.peak_setup_bytes) << " far_gb=" << gb(r.far_bytes)
       << " krylov_gb=" << gb(r.krylov_bytes) << " peak_setup_gb=" << gb(r.peak_setup_rss)
       << " peak_gb=" << gb(r.peak_rss);
    return os.str();
}

}  // namespace mlfmm_scaling
