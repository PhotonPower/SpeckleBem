#pragma once
/// @file ag_sphere_mlfmm_support.hpp
/// The Ag sphere of the Phase 4 DoD (docs/01, docs/05 row "Ag sphere d = 4 um, lambda/27, MLFMM")
/// solved through Simulation with compression "mlfmm" and compared with Mie (WP22a). Shared by
/// tests/validation_large/test_ag_sphere_mlfmm_large.cpp and the study executable
/// benchmarks/ag_sphere_mlfmm.cpp, so it must not depend on Catch2.
///
/// Setup (docs/06): sphere centred at the origin in vacuum, lambda = 500 nm, Ag eps_r = -9.794 -
/// 0.313j (material::silver_500nm), plane wave k = +z, E along x (1 V/m). Meshes:
///  * "ico": geometry::make_icosphere (20 4^n triangles);
///  * "octa": the octahedron-based sphere of the paper (8 4^n triangles; n = 7 gives the paper's
///    131 072 triangles and 2N = 393 216 unknowns), built here by the same recursive midpoint
///    subdivision with projection as the icosphere.
/// eps_rr of docs/05 (E = sqrt(sigma), RMS over the angles / max E_ref) in the xz-plane (phi = 0,
/// bistatic_rcs plane normal +y) and the yz-plane (phi = pi / 2, plane normal -x), theta = 0 ...
/// 180 degrees in kRcsAngles points (1 degree, as benchmarks/results/mie_sphere_dense.md) and, as a
/// sampling check, in kRcsAnglesFine points.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/reference/mie.hpp"
#include "specklebem/simulation.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "system_memory.hpp"

namespace ag_sphere_mlfmm {

using namespace specklebem;

inline constexpr Real kLambda = 500e-9;
inline constexpr Index kRcsAngles = 181;       ///< 1 degree spacing
inline constexpr Index kRcsAnglesFine = 1801;  ///< 0.1 degree spacing (sampling check)

/// Octahedron-based sphere: regular octahedron refined `subdivisions` times by midpoint
/// subdivision with the midpoints projected onto the sphere (as geometry::make_icosphere), 8 4^n
/// triangles, 4 4^n + 2 vertices, 12 4^n edges; counter-clockwise seen from outside (normals out
/// of R2 into R1, docs/06).
/// @throws std::invalid_argument for subdivisions outside [0, 9] or radius not finite and > 0.
inline geometry::TriangleMesh make_octasphere(Real radius, int subdivisions) {
    if (subdivisions < 0 || subdivisions > 9)
        throw std::invalid_argument("make_octasphere: subdivisions must lie in [0, 9]");
    if (!(radius > 0.0) || !std::isfinite(radius))
        throw std::invalid_argument("make_octasphere: radius must be finite and > 0");
    using Face = std::array<Index, 3>;
    std::vector<Vec3> v = {Vec3::UnitX(), -Vec3::UnitX(), Vec3::UnitY(),
                           -Vec3::UnitY(), Vec3::UnitZ(), -Vec3::UnitZ()};
    std::vector<Face> f = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
                           {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
    for (Face& t : f) {
        const Vec3& a = v[static_cast<std::size_t>(t[0])];
        const Vec3& b = v[static_cast<std::size_t>(t[1])];
        const Vec3& c = v[static_cast<std::size_t>(t[2])];
        if ((b - a).cross(c - a).dot(a + b + c) < 0.0)
            std::swap(t[1], t[2]);
    }
    for (int level = 0; level < subdivisions; ++level) {
        const std::size_t nf = f.size();
        const auto stride = static_cast<std::uint64_t>(v.size() + 3 * nf / 2);
        std::unordered_map<std::uint64_t, Index> cache;
        cache.reserve(3 * nf / 2);
        const auto midpoint = [&](Index a, Index b) -> Index {
            if (a > b)
                std::swap(a, b);
            const std::uint64_t key =
                static_cast<std::uint64_t>(a) * stride + static_cast<std::uint64_t>(b);
            const auto [it, inserted] = cache.try_emplace(key, 0);
            if (inserted) {
                const Vec3 m =
                    (v[static_cast<std::size_t>(a)] + v[static_cast<std::size_t>(b)]).normalized();
                it->second = static_cast<Index>(v.size());
                v.push_back(m);
            }
            return it->second;
        };
        std::vector<Face> refined;
        refined.reserve(4 * nf);
        for (const Face& t : f) {
            const Index ab = midpoint(t[0], t[1]);
            const Index bc = midpoint(t[1], t[2]);
            const Index ca = midpoint(t[2], t[0]);
            refined.push_back({t[0], ab, ca});
            refined.push_back({t[1], bc, ab});
            refined.push_back({t[2], ca, bc});
            refined.push_back({ab, bc, ca});
        }
        f = std::move(refined);
    }
    Vertices vertices(static_cast<Index>(v.size()), 3);
    for (std::size_t i = 0; i < v.size(); ++i)
        vertices.row(static_cast<Index>(i)) = (radius * v[i]).transpose();
    Triangles triangles(static_cast<Index>(f.size()), 3);
    for (std::size_t t = 0; t < f.size(); ++t)
        for (Index k = 0; k < 3; ++k)
            triangles(static_cast<Index>(t), k) = f[t][static_cast<std::size_t>(k)];
    return geometry::TriangleMesh(std::move(vertices), std::move(triangles));
}

/// Sphere mesh "ico" (icosphere) or "octa" (octahedron-based) of the given diameter.
/// @throws std::invalid_argument for another kind.
inline geometry::TriangleMesh make_mesh(const std::string& kind, Real diameter, int subdivisions) {
    if (kind == "ico")
        return geometry::make_icosphere(0.5 * diameter, subdivisions);
    if (kind == "octa")
        return make_octasphere(0.5 * diameter, subdivisions);
    throw std::invalid_argument("make_mesh: mesh kind must be \"ico\" or \"octa\", got \"" + kind +
                                "\"");
}

/// Edge statistics of a mesh [m].
struct EdgeStats {
    Real mean = 0, min = 0, max = 0;
};

inline EdgeStats edge_stats(const geometry::TriangleMesh& mesh) {
    EdgeStats s;
    s.min = 1e300;
    Real sum = 0.0;
    for (Index e = 0; e < mesh.num_edges(); ++e) {
        const Vec3 a = mesh.vertices().row(mesh.edges()(e, 0)).transpose();
        const Vec3 b = mesh.vertices().row(mesh.edges()(e, 1)).transpose();
        const Real l = (b - a).norm();
        sum += l;
        s.min = std::min(s.min, l);
        s.max = std::max(s.max, l);
    }
    s.mean = sum / static_cast<Real>(mesh.num_edges());
    return s;
}

/// One case of the study.
struct Case {
    std::string mesh = "octa";  ///< "ico" or "octa"
    int subdivisions = 7;
    Real diameter = 4e-6;  ///< [m]
    Real digits = 3.0;     ///< accuracy_digits d0
    Real tolerance = 1e-3;
    int max_iter = 3000;
    std::size_t max_exact_far_bytes = 0;  ///< 0: automatic budget (2 x near field)
    /// Empty: formulation::recommend (ICTF + left Jacobi for Ag).
    std::optional<formulation::Kind> formulation;
    std::optional<bool> jacobi;
};

/// Mesh, octree and memory estimates of a case, without assembly.
struct Estimate {
    Index triangles = 0;
    Index unknowns = 0;  ///< 2N
    EdgeStats edges;
    Real max_support_radius = 0;  ///< r_max [m]
    int levels = 0;               ///< octree levels incl. the root (after the leaf rule)
    Real leaf_edge = 0;           ///< [m]
    Index leaves = 0;
    std::size_t near_bytes = 0;  ///< mlfmm::estimate_near_bytes
    std::size_t exact_budget = 0;
};

/// Result of one solved case.
struct Result {
    Estimate est;
    std::string formulation;
    bool jacobi = false;
    Real eps_xz = 0, eps_yz = 0;            ///< kRcsAngles
    Real eps_xz_fine = 0, eps_yz_fine = 0;  ///< kRcsAnglesFine
    int iterations = 0;
    bool converged = false;
    Real true_residual = 0;
    Real assembly_s = 0;  ///< Simulation::assemble (octree, far setup, exact parts, near field,
                          ///< rhs, Jacobi diagonal)
    Real solve_s = 0;     ///< GMRES wall time
    Real post_s = 0;      ///< RCS evaluation
    std::size_t near_bytes = 0;
    std::size_t near_nnz = 0;
    std::array<std::size_t, 2> exact_bytes{};  ///< per region (R1, R2)
    std::array<Index, 2> exact_pairs{};
    std::array<Index, 2> truncated_pairs{};
    std::size_t far_bytes = 0;  ///< far operator without the exact parts (tables, workspaces)
    std::size_t operator_bytes = 0;
    Real krylov_bytes = 0;  ///< (iterations + 1) 2N 16 bytes of the full-GMRES basis
    Real peak_rss = 0;      ///< process peak working set [bytes]
    std::string describe;   ///< MlfmmOperator::describe()
    std::string report;     ///< Simulation::report()
};

inline SimulationConfig make_config(const Case& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = material::silver_500nm();
    cfg.compression = "mlfmm";
    cfg.mlfmm.accuracy_digits = c.digits;
    cfg.mlfmm.max_exact_far_bytes = c.max_exact_far_bytes;
    cfg.formulation = c.formulation;
    cfg.diagonal_preconditioner = c.jacobi;
    cfg.gmres.tolerance = c.tolerance;
    cfg.gmres.max_iter = c.max_iter;
    cfg.gmres.restart = 0;
    cfg.gmres.verbose = true;
    return cfg;
}

inline std::shared_ptr<excitation::PlaneWave> make_wave() {
    return std::make_shared<excitation::PlaneWave>(kLambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
}

/// Mesh statistics, octree after the leaf rule and the near-field estimate (cheap: no assembly).
inline Estimate estimate(const geometry::TriangleMesh& mesh, const Case& c) {
    const Simulation sim(mesh, make_wave(), make_config(c));
    const basis::RwgSpace& space = *sim.problem().space;
    Estimate e;
    e.triangles = mesh.num_triangles();
    e.unknowns = sim.num_unknowns();
    e.edges = edge_stats(mesh);
    e.max_support_radius = mlfmm::max_support_radius(space);
    const mlfmm::OctreeParams op = mlfmm::leaf_rule_params(space, kLambda, sim.config().mlfmm);
    const mlfmm::Octree tree(space, kLambda, op);
    e.levels = tree.levels();
    e.leaf_edge = tree.box_size(tree.leaf_level());
    e.leaves = static_cast<Index>(tree.boxes_at_level(tree.leaf_level()).size());
    e.near_bytes = mlfmm::estimate_near_bytes(tree);
    e.exact_budget = c.max_exact_far_bytes > 0
                         ? c.max_exact_far_bytes
                         : std::max(static_cast<std::size_t>(mlfmm::kExactFarNearFactor *
                                                             static_cast<Real>(e.near_bytes)),
                                    mlfmm::kExactFarMinBytes);
    return e;
}

/// eps_rr of docs/05 over `na` angles theta in [0, pi] in the plane with normal plane_normal
/// (bistatic_rcs) against Mie at azimuth mie_phi.
inline Real eps_rr(const post::SurfaceSolution& s, const reference::MieSolution& mie, Index na,
                   const Vec3& plane_normal, Real mie_phi) {
    const VectorXr theta = VectorXr::LinSpaced(na, 0.0, constants::pi);
    const VectorXr sigma = post::bistatic_rcs(s, plane_normal, theta);
    Real sum = 0.0;
    Real max_ref = 0.0;
    for (Index i = 0; i < na; ++i) {
        const Real ref = std::sqrt(mie.bistatic_rcs(theta(i), mie_phi));
        const Real d = ref - std::sqrt(sigma(i));
        sum += d * d;
        max_ref = std::max(max_ref, ref);
    }
    return std::sqrt(sum / static_cast<Real>(na)) / max_ref;
}

/// Assembles and solves the case through Simulation (compression "mlfmm") and compares the
/// bistatic RCS with Mie.
inline Result run(const Case& c) {
    using Clock = std::chrono::steady_clock;
    const geometry::TriangleMesh mesh = make_mesh(c.mesh, c.diameter, c.subdivisions);
    Result r;
    r.est = estimate(mesh, c);
    Simulation sim(mesh, make_wave(), make_config(c));
    r.formulation = formulation::make_formulation(sim.formulation_kind())->name();
    r.jacobi = sim.diagonal_preconditioner();
    {
        const auto t0 = Clock::now();
        sim.assemble();
        r.assembly_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    }
    const auto op = std::dynamic_pointer_cast<const mlfmm::MlfmmOperator>(sim.system_operator());
    if (op == nullptr)
        throw std::logic_error("ag_sphere_mlfmm::run: the system operator is not an MLFMM");
    r.describe = op->describe();
    r.near_bytes = op->near_operator().memory_bytes();
    r.near_nnz = static_cast<std::size_t>(op->near_operator().nonzeros());
    std::size_t exact_total = 0;
    for (int region = 0; region < 2; ++region) {
        const mlfmm::ExactPartInfo& xi = op->far_operator().exact_info(region);
        const auto i = static_cast<std::size_t>(region);
        r.exact_bytes[i] = xi.bytes;
        r.exact_pairs[i] = xi.pairs;
        r.truncated_pairs[i] = xi.truncated_pairs;
        exact_total += xi.bytes;
    }
    const solver::GmresResult g = sim.solve();
    r.iterations = g.iterations;
    r.converged = g.converged;
    r.true_residual = g.true_relative_residual;
    r.solve_s = g.wall_seconds;
    // After the solve: the far apply workspaces are allocated now.
    r.far_bytes = op->far_operator().memory_bytes() - exact_total;
    r.operator_bytes = op->memory_bytes();
    r.krylov_bytes = static_cast<Real>(g.iterations + 1) * static_cast<Real>(sim.num_unknowns()) *
                     static_cast<Real>(sizeof(Complex));
    const auto t0 = Clock::now();
    const reference::MieSolution mie(reference::MieParams{0.5 * c.diameter, kLambda,
                                                          material::silver_500nm(),
                                                          material::vacuum(), 0});
    r.eps_xz = eps_rr(sim.solution(), mie, kRcsAngles, Vec3::UnitY(), 0.0);
    r.eps_yz = eps_rr(sim.solution(), mie, kRcsAngles, -Vec3::UnitX(), 0.5 * constants::pi);
    r.eps_xz_fine = eps_rr(sim.solution(), mie, kRcsAnglesFine, Vec3::UnitY(), 0.0);
    r.eps_yz_fine = eps_rr(sim.solution(), mie, kRcsAnglesFine, -Vec3::UnitX(), 0.5 * constants::pi);
    r.post_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// Case label, e.g. "Ag d = 4 um, octa n = 7".
inline std::string label(const Case& c) {
    std::ostringstream os;
    os << "Ag d = " << c.diameter * 1e6 << " um, " << c.mesh << " n = " << c.subdivisions
       << ", d0 = " << c.digits << ", tol = " << c.tolerance;
    return os.str();
}

/// Multi-line summary of an estimate.
inline std::string summary(const Estimate& e) {
    std::ostringstream os;
    os << "triangles " << e.triangles << ", 2N = " << e.unknowns << ", edges mean "
       << e.edges.mean * 1e9 << " nm (lambda / " << kLambda / e.edges.mean << "), min "
       << e.edges.min * 1e9 << " nm, max " << e.edges.max * 1e9 << " nm (lambda / "
       << kLambda / e.edges.max << "), r_max " << e.max_support_radius * 1e9 << " nm; octree "
       << e.levels << " levels, leaf edge " << e.leaf_edge * 1e9 << " nm (lambda / "
       << kLambda / e.leaf_edge << ", r_max / a = " << e.max_support_radius / e.leaf_edge << "), "
       << e.leaves << " leaves; near field estimate " << static_cast<Real>(e.near_bytes) / 1e9
       << " GB, exact-part budget " << static_cast<Real>(e.exact_budget) / 1e9 << " GB";
    return os.str();
}

/// Multi-line summary of a result (estimate, accuracy, timings, memory).
inline std::string summary(const Result& r) {
    std::ostringstream os;
    os << summary(r.est) << "\n"
       << "formulation " << r.formulation << (r.jacobi ? " + left Jacobi" : "") << "; eps_rr xz = "
       << r.eps_xz << ", yz = " << r.eps_yz << " (" << kRcsAngles << " angles), " << r.eps_xz_fine
       << " / " << r.eps_yz_fine << " (" << kRcsAnglesFine << " angles)\n"
       << "GMRES " << r.iterations << " iterations, converged " << (r.converged ? "yes" : "no")
       << ", true residual " << r.true_residual << "; assembly " << r.assembly_s << " s, solve "
       << r.solve_s << " s (" << r.solve_s / std::max(1, r.iterations) << " s/it), RCS "
       << r.post_s << " s\n"
       << "memory: near " << static_cast<Real>(r.near_bytes) / 1e9 << " GB (nnz " << r.near_nnz
       << ", " << static_cast<Real>(r.near_nnz) / static_cast<Real>(r.est.unknowns)
       << " per row), exact R1 " << static_cast<Real>(r.exact_bytes[0]) / 1e9 << " GB ("
       << r.exact_pairs[0] << " pairs), exact R2 " << static_cast<Real>(r.exact_bytes[1]) / 1e9
       << " GB (" << r.exact_pairs[1] << " pairs, "
       << static_cast<Real>(r.exact_pairs[1]) / static_cast<Real>(r.est.unknowns / 2)
       << " per basis; truncated " << r.truncated_pairs[1] << "), far tables + workspaces "
       << static_cast<Real>(r.far_bytes) / 1e9 << " GB, operator total "
       << static_cast<Real>(r.operator_bytes) / 1e9 << " GB, Krylov basis " << r.krylov_bytes / 1e9
       << " GB, peak RSS " << r.peak_rss / 1e9 << " GB";
    return os.str();
}

}  // namespace ag_sphere_mlfmm
