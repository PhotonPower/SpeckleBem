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
///    131 072 triangles and 2N = 393 216 unknowns), octasphere::make_octasphere
///    (octasphere.hpp: the same recursive midpoint subdivision with projection as the icosphere).
/// eps_rr of docs/05 (E = sqrt(sigma), RMS over the angles / max E_ref) in the xz-plane (phi = 0,
/// bistatic_rcs plane normal +y) and the yz-plane (phi = pi / 2, plane normal -x), theta = 0 ...
/// 180 degrees in kRcsAngles points (1 degree, as benchmarks/results/mie_sphere_dense.md) and, as a
/// sampling check, in kRcsAnglesFine points.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "octasphere.hpp"
#include "system_memory.hpp"

namespace ag_sphere_mlfmm {

using namespace specklebem;
using octasphere::edge_stats;
using octasphere::EdgeStats;
using octasphere::make_octasphere;

inline constexpr Real kLambda = 500e-9;
inline constexpr Index kRcsAngles = 181;       ///< 1 degree spacing
inline constexpr Index kRcsAnglesFine = 1801;  ///< 0.1 degree spacing (sampling check)

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
    e.exact_budget =
        c.max_exact_far_bytes > 0
            ? c.max_exact_far_bytes
            : std::min(std::max(static_cast<std::size_t>(mlfmm::kExactFarNearFactor *
                                                         static_cast<Real>(e.near_bytes)),
                                mlfmm::kExactFarMinBytes),
                       std::size_t{16} * static_cast<std::size_t>(e.unknowns) *
                           static_cast<std::size_t>(e.unknowns));
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
    const reference::MieSolution mie(reference::MieParams{
        0.5 * c.diameter, kLambda, material::silver_500nm(), material::vacuum(), 0});
    r.eps_xz = eps_rr(sim.solution(), mie, kRcsAngles, Vec3::UnitY(), 0.0);
    r.eps_yz = eps_rr(sim.solution(), mie, kRcsAngles, -Vec3::UnitX(), 0.5 * constants::pi);
    r.eps_xz_fine = eps_rr(sim.solution(), mie, kRcsAnglesFine, Vec3::UnitY(), 0.0);
    r.eps_yz_fine =
        eps_rr(sim.solution(), mie, kRcsAnglesFine, -Vec3::UnitX(), 0.5 * constants::pi);
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
       << "formulation " << r.formulation << (r.jacobi ? " + left Jacobi" : "")
       << "; eps_rr xz = " << r.eps_xz << ", yz = " << r.eps_yz << " (" << kRcsAngles
       << " angles), " << r.eps_xz_fine << " / " << r.eps_yz_fine << " (" << kRcsAnglesFine
       << " angles)\n"
       << "GMRES " << r.iterations << " iterations, converged " << (r.converged ? "yes" : "no")
       << ", true residual " << r.true_residual << "; assembly " << r.assembly_s << " s, solve "
       << r.solve_s << " s (" << r.solve_s / std::max(1, r.iterations) << " s/it), RCS " << r.post_s
       << " s\n"
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
