#pragma once
/// @file pml_ring_support.hpp
/// Validation 7h of the PML route (WP-PML5a/5b/5c; ADR 0008 amendment 2026-10-18 decision 6,
/// docs/05 "PML route" row 7h): the fast path (Simulation with compression "mlfmm" on the PML
/// route: MLFMM on the real functions plus the PML ring, near part exact, far part ACA by
/// default, GMRES with the recommended diagonal preconditioner) against the dense PML matrix and
/// its LU solve on the same case:
///  * matvec: max over random vectors (fixed seeds) of |Z_fast x - Z_dense x| / |Z_dense x|
///    (< 1e-3 at d0 = 3, < 1e-5 at d0 = 5);
///  * solve: R = P_refl / P_inc of the windowed far field (pml_rough_ag::FarField::reflectance)
///    within 1e-4 relative, and the windowed far field |F|^2 over the reflection hemisphere in
///    the ADR 0006 / docs/05 L2 metric (pml_rough_ag::compare, l2) < 1e-3.
/// The cases are pml_rough_ag::RoughCase (flat for rms = 0); the fast run comes first (its
/// operator applied to the matvec vectors, solve, far field, the process peak RSS of the fast
/// path), then the dense run (matvec against the stored fast products, LU in place, far field),
/// so both paths' peaks are measured in one process. Shared by tests/validation/test_pml_ring.cpp,
/// tests/validation_large/test_pml_ring_large.cpp and benchmarks/pml_ring.cpp; no Catch2.
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_rough_ag_support.hpp"

namespace pml_ring {

using namespace specklebem;
using pml_rough_ag::FarField;
using pml_rough_ag::RoughCase;

/// Seeds of the matvec vectors (entries uniform in [-1/2, 1/2)^2, real and imaginary part).
inline const std::vector<std::uint64_t> kSeeds = {101, 202, 303};

[[nodiscard]] inline VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

/// Fast-path settings.
struct FastSettings {
    Real digits = 3.0;            ///< MlfmmParams::accuracy_digits
    Real gmres_tolerance = 1e-7;  ///< relative residual of the GMRES solve
    int max_iter = 5000;
    mlfmm::PmlRingParams ring;  ///< MlfmmParams::pml_ring (WP-PML5b: ACA by default)
};

/// One ROW-line segment with the far ring statistics (WP-PML5b; exact path: "exact far ring").
[[nodiscard]] inline std::string far_segment(const mlfmm::PmlRingFarStats& s) {
    char buf[900];
    if (!s.aca) {
        std::snprintf(buf, sizeof(buf), "ring far exact (5a) %lld blocks %.1f MB",
                      static_cast<long long>(s.exact_blocks),
                      static_cast<Real>(s.exact_bytes) / 1e6);
        return buf;
    }
    std::string hist;
    for (const Index h : s.rank_histogram) hist += (hist.empty() ? "" : " ") + std::to_string(h);
    std::snprintf(
        buf, sizeof(buf),
        "ring far ACA eps %.0e eta %.2f: aca %lld blocks %.1f MB, exact %lld + cap %lld blocks "
        "%.1f "
        "MB, dropped %lld (R1 %lld, R2 %lld, max bound %.1e) | ranks min %lld med %lld max %lld "
        "mean JJ %.2f JM %.2f MJ %.2f MM %.2f | hist [%s] | gen rows %lld cols %lld | pair evals "
        "%lld, reduced %lld (R2 beyond %.3g m)",
        s.epsilon, s.eta, static_cast<long long>(s.aca_blocks),
        static_cast<Real>(s.aca_bytes) / 1e6, static_cast<long long>(s.exact_blocks),
        static_cast<long long>(s.fallback_blocks), static_cast<Real>(s.exact_bytes) / 1e6,
        static_cast<long long>(s.dropped_blocks), static_cast<long long>(s.region_dropped[0]),
        static_cast<long long>(s.region_dropped[1]), s.max_dropped_bound,
        static_cast<long long>(s.rank_min), static_cast<long long>(s.rank_median),
        static_cast<long long>(s.rank_max), s.rank_mean[0], s.rank_mean[1], s.rank_mean[2],
        s.rank_mean[3], hist.c_str(), static_cast<long long>(s.generated_rows),
        static_cast<long long>(s.generated_columns), static_cast<long long>(s.pair_evaluations),
        static_cast<long long>(s.pair_reductions), s.drop_distance[1]);
    return buf;
}

/// Result of the fast run (Simulation with compression "mlfmm").
struct FastRun {
    std::vector<VectorXc> products;  ///< Z_fast x for x = random_vector(2N, kSeeds[i])
    VectorXc x;                      ///< currents
    FarField far;
    Index unknowns = 0, stretched = 0;
    int levels = 0;
    Real leaf_lambda = 0;  ///< leaf edge / lambda_1
    std::size_t operator_bytes = 0, ring_near_bytes = 0, ring_far_bytes = 0, rr_near_bytes = 0,
                rr_far_bytes = 0;
    Index ring_near_pairs = 0, ring_far_pairs = 0;
    mlfmm::PmlRingFarStats far_stats;  ///< ring far part (ACA / exact)
    /// Simulation::assemble(): the operator (MLFMM + ring), the right-hand side and the Jacobi
    /// diagonal (ROW "fast assemble"; the operator alone is about ring_near_s + ring_far_s + the
    /// MLFMM setup, the diagonal is read from the near parts, MlfmmOperator::diagonal)
    Real assemble_s = 0;
    Real ring_near_s = 0, ring_far_s = 0;
    Real matvec_s = 0;  ///< one apply (mean over the matvec vectors)
    Real solve_s = 0;
    int iterations = 0;
    bool converged = false;
    Real true_residual = 0;
    Real peak_rss = 0;  ///< process peak after the fast run [bytes]
    std::string report;
};

/// Result of the dense run (and the comparison of the stored fast products).
struct DenseRun {
    VectorXc x;
    FarField far;
    std::vector<Real> matvec_errors;  ///< |Z_fast x - Z_dense x| / |Z_dense x| per vector
    Real assembly_s = 0, lu_s = 0, solve_s = 0, matvec_s = 0;
    Real residual = 0;  ///< |b - Z x| / |b| on 64 sampled rows
    Real dense_bytes = 0;
    Real peak_rss = 0;  ///< process peak after the dense run [bytes]
    bool lapack = false;
};

[[nodiscard]] inline Real seconds(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - a).count();
}

/// The fast path on case c: build, matvec products, GMRES solve, far field (default window).
inline FastRun run_fast(const RoughCase& c, const FastSettings& f) {
    FastRun r;
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    cfg.compression = "mlfmm";
    cfg.solver = SolverKind::Gmres;
    cfg.mlfmm.accuracy_digits = f.digits;
    cfg.mlfmm.pml_ring = f.ring;
    cfg.gmres.tolerance = f.gmres_tolerance;
    cfg.gmres.max_iter = f.max_iter;
    Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam, cfg);
    auto t0 = std::chrono::steady_clock::now();
    sim.assemble();
    r.assemble_s = seconds(t0);
    const auto* Z = dynamic_cast<const mlfmm::MlfmmOperator*>(sim.system_operator().get());
    if (Z == nullptr || Z->ring_operator() == nullptr)
        throw std::logic_error("pml_ring::run_fast: the operator is not an MLFMM with a PML ring");
    const mlfmm::PmlRingOperator& ring = *Z->ring_operator();
    r.unknowns = Z->rows();
    r.stretched = ring.stretched_count();
    r.levels = Z->octree().levels();
    r.leaf_lambda = Z->octree().box_size(Z->octree().leaf_level()) / pml_rough_ag::kLambda;
    r.operator_bytes = Z->memory_bytes();
    r.ring_near_bytes = ring.near_bytes();
    r.ring_far_bytes = ring.far_bytes();
    r.rr_near_bytes = Z->near_operator().memory_bytes();
    r.rr_far_bytes = Z->far_operator().memory_bytes();
    r.ring_near_pairs = ring.near_pairs();
    r.ring_far_pairs = ring.far_pairs();
    r.far_stats = ring.far_stats();
    r.ring_near_s = ring.near_seconds();
    r.ring_far_s = ring.far_seconds();
    t0 = std::chrono::steady_clock::now();
    for (const std::uint64_t seed : kSeeds)
        r.products.push_back(*Z * random_vector(Z->cols(), seed));
    r.matvec_s = seconds(t0) / static_cast<Real>(kSeeds.size());
    const solver::GmresResult g = sim.solve();
    r.solve_s = g.wall_seconds;
    r.iterations = g.iterations;
    r.converged = g.converged;
    r.true_residual = g.true_relative_residual;
    r.x = sim.solution().currents;
    r.far = pml_rough_ag::evaluate_far_field(sim, sim.solution(), c, *sim.far_field_window(),
                                             beam->power());
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// The dense path on case c (LU in place), with the matvec comparison of the fast products.
inline DenseRun run_dense(const RoughCase& c, const FastRun& fast) {
    DenseRun r;
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    const Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam,
                         pml_rough_ag::make_config(c));
    r.dense_bytes = est.size.dense_bytes;
    const VectorXc b = op::assemble_rhs(sim.problem());
    auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<op::LinearOperator> lin = op::DenseStrategy{}.build(sim.problem());
    const auto dense = std::dynamic_pointer_cast<op::DenseOperator>(lin);
    if (!dense)
        throw std::logic_error("pml_ring::run_dense: not a DenseOperator");
    MatrixXc Z = std::move(dense->matrix());
    lin.reset();
    r.assembly_s = seconds(t0);
    if (Z.rows() != fast.unknowns)
        throw std::logic_error("pml_ring::run_dense: the fast and dense systems differ in size");
    t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kSeeds.size(); ++i) {
        const VectorXc yd = Z * random_vector(Z.cols(), kSeeds[i]);
        r.matvec_errors.push_back((fast.products[i] - yd).norm() / yd.norm());
    }
    r.matvec_s = seconds(t0) / static_cast<Real>(kSeeds.size());
    const Index n = Z.rows();
    const Index stride = std::max<Index>(1, n / 64);
    std::vector<Index> rows;
    for (Index i = stride / 2; i < n; i += stride) rows.push_back(i);
    MatrixXc Zs(static_cast<Index>(rows.size()), n);
    for (std::size_t i = 0; i < rows.size(); ++i) Zs.row(static_cast<Index>(i)) = Z.row(rows[i]);
    {
        t0 = std::chrono::steady_clock::now();
        const pml_spp::InPlaceLu lu(std::move(Z), c.use_lapack);
        r.lapack = lu.lapack();
        r.lu_s = seconds(t0);
        t0 = std::chrono::steady_clock::now();
        r.x = lu.solve(b);
        r.solve_s = seconds(t0);
    }
    VectorXc bs(static_cast<Index>(rows.size()));
    for (std::size_t i = 0; i < rows.size(); ++i) bs(static_cast<Index>(i)) = b(rows[i]);
    r.residual = (bs - Zs * r.x).norm() / bs.norm();
    const post::SurfaceSolution s{&sim.problem(), r.x};
    r.far = pml_rough_ag::evaluate_far_field(sim, s, c, *sim.far_field_window(), beam->power());
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// Matvec-only comparison (no solve): the fast operator built from the Simulation's Problem
/// (the operator of Simulation::assemble without rhs and diagonal), its products, released;
/// then the dense matrix and its products. For the d0 = 5 gate at the 7c size.
struct MatvecRun {
    std::vector<Real> errors;  ///< |Z_fast x - Z_dense x| / |Z_dense x| per kSeeds vector
    Index unknowns = 0, stretched = 0;
    int levels = 0;
    Real leaf_lambda = 0;
    std::size_t fast_bytes = 0, ring_near_bytes = 0, ring_far_bytes = 0;
    Real fast_s = 0, ring_near_s = 0, ring_far_s = 0;  ///< operator construction [s]
    Real dense_s = 0;                                  ///< dense assembly [s]
    Real fast_matvec_s = 0, dense_matvec_s = 0;        ///< one apply (mean)
    Real peak_fast = 0, peak_dense = 0;                ///< process peak after each part [bytes]
    mlfmm::PmlRingFarStats far_stats;
};

inline MatvecRun run_matvec(const RoughCase& c, const FastSettings& f) {
    MatvecRun r;
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    cfg.compression = "mlfmm";
    cfg.solver = SolverKind::Gmres;
    cfg.mlfmm.accuracy_digits = f.digits;
    cfg.mlfmm.pml_ring = f.ring;
    const Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam,
                         cfg);
    std::vector<VectorXc> products;
    {
        auto t0 = std::chrono::steady_clock::now();
        const mlfmm::MlfmmOperator Z(sim.problem(), cfg.mlfmm);
        r.fast_s = seconds(t0);
        if (Z.ring_operator() == nullptr)
            throw std::logic_error("pml_ring::run_matvec: no PML ring");
        const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
        r.unknowns = Z.rows();
        r.stretched = ring.stretched_count();
        r.levels = Z.octree().levels();
        r.leaf_lambda = Z.octree().box_size(Z.octree().leaf_level()) / pml_rough_ag::kLambda;
        r.fast_bytes = Z.memory_bytes();
        r.ring_near_bytes = ring.near_bytes();
        r.ring_far_bytes = ring.far_bytes();
        r.ring_near_s = ring.near_seconds();
        r.ring_far_s = ring.far_seconds();
        r.far_stats = ring.far_stats();
        t0 = std::chrono::steady_clock::now();
        for (const std::uint64_t seed : kSeeds)
            products.push_back(Z * random_vector(Z.cols(), seed));
        r.fast_matvec_s = seconds(t0) / static_cast<Real>(kSeeds.size());
        r.peak_fast = system_memory::peak_rss_bytes();
    }
    auto t0 = std::chrono::steady_clock::now();
    const std::shared_ptr<op::LinearOperator> D = op::DenseStrategy{}.build(sim.problem());
    r.dense_s = seconds(t0);
    if (D->rows() != r.unknowns)
        throw std::logic_error("pml_ring::run_matvec: the fast and dense systems differ in size");
    t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kSeeds.size(); ++i) {
        const VectorXc yd = *D * random_vector(D->cols(), kSeeds[i]);
        r.errors.push_back((products[i] - yd).norm() / yd.norm());
    }
    r.dense_matvec_s = seconds(t0) / static_cast<Real>(kSeeds.size());
    r.peak_dense = system_memory::peak_rss_bytes();
    return r;
}

[[nodiscard]] inline Real max_of(const std::vector<Real>& v) {
    Real m = 0;
    for (const Real e : v) m = std::max(m, e);
    return m;
}

/// One machine-readable line of a fast-vs-dense comparison.
[[nodiscard]] inline std::string row_line(const RoughCase& c, const FastSettings& f,
                                          const FastRun& fr, const DenseRun& dr) {
    const pml_rough_ag::Metrics m = pml_rough_ag::compare(fr.far, dr.far);
    char buf[2400];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "ROW | pml_ring | %s | %s | rms %.3g | L %.4g | taper %.4g | gap %.4g | pml %.4g | h %.4g "
        "| sigma %.3g | d0 %.1f | tol %.1e | 2N %lld | stretched %lld (%.1f %%) | levels %d | leaf "
        "%.3f lambda | matvec %.3e | R fast %.8f | R dense %.8f | dR %+.3e | l2 %.3e | l2_off30 "
        "%.3e | its %d | true res %.2e | dense res %.1e | fast op %.1f MB (rr near %.1f, rr far "
        "%.1f, ring near %.1f, ring far %.1f) | dense %.1f MB | ring pairs near %lld far %lld | "
        "fast assemble (op + rhs + diag) %.1f s (ring near %.1f, far %.1f) | matvec fast %.3f s "
        "dense %.3f s | "
        "GMRES %.1f s | dense assembly %.1f s | LU %.1f s | peak fast %.2f GB dense %.2f GB",
        c.tag.c_str(), c.material.c_str(), c.rms, c.patch, c.taper, c.gap, c.pml_width, c.mesh_size,
        c.sigma_max, f.digits, f.gmres_tolerance, static_cast<long long>(fr.unknowns),
        static_cast<long long>(fr.stretched),
        200.0 * static_cast<Real>(fr.stretched) / static_cast<Real>(fr.unknowns), fr.levels,
        fr.leaf_lambda, max_of(dr.matvec_errors), fr.far.reflectance(), dr.far.reflectance(), m.dR,
        m.l2, m.l2_off30, fr.iterations, fr.true_residual, dr.residual,
        static_cast<Real>(fr.operator_bytes) / 1e6, static_cast<Real>(fr.rr_near_bytes) / 1e6,
        static_cast<Real>(fr.rr_far_bytes) / 1e6, static_cast<Real>(fr.ring_near_bytes) / 1e6,
        static_cast<Real>(fr.ring_far_bytes) / 1e6, dr.dense_bytes / 1e6,
        static_cast<long long>(fr.ring_near_pairs), static_cast<long long>(fr.ring_far_pairs),
        fr.assemble_s, fr.ring_near_s, fr.ring_far_s, fr.matvec_s, dr.matvec_s, fr.solve_s,
        dr.assembly_s, dr.lu_s, fr.peak_rss / 1e9, dr.peak_rss / 1e9);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_ring::row_line: formatting failed or line truncated");
    return std::string(buf) + " | " + far_segment(fr.far_stats);
}

/// One machine-readable line of a matvec-only comparison (run_matvec).
[[nodiscard]] inline std::string matvec_line(const RoughCase& c, const FastSettings& f,
                                             const MatvecRun& r) {
    char buf[1200];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "MATVEC | pml_ring | %s | %s | rms %.3g | L %.4g | h %.4g | d0 %.1f | 2N %lld | stretched "
        "%lld | levels %d | leaf %.3f lambda | matvec %.3e (%.3e, %.3e, %.3e) | fast op %.1f MB "
        "(ring near %.1f, ring far %.1f) | fast op build %.1f s (ring near %.1f, far %.1f) | "
        "dense assembly %.1f s | matvec fast %.3f s dense %.3f s | peak fast %.2f GB dense %.2f GB",
        c.tag.c_str(), c.material.c_str(), c.rms, c.patch, c.mesh_size, f.digits,
        static_cast<long long>(r.unknowns), static_cast<long long>(r.stretched), r.levels,
        r.leaf_lambda, max_of(r.errors), r.errors.size() > 0 ? r.errors[0] : 0.0,
        r.errors.size() > 1 ? r.errors[1] : 0.0, r.errors.size() > 2 ? r.errors[2] : 0.0,
        static_cast<Real>(r.fast_bytes) / 1e6, static_cast<Real>(r.ring_near_bytes) / 1e6,
        static_cast<Real>(r.ring_far_bytes) / 1e6, r.fast_s, r.ring_near_s, r.ring_far_s, r.dense_s,
        r.fast_matvec_s, r.dense_matvec_s, r.peak_fast / 1e9, r.peak_dense / 1e9);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_ring::matvec_line: formatting failed or line truncated");
    return std::string(buf) + " | " + far_segment(r.far_stats);
}

/// ACA ring against the exact (WP-PML5a) ring of the same Problem and octree, without the dense
/// matrix (sizes beyond the dense budget; WP-PML5b scaling): the fast operator (MLFMM + ACA ring)
/// is built and applied first, then the exact ring alone (released before return).
struct RingCompare {
    Index unknowns = 0, stretched = 0;
    int levels = 0;
    Real leaf_lambda = 0;
    std::vector<Real> ring_errors;  ///< |ring_aca x - ring_exact x| / |ring_exact x|
    std::vector<Real> full_errors;  ///< |ring_aca x - ring_exact x| / |Z_exact x|
    std::size_t fast_bytes = 0, ring_near_bytes = 0, ring_far_bytes = 0, exact_far_bytes = 0;
    std::size_t exact_estimate = 0;  ///< PmlRingEstimate::far_bytes
    Real fast_s = 0, ring_near_s = 0, ring_far_s = 0, exact_far_s = 0;
    Real fast_matvec_s = 0, ring_matvec_s = 0, exact_ring_matvec_s = 0;
    Real peak_fast = 0, peak_exact = 0;
    bool exact_run = false;
    mlfmm::PmlRingFarStats far_stats;
};

inline RingCompare run_ring_compare(const RoughCase& c, const FastSettings& f, bool with_exact) {
    RingCompare r;
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    cfg.compression = "mlfmm";
    cfg.solver = SolverKind::Gmres;
    cfg.mlfmm.accuracy_digits = f.digits;
    cfg.mlfmm.pml_ring = f.ring;
    const Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam,
                         cfg);
    std::vector<VectorXc> xs, zf, ra;
    auto t0 = std::chrono::steady_clock::now();
    const mlfmm::MlfmmOperator Z(sim.problem(), cfg.mlfmm);
    r.fast_s = seconds(t0);
    if (Z.ring_operator() == nullptr)
        throw std::logic_error("pml_ring::run_ring_compare: no PML ring");
    const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
    r.unknowns = Z.rows();
    r.stretched = ring.stretched_count();
    r.levels = Z.octree().levels();
    r.leaf_lambda = Z.octree().box_size(Z.octree().leaf_level()) / pml_rough_ag::kLambda;
    r.fast_bytes = Z.memory_bytes();
    r.ring_near_bytes = ring.near_bytes();
    r.ring_far_bytes = ring.far_bytes();
    r.exact_estimate = ring.estimate().far_bytes;
    r.ring_near_s = ring.near_seconds();
    r.ring_far_s = ring.far_seconds();
    r.far_stats = ring.far_stats();
    for (const std::uint64_t seed : kSeeds) xs.push_back(random_vector(Z.cols(), seed));
    t0 = std::chrono::steady_clock::now();
    for (const VectorXc& x : xs) zf.push_back(Z * x);
    r.fast_matvec_s = seconds(t0) / static_cast<Real>(xs.size());
    t0 = std::chrono::steady_clock::now();
    for (const VectorXc& x : xs) ra.push_back(ring * x);
    r.ring_matvec_s = seconds(t0) / static_cast<Real>(xs.size());
    r.peak_fast = system_memory::peak_rss_bytes();
    if (!with_exact)
        return r;
    // The exact far ring alone (its near ring would be the same entries as the ACA ring's):
    // ring_exact x = near x + far_exact x.
    mlfmm::PmlRingParams ep = f.ring;
    ep.use_aca = false;
    ep.assemble_near = false;
    const mlfmm::PmlRingOperator exact(sim.problem(), Z.octree(), ep, f.digits);
    r.exact_far_s = exact.far_seconds();
    r.exact_far_bytes = exact.far_bytes();
    t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const VectorXc re = ring.near_operator() * xs[i] + exact * xs[i];
        const VectorXc ze = zf[i] - ra[i] + re;  // Z_rr x + exact ring x
        r.ring_errors.push_back((ra[i] - re).norm() / re.norm());
        r.full_errors.push_back((ra[i] - re).norm() / ze.norm());
    }
    r.exact_ring_matvec_s = seconds(t0) / static_cast<Real>(xs.size());
    r.peak_exact = system_memory::peak_rss_bytes();
    r.exact_run = true;
    return r;
}

[[nodiscard]] inline std::string ring_compare_line(const RoughCase& c, const FastSettings& f,
                                                   const RingCompare& r) {
    char buf[1400];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "RING | pml_ring | %s | %s | rms %.3g | L %.4g | h %.4g | pml %.4g | d0 %.1f | 2N %lld | "
        "stretched %lld | levels %d | leaf %.3f lambda | ring err %.3e | full err %.3e | fast op "
        "%.1f MB (ring near %.1f, ring far %.1f; exact far %.1f, estimate %.1f) | fast op build "
        "%.1f s (ring near %.1f, far %.1f; exact far %.1f) | matvec fast %.3f s ring %.3f s exact "
        "ring %.3f s | peak fast %.2f GB exact %.2f GB",
        c.tag.c_str(), c.material.c_str(), c.rms, c.patch, c.mesh_size, c.pml_width, f.digits,
        static_cast<long long>(r.unknowns), static_cast<long long>(r.stretched), r.levels,
        r.leaf_lambda, r.exact_run ? max_of(r.ring_errors) : -1.0,
        r.exact_run ? max_of(r.full_errors) : -1.0, static_cast<Real>(r.fast_bytes) / 1e6,
        static_cast<Real>(r.ring_near_bytes) / 1e6, static_cast<Real>(r.ring_far_bytes) / 1e6,
        static_cast<Real>(r.exact_far_bytes) / 1e6, static_cast<Real>(r.exact_estimate) / 1e6,
        r.fast_s, r.ring_near_s, r.ring_far_s, r.exact_far_s, r.fast_matvec_s, r.ring_matvec_s,
        r.exact_ring_matvec_s, r.peak_fast / 1e9, r.peak_exact / 1e9);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_ring::ring_compare_line: formatting failed or line truncated");
    return std::string(buf) + " | " + far_segment(r.far_stats);
}

/// WP-PML5b scaling cases: the rough Ag U case with the patch edge L (the map generated on L, the
/// same statistics sigma 50 nm, Lc 500 nm, seed 1; zones, PML, mesh and beam as U): L = 8.5 um
/// gives 2N = 59 600, L = 16 um 2N = 152 960 (benchmarks/results/pml_ring.md, 5b).
[[nodiscard]] inline RoughCase scaling_case(Real L) {
    RoughCase c = pml_rough_ag::u_case();
    c.patch = L;
    c.map_patch = L;
    char tag[32];
    std::snprintf(tag, sizeof(tag), "u_L%.4g", L * 1e6);
    c.tag = tag;
    return c;
}

/// The CI-size flat Ag case of the validation test (7a-type: flat Ag, normal incidence, w0 =
/// lambda, h = lambda / 4): patch 1 um, taper 0.25 um, gap 0.5 um (= lambda, the 7a window
/// width), W_pml = lambda with sigma_max 2 (the measured W_pml = lambda configuration of
/// WP-PML4b), 2N = 4592. The dark band (pml_start = 2.5 w0) is relaxed to 0.05: the comparison
/// is fast vs dense on the same discrete problem, not against Fresnel.
[[nodiscard]] inline RoughCase flat_ag_ci_case() {
    RoughCase c;
    c.tag = "flat_ci";
    c.material = "ag";
    c.rms = 0.0;
    c.correlation_length = 0.0;
    c.patch = 1.0e-6;
    c.map_patch = 1.0e-6;
    c.taper = 0.25e-6;
    c.gap = 0.5e-6;
    c.pml_width = 0.5e-6;
    c.sigma_max = 2.0;
    c.dark_band_tolerance = 5e-2;
    c.n_theta = 60;
    c.n_phi = 120;
    return c;
}

}  // namespace pml_ring
