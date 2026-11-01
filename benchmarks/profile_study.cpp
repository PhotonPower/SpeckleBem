// Phase 7 profile (WP-PF1): where the per-surface and the per-solve time of the production PML
// route goes. One case per process: the rough PML route of the preconditioner study
// (pml_rough_ag::u_case(): rough Ag, sigma 50 nm, Lc 500 nm, seed 1, h = lambda/4, W_pml = 2
// lambda, w0 = lambda; --patch L --map-patch L for the 16 um size, --material si for the Si
// control), compression "mlfmm", d0 = 3, the route's default preconditioner (ILUT of the near
// blocks A and D, fill 2 / drop 1e-2, WP-SC4), full GMRES, tolerance 1e-7 (pml_route::fast_config,
// as the WP-SC4 / WP-SC5 rows). Stages:
//   assembly  Simulation::assemble() wall time against its parts: MlfmmOperator::setup_stats
//             (octree, PML plan, far part split by MlfmmFarOperator::setup_stats, real-real near
//             part, ring split into near ring (cache / scatter passes) and ACA far ring), the
//             right-hand side and the ILUT setup (both from Simulation::report(); the ILUT split
//             into A and D, each extraction + ordering + elimination, A and D in parallel);
//   gmres     solver::gmres on the assembled operator and the Simulation's ILUT, both wrapped in
//             timing adapters (calls and seconds): matvec, ILUT apply, orthogonalisation
//             (GmresResult::orthogonalization_seconds) and the rest, per iteration and as shares
//             of the GMRES wall time; the matvecs inside GMRES split by MlfmmOperator /
//             MlfmmFarOperator / PmlRingOperator::apply_stats (near, far passes per region,
//             ring near / far);
//   matvec    --matvecs K standalone applies of a fixed random vector with the same split;
//   ilut      --ilut-applies K standalone ILUT applies, and the forward / backward solves of A
//             and D timed separately by a replica of the library's triangular solves on the
//             public factors (sequential; the library runs A and D in parallel), checked bitwise
//             against IncompleteLuPreconditioner::apply (a mismatch throws: no split is printed);
//   post      the windowed far field on the reflection hemisphere (Gauss-Legendre 90 x 180, the
//             records' R) and the two 1-degree cuts (as pml_rough_ag::evaluate_far_field), and the
//             scattered field (post::exterior_scattered_field, degree 6) on a --plane-n x
//             --plane-n plane at z = --plane-z over +-0.9 pml_start (the PML route admits no
//             points beyond the stretch start; the WP-S2 161 x 161 sampling, N stated);
//   memory    the peak working set after assembly, after GMRES and at the end.
// --no-solve (the thread-scaling runs) skips GMRES and every far-field / plane evaluation: the
// assembly, the standalone matvecs and ILUT applies and the memory line remain (the ROW line
// prints "skipped" for its, true res, R and GMRES).
// Output: SETUP, PROFILE (one line per stage group) and ROW lines (stdout, also appended to
// --summary FILE), the Simulation report. omp_get_max_threads() is printed (set OMP_NUM_THREADS
// before launching for the thread-scaling runs). Record: benchmarks/results/profile.md
// (+ profile_rows.txt).
//
// Usage: specklebem_profile_study [--tag NAME] [--material ag|si] [--patch L] [--map-patch L]
//            [--d0 3] [--tol 1e-7] [--max-iter 5000] [--matvecs 3] [--ilut-applies 10]
//            [--plane-n 161] [--plane-z -1e-6] [--no-post] [--no-solve] [--summary FILE]
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/incomplete_lu_preconditioner.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

#include "box_validity_support.hpp"
#include "pml_rough_ag_support.hpp"
#include "pml_route_case.hpp"
#include "system_memory.hpp"

namespace {

using namespace specklebem;
using Clock = std::chrono::steady_clock;

struct Options {
    pml_rough_ag::RoughCase c = pml_rough_ag::u_case();
    Real digits = 3.0;
    Real tolerance = 1e-7;
    int max_iter = 5000;
    int matvecs = 3;
    int ilut_applies = 10;
    int plane_n = 161;
    Real plane_z = -1e-6;
    bool post = true;
    bool solve = true;
    std::string summary;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        const auto real = [&]() { return std::stod(value()); };
        if (a == "--tag")
            o.c.tag = value();
        else if (a == "--material")
            o.c.material = value();
        else if (a == "--patch")
            o.c.patch = real();
        else if (a == "--map-patch")
            o.c.map_patch = real();
        else if (a == "--d0")
            o.digits = real();
        else if (a == "--tol")
            o.tolerance = real();
        else if (a == "--max-iter")
            o.max_iter = std::stoi(value());
        else if (a == "--matvecs")
            o.matvecs = std::stoi(value());
        else if (a == "--ilut-applies")
            o.ilut_applies = std::stoi(value());
        else if (a == "--plane-n")
            o.plane_n = std::stoi(value());
        else if (a == "--plane-z")
            o.plane_z = real();
        else if (a == "--no-post")
            o.post = false;
        else if (a == "--no-solve")
            o.solve = false;
        else if (a == "--summary")
            o.summary = value();
        else
            throw std::invalid_argument("unknown option " + a);
    }
    if (o.matvecs < 1 || o.ilut_applies < 1 || o.plane_n < 2)
        throw std::invalid_argument("--matvecs, --ilut-applies must be >= 1, --plane-n >= 2");
    if (!(o.plane_z < 0.0))
        throw std::invalid_argument("--plane-z must be < 0 (R1, the vacuum side)");
    return o;
}

void emit(const std::string& path, const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    if (path.empty())
        return;
    std::ofstream f(core::path_from_utf8(path), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

Real seconds_since(Clock::time_point t0) {
    return std::chrono::duration<Real>(Clock::now() - t0).count();
}

int threads() {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

std::string strf(const char* f, auto... args) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf), f, args...);
    return buf;
}

Real pct(Real part, Real whole) {
    return whole > 0.0 ? 100.0 * part / whole : 0.0;
}

/// Counts and times the applications of an operator (GMRES's matvecs).
class TimedOperator final : public op::LinearOperator {
public:
    explicit TimedOperator(const op::LinearOperator& Z) : Z_(Z) {}
    [[nodiscard]] Index rows() const override { return Z_.rows(); }
    [[nodiscard]] Index cols() const override { return Z_.cols(); }
    void apply(const VectorXc& x, VectorXc& y) const override {
        const auto t0 = Clock::now();
        Z_.apply(x, y);
        seconds_ += seconds_since(t0);
        ++calls_;
    }
    [[nodiscard]] std::string describe() const override { return Z_.describe(); }
    [[nodiscard]] std::size_t memory_bytes() const override { return Z_.memory_bytes(); }
    [[nodiscard]] Index calls() const { return calls_; }
    [[nodiscard]] Real seconds() const { return seconds_; }

private:
    const op::LinearOperator& Z_;
    mutable Index calls_ = 0;
    mutable Real seconds_ = 0;
};

/// Counts and times the applications of a preconditioner.
class TimedPreconditioner final : public solver::Preconditioner {
public:
    explicit TimedPreconditioner(const solver::Preconditioner& M) : M_(M) {}
    void apply(const VectorXc& r, VectorXc& z) const override {
        const auto t0 = Clock::now();
        M_.apply(r, z);
        seconds_ += seconds_since(t0);
        ++calls_;
    }
    [[nodiscard]] std::string name() const override { return M_.name(); }
    [[nodiscard]] Index calls() const { return calls_; }
    [[nodiscard]] Real seconds() const { return seconds_; }

private:
    const solver::Preconditioner& M_;
    mutable Index calls_ = 0;
    mutable Real seconds_ = 0;
};

/// Times of a replica of the library's triangular solves of one ILUT block.
struct TriSolveTimes {
    Real permute = 0;   ///< gather by the permutation + scatter back
    Real forward = 0;   ///< L solve (unit diagonal)
    Real backward = 0;  ///< U solve (with the diagonal)
};

/// z = P^T U^{-1} L^{-1} P r with the loops of solve_block (src/solver/incomplete_lu_preconditioner
/// .cpp), each phase timed; the same operations in the same order, so z is bitwise the library's.
/// It mirrors solve_block in src/solver/incomplete_lu_preconditioner.cpp loop by loop: a change
/// there has to be repeated here (run() throws when the replica no longer matches the library).
void timed_solve_block(const solver::IncompleteLuPreconditioner::Factor& f, const Complex* r,
                       Complex* z, TriSolveTimes& t) {
    const auto sz = [](Index i) { return static_cast<std::size_t>(i); };
    const Index n = f.n;
    auto t0 = Clock::now();
    std::vector<Complex> y(sz(n));
    for (Index k = 0; k < n; ++k) y[sz(k)] = r[f.perm[sz(k)]];
    t.permute += seconds_since(t0);
    t0 = Clock::now();
    for (Index k = 0; k < n; ++k) {
        Complex s = y[sz(k)];
        for (Index e = f.ptr[sz(k)]; e < f.mid[sz(k)]; ++e) s -= f.val[sz(e)] * y[sz(f.col[sz(e)])];
        y[sz(k)] = s;
    }
    t.forward += seconds_since(t0);
    t0 = Clock::now();
    for (Index k = n - 1; k >= 0; --k) {
        Complex s = y[sz(k)];
        for (Index e = f.mid[sz(k)]; e < f.ptr[sz(k) + 1]; ++e)
            s -= f.val[sz(e)] * y[sz(f.col[sz(e)])];
        y[sz(k)] = s / f.diag[sz(k)];
    }
    t.backward += seconds_since(t0);
    t0 = Clock::now();
    for (Index k = 0; k < n; ++k) z[f.perm[sz(k)]] = y[sz(k)];
    t.permute += seconds_since(t0);
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

/// Seconds of "<key> <number> s" in Simulation::report()'s assembly line; -1 if absent.
Real report_seconds(const std::string& report, const std::string& key) {
    const std::size_t line = report.find("assembly:");
    if (line == std::string::npos)
        return -1.0;
    const std::size_t end = report.find('\n', line);
    const std::string text = report.substr(line, end - line);
    const std::size_t p = text.find(key + " ");
    if (p == std::string::npos)
        return -1.0;
    return std::stod(text.substr(p + key.size() + 1));
}

/// The matvec split (summed stats of `calls` applies) as a PROFILE segment, per call.
std::string matvec_segment(const mlfmm::MlfmmApplyStats& a, const mlfmm::FarApplyStats& f,
                           const mlfmm::PmlRingApplyStats& r, Real measured, Index calls) {
    const Real n = static_cast<Real>(calls);
    const Real per = measured / n;
    const auto sum2 = [](const std::array<Real, 2>& v) { return v[0] + v[1]; };
    const Real agg = sum2(f.aggregation), up = sum2(f.upward), tr = sum2(f.translation),
               down = sum2(f.downward), rec = sum2(f.reception), ex = sum2(f.exact);
    const Real parts = a.near + a.far + a.ring + a.other;
    std::string s = strf(
        "calls %lld | per call %.4f s | near %.4f s (%.1f %%) | far %.4f s (%.1f %%): aggregation "
        "%.4f (%.1f %%), upward %.4f (%.1f %%), translation %.4f (%.1f %%), downward %.4f "
        "(%.1f %%), reception %.4f (%.1f %%), exact %.4f (%.1f %%), far other %.4f",
        static_cast<long long>(calls), per, a.near / n, pct(a.near, measured), a.far / n,
        pct(a.far, measured), agg / n, pct(agg, measured), up / n, pct(up, measured), tr / n,
        pct(tr, measured), down / n, pct(down, measured), rec / n, pct(rec, measured), ex / n,
        pct(ex, measured), f.other / n);
    s += strf(
        " | far R1 agg %.4f up %.4f trans %.4f down %.4f recv %.4f exact %.4f | far R2 agg %.4f "
        "up %.4f trans %.4f down %.4f recv %.4f exact %.4f",
        f.aggregation[0] / n, f.upward[0] / n, f.translation[0] / n, f.downward[0] / n,
        f.reception[0] / n, f.exact[0] / n, f.aggregation[1] / n, f.upward[1] / n,
        f.translation[1] / n, f.downward[1] / n, f.reception[1] / n, f.exact[1] / n);
    s += strf(
        " | ring %.4f s (%.1f %%): near %.4f (%.1f %%), far ACA %.4f (%.1f %%) | other "
        "(additions + rest) %.4f s (%.1f %%) | parts %.4f s = %.1f %% of the measured matvec",
        a.ring / n, pct(a.ring, measured), r.near / n, pct(r.near, measured), r.far / n,
        pct(r.far, measured), a.other / n, pct(a.other, measured), parts / n, pct(parts, measured));
    return s;
}

void run(const Options& o) {
    const pml_rough_ag::RoughCase& c = o.c;
    const int nthreads = threads();
    std::printf("profile_study: %s %s, OpenMP max threads %d\n", c.tag.c_str(), c.material.c_str(),
                nthreads);
    auto t0 = Clock::now();
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    const SimulationConfig cfg = pml_route::fast_config(c, o.digits, o.tolerance, o.max_iter);
    Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam, cfg);
    const Real construct_s = seconds_since(t0);
    if (sim.preconditioner_kind() != PreconditionerKind::IncompleteLu)
        throw std::logic_error("profile_study: the route's default preconditioner is not the ILUT");

    // ---- assembly -------------------------------------------------------------------------
    t0 = Clock::now();
    sim.assemble();
    const Real assemble_s = seconds_since(t0);
    const Real peak_assembly = system_memory::peak_rss_bytes();
    const auto* Z = dynamic_cast<const mlfmm::MlfmmOperator*>(sim.system_operator().get());
    if (Z == nullptr || Z->ring_operator() == nullptr)
        throw std::logic_error("profile_study: not an MLFMM operator with a PML ring");
    const solver::IncompleteLuPreconditioner* P = sim.incomplete_lu_preconditioner();
    if (P == nullptr)
        throw std::logic_error("profile_study: no ILUT after assemble()");
    const mlfmm::PmlRingOperator& ring = *Z->ring_operator();
    const mlfmm::MlfmmFarOperator& far = Z->far_operator();
    const Index n2 = Z->rows();
    const std::string report = sim.report();
    const Real z_s = report_seconds(report, "Z");
    const Real rhs_s = report_seconds(report, "rhs");
    const Real ilut_s = report_seconds(report, "ILUT preconditioner");
    if (z_s < 0.0 || rhs_s < 0.0 || ilut_s < 0.0)
        throw std::logic_error("profile_study: assembly times not found in Simulation::report()");

    const std::string head = "PROFILE | profile | " + c.tag + " | " + c.material + " | 2N " +
                             std::to_string(n2) + " | threads " + std::to_string(nthreads) + " | ";
    const Real leaf = Z->octree().box_size(Z->octree().leaf_level()) / pml_rough_ag::kLambda;
    emit(o.summary,
         strf("SETUP | profile | %s | %s | rms %.3g | L %.4g | Lgen %.4g | h %.4g | d0 %.1f | tol "
              "%.1e | 2N %lld | stretched %lld | levels %d | leaf %.3f lambda | threads %d | ILUT "
              "fill %.3g drop %.1e | op %.1f MB (rr near %.1f, far %.1f, ring near %.1f, ring far "
              "%.1f) | P %.1f MB | construct %.1f s | assemble %.1f s",
              c.tag.c_str(), c.material.c_str(), c.rms, c.patch, c.map_patch, c.mesh_size, o.digits,
              cfg.gmres.tolerance, static_cast<long long>(n2),
              static_cast<long long>(ring.stretched_count()), Z->octree().levels(), leaf, nthreads,
              P->params().fill_factor, P->params().drop_tol,
              static_cast<Real>(Z->memory_bytes()) / 1e6,
              static_cast<Real>(Z->near_operator().memory_bytes()) / 1e6,
              static_cast<Real>(far.memory_bytes()) / 1e6,
              static_cast<Real>(ring.near_bytes()) / 1e6, static_cast<Real>(ring.far_bytes()) / 1e6,
              static_cast<Real>(P->memory_bytes()) / 1e6, construct_s, assemble_s));

    const mlfmm::MlfmmSetupStats& st = Z->setup_stats();
    const mlfmm::FarSetupStats& fs = far.setup_stats();
    const solver::IncompleteLuPreconditioner::Factor& fa = P->factors_a();
    const solver::IncompleteLuPreconditioner::Factor& fd = P->factors_d();
    const Real z_parts = st.tree + st.plan + st.far + st.near + st.ring;
    const Real sum = st.total + rhs_s + ilut_s;
    emit(o.summary,
         head + strf("assembly | total %.2f s | Z %.2f s (%.1f %%; report %.2f): tree %.2f, PML "
                     "plan %.2f (%.1f %%), far %.2f (%.1f %%), near rr %.2f (%.1f %%), ring %.2f "
                     "(%.1f %%), Z other %.2f | rhs %.2f s (%.1f %%) | ILUT %.2f s (%.1f %%): A "
                     "%.2f (extract %.2f, order %.2f, eliminate %.2f), D %.2f (extract %.2f, order "
                     "%.2f, eliminate %.2f), A || D | stages %.2f s = %.1f %% of total (rest %.2f "
                     "s)",
                     assemble_s, st.total, pct(st.total, assemble_s), z_s, st.tree, st.plan,
                     pct(st.plan, assemble_s), st.far, pct(st.far, assemble_s), st.near,
                     pct(st.near, assemble_s), st.ring, pct(st.ring, assemble_s),
                     st.total - z_parts, rhs_s, pct(rhs_s, assemble_s), ilut_s,
                     pct(ilut_s, assemble_s), P->setup_seconds_a(), fa.extract_s, fa.order_s,
                     fa.factor_s, P->setup_seconds_d(), fd.extract_s, fd.order_s, fd.factor_s, sum,
                     pct(sum, assemble_s), assemble_s - sum));
    const op::SparseStretchedStats& ns = ring.near_stats();
    const mlfmm::PmlRingFarStats& rf = ring.far_stats();
    emit(o.summary,
         head + strf("setup split | far R1: plan %.2f, exact %.2f, translators %.2f, patterns "
                     "%.2f | far R2: plan %.2f, exact %.2f, translators %.2f, patterns %.2f | far "
                     "other %.2f (total %.2f) | ring near %.2f (cache %.2f, scatter %.2f) | ring "
                     "far %.2f (ACA blocks %lld, exact %lld; pair evaluations %lld + memo %lld) | "
                     "ring other %.2f",
                     fs.plan[0], fs.exact[0], fs.translators[0], fs.patterns[0], fs.plan[1],
                     fs.exact[1], fs.translators[1], fs.patterns[1],
                     fs.total - (fs.plan[0] + fs.exact[0] + fs.translators[0] + fs.patterns[0] +
                                 fs.plan[1] + fs.exact[1] + fs.translators[1] + fs.patterns[1]),
                     fs.total, ring.near_seconds(), ns.cache_seconds, ns.scatter_seconds,
                     ring.far_seconds(), static_cast<long long>(rf.aca_blocks),
                     static_cast<long long>(rf.exact_blocks),
                     static_cast<long long>(rf.pair_evaluations),
                     static_cast<long long>(rf.pair_memo_hits),
                     st.ring - ring.near_seconds() - ring.far_seconds()));

    // ---- GMRES ----------------------------------------------------------------------------
    Z->reset_apply_stats();
    const TimedOperator Zt(*Z);
    const TimedPreconditioner Pt(*P);
    solver::GmresResult g;
    if (o.solve)
        g = solver::gmres(Zt, sim.rhs(), Pt, sim.config().gmres);
    const Real peak_gmres = system_memory::peak_rss_bytes();
    const mlfmm::MlfmmApplyStats ga = Z->apply_stats();
    const mlfmm::FarApplyStats gf = far.apply_stats();
    const mlfmm::PmlRingApplyStats gr = ring.apply_stats();
    const Real its = static_cast<Real>(std::max(g.iterations, 1));
    const Real mv = Zt.seconds(), pc = Pt.seconds(), orth = g.orthogonalization_seconds;
    const Real parts = mv + pc + orth;
    if (!o.solve) {
        emit(o.summary, head + "gmres | skipped (--no-solve)");
    } else {
        emit(o.summary,
             head +
                 strf("gmres | its %d | wall %.3f s | per it %.4f s | matvec %.3f s (%lld calls, "
                      "%.4f s per call, %.1f %%) | ILUT apply %.3f s (%lld calls, %.4f s per call, "
                      "%.1f %%) | orth %.3f s (%.5f s per it, %.1f %%) | rest %.3f s (%.1f %%) | "
                      "parts %.4f s per it = %.1f %% of the iteration time",
                      g.iterations, g.wall_seconds, g.wall_seconds / its, mv,
                      static_cast<long long>(Zt.calls()),
                      mv / static_cast<Real>(std::max<Index>(Zt.calls(), 1)),
                      pct(mv, g.wall_seconds), pc, static_cast<long long>(Pt.calls()),
                      pc / static_cast<Real>(std::max<Index>(Pt.calls(), 1)),
                      pct(pc, g.wall_seconds), orth, orth / its, pct(orth, g.wall_seconds),
                      g.wall_seconds - parts, pct(g.wall_seconds - parts, g.wall_seconds),
                      parts / its, pct(parts, g.wall_seconds)));
        emit(o.summary, head + "matvec in GMRES | " + matvec_segment(ga, gf, gr, mv, Zt.calls()));
    }

    // ---- standalone matvecs ---------------------------------------------------------------
    const VectorXc x = random_vector(n2, 20261031);
    VectorXc y;
    Z->reset_apply_stats();
    t0 = Clock::now();
    for (int k = 0; k < o.matvecs; ++k) Z->apply(x, y);
    const Real mv_s = seconds_since(t0);
    emit(o.summary, head + "matvec standalone | " +
                        matvec_segment(Z->apply_stats(), far.apply_stats(), ring.apply_stats(),
                                       mv_s, o.matvecs));

    // ---- ILUT apply -----------------------------------------------------------------------
    VectorXc z;
    t0 = Clock::now();
    for (int k = 0; k < o.ilut_applies; ++k) P->apply(x, z);
    const Real pa_s = seconds_since(t0) / static_cast<Real>(o.ilut_applies);
    TriSolveTimes ta, td;
    VectorXc zr(n2);
    for (int k = 0; k < o.ilut_applies; ++k) {
        timed_solve_block(fa, x.data(), zr.data(), ta);
        timed_solve_block(fd, x.data() + n2 / 2, zr.data() + n2 / 2, td);
    }
    if (!(zr.array() == z.array()).all()) {
        throw std::runtime_error(
            "profile_study: ILUT replica differs from IncompleteLuPreconditioner::apply (update "
            "timed_solve_block)");
    }
    const Real k_il = static_cast<Real>(o.ilut_applies);
    emit(o.summary,
         head + strf("ilut apply | in GMRES %.4f s per call | standalone %.4f s per call | replica "
                     "(sequential, per call): A forward %.4f backward %.4f permute %.4f (nnz L+U "
                     "%lld), D forward %.4f backward %.4f permute %.4f (nnz L+U %lld) | replica "
                     "bitwise yes",
                     pc / static_cast<Real>(std::max<Index>(Pt.calls(), 1)), pa_s,
                     ta.forward / k_il, ta.backward / k_il, ta.permute / k_il,
                     static_cast<long long>(P->nonzeros_a()), td.forward / k_il, td.backward / k_il,
                     td.permute / k_il, static_cast<long long>(P->nonzeros_d())));

    // ---- post-processing ------------------------------------------------------------------
    const post::SurfaceSolution s{&sim.problem(), g.x};
    Real R = 0, far_s = 0, cuts_s = 0, plane_s = 0;
    Index hemi_dirs = 0, cut_dirs = 0, plane_points = 0;
    Real plane_half = 0;
    if (o.solve) {
        // As pml_rough_ag::evaluate_far_field (the records' R), with the hemisphere and the cuts
        // timed separately.
        post::FieldOptions fo = sim.far_field_options(c.ff_degree);
        fo.window = *sim.far_field_window();
        const post::SphereGrid grid = post::hemisphere_grid(-1.0, c.n_theta, c.n_phi);
        box_validity::FieldMatrix F;
        t0 = Clock::now();
        post::far_field(s, grid.dirs, F, fo);
        far_s = seconds_since(t0);
        const VectorXr hemi = F.rowwise().squaredNorm();
        const Real eta1 = material::vacuum().wave_impedance(sim.problem().omega).real();
        R = grid.weights.dot(hemi) / (2.0 * eta1) / beam->power();
        hemi_dirs = grid.dirs.rows();
        if (o.post) {
            t0 = Clock::now();
            post::far_field(s, box_validity::cut_directions(false), F, fo);
            post::far_field(s, box_validity::cut_directions(true), F, fo);
            cuts_s = seconds_since(t0);
            cut_dirs = 2 * box_validity::kCut;
        }
    }
    if (o.solve && o.post) {
        plane_half = 0.9 * est.size.half[2];
        const Index m = o.plane_n;
        Vertices pts(m * m, 3);
        for (Index i = 0; i < m; ++i) {
            for (Index j = 0; j < m; ++j) {
                const Real u = -plane_half +
                               2.0 * plane_half * static_cast<Real>(i) / static_cast<Real>(m - 1);
                const Real v = -plane_half +
                               2.0 * plane_half * static_cast<Real>(j) / static_cast<Real>(m - 1);
                pts.row(i * m + j) = Vec3(u, v, o.plane_z).transpose();
            }
        }
        Eigen::Matrix<Complex, Eigen::Dynamic, 3> E, H;
        t0 = Clock::now();
        post::exterior_scattered_field(s, pts, E, H, post::FieldOptions{});
        plane_s = seconds_since(t0);
        plane_points = pts.rows();
    }
    emit(o.summary,
         head + strf("post |%s far field hemisphere %lld dirs (GL %d x %d, window) %.2f s (%.3f ms "
                     "per dir) | cuts %lld dirs %.2f s | plane %d x %d = %lld points at z %.3g m "
                     "over +-%.3g m (exterior_scattered_field, degree 6) %.2f s (%.3f ms per "
                     "point)",
                     o.solve ? "" : " skipped (--no-solve) |", static_cast<long long>(hemi_dirs),
                     c.n_theta, c.n_phi, far_s,
                     1e3 * far_s / static_cast<Real>(std::max<Index>(hemi_dirs, 1)),
                     static_cast<long long>(cut_dirs), cuts_s, o.plane_n, o.plane_n,
                     static_cast<long long>(plane_points), o.plane_z, plane_half, plane_s,
                     1e3 * plane_s / static_cast<Real>(std::max<Index>(plane_points, 1))));
    emit(o.summary,
         head + strf("memory | peak after assembly %.2f GB | after GMRES %.2f GB | final %.2f GB "
                     "| operator %.1f MB | ILUT %.1f MB | Krylov basis %.2f GB",
                     peak_assembly / 1e9, peak_gmres / 1e9, system_memory::peak_rss_bytes() / 1e9,
                     static_cast<Real>(Z->memory_bytes()) / 1e6,
                     static_cast<Real>(P->memory_bytes()) / 1e6,
                     16.0 * static_cast<Real>(g.iterations + 1) * static_cast<Real>(n2) / 1e9));
    // --no-solve: no iteration count, residual, R or GMRES time exists, so none is printed as 0.
    std::string solve_part = strf(
        "its skipped | converged skipped | true res skipped | R skipped "
        "| assemble %.1f s | GMRES skipped",
        assemble_s);
    if (o.solve) {
        solve_part = strf(
            "its %d | converged %s | true res %.2e | R %.8f | assemble %.1f s | "
            "GMRES %.1f s",
            g.iterations, g.converged ? "yes" : "NO", g.true_relative_residual, R, assemble_s,
            g.wall_seconds);
    }
    emit(o.summary,
         strf("ROW | profile | %s | %s | 2N %lld | threads %d | %s | matvec %.4f s | peak %.2f GB",
              c.tag.c_str(), c.material.c_str(), static_cast<long long>(n2), nthreads,
              solve_part.c_str(), mv_s / static_cast<Real>(o.matvecs),
              system_memory::peak_rss_bytes() / 1e9));
    std::printf("%s\n", report.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "profile_study: %s\n", e.what());
        return 1;
    }
}
