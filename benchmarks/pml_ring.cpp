// Fast path of the PML route vs dense (WP-PML5a, validation 7h; ADR 0008 amendment 2026-10-18,
// docs/05 "PML route" row 7h): one case of tests/support/pml_ring_support.hpp per process. The
// fast run (Simulation with compression "mlfmm" on the PML route: MLFMM on the real functions plus
// the PML ring, GMRES) comes first, then the dense run (DenseStrategy, LU in place) with the
// matvec comparison of the stored fast products, R and the windowed far field (L2 metric of
// pml_rough_ag::compare). --fast-only skips the dense run (sizes beyond the dense budget).
// --compare-far TEST REF [--summary FILE] (first option) compares two far fields stored by earlier
// runs (FILE.TAG.far / FILE.TAG.dense.far, written with --summary) in the 7c/7d metric.
// --estimate-only builds only the mesh and the octree and prints the PML ring's pre-assembly sizes
// (mlfmm::estimate_pml_ring) and the dense size: the memory model of WP-PML5a and its projection.
// WP-PML5b: the far ring is ACA by default (--exact-far: the exact ring of 5a; --eta, --aca-eps,
// --max-rank, --leaf-columns: PmlRingParams). --ring-compare builds the fast operator (MLFMM + ACA
// ring) and then the exact ring of the same Problem and octree and prints the RING line (ring and
// full-operator matvec differences, memory, times, ranks; no dense matrix, no solve);
// --ring-compare --no-exact builds the ACA operator only (sizes beyond the exact ring's memory).
// --assembly-study (WP-PML5c) times the near-ring assembly on the octree's ring pattern without
// solving, in one process and alternating: the canonical-pair cache (default, "5c"), the
// per-ordering evaluation of WP-PML5a ("5a"), and the dense assembler's evaluation of exactly
// these pairs ("dense": each needed unordered stretched pair once by the balanced parity rule of
// DenseStrategy, every unstretched ordering once, colour by colour with the dense schedule;
// blocks evaluated, not scattered), plus op::assemble_diagonal against the diagonal read from
// the near parts (equality reported). ASSEMBLY lines; --reps R repeats the 5c / 5a / dense triple.
//
// Output (stdout): SETUP and ESTIMATE lines, the ROW line (pml_ring::row_line; also appended to
// --summary FILE), the fast Simulation report. Record: benchmarks/results/pml_ring.md.
//
// Usage: specklebem_pml_ring [--case u|flat_u|flat_ci] [--tag NAME] [--material ag|si]
//            [--patch L] [--map-patch L] [--taper W] [--gap W] [--pml W] [--h H] [--sigma S]
//            [--rms SIGMA (0: flat)] [--lc LC] [--w0 W0] [--dark-band-tol T] [--d0 3]
//            [--tol 1e-7] [--n-theta 90] [--n-phi 180] [--summary FILE] [--fast-only]
//            [--estimate-only] [--force] [--exact-far] [--eta 2] [--aca-eps 0] [--max-rank 0]
//            [--leaf-columns] [--ring-compare] [--no-exact] [--seed 1] [--assembly-study]
//            [--reps 1] [--dense-compare] [--quick]
// --case must come first (it resets the case).
#include "specklebem/compression/mlfmm/pml_ring.hpp"

#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

#include "pml_ring_support.hpp"

namespace {

using namespace pml_ring;

struct Options {
    RoughCase c = pml_rough_ag::u_case();
    FastSettings f;
    std::string summary;
    bool fast_only = false;
    bool estimate_only = false;
    bool force = false;
    bool ring_compare = false;
    bool no_exact = false;
    bool assembly_study = false;
    bool dense_compare = false;
    bool quick = false;  ///< --assembly-study: 5a and 5c only (no canonical-only, dense, profile)
    int reps = 1;
};

Options parse(int argc, char** argv) {
    Options o;
    o.c.tag = "u";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        const auto real = [&]() { return std::stod(value()); };
        if (a == "--case" && i != 1)
            throw std::invalid_argument("--case must be the first option (it resets the case)");
        if (a == "--case") {
            const std::string name = value();
            if (name == "u") {
                o.c = pml_rough_ag::u_case();
            } else if (name == "flat_u") {  // flat Ag with the zones of U
                o.c = pml_rough_ag::u_case();
                o.c.tag = "flat_u";
                o.c.rms = 0.0;
                o.c.correlation_length = 0.0;
                o.c.map_patch = o.c.patch;
            } else if (name == "flat_ci") {
                o.c = flat_ag_ci_case();
            } else {
                throw std::invalid_argument("unknown --case " + name);
            }
        } else if (a == "--tag") {
            o.c.tag = value();
        } else if (a == "--material") {
            o.c.material = value();
        } else if (a == "--patch") {
            o.c.patch = real();
        } else if (a == "--map-patch") {
            o.c.map_patch = real();
        } else if (a == "--taper") {
            o.c.taper = real();
        } else if (a == "--gap") {
            o.c.gap = real();
        } else if (a == "--pml") {
            o.c.pml_width = real();
        } else if (a == "--h") {
            o.c.mesh_size = real();
        } else if (a == "--sigma") {
            o.c.sigma_max = real();
        } else if (a == "--rms") {
            o.c.rms = real();
        } else if (a == "--lc") {
            o.c.correlation_length = real();
        } else if (a == "--w0") {
            o.c.waist = real();
        } else if (a == "--dark-band-tol") {
            o.c.dark_band_tolerance = real();
        } else if (a == "--d0") {
            o.f.digits = real();
        } else if (a == "--tol") {
            o.f.gmres_tolerance = real();
        } else if (a == "--n-theta") {
            o.c.n_theta = std::stoi(value());
        } else if (a == "--n-phi") {
            o.c.n_phi = std::stoi(value());
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--fast-only") {
            o.fast_only = true;
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--force") {
            o.force = true;
        } else if (a == "--exact-far") {
            o.f.ring.use_aca = false;
        } else if (a == "--eta") {
            o.f.ring.eta = real();
        } else if (a == "--aca-eps") {
            o.f.ring.aca_epsilon = real();
        } else if (a == "--max-rank") {
            o.f.ring.max_rank = std::stoll(value());
        } else if (a == "--leaf-columns") {
            o.f.ring.coarse_columns = false;
        } else if (a == "--ring-compare") {
            o.ring_compare = true;
        } else if (a == "--no-exact") {
            o.no_exact = true;
        } else if (a == "--seed") {
            o.c.seed = static_cast<std::uint64_t>(std::stoull(value()));
        } else if (a == "--assembly-study") {
            o.assembly_study = true;
        } else if (a == "--dense-compare") {
            o.dense_compare = true;
        } else if (a == "--quick") {
            o.quick = true;
        } else if (a == "--reps") {
            o.reps = std::stoi(value());
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    return o;
}

void append_summary(const std::string& path, const std::string& line) {
    if (path.empty())
        return;
    std::ofstream f(core::path_from_utf8(path), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

/// Octree and ring estimate of the fast path (the MlfmmOperator's tree: leaf rule of d0).
std::string estimate_line(const RoughCase& c, const FastSettings& f) {
    const geometry::OpenSurfaceMesh om = pml_rough_ag::make_open_mesh(c);
    const basis::RwgSpace space(om.mesh);
    const geometry::CoordinateStretch st = geometry::make_pml_stretch(om, c.sigma_max, c.order);
    mlfmm::MlfmmParams p;
    p.accuracy_digits = f.digits;
    const mlfmm::Octree tree = mlfmm::make_octree(space, pml_rough_ag::kLambda, p);
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(space, st);
    const mlfmm::PmlRingEstimate e = mlfmm::estimate_pml_ring(tree, space, s, f.ring);
    char buf[1000];
    std::snprintf(
        buf, sizeof(buf),
        "ESTIMATE | %s | L %.4g | pml %.4g | h %.4g | d0 %.1f | 2N %lld | stretched %lld (%.1f %%) "
        "| levels %d | leaf %.3f lambda | row leaves %lld | ring near %lld pairs %.1f MB | ring "
        "far "
        "%lld pairs (%lld stored) %.1f MB | rr near %lld pairs %.1f MB | dense %.1f MB | ring / "
        "dense %.3f",
        c.tag.c_str(), c.patch, c.pml_width, c.mesh_size, f.digits, static_cast<long long>(2 * e.n),
        static_cast<long long>(e.stretched),
        100.0 * static_cast<Real>(e.stretched) / static_cast<Real>(e.n), tree.levels(),
        tree.box_size(tree.leaf_level()) / pml_rough_ag::kLambda,
        static_cast<long long>(e.row_leaves), static_cast<long long>(e.near_pairs),
        static_cast<Real>(e.near_bytes) / 1e6, static_cast<long long>(e.far_pairs),
        static_cast<long long>(e.far_stored), static_cast<Real>(e.far_bytes) / 1e6,
        static_cast<long long>(e.rr_near_pairs), static_cast<Real>(4 * e.rr_near_pairs * 24) / 1e6,
        static_cast<Real>(e.dense_bytes) / 1e6,
        static_cast<Real>(e.near_bytes + e.far_bytes) / static_cast<Real>(e.dense_bytes));
    char aca[300];
    std::snprintf(aca, sizeof(aca),
                  " | ACA %s eta %.2f: admissible %lld blocks (%lld pairs), exact %lld blocks "
                  "%.1f MB",
                  f.ring.use_aca ? "on" : "off", f.ring.eta,
                  static_cast<long long>(e.far_aca_blocks), static_cast<long long>(e.far_aca_pairs),
                  static_cast<long long>(e.far_exact_blocks),
                  static_cast<Real>(e.far_exact_bytes) / 1e6);
    return std::string(buf) + aca;
}

/// The dense assembler's evaluation of exactly the pairs of `pattern` (WP-PML5c gate reference):
/// greedy triangle colouring, colour by colour, OpenMP schedule(dynamic, 1) over the test
/// triangles of a colour in order of decreasing longest edge (DenseStrategy::build); per test
/// triangle t the pattern's source triangles s, evaluated iff the pair is unstretched or
/// identical, or stretched with the balanced parity rule ((t + s) even) == (t < s) (each unordered
/// stretched pair once, as DenseStrategy's pass 1). Blocks are evaluated, not scattered. Returns
/// the wall time [s]; `evaluated` = the number of evaluate() calls.
Real dense_same_pairs_seconds(const op::Problem& p, const op::BasisPattern& pattern,
                              Index& evaluated) {
    const basis::RwgSpace& space = *p.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index F = mesh.num_triangles();
    const op::PairBlockEvaluator eval(p);
    std::vector<int> colour(static_cast<std::size_t>(F), -1);
    std::vector<std::vector<Index>> groups;
    std::vector<Real> edge(static_cast<std::size_t>(F));
    for (Index t = 0; t < F; ++t) {
        const basis::RwgSpace::Support sup = space.support(t);
        std::array<bool, 4> used{};
        for (int a = 0; a < sup.count; ++a) {
            const Index nb = sup.n[a];
            const Index other =
                space.plus_triangle(nb) == t ? space.minus_triangle(nb) : space.plus_triangle(nb);
            const int c = colour[static_cast<std::size_t>(other)];
            if (c >= 0)
                used[static_cast<std::size_t>(c)] = true;
        }
        int c = 0;
        while (used[static_cast<std::size_t>(c)]) ++c;
        colour[static_cast<std::size_t>(t)] = c;
        if (static_cast<std::size_t>(c) >= groups.size())
            groups.resize(static_cast<std::size_t>(c) + 1);
        groups[static_cast<std::size_t>(c)].push_back(t);
        const auto v = [&](Eigen::Index i) -> Vec3 {
            return mesh.vertices().row(mesh.triangles()(t, i)).transpose();
        };
        edge[static_cast<std::size_t>(t)] =
            std::max({(v(1) - v(0)).squaredNorm(), (v(2) - v(1)).squaredNorm(),
                      (v(0) - v(2)).squaredNorm()});
    }
    for (std::vector<Index>& g : groups)
        std::stable_sort(g.begin(), g.end(), [&](Index a, Index b) {
            return edge[static_cast<std::size_t>(a)] > edge[static_cast<std::size_t>(b)];
        });
    std::vector<char> str(static_cast<std::size_t>(F), 0);
    for (Index t = 0; t < F; ++t)
        str[static_cast<std::size_t>(t)] =
            kernels::is_stretched_triangle(*p.stretch, mesh, t) ? 1 : 0;
    std::vector<Index> count(static_cast<std::size_t>(F), 0);
    const auto t0 = std::chrono::steady_clock::now();
    for (const std::vector<Index>& g : groups) {
        const auto ng = static_cast<Index>(g.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (Index gi = 0; gi < ng; ++gi) {
            const Index t = g[static_cast<std::size_t>(gi)];
            const basis::RwgSpace::Support st = space.support(t);
            if (st.count == 0)
                continue;
            std::vector<Index> sources;
            for (int a = 0; a < st.count; ++a) {
                const auto grp = static_cast<std::size_t>(
                    pattern.group_of_row[static_cast<std::size_t>(st.n[a])]);
                for (Index k = pattern.col_ptr[grp]; k < pattern.col_ptr[grp + 1]; ++k) {
                    const Index c = pattern.cols[static_cast<std::size_t>(k)];
                    sources.push_back(space.plus_triangle(c));
                    sources.push_back(space.minus_triangle(c));
                }
            }
            std::sort(sources.begin(), sources.end());
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
            op::TrianglePairBlocks blk;
            for (const Index s : sources) {
                const bool stretched =
                    str[static_cast<std::size_t>(t)] != 0 || str[static_cast<std::size_t>(s)] != 0;
                if (stretched && s != t && ((t + s) % 2 == 0) != (t < s))
                    continue;  // the other ordering evaluates this pair
                eval.evaluate(t, s, blk);
                ++count[static_cast<std::size_t>(t)];
            }
        }
    }
    const Real sec = seconds(t0);
    evaluated = 0;
    for (const Index c : count) evaluated += c;
    return sec;
}

/// Where the canonical-pair pass spends its time (WP-PML5c record): every unordered stretched
/// pair {a, b} of `pattern` (a <= b, as the cache) evaluated once with per-pair timing, summed
/// per class of the object region's rule (kernels::stretched_pair_info at k_2): identical, shared
/// edge, shared vertex, near / far with the target reached, near / far at the degree cap. One
/// PROFILE line per class: pairs, thread-seconds, share, point pairs per call (R2 rule).
void profile_canonical_pairs(const op::Problem& p, const op::BasisPattern& pattern,
                             const std::string& tag, Real digits, const std::string& sum) {
    const basis::RwgSpace& space = *p.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index F = mesh.num_triangles();
    std::vector<char> str(static_cast<std::size_t>(F), 0);
    for (Index t = 0; t < F; ++t)
        str[static_cast<std::size_t>(t)] =
            kernels::is_stretched_triangle(*p.stretch, mesh, t) ? 1 : 0;
    std::vector<std::array<Index, 2>> pairs;
    {
        std::vector<std::vector<Index>> partners(static_cast<std::size_t>(F));
        std::vector<Index> sources;
        for (Index t = 0; t < F; ++t) {
            const basis::RwgSpace::Support st = space.support(t);
            sources.clear();
            for (int a = 0; a < st.count; ++a) {
                const auto grp = static_cast<std::size_t>(
                    pattern.group_of_row[static_cast<std::size_t>(st.n[a])]);
                for (Index k = pattern.col_ptr[grp]; k < pattern.col_ptr[grp + 1]; ++k) {
                    const Index c = pattern.cols[static_cast<std::size_t>(k)];
                    sources.push_back(space.plus_triangle(c));
                    sources.push_back(space.minus_triangle(c));
                }
            }
            for (const Index s : sources)
                if (str[static_cast<std::size_t>(t)] != 0 || str[static_cast<std::size_t>(s)] != 0)
                    partners[static_cast<std::size_t>(std::min(t, s))].push_back(std::max(t, s));
        }
        for (Index a = 0; a < F; ++a) {
            std::vector<Index>& v = partners[static_cast<std::size_t>(a)];
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            for (const Index b : v) pairs.push_back({a, b});
        }
    }
    const op::PairBlockEvaluator eval(p);
    const Complex k2 = p.object.wavenumber(p.omega);
    constexpr int kClasses = 5;
    const char* names[kClasses] = {"identical", "shared edge", "shared vertex", "near/far reached",
                                   "near/far capped"};
    const auto n = static_cast<Index>(pairs.size());
    std::vector<int> cls(pairs.size());
    std::vector<Real> sec(pairs.size()), pts(pairs.size());
    const auto t0 = std::chrono::steady_clock::now();
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (Index i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const kernels::StretchedPairInfo info = kernels::stretched_pair_info(
            mesh, pairs[k][0], pairs[k][1], k2, p.kernel_options, *p.stretch);
        int c = 4;
        switch (info.proximity) {
            case kernels::Proximity::identical:
                c = 0;
                break;
            case kernels::Proximity::shared_edge:
                c = 1;
                break;
            case kernels::Proximity::shared_vertex:
                c = 2;
                break;
            default:
                c = info.target_reached ? 3 : 4;
                break;
        }
        cls[k] = c;
        pts[k] = static_cast<Real>(info.points);
        op::TrianglePairBlocks blk;
        const auto a = std::chrono::steady_clock::now();
        eval.evaluate(pairs[k][0], pairs[k][1], blk);
        sec[k] = seconds(a);
    }
    const Real wall = seconds(t0);
    std::array<Real, kClasses> s{}, q{};
    std::array<Index, kClasses> cnt{};
    Real total = 0;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        s[static_cast<std::size_t>(cls[k])] += sec[k];
        q[static_cast<std::size_t>(cls[k])] += pts[k];
        cnt[static_cast<std::size_t>(cls[k])] += 1;
        total += sec[k];
    }
    char buf[600];
    for (std::size_t c = 0; c < static_cast<std::size_t>(kClasses); ++c) {
        std::snprintf(buf, sizeof(buf),
                      "PROFILE | %s | d0 %.1f | %s | pairs %lld | %.1f thread-s (%.1f %%) | %.0f "
                      "point pairs per call (R2 rule) | %.2f ms per pair",
                      tag.c_str(), digits, names[c], static_cast<long long>(cnt[c]), s[c],
                      total > 0 ? 100.0 * s[c] / total : 0.0,
                      cnt[c] > 0 ? q[c] / static_cast<Real>(cnt[c]) : 0.0,
                      cnt[c] > 0 ? 1e3 * s[c] / static_cast<Real>(cnt[c]) : 0.0);
        std::printf("%s\n", buf);
        append_summary(sum, buf);
    }
    std::snprintf(buf, sizeof(buf),
                  "PROFILE | %s | d0 %.1f | total | pairs %lld | %.1f thread-s | wall %.1f s",
                  tag.c_str(), digits, static_cast<long long>(n), total, wall);
    std::printf("%s\n", buf);
    std::fflush(stdout);
    append_summary(sum, buf);
}

/// Translation classes with a rounding-tolerant key (WP-PML5c design study; the library's
/// translation_classes key is exact, bit for bit, and finds fewer copies): a pair {a, b} (a < b,
/// sharing a vertex) whose six vertices all lie in |y| <= y.start (x strip: sigma_y = 0, the
/// stretch depends on x only) is invariant under translations along y, one with all |x| <= x.start
/// under translations along x; corner pairs are not. Key: the invariant axis, the six vertices in
/// local order (both triangles) quantised to 1e-7 h with the translated coordinate taken relative
/// to a's vertex 0, and per local vertex of each triangle the sign of the basis function whose free
/// vertex it is (0: none). Prints the class count, the fraction of the touching thread time a
/// per-class evaluation would keep (times measured per pair) and the block difference of class
/// members against their representative (slots mapped by free vertex), on up to `samples` classes:
/// CLASSES line.
void translation_classes(const op::Problem& p, const op::BasisPattern& pattern, Real h,
                         const std::string& tag, Real digits, int samples, const std::string& sum) {
    const basis::RwgSpace& space = *p.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Triangles& T = mesh.triangles();
    const Vertices& V = mesh.vertices();
    const Index F = mesh.num_triangles();
    const geometry::CoordinateStretch& S = *p.stretch;
    const Real xs = S.profile_x() ? S.profile_x()->start : 0.0;
    const Real ys = S.profile_y() ? S.profile_y()->start : 0.0;
    std::vector<char> str(static_cast<std::size_t>(F), 0);
    for (Index t = 0; t < F; ++t)
        str[static_cast<std::size_t>(t)] = kernels::is_stretched_triangle(S, mesh, t) ? 1 : 0;
    const auto touching = [&](Index a, Index b) {
        for (Eigen::Index i = 0; i < 3; ++i)
            for (Eigen::Index j = 0; j < 3; ++j)
                if (T(a, i) == T(b, j))
                    return true;
        return false;
    };
    std::vector<std::array<Index, 2>> pairs;
    {
        std::vector<std::vector<Index>> partners(static_cast<std::size_t>(F));
        std::vector<Index> sources;
        for (Index t = 0; t < F; ++t) {
            const basis::RwgSpace::Support st = space.support(t);
            sources.clear();
            for (int a = 0; a < st.count; ++a) {
                const auto grp = static_cast<std::size_t>(
                    pattern.group_of_row[static_cast<std::size_t>(st.n[a])]);
                for (Index k = pattern.col_ptr[grp]; k < pattern.col_ptr[grp + 1]; ++k) {
                    const Index c = pattern.cols[static_cast<std::size_t>(k)];
                    sources.push_back(space.plus_triangle(c));
                    sources.push_back(space.minus_triangle(c));
                }
            }
            for (const Index s : sources)
                if (s != t &&
                    (str[static_cast<std::size_t>(t)] != 0 ||
                     str[static_cast<std::size_t>(s)] != 0) &&
                    touching(t, s))
                    partners[static_cast<std::size_t>(std::min(t, s))].push_back(std::max(t, s));
        }
        for (Index a = 0; a < F; ++a) {
            std::vector<Index>& v = partners[static_cast<std::size_t>(a)];
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            for (const Index b : v) pairs.push_back({a, b});
        }
    }
    // Local free-vertex index and sign per slot of triangle t.
    const auto free_local = [&](Index t, int slot) {
        const basis::RwgSpace::Support st = space.support(t);
        const Index n = st.n[slot];
        const Index fv =
            space.plus_triangle(n) == t ? space.plus_free_vertex(n) : space.minus_free_vertex(n);
        for (Eigen::Index i = 0; i < 3; ++i)
            if (T(t, i) == fv)
                return static_cast<int>(i);
        return -1;
    };
    const Real q = 1e-7 * h;
    std::map<std::vector<long long>, std::vector<std::size_t>> classes;
    Index corner = 0;
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        const Index a = pairs[k][0], b = pairs[k][1];
        bool xstrip = true, ystrip = true;
        for (const Index t : {a, b})
            for (Eigen::Index i = 0; i < 3; ++i) {
                const Vec3 v = V.row(T(t, i)).transpose();
                xstrip = xstrip && std::abs(v(1)) <= ys;
                ystrip = ystrip && std::abs(v(0)) <= xs;
            }
        if (!xstrip && !ystrip) {
            ++corner;
            continue;
        }
        const int axis = xstrip ? 1 : 0;  // translated coordinate
        const Real ref = V(T(a, 0), axis);
        std::vector<long long> key = {axis};
        for (const Index t : {a, b}) {
            for (Eigen::Index i = 0; i < 3; ++i)
                for (Eigen::Index d = 0; d < 3; ++d) {
                    const Real c = V(T(t, i), d) - (d == axis ? ref : 0.0);
                    key.push_back(std::llround(c / q));
                }
            std::array<long long, 3> sg{};
            const basis::RwgSpace::Support st = space.support(t);
            for (int s = 0; s < st.count; ++s)
                sg[static_cast<std::size_t>(free_local(t, s))] = st.sign[s];
            key.insert(key.end(), sg.begin(), sg.end());
        }
        classes[key].push_back(k);
    }
    // Per-pair times of the touching pairs (one evaluation each) and the member differences.
    const op::PairBlockEvaluator eval(p);
    const auto n = static_cast<Index>(pairs.size());
    std::vector<Real> sec(pairs.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (Index i = 0; i < n; ++i) {
        op::TrianglePairBlocks blk;
        const auto t0 = std::chrono::steady_clock::now();
        eval.evaluate(pairs[static_cast<std::size_t>(i)][0], pairs[static_cast<std::size_t>(i)][1],
                      blk);
        sec[static_cast<std::size_t>(i)] = seconds(t0);
    }
    Real total = 0, kept = 0;
    for (std::size_t k = 0; k < pairs.size(); ++k) total += sec[k];
    std::vector<const std::vector<std::size_t>*> multi;
    std::vector<char> in_class(pairs.size(), 0);
    for (const auto& [key, members] : classes) {
        kept += sec[members.front()];
        for (const std::size_t k : members) in_class[k] = 1;
        if (members.size() > 1)
            multi.push_back(&members);
    }
    for (std::size_t k = 0; k < pairs.size(); ++k)
        if (!in_class[k])
            kept += sec[k];  // corner pairs: evaluated as now
    // Member vs representative blocks on a stride of the multi-member classes.
    const std::size_t stride =
        std::max<std::size_t>(1, multi.size() / static_cast<std::size_t>(std::max(samples, 1)));
    Real worst = 0, worst_l = 0;
    Index compared = 0;
    for (std::size_t c = 0; c < multi.size(); c += stride) {
        const std::vector<std::size_t>& m = *multi[c];
        const auto& r = pairs[m.front()];
        op::TrianglePairBlocks br;
        eval.evaluate(r[0], r[1], br);
        for (std::size_t j = 1; j < std::min<std::size_t>(m.size(), 3); ++j) {
            const auto& o = pairs[m[j]];
            op::TrianglePairBlocks bo;
            eval.evaluate(o[0], o[1], bo);
            const basis::RwgSpace::Support so = space.support(o[0]), sb = space.support(o[1]);
            const basis::RwgSpace::Support sr = space.support(r[0]), srb = space.support(r[1]);
            Real diff = 0, scale = 0, diff_l = 0, scale_l = 0;
            for (int a = 0; a < so.count; ++a) {
                int ar = -1;
                for (int x = 0; x < sr.count; ++x)
                    if (free_local(r[0], x) == free_local(o[0], a))
                        ar = x;
                for (int b = 0; b < sb.count; ++b) {
                    int brr = -1;
                    for (int x = 0; x < srb.count; ++x)
                        if (free_local(r[1], x) == free_local(o[1], b))
                            brr = x;
                    const auto ko = static_cast<std::size_t>(3 * a + b);
                    const auto kr = static_cast<std::size_t>(3 * ar + brr);
                    for (const auto X :
                         {&op::TrianglePairBlocks::jj, &op::TrianglePairBlocks::jm,
                          &op::TrianglePairBlocks::mj, &op::TrianglePairBlocks::mm}) {
                        const Real d = std::abs((bo.*X)[ko] - (br.*X)[kr]);
                        const Real s = std::abs((bo.*X)[ko]);
                        diff = std::max(diff, d);
                        scale = std::max(scale, s);
                        if (X == &op::TrianglePairBlocks::jj || X == &op::TrianglePairBlocks::mm) {
                            diff_l = std::max(diff_l, d);
                            scale_l = std::max(scale_l, s);
                        }
                    }
                }
            }
            worst = std::max(worst, scale > 0 ? diff / scale : 0.0);
            worst_l = std::max(worst_l, scale_l > 0 ? diff_l / scale_l : 0.0);
            ++compared;
        }
    }
    char buf[700];
    std::snprintf(buf, sizeof(buf),
                  "CLASSES (tolerant key, not the library's exact one) | %s | d0 %.1f | stretched "
                  "touching pairs %lld (corner %lld) | "
                  "translation classes %lld | thread time %.1f s, per-class evaluation keeps %.1f "
                  "s (%.1f %%) | member vs representative: %lld compared, max |dB| / max |B| %.2e "
                  "(JJ/MM %.2e)",
                  tag.c_str(), digits, static_cast<long long>(pairs.size()),
                  static_cast<long long>(corner), static_cast<long long>(classes.size()), total,
                  kept, total > 0 ? 100.0 * kept / total : 0.0, static_cast<long long>(compared),
                  worst, worst_l);
    std::printf("%s\n", buf);
    std::fflush(stdout);
    append_summary(sum, buf);
}

/// True if two sparse matrices have bitwise equal values (same structure assumed by nnz).
bool same_values(const op::SparseOperator::Matrix& a, const op::SparseOperator::Matrix& b) {
    const auto nnz = static_cast<std::size_t>(a.nonZeros());
    return a.nonZeros() == b.nonZeros() &&
           std::equal(a.valuePtr(), a.valuePtr() + nnz, b.valuePtr(),
                      [](const Complex& x, const Complex& y) {
                          return x.real() == y.real() && x.imag() == y.imag();
                      });
}

/// --assembly-study (file comment): near-ring assembly 5c / 5a / dense-same-pairs, diagonal.
void assembly_study(const RoughCase& c, const FastSettings& f, int reps, bool dense_compare,
                    bool quick, const std::string& sum) {
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    const Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam,
                         pml_rough_ag::make_config(c));
    const op::Problem& p = sim.problem();
    mlfmm::MlfmmParams mp;
    mp.accuracy_digits = f.digits;
    const mlfmm::Octree tree = mlfmm::make_octree(*p.space, pml_rough_ag::kLambda, mp);
    auto t0 = std::chrono::steady_clock::now();
    const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree, *p.space, *p.stretch);
    const Real plan_s = seconds(t0);
    int threads = 1;
#ifdef SPECKLEBEM_HAVE_OPENMP
    threads = omp_get_max_threads();
#endif
    char buf[900];
    const auto emit = [&]() {
        std::printf("%s\n", buf);
        std::fflush(stdout);
        append_summary(sum, buf);
    };
    std::snprintf(buf, sizeof(buf),
                  "ASSEMBLY | %s | d0 %.1f | 2N %lld | stretched %lld | ring near pairs %lld | "
                  "plan %.2f s | assembly cache estimate %.1f MB | threads %d",
                  c.tag.c_str(), f.digits, static_cast<long long>(2 * p.space->size()),
                  static_cast<long long>(plan.estimate.stretched),
                  static_cast<long long>(plan.split.ring_pairs), plan_s,
                  static_cast<Real>(plan.estimate.near_assembly_bytes) / 1e6, threads);
    emit();
    std::shared_ptr<op::SparseOperator> first;   // the 5a (per-ordering) matrix of rep 0
    std::shared_ptr<op::SparseOperator> ring5c;  // the default (translation classes)
    for (int r = 0; r < reps; ++r) {
        for (const auto ev : {op::StretchedPairEvaluation::per_ordering,
                              op::StretchedPairEvaluation::translation_classes,
                              op::StretchedPairEvaluation::canonical_cache}) {
            if (quick && ev == op::StretchedPairEvaluation::canonical_cache)
                continue;
            op::SparseStretchedStats st;
            op::SparseStretchedOptions o;
            o.evaluation = ev;
            o.stats = &st;
            t0 = std::chrono::steady_clock::now();
            const auto Z = op::assemble_sparse_stretched(p, plan.split.ring, o);
            const Real sec = seconds(t0);
            const char* name = ev == op::StretchedPairEvaluation::translation_classes ? "5c"
                               : ev == op::StretchedPairEvaluation::canonical_cache
                                   ? "5c canonical only"
                                   : "5a";
            if (ev == op::StretchedPairEvaluation::per_ordering && !first)
                first = Z;
            if (ev == op::StretchedPairEvaluation::translation_classes)
                ring5c = Z;
            std::snprintf(buf, sizeof(buf),
                          "ASSEMBLY | %s | d0 %.1f | rep %d | near ring %s | %.1f s (cache %.1f, "
                          "scatter %.1f) | ordered pairs %lld (stretched %lld) | evaluated "
                          "stretched %lld (touching %lld, %lld copied from %lld translation "
                          "classes) | cache %.1f MB",
                          c.tag.c_str(), f.digits, r, name, sec, st.cache_seconds,
                          st.scatter_seconds, static_cast<long long>(st.ordered_pairs),
                          static_cast<long long>(st.stretched_pairs),
                          static_cast<long long>(st.canonical_pairs > 0
                                                     ? st.canonical_pairs - st.translated_pairs
                                                     : st.stretched_pairs),
                          static_cast<long long>(st.touching_pairs),
                          static_cast<long long>(st.translated_pairs),
                          static_cast<long long>(st.translation_classes),
                          static_cast<Real>(st.cache_bytes) / 1e6);
            emit();
            if (first) {
                const auto& a = Z->matrix();
                const auto& b = first->matrix();
                Real worst = 0, worst_abs = 0, scale = 0;
                Index zero_mismatch = 0;
                for (Index k = 0; k < b.nonZeros(); ++k) {
                    const Complex x = a.valuePtr()[k], y = b.valuePtr()[k];
                    scale = std::max(scale, std::abs(y));
                    worst_abs = std::max(worst_abs, std::abs(x - y));
                    if (y != Complex(0.0, 0.0))
                        worst = std::max(worst, std::abs(x - y) / std::abs(y));
                    else if (x != Complex(0.0, 0.0))
                        ++zero_mismatch;
                }
                std::snprintf(
                    buf, sizeof(buf),
                    "ASSEMBLY | %s | d0 %.1f | rep %d | near ring %s vs 5a | bitwise %s | "
                    "max per-entry rel diff %.2e | max |diff| / max |entry| %.2e | "
                    "zero entries changed %lld",
                    c.tag.c_str(), f.digits, r, name, same_values(a, b) ? "yes" : "no", worst,
                    scale > 0 ? worst_abs / scale : 0.0, static_cast<long long>(zero_mismatch));
                emit();
            }
        }
        if (quick)
            continue;
        Index evaluated = 0;
        const Real dsec = dense_same_pairs_seconds(p, plan.split.ring, evaluated);
        std::snprintf(buf, sizeof(buf),
                      "ASSEMBLY | %s | d0 %.1f | rep %d | dense same pairs %.1f s | evaluated "
                      "%lld triangle pairs",
                      c.tag.c_str(), f.digits, r, dsec, static_cast<long long>(evaluated));
        emit();
    }
    if (dense_compare) {
        // Per-entry differences of the near-ring matrices against the dense PML matrix (summation
        // order A + A^T + rest): the rounding level the dense and sparse assemblers already
        // differ by.
        const auto D = op::DenseStrategy().build(p);
        const auto* Dd = dynamic_cast<const op::DenseOperator*>(D.get());
        for (const auto& [name, Z] :
             {std::pair<const char*, std::shared_ptr<op::SparseOperator>>{"5a", first},
              {"5c", ring5c}}) {
            const auto& M = Z->matrix();
            Real worst = 0, worst_abs = 0, scale = 0;
            for (Index r = 0; r < M.rows(); ++r) {
                for (Index k = M.outerIndexPtr()[r]; k < M.outerIndexPtr()[r + 1]; ++k) {
                    const Complex x = M.valuePtr()[k];
                    const Complex y = Dd->matrix()(r, M.innerIndexPtr()[k]);
                    scale = std::max(scale, std::abs(y));
                    worst_abs = std::max(worst_abs, std::abs(x - y));
                    if (y != Complex(0.0, 0.0))
                        worst = std::max(worst, std::abs(x - y) / std::abs(y));
                }
            }
            std::snprintf(buf, sizeof(buf),
                          "ASSEMBLY | %s | d0 %.1f | near ring %s vs dense | max per-entry rel "
                          "diff %.2e | max |diff| / max |entry| %.2e",
                          c.tag.c_str(), f.digits, name, worst,
                          scale > 0 ? worst_abs / scale : 0.0);
            emit();
        }
    }
    if (!quick) {
        profile_canonical_pairs(p, plan.split.ring, c.tag, f.digits, sum);
        translation_classes(p, plan.split.ring, c.mesh_size, c.tag, f.digits, 200, sum);
    }
    // Diagonal: op::assemble_diagonal (5a) against the near parts (5c, MlfmmOperator::diagonal).
    t0 = std::chrono::steady_clock::now();
    const VectorXc dref = op::assemble_diagonal(p);
    const Real diag_s = seconds(t0);
    op::Problem real_problem = p;
    real_problem.stretch = nullptr;
    const auto rr = op::assemble_sparse(real_problem, plan.split.real);
    const Index N = p.space->size();
    t0 = std::chrono::steady_clock::now();
    VectorXc d(2 * N);
    for (Index r = 0; r < 2 * N; ++r) {
        const bool s = plan.stretched[static_cast<std::size_t>(r % N)] != 0;
        d(r) = (s ? ring5c : rr)->matrix().coeff(r, r);
    }
    const Real read_s = seconds(t0);
    Real worst = 0;
    bool real_bitwise = true;
    for (Index r = 0; r < 2 * N; ++r) {
        if (plan.stretched[static_cast<std::size_t>(r % N)] != 0)
            worst = std::max(worst, std::abs(d(r) - dref(r)) / std::abs(dref(r)));
        else
            real_bitwise = real_bitwise && d(r) == dref(r);
    }
    std::snprintf(buf, sizeof(buf),
                  "ASSEMBLY | %s | d0 %.1f | diagonal assemble_diagonal %.1f s | from near parts "
                  "%.3f s | stretched rows max rel diff %.2e | real rows bitwise %s",
                  c.tag.c_str(), f.digits, diag_s, read_s, worst, real_bitwise ? "yes" : "NO");
    emit();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // --compare-far TEST REF [--summary FILE]: the 7c/7d metric between two stored far fields
        // (e.g. the fast-path 7d Si control: L x 1.2 against U).
        if (argc >= 4 && std::string(argv[1]) == "--compare-far") {
            const pml_rough_ag::FarField t = pml_rough_ag::read_far_field(argv[2]);
            const pml_rough_ag::FarField r = pml_rough_ag::read_far_field(argv[3]);
            const pml_rough_ag::Metrics m = pml_rough_ag::compare(t, r);
            const std::string line = std::string("COMPARE_FAR | ") + argv[2] + " vs " + argv[3] +
                                     " | " + pml_rough_ag::compare_line(t, r, m);
            std::printf("%s\n", line.c_str());
            if (argc >= 6 && std::string(argv[4]) == "--summary")
                append_summary(argv[5], line);
            return 0;
        }
        const Options o = parse(argc, argv);
        const RoughCase& c = o.c;
        const pml_rough_ag::Size s = pml_rough_ag::size_of(c);
        const Real avail = system_memory::available_memory_bytes();
        std::printf(
            "SETUP | %s | %s | rms %.3g | L %.4g | half widths %.4g / %.4g / %.4g / %.4g m | "
            "triangles %lld | 2N %lld | dense %.2f GB | available %.1f GB\n",
            c.tag.c_str(), c.material.c_str(), c.rms, c.patch, s.half[0], s.half[1], s.half[2],
            s.half[3], static_cast<long long>(s.triangles), static_cast<long long>(s.unknowns),
            s.dense_bytes / 1e9, avail / 1e9);
        const std::string est = estimate_line(c, o.f);
        std::printf("%s\n", est.c_str());
        std::fflush(stdout);
        append_summary(o.summary, est);
        if (o.estimate_only)
            return 0;
        if (o.ring_compare) {
            const RingCompare r = run_ring_compare(c, o.f, !o.no_exact);
            const std::string line = ring_compare_line(c, o.f, r);
            std::printf("%s\n", line.c_str());
            append_summary(o.summary, line);
            std::fflush(stdout);
            return 0;
        }
        if (o.assembly_study) {
            assembly_study(c, o.f, o.reps, o.dense_compare, o.quick, o.summary);
            return 0;
        }
        if (!o.fast_only && !o.force && !(avail > s.dense_bytes + 4e9)) {
            std::fprintf(stderr,
                         "pml_ring: dense %.1f GB + 4 GB margin exceeds the available %.1f GB "
                         "(--fast-only or --force)\n",
                         s.dense_bytes / 1e9, avail / 1e9);
            return 2;
        }
        const FastRun fr = run_fast(c, o.f);
        std::printf("%s\n", fr.report.c_str());
        std::fflush(stdout);
        if (o.fast_only) {
            char buf[600];
            std::snprintf(
                buf, sizeof(buf),
                "FAST | %s | %s | rms %.3g | L %.4g | h %.4g | d0 %.1f | 2N %lld | stretched %lld "
                "| "
                "levels %d | R %.8f | its %d | true res %.2e | op %.1f MB (ring near %.1f, far "
                "%.1f) | assemble (op + rhs + diag) %.1f s (ring near %.1f, far %.1f) | GMRES %.1f "
                "s | matvec %.3f s | peak %.2f GB",
                c.tag.c_str(), c.material.c_str(), c.rms, c.patch, c.mesh_size, o.f.digits,
                static_cast<long long>(fr.unknowns), static_cast<long long>(fr.stretched),
                fr.levels, fr.far.reflectance(), fr.iterations, fr.true_residual,
                static_cast<Real>(fr.operator_bytes) / 1e6,
                static_cast<Real>(fr.ring_near_bytes) / 1e6,
                static_cast<Real>(fr.ring_far_bytes) / 1e6, fr.assemble_s, fr.ring_near_s,
                fr.ring_far_s, fr.solve_s, fr.matvec_s, fr.peak_rss / 1e9);
            const std::string line = std::string(buf) + " | " + far_segment(fr.far_stats);
            std::printf("%s\n", line.c_str());
            append_summary(o.summary, line);
            // The far field for d0 comparisons between fast runs (--summary FILE.far: |F|^2).
            if (!o.summary.empty())
                pml_rough_ag::write_far_field(o.summary + "." + c.tag + ".far", fr.far);
            return 0;
        }
        const DenseRun dr = run_dense(c, fr);
        const std::string line = row_line(c, o.f, fr, dr);
        std::printf("%s\n", line.c_str());
        append_summary(o.summary, line);
        const pml_rough_ag::Metrics m = pml_rough_ag::compare(fr.far, dr.far);
        const std::string cmp = pml_rough_ag::compare_line(fr.far, dr.far, m);
        std::printf("%s\n", cmp.c_str());
        append_summary(o.summary, cmp);
        if (!o.summary.empty()) {  // far fields for cross-case metrics (--compare-far)
            pml_rough_ag::write_far_field(o.summary + "." + c.tag + ".far", fr.far);
            pml_rough_ag::write_far_field(o.summary + "." + c.tag + ".dense.far", dr.far);
        }
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_ring: %s\n", ex.what());
        return 1;
    }
}
