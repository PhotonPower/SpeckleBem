/// @file far_operator.cpp
/// Multilevel FMM far operator (WP20a) with the region policy of ADR 0008 §6 (WP21); the passes,
/// signs, weight conventions and the policy are derived in the header comment of far_operator.hpp.
#include "specklebem/compression/mlfmm/far_operator.hpp"

#include "specklebem/compression/mlfmm/interpolation.hpp"
#include "specklebem/compression/mlfmm/plane_wave.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <sstream>
#include <stdexcept>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

namespace specklebem::mlfmm {

namespace {

/// Fields per box: f = 2 * source + component, source 0 = J, 1 = M; component 0 = theta_hat,
/// 1 = phi_hat. Box storage (box, field, direction), contiguous per field.
constexpr std::size_t kFields = 4;
constexpr int kOffsetSlots = 343;  ///< integer offsets in [-3, 3]^3

int max_threads() {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

std::size_t thread_id() {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return static_cast<std::size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

/// First exception thrown inside an OpenMP loop body (exceptions must not leave the region).
class ErrorSlot {
public:
    void capture() {
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp critical(specklebem_far_operator_exception)
#endif
        {
            if (!ptr_)
                ptr_ = std::current_exception();
        }
    }
    void rethrow() const {
        if (ptr_)
            std::rethrow_exception(ptr_);
    }

private:
    std::exception_ptr ptr_;
};

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

std::size_t sz(Index i) {
    return static_cast<std::size_t>(std::max<Index>(i, 0));
}

/// Slot of the integer offset obs - src (each component in [-3, 3] for interaction lists).
std::size_t offset_slot(const std::array<Index, 3>& obs, const std::array<Index, 3>& src) {
    return sz(49 * (obs[0] - src[0] + 3) + 7 * (obs[1] - src[1] + 3) + (obs[2] - src[2] + 3));
}

struct Level {
    FarLevelInfo info;
    SphereSampling sampling{0};
    std::array<int, kOffsetSlots> slot{};  ///< offset -> translator index, -1 if absent
    std::vector<VectorXc> translators;     ///< w_q T_L(k, offset * a, khat_q)
    /// Transition to the parent level (levels > 2): interpolation of this level's sampling to the
    /// parent's (null: identical samplings) and the phase shifts per child slot at the parent
    /// directions, up = e^{+jk khat.(c_child - c_parent)}, down = e^{-jk khat.(...)}.
    std::unique_ptr<SphereInterpolator> interp;
    std::array<VectorXc, 8> up, down;
};

/// Uninitialised storage of n complex values (std::complex<double> is an implicit-lifetime type).
/// The passes zero every box segment before accumulating into it, so the memory is first touched
/// by the threads that own the boxes and a reused workspace needs no separate zeroing sweep.
class FieldBuffer {
public:
    explicit FieldBuffer(std::size_t n)
        : n_(n), p_(n > 0 ? std::allocator<Complex>().allocate(n) : nullptr) {}
    ~FieldBuffer() {
        if (p_ != nullptr)
            std::allocator<Complex>().deallocate(p_, n_);
    }
    FieldBuffer(const FieldBuffer&) = delete;
    FieldBuffer& operator=(const FieldBuffer&) = delete;
    [[nodiscard]] Complex* data() const { return p_; }

private:
    std::size_t n_;
    Complex* p_;
};

/// Pass storage of one apply: the outgoing and incoming fields of every active region and level
/// in one block (pointers per region and level) and the per-thread scratch.
struct Workspace {
    explicit Workspace(std::size_t n) : fields(n) {}
    FieldBuffer fields;
    std::array<std::vector<Complex*>, 2> out, in;
    std::vector<Complex> scratch;
};

struct Region {
    bool active = false;
    Complex k;
    Complex alpha, beta, gamma, delta;  ///< e c_L, e c_K, h c_K, m c_L
    int first_level = 0;                ///< coarsest expansion level (levels[0])
    std::vector<Level> levels;          ///< expansion levels first_level ... leaf (may be empty)
    std::vector<FarLevelInfo> info;     ///< every level 2 ... leaf with its decision
    std::unique_ptr<RadiationPatterns> patterns;
    std::vector<std::size_t> antipode;  ///< leaf directions: index of -khat_q
    Real pattern_seconds = 0;
    std::vector<std::array<Index, 2>> exact_pairs;  ///< (observer box, source box), decision exact
    std::shared_ptr<op::SparseOperator> exact;      ///< their exact region-i entries
    Real exact_seconds = 0;
    Real decay = 0;                   ///< alpha = -Im k
    bool force_exact = false;         ///< MlfmmParams::exact_far_regions
    Index truncated_basis_pairs = 0;  ///< basis pairs of exact box pairs dropped by the bound
    Real truncated_basis_bound = 0;   ///< their largest delta
};

/// Test hook: workspace allocations still to fail (testing::fail_next_workspace_allocations).
std::atomic<int> g_failing_workspaces{0};

/// The formulation with the weights of one region set to zero: assemble_sparse then yields only
/// the other region's entries (the masked region is skipped, DenseStrategy semantics).
class RegionMask final : public formulation::Formulation {
public:
    RegionMask(const formulation::Formulation& f, std::size_t keep) : f_(f), keep_(keep) {}
    [[nodiscard]] formulation::Kind kind() const override { return f_.kind(); }
    [[nodiscard]] std::string name() const override { return f_.name(); }
    [[nodiscard]] formulation::Weights weights(Complex eta1, Complex eta2) const override {
        formulation::Weights w = f_.weights(eta1, eta2);
        const Complex zero(0.0, 0.0);
        if (keep_ == 0)
            w.a2 = w.b2 = zero;
        else
            w.a1 = w.b1 = zero;
        return w;
    }

private:
    const formulation::Formulation& f_;
    std::size_t keep_;
};

kernels::RegionParams region_params(const material::Material& m, Real omega) {
    return {m.wavenumber(omega), m.wave_impedance(omega), omega, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

/// Decay factor of the policy (far_operator.hpp): (1 + alpha d) e^{-alpha d}, 1 for d <= 0.
Real decay_factor(Real alpha, Real d) {
    if (!(d > 0.0))
        return 1.0;
    const Real x = alpha * d;
    return (1.0 + x) * std::exp(-x);
}

/// Bounding box (lo, hi) of the support vertices of a box's basis functions.
using Extent = std::array<Real, 6>;

Real extent_distance(const Extent& a, const Extent& b) {
    Real d2 = 0.0;
    for (std::size_t c = 0; c < 3; ++c) {
        const Real g = std::max({0.0, b[c] - a[c + 3], a[c] - b[c + 3]});
        d2 += g * g;
    }
    return std::sqrt(d2);
}

}  // namespace

Real block_check_tolerance(Real digits) {
    return std::pow(10.0, -digits);
}

const char* to_string(FarDecision d) {
    switch (d) {
        case FarDecision::expansion:
            return "expansion";
        case FarDecision::truncation:
            return "truncation";
        case FarDecision::exact:
            return "exact";
    }
    return "?";
}

namespace testing {
void fail_next_workspace_allocations(int count) {
    g_failing_workspaces.store(std::max(count, 0));
}
}  // namespace testing

struct MlfmmFarOperator::Impl {
    explicit Impl(const Octree& t) : tree(t) {}
    const Octree& tree;
    Index n = 0;
    Real digits = 0;
    int interp_order = 0;
    std::array<Region, 2> region;
    std::vector<Index> local;            ///< box -> position within its level
    std::vector<Extent> extent;          ///< box -> bounding box of its bases' supports
    Real rmax = 0;                       ///< max_support_radius
    PatternOptions popt;                 ///< leaf pattern / block check pattern quadrature
    std::size_t field_entries = 0;       ///< complex entries of one workspace's fields
    std::size_t max_nd = 0, max_ws = 0;  ///< largest sampling / interpolator workspace

    /// Pool of reusable workspaces: apply() takes one (allocating only if none is free) and
    /// returns it, so serial applies reuse one workspace and concurrent ones get their own.
    /// Capacity >= workspaces (reserved in take()), so give_back never allocates.
    mutable std::mutex pool_mutex;
    mutable std::vector<std::unique_ptr<Workspace>> pool;
    mutable std::size_t workspaces = 0;  ///< allocated so far (pooled or in use)

    [[nodiscard]] std::unique_ptr<Workspace> take() const;
    [[nodiscard]] std::unique_ptr<Workspace> new_workspace() const;
    void give_back(std::unique_ptr<Workspace> w) const noexcept;

    void compute_extents(const basis::RwgSpace& space);
    /// Policy decisions, orders and samplings of every level (file comment of the header);
    /// throws TruncationOrderError before any expensive setup.
    void plan_region(Region& r, const op::Problem& problem, std::size_t index, bool force_exact);
    /// Block check of one level (header comment): max relative Frobenius error, pairs used.
    [[nodiscard]] std::pair<Real, Index> block_check(const basis::RwgSpace& space, int level,
                                                     const kernels::RegionParams& rp,
                                                     const SphereSampling& s, int order) const;
    /// Block error of the lossless analogue (k = Re k) of a region at a level; -1 if its order
    /// search is not achievable.
    [[nodiscard]] Real reference_block_error(const basis::RwgSpace& space, int level,
                                             const kernels::RegionParams& rp) const;
    /// Exact region-i entries of the exact box pairs (op::assemble_sparse, other region masked).
    void assemble_exact(Region& r, const op::Problem& problem, std::size_t index) const;
    /// Translators, interpolators, phase shifts and leaf patterns of the expansion levels.
    void build_region(Region& r, const basis::RwgSpace& space);
    void apply_region(const Region& r, std::span<Complex* const> out, std::span<Complex* const> in,
                      Complex* scratch, const VectorXc& x, VectorXc& y) const;
};

std::unique_ptr<Workspace> MlfmmFarOperator::Impl::new_workspace() const {
    int failing = g_failing_workspaces.load();
    while (failing > 0 && !g_failing_workspaces.compare_exchange_weak(failing, failing - 1)) {
    }
    if (failing > 0)
        throw std::bad_alloc();
    auto w = std::make_unique<Workspace>(field_entries);
    Complex* base = w->fields.data();
    for (std::size_t i = 0; i < 2; ++i) {
        for (const Level& lv : region[i].levels) {
            const std::size_t size = sz(lv.info.boxes) * kFields * sz(lv.sampling.size());
            w->out[i].push_back(base);
            w->in[i].push_back(base + size);
            base += 2 * size;
        }
    }
    return w;
}

std::unique_ptr<Workspace> MlfmmFarOperator::Impl::take() const {
    std::unique_ptr<Workspace> w;
    {
        const std::lock_guard<std::mutex> lock(pool_mutex);
        if (!pool.empty()) {
            w = std::move(pool.back());
            pool.pop_back();
        }
    }
    if (!w) {
        // Built completely before it is counted: a failure leaves the pool unchanged. The
        // reserve keeps capacity >= workspaces, so give_back's push_back cannot allocate.
        w = new_workspace();
        const std::lock_guard<std::mutex> lock(pool_mutex);
        pool.reserve(workspaces + 1);
        ++workspaces;
    }
    // Per-thread scratch: two direction buffers (or the 4 reception fields) + the interpolator
    // workspace; the thread count may change between calls.
    const std::size_t need = sz(std::max(max_threads(), 1)) * (4 * max_nd + max_ws);
    if (w->scratch.size() < need) {
        try {
            w->scratch.resize(need);
        } catch (...) {
            give_back(std::move(w));
            throw;
        }
    }
    return w;
}

void MlfmmFarOperator::Impl::give_back(std::unique_ptr<Workspace> w) const noexcept {
    const std::lock_guard<std::mutex> lock(pool_mutex);
    pool.push_back(std::move(w));  // capacity >= workspaces > pool.size(): no allocation
}

void MlfmmFarOperator::Impl::compute_extents(const basis::RwgSpace& space) {
    const Real inf = std::numeric_limits<Real>::infinity();
    extent.assign(tree.boxes().size(), Extent{inf, inf, inf, -inf, -inf, -inf});
    const geometry::TriangleMesh& mesh = space.mesh();
    const Vertices& v = mesh.vertices();
    const std::vector<Index>& perm = tree.permutation();
    for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
        const Box& box = tree.boxes()[sz(b)];
        Extent& e = extent[sz(b)];
        for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
            const Index bn = perm[sz(p)];
            for (const Index vi : {mesh.edges()(bn, 0), mesh.edges()(bn, 1),
                                   space.plus_free_vertex(bn), space.minus_free_vertex(bn)}) {
                for (Eigen::Index c = 0; c < 3; ++c) {
                    const auto cc = static_cast<std::size_t>(c);
                    e[cc] = std::min(e[cc], v(vi, c));
                    e[cc + 3] = std::max(e[cc + 3], v(vi, c));
                }
            }
        }
    }
    for (int l = tree.leaf_level() - 1; l >= 0; --l) {
        for (const Index b : tree.boxes_at_level(l)) {
            for (const Index c : tree.boxes()[sz(b)].children) {
                if (c < 0)
                    continue;
                for (std::size_t j = 0; j < 3; ++j) {
                    extent[sz(b)][j] = std::min(extent[sz(b)][j], extent[sz(c)][j]);
                    extent[sz(b)][j + 3] = std::max(extent[sz(b)][j + 3], extent[sz(c)][j + 3]);
                }
            }
        }
    }
}

std::pair<Real, Index> MlfmmFarOperator::Impl::block_check(const basis::RwgSpace& space, int level,
                                                           const kernels::RegionParams& rp,
                                                           const SphereSampling& s,
                                                           int order) const {
    const auto& boxes = tree.boxes();
    // Nearest offset classes (sorted |offset| triple): per class the pair with the most bases.
    std::map<std::array<Index, 4>, std::array<Index, 3>> classes;  // key -> (A, B, min count)
    for (const Index ia : tree.boxes_at_level(level)) {
        const Box& A = boxes[sz(ia)];
        for (const Index ib : A.interaction_list) {
            const Box& B = boxes[sz(ib)];
            std::array<Index, 3> o{};
            for (std::size_t c = 0; c < 3; ++c) o[c] = std::abs(A.ijk[c] - B.ijk[c]);
            std::sort(o.begin(), o.end());
            const std::array<Index, 4> key{o[0] * o[0] + o[1] * o[1] + o[2] * o[2], o[0], o[1],
                                           o[2]};
            const Index count = std::min(A.num_elements, B.num_elements);
            auto it = classes.find(key);
            if (it == classes.end() || count > it->second[2])
                classes[key] = {ia, ib, count};
        }
    }
    const geometry::TriangleMesh& mesh = space.mesh();
    const Vertices& v = mesh.vertices();
    const std::vector<Index>& perm = tree.permutation();
    // All bases of a small box; else the half farthest from the centre (corner-near) and an
    // evenly spaced selection of the rest.
    const auto select = [&](const Box& box) {
        std::vector<Index> all(perm.begin() + box.first_element,
                               perm.begin() + box.first_element + box.num_elements);
        constexpr auto kMax = static_cast<std::size_t>(kBlockCheckBases);
        if (all.size() <= kMax)
            return all;
        const auto dist = [&](Index bn) {
            const Vec3 mid =
                0.5 * (v.row(mesh.edges()(bn, 0)) + v.row(mesh.edges()(bn, 1))).transpose();
            return (mid - box.center).norm();
        };
        std::stable_sort(all.begin(), all.end(),
                         [&](Index a, Index b) { return dist(a) > dist(b); });
        std::vector<Index> out(all.begin(), all.begin() + kMax / 2);
        const std::size_t rest = all.size() - kMax / 2;
        for (std::size_t i = 0; i < kMax - kMax / 2; ++i)
            out.push_back(all[kMax / 2 + (2 * i + 1) * rest / (2 * (kMax - kMax / 2))]);
        return out;
    };
    // Antipodes of the sampling (RadiationPatterns convention).
    const Index nd = s.size();
    std::vector<Index> anti(sz(nd));
    for (int it = 0; it < s.num_theta(); ++it) {
        for (int ip = 0; ip < s.num_phi(); ++ip) {
            anti[sz(s.index(it, ip))] =
                s.index(s.num_theta() - 1 - it, (ip + s.num_phi() / 2) % s.num_phi());
        }
    }
    kernels::OperatorOptions exact_opt;
    exact_opt.target_accuracy = 0.01 * std::pow(10.0, -digits);
    exact_opt.decay_aware_target = false;
    const Real c16 = 16.0 * constants::pi * constants::pi;
    Real worst = 0.0;
    Index used = 0;
    for (const auto& [key, pair] : classes) {
        if (used == kBlockCheckPairs)
            break;
        ++used;
        const Box& A = boxes[sz(pair[0])];
        const Box& B = boxes[sz(pair[1])];
        const std::vector<Index> ma = select(A);
        const std::vector<Index> nb = select(B);
        const std::vector<Complex> pa = basis_patterns(space, ma, A.center, rp.k, s, popt);
        const std::vector<Complex> pb = basis_patterns(space, nb, B.center, rp.k, s, popt);
        const VectorXc t = translator(rp.k, A.center - B.center, s, order);
        const auto na = static_cast<Index>(ma.size());
        const auto nbs = static_cast<Index>(nb.size());
        MatrixXc rw(na, 2 * nd), vm(nbs, 2 * nd), wm(nbs, 2 * nd);
        for (Index q = 0; q < nd; ++q) {
            const Complex sq = s.weights()(q) * t(q);
            const std::size_t qa = sz(anti[sz(q)]);
            for (Index i = 0; i < na; ++i) {  // R_theta = V_theta(q'), R_phi = -V_phi(q')
                rw(i, q) = sq * pa[(sz(i) * sz(nd) + qa) * 2];
                rw(i, nd + q) = -sq * pa[(sz(i) * sz(nd) + qa) * 2 + 1];
            }
            for (Index j = 0; j < nbs; ++j) {
                const Complex vt = pb[(sz(j) * sz(nd) + sz(q)) * 2];
                const Complex vp = pb[(sz(j) * sz(nd) + sz(q)) * 2 + 1];
                vm(j, q) = vt;
                vm(j, nd + q) = vp;
                wm(j, q) = -vp;
                wm(j, nd + q) = vt;
            }
        }
        const MatrixXc Lf = (rp.omega * rp.mu * rp.k / c16) * (rw * vm.transpose());
        const MatrixXc Kf = (rp.k * rp.k / c16) * (rw * wm.transpose());
        // Exact entries from the support triangles (each triangle pair once).
        MatrixXc Le = MatrixXc::Zero(na, nbs), Ke = MatrixXc::Zero(na, nbs);
        std::vector<Index> row(sz(space.size()), -1), col(sz(space.size()), -1), ta, tb;
        for (Index i = 0; i < na; ++i) {
            row[sz(ma[sz(i)])] = i;
            ta.push_back(space.plus_triangle(ma[sz(i)]));
            ta.push_back(space.minus_triangle(ma[sz(i)]));
        }
        for (Index j = 0; j < nbs; ++j) {
            col[sz(nb[sz(j)])] = j;
            tb.push_back(space.plus_triangle(nb[sz(j)]));
            tb.push_back(space.minus_triangle(nb[sz(j)]));
        }
        for (std::vector<Index>* tv : {&ta, &tb}) {
            std::sort(tv->begin(), tv->end());
            tv->erase(std::unique(tv->begin(), tv->end()), tv->end());
        }
        Eigen::Matrix<Complex, 3, 3> L, K;
        for (const Index ti : ta) {
            const basis::RwgSpace::Support st = space.support(ti);
            for (const Index si : tb) {
                const basis::RwgSpace::Support ss = space.support(si);
                kernels::element_blocks(space, ti, si, rp, exact_opt, L, K);
                for (int x = 0; x < st.count; ++x) {
                    const Index i = row[sz(st.n[x])];
                    for (int y = 0; i >= 0 && y < ss.count; ++y) {
                        const Index j = col[sz(ss.n[y])];
                        if (j < 0)
                            continue;
                        Le(i, j) += L(x, y);
                        Ke(i, j) += K(x, y);
                    }
                }
            }
        }
        worst = std::max({worst, (Lf - Le).norm() / Le.norm(), (Kf - Ke).norm() / Ke.norm()});
        if (!std::isfinite(worst))
            worst = std::numeric_limits<Real>::infinity();
    }
    return {worst, used};
}

Real MlfmmFarOperator::Impl::reference_block_error(const basis::RwgSpace& space, int level,
                                                   const kernels::RegionParams& rp) const {
    // Lossless analogue: k = Re k, eps from k^2 = w^2 mu eps, its own order and sampling.
    kernels::RegionParams ref = rp;
    ref.k = Complex(rp.k.real(), 0.0);
    ref.eps = ref.k * ref.k / (rp.omega * rp.omega * rp.mu);
    ref.eta = rp.omega * rp.mu / ref.k;
    TruncationSearch search;
    int sampling_order = 0;
    if (level == tree.leaf_level()) {
        const LeafSampling ls = leaf_sampling(space, tree, ref.k, digits);
        search = ls.search;
        sampling_order = ls.sampling_order;
    } else {
        TruncationSearchOptions opt;
        opt.box_diagonal = std::sqrt(3.0) * tree.box_size(level) + 2.0 * rmax;
        search = search_truncation_order(ref.k, tree.box_size(level), digits, opt);
        sampling_order = search.order;
    }
    if (!search.achievable)
        return -1.0;
    return block_check(space, level, ref, SphereSampling(sampling_order), search.order).first;
}

void MlfmmFarOperator::Impl::plan_region(Region& r, const op::Problem& problem, std::size_t index,
                                         bool force_exact) {
    const basis::RwgSpace& space = *problem.space;
    const int leaf = tree.leaf_level();
    const Real lambda = 2.0 * constants::pi / r.k.real();
    const Real alpha = -r.k.imag();
    const Real trunc_tol = std::pow(10.0, -(digits + 1.0));
    const kernels::RegionParams rp =
        region_params(index == 0 ? problem.exterior : problem.object, problem.omega);
    r.info.assign(sz(leaf - 1), FarLevelInfo{});
    std::vector<Level> expansion;  // leaf first
    bool expanding = !force_exact;
    Real first_fallback_edge = 0.0;
    for (int l = leaf; l >= 2; --l) {
        const auto t0 = std::chrono::steady_clock::now();
        FarLevelInfo& f = r.info[sz(l - 2)];
        const Real a = tree.box_size(l);
        const std::vector<Index>& boxes = tree.boxes_at_level(l);
        f.level = l;
        f.boxes = static_cast<Index>(boxes.size());
        f.box_size = a;
        if (expanding) {
            TruncationSearch search;
            int sampling_order = 0;
            bool searched = true;
            try {
                if (l == leaf) {
                    const LeafSampling ls = leaf_sampling(space, tree, r.k, digits);
                    search = ls.search;
                    sampling_order = ls.sampling_order;
                } else {
                    TruncationSearchOptions opt;
                    opt.box_diagonal = std::sqrt(3.0) * a + 2.0 * rmax;
                    search = search_truncation_order(r.k, a, digits, opt);
                    sampling_order = search.order;
                }
            } catch (const std::underflow_error&) {
                searched = false;  // the interaction underflows: certainly no expansion
            }
            if (searched) {
                f.search_achievable = search.achievable;
                f.search_error = search.error;
                f.truncation_order = search.order;
            }
            if (searched && search.achievable) {
                Level lv;
                lv.sampling = SphereSampling(sampling_order);
                f.sampling_order = sampling_order;
                f.directions = lv.sampling.size();
                std::tie(f.block_error, f.block_pairs) =
                    block_check(space, l, rp, lv.sampling, search.order);
                bool accept = !(alpha > 0.0) || f.block_error <= block_check_tolerance(digits);
                if (!accept) {
                    f.reference_error = reference_block_error(space, l, rp);
                    accept = f.reference_error >= 0.0 &&
                             f.block_error <= kLossDegradationFactor * f.reference_error;
                }
                if (accept) {
                    f.decision = FarDecision::expansion;
                    f.pattern_bytes =
                        2 * kFields * sizeof(Complex) * boxes.size() * sz(lv.sampling.size());
                    f.setup_seconds = seconds_since(t0);
                    SBEM_INFO(
                        "MlfmmFarOperator: region R{} level {} (a = {:.3g} lambda_i): expansion, "
                        "L = {}, search error {:.2e}, block check {:.2e} ({} pairs)",
                        index + 1, l, a / lambda, f.truncation_order, f.search_error, f.block_error,
                        f.block_pairs);
                    expansion.push_back(std::move(lv));
                    continue;
                }
            }
            expanding = false;
            first_fallback_edge = a;
        }
        // Per box pair: truncation by the decay bound, else exact.
        f.decision = FarDecision::truncation;
        for (const Index ia : boxes) {
            for (const Index ib : tree.boxes()[sz(ia)].interaction_list) {
                const Real delta =
                    decay_factor(alpha, extent_distance(extent[sz(ia)], extent[sz(ib)]));
                if (!force_exact && delta <= trunc_tol) {
                    ++f.truncated_pairs;
                    f.decay_bound = std::max(f.decay_bound, delta);
                } else {
                    f.exact_bound = f.exact_pairs == 0 ? delta : std::min(f.exact_bound, delta);
                    ++f.exact_pairs;
                    r.exact_pairs.push_back({ia, ib});
                }
            }
        }
        if (f.exact_pairs > 0)
            f.decision = FarDecision::exact;
        f.setup_seconds = seconds_since(t0);
        SBEM_INFO(
            "MlfmmFarOperator: region R{} level {} (a = {:.3g} lambda_i): {} (search {} error "
            "{:.2e}, block check {:.2e}); {} box pairs truncated (delta <= {:.2e}, bound 1e-{}), "
            "{} exact (delta >= {:.2e})",
            index + 1, l, a / lambda, to_string(f.decision),
            f.search_achievable ? "achievable," : "not achievable,", f.search_error, f.block_error,
            f.truncated_pairs, f.decay_bound, digits + 1.0, f.exact_pairs, f.exact_bound);
        if (f.exact_pairs > 0 && !force_exact && !(alpha * first_fallback_edge >= 1.0)) {
            // The fallback needs decay over a box edge; otherwise it would make the region dense.
            const Real D = std::sqrt(3.0) * a + 2.0 * rmax;
            const auto cause = rmax / a > 0.3     ? TruncationOrderError::Cause::mesh_or_leaf_size
                               : alpha * D >= 1.0 ? TruncationOrderError::Cause::lossy_region
                                                  : TruncationOrderError::Cause::digits;
            std::ostringstream os;
            os << "MlfmmFarOperator: region R" << index + 1 << " (k = " << r.k << " 1/m), level "
               << l << " (box edge " << a / lambda << " lambda_i): no expansion meets 10^-"
               << digits << " (order search "
               << (f.search_achievable ? "achievable" : "not achievable") << ", best "
               << f.search_error << " at L = " << f.truncation_order << "; block check "
               << f.block_error << "), and the decay is too weak for the ADR 0008 §6 fallback "
               << "(-Im k a = " << alpha * first_fallback_edge << " < 1); ";
            if (cause == TruncationOrderError::Cause::mesh_or_leaf_size)
                os << "refine the mesh or use larger leaves";
            else if (cause == TruncationOrderError::Cause::lossy_region)
                os << "the loss spoils the expansion at this box size: use smaller or larger "
                      "leaves, fewer digits or the dense operator";
            else
                os << "use fewer digits or larger leaves";
            os << " (r_max / a = " << rmax / a << ")";
            throw TruncationOrderError(os.str(), static_cast<int>(index), l, cause);
        }
    }
    std::reverse(expansion.begin(), expansion.end());
    r.first_level = leaf - static_cast<int>(expansion.size()) + 1;
    for (std::size_t li = 0; li < expansion.size(); ++li)
        expansion[li].info = r.info[sz(r.first_level - 2) + li];
    r.levels = std::move(expansion);
}

void MlfmmFarOperator::Impl::assemble_exact(Region& r, const op::Problem& problem,
                                            std::size_t index) const {
    if (r.exact_pairs.empty())
        return;
    const auto t0 = std::chrono::steady_clock::now();
    const auto& boxes = tree.boxes();
    const std::vector<Index>& leaves = tree.boxes_at_level(tree.leaf_level());
    const std::vector<Index>& perm = tree.permutation();
    std::vector<Index> first(leaves.size());
    for (std::size_t g = 0; g < leaves.size(); ++g) first[g] = boxes[sz(leaves[g])].first_element;
    // Source boxes per observer leaf: the leaves of A are those whose elements lie in A's range.
    std::vector<std::vector<Index>> src(leaves.size());
    for (const auto& [ia, ib] : r.exact_pairs) {
        const Box& A = boxes[sz(ia)];
        const auto lo = std::lower_bound(first.begin(), first.end(), A.first_element);
        const auto hi = std::lower_bound(lo, first.end(), A.first_element + A.num_elements);
        for (auto it = lo; it != hi; ++it) src[sz(it - first.begin())].push_back(ib);
    }
    // One group per row: the basis pairs of the exact box pairs whose own bound exceeds the
    // truncation tolerance (all of them for a forced region).
    const basis::RwgSpace& space = *problem.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index nb = space.size();
    std::vector<Extent> be(sz(nb));
    for (Index q = 0; q < nb; ++q) {
        const Real inf = std::numeric_limits<Real>::infinity();
        Extent e{inf, inf, inf, -inf, -inf, -inf};
        for (const Index vi : {mesh.edges()(q, 0), mesh.edges()(q, 1), space.plus_free_vertex(q),
                               space.minus_free_vertex(q)}) {
            for (std::size_t c = 0; c < 3; ++c) {
                const Real x = mesh.vertices()(vi, static_cast<Eigen::Index>(c));
                e[c] = std::min(e[c], x);
                e[c + 3] = std::max(e[c + 3], x);
            }
        }
        be[sz(q)] = e;
    }
    const Real tol = std::pow(10.0, -(digits + 1.0));
    op::BasisPattern pat;
    pat.group_of_row.assign(sz(nb), 0);
    pat.col_ptr = {0, 0};  // group 0: rows without exact pairs
    for (std::size_t g = 0; g < leaves.size(); ++g) {
        if (src[g].empty())
            continue;
        const Box& X = boxes[sz(leaves[g])];
        for (Index p = X.first_element; p < X.first_element + X.num_elements; ++p) {
            const Index row = perm[sz(p)];
            const std::size_t start = pat.cols.size();
            for (const Index b : src[g]) {
                const Box& B = boxes[sz(b)];
                for (Index q = B.first_element; q < B.first_element + B.num_elements; ++q) {
                    const Index c = perm[sz(q)];
                    const Real delta =
                        decay_factor(r.decay, extent_distance(be[sz(row)], be[sz(c)]));
                    if (r.force_exact || delta > tol) {
                        pat.cols.push_back(c);
                    } else {
                        ++r.truncated_basis_pairs;
                        r.truncated_basis_bound = std::max(r.truncated_basis_bound, delta);
                    }
                }
            }
            std::sort(pat.cols.begin() + static_cast<std::ptrdiff_t>(start), pat.cols.end());
            pat.group_of_row[sz(row)] = static_cast<Index>(pat.col_ptr.size() - 1);
            pat.col_ptr.push_back(static_cast<Index>(pat.cols.size()));
        }
    }
    const RegionMask mask(*problem.formulation, index);
    op::Problem masked = problem;
    masked.formulation = &mask;
    r.exact = op::assemble_sparse(masked, pat);
    r.exact_seconds = seconds_since(t0);
    SBEM_INFO(
        "MlfmmFarOperator: region R{} exact part: {} box pairs, nnz {} ({:.1f} MB, {:.2f} s); {} "
        "basis pairs of them truncated (delta <= {:.2e})",
        index + 1, r.exact_pairs.size(), r.exact->nonzeros(),
        static_cast<Real>(r.exact->memory_bytes()) / 1048576.0, r.exact_seconds,
        r.truncated_basis_pairs, r.truncated_basis_bound);
}

void MlfmmFarOperator::Impl::build_region(Region& r, const basis::RwgSpace& space) {
    for (std::size_t li = 0; li < r.levels.size(); ++li) {
        const auto t0 = std::chrono::steady_clock::now();
        Level& lv = r.levels[li];
        const int l = lv.info.level;
        const Real a = lv.info.box_size;
        const std::vector<Index>& boxes = tree.boxes_at_level(l);
        lv.slot.fill(-1);
        for (const Index ia : boxes) {
            const Box& A = tree.boxes()[sz(ia)];
            for (const Index ib : A.interaction_list) {
                const Box& B = tree.boxes()[sz(ib)];
                for (std::size_t a2 = 0; a2 < 3; ++a2) {
                    if (std::abs(A.ijk[a2] - B.ijk[a2]) > 3)
                        throw std::logic_error("MlfmmFarOperator: interaction offset > 3");
                }
                const std::size_t s = offset_slot(A.ijk, B.ijk);
                if (lv.slot[s] >= 0)
                    continue;
                lv.slot[s] = static_cast<int>(lv.translators.size());
                const Vec3 d(static_cast<Real>(A.ijk[0] - B.ijk[0]),
                             static_cast<Real>(A.ijk[1] - B.ijk[1]),
                             static_cast<Real>(A.ijk[2] - B.ijk[2]));
                VectorXc t = translator(r.k, a * d, lv.sampling, lv.info.truncation_order);
                t.array() *= lv.sampling.weights().array().cast<Complex>();
                lv.translators.push_back(std::move(t));
            }
        }
        lv.info.translators = static_cast<Index>(lv.translators.size());
        if (li > 0) {
            const SphereSampling& parent = r.levels[li - 1].sampling;
            if (parent.order() != lv.sampling.order()) {
                lv.interp = std::make_unique<SphereInterpolator>(lv.sampling, parent, interp_order);
            }
            const auto& khat = parent.directions();
            const Complex jk = Complex(0.0, 1.0) * r.k;
            for (std::size_t s = 0; s < 8; ++s) {
                const Vec3 d =
                    0.5 * a *
                    Vec3((s & 1U) ? 1.0 : -1.0, (s & 2U) ? 1.0 : -1.0, (s & 4U) ? 1.0 : -1.0);
                const VectorXr phase = khat * d;
                lv.up[s] = (jk * phase.cast<Complex>()).array().exp();
                lv.down[s] = (-jk * phase.cast<Complex>()).array().exp();
            }
        }
        lv.info.setup_seconds += seconds_since(t0);
        r.info[sz(l - 2)] = lv.info;
    }
    if (r.levels.empty())
        return;
    const auto t0 = std::chrono::steady_clock::now();
    r.patterns =
        std::make_unique<RadiationPatterns>(space, tree, r.k, r.levels.back().sampling, popt);
    r.pattern_seconds = seconds_since(t0);
    r.antipode.resize(sz(r.patterns->num_directions()));
    for (std::size_t q = 0; q < r.antipode.size(); ++q) {
        r.antipode[q] = sz(r.patterns->antipode(static_cast<Index>(q)));
    }
}

MlfmmFarOperator::MlfmmFarOperator(const op::Problem& problem, const Octree& tree,
                                   const MlfmmParams& params)
    : impl_(std::make_unique<Impl>(tree)) {
    op::validate(problem);
    if (params.truncation_L != 0 || !params.precompute_translators ||
        params.use_fft_interpolation) {
        throw std::invalid_argument(
            "MlfmmFarOperator: unsupported params (truncation_L must be 0, "
            "precompute_translators true, use_fft_interpolation false)");
    }
    Impl& m = *impl_;
    m.digits = params.accuracy_digits;
    m.interp_order = interpolation_order(params.accuracy_digits);
    const basis::RwgSpace& space = *problem.space;
    m.n = space.size();
    if (m.n != static_cast<Index>(tree.permutation().size())) {
        throw std::invalid_argument("MlfmmFarOperator: the octree was not built on problem.space");
    }
    if (tree.levels() >= 3) {
        // Z_far leaves out the jump terms of K, which only coincident triangles carry. Far pairs
        // have midpoints more than one leaf edge apart, two bases sharing a triangle at most
        // r_max (each midpoint lies in the shared triangle): r_max < a_leaf excludes them.
        m.rmax = max_support_radius(space);
        const Real rmax = m.rmax;
        const Real a = tree.box_size(tree.leaf_level());
        if (!(rmax < a)) {
            std::ostringstream os;
            os << "MlfmmFarOperator: the largest RWG support radius (" << rmax
               << " m) must be smaller than the leaf box edge (" << a
               << " m), otherwise far pairs could share a triangle (jump terms); refine the "
                  "mesh or use larger leaves";
            throw std::invalid_argument(os.str());
        }
    }
    m.local.assign(tree.boxes().size(), -1);
    for (int l = 0; l < tree.levels(); ++l) {
        const std::vector<Index>& b = tree.boxes_at_level(l);
        for (std::size_t i = 0; i < b.size(); ++i) m.local[sz(b[i])] = static_cast<Index>(i);
    }
    const std::array<const material::Material*, 2> mat = {&problem.exterior, &problem.object};
    std::array<Complex, 2> eta;
    for (std::size_t i = 0; i < 2; ++i) eta[i] = mat[i]->wave_impedance(problem.omega);
    const formulation::Weights w = problem.formulation->weights(eta[0], eta[1]);
    const std::array<Complex, 2> e = {w.a1 / eta[0], w.a2 / eta[1]};
    const std::array<Complex, 2> h = {w.b1 * eta[0], w.b2 * eta[1]};
    const std::array<Complex, 2> mm = {w.b1 / eta[0], w.b2 / eta[1]};
    m.popt.target_accuracy = 1e-2 * std::pow(10.0, -params.accuracy_digits);
    if (tree.levels() >= 3)
        m.compute_extents(space);
    const Real c16 = 16.0 * constants::pi * constants::pi;
    for (std::size_t i = 0; i < 2; ++i) {
        Region& r = m.region[i];
        const Complex zero(0.0, 0.0);
        r.active = (e[i] != zero || h[i] != zero || mm[i] != zero) && tree.levels() >= 3;
        r.k = mat[i]->wavenumber(problem.omega);
        const Complex cl = problem.omega * constants::mu0 * mat[i]->mu_r * r.k / c16;
        const Complex ck = r.k * r.k / c16;
        r.alpha = e[i] * cl;
        r.beta = e[i] * ck;
        r.gamma = h[i] * ck;
        r.delta = mm[i] * cl;
        r.decay = -r.k.imag();
        r.force_exact = params.exact_far_regions[i];
        if (r.active)
            m.plan_region(r, problem, i, r.force_exact);
    }
    for (std::size_t i = 0; i < 2; ++i) {
        Region& r = m.region[i];
        if (!r.active)
            continue;
        m.assemble_exact(r, problem, i);
        m.build_region(r, space);
        for (const Level& lv : r.levels) {
            m.field_entries += 2 * sz(lv.info.boxes) * kFields * sz(lv.sampling.size());
            m.max_nd = std::max(m.max_nd, sz(lv.sampling.size()));
            if (lv.interp)
                m.max_ws = std::max(m.max_ws, sz(lv.interp->workspace_size()));
        }
    }
    SBEM_INFO(
        "MlfmmFarOperator: 2N = {}, {} octree levels, d0 = {}, {:.1f} MB (incl. one apply "
        "workspace of {:.1f} MB)",
        rows(), tree.levels(), m.digits, static_cast<Real>(memory_bytes()) / 1048576.0,
        static_cast<Real>(workspace_bytes()) / 1048576.0);
    SBEM_DEBUG("{}", describe());
}

MlfmmFarOperator::~MlfmmFarOperator() = default;

Index MlfmmFarOperator::rows() const {
    return 2 * impl_->n;
}
Index MlfmmFarOperator::cols() const {
    return 2 * impl_->n;
}
const Octree& MlfmmFarOperator::octree() const {
    return impl_->tree;
}

bool MlfmmFarOperator::region_active(int region) const {
    if (region < 0 || region > 1)
        throw std::out_of_range("MlfmmFarOperator: region must be 0 or 1");
    return impl_->region[static_cast<std::size_t>(region)].active;
}

const std::vector<FarLevelInfo>& MlfmmFarOperator::levels(int region) const {
    (void)region_active(region);
    return impl_->region[static_cast<std::size_t>(region)].info;
}

const RadiationPatterns& MlfmmFarOperator::patterns(int region) const {
    if (!region_active(region) || !impl_->region[static_cast<std::size_t>(region)].patterns)
        throw std::invalid_argument("MlfmmFarOperator: region inactive or without expansion");
    return *impl_->region[static_cast<std::size_t>(region)].patterns;
}

const op::SparseOperator* MlfmmFarOperator::exact_part(int region) const {
    (void)region_active(region);
    return impl_->region[static_cast<std::size_t>(region)].exact.get();
}

void MlfmmFarOperator::apply(const VectorXc& x, VectorXc& y) const {
    if (x.size() != cols())
        throw std::invalid_argument("MlfmmFarOperator::apply: x has the wrong size");
    VectorXc out = VectorXc::Zero(rows());
    const Impl& m = *impl_;
    if (m.field_entries > 0) {
        // Returned to the pool also when a pass throws (the passes re-zero what they use).
        struct Lease {
            const Impl& m;
            std::unique_ptr<Workspace> w;
            ~Lease() { m.give_back(std::move(w)); }
        } lease{m, m.take()};
        for (std::size_t i = 0; i < 2; ++i) {
            if (!m.region[i].levels.empty())
                m.apply_region(m.region[i], lease.w->out[i], lease.w->in[i],
                               lease.w->scratch.data(), x, out);
        }
    }
    for (const Region& r : m.region) {
        if (r.exact) {
            VectorXc t;
            r.exact->apply(x, t);
            out += t;
        }
    }
    y = std::move(out);
}

std::size_t MlfmmFarOperator::workspace_bytes() const {
    return impl_->field_entries * sizeof(Complex);
}

std::size_t MlfmmFarOperator::workspaces() const {
    const std::lock_guard<std::mutex> lock(impl_->pool_mutex);
    return impl_->workspaces;
}

void MlfmmFarOperator::Impl::apply_region(const Region& r, std::span<Complex* const> out,
                                          std::span<Complex* const> in, Complex* scratch,
                                          const VectorXc& x, VectorXc& y) const {
    const int leaf = tree.leaf_level();
    const std::size_t nl = r.levels.size();
    // Per-thread scratch (Impl::take): two direction buffers (or the 4 reception fields) + the
    // interpolator workspace.
    const std::size_t stride = 4 * max_nd + max_ws;
    const auto thread_scratch = [&]() { return scratch + thread_id() * stride; };
    // out / in are uninitialised or hold a previous apply: every pass zeroes the box segments it
    // accumulates into (by the thread that owns the box) before writing to them.
    const auto field = [](Complex* v, Index pos, std::size_t f, std::size_t nd) {
        return v + (sz(pos) * kFields + f) * nd;
    };
    const std::vector<Index>& perm = tree.permutation();
    const auto& boxes = tree.boxes();
    ErrorSlot error;

    // 1. Leaf aggregation.
    {
        const std::vector<Index>& leaves = tree.boxes_at_level(leaf);
        const std::size_t nd = sz(r.levels.back().sampling.size());
        const Complex* V = r.patterns->data().data();
        const auto nb = static_cast<Index>(leaves.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (Index ib = 0; ib < nb; ++ib) {
            const Box& box = boxes[sz(leaves[sz(ib)])];
            Complex* F = field(out.back(), ib, 0, nd);
            std::fill(F, F + kFields * nd, Complex(0.0, 0.0));
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                const Index basis = perm[sz(p)];
                const Complex xj = x(basis);
                const Complex xm = x(n + basis);
                const Complex* v = V + sz(p) * nd * 2;
                for (std::size_t q = 0; q < nd; ++q) {
                    F[q] += xj * v[2 * q];
                    F[nd + q] += xj * v[2 * q + 1];
                    F[2 * nd + q] += xm * v[2 * q];
                    F[3 * nd + q] += xm * v[2 * q + 1];
                }
            }
        }
    }
    // 2. Upward pass: children on level l + 1 -> parents on level l.
    for (int l = leaf - 1; l >= r.first_level; --l) {
        const auto li = static_cast<std::size_t>(l - r.first_level);
        const Level& child = r.levels[li + 1];
        const std::size_t ndp = sz(r.levels[li].sampling.size());
        const std::size_t ndc = sz(child.sampling.size());
        const std::vector<Index>& parents = tree.boxes_at_level(l);
        const auto nb = static_cast<Index>(parents.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (Index ip = 0; ip < nb; ++ip) {
            try {
                Complex* tmp = thread_scratch();
                const std::span<Complex> ws(tmp + 2 * max_nd, max_ws);
                const Box& P = boxes[sz(parents[sz(ip)])];
                Complex* Fp = field(out[li], ip, 0, ndp);
                std::fill(Fp, Fp + kFields * ndp, Complex(0.0, 0.0));
                for (std::size_t s = 0; s < 8; ++s) {
                    const Index c = P.children[s];
                    if (c < 0)
                        continue;
                    for (std::size_t f = 0; f < kFields; ++f) {
                        const Complex* src = field(out[li + 1], local[sz(c)], f, ndc);
                        if (child.interp) {
                            child.interp->interpolate(std::span<const Complex>(src, ndc),
                                                      std::span<Complex>(tmp, ndp), ws,
                                                      PoleParity::odd);
                            src = tmp;
                        }
                        Complex* dst = field(out[li], ip, f, ndp);
                        const Complex* shift = child.up[s].data();
                        for (std::size_t q = 0; q < ndp; ++q) dst[q] += shift[q] * src[q];
                    }
                }
            } catch (...) {
                error.capture();
            }
        }
        error.rethrow();
    }
    // 3. Translation on every level.
    for (std::size_t li = 0; li < nl; ++li) {
        const Level& lv = r.levels[li];
        const std::size_t nd = sz(lv.sampling.size());
        const std::vector<Index>& obs = tree.boxes_at_level(lv.info.level);
        const auto nb = static_cast<Index>(obs.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (Index ia = 0; ia < nb; ++ia) {
            const Box& A = boxes[sz(obs[sz(ia)])];
            Complex* G = field(in[li], ia, 0, nd);
            std::fill(G, G + kFields * nd, Complex(0.0, 0.0));
            for (const Index b : A.interaction_list) {
                const Box& B = boxes[sz(b)];
                const Complex* T = lv.translators[sz(lv.slot[offset_slot(A.ijk, B.ijk)])].data();
                const Complex* F = field(out[li], local[sz(b)], 0, nd);
                for (std::size_t f = 0; f < kFields; ++f) {
                    for (std::size_t q = 0; q < nd; ++q) G[f * nd + q] += T[q] * F[f * nd + q];
                }
            }
        }
    }
    // 4. Downward pass: parents on level l -> children on level l + 1.
    for (int l = r.first_level; l < leaf; ++l) {
        const auto li = static_cast<std::size_t>(l - r.first_level);
        const Level& child = r.levels[li + 1];
        const std::size_t ndp = sz(r.levels[li].sampling.size());
        const std::size_t ndc = sz(child.sampling.size());
        const std::vector<Index>& children = tree.boxes_at_level(l + 1);
        const auto nb = static_cast<Index>(children.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (Index ic = 0; ic < nb; ++ic) {
            try {
                Complex* tmp = thread_scratch();
                Complex* tmp2 = tmp + max_nd;
                const std::span<Complex> ws(tmp + 2 * max_nd, max_ws);
                const Box& C = boxes[sz(children[sz(ic)])];
                const std::size_t s =
                    sz((C.ijk[0] & 1) | (C.ijk[1] & 1) << 1 | (C.ijk[2] & 1) << 2);
                const Complex* shift = child.down[s].data();
                for (std::size_t f = 0; f < kFields; ++f) {
                    const Complex* gp = field(in[li], local[sz(C.parent)], f, ndp);
                    for (std::size_t q = 0; q < ndp; ++q) tmp[q] = shift[q] * gp[q];
                    const Complex* src = tmp;
                    if (child.interp) {
                        child.interp->anterpolate(std::span<const Complex>(tmp, ndp),
                                                  std::span<Complex>(tmp2, ndc), ws,
                                                  PoleParity::odd);
                        src = tmp2;
                    }
                    Complex* dst = field(in[li + 1], ic, f, ndc);
                    for (std::size_t q = 0; q < ndc; ++q) dst[q] += src[q];
                }
            } catch (...) {
                error.capture();
            }
        }
        error.rethrow();
    }
    // 5. Leaf reception (unweighted: the weights are in the incoming fields).
    {
        const std::vector<Index>& leaves = tree.boxes_at_level(leaf);
        const std::size_t nd = sz(r.levels.back().sampling.size());
        const Complex* V = r.patterns->data().data();
        const auto nb = static_cast<Index>(leaves.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (Index ib = 0; ib < nb; ++ib) {
            const Box& box = boxes[sz(leaves[sz(ib)])];
            const Complex* G = field(in.back(), ib, 0, nd);
            Complex* u = thread_scratch();
            // R_p(khat_q) = (V_theta, -V_phi)(p, q') with q' = antipode(q) (patterns.hpp), so
            // sum_q R_p(q) . U(q) = sum_q' V_p(q') . U~(q'), U~(q') = (U_theta,
            // -U_phi)(antipode(q')): u holds U~ in the order (J theta, J phi, M theta, M phi),
            // interleaved per q'.
            for (std::size_t q = 0; q < nd; ++q) {
                const std::size_t a = r.antipode[q];
                const Complex jt = G[a], jp = G[nd + a], mt = G[2 * nd + a], mp = G[3 * nd + a];
                u[4 * q] = r.alpha * jt + r.beta * mp;
                u[4 * q + 1] = -(r.alpha * jp - r.beta * mt);
                u[4 * q + 2] = -r.gamma * jp + r.delta * mt;
                u[4 * q + 3] = -(r.gamma * jt + r.delta * mp);
            }
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                Complex sj(0.0, 0.0), sm(0.0, 0.0);
                const Complex* v = V + sz(p) * nd * 2;
                for (std::size_t q = 0; q < nd; ++q) {
                    sj += v[2 * q] * u[4 * q] + v[2 * q + 1] * u[4 * q + 1];
                    sm += v[2 * q] * u[4 * q + 2] + v[2 * q + 1] * u[4 * q + 3];
                }
                const Index basis = perm[sz(p)];
                y(basis) += sj;
                y(n + basis) += sm;
            }
        }
    }
}

std::string MlfmmFarOperator::describe() const {
    const Impl& m = *impl_;
    std::ostringstream os;
    os << "MlfmmFarOperator: 2N = " << rows() << ", d0 = " << m.digits
       << ", interpolation order p = " << m.interp_order << ", octree levels " << m.tree.levels()
       << " (far levels 2 ... " << m.tree.leaf_level() << "); policy: expansion if the search "
       << "is achievable (lossy regions: and block check <= " << block_check_tolerance(m.digits)
       << " or <= " << kLossDegradationFactor
       << " x the lossless analogue), else per box pair truncation if delta <= "
       << std::pow(10.0, -(m.digits + 1.0)) << ", exact otherwise";
    for (std::size_t i = 0; i < 2; ++i) {
        const Region& r = m.region[i];
        os << "\n  region R" << i + 1 << ": ";
        if (!r.active) {
            os << "inactive (zero weights or no interaction lists)";
            continue;
        }
        const Real lambda = 2.0 * constants::pi / r.k.real();
        os << "k = " << r.k << " 1/m";
        if (r.patterns) {
            os << ", leaf patterns "
               << static_cast<Real>(r.patterns->data().size() * sizeof(Complex)) / 1048576.0
               << " MB (quadrature degree <= " << r.patterns->max_quad_degree() << ", "
               << r.pattern_seconds << " s)";
        }
        if (r.exact) {
            os << ", exact part nnz " << r.exact->nonzeros() << " ("
               << static_cast<Real>(r.exact->memory_bytes()) / 1048576.0 << " MB, "
               << r.exact_seconds << " s; " << r.truncated_basis_pairs
               << " basis pairs of exact box pairs truncated, max delta " << r.truncated_basis_bound
               << ")";
        }
        for (const FarLevelInfo& f : r.info) {
            os << "\n    level " << f.level << ": " << f.boxes
               << " boxes, a = " << f.box_size / lambda << " lambda_i, " << to_string(f.decision)
               << "; search " << (f.search_achievable ? "achievable" : "not achievable/not run")
               << " (error " << f.search_error << ", L = " << f.truncation_order
               << "), block check ";
            if (f.block_error < 0.0)
                os << "not run";
            else
                os << f.block_error << " (" << f.block_pairs << " pairs)";
            if (f.reference_error >= 0.0)
                os << ", lossless analogue " << f.reference_error;
            if (f.decision == FarDecision::expansion) {
                os << ", sampling L = " << f.sampling_order << " (" << f.directions
                   << " directions), " << f.translators << " translators, fields "
                   << static_cast<Real>(f.pattern_bytes) / 1048576.0 << " MB per apply";
            } else {
                os << "; box pairs: " << f.truncated_pairs << " truncated (max delta "
                   << f.decay_bound << "), " << f.exact_pairs << " exact (min delta "
                   << f.exact_bound << ")";
            }
            os << ", setup " << f.setup_seconds << " s";
        }
    }
    const std::size_t w = workspaces();
    os << "\n  memory " << static_cast<Real>(memory_bytes()) / 1048576.0 << " MB incl. "
       << std::max<std::size_t>(w, 1) << " apply workspace(s) of "
       << static_cast<Real>(workspace_bytes()) / 1048576.0 << " MB (" << w
       << " allocated so far; at least one is counted)";
    return os.str();
}

std::size_t MlfmmFarOperator::memory_bytes() const {
    std::size_t b =
        sizeof(Impl) + impl_->local.size() * sizeof(Index) + impl_->extent.size() * sizeof(Extent);
    for (const Region& r : impl_->region) {
        if (r.exact)
            b += r.exact->memory_bytes();
        if (r.patterns)
            b += r.patterns->data().size() * sizeof(Complex);
        for (std::size_t li = 0; li < r.levels.size(); ++li) {
            const Level& lv = r.levels[li];
            b += lv.translators.size() * sz(lv.sampling.size()) * sizeof(Complex);
            if (li > 0)  // phase shifts at the parent directions
                b += 16 * sz(r.levels[li - 1].sampling.size()) * sizeof(Complex);
            if (lv.interp)
                b += lv.interp->memory_bytes();
        }
    }
    // Apply workspaces: the pooled ones, at least the one a serial solve needs.
    return b + std::max<std::size_t>(workspaces(), 1) * workspace_bytes();
}

}  // namespace specklebem::mlfmm
