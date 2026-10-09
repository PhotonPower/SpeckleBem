/// @file far_operator.cpp
/// Multilevel FMM far operator (WP20a); the passes, signs and weight conventions are derived in
/// the header comment of far_operator.hpp.
#include "specklebem/compression/mlfmm/far_operator.hpp"

#include "specklebem/compression/mlfmm/interpolation.hpp"
#include "specklebem/compression/mlfmm/plane_wave.hpp"
#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
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

struct Region {
    bool active = false;
    Complex k;
    Complex alpha, beta, gamma, delta;  ///< e c_L, e c_K, h c_K, m c_L
    std::vector<Level> levels;          ///< levels 2 ... leaf
    std::vector<FarLevelInfo> info;
    std::unique_ptr<RadiationPatterns> patterns;
    std::vector<std::size_t> antipode;  ///< leaf directions: index of -khat_q
    Real pattern_seconds = 0;
};

}  // namespace

struct MlfmmFarOperator::Impl {
    explicit Impl(const Octree& t) : tree(t) {}
    const Octree& tree;
    Index n = 0;
    Real digits = 0;
    int interp_order = 0;
    std::array<Region, 2> region;
    std::vector<Index> local;  ///< box -> position within its level

    /// Truncation search and samplings of every level (throws before any expensive setup).
    void plan_region(Region& r, const basis::RwgSpace& space, int index);
    /// Translators, interpolators, phase shifts and leaf patterns.
    void build_region(Region& r, const basis::RwgSpace& space, const PatternOptions& popt);
    void apply_region(const Region& r, const VectorXc& x, VectorXc& y) const;
};

void MlfmmFarOperator::Impl::plan_region(Region& r, const basis::RwgSpace& space, int index) {
    const int leaf = tree.leaf_level();
    const Real rmax = max_support_radius(space);
    const Real lambda = 2.0 * constants::pi / r.k.real();
    for (int l = 2; l <= leaf; ++l) {
        const auto t0 = std::chrono::steady_clock::now();
        Level lv;
        const Real a = tree.box_size(l);
        const std::vector<Index>& boxes = tree.boxes_at_level(l);
        TruncationSearch search;
        int sampling_order = 0;
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
        if (!search.achievable) {
            std::ostringstream os;
            os << "MlfmmFarOperator: region R" << index + 1 << " (k = " << r.k << " 1/m), level "
               << l << " (box edge " << a / lambda << " lambda): no truncation order meets 10^-"
               << digits << " (best " << search.error << " at L = " << search.order
               << "); the lossy-region policy (ADR 0008 §6) is not implemented (WP21)";
            throw std::runtime_error(os.str());
        }
        lv.sampling = SphereSampling(sampling_order);
        lv.info = {l,
                   static_cast<Index>(boxes.size()),
                   a,
                   search.order,
                   sampling_order,
                   lv.sampling.size(),
                   search.error,
                   0,
                   2 * kFields * sizeof(Complex) * boxes.size() * sz(lv.sampling.size()),
                   seconds_since(t0)};
        r.levels.push_back(std::move(lv));
    }
}

void MlfmmFarOperator::Impl::build_region(Region& r, const basis::RwgSpace& space,
                                          const PatternOptions& popt) {
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
                const Vec3 d = 0.5 * a *
                               Vec3((s & 1U) ? 1.0 : -1.0, (s & 2U) ? 1.0 : -1.0,
                                    (s & 4U) ? 1.0 : -1.0);
                const VectorXr phase = khat * d;
                lv.up[s] = (jk * phase.cast<Complex>()).array().exp();
                lv.down[s] = (-jk * phase.cast<Complex>()).array().exp();
            }
        }
        lv.info.setup_seconds += seconds_since(t0);
        r.info.push_back(lv.info);
    }
    const auto t0 = std::chrono::steady_clock::now();
    r.patterns = std::make_unique<RadiationPatterns>(space, tree, r.k, r.levels.back().sampling, popt);
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
    PatternOptions popt;
    popt.target_accuracy = 1e-2 * std::pow(10.0, -params.accuracy_digits);
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
        if (r.active)
            m.plan_region(r, space, static_cast<int>(i));
    }
    for (Region& r : m.region) {
        if (r.active)
            m.build_region(r, space, popt);
    }
    SBEM_INFO("MlfmmFarOperator: 2N = {}, {} octree levels, d0 = {}, {:.1f} MB stored", rows(),
              tree.levels(), m.digits, static_cast<Real>(memory_bytes()) / 1048576.0);
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
    if (!region_active(region))
        throw std::invalid_argument("MlfmmFarOperator: region inactive or no far levels");
    return *impl_->region[static_cast<std::size_t>(region)].patterns;
}

void MlfmmFarOperator::apply(const VectorXc& x, VectorXc& y) const {
    if (x.size() != cols())
        throw std::invalid_argument("MlfmmFarOperator::apply: x has the wrong size");
    VectorXc out = VectorXc::Zero(rows());
    for (const Region& r : impl_->region) {
        if (r.active)
            impl_->apply_region(r, x, out);
    }
    y = std::move(out);
}

void MlfmmFarOperator::Impl::apply_region(const Region& r, const VectorXc& x,
                                          VectorXc& y) const {
    const int leaf = tree.leaf_level();
    const std::size_t nl = r.levels.size();
    std::vector<std::vector<Complex>> out(nl), in(nl);
    std::size_t max_nd = 0, max_ws = 0;
    for (std::size_t li = 0; li < nl; ++li) {
        const Level& lv = r.levels[li];
        const std::size_t size = sz(lv.info.boxes) * kFields * sz(lv.sampling.size());
        out[li].assign(size, Complex(0.0, 0.0));
        in[li].assign(size, Complex(0.0, 0.0));
        max_nd = std::max(max_nd, sz(lv.sampling.size()));
        if (lv.interp)
            max_ws = std::max(max_ws, sz(lv.interp->workspace_size()));
    }
    const auto nthreads = static_cast<std::size_t>(std::max(max_threads(), 1));
    // Per-thread scratch: two direction buffers (or the 4 reception fields) + workspace.
    const std::size_t stride = 4 * max_nd + max_ws;
    std::vector<Complex> scratch(nthreads * stride);
    const auto thread_scratch = [&]() { return scratch.data() + thread_id() * stride; };
    const auto field = [](auto& v, Index pos, std::size_t f, std::size_t nd) {
        return v.data() + (sz(pos) * kFields + f) * nd;
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
    for (int l = leaf - 1; l >= 2; --l) {
        const auto li = static_cast<std::size_t>(l - 2);
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
            for (const Index b : A.interaction_list) {
                const Box& B = boxes[sz(b)];
                const Complex* T =
                    lv.translators[sz(lv.slot[offset_slot(A.ijk, B.ijk)])].data();
                const Complex* F = field(out[li], local[sz(b)], 0, nd);
                for (std::size_t f = 0; f < kFields; ++f) {
                    for (std::size_t q = 0; q < nd; ++q) G[f * nd + q] += T[q] * F[f * nd + q];
                }
            }
        }
    }
    // 4. Downward pass: parents on level l -> children on level l + 1.
    for (int l = 2; l < leaf; ++l) {
        const auto li = static_cast<std::size_t>(l - 2);
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
                const std::size_t s = sz((C.ijk[0] & 1) | (C.ijk[1] & 1) << 1 | (C.ijk[2] & 1) << 2);
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
            // sum_q R_p(q) . U(q) = sum_q' V_p(q') . U~(q'), U~(q') = (U_theta, -U_phi)(antipode(q')):
            // u holds U~ in the order (J theta, J phi, M theta, M phi), interleaved per q'.
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
       << " (far levels 2 ... " << m.tree.leaf_level() << ")";
    std::size_t apply_bytes = 0;
    for (std::size_t i = 0; i < 2; ++i) {
        const Region& r = m.region[i];
        os << "\n  region R" << i + 1 << ": ";
        if (!r.active) {
            os << "inactive (zero weights or no interaction lists)";
            continue;
        }
        const Real lambda = 2.0 * constants::pi / r.k.real();
        os << "k = " << r.k << " 1/m, leaf patterns "
           << static_cast<Real>(r.patterns->data().size() * sizeof(Complex)) / 1048576.0
           << " MB (quadrature degree <= " << r.patterns->max_quad_degree() << ", "
           << r.pattern_seconds << " s)";
        for (const FarLevelInfo& f : r.info) {
            apply_bytes += f.pattern_bytes;
            os << "\n    level " << f.level << ": " << f.boxes << " boxes, a = "
               << f.box_size / lambda << " lambda, L = " << f.truncation_order
               << ", sampling L = " << f.sampling_order << " (" << f.directions
               << " directions), search error " << f.search_error << ", " << f.translators
               << " translators, fields " << static_cast<Real>(f.pattern_bytes) / 1048576.0
               << " MB per apply, setup " << f.setup_seconds << " s";
        }
    }
    os << "\n  stored " << static_cast<Real>(memory_bytes()) / 1048576.0
       << " MB, per-apply fields " << static_cast<Real>(apply_bytes) / 1048576.0 << " MB";
    return os.str();
}

std::size_t MlfmmFarOperator::memory_bytes() const {
    std::size_t b = sizeof(Impl) + impl_->local.size() * sizeof(Index);
    for (const Region& r : impl_->region) {
        if (!r.active)
            continue;
        b += r.patterns->data().size() * sizeof(Complex);
        for (std::size_t li = 0; li < r.levels.size(); ++li) {
            const Level& lv = r.levels[li];
            b += lv.translators.size() * sz(lv.sampling.size()) * sizeof(Complex);
            if (li > 0)  // phase shifts at the parent directions
                b += 16 * sz(r.levels[li - 1].sampling.size()) * sizeof(Complex);
        }
    }
    return b;
}

}  // namespace specklebem::mlfmm
