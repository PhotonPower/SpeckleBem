/// @file pml_ring.cpp
/// Exact PML ring of the fast PML route (WP-PML5a): classification, near-pattern split, size
/// estimate, near ring (op::assemble_sparse_stretched) and exact far ring blocks per row leaf.
/// Layout and symmetry argument: pml_ring.hpp.
#include "specklebem/compression/mlfmm/pml_ring.hpp"

#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "pml_ring_far.hpp"

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

namespace specklebem::mlfmm {

namespace {

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

Real mb(std::size_t bytes) {
    return static_cast<Real>(bytes) / 1048576.0;
}

/// First exception thrown inside an OpenMP loop body (exceptions must not leave the region).
class ErrorSlot {
public:
    void capture() {
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp critical(specklebem_pml_ring_exception)
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

/// Leaf structure of an octree, by leaf index g (position in boxes_at_level(leaf), Morton order;
/// the leaves' element ranges are ascending in g).
struct Leaves {
    std::vector<Index> first, count;        ///< element range [first, first + count)
    std::vector<std::array<Index, 3>> ijk;  ///< integer coordinates
    std::vector<std::vector<Index>> near;   ///< the leaf and its near_list, ascending
    std::vector<Index> leaf_of_pos;         ///< permuted position -> leaf index
};

/// @throws std::invalid_argument for a tree not built on `space` or with elevated functions.
Leaves leaves_of(const Octree& tree, const basis::RwgSpace& space, const char* who) {
    const Index N = space.size();
    if (static_cast<Index>(tree.permutation().size()) != N) {
        throw std::invalid_argument(std::string(who) + ": the octree holds " +
                                    std::to_string(tree.permutation().size()) +
                                    " basis functions, the space " + std::to_string(N));
    }
    if (tree.has_elevated()) {
        throw std::invalid_argument(
            std::string(who) +
            ": the PML ring needs every basis function on the leaf level (no local leaf rule, "
            "MlfmmParams::leaf_radius_quantile = 1); WP-PML5a defines near / far by leaf "
            "adjacency");
    }
    const auto& boxes = tree.boxes();
    const std::vector<Index>& lv = tree.boxes_at_level(tree.leaf_level());
    Leaves L;
    const std::size_t nl = lv.size();
    L.first.resize(nl);
    L.count.resize(nl);
    L.ijk.resize(nl);
    L.near.resize(nl);
    std::vector<Index> leaf_index(boxes.size(), -1);
    for (std::size_t g = 0; g < nl; ++g) leaf_index[sz(lv[g])] = static_cast<Index>(g);
    L.leaf_of_pos.assign(sz(N), -1);
    for (std::size_t g = 0; g < nl; ++g) {
        const Box& b = boxes[sz(lv[g])];
        L.first[g] = b.first_element;
        L.count[g] = b.num_elements;
        L.ijk[g] = b.ijk;
        L.near[g].push_back(static_cast<Index>(g));
        for (const Index nb : b.near_list) L.near[g].push_back(leaf_index[sz(nb)]);
        std::sort(L.near[g].begin(), L.near[g].end());
        for (Index p = b.first_element; p < b.first_element + b.num_elements; ++p)
            L.leaf_of_pos[sz(p)] = static_cast<Index>(g);
    }
    return L;
}

bool leaves_near(const Leaves& L, Index a, Index b) {
    const auto& u = L.ijk[sz(a)];
    const auto& v = L.ijk[sz(b)];
    return std::abs(u[0] - v[0]) <= 1 && std::abs(u[1] - v[1]) <= 1 && std::abs(u[2] - v[2]) <= 1;
}

void check_classification(const std::vector<std::uint8_t>& stretched, Index N, const char* who) {
    if (static_cast<Index>(stretched.size()) != N) {
        throw std::invalid_argument(std::string(who) + ": the classification has " +
                                    std::to_string(stretched.size()) +
                                    " entries, N = " + std::to_string(N));
    }
}

/// Bytes of a SparseOperator from assemble_sparse with `pairs` pattern pairs.
std::size_t sparse_bytes(Index pairs, Index N) {
    return 4 * sz(pairs) * (sizeof(Complex) + sizeof(Index)) + (2 * sz(N) + 1) * sizeof(Index);
}

/// Per row leaf bytes besides the blocks: rows and near ranges.
std::size_t row_leaf_overhead(Index ns, std::size_t near_count) {
    return sz(ns) * sizeof(Index) + near_count * 2 * sizeof(Index) + 64;
}

}  // namespace

std::vector<std::uint8_t> classify_stretched(const basis::RwgSpace& space,
                                             const geometry::CoordinateStretch& stretch) {
    return op::pml_masked_functions(space, stretch);
}

SplitNearPattern split_near_pattern(const Octree& tree, const basis::RwgSpace& space,
                                    const std::vector<std::uint8_t>& stretched) {
    const Index N = space.size();
    check_classification(stretched, N, "split_near_pattern");
    const op::BasisPattern base = near_pattern(tree, space);
    const std::size_t G = base.col_ptr.size() - 1;
    const auto is_s = [&](Index n) { return stretched[sz(n)] != 0; };
    SplitNearPattern out;
    op::BasisPattern& real = out.real;
    op::BasisPattern& ring = out.ring;
    real.group_of_row.assign(sz(N), -1);
    ring.group_of_row.assign(sz(N), -1);
    real.col_ptr.reserve(G + 2);
    ring.col_ptr.reserve(2 * G + 1);
    real.col_ptr.push_back(0);
    ring.col_ptr.push_back(0);
    // Rows per base group: real / stretched counts.
    std::vector<Index> nr(G, 0), ns(G, 0);
    for (Index m = 0; m < N; ++m) {
        const auto g = sz(base.group_of_row[sz(m)]);
        (is_s(m) ? ns : nr)[g] += 1;
    }
    for (std::size_t g = 0; g < G; ++g) {
        const Index* cb = base.cols.data() + base.col_ptr[g];
        const Index* ce = base.cols.data() + base.col_ptr[g + 1];
        Index cs = 0;  // stretched columns of the group
        // real group g: the real columns
        for (const Index* c = cb; c != ce; ++c) {
            if (is_s(*c))
                ++cs;
            else
                real.cols.push_back(*c);
        }
        real.col_ptr.push_back(static_cast<Index>(real.cols.size()));
        // ring group 2g: real rows x stretched columns; 2g + 1: stretched rows x all columns
        for (const Index* c = cb; c != ce; ++c) {
            if (is_s(*c))
                ring.cols.push_back(*c);
        }
        ring.col_ptr.push_back(static_cast<Index>(ring.cols.size()));
        ring.cols.insert(ring.cols.end(), cb, ce);
        ring.col_ptr.push_back(static_cast<Index>(ring.cols.size()));
        const Index ct = ce - cb;
        out.real_pairs += nr[g] * (ct - cs);
        out.ring_pairs += nr[g] * cs + ns[g] * ct;
    }
    const auto empty = static_cast<Index>(G);  // real pattern: one empty group for stretched rows
    real.col_ptr.push_back(static_cast<Index>(real.cols.size()));
    for (Index m = 0; m < N; ++m) {
        const Index g = base.group_of_row[sz(m)];
        if (is_s(m)) {
            real.group_of_row[sz(m)] = empty;
            ring.group_of_row[sz(m)] = 2 * g + 1;
        } else {
            real.group_of_row[sz(m)] = g;
            ring.group_of_row[sz(m)] = 2 * g;
        }
    }
    return out;
}

namespace {

void check_aca_params(const PmlRingParams& p) {
    if (!p.use_aca)
        return;
    if (!(p.eta > 0.0 && std::isfinite(p.eta)))
        throw std::invalid_argument("PmlRingParams::eta must be positive and finite");
    if (!(p.aca_epsilon >= 0.0 && p.aca_epsilon < 1.0))
        throw std::invalid_argument("PmlRingParams::aca_epsilon must lie in [0, 1) (0: automatic)");
    if (p.max_rank < 0)
        throw std::invalid_argument("PmlRingParams::max_rank must be >= 0 (0: automatic)");
}

/// PmlRingEstimate::far_memo_bytes: the pair memo budget per assembling thread (WP-PML5d).
std::size_t far_memo_bound(const PmlRingParams& p, std::size_t row_clusters) {
    std::size_t threads = 1;
#ifdef SPECKLEBEM_HAVE_OPENMP
    threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
    const std::size_t n = std::min(threads, row_clusters);
    return n > 0 && p.pair_memo_bytes > std::numeric_limits<std::size_t>::max() / n
               ? std::numeric_limits<std::size_t>::max()
               : p.pair_memo_bytes * n;
}

/// estimate_pml_ring; with use_aca also the far partition (moved into *partition if given).
PmlRingEstimate estimate_impl(const Octree& tree, const basis::RwgSpace& space,
                              const std::vector<std::uint8_t>& stretched,
                              const PmlRingParams& params,
                              std::vector<detail::FarRowLeaf>* partition);

}  // namespace

PmlRingEstimate estimate_pml_ring(const Octree& tree, const basis::RwgSpace& space,
                                  const std::vector<std::uint8_t>& stretched,
                                  const PmlRingParams& params) {
    return estimate_impl(tree, space, stretched, params, nullptr);
}

std::vector<PmlRingFarBlock> plan_pml_ring_far(const Octree& tree, const basis::RwgSpace& space,
                                               const std::vector<std::uint8_t>& stretched,
                                               const PmlRingParams& params) {
    std::vector<detail::FarRowLeaf> part;
    (void)estimate_impl(tree, space, stretched, params, &part);
    std::vector<PmlRingFarBlock> out;
    for (const detail::FarRowLeaf& R : part)
        out.insert(out.end(), R.blocks.begin(), R.blocks.end());
    return out;
}

namespace {

PmlRingEstimate estimate_impl(const Octree& tree, const basis::RwgSpace& space,
                              const std::vector<std::uint8_t>& stretched,
                              const PmlRingParams& params,
                              std::vector<detail::FarRowLeaf>* partition) {
    const Index N = space.size();
    check_classification(stretched, N, "estimate_pml_ring");
    check_aca_params(params);
    const Leaves L = leaves_of(tree, space, "estimate_pml_ring");
    const std::vector<Index>& perm = tree.permutation();
    const std::size_t nl = L.first.size();
    std::vector<Index> ns(nl, 0);
    for (std::size_t g = 0; g < nl; ++g) {
        for (Index p = L.first[g]; p < L.first[g] + L.count[g]; ++p)
            ns[g] += stretched[sz(perm[sz(p)])] != 0 ? 1 : 0;
    }
    PmlRingEstimate e;
    e.n = N;
    for (std::size_t g = 0; g < nl; ++g) e.stretched += ns[g];
    const Index n_real = N - e.stretched;
    std::size_t far_bytes = 0;
    for (std::size_t g = 0; g < nl; ++g) {
        Index near_total = 0, near_s = 0;
        for (const Index h : L.near[g]) {
            near_total += L.count[sz(h)];
            near_s += ns[sz(h)];
        }
        const Index nr = L.count[g] - ns[g];
        e.near_pairs += ns[g] * near_total + nr * near_s;
        e.rr_near_pairs += nr * (near_total - near_s);
        if (ns[g] == 0)
            continue;
        ++e.row_leaves;
        const Index nf = N - near_total;
        e.far_stored += ns[g] * nf;
        e.far_pairs += ns[g] * nf + ns[g] * (n_real - (near_total - near_s));
        far_bytes +=
            4 * sz(ns[g]) * sz(nf) * sizeof(Complex) + row_leaf_overhead(ns[g], L.near[g].size());
    }
    e.near_bytes = sparse_bytes(e.near_pairs, N);
    e.far_bytes = far_bytes;
    e.far_exact_bytes = far_bytes;
    if (params.use_aca) {
        std::vector<detail::FarRowLeaf> part = detail::partition_far_ring(
            tree, space, stretched, true, params.eta, params.coarse_columns, params.row_clusters);
        e.far_exact_bytes = 0;
        for (const detail::FarRowLeaf& R : part) {
            const auto rows = static_cast<Index>(R.rows.size());
            e.far_exact_bytes += row_leaf_overhead(rows, 0);
            for (const PmlRingFarBlock& b : R.blocks) {
                if (b.admissible) {
                    ++e.far_aca_blocks;
                    e.far_aca_pairs += rows * b.count;
                } else {
                    ++e.far_exact_blocks;
                    e.far_exact_bytes += 4 * sz(rows) * sz(b.count) * sizeof(Complex);
                }
            }
        }
        e.far_memo_bytes = far_memo_bound(params, part.size());
        if (partition != nullptr)
            *partition = std::move(part);
    }
    e.dense_bytes = 16 * sz(2 * N) * sz(2 * N);
    e.budget = params.max_far_bytes > 0 ? params.max_far_bytes : e.dense_bytes;
    return e;
}

}  // namespace

void check_pml_ring_budget(const PmlRingEstimate& e, const PmlRingParams& params) {
    if (e.far_exact_bytes <= e.budget)
        return;
    std::ostringstream os;
    if (params.use_aca) {
        os << "PmlRingOperator: the exact (non-admissible) far ring blocks would need "
           << mb(e.far_exact_bytes) << " MB (" << e.far_exact_blocks
           << " far leaf blocks of stretched rows, 64 bytes per pair; " << e.far_aca_blocks
           << " admissible blocks go through ACA)";
    } else {
        os << "PmlRingOperator: the exact far ring blocks (WP-PML5a) would need "
           << mb(e.far_exact_bytes) << " MB (" << e.far_stored << " stored pairs of " << e.stretched
           << " stretched rows x far columns, 64 bytes each)";
    }
    os << ", more than the budget of " << mb(e.budget)
       << " MB (PmlRingParams::max_far_bytes, MlfmmParams::pml_ring"
       << (params.max_far_bytes > 0 ? ")" : "; automatic: the dense matrix)")
       << "; raise the budget, shrink the PML ring (W_pml, mesh)"
       << (params.use_aca ? ", raise PmlRingParams::eta" : ", use the ACA far ring (use_aca)")
       << " or use the dense operator";
    throw std::runtime_error(os.str());
}

PmlRingTreeFingerprint pml_ring_tree_fingerprint(const Octree& tree) {
    PmlRingTreeFingerprint f;
    f.levels = tree.levels();
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a 64
    const auto mix = [&h](std::uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (8 * b)) & 0xffU;
            h *= 1099511628211ULL;
        }
    };
    const auto mix_index = [&mix](Index v) { mix(static_cast<std::uint64_t>(v)); };
    for (const Index p : tree.permutation()) mix_index(p);
    if (f.levels > 0) {
        f.leaf_size = tree.box_size(tree.leaf_level());
        const std::vector<Index>& lv = tree.boxes_at_level(tree.leaf_level());
        f.leaves = static_cast<Index>(lv.size());
        for (Eigen::Index d = 0; d < 3; ++d)
            mix(std::bit_cast<std::uint64_t>(tree.root_center()(d)));
        for (const Index b : lv) {
            const Box& box = tree.boxes()[sz(b)];
            for (const Index c : box.ijk) mix_index(c);
            mix_index(box.first_element);
            mix_index(box.num_elements);
            mix_index(static_cast<Index>(box.near_list.size()));
            for (const Index nb : box.near_list) mix_index(nb);
        }
    }
    f.hash = h;
    return f;
}

PmlRingPlan plan_pml_ring(const Octree& tree, const basis::RwgSpace& space,
                          const geometry::CoordinateStretch& stretch, const PmlRingParams& params) {
    PmlRingPlan plan;
    plan.tree = pml_ring_tree_fingerprint(tree);
    plan.params = params;
    plan.stretched = classify_stretched(space, stretch);
    // The estimate and (with use_aca) the far partition in one pass (WP-PML5b + 5c).
    std::vector<detail::FarRowLeaf> part;
    plan.estimate = estimate_impl(tree, space, plan.stretched, params, &part);
    for (detail::FarRowLeaf& R : part)
        plan.far_blocks.insert(plan.far_blocks.end(), R.blocks.begin(), R.blocks.end());
    part.clear();
    plan.split = split_near_pattern(tree, space, plan.stretched);
    plan.estimate.near_assembly_bytes =
        op::estimate_sparse_stretched_cache(space, stretch, plan.split.ring).bytes;
    return plan;
}

std::size_t pml_ring_peak_bytes(const PmlRingEstimate& e) {
    // Saturating sums: an unlimited pair memo budget (far_memo_bytes = SIZE_MAX) must not wrap
    // the peak to a small value and silence warn_pml_ring_memory (WP-PML5d review W2).
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    const std::size_t far =
        e.far_exact_bytes > kMax - e.far_memo_bytes ? kMax : e.far_exact_bytes + e.far_memo_bytes;
    const std::size_t inner = std::max(e.near_assembly_bytes, far);
    return e.near_bytes > kMax - inner ? kMax : e.near_bytes + inner;
}

bool warn_pml_ring_memory(const PmlRingEstimate& e, std::size_t available) {
    const std::size_t peak = pml_ring_peak_bytes(e);
    if (available == 0 || peak <= available)
        return false;
    SBEM_WARN(
        "PmlRingOperator: the ring needs {:.1f} MB at its peak known before assembly (near ring "
        "{:.1f} MB plus the larger of its assembly cache {:.1f} MB and the far ring blocks stored "
        "exactly {:.1f} MB, {} stretched rows{}), more than the {:.1f} MB of physical memory "
        "available now; the run will swap or fail. Shrink the PML ring (W_pml, mesh), set "
        "MlfmmParams::pml_ring.max_far_bytes to refuse such runs, or use the dense operator",
        mb(peak), mb(e.near_bytes), mb(e.near_assembly_bytes), mb(e.far_exact_bytes), e.stretched,
        e.far_exact_bytes < e.far_bytes ? "; the ACA blocks come on top" : "", mb(available));
    return true;
}

namespace {

/// Far blocks of one leaf A with stretched functions (pml_ring.hpp): rows A_s, columns = the
/// positions of every leaf not near A, ascending; per column j the four sub-block columns
/// [JJ(:, j), JM(:, j), MJ(:, j), MM(:, j)] contiguously (4 ns entries).
struct RowLeaf {
    Index leaf = -1;
    std::vector<Index> rows;                     ///< stretched basis indices, ascending position
    std::vector<std::array<Index, 2>> near_pos;  ///< position ranges of the near leaves
    Index nf = 0;
    std::vector<Complex> w;

    /// Column of position p (p must not lie in a near range).
    [[nodiscard]] Index column(Index p) const {
        Index skipped = 0;
        for (const auto& r : near_pos) {
            if (r[1] <= p)
                skipped += r[1] - r[0];
            else
                break;
        }
        return p - skipped;
    }
};

}  // namespace

struct PmlRingOperator::Impl {
    Index n = 0;
    std::vector<std::uint8_t> stretched;  ///< by basis index
    std::vector<Index> perm;              ///< position -> basis
    std::vector<Index> inv_perm;          ///< basis -> position
    std::vector<std::uint8_t> stretched_pos;
    Leaves leaves;
    std::vector<RowLeaf> row_leaves;
    std::shared_ptr<op::SparseOperator> near;
    PmlRingEstimate est;
    std::size_t far_alloc = 0;
    Real near_s = 0, far_s = 0;
    std::unique_ptr<detail::AcaFarRing> aca;  ///< ACA far ring (use_aca); row_leaves empty then
    PmlRingFarStats exact_stats;              ///< far_stats() of the exact path
    op::SparseStretchedStats near_stats;

    void assemble_far(const op::Problem& p);
    /// Transposed-product chunks: row leaves [chunk_first[k], chunk_first[k + 1]) (apply_far).
    std::vector<Index> chunk_first;
    /// Fixed chunk count (not the thread count): the result is thread-count independent.
    static constexpr Index kTransposeChunks = 64;

    void partition_chunks();
    void apply_far(const VectorXc& x, VectorXc& y) const;
};

void PmlRingOperator::Impl::assemble_far(const op::Problem& p) {
    const basis::RwgSpace& space = *p.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index F = mesh.num_triangles();
    const op::PairBlockEvaluator eval(p);
    // Row slot of every stretched basis: (row leaf, row index).
    std::vector<Index> rl_of(sz(n), -1), row_of(sz(n), -1);
    for (std::size_t r = 0; r < row_leaves.size(); ++r) {
        const RowLeaf& R = row_leaves[r];
        for (std::size_t i = 0; i < R.rows.size(); ++i) {
            rl_of[sz(R.rows[i])] = static_cast<Index>(r);
            row_of[sz(R.rows[i])] = static_cast<Index>(i);
        }
    }
    const auto leaf_of = [&](Index basis) { return leaves.leaf_of_pos[sz(inv_perm[sz(basis)])]; };
    // Greedy colouring: triangles sharing a basis function get different colours (as the dense
    // assembler; shared with the ACA path); the test triangles are those carrying a stretched
    // function.
    const std::vector<int> colour = detail::greedy_triangle_colours(space);
    std::vector<std::vector<Index>> groups;
    for (Index t = 0; t < F; ++t) {
        const basis::RwgSpace::Support sup = space.support(t);
        const int c = colour[sz(t)];
        bool test = false;
        for (int a = 0; a < sup.count; ++a) test = test || stretched[sz(sup.n[a])] != 0;
        if (!test)
            continue;
        if (static_cast<std::size_t>(c) >= groups.size())
            groups.resize(static_cast<std::size_t>(c) + 1);
        groups[static_cast<std::size_t>(c)].push_back(t);
    }
    ErrorSlot error;
    for (const std::vector<Index>& group : groups) {
        const auto ng = static_cast<Index>(group.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (Index gi = 0; gi < ng; ++gi) {
            try {
                const Index t = group[sz(gi)];
                const basis::RwgSpace::Support st = space.support(t);
                std::array<Index, 3> lt{-1, -1, -1};  // leaf of each stretched test function
                for (int a = 0; a < st.count; ++a) {
                    if (stretched[sz(st.n[a])] != 0)
                        lt[static_cast<std::size_t>(a)] = leaf_of(st.n[a]);
                }
                op::TrianglePairBlocks blk;
                for (Index s = 0; s < F; ++s) {
                    const basis::RwgSpace::Support ss = space.support(s);
                    if (ss.count == 0)
                        continue;
                    std::array<Index, 3> ls{};
                    for (int b = 0; b < ss.count; ++b)
                        ls[static_cast<std::size_t>(b)] = leaf_of(ss.n[b]);
                    bool needed = false;
                    for (int a = 0; a < st.count && !needed; ++a) {
                        if (lt[static_cast<std::size_t>(a)] < 0)
                            continue;
                        for (int b = 0; b < ss.count && !needed; ++b)
                            needed = !leaves_near(leaves, lt[static_cast<std::size_t>(a)],
                                                  ls[static_cast<std::size_t>(b)]);
                    }
                    if (!needed)
                        continue;
                    eval.evaluate(t, s, blk);
                    for (int a = 0; a < st.count; ++a) {
                        const Index la = lt[static_cast<std::size_t>(a)];
                        if (la < 0)
                            continue;
                        const Index m = st.n[a];
                        RowLeaf& R = row_leaves[sz(rl_of[sz(m)])];
                        const auto ns = R.rows.size();
                        const std::size_t i = sz(row_of[sz(m)]);
                        for (int b = 0; b < ss.count; ++b) {
                            if (leaves_near(leaves, la, ls[static_cast<std::size_t>(b)]))
                                continue;
                            const std::size_t j = sz(R.column(inv_perm[sz(ss.n[b])]));
                            const auto k = static_cast<std::size_t>(3 * a + b);
                            Complex* col = R.w.data() + 4 * ns * j;
                            col[i] += blk.jj[k];
                            col[ns + i] += blk.jm[k];
                            col[2 * ns + i] += blk.mj[k];
                            col[3 * ns + i] += blk.mm[k];
                        }
                    }
                }
            } catch (...) {
                error.capture();
            }
        }
        error.rethrow();
    }
}

void PmlRingOperator::Impl::partition_chunks() {
    const auto nr = static_cast<Index>(row_leaves.size());
    const Index K = std::min<Index>(nr, kTransposeChunks);
    chunk_first.assign(sz(K) + 1, nr);
    if (K == 0)
        return;
    // Consecutive row leaves, balanced by stored entries (the apply cost); depends only on the
    // ring's structure, never on the thread count.
    Real total = 0;
    for (const RowLeaf& R : row_leaves) total += static_cast<Real>(R.w.size());
    chunk_first[0] = 0;
    Real acc = 0;
    Index k = 0;
    for (Index r = 0; r + 1 < nr && k + 1 < K; ++r) {
        acc += static_cast<Real>(row_leaves[sz(r)].w.size());
        if (acc >= total * static_cast<Real>(k + 1) / static_cast<Real>(K))
            chunk_first[sz(++k)] = r + 1;
    }
}

void PmlRingOperator::Impl::apply_far(const VectorXc& x, VectorXc& y) const {
    if (row_leaves.empty())
        return;
    const Index N = n;
    // x by permuted position, split into real and imaginary parts.
    std::vector<Real> xr(4 * sz(N));  // [Jr, Ji, Mr, Mi] per position
    for (Index p = 0; p < N; ++p) {
        const Index b = perm[sz(p)];
        const Complex cj = x(b), cm = x(N + b);
        xr[4 * sz(p)] = cj.real();
        xr[4 * sz(p) + 1] = cj.imag();
        xr[4 * sz(p) + 2] = cm.real();
        xr[4 * sz(p) + 3] = cm.imag();
    }
    // One pass over the stored blocks per chunk of row leaves: the direct products go to the
    // leaves' own (stretched) rows, the transposed products into the chunk's buffer (real
    // positions, [Jr, Ji, Mr, Mi]); the buffers are then summed in chunk order.
    const auto K = static_cast<Index>(chunk_first.size()) - 1;
    std::vector<Real> buf(sz(K) * 4 * sz(N), 0.0);
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Real> xs, acc;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (Index k = 0; k < K; ++k) {
            Real* u = buf.data() + sz(k) * 4 * sz(N);
            for (Index r = chunk_first[sz(k)]; r < chunk_first[sz(k) + 1]; ++r) {
                const RowLeaf& R = row_leaves[sz(r)];
                const std::size_t ns = R.rows.size();
                xs.assign(4 * ns, 0.0);  // [Jr, Ji, Mr, Mi] blocks of ns
                acc.assign(4 * ns, 0.0);
                for (std::size_t i = 0; i < ns; ++i) {
                    const Index b = R.rows[i];
                    xs[i] = x(b).real();
                    xs[ns + i] = x(b).imag();
                    xs[2 * ns + i] = x(N + b).real();
                    xs[3 * ns + i] = x(N + b).imag();
                }
                // The column runs between the near ranges (shared kernel with the ACA path's
                // exact blocks, detail::apply_exact_columns).
                std::size_t j = 0;
                Index p0 = 0;
                const auto run = [&](Index a, Index e) {
                    detail::apply_exact_columns(R.w.data(), ns, xs.data(), xr.data(), a, e, j,
                                                acc.data(), u, stretched_pos.data());
                    j += sz(e - a);
                };
                for (const auto& rg : R.near_pos) {
                    run(p0, rg[0]);
                    p0 = rg[1];
                }
                run(p0, N);
                for (std::size_t i = 0; i < ns; ++i) {
                    const Index b = R.rows[i];
                    y(b) += Complex(acc[i], acc[ns + i]);
                    y(N + b) += Complex(acc[2 * ns + i], acc[3 * ns + i]);
                }
            }
        }
        // Real rows: the chunk buffers in chunk order (implicit barrier above).
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp for schedule(static)
#endif
        for (Index p = 0; p < N; ++p) {
            if (stretched_pos[sz(p)] != 0)
                continue;
            Real s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (Index k = 0; k < K; ++k) {
                const Real* up = buf.data() + sz(k) * 4 * sz(N) + 4 * sz(p);
                s0 += up[0];
                s1 += up[1];
                s2 += up[2];
                s3 += up[3];
            }
            const Index b = perm[sz(p)];
            y(b) += Complex(s0, s1);
            y(N + b) += Complex(s2, s3);
        }
    }
}

namespace {

/// @throws std::invalid_argument for an invalid Problem, one without a non-identity stretch, or
///         accuracy_digits not positive and finite.
void check_ring_problem(const op::Problem& problem, Real accuracy_digits) {
    op::validate(problem);
    if (!(accuracy_digits > 0.0 && std::isfinite(accuracy_digits)))
        throw std::invalid_argument("PmlRingOperator: accuracy_digits must be positive");
    if (!op::has_stretch(problem)) {
        throw std::invalid_argument(
            "PmlRingOperator: the Problem has no (non-identity) PML stretch; without a stretch "
            "every function is real and the MLFMM needs no ring");
    }
}

PmlRingPlan checked_plan(const op::Problem& problem, const Octree& tree,
                         const PmlRingParams& params, Real accuracy_digits) {
    check_ring_problem(problem, accuracy_digits);
    (void)leaves_of(tree, *problem.space, "PmlRingOperator");  // tree checks first
    return plan_pml_ring(tree, *problem.space, *problem.stretch, params);
}

/// True iff the parameters that shape the far partition (and the estimate's far_exact_bytes /
/// far_aca_* fields) agree.
bool same_far_partition(const PmlRingParams& a, const PmlRingParams& b) {
    if (a.use_aca != b.use_aca)
        return false;
    if (!a.use_aca)
        return true;
    return a.eta == b.eta && a.coarse_columns == b.coarse_columns &&
           a.row_clusters == b.row_clusters;
}

}  // namespace

PmlRingOperator::PmlRingOperator(const op::Problem& problem, const Octree& tree,
                                 const PmlRingParams& params, Real accuracy_digits)
    : PmlRingOperator(problem, tree, checked_plan(problem, tree, params, accuracy_digits), params,
                      accuracy_digits) {}

PmlRingOperator::PmlRingOperator(const op::Problem& problem, const Octree& tree, PmlRingPlan plan,
                                 const PmlRingParams& params, Real accuracy_digits)
    : impl_(std::make_unique<Impl>()) {
    check_ring_problem(problem, accuracy_digits);
    Impl& m = *impl_;
    const basis::RwgSpace& space = *problem.space;
    m.n = space.size();
    m.leaves = leaves_of(tree, space, "PmlRingOperator");
    // The plan must belong to this problem (the classification is O(N), checked in full).
    if (plan.stretched != classify_stretched(space, *problem.stretch) || plan.estimate.n != m.n) {
        throw std::invalid_argument(
            "PmlRingOperator: the PmlRingPlan was not made for this problem (classification or "
            "size differs; use plan_pml_ring on the same tree, space and stretch)");
    }
    // ... and to this octree: the near-ring pattern must match the far ring's near ranges.
    if (plan.tree != pml_ring_tree_fingerprint(tree)) {
        const PmlRingTreeFingerprint f = pml_ring_tree_fingerprint(tree);
        throw std::invalid_argument(
            "PmlRingOperator: the PmlRingPlan was made on another octree (plan: " +
            std::to_string(plan.tree.levels) + " levels, " + std::to_string(plan.tree.leaves) +
            " leaves; this tree: " + std::to_string(f.levels) + " levels, " +
            std::to_string(f.leaves) +
            " leaves, or a different leaf size / permutation); use plan_pml_ring on the same tree, "
            "space and stretch");
    }
    // ... and to these far-ring parameters: the far partition and its estimate depend on them.
    if (!same_far_partition(plan.params, params)) {
        throw std::invalid_argument(
            "PmlRingOperator: the PmlRingPlan was made with other far-ring partition parameters "
            "(PmlRingParams::use_aca, eta, coarse_columns, row_clusters); use plan_pml_ring with "
            "the parameters of the operator");
    }
    check_aca_params(params);
    m.stretched = std::move(plan.stretched);
    m.perm = tree.permutation();
    m.inv_perm = tree.inverse_permutation();
    m.stretched_pos.assign(sz(m.n), 0);
    for (Index p = 0; p < m.n; ++p) m.stretched_pos[sz(p)] = m.stretched[sz(m.perm[sz(p)])];
    m.est = plan.estimate;
    m.est.budget = params.max_far_bytes > 0 ? params.max_far_bytes : m.est.dense_bytes;
    // The memo bound of this operator's parameters (a shared plan may come with another budget).
    if (params.use_aca) {
        std::size_t clusters = 0;  // runs of blocks with one row box (far_partition_from_blocks)
        for (std::size_t i = 0; i < plan.far_blocks.size(); ++i)
            clusters += i == 0 || plan.far_blocks[i].row_box != plan.far_blocks[i - 1].row_box;
        m.est.far_memo_bytes = far_memo_bound(params, clusters);
    }
    SBEM_INFO(
        "PmlRingOperator: N = {}, {} stretched ({:.1f} %), {} row leaves; near ring {} pairs "
        "({:.1f} MB; assembly cache <= {:.1f} MB), far ring {} stored pairs ({:.1f} MB if exact; "
        "ACA {}: {} admissible blocks ({} pairs), {} exact blocks ({:.1f} MB); budget {:.1f} MB), "
        "dense matrix {:.1f} MB",
        m.n, m.est.stretched,
        m.n > 0 ? 100.0 * static_cast<Real>(m.est.stretched) / static_cast<Real>(m.n) : 0.0,
        m.est.row_leaves, m.est.near_pairs, mb(m.est.near_bytes), mb(m.est.near_assembly_bytes),
        m.est.far_stored, mb(m.est.far_bytes), params.use_aca ? "on" : "off", m.est.far_aca_blocks,
        m.est.far_aca_pairs, m.est.far_exact_blocks, mb(m.est.far_exact_bytes), mb(m.est.budget),
        mb(m.est.dense_bytes));
    check_pml_ring_budget(m.est, params);
    // The ACA far partition of the plan (rows and extents rebuilt from the tree; checked).
    std::vector<detail::FarRowLeaf> partition;
    if (params.use_aca)
        partition =
            detail::far_partition_from_blocks(tree, space, m.stretched, std::move(plan.far_blocks));
    // Near ring (WP-PML5c: canonical-pair cache and translation classes).
    auto t0 = std::chrono::steady_clock::now();
    {
        if (!params.assemble_near) {  // diagnostics: the far ring alone (an empty near pattern)
            plan.split.ring.group_of_row.assign(sz(m.n), 0);
            plan.split.ring.col_ptr = {0, 0};
            plan.split.ring.cols.clear();
        }
        op::SparseStretchedOptions opt;
        opt.stats = &m.near_stats;
        m.near = op::assemble_sparse_stretched(problem, plan.split.ring, opt);
        plan.split = SplitNearPattern{};  // the patterns are no longer needed
    }
    m.near_s = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (params.use_aca) {
        // ACA far ring (WP-PML5b): tolerance 10^-(d0 + 1) unless overridden; the magnitude-bound
        // threshold stays 10^-(d0 + 1).
        const Real tau = std::pow(10.0, -(accuracy_digits + 1.0));
        const Real eps = params.aca_epsilon > 0.0 ? params.aca_epsilon : tau;
        m.aca = std::make_unique<detail::AcaFarRing>(
            problem, detail::greedy_triangle_colours(space), std::move(partition), m.perm,
            m.inv_perm, m.stretched_pos, params, eps, tau);
        m.far_s = seconds_since(t0);
        const PmlRingFarStats& st = m.aca->stats();
        SBEM_INFO(
            "PmlRingOperator: near ring {:.1f} MB in {:.2f} s, ACA far ring {:.1f} MB in {:.2f} s "
            "(exact estimate {:.1f} MB): {} ACA blocks ({:.1f} MB, ranks min / median / max {} / "
            "{} / {}, mean JJ {:.1f} JM {:.1f} MJ {:.1f} MM {:.1f}), {} exact + {} rank-cap "
            "blocks ({} without an ACA attempt; {:.1f} MB), {} dropped; eps {:.1e}, eta {}",
            mb(m.near->memory_bytes()), m.near_s, mb(m.aca->bytes()), m.far_s, mb(m.est.far_bytes),
            st.aca_blocks, mb(st.aca_bytes), st.rank_min, st.rank_median, st.rank_max,
            st.rank_mean[0], st.rank_mean[1], st.rank_mean[2], st.rank_mean[3], st.exact_blocks,
            st.fallback_blocks, st.skipped_attempts, mb(st.exact_bytes), st.dropped_blocks, eps,
            params.eta);
        // The budget checks only the blocks known to be exact before assembly; say so when the
        // stored blocks (exact + ACA factors) end above it (review suggestion; e.g. an explicit
        // max_rank above the break-even, or a budget between far_exact_bytes and the result).
        if (st.exact_bytes + st.aca_bytes > m.est.budget) {
            SBEM_WARN(
                "PmlRingOperator: the assembled far ring blocks take {:.1f} MB (exact {:.1f} MB + "
                "ACA factors {:.1f} MB), more than the far budget of {:.1f} MB "
                "(PmlRingParams::max_far_bytes; checked before assembly against the {:.1f} MB "
                "of blocks known to be exact)",
                mb(st.exact_bytes + st.aca_bytes), mb(st.exact_bytes), mb(st.aca_bytes),
                mb(m.est.budget), mb(m.est.far_exact_bytes));
        }
        return;
    }
    // Exact far ring blocks (WP-PML5a): rows, near ranges, allocation, assembly.
    const Leaves& L = m.leaves;
    for (std::size_t g = 0; g < L.first.size(); ++g) {
        RowLeaf R;
        R.leaf = static_cast<Index>(g);
        for (Index p = L.first[g]; p < L.first[g] + L.count[g]; ++p) {
            if (m.stretched_pos[sz(p)] != 0)
                R.rows.push_back(m.perm[sz(p)]);
        }
        if (R.rows.empty())
            continue;
        Index near_total = 0;
        for (const Index h : L.near[g]) {
            R.near_pos.push_back({L.first[sz(h)], L.first[sz(h)] + L.count[sz(h)]});
            near_total += L.count[sz(h)];
        }
        R.nf = m.n - near_total;
        m.row_leaves.push_back(std::move(R));
    }
    for (RowLeaf& R : m.row_leaves) {
        R.w.assign(4 * R.rows.size() * sz(R.nf), Complex(0.0, 0.0));
        m.far_alloc += R.w.size() * sizeof(Complex) +
                       row_leaf_overhead(static_cast<Index>(R.rows.size()), R.near_pos.size());
    }
    m.partition_chunks();
    m.assemble_far(problem);
    m.far_s = seconds_since(t0);
    m.exact_stats.exact_blocks = static_cast<Index>(m.row_leaves.size());
    m.exact_stats.exact_pairs = m.est.far_stored;
    for (const RowLeaf& R : m.row_leaves) m.exact_stats.exact_bytes += R.w.size() * sizeof(Complex);
    SBEM_INFO("PmlRingOperator: near ring {:.1f} MB in {:.2f} s, far ring {:.1f} MB in {:.2f} s",
              mb(m.near->memory_bytes()), m.near_s, mb(m.far_alloc), m.far_s);
}

PmlRingOperator::~PmlRingOperator() = default;

Index PmlRingOperator::rows() const {
    return 2 * impl_->n;
}

Index PmlRingOperator::cols() const {
    return 2 * impl_->n;
}

void PmlRingOperator::apply(const VectorXc& x, VectorXc& y) const {
    if (x.size() != cols())
        throw std::invalid_argument("PmlRingOperator::apply: x has the wrong size");
    if (&x == &y) {
        const VectorXc copy = x;
        apply(copy, y);
        return;
    }
    impl_->near->apply(x, y);
    if (impl_->aca)
        impl_->aca->apply(x, y);
    else
        impl_->apply_far(x, y);
}

std::string PmlRingOperator::describe() const {
    const Impl& m = *impl_;
    const PmlRingEstimate& e = m.est;
    std::ostringstream os;
    os << "PmlRingOperator (" << (m.aca ? "ACA far ring, WP-PML5b" : "exact PML ring, WP-PML5a")
       << "): N = " << m.n << ", " << e.stretched << " stretched / " << m.n - e.stretched
       << " real functions, " << e.row_leaves << " leaves with stretched functions; near ring "
       << e.near_pairs << " pairs, " << mb(near_bytes()) << " MB, assembled in " << m.near_s
       << " s; far ring " << e.far_pairs << " pairs (" << e.far_stored
       << " stored, the rest applied transposed), " << mb(far_bytes()) << " MB, assembled in "
       << m.far_s << " s (budget " << mb(e.budget) << " MB; exact far ring " << mb(e.far_bytes)
       << " MB; dense matrix " << mb(e.dense_bytes) << " MB)";
    if (m.aca) {
        const PmlRingFarStats& st = m.aca->stats();
        os << "; ACA (eps " << st.epsilon << ", eta " << st.eta << "): " << st.aca_blocks
           << " blocks, " << mb(st.aca_bytes) << " MB, " << st.aca_pairs
           << " stored pairs, ranks min / median / max " << st.rank_min << " / " << st.rank_median
           << " / " << st.rank_max << ", mean per sub-block JJ " << st.rank_mean[0] << " JM "
           << st.rank_mean[1] << " MJ " << st.rank_mean[2] << " MM " << st.rank_mean[3]
           << ", histogram [0, 1, 2-3, 4-7, ...]:";
        for (const Index h : st.rank_histogram) os << " " << h;
        os << "; exact blocks " << st.exact_blocks << " + rank-cap fallbacks " << st.fallback_blocks
           << " (" << st.skipped_attempts << " without an ACA attempt), " << mb(st.exact_bytes)
           << " MB, " << st.exact_pairs
           << " pairs; dropped (magnitude bound <= " << st.drop_tolerance << ") "
           << st.dropped_blocks << " blocks, " << st.dropped_pairs << " pairs, region parts R1 "
           << st.region_dropped[0] << " / R2 " << st.region_dropped[1]
           << " blocks, largest dropped bound " << st.max_dropped_bound
           << "; per triangle pair: drop distances R1 " << st.drop_distance[0] << " / R2 "
           << st.drop_distance[1] << " m, " << st.pair_reductions << " of "
           << st.pair_distinct + st.pair_skips << " distinct pairs reduced (" << st.pair_skips
           << " skipped); pair evaluations " << st.pair_evaluations << " + memo hits "
           << st.pair_memo_hits << " (distinct " << st.pair_distinct << "; memo "
           << mb(st.memo_budget) << " MB per block, largest " << mb(st.memo_bytes_max) << " MB, "
           << st.memo_limited_blocks << " blocks at the budget); generated rows "
           << st.generated_rows << ", columns " << st.generated_columns;
    }
    return os.str();
}

std::size_t PmlRingOperator::near_bytes() const {
    return impl_->near->memory_bytes();
}

std::size_t PmlRingOperator::far_bytes() const {
    return impl_->aca ? impl_->aca->bytes() : impl_->far_alloc;
}

const PmlRingFarStats& PmlRingOperator::far_stats() const {
    return impl_->aca ? impl_->aca->stats() : impl_->exact_stats;
}

const std::vector<PmlRingFarBlock>& PmlRingOperator::far_blocks() const {
    static const std::vector<PmlRingFarBlock> none;
    return impl_->aca ? impl_->aca->blocks() : none;
}

std::size_t PmlRingOperator::memory_bytes() const {
    const Impl& m = *impl_;
    std::size_t b = sizeof(Impl) + near_bytes() + far_bytes();
    b += m.stretched.size() + m.stretched_pos.size() +
         (m.perm.size() + m.inv_perm.size() + m.leaves.leaf_of_pos.size()) * sizeof(Index);
    b += m.leaves.first.size() * (2 * sizeof(Index) + sizeof(std::array<Index, 3>));
    for (const auto& v : m.leaves.near) b += v.size() * sizeof(Index);
    return b;
}

Index PmlRingOperator::stretched_count() const {
    return impl_->est.stretched;
}

Index PmlRingOperator::near_pairs() const {
    return impl_->est.near_pairs;
}

Index PmlRingOperator::far_pairs() const {
    return impl_->est.far_pairs;
}

Index PmlRingOperator::far_stored_pairs() const {
    return impl_->est.far_stored;
}

Real PmlRingOperator::near_seconds() const {
    return impl_->near_s;
}

Real PmlRingOperator::far_seconds() const {
    return impl_->far_s;
}

const op::SparseStretchedStats& PmlRingOperator::near_stats() const {
    return impl_->near_stats;
}

std::size_t PmlRingOperator::peak_bytes() const {
    const std::size_t memo = impl_->aca ? impl_->aca->stats().memo_peak_bytes : 0;
    return near_bytes() + std::max(impl_->near_stats.cache_bytes, far_bytes() + memo);
}

const PmlRingEstimate& PmlRingOperator::estimate() const {
    return impl_->est;
}

const std::vector<std::uint8_t>& PmlRingOperator::stretched() const {
    return impl_->stretched;
}

const op::SparseOperator& PmlRingOperator::near_operator() const {
    return *impl_->near;
}

bool PmlRingOperator::far_contains(Index m, Index n) const {
    const Impl& d = *impl_;
    if (m < 0 || m >= d.n || n < 0 || n >= d.n)
        throw std::out_of_range("PmlRingOperator::far_contains: basis index out of range");
    if (d.stretched[sz(m)] == 0 && d.stretched[sz(n)] == 0)
        return false;
    const Index lm = d.leaves.leaf_of_pos[sz(d.inv_perm[sz(m)])];
    const Index ln = d.leaves.leaf_of_pos[sz(d.inv_perm[sz(n)])];
    return !leaves_near(d.leaves, lm, ln);
}

}  // namespace specklebem::mlfmm
