/// @file pml_ring_far.cpp
/// Far part of the PML ring (pml_ring_far.hpp): colouring, far partition, block entry generator,
/// the exact column kernel shared with WP-PML5a and the ACA far ring of WP-PML5b.
#include "pml_ring_far.hpp"

#include "specklebem/compression/aca/aca.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/formulation/formulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

namespace specklebem::mlfmm::detail {

namespace {

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// First exception thrown inside an OpenMP loop body (exceptions must not leave the region).
class ErrorSlot {
public:
    void capture() {
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp critical(specklebem_pml_ring_far_exception)
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

/// The slot of basis b in the support of triangle t.
int slot_of(const basis::RwgSpace::Support& sup, Index b) {
    for (int a = 0; a < sup.count; ++a)
        if (sup.n[a] == b)
            return a;
    throw std::logic_error("pml_ring_far: basis function not on its support triangle");
}

/// The formulation with the weights of the dropped regions set to zero: the PairBlockEvaluator
/// built from it skips those regions (BlockWeights::active).
class RegionMaskFormulation final : public formulation::Formulation {
public:
    RegionMaskFormulation(const formulation::Formulation& base, std::array<bool, 2> keep)
        : base_(base), keep_(keep) {}
    [[nodiscard]] formulation::Kind kind() const override { return base_.kind(); }
    [[nodiscard]] std::string name() const override { return base_.name(); }
    [[nodiscard]] formulation::Weights weights(Complex eta1, Complex eta2) const override {
        formulation::Weights w = base_.weights(eta1, eta2);
        if (!keep_[0])
            w.a1 = w.b1 = Complex(0.0, 0.0);
        if (!keep_[1])
            w.a2 = w.b2 = Complex(0.0, 0.0);
        return w;
    }

private:
    const formulation::Formulation& base_;
    std::array<bool, 2> keep_;
};

/// "never" for the sentinel 1e300 of PmlRingFarStats::drop_distance, else "<d> m".
std::string distance_text(Real d) {
    if (d >= 1e299)
        return "never";
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.3g m", d);
    return buf;
}

/// Bin of a rank in PmlRingFarStats::rank_histogram.
std::size_t rank_bin(Index r) {
    std::size_t b = 0;
    while (r > 0) {
        r >>= 1;
        ++b;
    }
    return b;
}

}  // namespace

std::vector<int> greedy_triangle_colours(const basis::RwgSpace& space) {
    const Index F = space.mesh().num_triangles();
    std::vector<int> colour(sz(F), -1);
    for (Index t = 0; t < F; ++t) {
        const basis::RwgSpace::Support sup = space.support(t);
        std::array<bool, 4> used{};
        for (int a = 0; a < sup.count; ++a) {
            const Index nb = sup.n[a];
            const Index other =
                space.plus_triangle(nb) == t ? space.minus_triangle(nb) : space.plus_triangle(nb);
            const int c = colour[sz(other)];
            if (c >= 0)
                used[static_cast<std::size_t>(c)] = true;
        }
        int c = 0;
        while (used[static_cast<std::size_t>(c)]) ++c;
        colour[sz(t)] = c;
    }
    return colour;
}

void Extent::add(const Extent& o) {
    for (int d = 0; d < 3; ++d) {
        const auto u = static_cast<std::size_t>(d);
        lo[u] = std::min(lo[u], o.lo[u]);
        hi[u] = std::max(hi[u], o.hi[u]);
    }
}

void Extent::add_support(const basis::RwgSpace& space, Index basis) {
    const geometry::TriangleMesh& mesh = space.mesh();
    const Triangles& tri = mesh.triangles();
    const Vertices& V = mesh.vertices();
    for (const Index t : {space.plus_triangle(basis), space.minus_triangle(basis)}) {
        for (int c = 0; c < 3; ++c) {
            const Index v = tri(t, c);
            for (int d = 0; d < 3; ++d) {
                const auto u = static_cast<std::size_t>(d);
                lo[u] = std::min(lo[u], V(v, d));
                hi[u] = std::max(hi[u], V(v, d));
            }
        }
    }
}

Real Extent::diameter() const {
    Real s = 0.0;
    for (std::size_t d = 0; d < 3; ++d) {
        if (hi[d] < lo[d])
            return 0.0;
        s += (hi[d] - lo[d]) * (hi[d] - lo[d]);
    }
    return std::sqrt(s);
}

Real extent_distance(const Extent& a, const Extent& b) {
    Real s = 0.0;
    for (std::size_t d = 0; d < 3; ++d) {
        const Real g = std::max({0.0, a.lo[d] - b.hi[d], b.lo[d] - a.hi[d]});
        s += g * g;
    }
    return std::sqrt(s);
}

std::vector<FarRowLeaf> partition_far_ring(const Octree& tree, const basis::RwgSpace& space,
                                           const std::vector<std::uint8_t>& stretched, bool use_aca,
                                           Real eta, bool coarse, bool row_clusters) {
    if (use_aca && !(eta > 0.0 && std::isfinite(eta)))
        throw std::invalid_argument("PmlRingParams::eta must be positive and finite");
    const Index N = space.size();
    if (static_cast<Index>(tree.permutation().size()) != N ||
        static_cast<Index>(stretched.size()) != N)
        throw std::invalid_argument("partition_far_ring: tree / classification not built on space");
    const auto& boxes = tree.boxes();
    const std::vector<Index>& perm = tree.permutation();
    const int L = tree.leaf_level();
    // Support extents of every box (leaves from their functions, parents merged).
    std::vector<Extent> ext(boxes.size());
    for (const Index b : tree.boxes_at_level(L)) {
        const Box& box = boxes[sz(b)];
        for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p)
            ext[sz(b)].add_support(space, perm[sz(p)]);
    }
    for (int l = L - 1; l >= 0; --l) {
        for (const Index b : tree.boxes_at_level(l)) {
            for (const Index c : boxes[sz(b)].children)
                if (c >= 0)
                    ext[sz(b)].add(ext[sz(c)]);
        }
    }
    const auto admissible_pair = [&](const Extent& r, Index c, Real& d, Real& dr, Real& dc) {
        d = extent_distance(r, ext[sz(c)]);
        dr = r.diameter();
        dc = ext[sz(c)].diameter();
        return d > 0.0 && std::max(dr, dc) <= eta * d;
    };
    std::vector<FarRowLeaf> out;
    if (use_aca && row_clusters) {
        // Row clusters (WP-PML5b): a dual descent over same-level box pairs (R, C) from the root;
        // a separated pair (no two leaves adjacent: the boxes are not adjacent on their level) is
        // a block if admissible or if both are leaves (exact), otherwise its children pairs are
        // visited; an adjacent pair is near on the leaf level, else subdivided.
        const auto nb = boxes.size();
        std::vector<Index> slot(nb, -1);  // box -> index into out
        std::vector<std::vector<Index>> rows_of(nb);
        std::vector<Extent> ext_s(nb);
        for (const Index b : tree.boxes_at_level(L)) {
            const Box& box = boxes[sz(b)];
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                const Index f = perm[sz(p)];
                if (stretched[sz(f)] != 0) {
                    rows_of[sz(b)].push_back(f);
                    ext_s[sz(b)].add_support(space, f);
                }
            }
        }
        for (int l = L - 1; l >= 0; --l) {
            for (const Index b : tree.boxes_at_level(l)) {
                for (const Index c : boxes[sz(b)].children) {
                    if (c < 0)
                        continue;
                    ext_s[sz(b)].add(ext_s[sz(c)]);
                    rows_of[sz(b)].insert(rows_of[sz(b)].end(), rows_of[sz(c)].begin(),
                                          rows_of[sz(c)].end());
                }
                const std::vector<Index>& inv = tree.inverse_permutation();
                std::sort(rows_of[sz(b)].begin(), rows_of[sz(b)].end(),
                          [&](Index x, Index y) { return inv[sz(x)] < inv[sz(y)]; });
            }
        }
        std::vector<std::array<Index, 2>> pairs;
        for (const Index r : tree.boxes_at_level(0))
            for (const Index c : tree.boxes_at_level(0)) pairs.push_back({r, c});
        while (!pairs.empty()) {
            const auto [r, c] = pairs.back();
            pairs.pop_back();
            const Box& R = boxes[sz(r)];
            const Box& C = boxes[sz(c)];
            if (rows_of[sz(r)].empty() || C.num_elements == 0)
                continue;
            bool adjacent = true;
            for (std::size_t d = 0; d < 3; ++d)
                adjacent = adjacent && std::abs(R.ijk[d] - C.ijk[d]) <= 1;
            const bool leaf = R.level == L;
            PmlRingFarBlock blk;
            bool block = false;
            if (!adjacent) {
                blk.admissible = admissible_pair(ext_s[sz(r)], c, blk.distance, blk.row_diameter,
                                                 blk.col_diameter);
                block = blk.admissible || leaf;
            } else if (leaf) {
                continue;  // near pair
            }
            if (block) {
                if (slot[sz(r)] < 0) {
                    slot[sz(r)] = static_cast<Index>(out.size());
                    FarRowLeaf F;
                    F.row_box = r;
                    F.rows = rows_of[sz(r)];
                    F.extent = ext_s[sz(r)];
                    out.push_back(std::move(F));
                }
                blk.row_box = r;
                blk.row_level = R.level;
                blk.box = c;
                blk.level = C.level;
                blk.first = C.first_element;
                blk.count = C.num_elements;
                blk.rows = static_cast<Index>(rows_of[sz(r)].size());
                blk.kind =
                    blk.admissible ? PmlRingFarBlock::Kind::aca : PmlRingFarBlock::Kind::exact;
                out[sz(slot[sz(r)])].blocks.push_back(blk);
                continue;
            }
            for (const Index rc : R.children) {
                if (rc < 0 || rows_of[sz(rc)].empty())
                    continue;
                for (const Index cc : C.children)
                    if (cc >= 0)
                        pairs.push_back({rc, cc});
            }
        }
        // Deterministic order: row clusters by level, then position; blocks by first column.
        std::sort(out.begin(), out.end(), [&](const FarRowLeaf& x, const FarRowLeaf& y) {
            const Box& bx = boxes[sz(x.row_box)];
            const Box& by = boxes[sz(y.row_box)];
            return bx.level != by.level ? bx.level < by.level : bx.first_element < by.first_element;
        });
        for (FarRowLeaf& F : out) {
            std::sort(F.blocks.begin(), F.blocks.end(),
                      [](const PmlRingFarBlock& x, const PmlRingFarBlock& y) {
                          return x.first < y.first;
                      });
        }
        return out;
    }
    std::vector<Index> stack;
    for (const Index a : tree.boxes_at_level(L)) {
        const Box& A = boxes[sz(a)];
        FarRowLeaf R;
        R.row_box = a;
        for (Index p = A.first_element; p < A.first_element + A.num_elements; ++p) {
            const Index b = perm[sz(p)];
            if (stretched[sz(b)] != 0) {
                R.rows.push_back(b);
                R.extent.add_support(space, b);
            }
        }
        if (R.rows.empty())
            continue;
        stack.clear();
        for (const Index r : tree.boxes_at_level(0)) stack.push_back(r);
        while (!stack.empty()) {
            const Index c = stack.back();
            stack.pop_back();
            const Box& C = boxes[sz(c)];
            if (C.num_elements == 0)
                continue;
            const Index span = Index{1} << (L - C.level);
            bool has_near = true;  // a leaf of C is A or adjacent to it
            for (std::size_t d = 0; d < 3; ++d) {
                const Index lo = C.ijk[d] * span, hi = lo + span - 1;
                has_near = has_near && A.ijk[d] >= lo - 1 && A.ijk[d] <= hi + 1;
            }
            const bool leaf = C.level == L;
            const auto push_children = [&] {
                for (auto it = C.children.rbegin(); it != C.children.rend(); ++it)
                    if (*it >= 0)
                        stack.push_back(*it);
            };
            if (has_near) {
                if (!leaf)
                    push_children();
                continue;
            }
            PmlRingFarBlock blk;
            blk.row_box = a;
            blk.row_level = L;
            blk.box = c;
            blk.level = C.level;
            blk.first = C.first_element;
            blk.count = C.num_elements;
            blk.rows = static_cast<Index>(R.rows.size());
            const bool adm =
                admissible_pair(R.extent, c, blk.distance, blk.row_diameter, blk.col_diameter);
            blk.admissible = use_aca && (coarse || leaf) && adm;
            if (blk.admissible || leaf) {
                blk.kind =
                    blk.admissible ? PmlRingFarBlock::Kind::aca : PmlRingFarBlock::Kind::exact;
                R.blocks.push_back(blk);
            } else {
                push_children();
            }
        }
        std::sort(
            R.blocks.begin(), R.blocks.end(),
            [](const PmlRingFarBlock& x, const PmlRingFarBlock& y) { return x.first < y.first; });
        out.push_back(std::move(R));
    }
    return out;
}

std::vector<FarRowLeaf> far_partition_from_blocks(const Octree& tree, const basis::RwgSpace& space,
                                                  const std::vector<std::uint8_t>& stretched,
                                                  std::vector<PmlRingFarBlock> blocks) {
    const Index N = space.size();
    if (static_cast<Index>(tree.permutation().size()) != N ||
        static_cast<Index>(stretched.size()) != N)
        throw std::invalid_argument(
            "far_partition_from_blocks: tree / classification not built on space");
    const auto& boxes = tree.boxes();
    const std::vector<Index>& perm = tree.permutation();
    std::vector<FarRowLeaf> out;
    for (PmlRingFarBlock& b : blocks) {
        if (out.empty() || out.back().row_box != b.row_box) {
            if (b.row_box < 0 || b.row_box >= static_cast<Index>(boxes.size()))
                throw std::invalid_argument(
                    "far_partition_from_blocks: a far block's row box is not a box of the tree");
            const Box& box = boxes[sz(b.row_box)];
            FarRowLeaf R;
            R.row_box = b.row_box;
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                const Index f = perm[sz(p)];
                if (stretched[sz(f)] != 0) {
                    R.rows.push_back(f);
                    R.extent.add_support(space, f);
                }
            }
            out.push_back(std::move(R));
        }
        if (b.rows != static_cast<Index>(out.back().rows.size()))
            throw std::invalid_argument(
                "far_partition_from_blocks: a far block's row count differs from the stretched "
                "functions of its row box (plan made on another tree or classification)");
        out.back().blocks.push_back(b);
    }
    return out;
}

}  // namespace specklebem::mlfmm::detail

namespace specklebem::mlfmm {

Real pml_ring_decay_bound(Real alpha, Real d, Real c) {
    const Real kappa = 2.0 / std::sqrt(4.0 + c * c);
    const Real F = (1.0 + c * c) * (1.0 + c * c) / (kappa * kappa * kappa);
    if (!(alpha > 0.0) || !(d > 0.0))
        return std::max(F, 1.0);
    const Real r = std::max(d, (1.0 - kappa) / (alpha * kappa));
    return F * (1.0 + alpha * r) * std::exp(-alpha * kappa * r);
}

}  // namespace specklebem::mlfmm

namespace specklebem::mlfmm::detail {

const op::TrianglePairBlocks* BlockSource::pair(std::size_t a, std::size_t s, bool seen) {
    // Memo lookup (WP-PML5d): the test triangle's strip, then the source triangle's; a pair not
    // found is evaluated into the first of them that exists (else into the scratch blocks).
    MemoEntry* slot = nullptr;
    for (MemoEntry* e : {test_memo_[a].empty() ? nullptr : &test_memo_[a][s],
                         src_memo_[s].empty() ? nullptr : &src_memo_[s][a]}) {
        if (e == nullptr)
            continue;
        if (e->state == 1) {
            ++memo_hits_;
            return &e->blk;
        }
        if (e->state == 2)
            return nullptr;
        if (slot == nullptr)
            slot = e;
    }
    const Index t = test_[a].t, u = src_[s].t;
    int mask = mask_;
    if (!eval_.tri_ext.empty()) {
        const Real d = extent_distance(eval_.tri_ext[sz(t)], eval_.tri_ext[sz(u)]);
        for (int i = 0; i < 2; ++i) {
            if ((mask & (1 << i)) != 0 && d > eval_.drop_distance[static_cast<std::size_t>(i)])
                mask &= ~(1 << i);
        }
    }
    if (!seen) {  // reductions, skips and distinct pairs count each pair once
        if (mask != mask_)
            ++reductions_;
        ++(mask == 0 ? skips_ : distinct_);
    }
    if (mask == 0) {
        if (slot != nullptr)
            slot->state = 2;
        return nullptr;
    }
    ++evaluations_;
    op::TrianglePairBlocks& out = slot != nullptr ? slot->blk : scratch_;
    eval_.eval[static_cast<std::size_t>(mask)]->evaluate(t, u, out);
    if (slot != nullptr)
        slot->state = 1;
    return &out;
}

void BlockSource::allocate_strip(std::vector<MemoEntry>& strip, std::size_t entries) {
    if (!strip.empty() || entries == 0 || memo_budget_ == 0)
        return;
    const std::size_t bytes = entries * sizeof(MemoEntry);
    if (memo_bytes_ > memo_budget_ || bytes > memo_budget_ - memo_bytes_) {
        memo_limited_ = true;
        return;
    }
    strip.resize(entries);
    memo_bytes_ += bytes;
}

BlockSource::BlockSource(const basis::RwgSpace& space, const PairEvaluators& eval, int mask,
                         const std::vector<int>& colour, const std::vector<Index>& perm,
                         const std::vector<Index>& rows, Index c0, Index n, std::size_t memo_budget)
    : eval_(eval),
      mask_(mask),
      m_(static_cast<Index>(rows.size())),
      n_(n),
      memo_budget_(memo_budget) {
    // Test triangles: (colour, triangle, slot, local row) of every row's two triangles.
    struct Entry {
        int colour;
        Index t;
        int slot;
        Index local;
    };
    std::vector<Entry> te;
    te.reserve(2 * rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Index b = rows[i];
        for (const Index t : {space.plus_triangle(b), space.minus_triangle(b)})
            te.push_back({colour[sz(t)], t, slot_of(space.support(t), b), static_cast<Index>(i)});
    }
    std::sort(te.begin(), te.end(), [](const Entry& x, const Entry& y) {
        return x.colour != y.colour ? x.colour < y.colour : x.t < y.t;
    });
    row_tris_.assign(rows.size(), {-1, -1});
    for (const Entry& e : te) {
        if (test_.empty() || test_.back().t != e.t)
            test_.push_back(Tri{e.t, {-1, -1, -1}});
        test_.back().local[static_cast<std::size_t>(e.slot)] = e.local;
        auto& rt = row_tris_[sz(e.local)];
        (rt[0] < 0 ? rt[0] : rt[1]) = static_cast<Index>(test_.size()) - 1;
    }
    // Source triangles: (triangle, slot, local column), ascending triangle.
    std::vector<Entry> se;
    se.reserve(2 * sz(n));
    for (Index j = 0; j < n; ++j) {
        const Index b = perm[sz(c0 + j)];
        for (const Index t : {space.plus_triangle(b), space.minus_triangle(b)})
            se.push_back({0, t, slot_of(space.support(t), b), j});
    }
    std::sort(se.begin(), se.end(), [](const Entry& x, const Entry& y) {
        return x.t != y.t ? x.t < y.t : x.slot < y.slot;
    });
    col_tris_.assign(sz(n), {-1, -1});
    for (const Entry& e : se) {
        if (src_.empty() || src_.back().t != e.t)
            src_.push_back(Tri{e.t, {-1, -1, -1}});
        src_.back().local[static_cast<std::size_t>(e.slot)] = e.local;
        auto& ct = col_tris_[sz(e.local)];
        (ct[0] < 0 ? ct[0] : ct[1]) = static_cast<Index>(src_.size()) - 1;
    }
    row_cache_.resize(rows.size());
    col_cache_.resize(sz(n));
    test_seen_.assign(test_.size(), 0);
    src_seen_.assign(src_.size(), 0);
    test_memo_.resize(test_.size());
    src_memo_.resize(src_.size());
}

const Complex* BlockSource::row(Index i) {
    std::vector<Complex>& out = row_cache_[sz(i)];
    if (!out.empty())
        return out.data();
    const std::size_t n = sz(n_);
    out.assign(4 * n, Complex(0.0, 0.0));
    for (const Index e : row_tris_[sz(i)]) {  // the row's test triangles in colour order
        const Tri& T = test_[sz(e)];
        int a = 0;
        while (T.local[static_cast<std::size_t>(a)] != i) ++a;
        const bool seen_t = full_seen_ || test_seen_[sz(e)] != 0;
        test_seen_[sz(e)] = 1;
        allocate_strip(test_memo_[sz(e)], src_.size());
        for (std::size_t s = 0; s < src_.size(); ++s) {
            const Tri& S = src_[s];
            const op::TrianglePairBlocks* blk = pair(sz(e), s, seen_t || src_seen_[s] != 0);
            if (blk == nullptr)
                continue;
            for (int b = 0; b < 3; ++b) {
                const Index j = S.local[static_cast<std::size_t>(b)];
                if (j < 0)
                    continue;
                const auto k = static_cast<std::size_t>(3 * a + b);
                const std::size_t jj = sz(j);
                out[jj] += blk->jj[k];
                out[n + jj] += blk->jm[k];
                out[2 * n + jj] += blk->mj[k];
                out[3 * n + jj] += blk->mm[k];
            }
        }
    }
    row_order_.push_back(i);
    return out.data();
}

const Complex* BlockSource::column(Index j) {
    std::vector<Complex>& out = col_cache_[sz(j)];
    if (!out.empty())
        return out.data();
    const std::size_t m = sz(m_);
    out.assign(4 * m, Complex(0.0, 0.0));
    const auto& ct = col_tris_[sz(j)];  // ascending (src_ is sorted)
    // Distinct-pair bookkeeping (pair_distinct): the column's source triangles before this call.
    std::array<bool, 2> seen_s{};
    for (std::size_t c = 0; c < 2; ++c) {
        seen_s[c] = full_seen_ || src_seen_[sz(ct[c])] != 0;
        src_seen_[sz(ct[c])] = 1;
        allocate_strip(src_memo_[sz(ct[c])], test_.size());
    }
    // Loop nesting as without the memo (test triangles outer, the column's source triangles
    // inner): the summation order into `out` is unchanged (WP-PML5d).
    for (std::size_t a0 = 0; a0 < test_.size(); ++a0) {
        const Tri& T = test_[a0];
        for (std::size_t c = 0; c < 2; ++c) {
            const Tri& S = src_[sz(ct[c])];
            int b = 0;
            while (S.local[static_cast<std::size_t>(b)] != j) ++b;
            const op::TrianglePairBlocks* blk =
                pair(a0, sz(ct[c]), seen_s[c] || test_seen_[a0] != 0);
            if (blk == nullptr)
                continue;
            for (int a = 0; a < 3; ++a) {
                const Index i = T.local[static_cast<std::size_t>(a)];
                if (i < 0)
                    continue;
                const auto k = static_cast<std::size_t>(3 * a + b);
                const std::size_t ii = sz(i);
                out[ii] += blk->jj[k];
                out[m + ii] += blk->jm[k];
                out[2 * m + ii] += blk->mj[k];
                out[3 * m + ii] += blk->mm[k];
            }
        }
    }
    col_order_.push_back(j);
    return out.data();
}

void BlockSource::release_cache() {
    for (const Index i : row_order_) std::vector<Complex>().swap(row_cache_[sz(i)]);
    for (const Index j : col_order_) std::vector<Complex>().swap(col_cache_[sz(j)]);
}

std::size_t BlockSource::cache_bytes() const {
    std::size_t b = 0;
    for (const Index i : row_order_) b += row_cache_[sz(i)].capacity() * sizeof(Complex);
    for (const Index j : col_order_) b += col_cache_[sz(j)].capacity() * sizeof(Complex);
    return b + memo_bytes_;
}

void BlockSource::full(std::vector<Complex>& w) {
    const std::size_t m = sz(m_);
    w.assign(4 * m * sz(n_), Complex(0.0, 0.0));
    const bool seen_all = full_seen_;
    for (std::size_t a0 = 0; a0 < test_.size(); ++a0) {
        const Tri& T = test_[a0];
        for (std::size_t s = 0; s < src_.size(); ++s) {
            const Tri& S = src_[s];
            const op::TrianglePairBlocks* blk =
                pair(a0, s, seen_all || test_seen_[a0] != 0 || src_seen_[s] != 0);
            if (blk == nullptr)
                continue;
            for (int a = 0; a < 3; ++a) {
                const Index i = T.local[static_cast<std::size_t>(a)];
                if (i < 0)
                    continue;
                for (int b = 0; b < 3; ++b) {
                    const Index j = S.local[static_cast<std::size_t>(b)];
                    if (j < 0)
                        continue;
                    const auto k = static_cast<std::size_t>(3 * a + b);
                    Complex* col = w.data() + 4 * m * sz(j);
                    const std::size_t ii = sz(i);
                    col[ii] += blk->jj[k];
                    col[m + ii] += blk->jm[k];
                    col[2 * m + ii] += blk->mj[k];
                    col[3 * m + ii] += blk->mm[k];
                }
            }
        }
    }
    full_seen_ = true;
}

void apply_exact_columns(const Complex* wc, std::size_t ns, const Real* xs, const Real* xr,
                         Index p0, Index p1, std::size_t j0, Real* acc, Real* u,
                         const std::uint8_t* stretched_pos) {
    const Real* w = reinterpret_cast<const Real*>(wc);
    std::size_t j = j0;
    for (Index p = p0; p < p1; ++p, ++j) {
        const Real* c = w + 8 * ns * j;  // JJ, JM, MJ, MM columns (re, im)
        const Real* xp = xr + 4 * sz(p);
        const Real jr = xp[0], ji = xp[1], mr = xp[2], mi = xp[3];
        Real ujr = 0, uji = 0, umr = 0, umi = 0;
        for (std::size_t i = 0; i < ns; ++i) {
            const Real ar = c[2 * i], ai = c[2 * i + 1];                // JJ
            const Real br = c[2 * (ns + i)], bi = c[2 * (ns + i) + 1];  // JM
            const Real cr = c[2 * (2 * ns + i)], ci = c[2 * (2 * ns + i) + 1];
            const Real dr = c[2 * (3 * ns + i)], di = c[2 * (3 * ns + i) + 1];
            // Direct: y_J,s += JJ x_J + JM x_M; y_M,s += MJ x_J + MM x_M.
            acc[i] += ar * jr - ai * ji + br * mr - bi * mi;
            acc[ns + i] += ar * ji + ai * jr + br * mi + bi * mr;
            acc[2 * ns + i] += cr * jr - ci * ji + dr * mr - di * mi;
            acc[3 * ns + i] += cr * ji + ci * jr + dr * mi + di * mr;
            // Transposed: u_J += JJ x_J,s + JM x_M,s; u_M += MJ x_J,s + MM x_M,s.
            const Real sjr = xs[i], sji = xs[ns + i];
            const Real smr = xs[2 * ns + i], smi = xs[3 * ns + i];
            ujr += ar * sjr - ai * sji + br * smr - bi * smi;
            uji += ar * sji + ai * sjr + br * smi + bi * smr;
            umr += cr * sjr - ci * sji + dr * smr - di * smi;
            umi += cr * sji + ci * sjr + dr * smi + di * smr;
        }
        if (stretched_pos[sz(p)] == 0) {  // a stretched row's leaf stores the pair
            Real* up = u + 4 * sz(p);
            up[0] += ujr;
            up[1] += uji;
            up[2] += umr;
            up[3] += umi;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ACA far ring
// ---------------------------------------------------------------------------------------------

namespace {

/// One sub-block's factors: U (m x k) and V (n x k), column-major.
struct Factors {
    Index rank = 0;
    std::vector<Complex> U, V;
};

struct Stored {
    Index first = 0, count = 0;
    bool low_rank = false;
    std::vector<Complex> w;    ///< exact: 4 m per column (5a layout)
    std::array<Factors, 4> f;  ///< low rank: JJ, JM, MJ, MM
};

struct LeafData {
    std::vector<Index> rows;
    std::vector<Index> row_pos;  ///< permuted positions of rows
    std::vector<Stored> blocks;
    Real cost = 0;  ///< stored entries (chunk balance)
};

/// Sub-block X of a BlockSource as an aca::EntryGenerator.
class SubBlock final : public aca::EntryGenerator {
public:
    SubBlock(BlockSource& src, int x) : src_(src), x_(static_cast<std::size_t>(x)) {}
    void row(Index i, Complex* out) const override {
        const Complex* r = src_.row(i) + x_ * sz(src_.cols());
        std::copy(r, r + src_.cols(), out);
    }
    void column(Index j, Complex* out) const override {
        const Complex* c = src_.column(j) + x_ * sz(src_.rows());
        std::copy(c, c + src_.rows(), out);
    }

private:
    BlockSource& src_;
    std::size_t x_;
};

}  // namespace

struct AcaFarRing::Impl {
    Index n = 0;
    std::vector<Index> perm;
    std::vector<std::uint8_t> stretched_pos;
    std::vector<LeafData> leaves;
    std::vector<PmlRingFarBlock> info;
    PmlRingFarStats stats;
    std::size_t bytes = 0;
    std::vector<Index> chunk_first;
    static constexpr Index kChunks = 64;

    void partition_chunks();
};

void AcaFarRing::Impl::partition_chunks() {
    const auto nr = static_cast<Index>(leaves.size());
    const Index K = std::min<Index>(nr, kChunks);
    chunk_first.assign(sz(K) + 1, nr);
    if (K == 0)
        return;
    Real total = 0;
    for (const LeafData& R : leaves) total += R.cost;
    chunk_first[0] = 0;
    Real acc = 0;
    Index k = 0;
    for (Index r = 0; r + 1 < nr && k + 1 < K; ++r) {
        acc += leaves[sz(r)].cost;
        if (acc >= total * static_cast<Real>(k + 1) / static_cast<Real>(K))
            chunk_first[sz(++k)] = r + 1;
    }
}

AcaFarRing::AcaFarRing(const op::Problem& problem, const std::vector<int>& colour,
                       std::vector<FarRowLeaf> partition, const std::vector<Index>& perm,
                       const std::vector<Index>& inv_perm,
                       const std::vector<std::uint8_t>& stretched_pos, const PmlRingParams& params,
                       Real epsilon, Real drop_tolerance)
    : impl_(std::make_unique<Impl>()) {
    if (!(epsilon > 0.0 && epsilon < 1.0) || !(drop_tolerance >= 0.0 && drop_tolerance < 1.0))
        throw std::invalid_argument(
            "AcaFarRing: the ACA tolerance must lie in (0, 1), the drop tolerance in [0, 1)");
    Impl& m = *impl_;
    const basis::RwgSpace& space = *problem.space;
    m.n = space.size();
    m.perm = perm;
    m.stretched_pos = stretched_pos;
    m.stats.aca = true;
    m.stats.epsilon = epsilon;
    m.stats.drop_tolerance = drop_tolerance;
    m.stats.eta = params.eta;
    m.stats.memo_budget = params.pair_memo_bytes;
    // Evaluators: both regions, R1 only, R2 only (the region-masked formulations exist only
    // during construction; the evaluators copy the weights).
    const op::PairBlockEvaluator both(problem);
    const std::array<bool, 2> active = both.weights().active;
    const RegionMaskFormulation f1(*problem.formulation, {true, false});
    const RegionMaskFormulation f2(*problem.formulation, {false, true});
    op::Problem p1 = problem, p2 = problem;
    p1.formulation = &f1;
    p2.formulation = &f2;
    const op::PairBlockEvaluator only1(p1), only2(p2);
    PairEvaluators pe;
    pe.eval = {nullptr, &only1, &only2, &both};
    const std::array<Real, 2> alpha = {-problem.exterior.wavenumber(problem.omega).imag(),
                                       -problem.object.wavenumber(problem.omega).imag()};
    Real c = 0.0;
    if (problem.stretch != nullptr) {
        if (problem.stretch->profile_x())
            c = std::max(c, problem.stretch->profile_x()->sigma_max);
        if (problem.stretch->profile_y())
            c = std::max(c, problem.stretch->profile_y()->sigma_max);
    }
    const Real tau = drop_tolerance;
    // Per-triangle-pair drop distances: the smallest d with pml_ring_decay_bound(alpha_i, d, c)
    // <= tau (the bound decreases beyond its maximum), by bisection; none for alpha_i = 0.
    for (std::size_t i = 0; i < 2; ++i) {
        if (!(alpha[i] > 0.0) || !(tau > 0.0) || pml_ring_decay_bound(alpha[i], 1.0, c) > tau)
            continue;
        Real lo = 0.0, hi = 1.0;
        for (int it = 0; it < 200; ++it) {
            const Real mid = 0.5 * (lo + hi);
            (pml_ring_decay_bound(alpha[i], mid, c) > tau ? lo : hi) = mid;
        }
        pe.drop_distance[i] = hi;
    }
    if (pe.drop_distance[0] < 1e300 || pe.drop_distance[1] < 1e300) {
        const Index F = space.mesh().num_triangles();
        const Triangles& tri = space.mesh().triangles();
        const Vertices& V = space.mesh().vertices();
        pe.tri_ext.resize(sz(F));
        for (Index t = 0; t < F; ++t) {
            Extent& e = pe.tri_ext[sz(t)];
            for (int k = 0; k < 3; ++k) {
                for (std::size_t d = 0; d < 3; ++d) {
                    const Real v = V(tri(t, k), static_cast<Index>(d));
                    e.lo[d] = std::min(e.lo[d], v);
                    e.hi[d] = std::max(e.hi[d], v);
                }
            }
        }
    }
    m.stats.drop_distance = pe.drop_distance;
    const auto nl = static_cast<Index>(partition.size());
    m.leaves.resize(sz(nl));
    ErrorSlot error;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (Index r = 0; r < nl; ++r) {
        try {
            FarRowLeaf& R = partition[sz(r)];
            LeafData& D = m.leaves[sz(r)];
            D.rows = R.rows;
            D.row_pos.reserve(R.rows.size());
            for (const Index b : R.rows) D.row_pos.push_back(inv_perm[sz(b)]);
            const Index rows = static_cast<Index>(R.rows.size());
            for (PmlRingFarBlock& b : R.blocks) {
                for (std::size_t i = 0; i < 2; ++i) {
                    b.bound[i] = active[i] ? pml_ring_decay_bound(alpha[i], b.distance, c) : 0.0;
                    b.region[i] = active[i] && b.bound[i] > tau;
                }
                if (!b.region[0] && !b.region[1]) {
                    b.kind = PmlRingFarBlock::Kind::dropped;
                    SBEM_DEBUG(
                        "PmlRingOperator: far block (row box {}, box {}, level {}, {} x {}) "
                        "dropped: bounds {:.3e} / {:.3e} <= {:.1e}",
                        b.row_box, b.box, b.level, rows, b.count, b.bound[0], b.bound[1], tau);
                    continue;
                }
                const int mask = (b.region[0] ? 1 : 0) | (b.region[1] ? 2 : 0);
                BlockSource src(space, pe, mask, colour, perm, R.rows, b.first, b.count,
                                params.pair_memo_bytes);
                Stored s;
                s.first = b.first;
                s.count = b.count;
                // Rank cap: the break-even rank (low rank never larger than exact), or an explicit
                // PmlRingParams::max_rank (diagnostics and tests; may exceed the break-even).
                const Index breakeven = rows * b.count / (rows + b.count);
                const Index cap = params.max_rank > 0 ? params.max_rank : breakeven;
                // Review W3: no ACA attempt below kPmlRingMinAcaRank (automatic cap only): with
                // the two confirming steps such a block converges only for sub-blocks of rank
                // <= 1, so the attempt would almost surely end at the cap and generate the
                // block twice.
                const bool attempt = b.admissible && cap >= 1 &&
                                     (params.max_rank > 0 || breakeven >= kPmlRingMinAcaRank);
                b.aca_skipped = b.admissible && !attempt;
                bool done = false;
                if (attempt) {
                    bool ok = true;
                    for (const int x : {0, 3, 1, 2}) {
                        aca::CompressParams cp;
                        cp.epsilon = epsilon;
                        cp.max_rank = cap;
                        cp.preferred_rows = src.cached_rows();
                        cp.preferred_columns = src.cached_columns();
                        const SubBlock gen(src, x);
                        const aca::LowRankBlock lr = aca::compress(rows, b.count, gen, cp);
                        if (!lr.converged()) {
                            ok = false;
                            break;
                        }
                        Factors& f = s.f[static_cast<std::size_t>(x)];
                        f.rank = lr.rank();
                        f.U.assign(lr.U.data(), lr.U.data() + lr.U.size());
                        f.V.assign(lr.V.data(), lr.V.data() + lr.V.size());
                        b.rank[static_cast<std::size_t>(x)] = lr.rank();
                    }
                    if (ok) {
                        s.low_rank = true;
                        b.kind = PmlRingFarBlock::Kind::aca;
                        done = true;
                    } else {
                        b.kind = PmlRingFarBlock::Kind::fallback;
                        b.rank = {0, 0, 0, 0};
                        s.f = {};
                        SBEM_DEBUG(
                            "PmlRingOperator: far block (row box {}, box {}, level {}, {} x {}) "
                            "reached the rank cap {}: stored exactly",
                            b.row_box, b.box, b.level, rows, b.count, cap);
                    }
                    b.generated_rows = src.generated_rows();
                    b.generated_columns = src.generated_columns();
                }
                if (!done) {
                    // Admissible blocks stored exactly (rank cap reached, or no attempt below
                    // kPmlRingMinAcaRank) count as fallbacks.
                    b.kind = b.admissible ? PmlRingFarBlock::Kind::fallback
                                          : PmlRingFarBlock::Kind::exact;
                    // Free the rows / columns of a failed attempt before the exact block is
                    // allocated (review W3; full() recomputes every entry, bitwise the same).
                    src.release_cache();
                    src.full(s.w);
                    D.cost += static_cast<Real>(s.w.size());
                } else {
                    for (const Factors& f : s.f)
                        D.cost += static_cast<Real>(f.rank * (rows + b.count));
                }
                b.pair_evaluations = src.pair_evaluations();
                b.pair_memo_hits = src.pair_memo_hits();
                b.pair_distinct = src.pair_distinct();
                b.memo_bytes = src.memo_bytes();
                b.memo_limited = src.memo_limited();
                b.pair_reductions = src.pair_reductions();
                b.pair_skips = src.pair_skips();
                D.blocks.push_back(std::move(s));
            }
        } catch (...) {
            error.capture();
        }
    }
    error.rethrow();
    // Statistics in row-leaf order (deterministic).
    PmlRingFarStats& st = m.stats;
    std::vector<Index> ranks;
    std::array<Real, 4> rank_sum = {0, 0, 0, 0};
    for (std::size_t r = 0; r < partition.size(); ++r) {
        const FarRowLeaf& R = partition[r];
        const auto rows = static_cast<Index>(R.rows.size());
        m.bytes += R.rows.size() * sizeof(Index) + 64;
        for (const PmlRingFarBlock& b : R.blocks) {
            const Index pairs = rows * b.count;
            for (std::size_t i = 0; i < 2; ++i) {
                if (active[i] && !b.region[i]) {
                    ++st.region_dropped[i];
                    st.max_dropped_bound = std::max(st.max_dropped_bound, b.bound[i]);
                }
            }
            switch (b.kind) {
                case PmlRingFarBlock::Kind::aca:
                    ++st.aca_blocks;
                    st.aca_pairs += pairs;
                    for (std::size_t x = 0; x < 4; ++x) {
                        ranks.push_back(b.rank[x]);
                        rank_sum[x] += static_cast<Real>(b.rank[x]);
                        st.aca_bytes += sz(b.rank[x]) * sz(rows + b.count) * sizeof(Complex);
                    }
                    break;
                case PmlRingFarBlock::Kind::exact:
                case PmlRingFarBlock::Kind::fallback:
                    (b.kind == PmlRingFarBlock::Kind::exact ? st.exact_blocks
                                                            : st.fallback_blocks)++;
                    st.skipped_attempts += b.aca_skipped ? 1 : 0;
                    st.exact_pairs += pairs;
                    st.exact_bytes += 4 * sz(pairs) * sizeof(Complex);
                    break;
                case PmlRingFarBlock::Kind::dropped:
                    ++st.dropped_blocks;
                    st.dropped_pairs += pairs;
                    break;
            }
            st.generated_rows += b.generated_rows;
            st.generated_columns += b.generated_columns;
            st.pair_evaluations += b.pair_evaluations;
            st.pair_memo_hits += b.pair_memo_hits;
            st.pair_distinct += b.pair_distinct;
            st.memo_bytes_max = std::max(st.memo_bytes_max, b.memo_bytes);
            st.memo_limited_blocks += b.memo_limited ? 1 : 0;
            st.pair_reductions += b.pair_reductions;
            st.pair_skips += b.pair_skips;
            m.info.push_back(b);
        }
    }
    // Pair memos alive at once (WP-PML5d): one block per assembling thread.
    std::size_t threads = 1;
#ifdef SPECKLEBEM_HAVE_OPENMP
    threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
    st.memo_peak_bytes = std::min(threads, partition.size()) * st.memo_bytes_max;
    m.bytes += st.aca_bytes + st.exact_bytes +
               m.info.size() * (sizeof(Stored) + sizeof(PmlRingFarBlock)) +
               m.perm.size() * sizeof(Index) + m.stretched_pos.size();
    if (!ranks.empty()) {
        std::sort(ranks.begin(), ranks.end());
        st.rank_min = ranks.front();
        st.rank_max = ranks.back();
        st.rank_median = ranks[ranks.size() / 2];
        for (std::size_t x = 0; x < 4; ++x)
            st.rank_mean[x] = rank_sum[x] / static_cast<Real>(st.aca_blocks);
        st.rank_histogram.assign(rank_bin(st.rank_max) + 1, 0);
        for (const Index k : ranks) ++st.rank_histogram[rank_bin(k)];
    }
    if (st.dropped_blocks > 0 || st.region_dropped[0] > 0 || st.region_dropped[1] > 0 ||
        st.pair_reductions > 0) {
        SBEM_INFO(
            "PmlRingOperator: magnitude bound (ADR 0008 §6) <= {:.1e}: {} far blocks dropped ({} "
            "pairs), region parts dropped R1 {} / R2 {} blocks, largest dropped bound {:.3e}; "
            "triangle pairs reduced {} of {} ({} skipped; region parts dropped beyond R1 {} / R2 "
            "{}, each with bound <= {:.1e})",
            tau, st.dropped_blocks, st.dropped_pairs, st.region_dropped[0], st.region_dropped[1],
            st.max_dropped_bound, st.pair_reductions, st.pair_distinct + st.pair_skips,
            st.pair_skips, distance_text(st.drop_distance[0]), distance_text(st.drop_distance[1]),
            tau);
    }
    m.partition_chunks();
}

AcaFarRing::~AcaFarRing() = default;

const PmlRingFarStats& AcaFarRing::stats() const {
    return impl_->stats;
}

const std::vector<PmlRingFarBlock>& AcaFarRing::blocks() const {
    return impl_->info;
}

std::size_t AcaFarRing::bytes() const {
    return impl_->bytes;
}

void AcaFarRing::apply(const VectorXc& x, VectorXc& y) const {
    const Impl& m = *impl_;
    if (m.leaves.empty())
        return;
    const Index N = m.n;
    std::vector<Real> xr(4 * sz(N));  // [Jr, Ji, Mr, Mi] per position
    for (Index p = 0; p < N; ++p) {
        const Index b = m.perm[sz(p)];
        const Complex cj = x(b), cm = x(N + b);
        xr[4 * sz(p)] = cj.real();
        xr[4 * sz(p) + 1] = cj.imag();
        xr[4 * sz(p) + 2] = cm.real();
        xr[4 * sz(p) + 3] = cm.imag();
    }
    const auto K = static_cast<Index>(m.chunk_first.size()) - 1;
    std::vector<Real> buf(sz(K) * 4 * sz(N), 0.0);
    const std::uint8_t* spos = m.stretched_pos.data();
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Real> xs, acc, t;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (Index k = 0; k < K; ++k) {
            Real* u = buf.data() + sz(k) * 4 * sz(N);
            for (Index r = m.chunk_first[sz(k)]; r < m.chunk_first[sz(k) + 1]; ++r) {
                const LeafData& R = m.leaves[sz(r)];
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
                for (const Stored& s : R.blocks) {
                    if (!s.low_rank) {
                        apply_exact_columns(s.w.data(), ns, xs.data(), xr.data(), s.first,
                                            s.first + s.count, 0, acc.data(), u, spos);
                        continue;
                    }
                    const std::size_t n = sz(s.count);
                    for (std::size_t xi = 0; xi < 4; ++xi) {
                        const Factors& f = s.f[xi];
                        if (f.rank == 0)
                            continue;
                        const std::size_t kr = sz(f.rank);
                        // src: J for JJ / MJ, M for JM / MM; dst: J for JJ / JM, M for MJ / MM.
                        const std::size_t so = (xi == 1 || xi == 3) ? 2 : 0;
                        const std::size_t dor = (xi >= 2) ? 2 * ns : 0;
                        const std::size_t sor = (xi == 1 || xi == 3) ? 2 * ns : 0;
                        const std::size_t uo = (xi >= 2) ? 2 : 0;
                        const Real* U = reinterpret_cast<const Real*>(f.U.data());
                        const Real* V = reinterpret_cast<const Real*>(f.V.data());
                        t.assign(4 * kr, 0.0);  // [direct re, im | transposed re, im] per l
                        for (std::size_t l = 0; l < kr; ++l) {
                            const Real* v = V + 2 * n * l;
                            const Real* xp = xr.data() + 4 * sz(s.first) + so;
                            Real tr = 0, ti = 0;
                            for (std::size_t j = 0; j < n; ++j, xp += 4) {
                                const Real vr = v[2 * j], vi = v[2 * j + 1];
                                tr += vr * xp[0] - vi * xp[1];
                                ti += vr * xp[1] + vi * xp[0];
                            }
                            const Real* uu = U + 2 * ns * l;
                            Real sr = 0, si = 0;
                            for (std::size_t i = 0; i < ns; ++i) {
                                const Real ur = uu[2 * i], ui = uu[2 * i + 1];
                                const Real xa = xs[sor + i], xb = xs[sor + ns + i];
                                sr += ur * xa - ui * xb;
                                si += ur * xb + ui * xa;
                            }
                            t[4 * l] = tr;
                            t[4 * l + 1] = ti;
                            t[4 * l + 2] = sr;
                            t[4 * l + 3] = si;
                        }
                        // Direct: acc_dst += U t_direct.
                        for (std::size_t l = 0; l < kr; ++l) {
                            const Real* uu = U + 2 * ns * l;
                            const Real tr = t[4 * l], ti = t[4 * l + 1];
                            for (std::size_t i = 0; i < ns; ++i) {
                                const Real ur = uu[2 * i], ui = uu[2 * i + 1];
                                acc[dor + i] += ur * tr - ui * ti;
                                acc[dor + ns + i] += ur * ti + ui * tr;
                            }
                        }
                        // Transposed: u_dst(C, real positions) += V t_transposed.
                        for (std::size_t l = 0; l < kr; ++l) {
                            const Real* v = V + 2 * n * l;
                            const Real sr = t[4 * l + 2], si = t[4 * l + 3];
                            Real* up = u + 4 * sz(s.first) + uo;
                            const std::uint8_t* sp = spos + s.first;
                            for (std::size_t j = 0; j < n; ++j, up += 4) {
                                if (sp[j] != 0)
                                    continue;
                                const Real vr = v[2 * j], vi = v[2 * j + 1];
                                up[0] += vr * sr - vi * si;
                                up[1] += vr * si + vi * sr;
                            }
                        }
                    }
                }
                // Direct products into the chunk buffer as well: a stretched row can belong to
                // row clusters of several levels (several chunks).
                for (std::size_t i = 0; i < ns; ++i) {
                    Real* up = u + 4 * sz(R.row_pos[i]);
                    up[0] += acc[i];
                    up[1] += acc[ns + i];
                    up[2] += acc[2 * ns + i];
                    up[3] += acc[3 * ns + i];
                }
            }
        }
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp for schedule(static)
#endif
        for (Index p = 0; p < N; ++p) {
            Real s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (Index k = 0; k < K; ++k) {
                const Real* up = buf.data() + sz(k) * 4 * sz(N) + 4 * sz(p);
                s0 += up[0];
                s1 += up[1];
                s2 += up[2];
                s3 += up[3];
            }
            const Index b = m.perm[sz(p)];
            y(b) += Complex(s0, s1);
            y(N + b) += Complex(s2, s3);
        }
    }
}

}  // namespace specklebem::mlfmm::detail
