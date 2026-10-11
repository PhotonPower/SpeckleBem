/// @file near_field.cpp
/// Near pattern of the octree (leaf neighbourhoods and the home-level rule of elevated basis
/// functions) and Z_near assembly (ADR 0008 §1, WP19a; local leaf rule, WP21L).
#include "specklebem/compression/mlfmm/near_field.hpp"

#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace specklebem::mlfmm {

namespace {

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// Levels below the leaf level that hold elevated basis functions, ascending.
std::vector<int> elevated_levels(const Octree& tree) {
    std::vector<int> out;
    for (int l = 0; l < tree.leaf_level(); ++l) {
        if (!tree.elevated_positions(l).empty())
            out.push_back(l);
    }
    return out;
}

/// anc[l] = ancestor of `leaf` on level l (anc[leaf level] = leaf).
void ancestors(const Octree& tree, Index leaf, std::vector<Index>& anc) {
    anc.assign(sz(tree.levels()), -1);
    anc[sz(tree.leaf_level())] = leaf;
    for (int l = tree.leaf_level() - 1; l >= 0; --l)
        anc[sz(l)] = tree.boxes()[sz(anc[sz(l + 1)])].parent;
}

}  // namespace

op::BasisPattern near_pattern(const Octree& tree, const basis::RwgSpace& space) {
    const Index N = space.size();
    const std::vector<Index>& perm = tree.permutation();
    if (static_cast<Index>(perm.size()) != N) {
        throw std::invalid_argument("near_pattern: the octree holds " +
                                    std::to_string(perm.size()) + " basis functions, the space " +
                                    std::to_string(N));
    }
    const std::vector<Box>& boxes = tree.boxes();
    const int leaf_level = tree.leaf_level();
    const std::vector<Index>& leaves = tree.boxes_at_level(leaf_level);
    const std::vector<int>& home = tree.home_levels();
    const std::vector<int> elevated = elevated_levels(tree);
    const Real half = 0.5 * tree.box_size(leaf_level) * (1.0 + 1e-6);
    const auto& v = space.mesh().vertices();
    const auto& e = space.mesh().edges();

    op::BasisPattern pat;
    pat.group_of_row.assign(sz(std::max<Index>(N, 0)), -1);
    pat.col_ptr.reserve(leaves.size() + 1);
    pat.col_ptr.push_back(0);
    std::vector<Index> anc;
    std::vector<int> hs;  // home levels of the leaf's rows, the leaf level first
    for (std::size_t g = 0; g < leaves.size(); ++g) {
        const Box& leaf = boxes[sz(leaves[g])];
        hs.clear();
        for (Index q = leaf.first_element; q < leaf.first_element + leaf.num_elements; ++q) {
            const Index n = perm[sz(q)];
            const Vec3 mid = 0.5 * (v.row(e(n, 0)) + v.row(e(n, 1))).transpose();
            if (!((mid - leaf.center).cwiseAbs().maxCoeff() <= half)) {
                throw std::invalid_argument(
                    "near_pattern: basis function " + std::to_string(n) +
                    " lies outside its leaf box (octree built on a different space)");
            }
            if (std::find(hs.begin(), hs.end(), home[sz(q)]) == hs.end())
                hs.push_back(home[sz(q)]);
        }
        std::sort(hs.begin(), hs.end(), [](int a, int b) { return a > b; });
        ancestors(tree, leaves[g], anc);
        for (const int h : hs) {
            // Rows of home level h: near to every function whose ancestor on min(h, h(b)) is
            // equal or adjacent to theirs (octree.hpp, home levels).
            const auto group = static_cast<Index>(pat.col_ptr.size() - 1);
            for (Index q = leaf.first_element; q < leaf.first_element + leaf.num_elements; ++q) {
                if (home[sz(q)] == h)
                    pat.group_of_row[sz(perm[sz(q)])] = group;
            }
            const std::size_t start = pat.cols.size();
            // h(b) >= h: the functions of the 27 boxes around the level-h ancestor.
            const auto append_active = [&](const Box& b) {
                for (Index q = b.first_element; q < b.first_element + b.num_elements; ++q) {
                    if (home[sz(q)] >= h)
                        pat.cols.push_back(perm[sz(q)]);
                }
            };
            const Box& H = boxes[sz(anc[sz(h)])];
            append_active(H);
            for (const Index nb : H.near_list) append_active(boxes[sz(nb)]);
            // h(b) = l < h: the elevated functions of the 27 boxes around the level-l ancestor.
            for (const int l : elevated) {
                if (l >= h)
                    break;
                const std::vector<Index>& pos = tree.elevated_positions(l);
                const auto append_elevated = [&](Index b) {
                    const Index first = tree.elevated_first(b);
                    for (Index i = first; i < first + tree.elevated_count(b); ++i)
                        pat.cols.push_back(perm[sz(pos[sz(i)])]);
                };
                const Index a = anc[sz(l)];
                append_elevated(a);
                for (const Index nb : boxes[sz(a)].near_list) append_elevated(nb);
            }
            std::sort(pat.cols.begin() + static_cast<std::ptrdiff_t>(start), pat.cols.end());
            pat.col_ptr.push_back(static_cast<Index>(pat.cols.size()));
        }
    }
    return pat;
}

std::shared_ptr<op::SparseOperator> assemble_near(const op::Problem& p, const Octree& tree) {
    op::validate(p);
    const op::BasisPattern pat = near_pattern(tree, *p.space);
    SBEM_INFO("near field: {} levels, {} leaves, leaf edge {:.4g} m, {} column-list entries",
              tree.levels(), tree.boxes_at_level(tree.leaf_level()).size(),
              tree.box_size(tree.leaf_level()), pat.cols.size());
    return op::assemble_sparse(p, pat);
}

std::size_t estimate_near_bytes(const Octree& tree) {
    const auto& boxes = tree.boxes();
    const int leaf_level = tree.leaf_level();
    const std::vector<int>& home = tree.home_levels();
    const std::vector<int> elevated = elevated_levels(tree);
    std::vector<Index> anc;
    std::vector<std::size_t> rows(sz(tree.levels()), 0);
    std::size_t pairs = 0;
    for (const Index b : tree.boxes_at_level(leaf_level)) {
        const Box& box = boxes[sz(b)];
        std::fill(rows.begin(), rows.end(), std::size_t{0});
        for (Index q = box.first_element; q < box.first_element + box.num_elements; ++q)
            ++rows[sz(home[sz(q)])];
        ancestors(tree, b, anc);
        for (int h = leaf_level; h >= 0; --h) {
            if (rows[sz(h)] == 0)
                continue;
            const Index a = anc[sz(h)];
            std::size_t cols = sz(tree.active_elements(a));
            for (const Index nb : boxes[sz(a)].near_list) cols += sz(tree.active_elements(nb));
            for (const int l : elevated) {
                if (l >= h)
                    break;
                const Index al = anc[sz(l)];
                cols += sz(tree.elevated_count(al));
                for (const Index nb : boxes[sz(al)].near_list) cols += sz(tree.elevated_count(nb));
            }
            pairs += rows[sz(h)] * cols;
        }
    }
    // op::assemble_sparse: 4 entries per pair (JJ, JM, MJ, MM), value + column index, 2N + 1
    // row pointers.
    return 4 * pairs * (sizeof(Complex) + sizeof(Index)) +
           (2 * tree.permutation().size() + 1) * sizeof(Index);
}

}  // namespace specklebem::mlfmm
