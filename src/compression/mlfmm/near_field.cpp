/// @file near_field.cpp
/// Near pattern of the octree leaves and Z_near assembly (ADR 0008 §1, WP19a).
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
    const std::vector<Index>& leaves = tree.boxes_at_level(tree.leaf_level());
    const Real half = 0.5 * tree.box_size(tree.leaf_level()) * (1.0 + 1e-6);
    const auto& v = space.mesh().vertices();
    const auto& e = space.mesh().edges();

    op::BasisPattern pat;
    pat.group_of_row.assign(sz(std::max<Index>(N, 0)), -1);
    pat.col_ptr.reserve(leaves.size() + 1);
    pat.col_ptr.push_back(0);
    for (std::size_t g = 0; g < leaves.size(); ++g) {
        const Box& leaf = boxes[sz(leaves[g])];
        for (Index q = leaf.first_element; q < leaf.first_element + leaf.num_elements; ++q) {
            const Index n = perm[sz(q)];
            const Vec3 mid = 0.5 * (v.row(e(n, 0)) + v.row(e(n, 1))).transpose();
            if (!((mid - leaf.center).cwiseAbs().maxCoeff() <= half)) {
                throw std::invalid_argument(
                    "near_pattern: basis function " + std::to_string(n) +
                    " lies outside its leaf box (octree built on a different space)");
            }
            pat.group_of_row[sz(n)] = static_cast<Index>(g);
        }
        const std::size_t start = pat.cols.size();
        const auto append = [&](const Box& b) {
            for (Index q = b.first_element; q < b.first_element + b.num_elements; ++q)
                pat.cols.push_back(perm[sz(q)]);
        };
        append(leaf);
        for (const Index nb : leaf.near_list) append(boxes[sz(nb)]);
        std::sort(pat.cols.begin() + static_cast<std::ptrdiff_t>(start), pat.cols.end());
        pat.col_ptr.push_back(static_cast<Index>(pat.cols.size()));
    }
    return pat;
}

std::shared_ptr<op::SparseOperator> assemble_near(const op::Problem& p, const Octree& tree) {
    op::validate(p);
    const op::BasisPattern pat = near_pattern(tree, *p.space);
    SBEM_INFO("near field: {} levels, {} leaves, leaf edge {:.4g} m, {} column-list entries",
              tree.levels(), pat.col_ptr.size() - 1, tree.box_size(tree.leaf_level()),
              pat.cols.size());
    return op::assemble_sparse(p, pat);
}

}  // namespace specklebem::mlfmm
