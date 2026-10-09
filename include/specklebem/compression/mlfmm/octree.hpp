#pragma once
/// @file octree.hpp
/// Adaptive octree over the RWG basis functions (grouped by edge midpoints),
/// following Gumerov, Duraiswami & Borovikova (2003). The leaf size is chosen
/// so that the smallest box holds ~100 elements (paper: 120 elements, 5 levels
/// for the 4 um sphere).
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/types.hpp"

#include <vector>

namespace specklebem::mlfmm {

struct Box {
    Index parent = -1;
    Index children[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int level = 0;
    Vec3 center;
    Index first_element = 0;  ///< range into the permuted element list
    Index num_elements = 0;
    std::vector<Index> near_list;  ///< adjacent boxes (direct interaction)
    std::vector<Index>
        interaction_list;  ///< well-separated boxes with the same parent neighbourhood
};

struct OctreeParams {
    int max_elements_per_leaf = 100;
    int max_levels = 12;
    Real min_box_size_lambda = 0.25;  ///< do not subdivide below ~lambda/4
};

class Octree {
public:
    Octree(const basis::RwgSpace& space, Real wavelength, const OctreeParams& p);

    [[nodiscard]] int levels() const { return levels_; }
    [[nodiscard]] const std::vector<Box>& boxes() const { return boxes_; }
    [[nodiscard]] const std::vector<Index>& boxes_at_level(int level) const {
        return level_boxes_[static_cast<std::size_t>(level)];
    }
    [[nodiscard]] Real box_size(int level) const;
    /// Element permutation (Morton order) and its inverse.
    [[nodiscard]] const std::vector<Index>& permutation() const { return perm_; }
    [[nodiscard]] const std::vector<Index>& inverse_permutation() const { return inv_perm_; }

    [[nodiscard]] std::string summary() const;

private:
    void build(const basis::RwgSpace& space);
    void build_lists();

    OctreeParams params_;
    Real root_size_ = 0;
    Vec3 root_center_;
    int levels_ = 0;
    std::vector<Box> boxes_;
    std::vector<std::vector<Index>> level_boxes_;
    std::vector<Index> perm_, inv_perm_;
};

}  // namespace specklebem::mlfmm
