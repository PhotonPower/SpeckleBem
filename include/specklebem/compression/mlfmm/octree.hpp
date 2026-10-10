#pragma once
/// @file octree.hpp
/// Octree over the RWG basis functions for the MLFMM (docs/04_theory_mlfmm.md "Data
/// structures", ADR 0008 §1), data layout after Gumerov, Duraiswami & Borovikova (2003).
///
/// Elements are the RWG functions, positioned at their edge midpoints. The root is the cube
/// built from the mesh vertex bounding box [lo, hi] (geometry::TriangleMesh::bounding_box()):
/// edge s_0 = e (1 + 2e-6) with e = max(hi - lo), anchored at the lower corner
/// root_min = lo - 1e-6 e (padded so that no midpoint lies on its boundary; anchoring instead
/// of centring keeps flat or thin geometry in one layer of boxes). Box (level l, integer
/// coordinates ijk in [0, 2^l)^3) has edge s_l = s_0 / 2^l and centre
/// root_min + (ijk + 1/2) s_l; a point on a box face shared with a higher-index box belongs
/// to that box, points on the upper root faces to the last box.
///
/// **Uniform depth.** All leaves lie on the finest level D = levels() - 1, so the level-wise
/// interaction lists of the MLFMM passes need no adaptive (U/V/W/X) lists. D is the first
/// level at which every box holds <= max_elements_per_leaf elements, unless a further split
/// would make the box edge smaller than (1 - kMinBoxSizeTolerance) * min_box_size_lambda *
/// lambda or exceed max_levels; then D is the deepest admissible level and leaves may hold
/// more elements. Only non-empty boxes are created.
///
/// **Order.** Boxes are stored level by level (root first), each level in Morton order (bits
/// interleaved x, y, z with x lowest; child slot = (i & 1) | (j & 1) << 1 | (k & 1) << 2).
/// The elements are permuted into Morton order of their leaf (ties by basis index), so every
/// box owns a contiguous range [first_element, first_element + num_elements) of the permuted
/// list and the children of a box partition its range.
///
/// **Lists** (same level, sorted by box index, i.e. Morton order):
///  * near_list: boxes whose ijk differ by at most 1 per axis, **excluding the box itself**;
///    the self-interaction is near as well (WP19 assembles leaf-self and leaf-near_list
///    blocks exactly). Filled on every level.
///  * interaction_list: children of the parent's near_list that are not near the box
///    (<= 189; the parent's own children are always near). Empty on levels 0 and 1 (every
///    pair is near there).
/// For two leaves A, B exactly one holds: A == B or B in A.near_list, or there is exactly one
/// level l >= 2 at which ancestor(B, l) is in ancestor(A, l).interaction_list.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace specklebem::mlfmm {

/// Relative tolerance of the leaf-size floor: a level is admitted if its box edge is
/// >= (1 - kMinBoxSizeTolerance) * min_box_size_lambda * lambda. The floor only guards
/// against the gradual low-frequency breakdown (ADR 0008 derives the truncation from the
/// actual box diagonal and checks the accuracy per level), so it need not be exact; the
/// tolerance keeps a root that rounds to just below 2^D * floor from losing its last level.
inline constexpr Real kMinBoxSizeTolerance = 1e-2;

/// Largest OctreeParams::max_levels: the finest level 20 needs 3 x 20 bits of a 64-bit Morton
/// code.
inline constexpr int kMaxOctreeLevels = 21;

struct Box {
    Index parent = -1;
    std::array<Index, 8> children = {-1, -1, -1, -1, -1, -1, -1, -1};  ///< by child slot
    int level = 0;
    std::array<Index, 3> ijk = {0, 0, 0};  ///< integer coordinates on its level
    Vec3 center = Vec3::Zero();
    Index first_element = 0;  ///< range into the permuted element list
    Index num_elements = 0;
    std::vector<Index> near_list;         ///< adjacent boxes of the same level (not itself)
    std::vector<Index> interaction_list;  ///< well-separated boxes, parents near
};

struct OctreeParams {
    int max_elements_per_leaf = 100;  ///< >= 1
    int max_levels = 12;              ///< number of levels incl. the root, in [1, kMaxOctreeLevels]
    /// Leaf edge floor in wavelengths (>= 0): no level with a box edge below
    /// (1 - kMinBoxSizeTolerance) * min_box_size_lambda * lambda.
    Real min_box_size_lambda = 0.25;
};

class Octree {
public:
    /// @param wavelength lambda of the region that bounds the leaf size [m]
    /// @throws std::invalid_argument for an empty space, a non-positive or non-finite
    ///         wavelength, invalid params or non-finite vertex coordinates.
    Octree(const basis::RwgSpace& space, Real wavelength, const OctreeParams& p);

    /// Number of levels including the root; the leaves are on level levels() - 1.
    [[nodiscard]] int levels() const { return levels_; }
    [[nodiscard]] int leaf_level() const { return levels_ - 1; }
    [[nodiscard]] const std::vector<Box>& boxes() const { return boxes_; }
    /// Box indices of one level in Morton order. @throws std::out_of_range
    [[nodiscard]] const std::vector<Index>& boxes_at_level(int level) const;
    /// Box edge on a level [m]. @throws std::out_of_range
    [[nodiscard]] Real box_size(int level) const;
    [[nodiscard]] const Vec3& root_center() const { return root_center_; }
    [[nodiscard]] Real wavelength() const { return wavelength_; }
    /// Box of the given level and integer coordinates, -1 if empty (absent).
    /// @throws std::out_of_range for an invalid level
    [[nodiscard]] Index find_box(int level, const std::array<Index, 3>& ijk) const;
    /// permutation()[p] = basis index at permuted position p; inverse_permutation()[n] = p.
    [[nodiscard]] const std::vector<Index>& permutation() const { return perm_; }
    [[nodiscard]] const std::vector<Index>& inverse_permutation() const { return inv_perm_; }

    /// Multi-line text: levels, boxes, box size and list sizes per level, elements per leaf.
    [[nodiscard]] std::string summary() const;

private:
    void build(const basis::RwgSpace& space);
    void build_lists();
    void check_level(int level) const;

    OctreeParams params_;
    Real wavelength_ = 0;
    Real root_size_ = 0;
    Vec3 root_center_ = Vec3::Zero();
    int levels_ = 0;
    std::vector<Box> boxes_;
    std::vector<std::vector<Index>> level_boxes_;
    std::vector<std::vector<std::uint64_t>> level_codes_;  ///< Morton codes per level, sorted
    std::vector<Index> perm_, inv_perm_;
};

}  // namespace specklebem::mlfmm
