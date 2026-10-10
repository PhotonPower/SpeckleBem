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
///
/// **Home levels** (local leaf rule, ADR 0008 amendment 2026-10-11). Every basis function b has a
/// home level h(b) <= D and a home box = the ancestor of its leaf on level h(b). Without support
/// radii (first constructor) every home level is the leaf level D. With support radii r(b) and a
/// largest support ratio rho (MlfmmOperator: max_support_ratio(d0)), h(b) = D if r(b) <= rho a_D,
/// otherwise the finest level l < D with r(b) <= kElevatedSupportRatioFactor rho a_l (a_l =
/// box_size(l), relative tolerance 1e-12; the constant explains why levels above the leaf need
/// half the leaf ratio), or 0 if no level qualifies (the far operator rejects home levels < 2 on
/// trees with >= 3 levels). Functions with
/// h(b) < D are *elevated*: they take part in the expansions from their home box upwards only,
/// and a pair (a, b) is near iff the ancestors on min(h(a), h(b)) coincide or touch; otherwise
/// exactly one level l <= min(h(a), h(b)), l >= 2, holds the two ancestors in each other's
/// interaction lists (the leaf-pair statement above, applied on level min(h(a), h(b))). The box
/// ranges [first_element, ...) still hold every function by its midpoint, elevated ones included;
/// active_elements(box) counts those with h >= the box's level.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/types.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace specklebem::mlfmm {

/// Relative tolerance of the leaf-size floor: a level is admitted if its box edge is
/// >= (1 - kMinBoxSizeTolerance) * min_box_size_lambda * lambda. The floor only guards
/// against the gradual low-frequency breakdown (ADR 0008 derives the truncation from the
/// actual box diagonal and checks the accuracy per level), so it need not be exact; the
/// tolerance keeps a root that rounds to just below 2^D * floor from losing its last level.
inline constexpr Real kMinBoxSizeTolerance = 1e-2;

/// Relative tolerance of the home-level test r(b) <= rho a_l (keeps a function whose radius sets
/// the leaf floor exactly on the leaf level despite rounding).
inline constexpr Real kHomeLevelTolerance = 1e-12;

/// Above the leaf level a basis function needs r(b) <= kElevatedSupportRatioFactor x rho x a_l
/// (rho the leaf ratio): the support ratio of the leaf's parent level in a tree built with the
/// global leaf rule (r_max / a_leaf <= rho, hence <= rho / 2 one level up), the validated
/// configuration for a level that also receives interpolated child fields. With the leaf ratio rho
/// on such a level (the ADR 0008 amendment of 2026-10-11 as written) the level's order grows to
/// ~1.5 k r_min (r_min the nearest translation distance), and the translators amplify the
/// interpolation error of the children's fields far beyond 10^-d0 (WP21L: O(1) errors of the Si
/// interior at a 2.3 lambda_2 level; the leaf level is not affected because its patterns are
/// exact samples).
inline constexpr Real kElevatedSupportRatioFactor = 0.5;

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
    /// The same tree with home levels from per-basis support radii (file comment).
    /// @param support_radii r(b) per basis index (size N, finite, >= 0), e.g. support_radii()
    /// @param max_support_ratio rho > 0 (finite): the leaf ratio rho (file comment)
    /// @throws std::invalid_argument as the first constructor, or for invalid radii / ratio.
    Octree(const basis::RwgSpace& space, Real wavelength, const OctreeParams& p,
           std::span<const Real> support_radii, Real max_support_ratio);

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

    /// Home level of the basis function at permuted position p (file comment).
    [[nodiscard]] int home_level(Index p) const { return home_[static_cast<std::size_t>(p)]; }
    /// Home levels by permuted position (size N).
    [[nodiscard]] const std::vector<int>& home_levels() const { return home_; }
    /// True if some basis function has a home level above the leaf level.
    [[nodiscard]] bool has_elevated() const { return elevated_total_ > 0; }
    /// Number of basis functions per home level (size levels(); the last entry is the leaf level).
    [[nodiscard]] std::vector<Index> home_level_counts() const;
    /// Permuted positions of the basis functions with home level `level` < leaf_level(),
    /// ascending (hence grouped by home box in Morton order); empty for the leaf level.
    /// @throws std::out_of_range for an invalid level
    [[nodiscard]] const std::vector<Index>& elevated_positions(int level) const;
    /// The elevated functions whose home box is `box`: elevated_positions(level of box)[first ...
    /// first + count) with first = elevated_first(box), count = elevated_count(box).
    [[nodiscard]] Index elevated_first(Index box) const {
        return elevated_first_[static_cast<std::size_t>(box)];
    }
    [[nodiscard]] Index elevated_count(Index box) const {
        return elevated_count_[static_cast<std::size_t>(box)];
    }
    /// Basis functions of the box with home level >= the box's level (those that take part in
    /// the box's expansion); num_elements without elevated functions.
    [[nodiscard]] Index active_elements(Index box) const {
        return active_[static_cast<std::size_t>(box)];
    }

    /// Multi-line text: levels, boxes, box size and list sizes per level, elements per leaf.
    [[nodiscard]] std::string summary() const;

private:
    void build(const basis::RwgSpace& space);
    void build_lists();
    void check_level(int level) const;
    /// Home levels, elevated lists and active counts (radii empty: every function at the leaf).
    void assign_home_levels(std::span<const Real> radii, Real ratio);

    OctreeParams params_;
    Real wavelength_ = 0;
    Real root_size_ = 0;
    Vec3 root_center_ = Vec3::Zero();
    int levels_ = 0;
    std::vector<Box> boxes_;
    std::vector<std::vector<Index>> level_boxes_;
    std::vector<std::vector<std::uint64_t>> level_codes_;  ///< Morton codes per level, sorted
    std::vector<Index> perm_, inv_perm_;
    std::vector<int> home_;                     ///< home level by permuted position
    std::vector<std::vector<Index>> elevated_;  ///< per level: elevated positions
    std::vector<Index> elevated_first_, elevated_count_, active_;  ///< per box
    Index elevated_total_ = 0;
};

}  // namespace specklebem::mlfmm
