#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace specklebem;
using mlfmm::Box;
using mlfmm::Octree;
using mlfmm::OctreeParams;

namespace {

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// Smallest admissible box edge for the given floor (the octree's level criterion).
Real floor_size(const OctreeParams& p, Real lambda) {
    return (1.0 - mlfmm::kMinBoxSizeTolerance) * p.min_box_size_lambda * lambda;
}

std::uint64_t morton(const std::array<Index, 3>& ijk, int bits) {
    std::uint64_t code = 0;
    for (int b = 0; b < bits; ++b) {
        for (std::size_t a = 0; a < 3; ++a) {
            code |= ((static_cast<std::uint64_t>(ijk[a]) >> b) & 1U)
                    << (3 * b + static_cast<int>(a));
        }
    }
    return code;
}

Index chebyshev(const Box& a, const Box& b) {
    Index d = 0;
    for (std::size_t i = 0; i < 3; ++i) d = std::max(d, std::abs(a.ijk[i] - b.ijk[i]));
    return d;
}

bool contains(const std::vector<Index>& sorted, Index v) {
    return std::binary_search(sorted.begin(), sorted.end(), v);
}

Vec3 midpoint(const basis::RwgSpace& space, Index n) {
    const auto& v = space.mesh().vertices();
    const auto& e = space.mesh().edges();
    return 0.5 * (v.row(e(n, 0)) + v.row(e(n, 1))).transpose();
}

/// Permutation, nesting, ranges, Morton order, geometry and the leaf criteria.
void check_structure(const Octree& tree, const basis::RwgSpace& space, Real lambda,
                     const OctreeParams& p) {
    const Index n = space.size();
    const auto& perm = tree.permutation();
    const auto& inv = tree.inverse_permutation();
    REQUIRE(static_cast<Index>(perm.size()) == n);
    REQUIRE(static_cast<Index>(inv.size()) == n);
    std::vector<int> seen(sz(n), 0);
    for (Index i = 0; i < n; ++i) {
        REQUIRE(perm[sz(i)] >= 0);
        REQUIRE(perm[sz(i)] < n);
        ++seen[sz(perm[sz(i)])];
        REQUIRE(inv[sz(perm[sz(i)])] == i);
    }
    REQUIRE(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }));

    // Root: the vertex bounding box, padded and anchored at its lower corner.
    const auto [lo, hi] = space.mesh().bounding_box();
    const Real extent = (hi - lo).maxCoeff();
    CHECK(tree.box_size(0) >= extent);
    CHECK(tree.box_size(0) <= extent * (1.0 + 1e-5));
    const Vec3 root_min = tree.root_center() - Vec3::Constant(0.5 * tree.box_size(0));
    CHECK((root_min - lo).cwiseAbs().maxCoeff() <= 1e-5 * extent);
    CHECK((root_min - lo).maxCoeff() <= 0.0);

    const auto& boxes = tree.boxes();
    REQUIRE(boxes[0].parent == -1);
    REQUIRE(boxes[0].num_elements == n);
    for (int l = 0; l < tree.levels(); ++l) {
        const auto& ids = tree.boxes_at_level(l);
        const Real s = tree.box_size(l);
        if (l >= 1)
            CHECK(s >= floor_size(p, lambda));
        Index next = 0;
        std::uint64_t prev_code = 0;
        for (std::size_t q = 0; q < ids.size(); ++q) {
            const Box& b = boxes[sz(ids[q])];
            REQUIRE(b.level == l);
            REQUIRE(b.num_elements > 0);
            REQUIRE(b.first_element == next);  // level ranges partition [0, n) in order
            next += b.num_elements;
            const std::uint64_t code = morton(b.ijk, l);
            if (q > 0)
                REQUIRE(code > prev_code);  // Morton order strictly monotone
            prev_code = code;
            REQUIRE(tree.find_box(l, b.ijk) == ids[q]);
            for (Index e = b.first_element; e < b.first_element + b.num_elements; ++e) {
                const Vec3 d = midpoint(space, perm[sz(e)]) - b.center;
                REQUIRE(d.cwiseAbs().maxCoeff() <= 0.5 * s * (1.0 + 1e-9));
            }
            // Children partition the range, in slot order, with consistent coordinates.
            Index child_next = b.first_element;
            int num_children = 0;
            for (std::size_t slot = 0; slot < 8; ++slot) {
                const Index c = b.children[slot];
                if (c < 0)
                    continue;
                ++num_children;
                const Box& ch = boxes[sz(c)];
                REQUIRE(ch.parent == ids[q]);
                REQUIRE(ch.level == l + 1);
                REQUIRE(ch.first_element == child_next);
                child_next += ch.num_elements;
                for (std::size_t a = 0; a < 3; ++a) {
                    REQUIRE(ch.ijk[a] == 2 * b.ijk[a] + static_cast<Index>((slot >> a) & 1U));
                }
            }
            if (l == tree.leaf_level()) {
                REQUIRE(num_children == 0);
            } else {
                REQUIRE(num_children > 0);
                REQUIRE(child_next == b.first_element + b.num_elements);
            }
        }
        REQUIRE(next == n);
    }
    // Leaf criterion: leaves respect the limit unless the lambda bound or max_levels stopped
    // the split; the leaf level is the first one meeting the limit.
    const int leaf = tree.leaf_level();
    const bool can_split =
        leaf + 1 < p.max_levels && tree.box_size(leaf) / 2 >= floor_size(p, lambda);
    Index leaf_max = 0;
    for (const Index b : tree.boxes_at_level(leaf)) {
        leaf_max = std::max(leaf_max, boxes[sz(b)].num_elements);
    }
    if (can_split)
        CHECK(leaf_max <= p.max_elements_per_leaf);
    if (leaf > 0) {
        Index above_max = 0;
        for (const Index b : tree.boxes_at_level(leaf - 1)) {
            above_max = std::max(above_max, boxes[sz(b)].num_elements);
        }
        CHECK(above_max > p.max_elements_per_leaf);
    }
}

/// Near / interaction list properties, an independent O(B^2) classification of every
/// same-level box pair from ijk, and the brute-force completeness over all leaf pairs.
/// Returns the number of leaf pairs checked.
std::size_t check_lists(const Octree& tree) {
    const auto& boxes = tree.boxes();
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const Box& b = boxes[i];
        const auto self = static_cast<Index>(i);
        REQUIRE(std::is_sorted(b.near_list.begin(), b.near_list.end()));
        REQUIRE(std::is_sorted(b.interaction_list.begin(), b.interaction_list.end()));
        for (const Index nb : b.near_list) {
            REQUIRE(nb != self);
            REQUIRE(contains(boxes[sz(nb)].near_list, self));  // symmetric
        }
        REQUIRE(b.interaction_list.size() <= 189);
        if (b.level < 2)
            REQUIRE(b.interaction_list.empty());
        for (const Index f : b.interaction_list) {
            REQUIRE(contains(boxes[sz(f)].interaction_list, self));  // symmetric
        }
    }
    // Expected lists from the integer coordinates alone: near = Chebyshev distance 1,
    // interaction = distance > 1 with parents at distance <= 1 (same or adjacent parent).
    // The level's boxes are in ascending index order, so the expected lists come out sorted.
    for (int l = 0; l < tree.levels(); ++l) {
        const auto& ids = tree.boxes_at_level(l);
        for (const Index a : ids) {
            const Box& ba = boxes[sz(a)];
            std::vector<Index> near, inter;
            for (const Index c : ids) {
                const Box& bc = boxes[sz(c)];
                const Index d = chebyshev(ba, bc);
                if (d == 1) {
                    near.push_back(c);
                } else if (d > 1 && l >= 2 &&
                           chebyshev(boxes[sz(ba.parent)], boxes[sz(bc.parent)]) <= 1) {
                    inter.push_back(c);
                }
            }
            REQUIRE(ba.near_list == near);
            REQUIRE(ba.interaction_list == inter);
        }
    }
    // Completeness: each leaf pair is near or translated at exactly one level.
    const int leaf = tree.leaf_level();
    const auto& leaves = tree.boxes_at_level(leaf);
    std::vector<std::vector<Index>> anc(leaves.size(), std::vector<Index>(sz(leaf + 1)));
    for (std::size_t a = 0; a < leaves.size(); ++a) {
        Index b = leaves[a];
        for (int l = leaf; l >= 0; --l) {
            anc[a][sz(l)] = b;
            b = boxes[sz(b)].parent;
        }
    }
    std::size_t pairs = 0, bad = 0;
    for (std::size_t a = 0; a < leaves.size(); ++a) {
        for (std::size_t c = 0; c < leaves.size(); ++c) {
            int count = (a == c || contains(boxes[sz(leaves[a])].near_list, leaves[c])) ? 1 : 0;
            for (int l = 2; l <= leaf; ++l) {
                if (contains(boxes[sz(anc[a][sz(l)])].interaction_list, anc[c][sz(l)]))
                    ++count;
            }
            if (count != 1)
                ++bad;
            ++pairs;
        }
    }
    CHECK(bad == 0);
    return pairs;
}

geometry::TriangleMesh rough_box(Real depth) {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = 2e-6;
    p.rms_roughness = 50e-9;
    p.correlation_length = 200e-9;
    p.mesh_size = 100e-9;
    p.seed = 20261009;
    p.box_depth = depth;
    return geometry::make_rough_surface_mesh(p);
}

/// Open square plate [0, side]^2 in the plane z = 0 (m x m cells, two triangles each); with
/// `jitter` the vertices alternate between z = +1e-15 and -1e-15 m.
geometry::TriangleMesh plate(Real side, int m, bool jitter) {
    const Index row = m + 1;
    Vertices v(row * row, 3);
    for (Index j = 0; j < row; ++j) {
        for (Index i = 0; i < row; ++i) {
            const Index r = j * row + i;
            v(r, 0) = side * static_cast<Real>(i) / m;
            v(r, 1) = side * static_cast<Real>(j) / m;
            v(r, 2) = jitter ? (((i + j) % 2 != 0) ? 1e-15 : -1e-15) : 0.0;
        }
    }
    Triangles t(2 * Index{m} * m, 3);
    Index k = 0;
    for (Index j = 0; j < m; ++j) {
        for (Index i = 0; i < m; ++i) {
            const Index a = j * row + i;
            t.row(k++) << a, a + 1, a + row + 1;
            t.row(k++) << a, a + row + 1, a + row;
        }
    }
    return geometry::TriangleMesh(v, t);
}

/// Number of leading levels on which every box lies in the bottom layer (ijk[2] == 0).
int flat_levels(const Octree& tree) {
    for (int l = 0; l < tree.levels(); ++l) {
        for (const Index b : tree.boxes_at_level(l)) {
            if (tree.boxes()[sz(b)].ijk[2] != 0)
                return l;
        }
    }
    return tree.levels();
}

}  // namespace

TEST_CASE("Octree: structure on icospheres, a thin rough box and flat plates", "[octree]") {
    struct Case {
        std::string label;
        geometry::TriangleMesh mesh;
        Real lambda;
        OctreeParams p;
        int min_flat_levels;  ///< leading levels expected one box layer thick
    };
    std::vector<Case> cases;
#ifdef NDEBUG
    const int fine = 4;  // sphere at lambda = 100 nm: 7680 basis functions
#else
    const int fine = 3;  // unoptimised builds (time budget): 1920 basis functions
#endif
    cases.push_back({"sphere n=3", geometry::make_icosphere(1e-6, 3), 500e-9, OctreeParams{}, 1});
    cases.push_back({"sphere fine", geometry::make_icosphere(1e-6, fine), 100e-9,
                     OctreeParams{32, 12, 0.25}, 1});
    // Thin box (z range < 0.5 um, root 2 um): anchored at the bottom, levels 0..2 are flat.
    cases.push_back({"rough box", rough_box(0.3e-6), 500e-9, OctreeParams{50, 12, 0.25}, 3});
    // Plates: zero extent in z, and +-1e-15 m jitter; leaves are one layer thick.
    cases.push_back({"plate", plate(1e-6, 24, false), 500e-9, OctreeParams{8, 12, 0.0}, 0});
    cases.push_back({"plate jitter", plate(1e-6, 24, true), 500e-9, OctreeParams{8, 12, 0.0}, 0});
    std::vector<std::array<Index, 3>> flat_plate_leaves;
    for (const Case& c : cases) {
        INFO(c.label);
        const basis::RwgSpace space(c.mesh);
        const Octree tree(space, c.lambda, c.p);
        CHECK(tree.levels() >= 3);
        check_structure(tree, space, c.lambda, c.p);
        const std::size_t pairs = check_lists(tree);
        CHECK(pairs > 0);
        if (c.min_flat_levels > 0) {
            CHECK(flat_levels(tree) >= c.min_flat_levels);
            continue;
        }
        // Plates: every level one layer thick, 4^l boxes at most, and the same leaves with
        // and without jitter.
        CHECK(flat_levels(tree) == tree.levels());
        for (int l = 0; l < tree.levels(); ++l) {
            CHECK(tree.boxes_at_level(l).size() <= (std::size_t{1} << (2 * l)));
        }
        std::vector<std::array<Index, 3>> leaves;
        for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
            leaves.push_back(tree.boxes()[sz(b)].ijk);
        }
        if (flat_plate_leaves.empty()) {
            flat_plate_leaves = leaves;
        } else {
            CHECK(leaves == flat_plate_leaves);
        }
    }
}

TEST_CASE("Octree: list completeness on deep trees", "[octree]") {
    const auto sphere = geometry::make_icosphere(1e-6, 3);
    const basis::RwgSpace space(sphere);
    // Small leaves without the lambda bound: 5 levels, ~800 leaves (~6.6e5 pairs).
    const OctreeParams p{6, 12, 0.0};
    const Octree tree(space, 500e-9, p);
    CHECK(tree.levels() >= 5);
    check_structure(tree, space, 500e-9, p);
    const auto leaves = tree.boxes_at_level(tree.leaf_level()).size();
    CHECK(check_lists(tree) == leaves * leaves);
    // A surface fills only part of the 6x6x6 parent neighbourhood (~61 at most here).
    std::size_t largest = 0;
    for (const Box& b : tree.boxes()) largest = std::max(largest, b.interaction_list.size());
    CHECK(largest > 26);
}

TEST_CASE("Octree: lambda bound and max_levels stop the subdivision", "[octree]") {
    const auto sphere = geometry::make_icosphere(1e-6, 3);
    const basis::RwgSpace space(sphere);
    const Real lambda = 500e-9;
    const OctreeParams bounded{1, 12, 0.25};
    const Octree tree(space, lambda, bounded);
    const Real leaf = tree.box_size(tree.leaf_level());
    CHECK(leaf >= floor_size(bounded, lambda));
    CHECK(leaf / 2 < floor_size(bounded, lambda));  // the bound, not the element limit, stopped
    check_structure(tree, space, lambda, bounded);

    const Octree capped(space, lambda, OctreeParams{1, 3, 0.0});
    CHECK(capped.levels() == 3);
    check_structure(capped, space, lambda, OctreeParams{1, 3, 0.0});

    const Octree root_only(space, lambda, OctreeParams{1, 1, 0.0});
    REQUIRE(root_only.levels() == 1);
    CHECK(root_only.boxes().size() == 1);
    CHECK(root_only.boxes()[0].near_list.empty());
    CHECK(root_only.boxes()[0].num_elements == space.size());
}

TEST_CASE("Octree: 4 um sphere at 500 nm reaches lambda/4 leaves", "[octree]") {
    // Regression: the root (vertex bounding cube of a 4 um sphere) is 8 lambda up to the
    // padding, so the lambda/4 level (8 lambda / 2^5) must be admitted: 6 levels.
    const auto sphere = geometry::make_icosphere(2e-6, 4);
    const basis::RwgSpace space(sphere);
    const Real lambda = 500e-9;
    const OctreeParams p{4, 12, 0.25};
    const Octree tree(space, lambda, p);
    CHECK(tree.box_size(0) / lambda >= 8.0);
    CHECK(tree.box_size(0) / lambda <= 8.0 * (1.0 + 1e-5));
    CHECK(tree.levels() == 6);
    const Real leaf = tree.box_size(tree.leaf_level());
    CHECK(leaf >= floor_size(p, lambda));
    CHECK(leaf <= 0.25 * lambda * (1.0 + 1e-5));
    check_structure(tree, space, lambda, p);
    Index leaf_max = 0;
    for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
        leaf_max = std::max(leaf_max, tree.boxes()[sz(b)].num_elements);
    }
    CHECK(leaf_max <= 4);
}

TEST_CASE("Octree: invalid input and summary", "[octree]") {
    const auto sphere = geometry::make_icosphere(1e-6, 1);
    const basis::RwgSpace space(sphere);
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    for (const Real w : {0.0, -1e-6, nan, inf}) {
        CHECK_THROWS_AS(Octree(space, w, OctreeParams{}), std::invalid_argument);
    }
    for (const OctreeParams& p :
         {OctreeParams{0, 12, 0.25}, OctreeParams{10, 0, 0.25}, OctreeParams{10, 22, 0.25},
          OctreeParams{10, 12, -0.1}, OctreeParams{10, 12, nan}}) {
        CHECK_THROWS_AS(Octree(space, 500e-9, p), std::invalid_argument);
    }
    Vertices v(3, 3);
    v << 0, 0, 0, 1, 0, 0, 0, 1, 0;
    Triangles t(1, 3);
    t << 0, 1, 2;
    const geometry::TriangleMesh single(v, t);
    const basis::RwgSpace empty(single);
    CHECK_THROWS_AS(Octree(empty, 500e-9, OctreeParams{}), std::invalid_argument);

    const Octree tree(space, 100e-9, OctreeParams{4, 12, 0.25});
    CHECK_THROWS_AS(tree.boxes_at_level(tree.levels()), std::out_of_range);
    CHECK_THROWS_AS(tree.box_size(-1), std::out_of_range);
    CHECK(tree.find_box(1, {-1, 0, 0}) == -1);
    CHECK(tree.find_box(1, {2, 0, 0}) == -1);
    CHECK_THAT(tree.summary(),
               Catch::Matchers::ContainsSubstring(std::to_string(tree.levels()) + " levels"));
    CHECK_THAT(tree.summary(), Catch::Matchers::ContainsSubstring("elements per leaf"));
}

TEST_CASE("Octree: build time for ~1e5 basis functions", "[octree]") {
#ifdef NDEBUG
    const int subdivisions = 6;  // 122,880 basis functions
#else
    const int subdivisions = 4;  // unoptimised builds: smoke test without a time bound
#endif
    const auto sphere = geometry::make_icosphere(2e-6, subdivisions);
    const basis::RwgSpace space(sphere);
    const auto t0 = std::chrono::steady_clock::now();
    const Octree tree(space, 500e-9, OctreeParams{});
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    WARN("octree build, N = " << space.size() << ": " << seconds << " s\n" << tree.summary());
    CHECK(tree.permutation().size() == static_cast<std::size_t>(space.size()));
#ifdef NDEBUG
    CHECK(tree.levels() == 6);  // 8 lambda root down to lambda/4 leaves
    CHECK(seconds <= 1.0);
#endif
}
