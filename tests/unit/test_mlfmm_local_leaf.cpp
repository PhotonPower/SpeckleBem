/// Unit tests of the local leaf rule of the MLFMM (WP21L, ADR 0008 amendment 2026-10-11): home
/// levels of the octree, the near pattern with elevated basis functions and its estimate, the
/// partition of all basis pairs into near / expansion / exact / truncated, the bitwise identity
/// of leaf_radius_quantile = 1 with the global leaf rule, and the accuracy of columns and rows of
/// elevated functions against exact entries. lambda0 = 500 nm.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/region_sparse_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace specklebem;
using formulation::Kind;

namespace {

constexpr Real kLambda = 500e-9;

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// Mesh, space and Problem (vacuum exterior). Not movable.
struct Setup {
    Setup(geometry::TriangleMesh m, const material::Material& object, Kind kind)
        : mesh(std::move(m)), space(mesh), form(formulation::make_formulation(kind)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = object;
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
    }
    Setup(const Setup&) = delete;
    Setup& operator=(const Setup&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
};

/// Rough box with the graded walls of ADR 0006: top 50 nm, coarse box cells 100 nm (lambda1 / 5),
/// sigma 50 nm, Lc 500 nm, fixed seed. The coarse cells (support radii 112-141 nm) are the
/// elevated functions of the local leaf rule; the top face has radii of ~56-70 nm.
geometry::TriangleMesh graded_box(Real size, Real depth) {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = size;
    p.rms_roughness = 50e-9;
    p.correlation_length = 500e-9;
    p.mesh_size = 50e-9;
    p.box_depth = depth;
    p.box_mesh_size = 100e-9;
    p.exterior_wavelength = kLambda;
    p.seed = 1;
    return geometry::make_rough_surface_mesh(p);
}

/// Open plate in z = 0 on a tensor grid of spacing h with two wide columns (widths w1 h and
/// w2 h): x = `cells` cells of h, one of w1 h, `cells` of h, one of w2 h, `cells` of h; y = `rows`
/// cells of h. The long triangles of a column of width w h have support radii ~sqrt(w^2 + 1/4) h,
/// the others <= 1.12 h. Default: 6 / 2 / 4 / 24 rows, 24 h x 24 h.
geometry::TriangleMesh strip_plate(Real h, int cells = 6, Real w1 = 2.0, Real w2 = 4.0,
                                   Index rows = 24) {
    std::vector<Real> xs = {0.0};
    const auto add = [&](int count, Real w) {
        for (int i = 0; i < count; ++i) xs.push_back(xs.back() + w);
    };
    add(cells, h);
    add(1, w1 * h);
    add(cells, h);
    add(1, w2 * h);
    add(cells, h);
    const auto nx = static_cast<Index>(xs.size());
    const Index ny = rows + 1;
    Vertices v(nx * ny, 3);
    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            v(j * nx + i, 0) = xs[sz(i)];
            v(j * nx + i, 1) = h * static_cast<Real>(j);
            v(j * nx + i, 2) = 0.0;
        }
    }
    Triangles t(2 * (nx - 1) * (ny - 1), 3);
    Index k = 0;
    for (Index j = 0; j + 1 < ny; ++j) {
        for (Index i = 0; i + 1 < nx; ++i) {
            const Index a = j * nx + i;
            t.row(k++) << a, a + 1, a + nx + 1;
            t.row(k++) << a, a + nx + 1, a + nx;
        }
    }
    return geometry::TriangleMesh(v, t);
}

kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = 1;
    opt.quad_degree_near = 2;
    opt.quad_degree_sing = 2;
    return opt;
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

/// anc[n][l] = box of basis n's leaf ancestor on level l.
std::vector<std::vector<Index>> ancestors(const mlfmm::Octree& tree) {
    std::vector<std::vector<Index>> anc(tree.permutation().size(),
                                        std::vector<Index>(sz(tree.levels()), -1));
    for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
        const mlfmm::Box& box = tree.boxes()[sz(b)];
        for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
            std::vector<Index>& a = anc[sz(tree.permutation()[sz(p)])];
            a[sz(tree.leaf_level())] = b;
            for (int l = tree.leaf_level() - 1; l >= 0; --l)
                a[sz(l)] = tree.boxes()[sz(a[sz(l + 1)])].parent;
        }
    }
    return anc;
}

bool touching(const mlfmm::Box& a, const mlfmm::Box& b) {
    return std::abs(a.ijk[0] - b.ijk[0]) <= 1 && std::abs(a.ijk[1] - b.ijk[1]) <= 1 &&
           std::abs(a.ijk[2] - b.ijk[2]) <= 1;
}

/// Home level by basis index.
std::vector<int> home_by_basis(const mlfmm::Octree& tree) {
    std::vector<int> h(tree.permutation().size());
    for (std::size_t p = 0; p < h.size(); ++p)
        h[sz(tree.permutation()[p])] = tree.home_level(static_cast<Index>(p));
    return h;
}

/// Pattern entry (row, col) present?
bool in_pattern(const op::BasisPattern& pat, Index row, Index col) {
    const auto g = sz(pat.group_of_row[sz(row)]);
    const auto b = pat.cols.begin() + pat.col_ptr[g];
    const auto e = pat.cols.begin() + pat.col_ptr[g + 1];
    return std::binary_search(b, e, col);
}

}  // namespace

TEST_CASE("mlfmm local leaf: home levels of a plate with wide columns", "[octree]") {
    const Real h = 25e-9;
    const geometry::TriangleMesh mesh = strip_plate(h);
    const basis::RwgSpace space(mesh);
    const std::vector<Real> r = mlfmm::support_radii(space);
    REQUIRE(static_cast<Index>(r.size()) == space.size());
    CHECK(*std::max_element(r.begin(), r.end()) == mlfmm::max_support_radius(space));
    // Root 24 h, 5 levels (leaf a = 1.5 h): regular functions on the leaf (r <= 1.12 h), the 2h
    // column on level 3 (a = 3 h), the 4h column on level 2 (a = 6 h) for the ratio 1.
    const mlfmm::OctreeParams op{1, 5, 0.0};
    const mlfmm::Octree plain(space, kLambda, op);
    const mlfmm::Octree tree(space, kLambda, op, r, 1.0, 1.0);
    REQUIRE(tree.levels() == 5);
    CHECK_FALSE(plain.has_elevated());
    CHECK(plain.home_level_counts().back() == space.size());
    REQUIRE(tree.has_elevated());
    // The geometry is unchanged by the home levels.
    CHECK(tree.permutation() == plain.permutation());
    CHECK(tree.boxes().size() == plain.boxes().size());
    const std::vector<int> home = home_by_basis(tree);
    for (Index n = 0; n < space.size(); ++n) {
        int expect = 0;
        for (int l = tree.leaf_level(); l >= 0; --l) {
            if (r[sz(n)] <= tree.box_size(l) * (1.0 + mlfmm::kHomeLevelTolerance)) {
                expect = l;
                break;
            }
        }
        if (home[sz(n)] != expect)
            FAIL("basis " << n << ": home level " << home[sz(n)] << ", expected " << expect);
    }
    const std::vector<Index> counts = tree.home_level_counts();
    INFO("home level counts: " << counts[2] << " / " << counts[3] << " / " << counts[4]);
    CHECK(counts[0] == 0);
    CHECK(counts[1] == 0);
    CHECK(counts[2] > 0);
    CHECK(counts[3] > 0);
    CHECK(counts[4] > counts[2] + counts[3]);
    Index total = 0;
    for (const Index c : counts) total += c;
    CHECK(total == space.size());
    // Elevated lists, per-box ranges and active counts.
    for (int l = 0; l < tree.levels(); ++l) {
        const std::vector<Index>& pos = tree.elevated_positions(l);
        CHECK(std::is_sorted(pos.begin(), pos.end()));
        if (l == tree.leaf_level())
            CHECK(pos.empty());
        else
            CHECK(static_cast<Index>(pos.size()) == counts[sz(l)]);
        Index listed = 0;
        for (const Index b : tree.boxes_at_level(l)) {
            const mlfmm::Box& box = tree.boxes()[sz(b)];
            Index elevated = 0, active = 0;
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                elevated += (tree.home_level(p) == l && l < tree.leaf_level()) ? 1 : 0;
                active += tree.home_level(p) >= l ? 1 : 0;
            }
            CHECK(tree.elevated_count(b) == elevated);
            CHECK(tree.active_elements(b) == active);
            CHECK(plain.active_elements(b) == box.num_elements);
            for (Index i = 0; i < tree.elevated_count(b); ++i) {
                const Index p = pos[sz(tree.elevated_first(b) + i)];
                CHECK((p >= box.first_element && p < box.first_element + box.num_elements));
            }
            listed += tree.elevated_count(b);
        }
        CHECK(listed == static_cast<Index>(pos.size()));
    }
    CHECK(tree.summary().find("home levels") != std::string::npos);
    CHECK(plain.summary().find("home levels") == std::string::npos);
    // Invalid radii / ratio.
    std::vector<Real> bad = r;
    bad.pop_back();
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, bad, 1.0, 1.0), std::invalid_argument);
    bad = r;
    bad[3] = -1.0;
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, bad, 1.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, r, 0.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, r, 1.0, 0.0), std::invalid_argument);
    // The elevated ratio applies above the leaf only: with rho_e = rho / 2 the 2h column
    // (r ~ 2.06 h > 0.5 x 3 h) moves from level 3 to level 2, the 4h column (r ~ 4.03 h >
    // 0.5 x 6 h) from level 2 to level 1, the leaf functions stay. No function lives one level
    // above the leaf (rho_e a_{D-1} = rho a_D).
    const mlfmm::Octree half(space, kLambda, op, r, 1.0, 0.5);
    const std::vector<int> home_half = home_by_basis(half);
    for (Index n = 0; n < space.size(); ++n) {
        const int expect = home[sz(n)] == tree.leaf_level() ? tree.leaf_level()
                           : r[sz(n)] <= 3.0 * h            ? 2
                                                            : 1;
        if (home_half[sz(n)] != expect)
            FAIL("basis " << n << " (rho_e = rho / 2): home level " << home_half[sz(n)]
                          << ", expected " << expect);
    }
    CHECK(half.home_level_counts()[3] == 0);
    CHECK(half.home_level_counts()[2] == counts[3]);
    CHECK(half.home_level_counts()[1] == counts[2]);
    CHECK_THROWS_AS(tree.elevated_positions(5), std::out_of_range);
}

TEST_CASE("mlfmm local leaf: near pattern and its estimate with elevated functions",
          "[near_field]") {
    const geometry::TriangleMesh mesh = strip_plate(25e-9);
    const basis::RwgSpace space(mesh);
    const mlfmm::Octree tree(space, kLambda, mlfmm::OctreeParams{1, 5, 0.0},
                             mlfmm::support_radii(space), 1.0, 1.0);
    REQUIRE(tree.has_elevated());
    const op::BasisPattern pat = mlfmm::near_pattern(tree, space);
    const auto anc = ancestors(tree);
    const std::vector<int> home = home_by_basis(tree);
    const Index n = space.size();
    // Brute force: (a, b) near iff the ancestors on min(h(a), h(b)) coincide or touch.
    Index near = 0, wrong = 0;
    for (Index a = 0; a < n; ++a) {
        for (Index b = 0; b < n; ++b) {
            const int m = std::min(home[sz(a)], home[sz(b)]);
            const bool expect =
                touching(tree.boxes()[sz(anc[sz(a)][sz(m)])], tree.boxes()[sz(anc[sz(b)][sz(m)])]);
            near += expect ? 1 : 0;
            wrong += expect != in_pattern(pat, a, b) ? 1 : 0;
        }
    }
    INFO(near << " near pairs of " << n * n);
    CHECK(wrong == 0);
    Index stored = 0;
    for (Index a = 0; a < n; ++a) {
        const auto g = sz(pat.group_of_row[sz(a)]);
        stored += pat.col_ptr[g + 1] - pat.col_ptr[g];
    }
    CHECK(stored == near);
    // The estimate counts the same pairs: 4 x 24 bytes per pair + 2N + 1 row pointers.
    CHECK(mlfmm::estimate_near_bytes(tree) == sz(4 * near * 24 + (2 * n + 1) * 8));
    // Without home levels: the WP19a leaf pattern, larger near field for the wide columns'
    // leaves but no elevated rows.
    const mlfmm::Octree plain(space, kLambda, mlfmm::OctreeParams{1, 5, 0.0});
    const op::BasisPattern p0 = mlfmm::near_pattern(plain, space);
    CHECK(p0.col_ptr.size() == plain.boxes_at_level(plain.leaf_level()).size() + 1);
    CHECK(pat.col_ptr.size() > p0.col_ptr.size());
}

TEST_CASE("mlfmm local leaf: every basis pair is near, expanded, exact or truncated once",
          "[mlfmm]") {
    // Plate h = 50 nm, 2.3 um x 0.6 um (5 levels, leaf a ~ 0.29 lambda, regular r / a <= 0.39;
    // ratio 0.6 on every level: structure only, no accuracy claim): the 3h column (r ~ 151 nm)
    // lives on level 3, the 4h column (r ~ 201 nm) on level 2. Vacuum exterior: expansion on every
    // far level; Ag interior: no expansion, its far pairs are truncated by the decay bound or
    // evaluated exactly (reach x* / alpha ~ 0.3 um).
    Setup s(strip_plate(50e-9, 13, 3.0, 4.0, 12), material::silver_500nm(), Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    const mlfmm::Octree tree(s.space, kLambda, mlfmm::OctreeParams{1, 5, 0.0},
                             mlfmm::support_radii(s.space), 0.6, 0.6);
    REQUIRE(tree.levels() == 5);
    REQUIRE(tree.home_level_counts()[2] > 0);
    REQUIRE(tree.home_level_counts()[3] > 0);
    mlfmm::MlfmmParams p;
    p.accuracy_digits = 3.0;
    p.automatic_leaf_size = false;
    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
    INFO(far.describe());
    REQUIRE(far.region_active(0));
    REQUIRE(far.region_active(1));
    const op::BasisPattern pat = mlfmm::near_pattern(tree, s.space);
    const auto anc = ancestors(tree);
    const std::vector<int> home = home_by_basis(tree);
    const auto& boxes = tree.boxes();
    const Index n = s.space.size();
    // Per pair: -1 near, else the unique far level (0 if the far level is not unique).
    std::vector<int> far_level(sz(n * n), 0);
    Index near = 0, errors = 0;
    std::set<int> elevated_far_levels;
    for (Index a = 0; a < n; ++a) {
        for (Index b = 0; b < n; ++b) {
            const int m = std::min(home[sz(a)], home[sz(b)]);
            const bool is_near =
                touching(boxes[sz(anc[sz(a)][sz(m)])], boxes[sz(anc[sz(b)][sz(m)])]);
            errors += is_near != in_pattern(pat, a, b) ? 1 : 0;
            int& level = far_level[sz(a * n + b)];
            if (is_near) {
                ++near;
                level = -1;
                continue;
            }
            // Exactly one level l in [2, m] with the ancestors in each other's lists.
            int hits = 0;
            for (int l = m; l >= 2; --l) {
                const mlfmm::Box& A = boxes[sz(anc[sz(a)][sz(l)])];
                if (std::binary_search(A.interaction_list.begin(), A.interaction_list.end(),
                                       anc[sz(b)][sz(l)])) {
                    level = l;
                    ++hits;
                }
            }
            if (hits != 1) {
                ++errors;
                level = 0;
            } else if (m < tree.leaf_level()) {
                elevated_far_levels.insert(level);
            }
        }
    }
    CHECK(errors == 0);
    INFO("near pairs " << near << " of " << n * n << ", elevated far levels "
                       << elevated_far_levels.size());
    CHECK(elevated_far_levels.size() == 2);
    std::array<Index, 3> kinds{};  // expansion, exact, truncated over both regions
    for (int region = 0; region < 2; ++region) {
        const std::vector<mlfmm::FarLevelInfo>& info = far.levels(region);
        const op::RegionSparseOperator* exact = far.exact_part(region);
        const auto in_exact = [&](Index a, Index b) {
            if (exact == nullptr)
                return false;
            const auto& rp = exact->row_ptr();
            const auto& cols = exact->col_indices();
            return std::binary_search(cols.begin() + rp[sz(a)], cols.begin() + rp[sz(a) + 1], b);
        };
        Index expansion = 0, exact_pairs = 0, truncated = 0, wrong = 0;
        for (Index a = 0; a < n; ++a) {
            for (Index b = 0; b < n; ++b) {
                const int level = far_level[sz(a * n + b)];
                if (level < 0) {
                    wrong += in_exact(a, b) ? 1 : 0;  // near pairs are never in the exact part
                    continue;
                }
                if (level == 0)
                    continue;  // counted as an error above
                if (info[sz(level - 2)].decision == mlfmm::FarDecision::expansion) {
                    ++expansion;
                    wrong += in_exact(a, b) ? 1 : 0;
                } else if (in_exact(a, b)) {
                    ++exact_pairs;
                } else {
                    ++truncated;
                }
            }
        }
        Index truncated_reported = far.exact_info(region).truncated_pairs;
        for (const mlfmm::FarLevelInfo& f : info) truncated_reported += f.truncated_basis_pairs;
        INFO("region " << region << ": expansion " << expansion << ", exact " << exact_pairs
                       << ", truncated " << truncated << " (reported " << truncated_reported
                       << ")");
        CHECK(wrong == 0);
        CHECK(near + expansion + exact_pairs + truncated == n * n);
        CHECK(exact_pairs == (exact != nullptr ? exact->pairs() : 0));
        CHECK(truncated == truncated_reported);
        kinds[0] += expansion;
        kinds[1] += exact_pairs;
        kinds[2] += truncated;
    }
    // The case exercises all kinds: R1 expands, R2 is exact or truncated.
    CHECK(kinds[0] > 0);
    CHECK(kinds[1] > 0);
    CHECK(kinds[2] > 0);
}

TEST_CASE("mlfmm local leaf: a function without home level >= 2 is rejected", "[mlfmm]") {
    Setup s(strip_plate(25e-9), material::silicon_500nm(), Kind::PMCHWT);
    // Ratio 0.3 on a tree with leaf a = 1.5 h: the 4h column (r ~ 4.03 h) needs a >= 13.4 h,
    // i.e. level 0 (a = 24 h): no home level >= 2.
    const mlfmm::Octree tree(s.space, kLambda, mlfmm::OctreeParams{1, 5, 0.0},
                             mlfmm::support_radii(s.space), 0.3, 0.3);
    REQUIRE(tree.home_level_counts()[0] + tree.home_level_counts()[1] > 0);
    mlfmm::MlfmmParams p;
    p.automatic_leaf_size = false;
    try {
        const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
        FAIL("expected TruncationOrderError");
    } catch (const mlfmm::TruncationOrderError& e) {
        INFO(e.what());
        CHECK(e.cause() == mlfmm::TruncationOrderError::Cause::mesh_or_leaf_size);
        CHECK(e.level() < 2);
        CHECK(std::string(e.what()).find("home level") != std::string::npos);
    }
}

TEST_CASE("mlfmm local leaf: quantile 1 reproduces the global leaf rule bitwise", "[mlfmm]") {
    Setup s(graded_box(1e-6, 1e-6), {Complex(2.25, 0.0), Complex(1.0, 0.0)}, Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    mlfmm::MlfmmParams p;
    p.leaf_radius_quantile = 1.0;
    p.octree.max_elements_per_leaf = 8;
    const mlfmm::MlfmmOperator local(s.problem, p);
    CHECK_FALSE(local.octree().has_elevated());
    // The WP21 path: the leaf-rule octree without home levels.
    mlfmm::MlfmmParams legacy = p;
    legacy.octree = mlfmm::leaf_rule_params(s.space, kLambda, p);
    legacy.automatic_leaf_size = false;
    const mlfmm::MlfmmOperator global(s.problem, legacy);
    REQUIRE(global.octree().levels() == local.octree().levels());
    const VectorXc x = random_vector(local.cols(), 7);
    VectorXc y1, y2;
    local.apply(x, y1);
    global.apply(x, y2);
    CHECK((y1.array() == y2.array()).all());
    CHECK(mlfmm::support_radius_quantile(s.space, 1.0) == mlfmm::max_support_radius(s.space));
    // The quantile: at least q N radii <= r_q, fewer below.
    const std::vector<Real> r = mlfmm::support_radii(s.space);
    for (const Real q : {0.5, 0.9, 0.99}) {
        const Real rq = mlfmm::support_radius_quantile(s.space, q);
        const auto le = std::count_if(r.begin(), r.end(), [&](Real v) { return v <= rq; });
        const auto lt = std::count_if(r.begin(), r.end(), [&](Real v) { return v < rq; });
        CHECK(static_cast<Real>(le) >= q * static_cast<Real>(r.size()) * (1.0 - 1e-12));
        CHECK(static_cast<Real>(lt) < q * static_cast<Real>(r.size()));
    }
    for (const Real q : {0.0, -0.5, 1.5}) {
        mlfmm::MlfmmParams bad = p;
        bad.leaf_radius_quantile = q;
        CHECK_THROWS_AS(mlfmm::leaf_rule_params(s.space, kLambda, bad), std::invalid_argument);
    }
    // The leaf floor follows the quantile radius.
    mlfmm::MlfmmParams half = p;
    half.leaf_radius_quantile = 0.5;
    const Real expected = std::max(0.25, mlfmm::support_radius_quantile(s.space, 0.5) /
                                             (0.6 * (1.0 - mlfmm::kMinBoxSizeTolerance) * kLambda));
    CHECK(std::abs(mlfmm::leaf_rule_params(s.space, kLambda, half).min_box_size_lambda -
                   expected) <= 1e-12 * expected);
}

TEST_CASE("mlfmm local leaf: columns and rows of elevated functions match exact entries",
          "[mlfmm]") {
    // Graded box 0.6 um x 0.6 um, depth 2 um (root ~2.07 um), n = 1.5 interior. The quantile of
    // the top-face radii (<= 0.6 lambda / 4) gives lambda / 4 leaves (5 levels); the coarse box
    // functions (r = 112-141 nm > 0.6 a_leaf) live on level 2 (rho_e = 0.3, 517 nm boxes), none
    // on level 3 (rho_e a_3 = rho a_4). Si and Ag with the dense matrix: validation-large.
    Setup s(graded_box(0.6e-6, 2e-6), {Complex(2.25, 0.0), Complex(1.0, 0.0)}, Kind::PMCHWT);
#ifdef NDEBUG
    s.problem.kernel_options.target_accuracy = 1e-5;
    constexpr bool kAccuracy = true;
#else
    s.problem.kernel_options = cheap_options();  // structure only in the sanitizer build (time)
    constexpr bool kAccuracy = false;
#endif
    const Index n = s.space.size();
    const std::vector<Real> r = mlfmm::support_radii(s.space);
    const Real top = 0.6 * 0.25 * kLambda * (1.0 - mlfmm::kMinBoxSizeTolerance);
    const auto small = std::count_if(r.begin(), r.end(), [&](Real v) { return v <= top; });
    mlfmm::MlfmmParams p;
    p.leaf_radius_quantile = static_cast<Real>(small) / static_cast<Real>(n);
    p.octree.max_elements_per_leaf = 4;
    const mlfmm::MlfmmOperator Z(s.problem, p);
    const mlfmm::Octree& tree = Z.octree();
    INFO("2N = " << 2 * n << ", quantile " << p.leaf_radius_quantile << "\n" << Z.describe());
    REQUIRE(tree.levels() == 5);
    REQUIRE(tree.has_elevated());
    const std::vector<Index> counts = tree.home_level_counts();
    CHECK(counts[3] == 0);
    REQUIRE(counts[2] > 0);
    CHECK(counts[2] + counts[4] == n);
    CHECK(Z.describe().find("home levels") != std::string::npos);
    // Both regions expand on level 2, where the elevated functions live.
    for (int region = 0; region < 2; ++region) {
        const std::vector<mlfmm::FarLevelInfo>& levels = Z.far_operator().levels(region);
        const mlfmm::FarLevelInfo& f = levels.front();
        CHECK(f.level == 2);
        CHECK(f.decision == mlfmm::FarDecision::expansion);
        CHECK(f.home_functions == counts[2]);
        CHECK(f.elevated_pattern_bytes > 0);
        CHECK(f.support_radius > levels.back().support_radius);
        CHECK(levels.back().home_functions == counts[4]);
    }
    const std::vector<Index>& elevated = tree.elevated_positions(2);
    REQUIRE(elevated.size() >= 4);
    std::vector<Index> probe;  // elevated and leaf functions, by basis index
    for (std::size_t i = 0; i < 4; ++i)
        probe.push_back(tree.permutation()[sz(elevated[i * (elevated.size() - 1) / 3])]);
    for (Index pp = 0; pp < n; pp += n / 3) {
        if (tree.home_level(pp) == tree.leaf_level())
            probe.push_back(tree.permutation()[sz(pp)]);
    }
    std::sort(probe.begin(), probe.end());
    probe.erase(std::unique(probe.begin(), probe.end()), probe.end());
    // Exact columns (all rows) and exact rows (all columns) of the probes.
    op::BasisPattern cols_pat;
    cols_pat.group_of_row.assign(sz(n), 0);
    cols_pat.cols = probe;
    cols_pat.col_ptr = {0, static_cast<Index>(probe.size())};
    const auto exact_cols = op::assemble_sparse(s.problem, cols_pat);
    op::BasisPattern rows_pat;
    rows_pat.group_of_row.assign(sz(n), 1);
    for (const Index b : probe) rows_pat.group_of_row[sz(b)] = 0;
    for (Index c = 0; c < n; ++c) rows_pat.cols.push_back(c);
    rows_pat.col_ptr = {0, n, n};
    const auto exact_rows = op::assemble_sparse(s.problem, rows_pat);
    Real worst_elevated = 0.0, worst_leaf = 0.0;
    for (const Index b : probe) {
        const bool is_elevated = tree.home_level(tree.inverse_permutation()[sz(b)]) == 2;
        for (const Index col : {b, n + b}) {
            const VectorXc y = Z * VectorXc::Unit(2 * n, col);
            const VectorXc ex = exact_cols->matrix().col(col);
            const Real err = (y - ex).norm() / ex.norm();
            CHECK(std::isfinite(err));
            Real& worst = is_elevated ? worst_elevated : worst_leaf;
            worst = std::max(worst, err);
        }
    }
    const VectorXc x = random_vector(2 * n, 11);
    const VectorXc y = Z * x;
    const VectorXc yex = exact_rows->matrix() * x;
    Real num = 0.0, den = 0.0;
    for (const Index b : probe) {
        for (const Index row : {b, n + b}) {
            num += std::norm(y(row) - yex(row));
            den += std::norm(yex(row));
        }
    }
    const Real row_err = std::sqrt(num / den);
    WARN("local leaf rule, graded box 0.6 x 2 um, n = 1.5, d0 = 3, 2N = "
         << 2 * n << ", " << counts[2] << " elevated: worst column error " << worst_elevated
         << " (elevated) / " << worst_leaf << " (leaf), row error " << row_err);
    if (kAccuracy) {
        // Single columns as in test_mlfmm_operator (<= 5e-3; K far-only error at lambda / 4
        // leaves), rows of a random x at the docs/05 tolerance.
        CHECK(worst_elevated <= 5e-3);
        CHECK(worst_leaf <= 5e-3);
        CHECK(row_err <= 1e-3);
    }
}
