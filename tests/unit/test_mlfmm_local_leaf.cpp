/// Unit tests of the local leaf rule of the MLFMM (WP21L, ADR 0008 amendment 2026-10-11): home
/// levels of the octree, the near pattern with elevated basis functions and its estimate, the
/// partition of all basis pairs into near / expansion / exact / truncated, the bitwise identity
/// of leaf_radius_quantile = 1 with the global leaf rule, and the accuracy of columns and rows of
/// elevated functions against exact entries. lambda0 = 500 nm.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/interpolation.hpp"
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
#include <cstdlib>
#include <memory>
#include <random>
#include <set>
#include <sstream>
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

/// Open plate in z = 0 on a tensor grid of spacing h with two wide columns (widths 2h and 4h):
/// x = 6 cells of h, one of 2h, 6 of h, one of 4h, 6 of h (24 h); y = 24 cells of h. The long
/// triangles of the wide columns have support radii ~2.06 h and ~4.03 h, the others <= 1.12 h.
geometry::TriangleMesh strip_plate(Real h) {
    std::vector<Real> xs = {0.0};
    const auto add = [&](int cells, Real w) {
        for (int i = 0; i < cells; ++i) xs.push_back(xs.back() + w);
    };
    add(6, h);
    add(1, 2.0 * h);
    add(6, h);
    add(1, 4.0 * h);
    add(6, h);
    const auto nx = static_cast<Index>(xs.size());
    const Index ny = 25;
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

/// The formulation with the exterior weights set to zero (region R2 only).
class InteriorOnly final : public formulation::Formulation {
public:
    explicit InteriorOnly(const formulation::Formulation& f) : f_(f) {}
    [[nodiscard]] Kind kind() const override { return f_.kind(); }
    [[nodiscard]] std::string name() const override { return f_.name(); }
    [[nodiscard]] formulation::Weights weights(Complex eta1, Complex eta2) const override {
        formulation::Weights w = f_.weights(eta1, eta2);
        w.a1 = w.b1 = Complex(0.0, 0.0);
        return w;
    }

private:
    const formulation::Formulation& f_;
};

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
    const mlfmm::Octree tree(space, kLambda, op, r, 1.0);
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
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, bad, 1.0), std::invalid_argument);
    bad = r;
    bad[3] = -1.0;
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, bad, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::Octree(space, kLambda, op, r, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(tree.elevated_positions(5), std::out_of_range);
}

TEST_CASE("mlfmm local leaf: near pattern and its estimate with elevated functions",
          "[near_field]") {
    const geometry::TriangleMesh mesh = strip_plate(25e-9);
    const basis::RwgSpace space(mesh);
    const mlfmm::Octree tree(space, kLambda, mlfmm::OctreeParams{1, 5, 0.0},
                             mlfmm::support_radii(space), 1.0);
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
            const bool expect = touching(tree.boxes()[sz(anc[sz(a)][sz(m)])],
                                         tree.boxes()[sz(anc[sz(b)][sz(m)])]);
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
    // Lossy interior only (exterior weights zero): Ag at h = 25 nm (leaf a = 37.5 nm). The
    // interior expands on the fine levels and falls back to truncation / exact on the coarse
    // ones, where the elevated functions of the wide columns live.
    Setup s(strip_plate(25e-9), material::silver_500nm(), Kind::PMCHWT);
    const InteriorOnly interior(*s.form);
    s.problem.formulation = &interior;
    s.problem.kernel_options = cheap_options();
    const mlfmm::Octree tree(s.space, kLambda, mlfmm::OctreeParams{1, 5, 0.0},
                             mlfmm::support_radii(s.space), 0.9);
    REQUIRE(tree.has_elevated());
    mlfmm::MlfmmParams p;
    p.accuracy_digits = 3.0;
    p.automatic_leaf_size = false;
    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
    REQUIRE(far.region_active(1));
    CHECK_FALSE(far.region_active(0));
    INFO(far.describe());
    const op::BasisPattern pat = mlfmm::near_pattern(tree, s.space);
    const auto anc = ancestors(tree);
    const std::vector<int> home = home_by_basis(tree);
    const std::vector<mlfmm::FarLevelInfo>& info = far.levels(1);
    const op::RegionSparseOperator* exact = far.exact_part(1);
    const auto in_exact = [&](Index a, Index b) {
        if (exact == nullptr)
            return false;
        const auto& rp = exact->row_ptr();
        const auto& cols = exact->col_indices();
        return std::binary_search(cols.begin() + rp[sz(a)], cols.begin() + rp[sz(a) + 1], b);
    };
    const Index n = s.space.size();
    Index near = 0, expansion = 0, exact_pairs = 0, truncated = 0, errors = 0;
    std::set<int> elevated_far_levels;
    for (Index a = 0; a < n; ++a) {
        for (Index b = 0; b < n; ++b) {
            const int m = std::min(home[sz(a)], home[sz(b)]);
            const auto& boxes = tree.boxes();
            const bool is_near =
                touching(boxes[sz(anc[sz(a)][sz(m)])], boxes[sz(anc[sz(b)][sz(m)])]);
            const bool stored = in_pattern(pat, a, b);
            if (is_near) {
                ++near;
                errors += (!stored || in_exact(a, b)) ? 1 : 0;
                continue;
            }
            errors += stored ? 1 : 0;
            // Exactly one level l in [2, m] with the ancestors in each other's lists.
            int level = -1, hits = 0;
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
                continue;
            }
            if (m < tree.leaf_level())
                elevated_far_levels.insert(level);
            if (info[sz(level - 2)].decision == mlfmm::FarDecision::expansion) {
                ++expansion;
                errors += in_exact(a, b) ? 1 : 0;
            } else if (in_exact(a, b)) {
                ++exact_pairs;
            } else {
                ++truncated;
            }
        }
    }
    Index truncated_reported = far.exact_info(1).truncated_pairs;
    for (const mlfmm::FarLevelInfo& f : info) truncated_reported += f.truncated_basis_pairs;
    INFO("near " << near << ", expansion " << expansion << ", exact " << exact_pairs
                 << ", truncated " << truncated << " (reported " << truncated_reported
                 << "), elevated far levels " << elevated_far_levels.size());
    CHECK(errors == 0);
    CHECK(near + expansion + exact_pairs + truncated == n * n);
    CHECK(exact_pairs == (exact != nullptr ? exact->pairs() : 0));
    CHECK(truncated == truncated_reported);
    // The case exercises all four kinds and elevated functions on more than one far level.
    CHECK(expansion > 0);
    CHECK(exact_pairs > 0);
    CHECK(truncated > 0);
    CHECK(elevated_far_levels.size() >= 2);
}

TEST_CASE("mlfmm local leaf: quantile 1 reproduces the global leaf rule bitwise", "[mlfmm]") {
    Setup s(graded_box(1e-6, 1e-6), material::silicon_500nm(), Kind::PMCHWT);
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
    // Graded box 1 um: the global rule gives 0.53 lambda leaves (3 levels); with the quantile
    // 0.5 the leaves are lambda / 4 (4 levels) and the coarse box functions live on level 2.
    Setup s(graded_box(1e-6, 1e-6), material::silicon_500nm(), Kind::PMCHWT);
#ifdef NDEBUG
    s.problem.kernel_options.target_accuracy = 1e-5;
    constexpr bool kAccuracy = true;
#else
    s.problem.kernel_options = cheap_options();  // structure only in the sanitizer build (time)
    constexpr bool kAccuracy = false;
#endif
    mlfmm::MlfmmParams p;
    p.leaf_radius_quantile = 0.5;
    p.octree.max_elements_per_leaf = 8;
    const mlfmm::MlfmmOperator Z(s.problem, p);
    const mlfmm::Octree& tree = Z.octree();
    INFO(Z.describe());
    REQUIRE(tree.has_elevated());
    REQUIRE(tree.levels() == 4);
    CHECK(Z.describe().find("home levels") != std::string::npos);
    // Both regions expand on level 2, where the elevated functions live.
    for (int region = 0; region < 2; ++region) {
        const mlfmm::FarLevelInfo& f = Z.far_operator().levels(region).front();
        CHECK(f.level == 2);
        CHECK(f.decision == mlfmm::FarDecision::expansion);
        CHECK(f.home_functions == tree.home_level_counts()[2]);
        CHECK(f.elevated_pattern_bytes > 0);
        CHECK(f.support_radius > Z.far_operator().levels(region).back().support_radius);
    }
    {
        const mlfmm::RadiationPatterns& pat = Z.far_operator().patterns(0);
        Real worst = 0;
        Index checked = 0;
        for (const Index lb : tree.boxes_at_level(tree.leaf_level())) {
            const mlfmm::Box& box = tree.boxes()[sz(lb)];
            for (Index pp = box.first_element; pp < box.first_element + box.num_elements; ++pp) {
                if (pat.slot(pp) < 0)
                    continue;
                const std::vector<Index> one = {tree.permutation()[sz(pp)]};
                const std::vector<Complex> v = mlfmm::basis_patterns(
                    s.space, one, box.center, pat.k(), pat.sampling(),
                    mlfmm::PatternOptions{0, 1e-5});
                for (Index q = 0; q < pat.num_directions(); ++q)
                    for (int c = 0; c < 2; ++c)
                        worst = std::max(worst, std::abs(v[sz(q * 2 + c)] - pat.radiation(pp, q, c)));
                ++checked;
            }
            if (checked > 200)
                break;
        }
        WARN("leaf pattern check: " << checked << " functions, max diff " << worst);
        const mlfmm::RadiationPatterns& p2 = Z.far_operator().patterns(1);
        const mlfmm::SphereSampling ps(Z.far_operator().levels(1).front().sampling_order);
        const mlfmm::SphereInterpolator I(p2.sampling(), ps, 14);
        for (Index pp : {Index{0}, Index{100}, Index{1000}}) {
            while (p2.slot(pp) < 0) ++pp;
            const mlfmm::Box& C = tree.boxes()[sz(p2.leaf_box(pp))];
            const mlfmm::Box& P = tree.boxes()[sz(C.parent)];
            const auto nd = sz(p2.num_directions());
            VectorXc vt(static_cast<Index>(nd)), vp(static_cast<Index>(nd));
            for (std::size_t q = 0; q < nd; ++q) {
                vt(static_cast<Index>(q)) = p2.radiation(pp, static_cast<Index>(q), 0);
                vp(static_cast<Index>(q)) = p2.radiation(pp, static_cast<Index>(q), 1);
            }
            const VectorXc it = I.interpolate(vt, mlfmm::PoleParity::odd);
            const VectorXc ip = I.interpolate(vp, mlfmm::PoleParity::odd);
            const std::vector<Index> one = {tree.permutation()[sz(pp)]};
            const std::vector<Complex> ref = mlfmm::basis_patterns(
                s.space, one, P.center, p2.k(), ps, mlfmm::PatternOptions{0, 1e-5});
            const Vec3 d = C.center - P.center;
            Real num = 0, den = 0;
            for (Index q = 0; q < ps.size(); ++q) {
                const Complex sh =
                    std::exp(Complex(0.0, 1.0) * p2.k() * Real(ps.directions().row(q) * d));
                num += std::norm(sh * it(q) - ref[sz(2 * q)]) + std::norm(sh * ip(q) - ref[sz(2 * q + 1)]);
                den += std::norm(ref[sz(2 * q)]) + std::norm(ref[sz(2 * q + 1)]);
            }
            WARN("interp check p " << pp << ": " << std::sqrt(num / den) << " (Lc "
                                   << p2.sampling().order() << " Lp " << ps.order() << ")");
        }
    }
    const Index n = s.space.size();
    const std::vector<Index>& elevated = tree.elevated_positions(2);
    REQUIRE(elevated.size() >= 4);
    std::vector<Index> probe;  // elevated and leaf functions, by basis index
    for (std::size_t i = 0; i < 4; ++i)
        probe.push_back(tree.permutation()[sz(elevated[i * (elevated.size() - 1) / 3])]);
    probe.push_back(tree.permutation()[0]);
    probe.push_back(tree.permutation()[tree.permutation().size() / 2]);
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
    Real worst_col = 0.0;
    for (const Index b : probe) {
        for (const Index col : {b, n + b}) {
            const VectorXc e = VectorXc::Unit(2 * n, col);
            const VectorXc y = Z * e;
            const VectorXc ex = exact_cols->matrix().col(col);
            const Real err = (y - ex).norm() / ex.norm();
            worst_col = std::max(worst_col, err);
            {
                const VectorXc yn = Z.near_operator() * e;
                Real nearerr = 0, farerr = 0, farn = 0, e2 = 0, n2 = 0, e3 = 0, n3 = 0;
                for (Index rr = 0; rr < 2 * n; ++rr) {
                    if (yn(rr) != Complex(0.0, 0.0)) {
                        nearerr += std::norm(y(rr) - ex(rr));
                    } else {
                        farerr += std::norm(y(rr) - ex(rr));
                        farn += std::norm(ex(rr));
                        if (tree.home_level(tree.inverse_permutation()[sz(rr % n)]) == 2) {
                            e2 += std::norm(y(rr) - ex(rr));
                            n2 += std::norm(ex(rr));
                        } else {
                            e3 += std::norm(y(rr) - ex(rr));
                            n3 += std::norm(ex(rr));
                        }
                    }
                }
                WARN("col " << col << " home "
                            << tree.home_level(tree.inverse_permutation()[sz(b)]) << " err "
                            << err << " near-row err " << std::sqrt(nearerr) / ex.norm()
                            << " far-row err " << std::sqrt(farerr / farn) << " elevated rows "
                            << std::sqrt(e2 / n2) << " (" << std::sqrt(n2) << ") leaf rows "
                            << std::sqrt(e3 / n3) << " (" << std::sqrt(n3) << ")");
            }
            CHECK(std::isfinite(err));
        }
    }
    {
        const auto anc = ancestors(tree);
        const std::vector<int> home = home_by_basis(tree);
        for (int reg = 0; reg < 2; ++reg)
        for (const Index b : probe) {
            for (const Index col : {b, n + b}) {
                class OneRegion final : public formulation::Formulation {
                public:
                    OneRegion(const formulation::Formulation& f, int r) : f_(f), r_(r) {}
                    [[nodiscard]] Kind kind() const override { return f_.kind(); }
                    [[nodiscard]] std::string name() const override { return f_.name(); }
                    [[nodiscard]] formulation::Weights weights(Complex e1,
                                                               Complex e2) const override {
                        formulation::Weights w = f_.weights(e1, e2);
                        if (r_ == 1)
                            w.a1 = w.b1 = Complex(0.0, 0.0);
                        else
                            w.a2 = w.b2 = Complex(0.0, 0.0);
                        return w;
                    }

                private:
                    const formulation::Formulation& f_;
                    int r_;
                };
                const OneRegion one(*s.form, reg);
                op::Problem pr = s.problem;
                pr.formulation = &one;
                const mlfmm::MlfmmFarOperator far(pr, tree, p);
                const auto ec = op::assemble_sparse(pr, cols_pat);
                const VectorXc e = VectorXc::Unit(2 * n, col);
                const VectorXc nearpart = Z.near_operator() * e;
                VectorXc yg = far * e;
                const VectorXc ex = ec->matrix().col(col);
                for (Index rr = 0; rr < 2 * n; ++rr)
                    if (nearpart(rr) != Complex(0.0, 0.0))
                        yg(rr) = ex(rr);
                WARN("region " << reg);
                std::array<Real, 4> en{}, nn{};
                for (Index rr = 0; rr < 2 * n; ++rr) {
                    const Index a = rr % n;
                    const int m = std::min(home[sz(a)], home[sz(b)]);
                    if (touching(tree.boxes()[sz(anc[sz(a)][sz(m)])],
                                 tree.boxes()[sz(anc[sz(b)][sz(m)])]))
                        continue;
                    int level = -1;
                    for (int l = m; l >= 2; --l) {
                        const mlfmm::Box& A = tree.boxes()[sz(anc[sz(a)][sz(l)])];
                        if (std::binary_search(A.interaction_list.begin(),
                                               A.interaction_list.end(), anc[sz(b)][sz(l)]))
                            level = l;
                    }
                    const std::size_t k = sz(level < 0 ? 0 : level);
                    en[k] += std::norm(yg(rr) - ex(rr));
                    nn[k] += std::norm(ex(rr));
                }
                WARN("col " << col << " level 2: " << std::sqrt(en[2] / nn[2]) << ", level 3: "
                            << std::sqrt(en[3] / nn[3]) << ", none: " << nn[0]);
            }
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
    WARN("local leaf rule, graded box 1 um Si, d0 = 3: worst column error " << worst_col
                                                                             << ", row error "
                                                                             << row_err);
    if (kAccuracy) {
        // Single columns as in test_mlfmm_operator (<= 5e-3; K far-only error at lambda / 4
        // leaves), rows of a random x at the docs/05 tolerance.
        CHECK(worst_col <= 5e-3);
        CHECK(row_err <= 1e-3);
    }
}

TEST_CASE("mlfmm local leaf: a function without home level >= 2 is rejected", "[mlfmm]") {
    const geometry::TriangleMesh mesh = strip_plate(25e-9);
    Setup s(strip_plate(25e-9), material::silicon_500nm(), Kind::PMCHWT);
    // Ratio 0.3: the 4h column (r ~ 4.03 h) needs a >= 13.4 h, i.e. level 1 (a = 12 h is too
    // small, level 0 qualifies): no home level >= 2.
    const mlfmm::Octree tree(s.space, kLambda, mlfmm::OctreeParams{1, 5, 0.0},
                             mlfmm::support_radii(s.space), 0.3);
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

namespace {
void level_errors(Setup& s, const mlfmm::MlfmmParams& p) {
    const mlfmm::Octree tree = mlfmm::make_octree(s.space, kLambda, p);
    const Index n = s.space.size();
    WARN("tree levels " << tree.levels() << " elevated " << tree.has_elevated());
    std::vector<Index> probe = {tree.permutation()[0], tree.permutation()[sz(n / 2)]};
    op::BasisPattern cols_pat;
    cols_pat.group_of_row.assign(sz(n), 0);
    std::sort(probe.begin(), probe.end());
    cols_pat.cols = probe;
    cols_pat.col_ptr = {0, static_cast<Index>(probe.size())};
    const auto anc = ancestors(tree);
    const std::vector<int> home = home_by_basis(tree);
    for (int reg = 0; reg < 2; ++reg) {
        class OneRegion final : public formulation::Formulation {
        public:
            OneRegion(const formulation::Formulation& f, int r) : f_(f), r_(r) {}
            [[nodiscard]] Kind kind() const override { return f_.kind(); }
            [[nodiscard]] std::string name() const override { return f_.name(); }
            [[nodiscard]] formulation::Weights weights(Complex e1, Complex e2) const override {
                formulation::Weights w = f_.weights(e1, e2);
                if (r_ == 1)
                    w.a1 = w.b1 = Complex(0.0, 0.0);
                else
                    w.a2 = w.b2 = Complex(0.0, 0.0);
                return w;
            }

        private:
            const formulation::Formulation& f_;
            int r_;
        };
        const OneRegion one(*s.form, reg);
        op::Problem pr = s.problem;
        pr.formulation = &one;
        const mlfmm::MlfmmFarOperator far(pr, tree, p);
        const auto ec = op::assemble_sparse(pr, cols_pat);
        if (tree.has_elevated()) {
            // Manual elevated -> elevated and leaf -> elevated at level 2.
            const std::vector<Index>& pos = tree.elevated_positions(2);
            const mlfmm::FarLevelInfo& f2 = far.levels(reg).front();
            const mlfmm::SphereSampling smp(f2.sampling_order);
            const Complex k = (reg == 0 ? pr.exterior : pr.object).wavenumber(pr.omega);
            const Index pb = pos[0];
            const Index bb = tree.permutation()[sz(pb)];
            Index box_b = -1;
            for (const Index bx : tree.boxes_at_level(2)) {
                const mlfmm::Box& B = tree.boxes()[sz(bx)];
                if (pb >= B.first_element && pb < B.first_element + B.num_elements)
                    box_b = bx;
            }
            const mlfmm::Box& B = tree.boxes()[sz(box_b)];
            const VectorXc e = VectorXc::Unit(2 * n, bb);
            const VectorXc yp = far * e;
            op::BasisPattern cp;
            cp.group_of_row.assign(sz(n), 0);
            cp.cols = {bb};
            cp.col_ptr = {0, 1};
            const auto exb = op::assemble_sparse(pr, cp);
            const VectorXc ex = exb->matrix().col(bb);
            int shown = 0;
            for (const Index ia : B.interaction_list) {
                const mlfmm::Box& A = tree.boxes()[sz(ia)];
                for (Index i = tree.elevated_first(ia);
                     i < tree.elevated_first(ia) + tree.elevated_count(ia) && shown < 3; ++i) {
                    const Index aa = tree.permutation()[sz(pos[sz(i)])];
                    const std::vector<Index> va = {aa}, vb = {bb};
                    const auto Pa = mlfmm::basis_patterns(s.space, va, A.center, k, smp,
                                                          mlfmm::PatternOptions{0, 1e-5});
                    const auto Pb = mlfmm::basis_patterns(s.space, vb, B.center, k, smp,
                                                          mlfmm::PatternOptions{0, 1e-5});
                    const VectorXc T =
                        mlfmm::translator(k, A.center - B.center, smp, f2.truncation_order);
                    Complex sum(0.0, 0.0);
                    for (int it = 0; it < smp.num_theta(); ++it)
                        for (int ip = 0; ip < smp.num_phi(); ++ip) {
                            const Index q = smp.index(it, ip);
                            const Index qa = smp.index(smp.num_theta() - 1 - it,
                                                       (ip + smp.num_phi() / 2) % smp.num_phi());
                            sum += smp.weights()(q) * T(q) *
                                   (Pa[sz(2 * qa)] * Pb[sz(2 * q)] -
                                    Pa[sz(2 * qa + 1)] * Pb[sz(2 * q + 1)]);
                        }
                    WARN("region " << reg << " elev->elev row " << aa << ": pass " << yp(aa)
                                   << " exact " << ex(aa) << " manual sum " << sum
                                   << " ratio pass/exact " << yp(aa) / ex(aa)
                                   << " exact/sum " << ex(aa) / sum);
                    ++shown;
                }
            }
            // Leaf source -> elevated rows.
            Index pl = 0;
            while (tree.home_level(pl) != tree.leaf_level()) ++pl;
            const Index bl = tree.permutation()[sz(pl)];
            const VectorXc el = VectorXc::Unit(2 * n, bl);
            const VectorXc ypl = far * el;
            cp.cols = {bl};
            const auto exl = op::assemble_sparse(pr, cp);
            const VectorXc exlc = exl->matrix().col(bl);
            Index leafbox = -1;
            for (const Index bx : tree.boxes_at_level(tree.leaf_level())) {
                const mlfmm::Box& C = tree.boxes()[sz(bx)];
                if (pl >= C.first_element && pl < C.first_element + C.num_elements)
                    leafbox = bx;
            }
            const mlfmm::Box& C = tree.boxes()[sz(leafbox)];
            const mlfmm::Box& P = tree.boxes()[sz(C.parent)];
            const Index pidx = C.parent;
            shown = 0;
            for (const Index ia : P.interaction_list) {
                const mlfmm::Box& A = tree.boxes()[sz(ia)];
                for (Index i = tree.elevated_first(ia);
                     i < tree.elevated_first(ia) + tree.elevated_count(ia) && shown < 3; ++i) {
                    const Index aa = tree.permutation()[sz(pos[sz(i)])];
                    WARN("region " << reg << " leaf->elev row " << aa << " (P " << pidx
                                   << "): pass " << ypl(aa) << " exact " << exlc(aa)
                                   << " ratio " << ypl(aa) / exlc(aa));
                    ++shown;
                }
            }
        }
        for (const Index b : probe) {
            for (const Index col : {b, n + b}) {
                const VectorXc e = VectorXc::Unit(2 * n, col);
                const VectorXc yg = far * e;
                const VectorXc ex = ec->matrix().col(col);
                std::vector<Real> en(sz(tree.levels()), 0.0), nn(sz(tree.levels()), 0.0);
                for (Index rr = 0; rr < 2 * n; ++rr) {
                    const Index a = rr % n;
                    const int m = std::min(home[sz(a)], home[sz(b)]);
                    if (touching(tree.boxes()[sz(anc[sz(a)][sz(m)])],
                                 tree.boxes()[sz(anc[sz(b)][sz(m)])]))
                        continue;
                    int level = 0;
                    for (int l = m; l >= 2; --l) {
                        const mlfmm::Box& A = tree.boxes()[sz(anc[sz(a)][sz(l)])];
                        if (std::binary_search(A.interaction_list.begin(),
                                               A.interaction_list.end(), anc[sz(b)][sz(l)]))
                            level = l;
                    }
                    en[sz(level)] += std::norm(yg(rr) - ex(rr));
                    nn[sz(level)] += std::norm(ex(rr));
                }
                std::ostringstream os;
                for (int l = 2; l < tree.levels(); ++l)
                    os << " level " << l << ": " << std::sqrt(en[sz(l)] / nn[sz(l)]);
                WARN("region " << reg << " col " << col << os.str());
            }
        }
    }
}
}  // namespace

TEST_CASE("debug levels", "[mlfmm_dbg]") {
    {
        Setup s(graded_box(1.6e-6, 1e-6), material::silicon_500nm(), Kind::PMCHWT);
        s.problem.kernel_options.target_accuracy = 1e-5;
        mlfmm::MlfmmParams p;
        p.leaf_radius_quantile = 0.5;
        p.octree.max_elements_per_leaf = 8;
        level_errors(s, p);
    }
    {
        geometry::RoughSurfaceParams rp;
        rp.edge_length_L = 1e-6;
        rp.rms_roughness = 30e-9;
        rp.correlation_length = 250e-9;
        rp.mesh_size = 50e-9;
        rp.box_depth = 0.3e-6;
        rp.box_mesh_size = 50e-9;
        rp.seed = 7;
        Setup s(geometry::make_rough_surface_mesh(rp), material::silicon_500nm(), Kind::PMCHWT);
        s.problem.kernel_options.target_accuracy = 1e-5;
        mlfmm::MlfmmParams p;
        p.leaf_radius_quantile = 1.0;
        p.octree.max_elements_per_leaf = 8;
        level_errors(s, p);
    }
}
