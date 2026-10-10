/// Unit tests of the MLFMM region policy (ADR 0008 §6, WP21): expansion / truncation / exact
/// decisions, per-region complementarity of the exact part, the leaf rule, the workspace pool's
/// exception safety and the error type. Icospheres at lambda0 = 500 nm.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/region_sparse_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace specklebem;
using formulation::Kind;

namespace {

constexpr Real kLambda = 500e-9;

/// Mesh, space, Problem (vacuum exterior) and octree. Not movable.
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

geometry::TriangleMesh rough_box(Real size, Real mesh_size) {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = size;
    p.rms_roughness = 30e-9;
    p.correlation_length = 250e-9;
    p.mesh_size = mesh_size;
    p.box_depth = 0.3e-6;
    p.box_mesh_size = mesh_size;
    p.seed = 7;
    return geometry::make_rough_surface_mesh(p);
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

/// 3 levels without the leaf rule: icosphere R = 1.5 lambda0 (leaf a = 0.75 lambda0).
mlfmm::MlfmmParams three_levels(Real digits) {
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, 3, 0.0};
    p.accuracy_digits = digits;
    p.automatic_leaf_size = false;
    return p;
}

VectorXc random_vector(Index n) {
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) {
        const auto t = static_cast<Real>(i);
        x(i) = Complex(std::sin(1.3 * t + 0.2), std::cos(0.7 * t));
    }
    return x;
}

}  // namespace

TEST_CASE("mlfmm policy: a lossless region keeps the expansion", "[mlfmm]") {
    Setup s(geometry::make_icosphere(1.5 * kLambda, 2), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
            Kind::PMCHWT);
    const mlfmm::MlfmmParams p = three_levels(3.0);
    const mlfmm::Octree tree(s.space, kLambda, p.octree);
    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
    const std::string d = far.describe();
    INFO(d);
    for (const int r : {0, 1}) {
        REQUIRE(far.levels(r).size() == 1);
        const mlfmm::FarLevelInfo& f = far.levels(r).front();
        CHECK(f.decision == mlfmm::FarDecision::expansion);
        CHECK(f.search_achievable);
        CHECK(f.block_pairs >= 1);
        CHECK(f.block_error >= 0.0);  // reported, decides only for lossy regions
        CHECK(f.block_error < 1e-1);
        CHECK(f.reference_error < 0.0);
        CHECK(far.exact_part(r) == nullptr);
    }
    CHECK(d.find("level 2: ") != std::string::npos);
    CHECK(d.find("expansion; search achievable") != std::string::npos);
    CHECK(std::string(mlfmm::to_string(mlfmm::FarDecision::truncation)) == "truncation");
    CHECK(std::string(mlfmm::to_string(mlfmm::FarDecision::exact)) == "exact");
    // describe() and memory_bytes() count one apply workspace before the first apply.
    const std::size_t before = far.memory_bytes();
    CHECK(far.workspaces() == 0);
    CHECK(d.find("1 apply workspace(s)") != std::string::npos);
    CHECK(d.find("(0 allocated so far") != std::string::npos);
    const VectorXc y = far * random_vector(far.cols());
    CHECK(far.workspaces() == 1);
    CHECK(far.memory_bytes() == before);
}

TEST_CASE("mlfmm policy: a very lossy region is truncated with the bound logged", "[mlfmm]") {
    // n = 0.1 - 20j (alpha = 20 k0 = 251 / um): the expansion fails and every far pair lies
    // beyond the decay bound (subdivision 3: r_max ~ 0.25 a, support distance >= 0.4 a = 150 nm).
    const Complex n(0.1, -20.0);
    Setup s(geometry::make_icosphere(1.5 * kLambda, 3), {n * n, Complex(1.0, 0.0)}, Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    const mlfmm::MlfmmParams p = three_levels(3.0);
    const mlfmm::Octree tree(s.space, kLambda, p.octree);
    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
    INFO(far.describe());
    REQUIRE(far.levels(1).size() == 1);
    const mlfmm::FarLevelInfo& f = far.levels(1).front();
    CHECK(f.decision == mlfmm::FarDecision::truncation);
    CHECK(f.truncated_pairs > 0);
    CHECK(f.exact_pairs == 0);
    CHECK(f.decay_bound > 0.0);
    CHECK(f.decay_bound <= 1e-4);
    CHECK(far.exact_part(1) == nullptr);
    CHECK_THROWS_AS(far.patterns(1), std::invalid_argument);
    CHECK(far.levels(0).front().decision == mlfmm::FarDecision::expansion);
    CHECK(far.describe().find("truncated (max delta") != std::string::npos);
}

TEST_CASE("mlfmm policy: Ag falls back to exact pairs; per-region complementarity", "[mlfmm]") {
    // Ag at 0.75 lambda0 leaves: no expansion order; near pairs of the interaction list exceed
    // the decay bound (exact), farther ones are truncated.
    Setup s(geometry::make_icosphere(1.5 * kLambda, 2), material::silver_500nm(), Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    const InteriorOnly interior(*s.form);
    s.problem.formulation = &interior;  // region R2 only: Z_mlfmm vs the dense R2 part
    const auto dense = op::DenseStrategy().build(s.problem);
    const VectorXc x = random_vector(dense->cols());
    const VectorXc yd = *dense * x;
    {
        const mlfmm::MlfmmOperator Z(s.problem, three_levels(3.0));
        const mlfmm::MlfmmFarOperator& far = Z.far_operator();
        INFO(Z.describe());
        CHECK(!far.region_active(0));
        REQUIRE(far.levels(1).size() == 1);
        const mlfmm::FarLevelInfo& f = far.levels(1).front();
        CHECK(f.decision == mlfmm::FarDecision::exact);
        CHECK(!f.search_achievable);
        CHECK(f.exact_pairs > 0);
        CHECK(f.truncated_pairs > 0);
        CHECK(f.decay_bound <= 1e-4);
        CHECK(f.exact_bound > 1e-4);
        REQUIRE(far.exact_part(1) != nullptr);
        // Truncated pairs contribute at most 1e-4 of their undamped bound; measured ~1e-9.
        const Real err = (Z * x - yd).norm() / yd.norm();
        INFO("Ag R2 part, automatic policy: relative matvec difference " << err);
        CHECK(err <= 1e-4);
    }
    // Forced exact far part: near + exact part reproduce the dense R2 part to round-off.
    mlfmm::MlfmmParams p = three_levels(3.0);
    p.exact_far_regions = {false, true};
    const mlfmm::MlfmmOperator Z(s.problem, p);
    const mlfmm::FarLevelInfo& f = Z.far_operator().levels(1).front();
    CHECK(f.decision == mlfmm::FarDecision::exact);
    CHECK(f.truncated_pairs == 0);
    const Real err = (Z * x - yd).norm() / yd.norm();
    INFO("Ag R2 part, forced exact: relative matvec difference " << err);
    CHECK(err <= 1e-12);
}

namespace {

/// Union of icospheres (subdivision s, radius R) centred at the given points: a closed surface of
/// several components, so that a small N spans a 4-level octree.
geometry::TriangleMesh spheres(Real radius, int subdivisions, const std::vector<Vec3>& centres) {
    const geometry::TriangleMesh one = geometry::make_icosphere(radius, subdivisions);
    const Index nv = one.num_vertices();
    const Index nt = one.num_triangles();
    const auto nc = static_cast<Index>(centres.size());
    Vertices v(nv * nc, 3);
    Triangles t(nt * nc, 3);
    for (Index s = 0; s < nc; ++s) {
        const Vec3& c = centres[static_cast<std::size_t>(s)];
        for (Index i = 0; i < nv; ++i) v.row(s * nv + i) = one.vertices().row(i) + c.transpose();
        t.middleRows(s * nt, nt) = one.triangles().array() + s * nv;
    }
    return {std::move(v), std::move(t)};
}

/// Both regions forced exact on a multi-level tree: near + exact parts reproduce the dense matrix
/// (round-off) and their basis pairs partition the N^2 pairs per region.
void check_forced_exact(const Setup& s, mlfmm::MlfmmParams p, int min_exact_levels) {
    const auto dense = op::DenseStrategy().build(s.problem);
    // Both regions forced exact store up to 2 x 40 bytes per far pair, more than the dense 64
    // bytes per pair the automatic budget allows (WP21f): diagnostics, no budget.
    p.max_exact_far_bytes = std::numeric_limits<std::size_t>::max();
    const mlfmm::MlfmmOperator Z(s.problem, p);
    INFO(Z.describe());
    const mlfmm::MlfmmFarOperator& far = Z.far_operator();
    const Index n = s.space.size();
    REQUIRE(Z.octree().levels() >= 4);
    const Index near_pairs = Z.near_operator().nonzeros() / 4;
    for (const int r : {0, 1}) {
        REQUIRE(far.exact_part(r) != nullptr);
        CHECK(near_pairs + far.exact_part(r)->pairs() == n * n);
        const mlfmm::ExactPartInfo& x = far.exact_info(r);
        CHECK(x.pairs == far.exact_part(r)->pairs());
        CHECK(x.pair_bound == x.pairs);  // forced: no per-basis truncation
        CHECK(x.truncated_pairs == 0);
        CHECK(x.counted_pairs == -1);  // under the budget: no count needed
        CHECK(x.first_level == Z.octree().leaf_level());
        int exact_levels = 0;
        for (const mlfmm::FarLevelInfo& f : far.levels(r)) {
            CHECK(f.decision != mlfmm::FarDecision::expansion);
            CHECK(f.truncated_pairs == 0);
            exact_levels += f.exact_pairs > 0 ? 1 : 0;
        }
        CHECK(exact_levels >= min_exact_levels);
    }
    for (int seed = 0; seed < 2; ++seed) {
        VectorXc x = random_vector(dense->cols());
        if (seed == 1)
            x = x.reverse().eval();
        const VectorXc yd = *dense * x;
        const Real err = (Z * x - yd).norm() / yd.norm();
        INFO("near + exact vs dense: " << err);
        CHECK(err <= 1e-12);
    }
}

}  // namespace

TEST_CASE("mlfmm policy: truncation decay exponent x*(d0)", "[mlfmm]") {
    for (const Real d0 : {1.0, 2.0, 3.0, 4.0, 5.0, 2.5}) {
        const Real x = mlfmm::truncation_decay_exponent(d0);
        const Real delta = (1.0 + x) * std::exp(-x);
        CHECK(std::abs(delta / std::pow(10.0, -(d0 + 1.0)) - 1.0) <= 1e-10);
    }
    CHECK(std::abs(mlfmm::truncation_decay_exponent(3.0) - 11.76) <= 0.01);
    CHECK(std::abs(mlfmm::truncation_decay_exponent(5.0) - 16.69) <= 0.01);
    CHECK_THROWS_AS(mlfmm::truncation_decay_exponent(0.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::truncation_decay_exponent(-1.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::truncation_decay_exponent(std::nan("")), std::invalid_argument);
}

TEST_CASE("mlfmm policy: levels after the first fallback report no search", "[mlfmm]") {
    // WP21f: a very lossy interior (n = 0.1 - 20j, all far pairs truncated: no exact part to
    // assemble) with 4 levels (leaf 0.375 lambda0, subdivision 3: r_max < a): the order search
    // fails on the leaf level, level 2 is not searched and says so ("search not run", not
    // "error 0, L = 0").
    const Complex n(0.1, -20.0);
    Setup s(geometry::make_icosphere(1.5 * kLambda, 3), {n * n, Complex(1.0, 0.0)}, Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    const InteriorOnly interior(*s.form);
    s.problem.formulation = &interior;
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, 4, 0.0};
    p.automatic_leaf_size = false;
    const mlfmm::Octree tree(s.space, kLambda, p.octree);
    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
    const std::string d = far.describe();
    INFO(d);
    REQUIRE(far.levels(1).size() == 2);
    const mlfmm::FarLevelInfo& coarse = far.levels(1)[0];
    const mlfmm::FarLevelInfo& leaf = far.levels(1)[1];
    CHECK(leaf.level == 3);
    CHECK(leaf.search_run);
    CHECK(!leaf.search_achievable);
    CHECK(leaf.decision != mlfmm::FarDecision::expansion);
    CHECK(coarse.level == 2);
    CHECK(!coarse.search_run);
    CHECK(!coarse.search_achievable);
    CHECK(d.find("search not achievable, error ") != std::string::npos);
    CHECK(d.find("search not run") != std::string::npos);
    CHECK(d.find("error 0,") == std::string::npos);
}

TEST_CASE("mlfmm policy: multi-level complementarity of the forced exact parts", "[mlfmm]") {
    // Three small spheres (subdivision 1, R = 50 nm, 2N = 720) spread over 1.4 um: 4 octree levels
    // (leaf ~0.18 um) with far pairs on levels 2 and 3, cheap enough for the sanitizer build.
    Setup s(
        spheres(0.1 * kLambda, 1,
                {Vec3(0, 0, 0), Vec3(0.9 * kLambda, 0, 0), Vec3(2.7 * kLambda, 0.6 * kLambda, 0)}),
        {Complex(2.25, 0.0), Complex(1.0, 0.0)}, Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, 4, 0.0};
    p.automatic_leaf_size = false;
    p.exact_far_regions = {true, true};
    check_forced_exact(s, p, 2);
}

TEST_CASE("mlfmm policy: forced exact parts on a 4-level icosphere", "[mlfmm]") {
#ifndef NDEBUG
    SKIP("2N = 3840 dense oracle: optimised builds only (the three-sphere case covers debug)");
#else
    // Icosphere R = 1.5 lambda0, subdivision 3 (2N = 3840), n = 1.5, 4 levels (leaf 0.375
    // lambda0, r_max / a ~ 0.5): exact pairs on levels 2 and 3.
    Setup s(geometry::make_icosphere(1.5 * kLambda, 3), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
            Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, 4, 0.0};
    p.automatic_leaf_size = false;
    p.exact_far_regions = {true, true};
    check_forced_exact(s, p, 2);
#endif
}

TEST_CASE("mlfmm policy: error causes lossy_region and digits", "[mlfmm]") {
    // Icosphere subdivision 3 with 3 levels (r_max / a ~ 0.25-0.29, below max_support_ratio for
    // both d0): R = lambda0 gives lambda / 2 leaves, R = lambda0 / 2 lambda / 4 leaves.
    const auto geo = [](Real radius) { return geometry::make_icosphere(radius, 3); };
    mlfmm::MlfmmParams p = three_levels(3.0);
    const auto expect = [&](const Setup& s, mlfmm::TruncationOrderError::Cause cause, int region) {
        const mlfmm::Octree tree(s.space, kLambda, p.octree);
        INFO("leaf a = " << tree.box_size(tree.leaf_level()) / kLambda << " lambda0, r_max / a = "
                         << mlfmm::max_support_radius(s.space) / tree.box_size(tree.leaf_level()));
        try {
            const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
            FAIL("expected TruncationOrderError\n" << far.describe());
        } catch (const mlfmm::TruncationOrderError& e) {
            INFO(e.what());
            CHECK(e.cause() == cause);
            CHECK(e.region() == region);
            CHECK(std::string(e.what()).find("x*/alpha") != std::string::npos);
            CHECK(std::string(e.what()).find("-1.00e+00") == std::string::npos);
        }
    };
    // n = 1.5 - 0.6j (alpha = 7.5 / um) at lambda / 2 leaves: the loss spoils the expansion
    // (alpha D ~ 4) and the exact pairs would reach x* / alpha = 1.6 um = 6.2 leaf edges > 4
    // (before the WP21 review: accepted since alpha a = 1.9 >= 1, with most pairs exact).
    {
        const Complex n(1.5, -0.6);
        Setup s(geo(kLambda), {n * n, Complex(1.0, 0.0)}, Kind::PMCHWT);
        s.problem.kernel_options = cheap_options();
        const InteriorOnly interior(*s.form);
        s.problem.formulation = &interior;  // region R2 only
        expect(s, mlfmm::TruncationOrderError::Cause::lossy_region, 1);
    }
    // Lossless interior at d0 = 5 with lambda / 4 leaves: no expansion order (ADR 0008: d0 = 5
    // needs leaves >= lambda / 2) at a fine mesh, no decay: cause digits, region R1 (planned
    // first).
    {
        p = three_levels(5.0);
        Setup s(geo(0.5 * kLambda), {Complex(2.25, 0.0), Complex(1.0, 0.0)}, Kind::PMCHWT);
        s.problem.kernel_options = cheap_options();
        expect(s, mlfmm::TruncationOrderError::Cause::digits, 0);
    }
}

TEST_CASE("mlfmm policy: exact-part budget", "[mlfmm]") {
    // The Ag case of "Ag falls back to exact pairs": with a budget of 1 kB the estimate exceeds
    // it before anything is assembled.
    Setup s(geometry::make_icosphere(1.5 * kLambda, 2), material::silver_500nm(), Kind::PMCHWT);
    s.problem.kernel_options = cheap_options();
    const InteriorOnly interior(*s.form);
    s.problem.formulation = &interior;  // region R2 only: no exterior block check
    mlfmm::MlfmmParams p = three_levels(3.0);
    const mlfmm::Octree tree(s.space, kLambda, p.octree);
    {
        const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
        INFO(far.describe());
        const mlfmm::ExactPartInfo& x = far.exact_info(1);
        CHECK(x.first_level == 2);
        CHECK(x.box_pairs == far.levels(1).front().exact_pairs);
        CHECK(x.pair_bound >= x.pairs + x.truncated_pairs);
        REQUIRE(far.exact_part(1) != nullptr);
        CHECK(x.pairs == far.exact_part(1)->pairs());
        CHECK(x.bytes == far.exact_part(1)->memory_bytes());
        // Automatic budget min(max(2 x near, 1 GiB), dense): for this small problem the dense
        // matrix size 16 (2N)^2 (WP21f; before: the 1 GiB floor, larger than dense).
        const auto N = static_cast<std::size_t>(s.space.size());
        const std::size_t dense = 16 * 4 * N * N;
        CHECK(dense < mlfmm::kExactFarMinBytes);
        CHECK(far.exact_budget() == dense);
        CHECK(far.exact_budget() ==
              std::min(std::max(static_cast<std::size_t>(
                                    mlfmm::kExactFarNearFactor *
                                    static_cast<Real>(mlfmm::estimate_near_bytes(tree))),
                                mlfmm::kExactFarMinBytes),
                       dense));
        // estimate_near_bytes equals the assembled near field.
        CHECK(mlfmm::estimate_near_bytes(tree) ==
              mlfmm::assemble_near(s.problem, tree)->memory_bytes());
        p.max_exact_far_bytes = x.bytes;  // exactly enough
        const mlfmm::MlfmmFarOperator fits(s.problem, tree, p);
        CHECK(fits.exact_budget() == x.bytes);
        CHECK(fits.exact_info(1).counted_pairs == x.pairs);  // bound > budget: counted
        CHECK(fits.exact_info(1).pairs == x.pairs);
    }
    p.max_exact_far_bytes = 1000;
    try {
        const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
        FAIL("expected TruncationOrderError");
    } catch (const mlfmm::TruncationOrderError& e) {
        INFO(e.what());
        CHECK(e.cause() == mlfmm::TruncationOrderError::Cause::exact_part_too_large);
        CHECK(e.region() == 1);
        CHECK(e.level() == 2);
        CHECK(std::string(e.what()).find("max_exact_far_bytes") != std::string::npos);
    }
}

TEST_CASE("mlfmm policy: automatic leaf rule", "[mlfmm]") {
    Setup s(geometry::make_icosphere(kLambda, 2), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
            Kind::PMCHWT);
    const Real rmax = mlfmm::max_support_radius(s.space);
    mlfmm::MlfmmParams p;
    p.octree.min_box_size_lambda = 0.1;
    for (const Real d0 : {2.0, 3.0, 4.0, 5.0}) {
        p.accuracy_digits = d0;
        const Real floor = mlfmm::leaf_rule_params(s.space, kLambda, p).min_box_size_lambda;
        const Real expected = std::max(
            d0 <= 3.0 ? 0.25 : 0.5,
            rmax / (mlfmm::max_support_ratio(d0) * (1.0 - mlfmm::kMinBoxSizeTolerance) * kLambda));
        CHECK(std::abs(floor - expected) <= 1e-12 * expected);
    }
    // WP21 review: 0.6 for d0 <= 3 (lambda / 4 leaves at r_max / a ~ 0.57 meet docs/05), 0.3 above.
    CHECK(mlfmm::max_support_ratio(2.0) == 0.6);
    CHECK(mlfmm::max_support_ratio(3.0) == 0.6);
    CHECK(mlfmm::max_support_ratio(3.5) == 0.3);
    CHECK(mlfmm::max_support_ratio(5.0) == 0.3);
    p.octree.min_box_size_lambda = 5.0;  // the user value is a lower bound
    CHECK(mlfmm::leaf_rule_params(s.space, kLambda, p).min_box_size_lambda == 5.0);
    p.automatic_leaf_size = false;
    p.octree.min_box_size_lambda = 0.1;
    CHECK(mlfmm::leaf_rule_params(s.space, kLambda, p).min_box_size_lambda == 0.1);
    p.automatic_leaf_size = true;
    p.accuracy_digits = 0.0;
    CHECK_THROWS_AS(mlfmm::leaf_rule_params(s.space, kLambda, p), std::invalid_argument);
    p.accuracy_digits = 3.0;
    CHECK_THROWS_AS(mlfmm::leaf_rule_params(s.space, 0.0, p), std::invalid_argument);
}

TEST_CASE("mlfmm policy: workspace pool exception safety and the error type", "[mlfmm]") {
    static_assert(std::is_base_of_v<std::runtime_error, mlfmm::TruncationOrderError>);
    Setup s(geometry::make_icosphere(1.5 * kLambda, 2), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
            Kind::PMCHWT);
    const mlfmm::MlfmmParams p = three_levels(3.0);
    const mlfmm::Octree tree(s.space, kLambda, p.octree);
    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
    const VectorXc x = random_vector(far.cols());
    VectorXc y;
    mlfmm::testing::fail_next_workspace_allocations(1);
    CHECK_THROWS_AS(far.apply(x, y), std::bad_alloc);
    CHECK(far.workspaces() == 0);  // a failed allocation is not counted
    far.apply(x, y);
    CHECK(far.workspaces() == 1);
    VectorXc y2;
    far.apply(x, y2);  // the pooled workspace is reused
    CHECK(far.workspaces() == 1);
    CHECK((y.array() == y2.array()).all());
    mlfmm::testing::fail_next_workspace_allocations(0);
}

TEST_CASE("mlfmm policy: basis_patterns equals the leaf patterns", "[mlfmm]") {
    Setup s(geometry::make_icosphere(1.5 * kLambda, 2), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
            Kind::PMCHWT);
    const mlfmm::Octree tree(s.space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const Complex k = s.problem.object.wavenumber(s.problem.omega);
    const mlfmm::SphereSampling sampling(9);
    const mlfmm::RadiationPatterns pat(s.space, tree, k, sampling);
    const mlfmm::Box& box =
        tree.boxes()[static_cast<std::size_t>(tree.boxes_at_level(tree.leaf_level())[3])];
    const std::vector<Index> bases(
        tree.permutation().begin() + box.first_element,
        tree.permutation().begin() + box.first_element + box.num_elements);
    const std::vector<Complex> v = mlfmm::basis_patterns(s.space, bases, box.center, k, sampling);
    const auto nd = static_cast<std::size_t>(sampling.size());
    REQUIRE(v.size() == bases.size() * nd * 2);
    bool equal = true;
    for (std::size_t i = 0; i < v.size(); ++i)
        equal =
            equal && v[i] == pat.data()[static_cast<std::size_t>(box.first_element) * nd * 2 + i];
    CHECK(equal);
    const std::vector<Index> bad = {s.space.size()};
    CHECK_THROWS_AS(mlfmm::basis_patterns(s.space, bad, box.center, k, sampling),
                    std::out_of_range);
    CHECK_THROWS_AS(mlfmm::basis_patterns(s.space, bases, box.center, Complex(1.0, 1.0), sampling),
                    std::invalid_argument);
}

TEST_CASE("mlfmm policy sweep: per-level decisions", "[.][mlfmm_policy_sweep]") {
    struct Mat {
        const char* name;
        material::Material m;
    };
    const Mat mats[] = {{"n=1.5", {Complex(2.25, 0.0), Complex(1.0, 0.0)}},
                        {"Si", material::silicon_500nm()},
                        {"Ag", material::silver_500nm()}};
    struct Geo {
        const char* name;
        geometry::TriangleMesh mesh;
    };
    Geo geos[] = {{"sphere R=1um n4", geometry::make_icosphere(2.0 * kLambda, 4)},
                  {"sphere R=0.5um n4", geometry::make_icosphere(kLambda, 4)},
                  {"box 2um 50nm", rough_box(2e-6, 50e-9)}};
    for (Geo& g : geos) {
        for (const Mat& mt : mats) {
            for (const Real d0 : {3.0, 5.0}) {
                const Setup s(g.mesh, mt.m, Kind::PMCHWT);
                mlfmm::MlfmmParams p;
                p.accuracy_digits = d0;
                p.octree.max_elements_per_leaf = 4;
                const mlfmm::Octree tree(s.space, kLambda,
                                         mlfmm::leaf_rule_params(s.space, kLambda, p));
                try {
                    const mlfmm::MlfmmFarOperator far(s.problem, tree, p);
                    WARN(g.name << ", " << mt.name << ", d0 = " << d0 << ", leaf "
                                << tree.box_size(tree.leaf_level()) / kLambda << " lambda0\n"
                                << far.describe());
                } catch (const std::exception& e) {
                    WARN(g.name << ", " << mt.name << ", d0 = " << d0 << ": " << e.what());
                }
            }
        }
    }
}
