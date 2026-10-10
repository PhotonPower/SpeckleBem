/// Unit tests of the MLFMM region policy (ADR 0008 §6, WP21): expansion / truncation / exact
/// decisions, per-region complementarity of the exact part, the leaf rule, the workspace pool's
/// exception safety and the error type. Icospheres at lambda0 = 500 nm.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
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
            d0 <= 3.0 ? 0.25 : 0.5, rmax / (0.3 * (1.0 - mlfmm::kMinBoxSizeTolerance) * kLambda));
        CHECK(std::abs(floor - expected) <= 1e-12 * expected);
    }
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
