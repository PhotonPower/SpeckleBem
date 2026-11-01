/// Unit tests of the stage statistics of the MLFMM operator (WP-PF1, the Phase 7 profile):
/// MlfmmOperator::setup_stats / apply_stats, MlfmmFarOperator::setup_stats / apply_stats and
/// PmlRingOperator::apply_stats. The stage seconds are >= 0, their sums stay below the measured
/// totals, the counters count every call (also concurrent ones), reset clears them, and the
/// timing never changes the arithmetic: apply() with timing on and off gives bitwise equal
/// results. A small icosphere with four octree levels (so that the upward and downward passes
/// run) and the tiny flat PML case of tests/unit/test_pml_ring.cpp; cheap kernel options (the
/// checks are structural).
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace specklebem;
using formulation::Kind;
using mlfmm::MlfmmOperator;

namespace {

constexpr Real kLambda = 500e-9;
/// Slack of the "parts <= total" checks: the parts are read from the same clock inside the
/// measured interval, so only the rounding of the summed doubles can make them exceed it.
constexpr Real kSlack = 1e-9;

kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = 1;
    opt.quad_degree_near = 2;
    opt.quad_degree_sing = 2;
    opt.stretched_touching_points = 4;
    return opt;
}

/// Exactly `levels` levels (one element per leaf requested, no size floor), d0 digits.
mlfmm::MlfmmParams params(int levels, Real digits) {
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, levels, 0.0};
    p.accuracy_digits = digits;
    p.automatic_leaf_size = false;  // deliberately small trees
    return p;
}

/// Icosphere mesh, space and Problem (vacuum exterior, n = 1.5 interior). Not movable.
struct SphereSetup {
    SphereSetup()
        : mesh(geometry::make_icosphere(kLambda, 3)),
          space(mesh),
          form(formulation::make_formulation(Kind::PMCHWT)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = {Complex(2.25, 0.0), Complex(1.0, 0.0)};
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
        problem.kernel_options = cheap_options();
    }
    SphereSetup(const SphereSetup&) = delete;
    SphereSetup& operator=(const SphereSetup&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
};

/// The tiny flat open surface with PML of tests/unit/test_pml_ring.cpp (h = 100 nm).
geometry::OpenSurfaceMesh tiny_surface() {
    geometry::OpenSurfaceParams op;
    op.patch_size = 0.2e-6;
    op.taper_width = 0.1e-6;
    op.gap_width = 0.1e-6;
    op.pml_width = 0.1e-6;
    const Real edge = op.patch_size + 2.0 * op.taper_width;
    const auto cells = static_cast<Index>(std::llround(edge / 100e-9));
    geometry::HeightMap m;
    m.dx = edge / static_cast<Real>(cells);
    m.dy = m.dx;
    m.z = MatrixXr::Zero(cells + 1, cells + 1);
    return geometry::make_open_surface_mesh(m, op);
}

/// Open mesh, space, PML stretch and Problem (vacuum over n = 1.5). Not movable.
struct PmlSetup {
    PmlSetup()
        : om(tiny_surface()),
          space(om.mesh),
          form(formulation::make_formulation(Kind::PMCHWT)),
          pml(geometry::make_pml_stretch(om, 1.0)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = {Complex(2.25, 0.0), Complex(1.0, 0.0)};
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
        problem.kernel_options = cheap_options();
        problem.stretch = &pml;
    }
    PmlSetup(const PmlSetup&) = delete;
    PmlSetup& operator=(const PmlSetup&) = delete;

    geometry::OpenSurfaceMesh om;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    geometry::CoordinateStretch pml;
    op::Problem problem;
};

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

bool bitwise_equal(const VectorXc& a, const VectorXc& b) {
    return a.size() == b.size() && (a.array() == b.array()).all();
}

Real far_pass_sum(const mlfmm::FarApplyStats& f) {
    Real s = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        s += f.aggregation[i] + f.upward[i] + f.translation[i] + f.downward[i] + f.reception[i] +
             f.exact[i];
    }
    return s;
}

/// Every far pass time >= 0, passes + other = total, total > 0 after a call.
void check_far_stats(const mlfmm::FarApplyStats& f, Index applies) {
    CHECK(f.applies == applies);
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK(f.aggregation[i] >= 0.0);
        CHECK(f.upward[i] >= 0.0);
        CHECK(f.translation[i] >= 0.0);
        CHECK(f.downward[i] >= 0.0);
        CHECK(f.reception[i] >= 0.0);
        CHECK(f.exact[i] >= 0.0);
    }
    CHECK(f.other >= -kSlack);
    CHECK(far_pass_sum(f) <= f.total + kSlack);
    CHECK(std::abs(far_pass_sum(f) + f.other - f.total) <= kSlack);
    if (applies > 0)
        CHECK(f.total > 0.0);
}

void check_operator_stats(const mlfmm::MlfmmApplyStats& s, Index applies) {
    CHECK(s.applies == applies);
    CHECK(s.near >= 0.0);
    CHECK(s.far >= 0.0);
    CHECK(s.ring >= 0.0);
    CHECK(s.other >= -kSlack);
    CHECK(s.near + s.far + s.ring <= s.total + kSlack);
    CHECK(std::abs(s.near + s.far + s.ring + s.other - s.total) <= kSlack);
    if (applies > 0)
        CHECK(s.total > 0.0);
}

}  // namespace

TEST_CASE("mlfmm: setup stage seconds are non-negative and below the totals", "[mlfmm]") {
    SphereSetup s;
    const MlfmmOperator Z(s.problem, params(4, 3.0));
    REQUIRE(Z.octree().levels() == 4);
    const mlfmm::MlfmmSetupStats& st = Z.setup_stats();
    CHECK(st.tree >= 0.0);
    CHECK(st.plan == 0.0);  // no stretch
    CHECK(st.far > 0.0);
    CHECK(st.near > 0.0);
    CHECK(st.ring == 0.0);
    CHECK(st.tree + st.plan + st.far + st.near + st.ring <= st.total + kSlack);
    const mlfmm::FarSetupStats& f = Z.far_operator().setup_stats();
    Real parts = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK(f.plan[i] >= 0.0);
        CHECK(f.exact[i] >= 0.0);
        CHECK(f.translators[i] >= 0.0);
        CHECK(f.patterns[i] >= 0.0);
        CHECK(f.exact[i] == Z.far_operator().exact_info(static_cast<int>(i)).seconds);
        parts += f.plan[i] + f.exact[i] + f.translators[i] + f.patterns[i];
    }
    CHECK(f.patterns[0] > 0.0);  // both regions expand on this tree
    CHECK(f.patterns[1] > 0.0);
    CHECK(parts <= f.total + kSlack);
    CHECK(f.total <= st.far + kSlack);
}

TEST_CASE("mlfmm: apply stage statistics count, sum and reset, timing keeps results bitwise",
          "[mlfmm]") {
    SphereSetup s;
    const MlfmmOperator Z(s.problem, params(4, 3.0));
    const mlfmm::MlfmmFarOperator& far = Z.far_operator();
    REQUIRE(far.levels(0).size() == 2);  // levels 2 and 3: the upward / downward passes run
    REQUIRE(far.levels(0).front().decision == mlfmm::FarDecision::expansion);
    CHECK(Z.stage_timing());
    CHECK(far.stage_timing());
    check_operator_stats(Z.apply_stats(), 0);
    check_far_stats(far.apply_stats(), 0);

    const VectorXc x = random_vector(Z.cols(), 11);
    const VectorXc y1 = Z * x;
    const VectorXc y2 = Z * x;
    const VectorXc y3 = Z * x;
    CHECK(bitwise_equal(y1, y2));
    CHECK(bitwise_equal(y1, y3));
    const mlfmm::MlfmmApplyStats st = Z.apply_stats();
    check_operator_stats(st, 3);
    CHECK(st.ring == 0.0);
    const mlfmm::FarApplyStats fs = far.apply_stats();
    check_far_stats(fs, 3);
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK(fs.aggregation[i] > 0.0);
        CHECK(fs.translation[i] > 0.0);
        CHECK(fs.reception[i] > 0.0);
    }
    // The far operator's own total is measured inside MlfmmOperator's far interval.
    CHECK(fs.total <= st.far + kSlack);

    // Timing off: no clock reads, statistics unchanged, results bitwise equal.
    Z.reset_apply_stats();
    check_operator_stats(Z.apply_stats(), 0);
    check_far_stats(far.apply_stats(), 0);
    Z.set_stage_timing(false);
    CHECK_FALSE(Z.stage_timing());
    CHECK_FALSE(far.stage_timing());
    const VectorXc y_off = Z * x;
    const VectorXc yf_off = far * x;
    CHECK(Z.apply_stats().applies == 0);
    CHECK(Z.apply_stats().total == 0.0);
    CHECK(far.apply_stats().applies == 0);
    CHECK(far.apply_stats().total == 0.0);
    CHECK(bitwise_equal(y_off, y1));
    Z.set_stage_timing(true);
    const VectorXc yf_on = far * x;
    CHECK(bitwise_equal(yf_on, yf_off));
    CHECK(far.apply_stats().applies == 1);
    CHECK(Z.apply_stats().applies == 0);  // the far part alone does not count for Z

    // Thread-safe accumulation: concurrent calls each count once.
    Z.reset_apply_stats();
    std::vector<VectorXc> out(4);
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < out.size(); ++i) th.emplace_back([&, i] { Z.apply(x, out[i]); });
    for (auto& t : th) t.join();
    for (const VectorXc& o : out) CHECK(bitwise_equal(o, y1));
    check_operator_stats(Z.apply_stats(), 4);
    check_far_stats(far.apply_stats(), 4);
}

TEST_CASE("mlfmm: PML ring and plan stage statistics, timing keeps results bitwise", "[mlfmm]") {
    PmlSetup s;
    const MlfmmOperator Z(s.problem, params(3, 3.0));
    const mlfmm::PmlRingOperator* ring = Z.ring_operator();
    REQUIRE(ring != nullptr);
    const mlfmm::MlfmmSetupStats& st = Z.setup_stats();
    CHECK(st.plan > 0.0);
    CHECK(st.ring > 0.0);
    CHECK(st.tree + st.plan + st.far + st.near + st.ring <= st.total + kSlack);
    CHECK(ring->near_seconds() + ring->far_seconds() <= st.ring + kSlack);

    const VectorXc x = random_vector(Z.cols(), 5);
    const VectorXc y1 = Z * x;
    const VectorXc y2 = Z * x;
    CHECK(bitwise_equal(y1, y2));
    const mlfmm::MlfmmApplyStats a = Z.apply_stats();
    check_operator_stats(a, 2);
    CHECK(a.ring > 0.0);
    const mlfmm::PmlRingApplyStats r = ring->apply_stats();
    CHECK(r.applies == 2);
    CHECK(r.near >= 0.0);
    CHECK(r.far >= 0.0);
    CHECK(r.near + r.far <= r.total + kSlack);
    CHECK(r.total <= a.ring + kSlack);
    check_far_stats(Z.far_operator().apply_stats(), 2);

    // Off: bitwise equal results for the whole operator and the ring alone, nothing counted.
    Z.reset_apply_stats();
    CHECK(ring->apply_stats().applies == 0);
    const VectorXc yr_on = *ring * x;
    CHECK(ring->apply_stats().applies == 1);
    Z.set_stage_timing(false);
    CHECK_FALSE(ring->stage_timing());
    const VectorXc y_off = Z * x;
    const VectorXc yr_off = *ring * x;
    CHECK(bitwise_equal(y_off, y1));
    CHECK(bitwise_equal(yr_off, yr_on));
    CHECK(ring->apply_stats().applies == 1);
    CHECK(Z.apply_stats().applies == 0);
    // An aliased call is counted once (the inner call on the copy).
    Z.set_stage_timing(true);
    Z.reset_apply_stats();
    VectorXc z = x;
    ring->apply(z, z);
    CHECK(bitwise_equal(z, yr_on));
    CHECK(ring->apply_stats().applies == 1);
}
