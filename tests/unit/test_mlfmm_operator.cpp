/// Unit tests of the full MLFMM operator Z = Z_near + Z_far (WP20b): complementarity of the near
/// and far parts, the Jacobi diagonal, concurrent applies and the workspace pool, the jump-term
/// guard, describe() / memory_bytes() and the strategy. Small icospheres (lambda0 = 500 nm) with
/// cheap kernel options (the structural checks are exact for any options; accuracy against dense
/// is the validation tests' job, tests/validation*/test_mlfmm_*.cpp).
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace specklebem;
using formulation::Kind;
using mlfmm::MlfmmOperator;

namespace {

constexpr Real kLambda = 500e-9;

kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = 1;
    opt.quad_degree_near = 2;
    opt.quad_degree_sing = 2;
    return opt;
}

/// Mesh, space and Problem (vacuum exterior, n = 1.5 interior). Not movable.
struct Setup {
    Setup(geometry::TriangleMesh m, Kind kind)
        : mesh(std::move(m)), space(mesh), form(formulation::make_formulation(kind)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = {Complex(2.25, 0.0), Complex(1.0, 0.0)};
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
        problem.kernel_options = cheap_options();
    }
    Setup(const Setup&) = delete;
    Setup& operator=(const Setup&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
};

/// Exactly `levels` levels (one element per leaf requested, no size floor), d0 digits.
mlfmm::MlfmmParams params(int levels, Real digits) {
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, levels, 0.0};
    p.accuracy_digits = digits;
    return p;
}

/// Icosphere R = lambda0 / 2, subdivision 2 (2N = 960), 3 levels: leaf a = lambda0 / 4,
/// r_max / a ~ 0.5.
geometry::TriangleMesh small_sphere() {
    return geometry::make_icosphere(0.5 * kLambda, 2);
}

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

/// Leaf box of every basis function.
std::vector<Index> leaf_of(const mlfmm::Octree& tree) {
    std::vector<Index> leaf(tree.permutation().size(), -1);
    for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
        const mlfmm::Box& box = tree.boxes()[static_cast<std::size_t>(b)];
        for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p)
            leaf[static_cast<std::size_t>(tree.permutation()[static_cast<std::size_t>(p)])] = b;
    }
    return leaf;
}

}  // namespace

TEST_CASE("mlfmm: near and far parts of MlfmmOperator are complementary", "[mlfmm]") {
    // For unit columns e_j (J and M columns): the near column is stored exactly on the rows whose
    // leaf touches the leaf of j and equals the exact entries there (bitwise); the far column is
    // exactly zero there and non-zero on every other row; near + far reproduces the exact column.
    Setup s(small_sphere(), Kind::PMCHWT);
    // Accurate near / far pairs (k-aware degrees for 1e-5), cheap touching pairs: the far
    // columns are then compared with entries of similar accuracy as the radiation patterns.
    s.problem.kernel_options.target_accuracy = 1e-5;
    const MlfmmOperator Z(s.problem, params(3, 3.0));
    REQUIRE(Z.octree().levels() == 3);
    const Index n = s.space.size();
    const std::vector<Index> basis = {0, n / 3, (2 * n) / 3, n - 1};
    // Exact entries of the columns basis (and + N), all rows: one group with these columns.
    op::BasisPattern pat;
    pat.group_of_row.assign(static_cast<std::size_t>(n), 0);
    pat.cols = basis;
    pat.col_ptr = {0, static_cast<Index>(basis.size())};
    const auto exact = op::assemble_sparse(s.problem, pat);
    const std::vector<Index> leaf = leaf_of(Z.octree());
    const auto& boxes = Z.octree().boxes();
    const auto& near = Z.near_operator().matrix();
    for (const Index b : basis) {
        const auto& lb = boxes[static_cast<std::size_t>(leaf[static_cast<std::size_t>(b)])].ijk;
        for (const Index col : {b, n + b}) {
            const VectorXc e = VectorXc::Unit(2 * n, col);
            const VectorXc yn = Z.near_operator() * e;
            const VectorXc yf = Z.far_operator() * e;
            const VectorXc yz = Z * e;
            const VectorXc ex = exact->matrix().col(col);
            Index near_rows = 0, far_rows = 0;
            bool structure = true, near_exact = true, far_zero = true, far_nonzero = true;
            for (Index r = 0; r < 2 * n; ++r) {
                const auto& lr =
                    boxes[static_cast<std::size_t>(leaf[static_cast<std::size_t>(r % n)])].ijk;
                const bool is_near = std::abs(lr[0] - lb[0]) <= 1 && std::abs(lr[1] - lb[1]) <= 1 &&
                                     std::abs(lr[2] - lb[2]) <= 1;
                const bool stored =
                    near.coeff(r, col) != Complex(0.0, 0.0) || yn(r) != Complex(0.0, 0.0);
                structure = structure && stored == is_near;
                if (is_near) {
                    ++near_rows;
                    near_exact = near_exact && yn(r) == ex(r);
                    far_zero = far_zero && yf(r) == Complex(0.0, 0.0);
                } else {
                    ++far_rows;
                    far_nonzero = far_nonzero && yf(r) != Complex(0.0, 0.0);
                }
            }
            INFO("column " << col << ": " << near_rows << " near rows, " << far_rows
                           << " far rows, |Z e - exact| / |exact| = "
                           << (yz - ex).norm() / ex.norm());
            CHECK(structure);
            CHECK(near_exact);
            CHECK(far_zero);
            CHECK(far_nonzero);
            CHECK(far_rows > 0);
            CHECK(bitwise_equal(yz, yn + yf));
            // Single columns, not the docs/05 metric (random x: tests/validation_large); measured
            // < 1e-3 (J columns) and 1.6e-3 (M columns: K far-only error at lambda / 4 leaves).
            CHECK((yz - ex).norm() <= 5e-3 * ex.norm());
        }
    }
}

TEST_CASE("mlfmm: the near field's diagonal is the Jacobi diagonal", "[mlfmm]") {
    // Simulation keeps op::assemble_diagonal for the Jacobi preconditioner; both are bitwise the
    // dense diagonal (same blocks and summation order), so either source gives the same solve.
    const Setup s(small_sphere(), Kind::ICTF);
    const MlfmmOperator Z(s.problem, params(3, 3.0));
    const VectorXc near_diag = Z.near_operator().matrix().diagonal();
    CHECK(bitwise_equal(near_diag, op::assemble_diagonal(s.problem)));
}

TEST_CASE("mlfmm: concurrent applies equal serial applies", "[mlfmm]") {
    SECTION("MlfmmOperator, 3 levels") {
        const Setup s(small_sphere(), Kind::PMCHWT);
        const MlfmmOperator Z(s.problem, params(3, 3.0));
        const std::array<VectorXc, 2> x = {random_vector(Z.cols(), 1), random_vector(Z.cols(), 2)};
        const std::array<VectorXc, 2> serial = {Z * x[0], Z * x[1]};
        CHECK(Z.far_operator().workspaces() == 1);  // serial calls reuse one workspace
        std::array<std::array<VectorXc, 3>, 2> y;
        {
            std::array<std::thread, 2> t;
            for (std::size_t i = 0; i < 2; ++i) {
                t[i] = std::thread([&, i]() {
                    for (VectorXc& yi : y[i]) Z.apply(x[i], yi);
                });
            }
            for (std::thread& ti : t) ti.join();
        }
        for (std::size_t i = 0; i < 2; ++i) {
            for (const VectorXc& yi : y[i]) CHECK(bitwise_equal(yi, serial[i]));
        }
        CHECK(Z.far_operator().workspaces() >= 1);
        CHECK(Z.far_operator().workspaces() <= 2);
    }
    SECTION("far operator, 4 levels (upward / downward passes on reused workspaces)") {
        // A reused workspace holds the previous apply's fields: every pass must zero what it
        // accumulates into. Alternating inputs on one workspace and on two threads.
        const Setup s(geometry::make_icosphere(0.5 * kLambda, 3), Kind::PMCHWT);
        const mlfmm::Octree tree(s.space, kLambda, mlfmm::OctreeParams{1, 4, 0.0});
        const mlfmm::MlfmmFarOperator far(s.problem, tree, params(4, 2.0));
        REQUIRE(far.levels(0).size() == 2);
        const std::array<VectorXc, 2> x = {random_vector(far.cols(), 3),
                                           random_vector(far.cols(), 4)};
        const VectorXc y0 = far * x[0];
        const VectorXc y1 = far * x[1];
        CHECK(bitwise_equal(far * x[0], y0));
        CHECK(far.workspaces() == 1);
        std::array<VectorXc, 2> y;
        std::thread t0([&]() { far.apply(x[0], y[0]); });
        std::thread t1([&]() { far.apply(x[1], y[1]); });
        t0.join();
        t1.join();
        CHECK(bitwise_equal(y[0], y0));
        CHECK(bitwise_equal(y[1], y1));
        CHECK(far.workspaces() <= 2);
        CHECK(far.workspace_bytes() > 0);
    }
}

TEST_CASE("mlfmm: MlfmmOperator describe, memory, aliasing and errors", "[mlfmm]") {
    const Setup s(small_sphere(), Kind::PMCHWT);
    const mlfmm::MlfmmStrategy strategy(params(3, 3.0));
    CHECK(strategy.name() == "mlfmm");
    const std::shared_ptr<op::LinearOperator> op = strategy.build(s.problem);
    const auto* Z = dynamic_cast<const MlfmmOperator*>(op.get());
    REQUIRE(Z != nullptr);
    CHECK(Z->rows() == 2 * s.space.size());
    CHECK(Z->cols() == Z->rows());
    const VectorXc x = random_vector(Z->cols(), 9);
    VectorXc y = x;
    Z->apply(y, y);  // aliasing
    CHECK(bitwise_equal(y, *Z * x));
    VectorXc bad;
    CHECK_THROWS_AS(Z->apply(VectorXc::Zero(3), bad), std::invalid_argument);
    const std::string d = Z->describe();
    INFO(d);
    for (const char* what : {"MlfmmOperator", "near: sparse", "far:", "level 2", "total"})
        CHECK(d.find(what) != std::string::npos);
    CHECK(Z->memory_bytes() > Z->near_operator().memory_bytes() + Z->far_operator().memory_bytes());
    CHECK(Z->far_operator().memory_bytes() > Z->far_operator().workspace_bytes());
}

TEST_CASE("mlfmm: far pairs must not share triangles (jump-term guard)", "[mlfmm]") {
    // Icosphere R = lambda0, subdivision 1 (edges ~ 0.55 lambda0) with 4 levels: leaf a =
    // lambda0 / 4 < r_max, so far pairs could share a triangle and miss the K jump term.
    const Setup s(geometry::make_icosphere(kLambda, 1), Kind::PMCHWT);
    CHECK_THROWS_MATCHES(
        MlfmmOperator(s.problem, params(4, 3.0)), std::invalid_argument,
        Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("support radius")));
    // Two levels: no far part, Z = Z_near = the dense matrix (every pair is near).
    const MlfmmOperator two(s.problem, params(2, 3.0));
    CHECK(two.far_operator().workspace_bytes() == 0);
    const VectorXc x = random_vector(two.cols(), 4);
    CHECK(bitwise_equal(two * x, two.near_operator() * x));
    op::Problem no_space = s.problem;
    no_space.space = nullptr;
    CHECK_THROWS_AS(MlfmmOperator(no_space, params(3, 3.0)), std::invalid_argument);
    CHECK_THROWS_AS(MlfmmOperator(s.problem, params(0, 3.0)), std::invalid_argument);
}
