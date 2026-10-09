#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/dense_operator.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;
using formulation::Kind;
using mlfmm::MlfmmFarOperator;
using mlfmm::Octree;

namespace {

constexpr Real kLambda = 500e-9;

Real omega0() {
    return 2.0 * constants::pi * constants::c0 / kLambda;
}

material::Material medium(Real n) {
    return {Complex(n * n, 0.0), Complex(1.0, 0.0)};
}

/// Mesh, RWG space, octree with exactly `levels` levels (one element per leaf requested, no
/// size floor) and the Problem (vacuum exterior). Not movable: the members reference each other.
struct Case {
    Case(geometry::TriangleMesh m, int levels, const material::Material& object, Kind kind,
         Real digits)
        : mesh(std::move(m)),
          space(mesh),
          tree(space, kLambda, mlfmm::OctreeParams{1, levels, 0.0}),
          form(formulation::make_formulation(kind)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = object;
        problem.formulation = form.get();
        problem.omega = omega0();
        // Dense quadrature target 0.1 x 10^-d0 (measured errors 0.05 ... 0.5 x the target).
        problem.kernel_options.target_accuracy = std::min(1e-4, 0.1 * std::pow(10.0, -digits));
        mlfmm::MlfmmParams p;
        p.accuracy_digits = digits;
        params = p;
    }
    Case(const Case&) = delete;
    Case& operator=(const Case&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    Octree tree;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
    mlfmm::MlfmmParams params;
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

/// Fixed-seed random vector, uniforms from the raw 64-bit output (identical on every library).
VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Leaf box of every basis function.
std::vector<Index> leaf_of(const Octree& tree) {
    std::vector<Index> leaf(tree.permutation().size(), -1);
    for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
        const mlfmm::Box& box = tree.boxes()[static_cast<std::size_t>(b)];
        for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
            leaf[static_cast<std::size_t>(tree.permutation()[static_cast<std::size_t>(p)])] = b;
        }
    }
    return leaf;
}

struct FarResult {
    Real far_err = 0;   ///< |y_far,MLFMM - y_far,dense| / |y_far,dense|
    Real full_err = 0;  ///< same difference / |Z_dense x| (docs/05 matvec metric)
    Real t_setup = 0, t_apply = 0, t_dense = 0;
    std::string describe;
    std::unique_ptr<MlfmmFarOperator> op;
};

/// Dense oracle: y_far = Z x - Z_near x, Z_near = the entries of near basis pairs (same leaf or
/// adjacent leaves, 3 x 3 x 3 neighbourhood) of all four blocks.
/// The far operator is built first (it throws if an order is not achievable).
FarResult measure(const Case& c) {
    FarResult r;
    auto t0 = std::chrono::steady_clock::now();
    r.op = std::make_unique<MlfmmFarOperator>(c.problem, c.tree, c.params);
    r.t_setup = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    const auto dense = op::DenseStrategy().build(c.problem);
    const MatrixXc& Z = dynamic_cast<const op::DenseOperator&>(*dense).matrix();
    r.t_dense = seconds_since(t0);
    const Index n = c.space.size();
    const VectorXc x = random_vector(2 * n, 20261009);
    const VectorXc y_full = Z * x;
    VectorXc y_near = VectorXc::Zero(2 * n);
    const std::vector<Index> leaf = leaf_of(c.tree);
    const auto& boxes = c.tree.boxes();
    for (Index m = 0; m < n; ++m) {
        const auto& a = boxes[static_cast<std::size_t>(leaf[static_cast<std::size_t>(m)])].ijk;
        for (Index k = 0; k < n; ++k) {
            const auto& b = boxes[static_cast<std::size_t>(leaf[static_cast<std::size_t>(k)])].ijk;
            if (std::abs(a[0] - b[0]) > 1 || std::abs(a[1] - b[1]) > 1 || std::abs(a[2] - b[2]) > 1)
                continue;
            for (const Index i : {m, n + m}) {
                y_near(i) += Z(i, k) * x(k) + Z(i, n + k) * x(n + k);
            }
        }
    }
    const VectorXc y_ref = y_full - y_near;
    t0 = std::chrono::steady_clock::now();
    const VectorXc y = *r.op * x;
    r.t_apply = seconds_since(t0);
    r.far_err = (y - y_ref).norm() / y_ref.norm();
    r.full_err = (y - y_ref).norm() / y_full.norm();
    r.describe = r.op->describe();
    return r;
}

void report(const std::string& name, const Case& c, const FarResult& r) {
    WARN(name << ": 2N = " << 2 * c.space.size() << ", levels " << c.tree.levels()
              << ", leaf a = " << c.tree.box_size(c.tree.leaf_level()) / kLambda
              << " lambda0, d0 = " << c.params.accuracy_digits << ": far-only error " << r.far_err
              << ", full-matvec error " << r.full_err << " (dense " << r.t_dense
              << " s, MLFMM setup " << r.t_setup << " s, apply " << r.t_apply << " s)\n"
              << r.describe);
}

/// PMCHWT: S Z with S = diag(I, -I) is complex-symmetric (L and K^PV are), so u^T S Z v =
/// v^T S Z u. The MLFMM keeps this only if the downward pass is the transpose of the upward pass
/// (anterpolation = I^T, parent weights before anterpolating, opposite phase shifts); child
/// weights or a sign error break it at O(1). Release: bitwise identical results for 1 and 3
/// threads (the single-thread apply is too slow under the sanitizers).
void check_symmetry_and_determinism(const MlfmmFarOperator& far) {
    const Index n = far.rows() / 2;
    const VectorXc u = random_vector(2 * n, 11);
    const VectorXc v = random_vector(2 * n, 12);
    VectorXc zu = far * u;
    VectorXc zv = far * v;
    zu.tail(n) *= -1.0;
    zv.tail(n) *= -1.0;
    const Complex uzv = u.transpose() * zv;
    const Complex vzu = v.transpose() * zu;
    const Real asym = std::abs(uzv - vzu) / (u.norm() * zv.norm());
    WARN("asymmetry |u^T S Z v - v^T S Z u| / (|u| |Z v|) = " << asym);
    CHECK(asym <= 1e-6);
#if defined(SPECKLEBEM_HAVE_OPENMP) && defined(NDEBUG)
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const VectorXc y1 = far * u;
    omp_set_num_threads(3);
    const VectorXc y3 = far * u;
    omp_set_num_threads(saved);
    CHECK((y1.array() == y3.array()).all());
#endif
}

/// Bigger cases are release-only (dense oracle and pattern cost under the sanitizers).
constexpr bool kRelease =
#ifdef NDEBUG
    true;
#else
    false;
#endif

}  // namespace

TEST_CASE("mlfmm: far operator vs dense far part, icosphere with 4 levels", "[mlfmm]") {
    // Leaf a = R / 4 = 0.75 lambda0 (vacuum) = 1.125 lambda (n = 1.5): the single-level FMM meets
    // 10^-3 from 0.75 lambda (ADR 0008, WP19b amendment); levels 2 and 3 translate, one
    // interpolation / anterpolation step. Subdivision 3 keeps r_max / a ~ 0.5 (coarser meshes
    // make the enlarged diagonal too large for any order).
    if (!kRelease)
        SKIP("dense oracle of 2N = 3840: release only (minutes under the sanitizers)");
    const Case c(geometry::make_icosphere(3.0 * kLambda, 3), 4, medium(1.5), Kind::PMCHWT, 3.0);
    const FarResult r = measure(c);
    report("icosphere n = 1.5, PMCHWT", c, r);
    CHECK(r.far_err <= 1e-3);
    CHECK(r.full_err <= 1e-3);
}

TEST_CASE("mlfmm: far operator symmetry and thread-count determinism", "[mlfmm]") {
    // 4 levels with small boxes (leaf lambda0 / 4, d0 = 2: low orders, cheap under the
    // sanitizers); the symmetry does not depend on the expansion accuracy.
    const Case c(geometry::make_icosphere(0.5 * kLambda, 3), 4, medium(1.5), Kind::PMCHWT, 2.0);
    const MlfmmFarOperator far(c.problem, c.tree, c.params);
    check_symmetry_and_determinism(far);
}

TEST_CASE("mlfmm: far operator vs dense far part, rough-surface box", "[mlfmm]") {
    // 2 um box, 3 levels: leaf a = 0.5 um = lambda0 (vacuum) = 1.5 lambda (n = 1.5).
    if (!kRelease)
        SKIP("dense oracle of 2N = 4224: release only (minutes under the sanitizers)");
    const Case c(rough_box(2e-6, 125e-9), 3, medium(1.5), Kind::ICTF, 3.0);
    const FarResult r = measure(c);
    report("rough box n = 1.5, ICTF", c, r);
    CHECK(r.far_err <= 1e-3);
    CHECK(r.full_err <= 1e-3);
}

TEST_CASE("mlfmm: with 3 levels the far operator equals the single-level far blocks", "[mlfmm]") {
    // Leaf level 2: no interpolation, so the far operator must reproduce the far_block sums with
    // the formulation weights, up to round-off amplified by
    // the cancellation in the direction sum (|w T_L| >> the result; measured 2e-10).
    const Case c(geometry::make_icosphere(kLambda, 2), 3, medium(1.5), Kind::ICTF, 3.0);
    const MlfmmFarOperator far(c.problem, c.tree, c.params);
    const Index n = c.space.size();
    const std::vector<Index>& perm = c.tree.permutation();
    // Observers: the first 2 leaves; sources: the first 4 boxes of the first observer's
    // interaction list (x vanishes elsewhere), so only a few far_block calls are needed.
    const std::vector<Index> observers(c.tree.boxes_at_level(2).begin(),
                                       c.tree.boxes_at_level(2).begin() + 2);
    const mlfmm::Box& first = c.tree.boxes()[static_cast<std::size_t>(observers[0])];
    REQUIRE(first.interaction_list.size() >= 4);
    std::vector<bool> source(c.tree.boxes().size(), false);
    VectorXc x = VectorXc::Zero(2 * n);
    const VectorXc xr = random_vector(2 * n, 7);
    for (std::size_t j = 0; j < 4; ++j) {
        const Index b = first.interaction_list[j];
        source[static_cast<std::size_t>(b)] = true;
        const mlfmm::Box& B = c.tree.boxes()[static_cast<std::size_t>(b)];
        for (Index p = B.first_element; p < B.first_element + B.num_elements; ++p) {
            const Index col = perm[static_cast<std::size_t>(p)];
            x(col) = xr(col);
            x(n + col) = xr(n + col);
        }
    }
    const VectorXc y = far * x;
    const std::array<material::Material, 2> mat = {c.problem.exterior, c.problem.object};
    const Real w = c.problem.omega;
    const formulation::Weights wt =
        c.form->weights(mat[0].wave_impedance(w), mat[1].wave_impedance(w));
    VectorXc ref = VectorXc::Zero(2 * n);
    std::vector<Index> rows;
    for (int i = 0; i < 2; ++i) {
        REQUIRE(far.region_active(i));
        const material::Material& m = mat[static_cast<std::size_t>(i)];
        const kernels::RegionParams region{m.wavenumber(w), m.wave_impedance(w), w,
                                           constants::eps0 * m.eps_r, constants::mu0 * m.mu_r};
        const Complex e = (i == 0 ? wt.a1 : wt.a2) / region.eta;
        const Complex h = (i == 0 ? wt.b1 : wt.b2) * region.eta;
        const Complex mm = (i == 0 ? wt.b1 : wt.b2) / region.eta;
        const int order = far.levels(i).back().truncation_order;
        for (const Index a : observers) {
            const mlfmm::Box& A = c.tree.boxes()[static_cast<std::size_t>(a)];
            for (Index r = 0; i == 0 && r < A.num_elements; ++r) {
                const Index row = perm[static_cast<std::size_t>(A.first_element + r)];
                rows.push_back(row);
                rows.push_back(n + row);
            }
            for (const Index b : A.interaction_list) {
                if (!source[static_cast<std::size_t>(b)])
                    continue;
                const mlfmm::Box& B = c.tree.boxes()[static_cast<std::size_t>(b)];
                const mlfmm::FarBlock f = mlfmm::far_block(far.patterns(i), a, b, order, region);
                for (Index r = 0; r < A.num_elements; ++r) {
                    const Index row = perm[static_cast<std::size_t>(A.first_element + r)];
                    for (Index s = 0; s < B.num_elements; ++s) {
                        const Index col = perm[static_cast<std::size_t>(B.first_element + s)];
                        ref(row) += e * (f.L(r, s) * x(col) - f.K(r, s) * x(n + col));
                        ref(n + row) += h * f.K(r, s) * x(col) + mm * f.L(r, s) * x(n + col);
                    }
                }
            }
        }
    }
    Real num = 0.0, den = 0.0;
    for (const Index r : rows) {
        num += std::norm(y(r) - ref(r));
        den += std::norm(ref(r));
    }
    const Real err = std::sqrt(num / den);
    INFO(rows.size() << " rows, relative difference " << err);
    REQUIRE(!rows.empty());
    CHECK(err <= 1e-8);
}

TEST_CASE("mlfmm: far operator error cases", "[mlfmm]") {
    // 3 levels, leaf a = R / 2 = 0.75 lambda0 (r_max / a ~ 0.5).
    const Real radius = 1.5 * kLambda;
    // Ag interior: d0 = 3 is not achievable from 0.5 lambda0 boxes on (ADR 0008 amendment), and
    // the lossy-region policy (WP21) is missing -> runtime_error naming region R2 (all order
    // searches run before any translator or pattern is computed).
    {
        const Case ag(geometry::make_icosphere(radius, 2), 3, material::silver_500nm(),
                      Kind::PMCHWT, 3.0);
        CHECK_THROWS_MATCHES(
            MlfmmFarOperator(ag.problem, ag.tree, ag.params), std::runtime_error,
            Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("region R2")));
    }
    const Case c(geometry::make_icosphere(radius, 2), 2, medium(1.5), Kind::PMCHWT, 3.0);
    for (int bad = 0; bad < 5; ++bad) {
        mlfmm::MlfmmParams p;
        if (bad == 0)
            p.truncation_L = 10;
        if (bad == 1)
            p.precompute_translators = false;
        if (bad == 2)
            p.use_fft_interpolation = true;
        if (bad == 3)
            p.accuracy_digits = 0.0;
        if (bad == 4)
            p.accuracy_digits = 6.0;
        CHECK_THROWS_AS(MlfmmFarOperator(c.problem, c.tree, p), std::invalid_argument);
    }
    const geometry::TriangleMesh other = geometry::make_icosphere(kLambda, 1);
    const basis::RwgSpace other_space(other);
    const Octree other_tree(other_space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    CHECK_THROWS_AS(MlfmmFarOperator(c.problem, other_tree, c.params), std::invalid_argument);
    op::Problem no_space = c.problem;
    no_space.space = nullptr;
    CHECK_THROWS_AS(MlfmmFarOperator(no_space, c.tree, c.params), std::invalid_argument);
    // Fewer than 3 levels: no interaction lists, Z_far = 0.
    const MlfmmFarOperator none(c.problem, c.tree, c.params);
    CHECK(!none.region_active(0));
    CHECK(none.levels(1).empty());
    CHECK_THROWS_AS(none.levels(2), std::out_of_range);
    CHECK_THROWS_AS(none.patterns(0), std::invalid_argument);
    CHECK((none * VectorXc::Ones(2 * c.space.size())).norm() == 0.0);
    VectorXc y;
    CHECK_THROWS_AS(none.apply(VectorXc::Zero(3), y), std::invalid_argument);
}

TEST_CASE("mlfmm: far operator accuracy sweep", "[.][mlfmm_far_sweep]") {
#ifndef NDEBUG
    SKIP("release-only sweep");
#endif
    // Far-only and full-matvec errors vs leaf size, d0 and interior. Icospheres with 4 levels
    // (leaf a = R / 4, subdivision 3: r_max / a = 0.13 lambda0 / a). Asserted where every active
    // region's leaf meets the single-level threshold of ADR 0008 (0.75 lambda for d0 = 3,
    // 1.5 lambda for d0 = 5, lambda in the medium); otherwise reported.
    struct Interior {
        std::string name;
        material::Material m;
        Real n;
    };
    const std::vector<Interior> interiors = {{"n = 1.5", medium(1.5), 1.5},
                                             {"Si", material::silicon_500nm(), 4.3}};
    for (const Real a : {0.5, 0.75, 1.0, 1.5}) {
        for (const Interior& in : interiors) {
            if (in.n > 2.0 && a > 0.5)
                continue;  // pattern memory of Si at L ~ 70+ on 1920 bases
            for (const Real d0 : {3.0, 5.0}) {
                const Case c(geometry::make_icosphere(4.0 * a * kLambda, 3), 4, in.m, Kind::PMCHWT,
                             d0);
                const std::string name =
                    "icosphere " + in.name + ", a = " + std::to_string(a) + " lambda0";
                const bool asserted = a >= (d0 <= 3.0 ? 0.74 : 1.49);
                try {
                    const FarResult r = measure(c);
                    report(name, c, r);
                    if (asserted)
                        CHECK(r.far_err <= std::pow(10.0, -d0));
                } catch (const std::runtime_error& e) {
                    WARN(name << ", d0 = " << d0 << ": " << e.what());
                    CHECK(!asserted);
                }
            }
        }
    }
    // Full-matvec error with lambda0 / 4 leaves (lambda0 / 2: the a = 0.5 rows above), input for
    // the ADR 0008 leaf policy: R = 0.5 um, subdivision 3, 4 levels (r_max / a ~ 0.5). The order
    // search may report the target as not achievable there (reported, not asserted).
    for (const Real d0 : {3.0, 5.0}) {
        const Case c(geometry::make_icosphere(kLambda, 3), 4, medium(1.5), Kind::PMCHWT, d0);
        try {
            report("icosphere R = 0.5 um, n = 1.5", c, measure(c));
        } catch (const std::runtime_error& e) {
            WARN("icosphere R = 0.5 um, lambda0 / 4 leaves, d0 = " << d0 << ": " << e.what());
        }
    }
}
