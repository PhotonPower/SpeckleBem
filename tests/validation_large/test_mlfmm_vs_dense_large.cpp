// MLFMM matvec against the dense matrix (WP20b; docs/05 row "MLFMM matvec, 2e4-1e5 unknowns":
// relative error < 1e-3 for d0 = 3, < 1e-5 for d0 = 5; ADR 0008 amendments). ctest label
// `validation-large`, manual / nightly:
//   ctest --preset release -L validation-large -R mlfmm
//
// Metric: max over 3 fixed-seed random x of |Z_dense x - Z_mlfmm x| / |Z_dense x|. One dense
// matrix per case (geometry, interior, formulation), shared by the MLFMM configurations:
//  * d0 = 3 with lambda / 4 leaves (the default floor; ADR 0008 WP20a amendment),
//  * d0 = 5 with lambda / 2 leaves (r_max / a ~ 0.2 on these meshes; the WP20a amendment asks
//    for r_max / a <= 0.3).
// lambda = 500 nm (vacuum exterior), dense quadrature target 1e-6 (0.1 x 10^-5; the near entries
// are identical in both operators, so only the far entries of the oracle matter).
// The 2N ~ 9e4 case has no dense oracle: 64 exact rows (op::assemble_sparse) are compared.
// SKIPs in unoptimised builds and when the dense matrix exceeds 60 % of the physical memory.
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "system_memory.hpp"

using namespace specklebem;
using formulation::Kind;

namespace {

[[maybe_unused]] constexpr Real kLambda = 500e-9;

[[maybe_unused]] double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

[[maybe_unused]] VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

/// Rough-surface box (Gaussian, sigma 30 nm, Lc 250 nm, fixed seed), uniform mesh.
[[maybe_unused]] geometry::TriangleMesh rough_box(Real size, Real depth, Real mesh_size) {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = size;
    p.rms_roughness = 30e-9;
    p.correlation_length = 250e-9;
    p.mesh_size = mesh_size;
    p.box_depth = depth;
    p.box_mesh_size = mesh_size;
    p.seed = 7;
    return geometry::make_rough_surface_mesh(p);
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
        problem.kernel_options.target_accuracy = 1e-6;
    }
    Setup(const Setup&) = delete;
    Setup& operator=(const Setup&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
};

struct Config {
    Real digits;
    Real leaf_lambda;  ///< min_box_size_lambda (leaf floor)
    bool asserted;     ///< false: report only (d0 = 5, ADR 0008 status still open)
};

[[maybe_unused]] mlfmm::MlfmmParams mlfmm_params(const Config& c) {
    mlfmm::MlfmmParams p;
    p.accuracy_digits = c.digits;
    p.octree.max_elements_per_leaf = 4;  // the floor decides the leaf size
    p.octree.min_box_size_lambda = c.leaf_lambda;
    return p;
}

[[maybe_unused]] Real mb(std::size_t b) {
    return static_cast<Real>(b) / 1048576.0;
}

/// One dense matrix, then each MLFMM configuration: error over 3 random x, timings, memory.
[[maybe_unused]] void run_case(const std::string& name, const Setup& s,
                               const std::vector<Config>& configs) {
    const Index unknowns = 2 * s.space.size();
    const Real dense_bytes = 16.0 * static_cast<Real>(unknowns) * static_cast<Real>(unknowns);
    const Real phys = system_memory::physical_memory_bytes();
    if (!(phys > 0.0) || dense_bytes + 4e9 > 0.6 * phys) {
        SKIP(name << ": 2N = " << unknowns << " needs ~" << dense_bytes / 1e9
                  << " GB for the dense matrix, more than 60 % of the " << phys / 1e9 << " GB");
    }
    auto t0 = std::chrono::steady_clock::now();
    const std::shared_ptr<op::LinearOperator> built = op::DenseStrategy().build(s.problem);
    const auto* Zd = dynamic_cast<const op::DenseOperator*>(built.get());
    REQUIRE(Zd != nullptr);
    const double dense_s = seconds_since(t0);
    std::vector<VectorXc> x, yd;
    t0 = std::chrono::steady_clock::now();
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        x.push_back(random_vector(unknowns, 20261010 + seed));
        yd.push_back(*Zd * x.back());
    }
    const double dense_apply_s = seconds_since(t0) / 3.0;
    const Real rmax = mlfmm::max_support_radius(s.space);
    WARN(name << ": " << s.mesh.num_triangles() << " triangles, 2N = " << unknowns
              << ", r_max = " << rmax << " m; dense assembly " << dense_s << " s, matvec "
              << dense_apply_s << " s, " << dense_bytes / 1048576.0 << " MB");
    for (const Config& c : configs) {
        std::ostringstream id;
        id << name << ", d0 = " << c.digits << ", leaf floor " << c.leaf_lambda << " lambda";
        try {
            t0 = std::chrono::steady_clock::now();
            const mlfmm::MlfmmOperator Zm(s.problem, mlfmm_params(c));
            const double setup_s = seconds_since(t0);
            const mlfmm::Octree& tree = Zm.octree();
            const Real a = tree.box_size(tree.leaf_level());
            Real err = 0.0;
            VectorXc y;
            t0 = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < x.size(); ++i) {
                Zm.apply(x[i], y);
                err = std::max(err, (y - yd[i]).norm() / yd[i].norm());
            }
            const double apply_s = seconds_since(t0) / 3.0;
            WARN(id.str() << ": " << tree.levels() << " levels, leaf a = " << a / kLambda
                          << " lambda0, r_max / a = " << rmax / a << ": matvec error " << err
                          << " (target " << std::pow(10.0, -c.digits) << "); setup " << setup_s
                          << " s, apply " << apply_s << " s; memory "
                          << mb(Zm.memory_bytes()) << " MB (near "
                          << mb(Zm.near_operator().memory_bytes()) << ", far "
                          << mb(Zm.far_operator().memory_bytes()) << ")\n"
                          << Zm.describe());
            if (c.asserted)
                CHECK(err < std::pow(10.0, -c.digits));
        } catch (const std::runtime_error& e) {
            WARN(id.str() << ": not built: " << e.what());
            CHECK(!c.asserted);
        }
    }
}

}  // namespace

TEST_CASE("mlfmm vs dense large: icosphere n = 1.5, PMCHWT", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // R = 1 um, subdivision 4: 2N = 15360, h ~ lambda / 7.6; root 4 lambda (5 levels at the
    // lambda / 4 floor, 4 at lambda / 2).
    const Setup s(geometry::make_icosphere(2.0 * kLambda, 4), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
                  Kind::PMCHWT);
    run_case("icosphere R = 1 um, n = 1.5, PMCHWT", s, {{3.0, 0.25, true}, {5.0, 0.5, false}});
#endif
}

TEST_CASE("mlfmm vs dense large: icosphere Si, ICTF", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Setup s(geometry::make_icosphere(2.0 * kLambda, 4), material::silicon_500nm(),
                  Kind::ICTF);
    run_case("icosphere R = 1 um, Si, ICTF", s, {{3.0, 0.25, true}, {5.0, 0.5, false}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough box n = 1.5, ICTF", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 2 um x 2 um x 0.3 um, mesh 50 nm (lambda / 10): 2N ~ 2.5e4; root 4 lambda.
    const Setup s(rough_box(2e-6, 0.3e-6, 50e-9), {Complex(2.25, 0.0), Complex(1.0, 0.0)}, Kind::ICTF);
    run_case("rough box 2 um, n = 1.5, ICTF", s, {{3.0, 0.25, true}, {5.0, 0.5, false}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough box Si, PMCHWT", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Setup s(rough_box(2e-6, 0.3e-6, 50e-9), material::silicon_500nm(), Kind::PMCHWT);
    run_case("rough box 2 um, Si, PMCHWT", s, {{3.0, 0.25, true}, {5.0, 0.5, false}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough plate with lambda leaves, d0 = 5",
          "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 4 um x 4 um x 0.3 um, mesh 100 nm (lambda / 5): 2N ~ 2.2e4; root 8 lambda, 4 levels at the
    // lambda floor (r_max / a ~ 0.2).
    const Setup s(rough_box(4e-6, 0.3e-6, 100e-9), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
                  Kind::PMCHWT);
    run_case("rough plate 4 um, n = 1.5, PMCHWT", s, {{5.0, 1.0, false}, {3.0, 1.0, true}});
#endif
}

TEST_CASE("mlfmm large: rough box 2N ~ 9e4 against exact rows", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 4 um x 4 um x 0.3 um, mesh 50 nm: 2N ~ 8.8e4 (dense 125 GB): 64 random basis rows m (J
    // and M rows) of Z x exactly (op::assemble_sparse, all columns) against the MLFMM, d0 = 3,
    // lambda / 4 leaves; MLFMM setup / apply time and memory for the scaling table.
    const Setup s(rough_box(4e-6, 0.3e-6, 50e-9), material::silicon_500nm(), Kind::ICTF);
    const Index n = s.space.size();
    std::vector<Index> rows;
    std::mt19937_64 rng(20261010);
    while (rows.size() < 64) {
        const auto r = static_cast<Index>(rng() % static_cast<std::uint64_t>(n));
        if (std::find(rows.begin(), rows.end(), r) == rows.end())
            rows.push_back(r);
    }
    op::BasisPattern pat;  // group 0: the rows with all columns; group 1: no columns
    pat.group_of_row.assign(static_cast<std::size_t>(n), 1);
    for (const Index r : rows) pat.group_of_row[static_cast<std::size_t>(r)] = 0;
    pat.cols.resize(static_cast<std::size_t>(n));
    for (Index c = 0; c < n; ++c) pat.cols[static_cast<std::size_t>(c)] = c;
    pat.col_ptr = {0, n, n};
    auto t0 = std::chrono::steady_clock::now();
    const auto exact = op::assemble_sparse(s.problem, pat);
    const double rows_s = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    const mlfmm::MlfmmOperator Zm(s.problem, mlfmm_params({3.0, 0.25, true}));
    const double setup_s = seconds_since(t0);
    Real num = 0.0, den = 0.0;
    double apply_s = 0.0;
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const VectorXc x = random_vector(2 * n, 20261010 + seed);
        const VectorXc ye = *exact * x;
        t0 = std::chrono::steady_clock::now();
        const VectorXc y = Zm * x;
        apply_s += seconds_since(t0) / 3.0;
        for (const Index r : rows) {
            for (const Index i : {r, n + r}) {
                num += std::norm(y(i) - ye(i));
                den += std::norm(ye(i));
            }
        }
    }
    const Real err = std::sqrt(num / den);
    const mlfmm::Octree& tree = Zm.octree();
    WARN("rough box 4 um, Si, ICTF: " << s.mesh.num_triangles() << " triangles, 2N = " << 2 * n
                                      << ", " << tree.levels() << " levels, leaf a = "
                                      << tree.box_size(tree.leaf_level()) / kLambda
                                      << " lambda0: error on 128 exact rows " << err
                                      << " (rows assembled in " << rows_s << " s); setup "
                                      << setup_s << " s, apply " << apply_s << " s, memory "
                                      << mb(Zm.memory_bytes()) << " MB (dense would need "
                                      << 16.0 * 4.0 * static_cast<Real>(n * n) / 1048576.0
                                      << " MB); peak RSS "
                                      << system_memory::peak_rss_bytes() / 1e9 << " GB\n"
                                      << Zm.describe());
    CHECK(2 * n > 80000);
    CHECK(err < 1e-3);
#endif
}
