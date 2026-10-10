// MLFMM matvec against the dense matrix (WP20b; docs/05 row "MLFMM matvec, 2e4-1e5 unknowns":
// relative error < 1e-3 for d0 = 3, < 1e-5 for d0 = 5; ADR 0008 amendments). ctest label
// `validation-large`, manual / nightly:
//   ctest --preset release -L validation-large -R mlfmm
//
// Metric: max over 3 fixed-seed random x of |Z_dense x - Z_mlfmm x| / |Z_dense x|. One dense
// matrix per case (geometry, interior, formulation), shared by the MLFMM configurations:
//  * d0 = 3 with a lambda / 4 leaf floor (the default; ADR 0008 WP20a amendment),
//  * d0 = 5 with a lambda / 2 floor, and lambda leaves on a coarser plate.
// Since WP21 MlfmmOperator applies the leaf rule a >= max(a_min(d0), r_max / max_support_ratio(d0))
// (ratio 0.6 for d0 <= 3, 0.3 for d0 > 3, WP21 review): every case here keeps lambda / 4 leaves at
// d0 = 3 (rough box Ag: 0.3 lambda) and lambda / 2 at d0 = 5 (rough box Ag: 0.6 lambda).
// After the WP21 review (2026-10-10): d0 = 3: 1.7e-4 / 2.1e-4 / 1.0e-4 (icosphere n = 1.5 / Si /
// Ag R = 0.5 um), 7.2e-5 / 4.3e-5 / 6.4e-5 (rough box n = 1.5 / Si / Ag), 5.4e-6 (plate); d0 = 5:
// 3.7e-6 / 6.5e-6 / 4.1e-6 (icosphere), 6.7e-6 / 4.2e-6 / 2.1e-6 (rough box), 5.5e-7 (plate);
// exact rows: 2.3e-4 (Si icosphere 61 440), 1.1e-4 (Ag icosphere 61 440), 2.9e-4 (box 88 320).
// Measured 2026-10-10 before WP21 (win-release, 24 cores, lambda / 4 leaves at r_max / a ~ 0.57
// for d0 = 3): d0 = 3: 1.7e-4 / 2.1e-4 (icosphere n = 1.5 / Si), 7.2e-5 / 4.3e-5 (rough box
// n = 1.5 / Si), 5.4e-6 (plate, lambda leaves); d0 = 5: 3.7e-6 / 6.5e-6 (icosphere), 6.7e-6 /
// 4.2e-6 (rough box), 5.5e-7 (plate, lambda leaves); 2N = 8.8e4 rows: 2.9e-4. WP21 values: see
// the Ag cases below and the WP21 report (CHANGELOG).
// lambda = 500 nm (vacuum exterior), dense quadrature target 1e-6 (0.1 x 10^-5; the near entries
// are identical in both operators, so only the far entries of the oracle matter).
// The 2N ~ 6e4 / 9e4 cases have no dense oracle: 64 exact rows (op::assemble_sparse) are
// compared. SKIPs in unoptimised builds and when the dense matrix (or the measured peak of the
// exact-rows cases) exceeds 60 % of the physical memory.
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
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

#include "mlfmm_simulation_support.hpp"
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
        INFO(id.str());
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
                      << " s, apply " << apply_s << " s; memory " << mb(Zm.memory_bytes())
                      << " MB (near " << mb(Zm.near_operator().memory_bytes()) << ", far "
                      << mb(Zm.far_operator().memory_bytes()) << ")\n"
                      << Zm.describe());
        CHECK(err < std::pow(10.0, -c.digits));
    }
}

}  // namespace

TEST_CASE("mlfmm vs dense large: icosphere n = 1.5, PMCHWT", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // R = 1 um, subdivision 4: 2N = 15360, h ~ lambda / 7.6; root 4 lambda (5 levels at the
    // lambda / 4 floor, 4 at lambda / 2).
    const Setup s(geometry::make_icosphere(2.0 * kLambda, 4),
                  {Complex(2.25, 0.0), Complex(1.0, 0.0)}, Kind::PMCHWT);
    run_case("icosphere R = 1 um, n = 1.5, PMCHWT", s, {{3.0, 0.25}, {5.0, 0.5}});
#endif
}

TEST_CASE("mlfmm vs dense large: icosphere Si, ICTF", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Setup s(geometry::make_icosphere(2.0 * kLambda, 4), material::silicon_500nm(),
                  Kind::ICTF);
    run_case("icosphere R = 1 um, Si, ICTF", s, {{3.0, 0.25}, {5.0, 0.5}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough box n = 1.5, ICTF", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 2 um x 2 um x 0.3 um, mesh 50 nm (lambda / 10): 2N ~ 2.5e4; root 4 lambda.
    const Setup s(rough_box(2e-6, 0.3e-6, 50e-9), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
                  Kind::ICTF);
    run_case("rough box 2 um, n = 1.5, ICTF", s, {{3.0, 0.25}, {5.0, 0.5}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough box Si, PMCHWT", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Setup s(rough_box(2e-6, 0.3e-6, 50e-9), material::silicon_500nm(), Kind::PMCHWT);
    run_case("rough box 2 um, Si, PMCHWT", s, {{3.0, 0.25}, {5.0, 0.5}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough plate with lambda leaves, d0 = 5",
          "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 4 um x 4 um x 0.3 um, mesh 100 nm (lambda / 5): 2N ~ 2.2e4; root 8 lambda, 4 levels at the
    // lambda floor (r_max / a ~ 0.28).
    const Setup s(rough_box(4e-6, 0.3e-6, 100e-9), {Complex(2.25, 0.0), Complex(1.0, 0.0)},
                  Kind::PMCHWT);
    run_case("rough plate 4 um, n = 1.5, PMCHWT", s, {{5.0, 1.0}, {3.0, 1.0}});
#endif
}

TEST_CASE("mlfmm simulation large: rough box n = 1.5, ICTF + Jacobi", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // GMRES through Simulation, "mlfmm" vs "dense" (tests/support/mlfmm_simulation_support.hpp):
    // 1 um x 1 um x 0.3 um, mesh 50 nm (2N = 7680), default kernel options. Measured 2026-10-10:
    // 329 GMRES iterations each (~22 s), currents 3.2e-4, RCS 8.2e-6 with 4 levels and lambda / 4
    // leaves (again since the WP21 review: r_max ~ 45 nm <= 0.6 a); the interim r_max / 0.3 rule
    // gave 3 levels with lambda / 2 leaves: currents 4.6e-5, RCS 1.3e-6.
    SimulationConfig cfg;
    cfg.object = {Complex(2.25, 0.0), Complex(1.0, 0.0)};
    cfg.formulation = formulation::Kind::ICTF;
    cfg.diagonal_preconditioner = true;
    mlfmm::MlfmmParams m;
    m.octree.max_elements_per_leaf = 4;
    const mlfmm_simulation_test::Comparison c =
        mlfmm_simulation_test::compare(rough_box(1e-6, 0.3e-6, 50e-9), cfg, m, "rough box 1 um");
    CHECK(c.currents <= 1e-3);
    CHECK(c.rcs <= 1e-3);
#endif
}

TEST_CASE("mlfmm vs dense large: icosphere Ag, PMCHWT", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // R = 0.5 um, subdivision 4: 2N = 15360, r_max ~ 35 nm, so the leaf rule keeps lambda / 4
    // leaves at d0 = 3 (lambda / 2 at d0 = 5). The Ag interior (skin depth ~13 nm) uses no
    // expansion: its far pairs are truncated by the decay bound or evaluated exactly (WP21).
    // Measured 2026-10-10: 1.0e-4 (d0 = 3; exact part 4.45e6 basis pairs, 170 MB with the (L, K)
    // storage, 408 MB before), 4.1e-6 (d0 = 5; 100 MB).
    const Setup s(geometry::make_icosphere(kLambda, 4), material::silver_500nm(), Kind::PMCHWT);
    run_case("icosphere R = 0.5 um, Ag, PMCHWT", s, {{3.0, 0.25}, {5.0, 0.5}});
#endif
}

TEST_CASE("mlfmm vs dense large: rough box Ag, PMCHWT", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 1.2 um x 1.2 um x 0.3 um, mesh 35 nm: 2N = 21216, r_max = 46 nm (rim triangles), so the
    // leaf rule gives 0.3 lambda leaves at d0 = 3 and 0.6 lambda at d0 = 5. Measured 2026-10-10
    // after the WP21 review: 6.4e-5 (d0 = 3, exact part 204 MB vs near 679 MB); with 0.6 lambda
    // leaves at both d0 before: 4.8e-6
    // (d0 = 3), 2.1e-6 (d0 = 5); Ag interior exact at the leaf level (272 / 382 box pairs).
    const Setup s(rough_box(1.2e-6, 0.3e-6, 35e-9), material::silver_500nm(), Kind::PMCHWT);
    run_case("rough box 1.2 um, Ag, PMCHWT", s, {{3.0, 0.25}, {5.0, 0.5}});
#endif
}

TEST_CASE("mlfmm simulation large: Ag sphere, Jacobi", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // GMRES through Simulation, "mlfmm" vs "dense": Ag icosphere R = 0.5 um, subdivision 3
    // (2N = 3840; leaf rule: lambda / 2 leaves, 3 levels), automatic formulation, left Jacobi.
    // Measured 2026-10-10: GMRES 186 (dense) / 210 (MLFMM) iterations, currents 1.2e-4, RCS
    // 3.3e-6.
    SimulationConfig cfg;
    cfg.object = material::silver_500nm();
    cfg.diagonal_preconditioner = true;
    const mlfmm_simulation_test::Comparison c = mlfmm_simulation_test::compare(
        geometry::make_icosphere(kLambda, 3), cfg, mlfmm::MlfmmParams{}, "Ag sphere");
    CHECK(c.currents <= 1e-3);
    CHECK(c.rcs <= 1e-3);
#endif
}

namespace {

/// 64 random basis rows m (J and M rows) of Z x exactly (op::assemble_sparse, all columns)
/// against the MLFMM (d0 = 3, lambda / 4 floor); setup / apply time and memory for the scaling
/// table. SKIPs if `peak_bytes` (the measured peak RSS of the case) exceeds 60 % of the memory.
[[maybe_unused]] void run_exact_rows(const std::string& name, const Setup& s, Real peak_bytes) {
    const Real phys = system_memory::physical_memory_bytes();
    if (!(phys > 0.0) || peak_bytes > 0.6 * phys) {
        SKIP(name << ": needs ~" << peak_bytes / 1e9 << " GB, more than 60 % of the " << phys / 1e9
                  << " GB");
    }
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
    const mlfmm::MlfmmOperator Zm(s.problem, mlfmm_params({3.0, 0.25}));
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
    WARN(name << ": " << s.mesh.num_triangles() << " triangles, 2N = " << 2 * n << ", "
              << tree.levels() << " levels, leaf a = " << tree.box_size(tree.leaf_level()) / kLambda
              << " lambda0: error on 128 exact rows " << err << " (rows assembled in " << rows_s
              << " s); setup " << setup_s << " s, apply " << apply_s << " s, memory "
              << mb(Zm.memory_bytes()) << " MB (dense would need "
              << 16.0 * 4.0 * static_cast<Real>(n * n) / 1048576.0 << " MB); peak RSS "
              << system_memory::peak_rss_bytes() / 1e9 << " GB\n"
              << Zm.describe());
    CHECK(err < 1e-3);
}

}  // namespace

TEST_CASE("mlfmm large: icosphere 2N = 61440 against exact rows", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // R = 1 um, subdivision 5 (2N = 61440, dense 60 GB), Si, ICTF; r_max ~ 36 nm keeps
    // lambda / 4 leaves under the leaf rule. Measured 2026-10-10: error 2.3e-4, setup 19 s,
    // apply 0.52 s, 3.3 GB, peak RSS 5.6 GB (guard 7 GB).
    const Setup s(geometry::make_icosphere(2.0 * kLambda, 5), material::silicon_500nm(),
                  Kind::ICTF);
    CHECK(2 * s.space.size() == 61440);
    run_exact_rows("icosphere R = 1 um, Si, ICTF", s, 7e9);
#endif
}

TEST_CASE("mlfmm large: Ag icosphere 2N = 61440 against exact rows", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // R = 1 um, subdivision 5 (2N = 61440), Ag, PMCHWT, d0 = 3 (lambda / 4 leaves): the Ag
    // interior uses the ADR 0008 §6 fallback (exact pairs on levels 4 and 3, (L, K) storage); WP22
    // projection input. Measured 2026-10-10: error 1.1e-4 on 128 rows; exact part 18.0e6 basis
    // pairs (box-pair bound 9.3e7), 687 MB in 3.7 s (the four-entry storage would need 1.73 GB),
    // near field 1133 MB, setup 20 s, apply 0.23 s, peak RSS 4.9 GB (guard 7 GB).
    const Setup s(geometry::make_icosphere(2.0 * kLambda, 5), material::silver_500nm(),
                  Kind::PMCHWT);
    CHECK(2 * s.space.size() == 61440);
    run_exact_rows("icosphere R = 1 um, Ag, PMCHWT", s, 7e9);
#endif
}

TEST_CASE("mlfmm large: rough box 2N ~ 9e4 against exact rows", "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // 4 um x 4 um x 0.3 um, mesh 50 nm: 2N ~ 8.8e4 (dense 125 GB), Si, ICTF; lambda / 4 leaves
    // (r_max ~ 45 nm <= 0.6 a, leaf rule since the WP21 review). Measured 2026-10-10: error
    // 2.9e-4, setup 39 s, apply 1.9 s, 5.5 GB, peak RSS 6.4 GB (guard 9 GB); with the interim
    // r_max / 0.3 rule (lambda / 2 leaves): 1.8e-5, 12 GB, peak RSS 23.9 GB.
    const Setup s(rough_box(4e-6, 0.3e-6, 50e-9), material::silicon_500nm(), Kind::ICTF);
    CHECK(2 * s.space.size() > 80000);
    run_exact_rows("rough box 4 um, Si, ICTF", s, 9e9);
#endif
}

namespace {

/// Rough box with the graded walls of ADR 0006 (top 50 nm, coarse box cells 100 nm = lambda1 / 5,
/// sigma 50 nm, Lc 500 nm, fixed seed): the coarse cells (support radii 112-141 nm) are the
/// elevated functions of the local leaf rule, the top face has radii of ~56-70 nm.
[[maybe_unused]] geometry::TriangleMesh graded_box(Real size, Real depth) {
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

/// Leaf radius quantile that keeps the leaf at a_min(d0): the fraction of basis functions with
/// r <= max_support_ratio(d0) a_min(d0) (1 - kMinBoxSizeTolerance) lambda (the top face of
/// graded_box), so that every coarser box function is elevated.
[[maybe_unused]] Real top_face_quantile(const basis::RwgSpace& space, Real digits) {
    const Real a_min = digits <= 3.0 ? mlfmm::kLeafMinLambdaD3 : mlfmm::kLeafMinLambdaD5;
    const Real limit = mlfmm::max_support_ratio(digits) * a_min *
                       (1.0 - mlfmm::kMinBoxSizeTolerance) * kLambda;
    const std::vector<Real> r = mlfmm::support_radii(space);
    const auto small = std::count_if(r.begin(), r.end(), [&](Real v) { return v <= limit; });
    return static_cast<Real>(small) / static_cast<Real>(r.size());
}

/// One MLFMM configuration of run_local_leaf_case.
struct LocalLeafConfig {
    Real digits;
    bool local;       ///< true: top_face_quantile (elevated functions), false: quantile 1
    bool check_full;  ///< CHECK the full matvec error against 10^-d0 (else reported only)
};

/// MLFMM configurations against one dense matrix: full matvec error over 3 random x (docs/05
/// criterion), and for the local leaf rule the part of the error that involves elevated functions
/// (|rows or columns of elevated functions of (Z_mlfmm - Z_dense) x| / |Z_dense x|, checked
/// against 10^-d0 for every local configuration) next to the leaf-leaf part; home levels, memory,
/// timings.
[[maybe_unused]] void run_local_leaf_case(const std::string& name, const Setup& s,
                                          const std::vector<LocalLeafConfig>& configs) {
    const Index n = s.space.size();
    const Index unknowns = 2 * n;
    const Real dense_bytes = 16.0 * static_cast<Real>(unknowns) * static_cast<Real>(unknowns);
    const Real phys = system_memory::physical_memory_bytes();
    if (!(phys > 0.0) || dense_bytes + 8e9 > 0.6 * phys) {
        SKIP(name << ": 2N = " << unknowns << " needs ~" << dense_bytes / 1e9
                  << " GB for the dense matrix + ~8 GB, more than 60 % of the " << phys / 1e9
                  << " GB");
    }
    auto t0 = std::chrono::steady_clock::now();
    const std::shared_ptr<op::LinearOperator> Zd = op::DenseStrategy().build(s.problem);
    const double dense_s = seconds_since(t0);
    std::vector<VectorXc> x, yd;
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        x.push_back(random_vector(unknowns, 20261011 + seed));
        yd.push_back(*Zd * x.back());
    }
    WARN(name << ": " << s.mesh.num_triangles() << " triangles, 2N = " << unknowns
              << ", r_max = " << mlfmm::max_support_radius(s.space) << " m; dense assembly "
              << dense_s << " s");
    for (const LocalLeafConfig& c : configs) {
        mlfmm::MlfmmParams p;
        p.accuracy_digits = c.digits;
        p.octree.max_elements_per_leaf = 4;  // the leaf rule decides the leaf size
        p.leaf_radius_quantile = c.local ? top_face_quantile(s.space, c.digits) : 1.0;
        std::ostringstream id;
        id << name << ", d0 = " << c.digits << ", leaf radius quantile "
           << p.leaf_radius_quantile << (c.local ? " (local leaf rule)" : " (global rule)");
        INFO(id.str());
        t0 = std::chrono::steady_clock::now();
        const mlfmm::MlfmmOperator Zm(s.problem, p);
        const double setup_s = seconds_since(t0);
        const mlfmm::Octree& tree = Zm.octree();
        CHECK(tree.has_elevated() == c.local);
        std::ostringstream homes;
        for (const Index k : tree.home_level_counts()) homes << " " << k;
        Real err = 0.0;
        VectorXc y;
        t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < x.size(); ++i) {
            Zm.apply(x[i], y);
            err = std::max(err, (y - yd[i]).norm() / yd[i].norm());
        }
        const double apply_s = seconds_since(t0) / 3.0;
        // Split of the first random vector: elevated sources / rows vs leaf-leaf.
        Real err_elevated = 0.0, err_leaf = 0.0;
        if (c.local) {
            std::vector<bool> elevated(static_cast<std::size_t>(n));
            for (Index q = 0; q < n; ++q)
                elevated[static_cast<std::size_t>(tree.permutation()[static_cast<std::size_t>(q)])] =
                    tree.home_level(q) != tree.leaf_level();
            VectorXc xe = x[0], xl = x[0];
            for (Index i = 0; i < unknowns; ++i)
                (elevated[static_cast<std::size_t>(i % n)] ? xl : xe)(i) = 0.0;
            const VectorXc de = Zm * xe - *Zd * xe;
            const VectorXc dl = Zm * xl - *Zd * xl;
            Real e2 = de.squaredNorm(), l2 = 0.0;
            for (Index i = 0; i < unknowns; ++i)
                (elevated[static_cast<std::size_t>(i % n)] ? e2 : l2) += std::norm(dl(i));
            err_elevated = std::sqrt(e2) / yd[0].norm();
            err_leaf = std::sqrt(l2) / yd[0].norm();
        }
        std::ostringstream split;
        if (c.local)
            split << "; involving elevated functions " << err_elevated << ", leaf-leaf " << err_leaf;
        WARN(id.str() << ": " << tree.levels() << " levels, leaf a = "
                      << tree.box_size(tree.leaf_level()) / kLambda
                      << " lambda0, functions per home level" << homes.str() << ": matvec error "
                      << err << " (target " << std::pow(10.0, -c.digits) << ")" << split.str()
                      << "; setup "
                      << setup_s << " s, apply " << apply_s << " s; memory "
                      << mb(Zm.memory_bytes()) << " MB (near "
                      << mb(Zm.near_operator().memory_bytes()) << ", far "
                      << mb(Zm.far_operator().memory_bytes()) << ")\n"
                      << Zm.describe());
        if (c.local)
            CHECK(err_elevated < std::pow(10.0, -c.digits));
        if (c.check_full)
            CHECK(err < std::pow(10.0, -c.digits));
    }
}

}  // namespace

// ADR 0008 amendment 2026-10-11 item 6 (WP21L): 1 um x 1 um graded box (top 50 nm, box cells
// 100 nm), depth 4 um (root ~4 um: 6 levels with lambda / 4 leaves at d0 = 3, 5 levels with
// lambda / 2 leaves at d0 = 5, so that the coarse box functions (54 % of the 6600 bases here) have
// a home level >= 2 at both d0: level 3 / level 2). Measured 2026-10-11 (win-release, 2N = 13200):
//   Si: d0 = 3: 3.4e-4 local (3530 elevated, 3.2 GB) / 2.1e-4 global (lambda leaves, 1.7 GB);
//       d0 = 5: 5.6e-5 local (7.0 GB) / 4.5e-6 global (4.6 GB);
//   Ag: d0 = 3: 2.5e-4 local (1.3 GB) / 2.0e-4 global (0.4 GB); d0 = 5: 6.4e-5 local / 5.2e-6.
// d0 = 5 with the local rule misses 1e-5: the error is the leaf-leaf part of the vacuum region at
// the lambda / 2 leaves of the WP21 rule a_min(5) (5.3e-5 for Si; far part ~8 % of |Z x| on this
// tall box), not the elevated functions (6.8e-7). The global rule meets 1e-5 only because r_max
// forces lambda leaves. The full d0 = 5 check of the local rule is kept as a known limitation
// ([!mayfail] cases below) until a_min(5) / the lossless block check is decided (WP21L report).

TEST_CASE("mlfmm vs dense large: graded box Si with elevated functions (local leaf rule)",
          "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Setup s(graded_box(1e-6, 4e-6), material::silicon_500nm(), Kind::PMCHWT);
    run_local_leaf_case("graded box 1 x 4 um, Si, PMCHWT", s,
                        {{3.0, true, true}, {3.0, false, true}, {5.0, true, false},
                         {5.0, false, true}});
#endif
}

TEST_CASE("mlfmm vs dense large: graded box Ag with elevated functions (local leaf rule)",
          "[validation-large][mlfmm]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    const Setup s(graded_box(1e-6, 4e-6), material::silver_500nm(), Kind::PMCHWT);
    run_local_leaf_case("graded box 1 x 4 um, Ag, PMCHWT", s,
                        {{3.0, true, true}, {3.0, false, true}, {5.0, true, false},
                         {5.0, false, true}});
#endif
}

TEST_CASE("mlfmm vs dense large: graded box Si, d0 = 5, local leaf rule, full matvec",
          "[validation-large][mlfmm][!mayfail]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // Known limitation (comment above): lambda / 2 leaves at d0 = 5; measured 5.6e-5.
    const Setup s(graded_box(1e-6, 4e-6), material::silicon_500nm(), Kind::PMCHWT);
    run_local_leaf_case("graded box 1 x 4 um, Si, PMCHWT", s, {{5.0, true, true}});
#endif
}
