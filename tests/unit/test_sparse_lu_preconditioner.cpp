// Unit tests of the near-field-inverse ceiling (WP-SC1, ADR 0012 decision 1):
// solver::SparseLuPreconditioner (exact sparse LU, budget checks, errors, thread-count
// independence), near_field_matrix / make_near_field_lu_preconditioner on the small PML mesh of
// test_block_preconditioner.cpp, the ADR 0011 left-GMRES invariance for the near-field LU, and
// the blocks two levels above the leaves (block_levels_above_leaf = 2). Random matrices use
// fixed seeds (std::mt19937_64); the BEM cases use cheap kernel options (the near entries are
// compared with the same matrices, so accuracy does not matter).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"
#include "specklebem/solver/sparse_lu_preconditioner.hpp"

#include <Eigen/LU>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;
using Catch::Matchers::ContainsSubstring;
using formulation::Kind;
using solver::SparseLuPreconditioner;
using SparseMatrix = op::SparseOperator::Matrix;
using Triplet = Eigen::Triplet<Complex, Index>;

namespace {

constexpr Real kLambda = 500e-9;

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    VectorXc v(n);
    for (Index i = 0; i < n; ++i) v(i) = Complex(nd(rng), nd(rng));
    return v;
}

/// Random sparse n x n complex matrix: `per_row` off-diagonal entries per row at random columns
/// (normal real and imaginary parts), diagonal = 1 + the row's absolute sum (diagonally
/// dominant, well conditioned).
SparseMatrix random_sparse(Index n, Index per_row, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    std::uniform_int_distribution<Index> col(0, n - 1);
    std::vector<Triplet> t;
    for (Index i = 0; i < n; ++i) {
        std::set<Index> cols;
        while (static_cast<Index>(cols.size()) < per_row) {
            const Index c = col(rng);
            if (c != i)
                cols.insert(c);
        }
        Real sum = 0;
        for (const Index c : cols) {
            const Complex v(nd(rng), nd(rng));
            sum += std::abs(v);
            t.emplace_back(i, c, v);
        }
        t.emplace_back(i, i, Complex(1.0 + sum, 0.5));
    }
    SparseMatrix A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

SparseMatrix from_triplets(Index rows, Index cols, const std::vector<Triplet>& t) {
    SparseMatrix A(rows, cols);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

bool bitwise_equal(const MatrixXc& a, const MatrixXc& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
}

Real rel_diff(const VectorXc& a, const VectorXc& b) {
    return (a - b).norm() / b.norm();
}

/// Dense solve of A x = r with partial pivoting (the reference).
VectorXc dense_solve(const MatrixXc& A, const VectorXc& r) {
    const Eigen::PartialPivLU<MatrixXc> lu(A);
    return lu.solve(r);
}

/// Cheap kernels (as test_block_preconditioner.cpp).
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

/// Exactly `levels` octree levels; optionally every far pair exact (no expansions: only the near
/// part is used by these tests, and tiny leaves would not pass the expansion search).
mlfmm::MlfmmParams tree_params(int levels, bool exact_far = false) {
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, levels, 0.0};
    p.accuracy_digits = 3.0;
    p.automatic_leaf_size = false;
    if (exact_far)
        p.exact_far_regions = {true, true};
    return p;
}

/// Closed-mesh problem (vacuum exterior). Not movable (the space refers to the mesh).
struct SphereSetup {
    SphereSetup(geometry::TriangleMesh m, Kind kind, material::Material object, Real lambda)
        : mesh(std::move(m)), space(mesh), form(formulation::make_formulation(kind)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = object;
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / lambda;
        problem.kernel_options = cheap_options();
    }
    SphereSetup(const SphereSetup&) = delete;
    SphereSetup& operator=(const SphereSetup&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
};

/// The tiny rough open surface of test_pml_ring.cpp / test_block_preconditioner.cpp: patch 0.2,
/// taper 0.2, gap 0.1, PML 0.1 um at h = 100 nm, Gaussian patch (seed 29), 2N = 352 at the
/// flat case's size.
geometry::OpenSurfaceMesh tiny_rough_open() {
    geometry::OpenSurfaceParams op;
    op.patch_size = 0.2e-6;
    op.taper_width = 0.2e-6;
    op.gap_width = 0.1e-6;
    op.pml_width = 0.1e-6;
    geometry::RoughSurfaceParams rp;
    rp.mesh_size = 100e-9;
    rp.edge_length_L = op.patch_size + 2.0 * op.taper_width;
    rp.rms_roughness = 20e-9;
    rp.correlation_length = 200e-9;
    rp.seed = 29;
    return geometry::make_open_surface_mesh(geometry::generate_gaussian_height_map(rp), op);
}

/// Open-surface problem with the PML stretch (sigma_max 1, n = 1.5 object).
struct PmlSetup {
    PmlSetup()
        : om(tiny_rough_open()),
          space(om.mesh),
          form(formulation::make_formulation(Kind::PMCHWT)),
          pml(geometry::make_pml_stretch(om, 1.0)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = material::Material{Complex(2.25, 0.0), 1.0};
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

MatrixXc dense_matrix(const op::Problem& p) {
    const auto D = std::dynamic_pointer_cast<op::DenseOperator>(op::DenseStrategy().build(p));
    REQUIRE(D != nullptr);
    return D->matrix();
}

}  // namespace

TEST_CASE("sparse_lu_preconditioner: apply = dense LU solve, aliasing, sizes",
          "[sparse_lu_preconditioner]") {
    const Index n = 200;
    const SparseMatrix A = random_sparse(n, 6, 101);
    const MatrixXc Ad = MatrixXc(A);
    const SparseLuPreconditioner P(A);
    CHECK(P.name() == "sparse_lu");
    CHECK(P.size() == n);
    CHECK(P.input_nonzeros() == A.nonZeros());
    CHECK(P.factor_nonzeros() == P.nonzeros_l() + P.nonzeros_u());
    CHECK(P.factor_nonzeros() >= A.nonZeros());  // L + U hold the pattern of P_r A P_c
    CHECK(P.fill_factor() ==
          static_cast<Real>(P.factor_nonzeros()) / static_cast<Real>(A.nonZeros()));
    CHECK(P.memory_bytes() > 0);
    CHECK(P.memory_bytes() >=
          static_cast<std::size_t>(P.factor_nonzeros()) * SparseLuPreconditioner::kBytesPerEntry);
    CHECK(P.estimated_bytes() ==
          static_cast<std::size_t>(A.nonZeros()) * SparseLuPreconditioner::kBytesPerEntry);
    CHECK(P.setup_seconds() >= 0.0);
    CHECK_THAT(P.describe(), ContainsSubstring("sparse LU: 200 unknowns"));
    for (const std::uint64_t seed : {102U, 103U, 104U}) {
        const VectorXc r = random_vector(n, seed);
        VectorXc z;
        P.apply(r, z);
        const VectorXc ref = dense_solve(Ad, r);
        INFO("seed " << seed << ": relative difference " << rel_diff(z, ref));
        CHECK(rel_diff(z, ref) <= 1e-12);
        // Aliasing: apply(z, z).
        VectorXc w = r;
        P.apply(w, w);
        CHECK(bitwise_equal(w, z));
    }
    // A RowMajor input that is not compressed is accepted (converted).
    SparseMatrix U = A;
    U.uncompress();
    const SparseLuPreconditioner PU(U);
    const VectorXc r = random_vector(n, 105);
    VectorXc z1, z2;
    P.apply(r, z1);
    PU.apply(r, z2);
    CHECK(bitwise_equal(z1, z2));
}

TEST_CASE("sparse_lu_preconditioner: errors and the memory budget", "[sparse_lu_preconditioner]") {
    // Non-square, empty, non-finite, invalid expected fill.
    CHECK_THROWS_AS(SparseLuPreconditioner(SparseMatrix(3, 4)), std::invalid_argument);
    CHECK_THROWS_AS(SparseLuPreconditioner(SparseMatrix(0, 0)), std::invalid_argument);
    {
        std::vector<Triplet> t = {{0, 0, Complex(1.0, 0.0)},
                                  {1, 1, Complex(std::numeric_limits<Real>::quiet_NaN(), 0.0)}};
        CHECK_THROWS_WITH(SparseLuPreconditioner(from_triplets(2, 2, t)),
                          ContainsSubstring("non-finite entry (1, 1)"));
    }
    const SparseMatrix A = random_sparse(200, 6, 111);
    for (const Real fill : {0.0, -1.0, std::numeric_limits<Real>::infinity()})
        CHECK_THROWS_AS(SparseLuPreconditioner(A, std::nullopt, fill), std::invalid_argument);

    // Singular: a zero column, a zero row (structural), two equal rows (numerical).
    const auto singular = [](const std::vector<Triplet>& t) {
        CHECK_THROWS_WITH(SparseLuPreconditioner(from_triplets(4, 4, t)),
                          ContainsSubstring("the sparse LU failed"));
        CHECK_THROWS_AS(SparseLuPreconditioner(from_triplets(4, 4, t)), std::runtime_error);
    };
    const Complex one(1.0, 0.0), two(2.0, 1.0);
    // Column 2 empty.
    singular({{0, 0, one}, {1, 1, one}, {2, 1, two}, {3, 3, one}, {2, 3, two}});
    // Row 1 empty.
    singular({{0, 0, one}, {0, 1, two}, {2, 2, one}, {3, 3, one}, {2, 1, one}, {3, 0, two}});
    // Rows 0 and 1 equal (rank 3).
    singular({{0, 0, one},
              {0, 1, two},
              {1, 0, one},
              {1, 1, two},
              {2, 2, one},
              {3, 3, one},
              {2, 0, two}});

    // Size mismatch of apply.
    const SparseLuPreconditioner P(A);
    VectorXc z;
    CHECK_THROWS_AS(P.apply(VectorXc::Ones(199), z), std::invalid_argument);

    // Budget, pre-check: the estimate (expected_fill x nnz x 24 B) above the budget throws
    // before the factorisation.
    CHECK_THROWS_WITH(SparseLuPreconditioner(A, std::size_t{1000}),
                      ContainsSubstring("estimated factor memory"));
    CHECK_THROWS_AS(SparseLuPreconditioner(A, std::size_t{1000}), std::runtime_error);
    const std::size_t est = P.estimated_bytes();
    CHECK_THROWS_WITH(SparseLuPreconditioner(A, 3 * est - 1, 3.0),
                      ContainsSubstring("estimated factor memory"));
    // Budget, post-check: an estimate below the budget, but factors above it (fill > 1).
    INFO("estimate " << est << " B, factors " << P.memory_bytes() << " B, fill "
                     << P.fill_factor());
    REQUIRE(P.fill_factor() > 1.0);
    REQUIRE(P.memory_bytes() > est);
    const std::size_t between = (est + P.memory_bytes()) / 2;
    CHECK_THROWS_WITH(
        SparseLuPreconditioner(A, between),
        ContainsSubstring("the factor memory") && ContainsSubstring("exceeds the budget"));
    CHECK_THROWS_AS(SparseLuPreconditioner(A, between), std::runtime_error);
    // The exact factor memory as the budget passes; the default budget (half of the available
    // memory) too.
    const SparseLuPreconditioner exact(A, P.memory_bytes());
    CHECK(exact.memory_bytes() == P.memory_bytes());
    CHECK(exact.budget_bytes() == P.memory_bytes());
    CHECK(P.budget_bytes() > 0);
}

TEST_CASE("sparse_lu_preconditioner: setup and apply are bitwise independent of the thread count",
          "[sparse_lu_preconditioner]") {
#ifndef SPECKLEBEM_HAVE_OPENMP
    SKIP("built without OpenMP");
#else
    // Large enough for dense supernodes (fill ~37): without the single-thread scope of the
    // factorisation and the solve, the solve differs between 1 and 24 threads (checked with
    // the scope removed, win-release with OpenBLAS).
    const Index n = 1000;
    const SparseMatrix A = random_sparse(n, 3, 121);
    const VectorXc r = random_vector(n, 122);
    const int max_threads = omp_get_max_threads();
    const int many = std::max(4, max_threads);
    omp_set_num_threads(1);
    const SparseLuPreconditioner P1(A);
    VectorXc z11, z1m, zm1, zmm;
    P1.apply(r, z11);
    omp_set_num_threads(many);
    const SparseLuPreconditioner Pm(A);
    P1.apply(r, z1m);
    Pm.apply(r, zmm);
    omp_set_num_threads(1);
    Pm.apply(r, zm1);
    omp_set_num_threads(max_threads);
    INFO("threads 1 and " << many << ", fill " << P1.fill_factor());
    CHECK(P1.factor_nonzeros() == Pm.factor_nonzeros());
    CHECK(bitwise_equal(z11, z1m));
    CHECK(bitwise_equal(z11, zmm));
    CHECK(bitwise_equal(z11, zm1));
#endif
}

TEST_CASE("sparse_lu_preconditioner: near-field LU of the PML route = dense solve of Z_n",
          "[sparse_lu_preconditioner]") {
    // The tiny rough PML surface (2N = 352) with a forced 3-level tree: Z_n = Z_rr's near part +
    // the near ring, the near-field LU applied to a vector equals the dense LU solve with the
    // explicit Z_n, and GMRES on the near-only operator converges in one iteration.
    PmlSetup c;
    const mlfmm::MlfmmOperator Z(c.problem, tree_params(3));
    REQUIRE(Z.ring_operator() != nullptr);
    const SparseMatrix& rr = Z.near_operator().matrix();
    const SparseMatrix& ring = Z.ring_operator()->near_operator().matrix();
    // The explicit sum (dense), and near_field_matrix equal to it.
    const MatrixXc Zn_dense = MatrixXc(rr) + MatrixXc(ring);
    const SparseMatrix Zn = near_field_matrix(Z);
    CHECK(Zn.nonZeros() == rr.nonZeros() + ring.nonZeros());
    CHECK(bitwise_equal(MatrixXc(Zn), Zn_dense));
    const Index n2 = Z.rows();
    INFO("2N = " << n2 << ", nnz(Z_n) = " << Zn.nonZeros());
    CHECK(Zn.nonZeros() < n2 * n2);  // far pairs exist: Z_n is not the whole matrix
    const std::unique_ptr<SparseLuPreconditioner> P = make_near_field_lu_preconditioner(Z);
    REQUIRE(P != nullptr);
    CHECK(P->size() == n2);
    CHECK(P->input_nonzeros() == Zn.nonZeros());
    for (const std::uint64_t seed : {131U, 132U}) {
        const VectorXc r = random_vector(n2, seed);
        VectorXc z;
        P->apply(r, z);
        const Real d = rel_diff(z, dense_solve(Zn_dense, r));
        INFO("seed " << seed << ": relative difference to the dense solve " << d << ", "
                     << P->describe());
        // Measured (seeds 131 / 132): 5.1e-14 / 7.9e-14 (win-release), 6.5e-14 / 7.2e-14
        // (win-debug).
        CHECK(d <= 1e-10);
    }
    // GMRES with M^{-1} = Z_n^{-1} on the near-only operator: one iteration.
    const op::SparseOperator near_only{SparseMatrix(Zn)};
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.verbose = false;
    const VectorXc b = random_vector(n2, 133);
    const solver::GmresResult res = solver::gmres(near_only, b, *P, g);
    INFO("iterations " << res.iterations << ", true residual " << res.true_relative_residual);
    CHECK(res.converged);
    CHECK(res.iterations == 1);
    CHECK(rel_diff(near_only.matrix() * res.x, b) <= 1e-10);
    // A budget below the estimate is refused (the factory passes it on).
    CHECK_THROWS_WITH(make_near_field_lu_preconditioner(Z, std::size_t{1000}),
                      ContainsSubstring("estimated factor memory"));
    // Without a stretch: Z_rr's near part alone.
    SphereSetup s(geometry::make_icosphere(0.5 * kLambda, 1), Kind::PMCHWT,
                  material::Material{Complex(2.25, 0.0), 1.0}, kLambda);
    const mlfmm::MlfmmOperator Zs(s.problem, tree_params(2));
    REQUIRE(Zs.ring_operator() == nullptr);
    const SparseMatrix Zsn = near_field_matrix(Zs);
    CHECK(bitwise_equal(MatrixXc(Zsn), MatrixXc(Zs.near_operator().matrix())));
}

TEST_CASE("sparse_lu_preconditioner: PMCHWT, ICTF and MCTF give the same preconditioned system",
          "[sparse_lu_preconditioner]") {
    // The ADR 0011 invariance (test_block_preconditioner.cpp) for M = Z_n: PMCHWT, ICTF and MCTF
    // scale the J-test and the M-test rows by two constants, so Z_n scales by the same row
    // factors and Z_n^{-1} Z is the same matrix up to rounding (for any row scaling, not only a
    // block-constant one). Ag sphere R = 0.5 um, icosphere n = 2 at lambda = 1 um (2N = 960), a
    // forced 3-level tree (lambda / 4 leaves; far pairs exact, so the MLFMM operator is the
    // exact system and Z_n its near part).
    const Real lambda = 1e-6;
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 2);
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.max_iter = 500;
    g.verbose = false;
    std::vector<solver::GmresResult> res;
    std::vector<MatrixXc> pz;
    Real fill = 0;
    Index nnz = 0;
    Index n2 = 0;
#ifdef NDEBUG
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF, Kind::MCTF};
#else
    // Sanitizer build: one pair (as the block test).
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF};
#endif
    for (const Kind kind : kinds) {
        SphereSetup s(mesh, kind, material::silver_500nm(), lambda);
        const auto wave =
            std::make_shared<excitation::PlaneWave>(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
        s.problem.excitation = wave.get();
        s.problem.omega = wave->omega();
        const VectorXc b = op::assemble_rhs(s.problem);
        const mlfmm::MlfmmOperator Z(s.problem, tree_params(3, true));
        REQUIRE(Z.octree().levels() == 3);
        n2 = Z.rows();
        const std::unique_ptr<SparseLuPreconditioner> P = make_near_field_lu_preconditioner(Z);
        fill = P->fill_factor();
        nnz = P->input_nonzeros();
        res.push_back(solver::gmres(Z, b, *P, g));
        REQUIRE(res.back().converged);
        // M^{-1} Z column by column (Z e_j by the operator); the sanitizer build takes every
        // 16th column (the operator applies dominate its cost).
#ifdef NDEBUG
        const Index stride = 1;
#else
        const Index stride = 16;
#endif
        MatrixXc PZ(n2, (n2 + stride - 1) / stride);
        VectorXc e = VectorXc::Zero(n2), col;
        for (Index j = 0; j < n2; j += stride) {
            e(j) = 1.0;
            Z.apply(e, col);
            e(j) = 0.0;
            P->apply(col, col);
            PZ.col(j / stride) = col;
        }
        pz.push_back(std::move(PZ));
    }
    std::string its;
    for (std::size_t f = 0; f < res.size(); ++f)
        its += (f == 0 ? "" : ", ") + std::to_string(res[f].iterations);
    INFO("iterations (PMCHWT, ICTF[, MCTF]) " << its << ", nnz(Z_n) " << nnz << " of " << n2 * n2
                                              << ", fill " << fill);
    CHECK(nnz < n2 * n2);
    CHECK(res[0].iterations > 1);
    for (std::size_t f = 1; f < res.size(); ++f) {
        const Real dpz = (pz[f] - pz[0]).norm() / pz[0].norm();
        INFO("formulation " << f << ": |M^-1 Z - M0^-1 Z0| / |M0^-1 Z0| = " << dpz);
        // Measured 2.3e-11 (win-release, ICTF and MCTF, all columns), 2.1e-11 (win-debug, ICTF,
        // every 16th column): unlike the block LUs, the global pivot order of the sparse LU
        // changes with the row scaling, so the rounding of Z_n^{-1} differs.
        CHECK(dpz <= 1e-10);
        CHECK(std::abs(res[f].iterations - res[0].iterations) <= 1);
        const std::size_t m =
            std::min(res[f].residual_history.size(), res[0].residual_history.size());
        for (std::size_t k = 0; k < m && res[0].residual_history[k] >= 3e-2; ++k) {
            CHECK(std::abs(res[f].residual_history[k] - res[0].residual_history[k]) <=
                  1e-9 * res[0].residual_history[k]);
        }
        CHECK((res[f].x - res[0].x).norm() <= 1e-8 * res[0].x.norm());
    }
}

TEST_CASE("sparse_lu_preconditioner: blocks two levels above the leaves (near-field blocks)",
          "[sparse_lu_preconditioner]") {
    // Icosphere R = lambda, subdivision 3 (2N = 3840), a forced 4-level tree (lambda / 4 leaves,
    // r_max / a ~ 0.54): the level-1 boxes (two levels above the leaves) partition the [J; M]
    // unknowns and the MLFMM blocks are those of Z_n (same-box far pairs zero, counted). The
    // dense route (exact blocks) on the 2N = 240 sphere at 1 um.
    SphereSetup s(geometry::make_icosphere(kLambda, 3), Kind::PMCHWT,
                  material::Material{Complex(2.25, 0.0), 1.0}, kLambda);
    const mlfmm::MlfmmParams params = tree_params(4);
    const mlfmm::MlfmmOperator Z(s.problem, params);
    REQUIRE(Z.octree().levels() == 4);
    const auto P = make_block_preconditioner(Z, s.problem, params, 2);
    REQUIRE(P != nullptr);
    const solver::BlockPartition& part = P->partition();
    CHECK(part.ids == Z.octree().boxes_at_level(1));
    std::vector<int> count(sz(Z.rows()), 0);
    for (const std::vector<Index>& b : part.blocks)
        for (const Index i : b) ++count[sz(i)];
    CHECK(std::all_of(count.begin(), count.end(), [](int c) { return c == 1; }));
    // The blocks of Z_n: compare with the block preconditioner of the dense Z_n.
    const SparseMatrix Zn_sparse = near_field_matrix(Z);
    // Reference blocks Z_n(I_k, I_k) by entry lookup (absent entries read as zero).
    std::vector<MatrixXc> ref_blocks;
    for (const std::vector<Index>& b : part.blocks) {
        const auto m = static_cast<Index>(b.size());
        MatrixXc B(m, m);
        for (Index q = 0; q < m; ++q)
            for (Index p = 0; p < m; ++p) B(p, q) = Zn_sparse.coeff(b[sz(p)], b[sz(q)]);
        ref_blocks.push_back(std::move(B));
    }
    const solver::BlockDiagonalPreconditioner Pn(part, std::move(ref_blocks));
    // Same-block pairs outside the near pattern.
    std::vector<std::uint8_t> stored(sz(Z.rows() * Z.rows()), 0);
    for (Index i = 0; i < Zn_sparse.outerSize(); ++i) {
        for (SparseMatrix::InnerIterator it(Zn_sparse, i); it; ++it)
            stored[sz(it.row() * Z.rows() + it.col())] = 1;
    }
    Index outside = 0;
    for (const std::vector<Index>& b : part.blocks) {
        for (const Index i : b)
            for (const Index j : b) outside += stored[sz(i * Z.rows() + j)] == 0 ? 1 : 0;
    }
    INFO(P->describe());
    CHECK(P->missing_entries() > 0);
    CHECK(P->missing_entries() == outside);
    CHECK(P->block_entries() > P->missing_entries());
    CHECK_THAT(P->describe(), ContainsSubstring("in no source (zero)"));
    const VectorXc r = random_vector(Z.rows(), 141);
    VectorXc z1, z2;
    P->apply(r, z1);
    Pn.apply(r, z2);
    CHECK(bitwise_equal(z1, z2));
    // The completeness check still refuses these blocks when asked for (the default).
    CHECK_THROWS_WITH(solver::extract_diagonal_blocks(
                          part, std::vector<const SparseMatrix*>{&Z.near_operator().matrix()}),
                      ContainsSubstring("stored in no source"));
    CHECK_THROWS_AS(make_block_preconditioner(Z, s.problem, params, 3), std::invalid_argument);
    // Dense route: the exact blocks of Z (no missing entries; on a 3-level tree level 2 is the
    // root: one block = Z).
    SphereSetup d(geometry::make_icosphere(0.5e-6, 1), Kind::PMCHWT,
                  material::Material{Complex(2.25, 0.0), 1.0}, 1e-6);
    const MatrixXc D = dense_matrix(d.problem);
    const auto Pd = make_block_preconditioner(op::DenseOperator(D), d.problem, tree_params(3), 2);
    CHECK(Pd->missing_entries() == 0);
    CHECK(Pd->num_blocks() == 1);
    CHECK(Pd->max_block_size() == D.rows());
    const VectorXc rd = random_vector(D.rows(), 142);
    VectorXc zd;
    Pd->apply(rd, zd);
    CHECK(rel_diff(zd, dense_solve(D, rd)) <= 1e-12);
}

TEST_CASE("sparse_lu_preconditioner: Simulation accepts block level 2 and rejects 3",
          "[sparse_lu_preconditioner]") {
    // n = 1.5 sphere at 1 um (2N = 240), dense route, PMCHWT: level 2 on the 3-level tree of
    // tree_params(3) is the root (one block = the dense LU: one iteration); level 3 throws.
    SimulationConfig c;
    c.wavelength = 1e-6;
    c.object = material::Material{Complex(2.25, 0.0), 1.0};
    c.formulation = Kind::PMCHWT;
    c.kernels = cheap_options();
    c.gmres.verbose = false;
    c.gmres.tolerance = 1e-10;
    c.mlfmm = tree_params(3);
    c.preconditioner = PreconditionerKind::BlockDiagonal;
    const auto wave =
        std::make_shared<excitation::PlaneWave>(1e-6, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 1);
    c.block_levels_above_leaf = 2;
    Simulation sim(mesh, wave, c);
    const solver::GmresResult r = sim.solve();
    REQUIRE(sim.block_preconditioner() != nullptr);
    INFO(sim.report());
    CHECK(r.converged);
    CHECK(sim.block_preconditioner()->num_blocks() == 1);
    CHECK(r.iterations <= 1);
    CHECK(sim.report().find("blocks: two levels above the leaves") != std::string::npos);
    for (const int bad : {3, -1}) {
        c.block_levels_above_leaf = bad;
        CHECK_THROWS_WITH(Simulation(mesh, wave, c), ContainsSubstring("block_levels_above_leaf"));
    }
}
