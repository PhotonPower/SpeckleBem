// Unit tests of the near-field block-diagonal preconditioner (WP-PC1, ADR 0011):
// solver::BlockDiagonalPreconditioner / extract_diagonal_blocks, mlfmm::level_unknown_blocks,
// make_block_preconditioner on the MLFMM (box and PML route) and dense operators, left GMRES
// invariance under the formulations' block-row scaling (as WP13 for Jacobi), and the Simulation
// option. Random matrices use fixed seeds (std::mt19937_64); the BEM cases are small icospheres
// and the tiny open surface of test_pml_ring.cpp with cheap kernel options (the block entries are
// compared with the dense matrix of the same options, so accuracy does not matter).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/direct.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
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
using solver::BlockDiagonalPreconditioner;
using solver::BlockPartition;
using SparseMatrix = op::SparseOperator::Matrix;
using Sources = std::vector<const SparseMatrix*>;

namespace {

constexpr Real kLambda = 500e-9;

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

MatrixXc random_matrix(Index rows, Index cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    MatrixXc A(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) A(i, j) = Complex(nd(rng), nd(rng));
    }
    return A;
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    return random_matrix(n, 1, seed).col(0);
}

/// shift I + R / sqrt(2 n): well conditioned for shift 3, and so is every principal block.
MatrixXc well_conditioned(Index n, std::uint64_t seed, Real shift = 3.0) {
    MatrixXc A = random_matrix(n, n, seed) / std::sqrt(2.0 * static_cast<Real>(n));
    A.diagonal().array() += Complex(shift, 0.0);
    return A;
}

/// Random partition of [0, n) into `blocks` blocks of random sizes (a random permutation cut at
/// random points), ids 100 + k.
BlockPartition random_partition(Index n, Index blocks, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Index> perm(sz(n));
    std::iota(perm.begin(), perm.end(), Index{0});
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<Index> cuts(sz(n - 1));
    std::iota(cuts.begin(), cuts.end(), Index{1});
    std::shuffle(cuts.begin(), cuts.end(), rng);
    cuts.resize(sz(blocks - 1));
    std::sort(cuts.begin(), cuts.end());
    cuts.push_back(n);
    BlockPartition p;
    Index start = 0;
    for (const Index c : cuts) {
        p.blocks.emplace_back(perm.begin() + start, perm.begin() + c);
        p.ids.push_back(100 + static_cast<Index>(p.ids.size()));
        start = c;
    }
    return p;
}

/// A(I, I) of a block index list.
MatrixXc principal(const MatrixXc& A, const std::vector<Index>& idx) {
    const auto m = static_cast<Index>(idx.size());
    MatrixXc D(m, m);
    for (Index b = 0; b < m; ++b) {
        for (Index a = 0; a < m; ++a) D(a, b) = A(idx[sz(a)], idx[sz(b)]);
    }
    return D;
}

bool bitwise_equal(const MatrixXc& a, const MatrixXc& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
}

/// Sparse matrix of the entries of A where keep(r, c) is true.
template <class Keep>
SparseMatrix sparse_part(const MatrixXc& A, Keep keep) {
    std::vector<Eigen::Triplet<Complex, Index>> t;
    for (Index r = 0; r < A.rows(); ++r) {
        for (Index c = 0; c < A.cols(); ++c) {
            if (keep(r, c))
                t.emplace_back(r, c, A(r, c));
        }
    }
    SparseMatrix S(A.rows(), A.cols());
    S.setFromTriplets(t.begin(), t.end());
    S.makeCompressed();
    return S;
}

/// Max over the blocks of max |D_k - Z(I_k, I_k)| / max |Z(I_k, I_k)|, and whether every block
/// is bitwise equal.
struct BlockDiff {
    Real rel = 0;
    bool bitwise = true;
};
BlockDiff compare_blocks(const std::vector<MatrixXc>& D, const BlockPartition& p,
                         const MatrixXc& Z) {
    BlockDiff d;
    for (std::size_t k = 0; k < D.size(); ++k) {
        const MatrixXc ref = principal(Z, p.blocks[k]);
        d.bitwise = d.bitwise && bitwise_equal(D[k], ref);
        d.rel = std::max(d.rel, (D[k] - ref).cwiseAbs().maxCoeff() / ref.cwiseAbs().maxCoeff());
    }
    return d;
}

/// Cheap kernels (the block entries are compared with the dense matrix of the same options).
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

/// Exactly `levels` octree levels (one element per leaf requested, no size floor).
mlfmm::MlfmmParams tree_params(int levels, Real digits = 3.0) {
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, levels, 0.0};
    p.accuracy_digits = digits;
    p.automatic_leaf_size = false;
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

/// Icosphere R = lambda / 2, subdivision 2 (2N = 960): 3 forced levels give lambda / 4 leaves.
geometry::TriangleMesh small_sphere() {
    return geometry::make_icosphere(0.5 * kLambda, 2);
}

/// The tiny flat open surface of test_pml_ring.cpp: patch 0.2, taper 0.1, gap 0.1, PML 0.1 um
/// at h = 100 nm (8 x 8 cells, 2N = 352), and the tiny rough one (Gaussian patch, seed 29).
geometry::OpenSurfaceMesh tiny_open(bool rough) {
    geometry::OpenSurfaceParams op;
    op.patch_size = 0.2e-6;
    op.taper_width = rough ? 0.2e-6 : 0.1e-6;
    op.gap_width = 0.1e-6;
    op.pml_width = 0.1e-6;
    if (rough) {
        geometry::RoughSurfaceParams rp;
        rp.mesh_size = 100e-9;
        rp.edge_length_L = op.patch_size + 2.0 * op.taper_width;
        rp.rms_roughness = 20e-9;
        rp.correlation_length = 200e-9;
        rp.seed = 29;
        return geometry::make_open_surface_mesh(geometry::generate_gaussian_height_map(rp), op);
    }
    const Real edge = op.patch_size + 2.0 * op.taper_width;
    const Index cells = 4;
    geometry::HeightMap m;
    m.dx = edge / static_cast<Real>(cells);
    m.dy = m.dx;
    m.z = MatrixXr::Zero(cells + 1, cells + 1);
    return geometry::make_open_surface_mesh(m, op);
}

/// Open-surface problem with the PML stretch (sigma_max 1, n = 1.5 object).
struct PmlSetup {
    explicit PmlSetup(bool rough)
        : om(tiny_open(rough)),
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

BlockPartition octree_partition(const mlfmm::Octree& tree, int level) {
    BlockPartition p;
    p.blocks = mlfmm::level_unknown_blocks(tree, level, &p.ids);
    return p;
}

}  // namespace

TEST_CASE("block_preconditioner: partition validation", "[block_preconditioner]") {
    const MatrixXc A = well_conditioned(6, 1);
    const auto make = [&](BlockPartition p) {
        return BlockDiagonalPreconditioner(std::move(p), A);
    };
    CHECK_NOTHROW(make({{{0, 2}, {1, 3, 5}, {4}}, {}}));
    CHECK_THROWS_WITH(make({{{0, 2}, {1, 3, 5}}, {}}), ContainsSubstring("5 of the 6 unknowns"));
    CHECK_THROWS_WITH(make({{{0, 2}, {1, 3, 5}, {4, 2}}, {}}),
                      ContainsSubstring("more than one block"));
    CHECK_THROWS_WITH(make({{{0, 2}, {1, 3, 5}, {4, 6}}, {}}), ContainsSubstring("outside [0, 6)"));
    CHECK_THROWS_WITH(make({{{0, 2}, {1, 3, 5}, {4}, {}}, {}}), ContainsSubstring("empty"));
    CHECK_THROWS_WITH(make({{{0, 1, 2}, {3, 4, 5}}, {7}}), ContainsSubstring("block ids"));
    CHECK_THROWS_AS(make({{{0, 1, 2}, {3, 4, 5}}, {7}}), std::invalid_argument);
    CHECK_THROWS_AS(BlockDiagonalPreconditioner({{{0, 1, 2}, {3, 4, 5}}, {}}, MatrixXc(6, 5)),
                    std::invalid_argument);
    // Explicit blocks: count and sizes.
    std::vector<MatrixXc> blocks = {principal(A, {0, 1, 2}), principal(A, {3, 4, 5})};
    CHECK_NOTHROW(BlockDiagonalPreconditioner({{{0, 1, 2}, {3, 4, 5}}, {}}, blocks));
    CHECK_THROWS_WITH(
        BlockDiagonalPreconditioner({{{0, 1, 2}, {3, 4, 5}}, {}}, std::vector<MatrixXc>{blocks[0]}),
        ContainsSubstring("1 block matrices for 2 blocks"));
    CHECK_THROWS_WITH(BlockDiagonalPreconditioner({{{0, 1, 2}, {3, 4}, {5}}, {}}, blocks),
                      ContainsSubstring("2 block matrices for 3 blocks"));
    CHECK_THROWS_WITH(BlockDiagonalPreconditioner({{{0, 1}, {2, 3, 4, 5}}, {}}, blocks),
                      ContainsSubstring("expected 2 x 2"));
    CHECK_THROWS_AS(BlockDiagonalPreconditioner({{{0, 1, 2}, {3, 4, 5}}, {}}, blocks, -1.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(BlockDiagonalPreconditioner({{}, {}}, std::vector<MatrixXc>{}),
                    std::invalid_argument);
    const BlockDiagonalPreconditioner P({{{0, 1, 2}, {3, 4, 5}}, {}}, A);
    VectorXc z;
    CHECK_THROWS_AS(P.apply(VectorXc::Ones(5), z), std::invalid_argument);
}

TEST_CASE("block_preconditioner: dense blocks, apply = blockwise solve, aliasing, memory",
          "[block_preconditioner]") {
    const Index n = 300;
    const MatrixXc A = well_conditioned(n, 7);
    const BlockPartition part = random_partition(n, 9, 8);
    const std::vector<MatrixXc> D = solver::extract_diagonal_blocks(part, A);
    REQUIRE(D.size() == part.blocks.size());
    CHECK(compare_blocks(D, part, A).bitwise);

    const BlockDiagonalPreconditioner P(part, A);
    CHECK(P.size() == n);
    CHECK(P.num_blocks() == 9);
    CHECK(P.name() == "block_diagonal");
    const VectorXc r = random_vector(n, 9);
    VectorXc z;
    P.apply(r, z);
    REQUIRE(z.size() == n);
    // Reference: Eigen's dense solve of every block (another substitution order).
    Real err = 0;
    Index mmin = n, mmax = 0;
    std::size_t lu_bytes = 0;
    for (std::size_t k = 0; k < part.blocks.size(); ++k) {
        const std::vector<Index>& idx = part.blocks[k];
        const auto m = static_cast<Index>(idx.size());
        mmin = std::min(mmin, m);
        mmax = std::max(mmax, m);
        lu_bytes += sz(m * m) * sizeof(Complex);
        VectorXc rk(m), zk(m);
        for (Index a = 0; a < m; ++a) {
            rk(a) = r(idx[sz(a)]);
            zk(a) = z(idx[sz(a)]);
        }
        const VectorXc ref = principal(A, idx).fullPivLu().solve(rk);
        err = std::max(err, (zk - ref).norm() / ref.norm());
    }
    INFO("blockwise solve vs Eigen full-pivot LU: " << err);
    CHECK(err <= 1e-13);
    // M z = r blockwise, i.e. z solves the block-diagonal system.
    CHECK(P.min_block_size() == mmin);
    CHECK(P.max_block_size() == mmax);
    CHECK(P.memory_bytes() >= lu_bytes);
    CHECK(P.memory_bytes() <= 2 * lu_bytes + 4096 + 64 * sz(n));
    CHECK(P.min_reciprocal_condition() > 0.01);
    CHECK(P.worst_block_id() >= 100);
    CHECK(P.worst_block_id() < 109);
    CHECK(P.setup_seconds() >= P.factorization_seconds());
    CHECK_THAT(P.describe(), ContainsSubstring("block-diagonal: 9 blocks"));
    CHECK_THROWS_AS(P.block_lu(9), std::out_of_range);
    const MatrixXc B0 = principal(A, part.blocks[0]);
    CHECK((P.block_lu(0).reconstructedMatrix() - B0).cwiseAbs().maxCoeff() <=
          1e-14 * B0.cwiseAbs().maxCoeff() * static_cast<Real>(B0.rows()));
    // r and z may alias.
    VectorXc w = r;
    P.apply(w, w);
    CHECK(bitwise_equal(w, z));
    // A single block over everything is the exact inverse.
    std::vector<Index> all(sz(n));
    std::iota(all.begin(), all.end(), Index{0});
    const BlockDiagonalPreconditioner full({{all}, {}}, A);
    VectorXc x;
    full.apply(A * r, x);
    CHECK((x - r).norm() <= 1e-12 * r.norm());
}

TEST_CASE("block_preconditioner: sparse sources, every same-block entry exactly once",
          "[block_preconditioner]") {
    const Index n = 120;
    const MatrixXc A = well_conditioned(n, 21);
    const BlockPartition part = random_partition(n, 6, 22);
    std::vector<Index> block_of(sz(n));
    for (std::size_t k = 0; k < part.blocks.size(); ++k) {
        for (const Index i : part.blocks[k]) block_of[sz(i)] = static_cast<Index>(k);
    }
    const auto same = [&](Index r, Index c) { return block_of[sz(r)] == block_of[sz(c)]; };
    // Two sources that split the same-block entries (and hold some off-block entries, which
    // must be ignored): even rows + (r + c) odd off-block entries / the rest.
    const auto in_a = [&](Index r, Index c) { return same(r, c) ? (r + 2 * c) % 3 == 0 : true; };
    const SparseMatrix Sa = sparse_part(A, [&](Index r, Index c) { return in_a(r, c); });
    const SparseMatrix Sb =
        sparse_part(A, [&](Index r, Index c) { return same(r, c) && !in_a(r, c); });
    const std::vector<MatrixXc> D = solver::extract_diagonal_blocks(part, Sources{&Sa, &Sb});
    CHECK(compare_blocks(D, part, A).bitwise);
    const BlockDiagonalPreconditioner Ps(part, Sources{&Sa, &Sb});
    const BlockDiagonalPreconditioner Pd(part, A);
    const VectorXc rv = random_vector(n, 23);
    VectorXc zs, zd;
    Ps.apply(rv, zs);
    Pd.apply(rv, zd);
    CHECK(bitwise_equal(zs, zd));
    CHECK(Ps.extraction_seconds() >= 0.0);

    // An entry in both sources (would be counted twice) and an entry in none are refused, with
    // the block id: (r1, c1) is an entry of Sa in block 2, (r0, c0) one of Sb.
    Index r0 = -1, c0 = -1, r1 = -1, c1 = -1;
    for (const Index r : part.blocks[2]) {
        for (const Index c : part.blocks[2]) {
            if (in_a(r, c) && r1 < 0) {
                r1 = r;
                c1 = c;
            }
            if (!in_a(r, c) && r0 < 0) {
                r0 = r;
                c0 = c;
            }
        }
    }
    REQUIRE(r0 >= 0);
    REQUIRE(r1 >= 0);
    const SparseMatrix Sdup = sparse_part(
        A, [&](Index r, Index c) { return (r == r1 && c == c1) || (same(r, c) && !in_a(r, c)); });
    CHECK_THROWS_WITH(solver::extract_diagonal_blocks(part, Sources{&Sa, &Sdup}),
                      ContainsSubstring("(" + std::to_string(r1) + ", " + std::to_string(c1) +
                                        ") of block 102 is stored in more than one source"));
    const SparseMatrix Smiss = sparse_part(
        A, [&](Index r, Index c) { return same(r, c) && !in_a(r, c) && !(r == r0 && c == c0); });
    CHECK_THROWS_WITH(solver::extract_diagonal_blocks(part, Sources{&Sa, &Smiss}),
                      ContainsSubstring("(" + std::to_string(r0) + ", " + std::to_string(c0) +
                                        ") of block 102 is stored in no source"));
    // Source checks.
    CHECK_THROWS_AS(solver::extract_diagonal_blocks(part, Sources{}), std::invalid_argument);
    CHECK_THROWS_AS(solver::extract_diagonal_blocks(part, Sources{&Sa, nullptr}),
                    std::invalid_argument);
    const SparseMatrix small =
        sparse_part(MatrixXc(A.topLeftCorner(n - 1, n - 1)), [](Index, Index) { return true; });
    CHECK_THROWS_AS(solver::extract_diagonal_blocks(part, Sources{&Sa, &small}),
                    std::invalid_argument);
}

TEST_CASE("block_preconditioner: singular and ill-conditioned blocks are refused",
          "[block_preconditioner]") {
    const Index n = 40;
    MatrixXc A = well_conditioned(n, 31);
    BlockPartition part;
    part.blocks = {{}, {}, {}};
    for (Index i = 0; i < n; ++i) part.blocks[sz(i % 3)].push_back(i);
    part.ids = {7, 8, 9};
    // Block 8 (the unknowns 1, 4, 7, ...): two equal rows -> singular.
    MatrixXc S = A;
    for (const Index c : part.blocks[1]) S(4, c) = S(1, c);
    CHECK_THROWS_AS(BlockDiagonalPreconditioner(part, S), std::runtime_error);
    CHECK_THROWS_WITH(BlockDiagonalPreconditioner(part, S),
                      ContainsSubstring("block 8 (13 unknowns) is singular or ill-conditioned"));
    // Nearly singular (rcond ~ 1e-15): refused by default, accepted with a lower limit.
    MatrixXc T = A;
    for (const Index c : part.blocks[2]) T(5, c) = T(2, c) * (1.0 + 1e-15);
    T(5, 5) += 1e-15;
    CHECK_THROWS_WITH(BlockDiagonalPreconditioner(part, T), ContainsSubstring("block 9"));
    CHECK_NOTHROW(BlockDiagonalPreconditioner(part, T, 1e-300));
    // Non-finite entries.
    MatrixXc U = A;
    U(0, 3) = Complex(std::numeric_limits<Real>::quiet_NaN(), 0.0);
    CHECK_THROWS_WITH(BlockDiagonalPreconditioner(part, U), ContainsSubstring("non-finite"));
    // A well-conditioned partition of the same matrix is fine.
    CHECK_NOTHROW(BlockDiagonalPreconditioner(part, A));
}

TEST_CASE("block_preconditioner: setup and apply are bitwise independent of the thread count",
          "[block_preconditioner]") {
#ifndef SPECKLEBEM_HAVE_OPENMP
    SKIP("built without OpenMP");
#else
    const Index n = 900;
    const MatrixXc A = well_conditioned(n, 41);
    // Blocks of up to ~200 unknowns: Eigen's blocked LU (and its BLAS calls) is used.
    const BlockPartition part = random_partition(n, 7, 42);
    const VectorXc r = random_vector(n, 43);
    // At least 4 threads, so that a CI run with OMP_NUM_THREADS=1 still compares two counts.
    const int max_threads = omp_get_max_threads();
    const int many = std::max(4, max_threads);
    omp_set_num_threads(1);
    const BlockDiagonalPreconditioner P1(part, A);
    VectorXc z11, z1m, zm1, zmm;
    P1.apply(r, z11);
    omp_set_num_threads(many);
    const BlockDiagonalPreconditioner Pm(part, A);
    P1.apply(r, z1m);
    Pm.apply(r, zmm);
    omp_set_num_threads(1);
    Pm.apply(r, zm1);
    omp_set_num_threads(max_threads);
    INFO("threads 1 and " << many);
    CHECK(bitwise_equal(z11, z1m));
    CHECK(bitwise_equal(z11, zmm));
    CHECK(bitwise_equal(z11, zm1));
    for (Index k = 0; k < P1.num_blocks(); ++k)
        CHECK(bitwise_equal(P1.block_lu(k).matrixLU(), Pm.block_lu(k).matrixLU()));
#endif
}

TEST_CASE("block_preconditioner: octree level blocks partition the [J; M] unknowns",
          "[block_preconditioner]") {
    const geometry::TriangleMesh mesh = small_sphere();
    const basis::RwgSpace space(mesh);
    const mlfmm::Octree tree(space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const auto N = static_cast<Index>(space.size());
    for (int level = 0; level < tree.levels(); ++level) {
        std::vector<Index> ids;
        const std::vector<std::vector<Index>> b = mlfmm::level_unknown_blocks(tree, level, &ids);
        REQUIRE(b.size() == tree.boxes_at_level(level).size());
        CHECK(ids == tree.boxes_at_level(level));
        std::vector<int> count(sz(2 * N), 0);
        for (std::size_t k = 0; k < b.size(); ++k) {
            const mlfmm::Box& box = tree.boxes()[sz(ids[k])];
            REQUIRE(static_cast<Index>(b[k].size()) == 2 * box.num_elements);
            for (Index a = 0; a < box.num_elements; ++a) {
                // J first (permuted order), then M with the offset N.
                CHECK(b[k][sz(a)] == tree.permutation()[sz(box.first_element + a)]);
                CHECK(b[k][sz(box.num_elements + a)] == b[k][sz(a)] + N);
            }
            for (const Index i : b[k]) ++count[sz(i)];
        }
        CHECK(std::all_of(count.begin(), count.end(), [](int c) { return c == 1; }));
    }
    CHECK_THROWS_AS(mlfmm::level_unknown_blocks(tree, tree.levels()), std::out_of_range);
}

TEST_CASE("block_preconditioner: MLFMM near-field blocks equal the dense blocks (box route)",
          "[block_preconditioner]") {
    SphereSetup s(small_sphere(), Kind::PMCHWT, material::Material{Complex(2.25, 0.0), 1.0},
                  kLambda);
    const mlfmm::MlfmmOperator Z(s.problem, tree_params(3));
    REQUIRE(Z.octree().levels() == 3);
    REQUIRE(Z.ring_operator() == nullptr);
    const MatrixXc D = dense_matrix(s.problem);
    for (const int up : {0, 1}) {
        const BlockPartition part = octree_partition(Z.octree(), Z.octree().leaf_level() - up);
        const std::vector<MatrixXc> blocks =
            solver::extract_diagonal_blocks(part, Sources{&Z.near_operator().matrix()});
        const BlockDiff d = compare_blocks(blocks, part, D);
        INFO("levels above the leaf " << up << ": " << part.blocks.size() << " blocks, max diff "
                                      << d.rel);
        CHECK(d.bitwise);  // the near field is bitwise the dense entries (WP19)
        // make_block_preconditioner builds the same partition and blocks.
        const auto P = make_block_preconditioner(Z, s.problem, tree_params(3), up);
        REQUIRE(P->num_blocks() == static_cast<Index>(part.blocks.size()));
        CHECK(P->partition().ids == part.ids);
        const BlockDiagonalPreconditioner Pd(part, D);
        const VectorXc r = random_vector(Z.rows(), 51);
        VectorXc z1, z2;
        P->apply(r, z1);
        Pd.apply(r, z2);
        CHECK(bitwise_equal(z1, z2));
    }
    // Two levels above the leaf the box pairs are no longer all near: refused by the default
    // extraction (no silent zeros); make_block_preconditioner offers level 2 explicitly as the
    // blocks of the near-field matrix (WP-SC1, test_sparse_lu_preconditioner.cpp) and refuses 3.
    const BlockPartition root = octree_partition(Z.octree(), 0);
    CHECK_THROWS_WITH(solver::extract_diagonal_blocks(root, Sources{&Z.near_operator().matrix()}),
                      ContainsSubstring("stored in no source"));
    CHECK(make_block_preconditioner(Z, s.problem, tree_params(3), 2)->missing_entries() > 0);
    CHECK_THROWS_AS(make_block_preconditioner(Z, s.problem, tree_params(3), 3),
                    std::invalid_argument);
    CHECK_THROWS_AS(make_block_preconditioner(Z, s.problem, tree_params(3), -1),
                    std::invalid_argument);
    // Dense route: the octree of make_octree(params) and the dense matrix's blocks.
    const op::DenseOperator dense(D);
    const auto Pd = make_block_preconditioner(dense, s.problem, tree_params(3), 0);
    const auto Pf = make_block_preconditioner(Z, s.problem, tree_params(3), 0);
    const VectorXc r = random_vector(Z.rows(), 52);
    VectorXc z1, z2;
    Pd->apply(r, z1);
    Pf->apply(r, z2);
    CHECK(bitwise_equal(z1, z2));
    // Wrong operator size / type.
    CHECK_THROWS_AS(make_block_preconditioner(op::DenseOperator(MatrixXc::Identity(10, 10)),
                                              s.problem, tree_params(3), 0),
                    std::invalid_argument);
    const op::SparseOperator near_copy{SparseMatrix(Z.near_operator().matrix())};
    CHECK_THROWS_WITH(make_block_preconditioner(near_copy, s.problem, tree_params(3), 0),
                      ContainsSubstring("neither an MLFMM nor a dense operator"));
}

TEST_CASE("block_preconditioner: PML route, Z_rr near + near ring hold every same-box entry once",
          "[block_preconditioner]") {
#ifdef NDEBUG
    const std::vector<bool> cases = {false, true};
#else
    // Sanitizer build: the rough case only (each case builds the MLFMM operator and the dense
    // PML matrix; the flat one adds no code path).
    const std::vector<bool> cases = {true};
#endif
    for (const bool rough : cases) {
        PmlSetup c(rough);
        const mlfmm::MlfmmOperator Z(c.problem, tree_params(3));
        REQUIRE(Z.ring_operator() != nullptr);
        const MatrixXc D = dense_matrix(c.problem);
        const std::vector<std::uint8_t>& stretched = Z.ring_operator()->stretched();
        const Index N = c.space.size();
        for (const int up : {0, 1}) {
            const BlockPartition part = octree_partition(Z.octree(), Z.octree().leaf_level() - up);
            const SparseMatrix& rr = Z.near_operator().matrix();
            const SparseMatrix& ring = Z.ring_operator()->near_operator().matrix();
            // Exactly once: extraction throws for an entry in both or in neither.
            const std::vector<MatrixXc> blocks =
                solver::extract_diagonal_blocks(part, Sources{&rr, &ring});
            // Each near part alone misses entries (both are needed).
            CHECK_THROWS_WITH(solver::extract_diagonal_blocks(part, Sources{&rr}),
                              ContainsSubstring("stored in no source"));
            CHECK_THROWS_WITH(solver::extract_diagonal_blocks(part, Sources{&ring}),
                              ContainsSubstring("stored in no source"));
            // Against the dense PML matrix: real-real entries bitwise, entries with a stretched
            // member to rounding (summation order of the stretched dense entries).
            const BlockDiff d = compare_blocks(blocks, part, D);
            Real rr_diff = 0;
            for (std::size_t k = 0; k < blocks.size(); ++k) {
                const std::vector<Index>& idx = part.blocks[k];
                for (std::size_t a = 0; a < idx.size(); ++a) {
                    for (std::size_t b = 0; b < idx.size(); ++b) {
                        if (stretched[sz(idx[a] % N)] == 0 && stretched[sz(idx[b] % N)] == 0) {
                            rr_diff = std::max(rr_diff, std::abs(blocks[k](static_cast<Index>(a),
                                                                           static_cast<Index>(b)) -
                                                                 D(idx[a], idx[b])));
                        }
                    }
                }
            }
            INFO((rough ? "rough" : "flat")
                 << ", levels above the leaf " << up << ": " << part.blocks.size()
                 << " blocks, max rel diff " << d.rel);
            CHECK(rr_diff == 0.0);
            CHECK(d.rel <= 1e-14);
            CHECK_NOTHROW(make_block_preconditioner(Z, c.problem, tree_params(3), up));
        }
    }
}

TEST_CASE("block_preconditioner: left GMRES is invariant under block-constant row scaling",
          "[block_preconditioner]") {
    // As the WP13 Jacobi test (test_gmres.cpp, issue #15): rows scaled by 1e-3 .. 1e3, here
    // with one factor per block, so the block preconditioner of S A is S P: M^{-1} A unchanged.
    const Index n = 200;
    const MatrixXc A = well_conditioned(n, 61);
    const VectorXc b = random_vector(n, 62);
    const BlockPartition part = random_partition(n, 10, 63);
    std::mt19937_64 rng(64);
    std::uniform_real_distribution<Real> ex(-3.0, 3.0);
    VectorXc s(n);
    for (const std::vector<Index>& blk : part.blocks) {
        const Complex f = std::pow(10.0, ex(rng)) * std::polar(1.0, ex(rng));
        for (const Index i : blk) s(i) = f;
    }
    const MatrixXc SA = s.asDiagonal() * A;
    const BlockDiagonalPreconditioner P(part, A);
    const BlockDiagonalPreconditioner PS(part, SA);
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.verbose = false;
    const solver::GmresResult r = solver::gmres(op::DenseOperator(A), b, P, g);
    const solver::GmresResult rs =
        solver::gmres(op::DenseOperator(SA), VectorXc(s.cwiseProduct(b)), PS, g);
    REQUIRE(r.converged);
    REQUIRE(rs.converged);
    CHECK(r.iterations == rs.iterations);
    REQUIRE(r.residual_history.size() == rs.residual_history.size());
    for (std::size_t k = 0; k < r.residual_history.size(); ++k) {
        CHECK(std::abs(r.residual_history[k] - rs.residual_history[k]) <=
              1e-10 * r.residual_history[k] + 1e-14);
    }
    CHECK((rs.x - r.x).norm() <= 1e-9 * r.x.norm());
}

TEST_CASE("block_preconditioner: PMCHWT, ICTF and MCTF give the same preconditioned system",
          "[block_preconditioner]") {
    // PMCHWT, ICTF and MCTF are block-row scalings of each other (formulation.hpp: the J-test
    // rows by one constant, the M-test rows by another, region-independent), and every block
    // holds whole J and M rows of a box, so M^{-1} Z is the same matrix up to rounding and left
    // GMRES runs the same iterates. Ag sphere R = 0.5 um, icosphere n = 1 at lambda = 1 um (2N =
    // 240), leaf boxes of a forced 3-level tree (48 blocks). Measured (win-release): M^{-1} Z
    // equal to 6.6e-16 relative; the residual histories equal to 1e-11 for the first 24
    // iterations, after which rounding grows (5e-7 at 32, 1e-2 at 40 iterations) as it does for
    // left Jacobi on these systems (5e-8 at 24, 7e-2 at 40); 88 iterations to 1e-10 for all three.
    const Real lambda = 1e-6;
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 1);
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.max_iter = 500;
    g.verbose = false;
    std::vector<solver::GmresResult> res;
    std::vector<MatrixXc> pz;
    Index blocks = 0;
#ifdef NDEBUG
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF, Kind::MCTF};
#else
    // Sanitizer build: one pair (the three dense Ag assemblies and solves dominate the cost);
    // the three-way check incl. MCTF runs in the release build and release CI only.
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF};
#endif
    for (const Kind kind : kinds) {
        SphereSetup s(mesh, kind, material::silver_500nm(), lambda);
        const auto wave =
            std::make_shared<excitation::PlaneWave>(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
        s.problem.excitation = wave.get();
        s.problem.omega = wave->omega();
        const MatrixXc Z = dense_matrix(s.problem);
        const VectorXc b = op::assemble_rhs(s.problem);
        const mlfmm::Octree tree(s.space, lambda, mlfmm::OctreeParams{1, 3, 0.0});
        const BlockDiagonalPreconditioner P(octree_partition(tree, tree.leaf_level()), Z);
        blocks = P.num_blocks();
        res.push_back(solver::gmres(op::DenseOperator(Z), b, P, g));
        REQUIRE(res.back().converged);
        // M^{-1} Z block row by block row (one LU solve with all columns per block, instead of
        // 2N applies, each an OpenMP region: the sanitizer build under ctest -j load).
        MatrixXc PZ(Z.rows(), Z.cols());
        for (Index k = 0; k < P.num_blocks(); ++k) {
            const std::vector<Index>& idx = P.partition().blocks[sz(k)];
            MatrixXc rows(static_cast<Index>(idx.size()), Z.cols());
            for (std::size_t a = 0; a < idx.size(); ++a)
                rows.row(static_cast<Index>(a)) = Z.row(idx[a]);
            const MatrixXc sol = P.block_lu(k).solve(rows);
            for (std::size_t a = 0; a < idx.size(); ++a)
                PZ.row(idx[a]) = sol.row(static_cast<Index>(a));
        }
        pz.push_back(std::move(PZ));
    }
    std::string its;
    for (std::size_t f = 0; f < res.size(); ++f)
        its += (f == 0 ? "" : ", ") + std::to_string(res[f].iterations);
    INFO("iterations (PMCHWT, ICTF[, MCTF]) " << its << " (" << blocks << " blocks)");
    CHECK(blocks > 4);
    for (std::size_t f = 1; f < res.size(); ++f) {
        CHECK((pz[f] - pz[0]).norm() <= 1e-13 * pz[0].norm());
        // Iterates equal up to rounding: while rounding has not grown (residual >= 3e-2), to
        // 1e-9; afterwards the counts agree to one iteration and the solutions to the tolerance.
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

namespace {

constexpr Real kLambdaFast = 1e-6;

SimulationConfig sim_config(const material::Material& object) {
    SimulationConfig c;
    c.wavelength = kLambdaFast;
    c.object = object;
    c.kernels = cheap_options();
    c.gmres.verbose = false;
    c.gmres.tolerance = 1e-10;
    c.mlfmm = tree_params(3);
    return c;
}

std::shared_ptr<excitation::Excitation> wave_1um() {
    return std::make_shared<excitation::PlaneWave>(kLambdaFast, Vec3::UnitZ(),
                                                   Vec3c(1.0, 0.0, 0.0));
}

geometry::TriangleMesh sphere_1um() {
    return geometry::make_icosphere(0.5e-6, 1);
}

bool contains(const std::string& s, const std::string& what) {
    return s.find(what) != std::string::npos;
}

}  // namespace

TEST_CASE("block_preconditioner: Simulation option, defaults and consistency",
          "[block_preconditioner]") {
    const material::Material ag = material::silver_500nm();
    {
        // Default unchanged: Jacobi for metals, none for dielectrics (automatic).
        const Simulation a(sphere_1um(), wave_1um(), sim_config(ag));
        CHECK(a.preconditioner_kind() == PreconditionerKind::Diagonal);
        CHECK(a.diagonal_preconditioner());
        CHECK(contains(a.report(), "diagonal (Jacobi) (automatic)"));
        const Simulation si(sphere_1um(), wave_1um(), sim_config(material::silicon_500nm()));
        CHECK(si.preconditioner_kind() == PreconditionerKind::None);
        CHECK(contains(si.report(), "preconditioner: none (automatic)"));
    }
    const auto with = [&](std::optional<PreconditionerKind> k, std::optional<bool> d,
                          int levels = 0) {
        SimulationConfig c = sim_config(ag);
        c.preconditioner = k;
        c.diagonal_preconditioner = d;
        c.block_levels_above_leaf = levels;
        return c;
    };
    // Consistent combinations.
    CHECK(Simulation(sphere_1um(), wave_1um(), with(PreconditionerKind::Diagonal, true))
              .preconditioner_kind() == PreconditionerKind::Diagonal);
    CHECK(Simulation(sphere_1um(), wave_1um(), with(PreconditionerKind::None, false))
              .preconditioner_kind() == PreconditionerKind::None);
    CHECK(Simulation(sphere_1um(), wave_1um(), with(std::nullopt, false)).preconditioner_kind() ==
          PreconditionerKind::None);
    const Simulation explicit_none(sphere_1um(), wave_1um(), with(PreconditionerKind::None, {}));
    CHECK(explicit_none.preconditioner_kind() == PreconditionerKind::None);
    CHECK(contains(explicit_none.report(), "preconditioner: none (explicit)"));
    // Contradictions and invalid values.
    for (const auto& [k, d] : std::vector<std::pair<PreconditionerKind, bool>>{
             {PreconditionerKind::Diagonal, false},
             {PreconditionerKind::None, true},
             {PreconditionerKind::BlockDiagonal, true},
             {PreconditionerKind::BlockDiagonal, false}}) {
        CHECK_THROWS_WITH(Simulation(sphere_1um(), wave_1um(), with(k, d)),
                          ContainsSubstring("contradicts diagonal_preconditioner"));
    }
    CHECK_THROWS_AS(
        Simulation(sphere_1um(), wave_1um(), with(static_cast<PreconditionerKind>(7), {})),
        std::invalid_argument);
    for (const int levels : {-1, 3}) {
        CHECK_THROWS_WITH(Simulation(sphere_1um(), wave_1um(),
                                     with(PreconditionerKind::BlockDiagonal, {}, levels)),
                          ContainsSubstring("block_levels_above_leaf"));
    }
    SimulationConfig bad = with(PreconditionerKind::BlockDiagonal, {});
    bad.mlfmm.octree.max_levels = 0;  // the dense route's partition needs valid octree params
    CHECK_THROWS_AS(Simulation(sphere_1um(), wave_1um(), bad), std::invalid_argument);
    CHECK(std::string(to_string(PreconditionerKind::BlockDiagonal)) == "block_diagonal");
    CHECK(std::string(to_string(PreconditionerKind::Diagonal)) == "diagonal");
    CHECK(std::string(to_string(PreconditionerKind::None)) == "none");
}

TEST_CASE("block_preconditioner: Simulation solves with the block preconditioner",
          "[block_preconditioner]") {
    // n = 1.5 sphere, PMCHWT, GMRES tolerance 1e-8: dense route on the 2N = 240 sphere at 1 um
    // (both block levels, against the direct solution), MLFMM on the 2N = 960 sphere at 500 nm
    // with the forced 3-level tree of the MLFMM unit tests (lambda / 4 leaves).
    const material::Material n15{Complex(2.25, 0.0), 1.0};
    struct Run {
        std::string compression;
        int levels;
    };
    for (const Run& run : {Run{"dense", 0}, Run{"dense", 1}, Run{"mlfmm", 0}}) {
#ifdef NDEBUG
        const bool dense = run.compression == "dense";
#else
        // Sanitizer build: the MLFMM route on the 2N = 240 sphere with a 2-level tree (every
        // pair near, no far part: the wiring only), as the 960 sphere's direct solve is slow.
        const bool dense = true;
#endif
        SimulationConfig c = sim_config(n15);
        c.formulation = Kind::PMCHWT;
        c.gmres.tolerance = 1e-8;
        if (!dense)
            c.wavelength = kLambda;
        if (dense && run.compression == "mlfmm")
            c.mlfmm = tree_params(2);
        const auto wave = dense ? wave_1um()
                                : std::make_shared<excitation::PlaneWave>(kLambda, Vec3::UnitZ(),
                                                                          Vec3c(1.0, 0.0, 0.0));
        const geometry::TriangleMesh mesh = dense ? sphere_1um() : small_sphere();
        SimulationConfig cd = c;
        cd.solver = SolverKind::Direct;
        Simulation direct(mesh, wave, cd);
        direct.solve();
        const VectorXc& x_ref = direct.solution().currents;
        c.compression = run.compression;
        c.preconditioner = PreconditionerKind::BlockDiagonal;
        c.block_levels_above_leaf = run.levels;
        Simulation sim(mesh, wave, c);
        CHECK(sim.block_preconditioner() == nullptr);
        CHECK_FALSE(sim.diagonal_preconditioner());
        CHECK(sim.preconditioner_kind() == PreconditionerKind::BlockDiagonal);
        const solver::GmresResult r = sim.solve();
        REQUIRE(sim.block_preconditioner() != nullptr);
        const std::string rep = sim.report();
        INFO(run.compression << ", levels above the leaf " << run.levels << ": " << r.iterations
                             << " iterations\n"
                             << rep);
        CHECK(r.converged);
        const Real err = (sim.solution().currents - x_ref).norm() / x_ref.norm();
        // dense: the direct solution to the GMRES tolerance; mlfmm: its matvec error (cheap
        // patterns on a forced tree) dominates.
        CHECK(err < (dense ? 1e-6 : 5e-2));
        CHECK(contains(rep, "block-diagonal (near field) (explicit)"));
        CHECK(contains(
            rep, run.levels == 0 ? "blocks: octree leaves" : "blocks: one level above the leaves"));
        CHECK(contains(rep, "preconditioner setup:"));
        CHECK(contains(rep, "block-diagonal preconditioner"));
        // Idempotent assembly: the preconditioner is built once.
        const solver::BlockDiagonalPreconditioner* p = sim.block_preconditioner();
        sim.assemble();
        CHECK(sim.block_preconditioner() == p);
    }
    // The direct solver does not build it.
    SimulationConfig cb = sim_config(material::silver_500nm());
    cb.solver = SolverKind::Direct;
    cb.preconditioner = PreconditionerKind::BlockDiagonal;
    Simulation d2(sphere_1um(), wave_1um(), cb);
    d2.solve();
    CHECK(d2.block_preconditioner() == nullptr);
    CHECK(contains(d2.report(), "unused by the direct solver"));
}

namespace {

/// Captures log lines of at least the given level while alive (as test_open_surface.cpp).
class LogCapture {
public:
    explicit LogCapture(spdlog::level::level_enum level)
        : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64)),
          saved_level_(spdlog::default_logger()->level()) {
        sink_->set_level(level);
        spdlog::default_logger()->set_level(std::min(saved_level_, level));
        spdlog::default_logger()->sinks().push_back(sink_);
    }
    ~LogCapture() {
        auto& sinks = spdlog::default_logger()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
        spdlog::default_logger()->set_level(saved_level_);
    }
    LogCapture(const LogCapture&) = delete;
    LogCapture& operator=(const LogCapture&) = delete;
    [[nodiscard]] std::size_t count(const std::string& text) const {
        const std::vector<std::string> lines = sink_->last_formatted();
        return static_cast<std::size_t>(
            std::count_if(lines.begin(), lines.end(),
                          [&](const std::string& l) { return l.find(text) != std::string::npos; }));
    }

private:
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
    spdlog::level::level_enum saved_level_;
};

/// The warning of the automatic block default on a coarse tree (simulation.cpp, ensure_block).
constexpr const char* kCoarseBlockWarning =
    "automatic block-diagonal preconditioner's largest block";

/// Tiny flat PML case of the default tests: n = 1.5 object, cheap kernels, GMRES 1e-9.
struct TinyPml {
    TinyPml() {
        cfg.wavelength = kLambda;
        cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
        cfg.kernels = cheap_options();
        cfg.gmres.tolerance = 1e-9;
        cfg.gmres.verbose = false;
        term.sigma_max = 1.0;
        term.dark_band_tolerance = 0.9;
        excitation::AngularSpectrumBeam::Params bp;
        bp.wavelength = kLambda;
        bp.waist_radius = 0.4e-6;
        bp.region_radius = 0.6e-6;
        beam = std::make_shared<excitation::AngularSpectrumBeam>(bp);
    }
    [[nodiscard]] std::unique_ptr<Simulation> make(const SimulationConfig& c) const {
        return std::make_unique<Simulation>(tiny_open(false), term, beam, c);
    }

    SimulationConfig cfg;
    PmlTermination term;
    std::shared_ptr<excitation::AngularSpectrumBeam> beam;
};

}  // namespace

TEST_CASE("block_preconditioner: the PML route defaults (mlfmm: blocks, dense: recommendation)",
          "[block_preconditioner]") {
    // ADR 0011 decision 3 was met on the "mlfmm" PML route (benchmarks/results/preconditioner.md),
    // and the automatic block default is limited to that measured route (WP-PC1 review): with
    // the GMRES solver and no explicit choice "mlfmm" uses the blocks one level above the
    // leaves; the dense PML route keeps the recommendation (Jacobi for metals, none for
    // dielectrics), as do the direct solver, explicit settings and the box route. Tiny flat
    // surface (2N = 352).
    const TinyPml t;
    SimulationConfig dense_ag = t.cfg;
    dense_ag.object = material::silver_500nm();
    const auto dag = t.make(dense_ag);
    CHECK(dag->preconditioner_kind() == PreconditionerKind::Diagonal);
    CHECK(contains(dag->report(), "preconditioner: diagonal (Jacobi) (automatic)"));
    const auto dn = t.make(t.cfg);
    CHECK(dn->preconditioner_kind() == PreconditionerKind::None);
    CHECK(contains(dn->report(), "preconditioner: none (automatic)"));
    for (const material::Material& object :
         {material::silver_500nm(), material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)}}) {
        SimulationConfig fast = t.cfg;
        fast.object = object;
        fast.compression = "mlfmm";
        const auto f = t.make(fast);
        CHECK(f->preconditioner_kind() == PreconditionerKind::BlockDiagonal);
        CHECK(f->config().block_levels_above_leaf == 1);
        CHECK(contains(f->report(),
                       "block-diagonal (near field) (automatic), blocks: one level "
                       "above the leaves"));
    }
    // Explicit settings win (diagonal_preconditioner kept for compatibility); the direct solver.
    SimulationConfig fj = t.cfg;
    fj.compression = "mlfmm";
    fj.diagonal_preconditioner = true;
    CHECK(t.make(fj)->preconditioner_kind() == PreconditionerKind::Diagonal);
    SimulationConfig fn = t.cfg;
    fn.compression = "mlfmm";
    fn.preconditioner = PreconditionerKind::None;
    const auto none = t.make(fn);
    CHECK(none->preconditioner_kind() == PreconditionerKind::None);
    CHECK(contains(none->report(), "preconditioner: none (explicit)"));
    SimulationConfig cd = t.cfg;
    cd.solver = SolverKind::Direct;
    const auto direct = t.make(cd);
    CHECK(direct->preconditioner_kind() == PreconditionerKind::None);  // the recommendation
    direct->solve();
    const VectorXc& x_ref = direct->solution().currents;

    // An explicit BlockDiagonal is allowed on the dense PML route (leaf blocks: on the two-level
    // default tree one level up would be the root, one dense LU).
    SimulationConfig db = t.cfg;
    db.preconditioner = PreconditionerKind::BlockDiagonal;
    db.block_levels_above_leaf = 0;
    const auto dense_block = t.make(db);
    const solver::GmresResult rb = dense_block->solve();
    REQUIRE(dense_block->block_preconditioner() != nullptr);
    INFO("dense PML, explicit leaf blocks: " << rb.iterations << " iterations, "
                                             << dense_block->block_preconditioner()->describe());
    CHECK(rb.converged);
    CHECK(dense_block->block_preconditioner()->num_blocks() > 1);
    CHECK((dense_block->solution().currents - x_ref).norm() <= 1e-7 * x_ref.norm());
    CHECK(contains(dense_block->report(), "block-diagonal (near field) (explicit)"));
}

TEST_CASE("block_preconditioner: the automatic block default warns on a coarse tree",
          "[block_preconditioner]") {
    // The tiny PML surface (2N = 352) gives the default octree two levels: one level above the
    // leaves is the root, so the automatic block preconditioner is one block = a dense LU and
    // GMRES converges in one iteration. Kept (no fallback), with an SBEM_WARN on the automatic
    // path only; an explicit BlockDiagonal does not warn.
    const TinyPml t;
    SimulationConfig fast = t.cfg;
    fast.compression = "mlfmm";
    SimulationConfig cd = t.cfg;
    cd.solver = SolverKind::Direct;
    const auto direct = t.make(cd);
    direct->solve();
    const VectorXc& x_ref = direct->solution().currents;
    {
        LogCapture log(spdlog::level::warn);
        const auto sim = t.make(fast);
        REQUIRE(sim->preconditioner_kind() == PreconditionerKind::BlockDiagonal);
        CHECK(log.count(kCoarseBlockWarning) == 0);  // at assembly, not at construction
        const solver::GmresResult r = sim->solve();
        const auto* Z = dynamic_cast<const mlfmm::MlfmmOperator*>(sim->system_operator().get());
        REQUIRE(Z != nullptr);
        REQUIRE(sim->block_preconditioner() != nullptr);
        const solver::BlockDiagonalPreconditioner& P = *sim->block_preconditioner();
        INFO(Z->octree().levels() << " octree levels, " << P.describe() << ", " << r.iterations
                                  << " iterations");
        CHECK(Z->octree().levels() == 2);
        CHECK(P.num_blocks() == 1);
        CHECK(P.max_block_size() == sim->num_unknowns());
        CHECK(r.converged);
        CHECK(r.iterations <= 1);
        CHECK(log.count(kCoarseBlockWarning) == 1);
        // Every box pair of the two-level tree is near: the fast operator is the dense one.
        CHECK((sim->solution().currents - x_ref).norm() <= 1e-7 * x_ref.norm());
        sim->assemble();  // idempotent: built (and warned) once
        CHECK(log.count(kCoarseBlockWarning) == 1);
    }
    {
        LogCapture log(spdlog::level::warn);
        SimulationConfig fe = fast;
        fe.preconditioner = PreconditionerKind::BlockDiagonal;
        const auto sim = t.make(fe);
        sim->assemble();
        REQUIRE(sim->block_preconditioner() != nullptr);
        CHECK(2 * sim->block_preconditioner()->max_block_size() > sim->num_unknowns());
        CHECK(log.count(kCoarseBlockWarning) == 0);
    }
}

TEST_CASE("block_preconditioner: the constructors validate the octree parameters of the blocks",
          "[block_preconditioner]") {
    // The dense route builds the block partition from mlfmm::make_octree(config.mlfmm): with an
    // explicit BlockDiagonal invalid octree parameters throw in the constructor, not in
    // assemble(); the automatic block default exists only with compression "mlfmm", whose
    // parameters are always validated.
    const TinyPml t;
    for (const bool explicit_block : {false, true}) {
        SimulationConfig dense = t.cfg;
        dense.mlfmm.octree.max_levels = 0;
        if (explicit_block) {
            dense.preconditioner = PreconditionerKind::BlockDiagonal;
            CHECK_THROWS_WITH(t.make(dense), ContainsSubstring("MLFMM octree parameters"));
            CHECK_THROWS_AS(t.make(dense), std::invalid_argument);
        } else {
            CHECK_NOTHROW(t.make(dense));  // unused by the dense route without blocks
        }
        SimulationConfig fast = dense;
        fast.compression = "mlfmm";
        CHECK_THROWS_WITH(t.make(fast), ContainsSubstring("MLFMM octree parameters"));
    }
    SimulationConfig floor = t.cfg;
    floor.preconditioner = PreconditionerKind::BlockDiagonal;
    floor.mlfmm.octree.min_box_size_lambda = -1.0;
    CHECK_THROWS_WITH(t.make(floor), ContainsSubstring("MLFMM octree parameters"));
}
