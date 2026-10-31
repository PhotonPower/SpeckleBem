// Unit tests of the near-field Schur-complement preconditioner (WP-SC2, ADR 0012 decision 2):
// solver::SchurComplementPreconditioner (exactness with one J and one M block, the Schur
// complement's M blocks and apply() against dense references, bitwise thread-count
// independence, errors), make_schur_preconditioner on the small PML mesh of
// test_block_preconditioner.cpp, the ADR 0011 left-GMRES invariance, and the Simulation option.
// Random matrices use fixed seeds (std::mt19937_64); the BEM cases use cheap kernel options (the
// preconditioner is compared with matrices of the same options, so accuracy does not matter).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"
#include "specklebem/solver/schur_preconditioner.hpp"

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
#include <numeric>
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
using solver::BlockPartition;
using solver::SchurComplementPreconditioner;
using SparseMatrix = op::SparseOperator::Matrix;
using Sources = std::vector<const SparseMatrix*>;
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

bool bitwise_equal(const MatrixXc& a, const MatrixXc& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
}

Real rel_diff(const VectorXc& a, const VectorXc& b) {
    return (a - b).norm() / b.norm();
}

/// Random sparse 2N x 2N system [A B; C D]: per row `per_row` off-diagonal entries at random
/// columns, normal values, scaled by `coupling` in B and C (columns in the other half); every
/// M row also stores all of D's columns when dense_d; diagonal = 1 + the row's absolute sum
/// (diagonally dominant, so A, its blocks and the Schur complements are well conditioned).
SparseMatrix random_system(Index N, Index per_row, std::uint64_t seed, Real coupling = 0.3,
                           bool dense_d = false) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    std::uniform_int_distribution<Index> col(0, 2 * N - 1);
    std::vector<Triplet> t;
    for (Index i = 0; i < 2 * N; ++i) {
        std::set<Index> cols;
        while (static_cast<Index>(cols.size()) < per_row) {
            const Index c = col(rng);
            if (c != i)
                cols.insert(c);
        }
        if (dense_d && i >= N) {
            for (Index c = N; c < 2 * N; ++c)
                if (c != i)
                    cols.insert(c);
        }
        Real sum = 0;
        for (const Index c : cols) {
            const Real scale = (c < N) != (i < N) ? coupling : 1.0;
            const Complex v = scale * Complex(nd(rng), nd(rng));
            sum += std::abs(v);
            t.emplace_back(i, c, v);
        }
        t.emplace_back(i, i, Complex(1.0 + sum, 0.5));
    }
    SparseMatrix A(2 * N, 2 * N);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

/// Random partition of [lo, lo + n) into `blocks` blocks of random sizes, ids 100 + k.
BlockPartition random_partition(Index lo, Index n, Index blocks, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Index> perm(sz(n));
    std::iota(perm.begin(), perm.end(), lo);
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

/// One block holding [lo, lo + n).
BlockPartition whole(Index lo, Index n) {
    BlockPartition p;
    p.blocks.emplace_back(sz(n));
    std::iota(p.blocks.front().begin(), p.blocks.front().end(), lo);
    return p;
}

/// M(I, J) of index lists (global indices, minus `offset`).
MatrixXc sub(const MatrixXc& M, const std::vector<Index>& I, const std::vector<Index>& J,
             Index offset = 0) {
    MatrixXc S(static_cast<Index>(I.size()), static_cast<Index>(J.size()));
    for (std::size_t b = 0; b < J.size(); ++b)
        for (std::size_t a = 0; a < I.size(); ++a)
            S(static_cast<Index>(a), static_cast<Index>(b)) = M(I[a] - offset, J[b] - offset);
    return S;
}

/// Dense block-diagonal matrix of M's blocks (indices minus `offset`).
MatrixXc block_diagonal(const MatrixXc& M, const BlockPartition& p, Index offset = 0) {
    MatrixXc out = MatrixXc::Zero(M.rows(), M.cols());
    for (const std::vector<Index>& b : p.blocks)
        for (const Index i : b)
            for (const Index j : b) out(i - offset, j - offset) = M(i - offset, j - offset);
    return out;
}

/// The dense references of the preconditioner for Z_n = [A B; C D] and the partitions: Ã (block
/// diagonal of A), the Schur complement D - C Ã^{-1} B masked to the pattern of D, and its block
/// diagonal over the M blocks.
struct Reference {
    MatrixXc A_blk, B, C, S_masked, S_blk;
};
Reference reference(const SparseMatrix& Zn, Index N, const BlockPartition& jp,
                    const BlockPartition& mp) {
    const MatrixXc Z = MatrixXc(Zn);
    Reference r;
    r.A_blk = block_diagonal(Z.topLeftCorner(N, N), jp);
    r.B = Z.topRightCorner(N, N);
    r.C = Z.bottomLeftCorner(N, N);
    const MatrixXc D = Z.bottomRightCorner(N, N);
    const Eigen::PartialPivLU<MatrixXc> lu(r.A_blk);
    const MatrixXc S = D - r.C * lu.solve(r.B);
    r.S_masked = MatrixXc::Zero(N, N);
    for (Index i = N; i < 2 * N; ++i) {
        for (SparseMatrix::InnerIterator it(Zn, i); it; ++it) {
            if (it.col() >= N)
                r.S_masked(i - N, it.col() - N) = S(i - N, it.col() - N);
        }
    }
    r.S_blk = block_diagonal(r.S_masked, mp, N);
    return r;
}

/// Dense block forward / back substitution with the references: y1 = Ã^{-1} r1, y2 = S_b^{-1}
/// (r2 - C y1), x1 = y1 - Ã^{-1} B y2, x2 = y2.
VectorXc reference_apply(const Reference& ref, const VectorXc& r) {
    const Index N = ref.B.rows();
    const Eigen::PartialPivLU<MatrixXc> la(ref.A_blk);
    const Eigen::PartialPivLU<MatrixXc> ls(ref.S_blk);
    const VectorXc y1 = la.solve(r.head(N));
    const VectorXc y2 = ls.solve(r.tail(N) - ref.C * y1);
    VectorXc x(2 * N);
    x.head(N) = y1 - la.solve(ref.B * y2);
    x.tail(N) = y2;
    return x;
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

/// Exactly `levels` octree levels; optionally every far pair exact.
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

/// The tiny open surfaces of test_block_preconditioner.cpp: patch 0.2, taper 0.1 (flat) / 0.2
/// (rough, Gaussian patch, seed 29), gap 0.1, PML 0.1 um at h = 100 nm.
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

/// The octree boxes of a level split into J and M partitions (as make_schur_preconditioner).
std::pair<BlockPartition, BlockPartition> box_partitions(const mlfmm::Octree& tree, int level,
                                                         Index N) {
    BlockPartition jp, mp;
    const std::vector<std::vector<Index>> boxes = mlfmm::level_unknown_blocks(tree, level, &jp.ids);
    mp.ids = jp.ids;
    for (const std::vector<Index>& b : boxes) {
        std::vector<Index> j, m;
        for (const Index i : b) (i < N ? j : m).push_back(i);
        jp.blocks.push_back(std::move(j));
        mp.blocks.push_back(std::move(m));
    }
    return {std::move(jp), std::move(mp)};
}

}  // namespace

TEST_CASE("schur_preconditioner: one J and one M block give the exact inverse of Z_n",
          "[schur_preconditioner]") {
    // With Ã = A and a dense D (the pattern holds every entry of C A^{-1} B), M = Z_n: apply()
    // is the dense solve and GMRES on the near-only operator converges in one iteration.
    const Index N = 150;
    const SparseMatrix Zn = random_system(N, 6, 11, 0.3, true);
    const SchurComplementPreconditioner P(Zn, N, whole(0, N), whole(N, N));
    CHECK(P.name() == "schur");
    CHECK(P.size() == 2 * N);
    CHECK(P.half_size() == N);
    CHECK(P.num_j_blocks() == 1);
    CHECK(P.num_m_blocks() == 1);
    CHECK(P.coupling_pairs() == 1);
    CHECK(P.masked_entries_s() == 0);
    CHECK_FALSE(P.borrows_sources());
    const MatrixXc Zd = MatrixXc(Zn);
    const Eigen::PartialPivLU<MatrixXc> lu(Zd);
    for (const std::uint64_t seed : {12U, 13U}) {
        const VectorXc r = random_vector(2 * N, seed);
        VectorXc z;
        P.apply(r, z);
        const Real d = rel_diff(z, lu.solve(r));
        INFO("seed " << seed << ": relative difference to the dense solve " << d);
        CHECK(d <= 1e-12);
    }
    const op::SparseOperator near_only{SparseMatrix(Zn)};
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.verbose = false;
    const VectorXc b = random_vector(2 * N, 14);
    const solver::GmresResult res = solver::gmres(near_only, b, P, g);
    INFO("iterations " << res.iterations << ", true residual " << res.true_relative_residual);
    CHECK(res.converged);
    CHECK(res.iterations == 1);

    // The PML route: the tiny rough surface with a 2-level tree (every pair near, Z_n = Z, no
    // far part) and the root as the block level: one J and one M block, Z_n dense.
    PmlSetup c(true);
    const mlfmm::MlfmmOperator Z(c.problem, tree_params(2));
    REQUIRE(Z.ring_operator() != nullptr);
    const auto S = make_schur_preconditioner(Z, c.problem, 1);
    REQUIRE(S != nullptr);
    CHECK(S->borrows_sources());
    CHECK(S->num_j_blocks() == 1);
    const SparseMatrix Zs = near_field_matrix(Z);
    const Index n2 = Z.rows();
    INFO(S->describe() << ", nnz(Z_n) " << Zs.nonZeros() << " of " << n2 * n2);
    CHECK(Zs.nonZeros() == n2 * n2);
    const Eigen::PartialPivLU<MatrixXc> lz{MatrixXc(Zs)};
    const VectorXc r = random_vector(n2, 15);
    VectorXc z;
    S->apply(r, z);
    const Real d = rel_diff(z, lz.solve(r));
    INFO("PML: relative difference to the dense solve " << d);
    // Measured: 2.0e-15 (win-release; the random system above 4.1e-16).
    CHECK(d <= 1e-12);
}

TEST_CASE(
    "schur_preconditioner: the Schur complement's M blocks and apply() match the dense "
    "references",
    "[schur_preconditioner]") {
    // Random sparse block system, N = 200, J blocks and M blocks from two different random
    // partitions (5 and 4 blocks): every M block of S̃ (reconstructed from its LU) equals the
    // dense D - C Ã^{-1} B masked to the pattern of D, and apply() equals the dense block
    // forward / back substitution with the same approximations.
    const Index N = 200;
    const SparseMatrix Zn = random_system(N, 8, 21);
    const BlockPartition jp = random_partition(0, N, 5, 22);
    const BlockPartition mp = random_partition(N, N, 4, 23);
    const SchurComplementPreconditioner P(Zn, N, jp, mp);
    const Reference ref = reference(Zn, N, jp, mp);
    INFO(P.describe());
    CHECK(P.num_j_blocks() == 5);
    CHECK(P.num_m_blocks() == 4);
    CHECK(P.min_j_block_size() >= 1);
    CHECK(P.max_m_block_size() <= N);
    CHECK(P.j_partition().ids == jp.ids);
    CHECK(P.m_partition().blocks == mp.blocks);
    // The sparse blocks exercise the zero / mask rules.
    CHECK(P.missing_entries_a() > 0);
    CHECK(P.masked_entries_s() > 0);
    Index nnz_b = 0, nnz_c = 0;
    for (Index i = 0; i < 2 * N; ++i) {
        for (SparseMatrix::InnerIterator it(Zn, i); it; ++it) {
            nnz_b += i < N && it.col() >= N ? 1 : 0;
            nnz_c += i >= N && it.col() < N ? 1 : 0;
        }
    }
    CHECK(P.nonzeros_b() == nnz_b);
    CHECK(P.nonzeros_c() == nnz_c);
    CHECK(P.coupling_pairs() > 0);
    CHECK(P.coupling_pairs() <= 20);
    CHECK(P.formation_macs() > 0.0);
    CHECK(P.memory_bytes() >
          static_cast<std::size_t>(P.j_block_entries() + P.m_block_entries()) * sizeof(Complex));
    CHECK(P.setup_seconds() >= P.a_factorization_seconds());
    CHECK_THAT(P.describe(), ContainsSubstring("Schur complement: J blocks 5"));
    CHECK_THAT(P.describe(), ContainsSubstring("outside the pattern of D (zero)"));
    Real worst = 0;
    for (Index b = 0; b < P.num_m_blocks(); ++b) {
        const std::vector<Index>& idx = mp.blocks[sz(b)];
        const MatrixXc expect = sub(ref.S_masked, idx, idx, N);
        const MatrixXc got = P.m_block_lu(b).reconstructedMatrix();
        worst =
            std::max(worst, (got - expect).cwiseAbs().maxCoeff() / expect.cwiseAbs().maxCoeff());
    }
    // Ã's blocks: reconstructed J blocks equal A(I_k, I_k) (the input entries, zeros where the
    // matrix stores nothing).
    Real worst_a = 0;
    for (Index k = 0; k < P.num_j_blocks(); ++k) {
        const std::vector<Index>& idx = jp.blocks[sz(k)];
        const MatrixXc expect = sub(ref.A_blk, idx, idx);
        const MatrixXc got = P.j_block_lu(k).reconstructedMatrix();
        worst_a =
            std::max(worst_a, (got - expect).cwiseAbs().maxCoeff() / expect.cwiseAbs().maxCoeff());
    }
    INFO("largest relative entry difference: Schur M blocks " << worst << ", A blocks " << worst_a);
    // Measured: 1.6e-16 / 1.3e-16 (win-release; apply() below 1.8-2.0e-16).
    CHECK(worst <= 1e-12);
    CHECK(worst_a <= 1e-12);
    for (const std::uint64_t seed : {24U, 25U, 26U}) {
        const VectorXc r = random_vector(2 * N, seed);
        VectorXc z;
        P.apply(r, z);
        const Real d = rel_diff(z, reference_apply(ref, r));
        INFO("seed " << seed << ": apply vs the dense substitution " << d);
        CHECK(d <= 1e-12);
        VectorXc w = r;
        P.apply(w, w);  // aliasing
        CHECK(bitwise_equal(w, z));
    }
    CHECK_THROWS_AS(P.j_block_lu(5), std::out_of_range);
    CHECK_THROWS_AS(P.m_block_lu(-1), std::out_of_range);

    // The borrowing constructor on two disjoint sources summing to Z_n (the PML-route layout):
    // the same blocks (bitwise: every entry is stored once), apply() equal to rounding (the
    // sparse products sum in another order).
    SparseMatrix lower = Zn;
    SparseMatrix upper = Zn;
    lower.prune([](Index row, Index col, const Complex&) { return col <= row; });
    upper.prune([](Index row, Index col, const Complex&) { return col > row; });
    lower.makeCompressed();
    upper.makeCompressed();
    REQUIRE(lower.nonZeros() + upper.nonZeros() == Zn.nonZeros());
    const SchurComplementPreconditioner Q(Sources{&lower, &upper}, N, jp, mp);
    CHECK(Q.borrows_sources());
    CHECK(Q.nonzeros_b() == nnz_b);
    CHECK(Q.memory_bytes() < P.memory_bytes());  // B and C not counted
    for (Index b = 0; b < P.num_m_blocks(); ++b)
        CHECK(bitwise_equal(Q.m_block_lu(b).matrixLU(), P.m_block_lu(b).matrixLU()));
    const VectorXc r = random_vector(2 * N, 27);
    VectorXc zp, zq;
    P.apply(r, zp);
    Q.apply(r, zq);
    CHECK(rel_diff(zq, zp) <= 1e-13);
}

TEST_CASE("schur_preconditioner: setup and apply are bitwise independent of the thread count",
          "[schur_preconditioner]") {
#ifndef SPECKLEBEM_HAVE_OPENMP
    SKIP("built without OpenMP");
#else
    // Blocks of ~100-250 unknowns: the blocked LU, the LU solves and the products of the
    // formation call the BLAS.
    const Index N = 700;
    const SparseMatrix Zn = random_system(N, 40, 31);
    const BlockPartition jp = random_partition(0, N, 4, 32);
    const BlockPartition mp = random_partition(N, N, 5, 33);
    const VectorXc r = random_vector(2 * N, 34);
    const int max_threads = omp_get_max_threads();
    const int many = std::max(4, max_threads);
    omp_set_num_threads(1);
    const SchurComplementPreconditioner P1(Zn, N, jp, mp);
    VectorXc z11, z1m, zm1, zmm;
    P1.apply(r, z11);
    omp_set_num_threads(many);
    const SchurComplementPreconditioner Pm(Zn, N, jp, mp);
    P1.apply(r, z1m);
    Pm.apply(r, zmm);
    omp_set_num_threads(1);
    Pm.apply(r, zm1);
    omp_set_num_threads(max_threads);
    INFO("threads 1 and " << many << ": " << P1.describe());
    CHECK(bitwise_equal(z11, z1m));
    CHECK(bitwise_equal(z11, zmm));
    CHECK(bitwise_equal(z11, zm1));
    for (Index k = 0; k < P1.num_j_blocks(); ++k)
        CHECK(bitwise_equal(P1.j_block_lu(k).matrixLU(), Pm.j_block_lu(k).matrixLU()));
    for (Index b = 0; b < P1.num_m_blocks(); ++b)
        CHECK(bitwise_equal(P1.m_block_lu(b).matrixLU(), Pm.m_block_lu(b).matrixLU()));
#endif
}

TEST_CASE("schur_preconditioner: errors", "[schur_preconditioner]") {
    const Index N = 20;
    const SparseMatrix Zn = random_system(N, 4, 41);
    const BlockPartition jp = random_partition(0, N, 3, 42);
    const BlockPartition mp = random_partition(N, N, 3, 43);
    const auto invalid = [&](BlockPartition j, BlockPartition m, const std::string& what) {
        CHECK_THROWS_WITH(SchurComplementPreconditioner(Zn, N, std::move(j), std::move(m)),
                          ContainsSubstring(what));
    };
    {
        BlockPartition j = jp;  // a missing J index (from a block of more than one)
        std::max_element(j.blocks.begin(), j.blocks.end(), [](const auto& a, const auto& b) {
            return a.size() < b.size();
        })->pop_back();
        invalid(j, mp, "the blocks hold 19 of the 20 unknowns");
        BlockPartition d = jp;  // a duplicate
        d.blocks[1].push_back(jp.blocks[0][0]);
        invalid(d, mp, "belongs to more than one block");
        BlockPartition x = jp;  // an M index in a J block
        x.blocks[2].push_back(N + 3);
        invalid(x, mp, "an M unknown");
        BlockPartition y = mp;  // a J index in an M block
        y.blocks[0].push_back(3);
        invalid(jp, y, "a J unknown");
        BlockPartition o = mp;  // outside [0, 2N)
        o.blocks[0].push_back(2 * N);
        invalid(jp, o, "outside [0, 40)");
        BlockPartition e = jp;  // an empty block
        e.blocks.emplace_back();
        e.ids.push_back(7);
        invalid(e, mp, "block 7 is empty");
        BlockPartition w = jp;  // ids of the wrong size
        w.ids.pop_back();
        invalid(w, mp, "2 block ids for 3 blocks");
        invalid(BlockPartition{}, mp, "no block");
    }
    // Sizes.
    CHECK_THROWS_WITH(SchurComplementPreconditioner(Zn, N + 1, jp, mp),
                      ContainsSubstring("expected 2N x 2N"));
    CHECK_THROWS_AS(SchurComplementPreconditioner(SparseMatrix(2 * N, 2 * N + 1), N, jp, mp),
                    std::invalid_argument);
    CHECK_THROWS_AS(SchurComplementPreconditioner(Zn, 0, jp, mp), std::invalid_argument);
    // Sources.
    CHECK_THROWS_WITH(SchurComplementPreconditioner(Sources{}, N, jp, mp),
                      ContainsSubstring("no source"));
    CHECK_THROWS_WITH(SchurComplementPreconditioner(Sources{nullptr}, N, jp, mp),
                      ContainsSubstring("null source"));
    SparseMatrix unc = Zn;
    unc.uncompress();
    CHECK_THROWS_WITH(SchurComplementPreconditioner(Sources{&unc}, N, jp, mp),
                      ContainsSubstring("not compressed"));
    const SparseMatrix small = random_system(N - 1, 4, 44);
    CHECK_THROWS_WITH(SchurComplementPreconditioner(Sources{&Zn, &small}, N, jp, mp),
                      ContainsSubstring("expected 2N x 2N"));
    // min_rcond.
    for (const Real bad : {-1.0, std::numeric_limits<Real>::quiet_NaN()}) {
        CHECK_THROWS_WITH(SchurComplementPreconditioner(Zn, N, jp, mp, solver::SchurParams{bad}),
                          ContainsSubstring("min_rcond"));
    }
    // apply() size.
    const SchurComplementPreconditioner P(Zn, N, jp, mp);
    VectorXc z;
    CHECK_THROWS_AS(P.apply(VectorXc::Ones(2 * N - 1), z), std::invalid_argument);
    // An uncompressed input of the owning constructor is accepted.
    const SchurComplementPreconditioner U(unc, N, jp, mp);
    const VectorXc r = random_vector(2 * N, 45);
    VectorXc z1, z2;
    P.apply(r, z1);
    U.apply(r, z2);
    CHECK(bitwise_equal(z1, z2));

    // Singular blocks: a J block of A with a zero row (2 x 2 system: A = [1 0; 0 0]), and an M
    // block of S̃ that vanishes (A = B = C = D = I, N = 1: S̃ = D - C A^{-1} B = 0).
    const Complex one(1.0, 0.0);
    {
        std::vector<Triplet> t = {{0, 0, one}, {2, 2, one}, {3, 3, one}, {1, 2, one}};
        SparseMatrix S(4, 4);
        S.setFromTriplets(t.begin(), t.end());
        const auto f = [&] {
            return SchurComplementPreconditioner(S, 2, whole(0, 2), whole(2, 2));
        };
        CHECK_THROWS_WITH(f(), ContainsSubstring("J block 0 (2 unknowns) is singular"));
        CHECK_THROWS_AS(f(), std::runtime_error);
    }
    {
        std::vector<Triplet> t = {{0, 0, one}, {0, 1, one}, {1, 0, one}, {1, 1, one}};
        SparseMatrix S(2, 2);
        S.setFromTriplets(t.begin(), t.end());
        const auto f = [&] {
            return SchurComplementPreconditioner(S, 1, whole(0, 1), whole(1, 1));
        };
        CHECK_THROWS_WITH(f(), ContainsSubstring("Schur-complement M block 0 (1 unknowns) is "
                                                 "singular"));
        CHECK_THROWS_AS(f(), std::runtime_error);
    }
    // Non-finite entries.
    {
        std::vector<Triplet> t = {{0, 0, one},
                                  {1, 1, Complex(std::numeric_limits<Real>::infinity(), 0.0)}};
        SparseMatrix S(2, 2);
        S.setFromTriplets(t.begin(), t.end());
        CHECK_THROWS_WITH(SchurComplementPreconditioner(S, 1, whole(0, 1), whole(1, 1)),
                          ContainsSubstring("non-finite"));
    }
}

TEST_CASE("schur_preconditioner: PMCHWT, ICTF and MCTF give the same preconditioned system",
          "[schur_preconditioner]") {
    // The ADR 0011 invariance: PMCHWT, ICTF and MCTF scale the J-test rows by one constant and
    // the M-test rows by another (region-independent), i.e. A, B by alpha and C, D by beta, so
    // Ã scales by alpha, S̃ = D - C Ã^{-1} B by beta and M by the same row scaling: M^{-1} Z is
    // the same matrix up to rounding and left GMRES runs the same iterates. Ag sphere R = 0.5 um,
    // icosphere n = 2 at lambda = 1 um (2N = 960), a forced 3-level tree (far pairs exact), the
    // leaf boxes as the J and M blocks.
    const Real lambda = 1e-6;
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 2);
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.max_iter = 500;
    g.verbose = false;
    std::vector<solver::GmresResult> res;
    std::vector<MatrixXc> pz;
    std::string desc;
    Index n2 = 0;
#ifdef NDEBUG
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF, Kind::MCTF};
#else
    // Sanitizer build: one pair (as the block and the near-field LU tests).
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF};
#endif
    for (const Kind kind : kinds) {
        SphereSetup s(mesh, kind, material::silver_500nm(), lambda);
        // On the stack: a shared_ptr<PlaneWave> next to angular_spectrum_beam.hpp trips a GCC 16
        // -Warray-bounds false positive in the inlined destructor.
        const excitation::PlaneWave wave(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0));
        s.problem.excitation = &wave;
        s.problem.omega = wave.omega();
        const VectorXc b = op::assemble_rhs(s.problem);
        const mlfmm::MlfmmOperator Z(s.problem, tree_params(3, true));
        REQUIRE(Z.octree().levels() == 3);
        n2 = Z.rows();
        const auto P = make_schur_preconditioner(Z, s.problem, 0);
        desc = P->describe();
        res.push_back(solver::gmres(Z, b, *P, g));
        REQUIRE(res.back().converged);
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
    INFO("iterations (PMCHWT, ICTF[, MCTF]) " << its << "; " << desc);
    CHECK(res[0].iterations > 1);
    for (std::size_t f = 1; f < res.size(); ++f) {
        const Real dpz = (pz[f] - pz[0]).norm() / pz[0].norm();
        INFO("formulation " << f << ": |M^-1 Z - M0^-1 Z0| / |M0^-1 Z0| = " << dpz);
        // Measured 1.7e-15 (win-release, ICTF and MCTF vs PMCHWT, all columns; 235 iterations for
        // all three): the block LUs with partial pivoting of alpha A and beta S̃ pivot as those of A
        // and S̃, so only the rounding of the scaled entries differs. The bound leaves three
        // orders of magnitude for other BLAS kernels and the sanitizer build.
        CHECK(dpz <= 1e-12);
        CHECK(std::abs(res[f].iterations - res[0].iterations) <= 1);
        const std::size_t m =
            std::min(res[f].residual_history.size(), res[0].residual_history.size());
        // Iterates equal up to rounding: for the first 40 iterations to 1e-9 (measured: <= 1e-15
        // up to iteration 30, 1.5e-13 at 40; afterwards the rounding of the Krylov process grows
        // by ~10x per 3-4 iterations, 2e-11 at 50 and 9e-10 at 60, as for the block and Jacobi
        // preconditioners); afterwards the counts agree to one iteration and the solutions to the
        // tolerance.
        for (std::size_t k = 0; k < std::min<std::size_t>(m, 40); ++k) {
            CHECK(std::abs(res[f].residual_history[k] - res[0].residual_history[k]) <=
                  1e-9 * res[0].residual_history[k]);
        }
        CHECK((res[f].x - res[0].x).norm() <= 1e-8 * res[0].x.norm());
    }
}

TEST_CASE("schur_preconditioner: make_schur_preconditioner on the PML route (Z_rr near + ring)",
          "[schur_preconditioner]") {
    // The tiny rough PML surface with a forced 3-level tree: the factory borrows Z_rr's near part
    // and the near ring, its J and M blocks are the J and M unknowns of the boxes one level above
    // the leaves, and it equals the owning preconditioner of the summed Z_n on the same
    // partition.
    PmlSetup c(true);
    const mlfmm::MlfmmOperator Z(c.problem, tree_params(3));
    REQUIRE(Z.ring_operator() != nullptr);
    const Index N = c.space.size();
    for (const int up : {0, 1}) {
        const auto P = make_schur_preconditioner(Z, c.problem, up);
        REQUIRE(P != nullptr);
        INFO("levels above the leaf " << up << ": " << P->describe());
        CHECK(P->borrows_sources());
        CHECK(P->size() == 2 * N);
        const int level = Z.octree().leaf_level() - up;
        CHECK(P->j_partition().ids == Z.octree().boxes_at_level(level));
        CHECK(P->num_m_blocks() == P->num_j_blocks());
        CHECK(P->num_j_blocks() > 1);
        CHECK(P->missing_entries_a() == 0);  // levels 0 and 1: every same-box pair is near
        CHECK(P->masked_entries_s() == 0);
        auto [jp, mp] = box_partitions(Z.octree(), level, N);
        const SchurComplementPreconditioner Q(near_field_matrix(Z), N, std::move(jp),
                                              std::move(mp));
        for (Index b = 0; b < P->num_m_blocks(); ++b) {
            CHECK(bitwise_equal(P->j_block_lu(b).matrixLU(), Q.j_block_lu(b).matrixLU()));
            CHECK(bitwise_equal(P->m_block_lu(b).matrixLU(), Q.m_block_lu(b).matrixLU()));
        }
        const VectorXc r = random_vector(2 * N, 51);
        VectorXc zp, zq;
        P->apply(r, zp);
        Q.apply(r, zq);
        CHECK(rel_diff(zp, zq) <= 1e-12);
    }
    CHECK_THROWS_AS(make_schur_preconditioner(Z, c.problem, 3), std::invalid_argument);
    CHECK_THROWS_AS(make_schur_preconditioner(Z, c.problem, -1), std::invalid_argument);
    const MatrixXc I = MatrixXc::Identity(Z.rows(), Z.cols());
    const op::DenseOperator D(I);
    CHECK_THROWS_WITH(make_schur_preconditioner(D, c.problem, 1),
                      ContainsSubstring("not an MLFMM operator"));
    const MatrixXc I4 = MatrixXc::Identity(4, 4);
    const op::DenseOperator wrong(I4);
    CHECK_THROWS_WITH(make_schur_preconditioner(wrong, c.problem, 1),
                      ContainsSubstring("not 2N x 2N"));
}

namespace {

/// Tiny flat PML case of test_block_preconditioner.cpp: n = 1.5 object, cheap kernels, GMRES
/// 1e-9, compression "mlfmm" (the default tree has two levels).
struct TinyPml {
    TinyPml() {
        cfg.wavelength = kLambda;
        cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
        cfg.kernels = cheap_options();
        cfg.gmres.tolerance = 1e-9;
        cfg.gmres.verbose = false;
        cfg.compression = "mlfmm";
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

bool contains(const std::string& s, const std::string& what) {
    return s.find(what) != std::string::npos;
}

}  // namespace

TEST_CASE("schur_preconditioner: Simulation option, PML solve reproduces the block solve",
          "[schur_preconditioner]") {
    // Tiny flat PML surface (2N = 352), compression "mlfmm", leaf blocks (on the two-level tree
    // one level up is the root): the Schur-preconditioned solve reproduces the block-
    // preconditioned one (currents to the GMRES tolerance, R = P_refl / P_inc to 1e-8). Measured
    // (win-release): 46 iterations for both, R = 0.0350102 for both.
    const TinyPml t;
    SimulationConfig cb = t.cfg;
    cb.preconditioner = PreconditionerKind::BlockDiagonal;
    cb.block_levels_above_leaf = 0;
    const auto blk = t.make(cb);
    const solver::GmresResult rb = blk->solve();
    REQUIRE(rb.converged);
    SimulationConfig cs = cb;
    cs.preconditioner = PreconditionerKind::Schur;
    const auto sch = t.make(cs);
    CHECK(sch->preconditioner_kind() == PreconditionerKind::Schur);
    CHECK(sch->schur_preconditioner() == nullptr);
    CHECK_FALSE(sch->diagonal_preconditioner());
    const solver::GmresResult rs = sch->solve();
    REQUIRE(sch->schur_preconditioner() != nullptr);
    CHECK(sch->block_preconditioner() == nullptr);
    const std::string rep = sch->report();
    INFO("block " << rb.iterations << " / Schur " << rs.iterations << " iterations\n" << rep);
    CHECK(rs.converged);
    CHECK(sch->schur_preconditioner()->num_j_blocks() > 1);
    const VectorXc& xb = blk->solution().currents;
    CHECK((sch->solution().currents - xb).norm() <= 1e-7 * xb.norm());
    const Real Rb = blk->reflected_power(18, 36) / blk->incident_power();
    const Real Rs = sch->reflected_power(18, 36) / sch->incident_power();
    INFO("R block " << Rb << ", Schur " << Rs);
    CHECK(std::abs(Rs - Rb) <= 1e-8);
    CHECK(contains(rep,
                   "preconditioner: Schur complement (near field) (explicit), blocks: "
                   "octree leaves"));
    CHECK(contains(rep, "preconditioner setup:"));
    CHECK(contains(rep, "Schur complement: J blocks"));
    CHECK(contains(rep, "Schur-complement preconditioner"));
    const solver::SchurComplementPreconditioner* p = sch->schur_preconditioner();
    sch->assemble();  // idempotent
    CHECK(sch->schur_preconditioner() == p);

    // Validation: explicit only, compression "mlfmm", consistent with diagonal_preconditioner.
    SimulationConfig dense = cs;
    dense.compression = "dense";
    CHECK_THROWS_WITH(t.make(dense), ContainsSubstring("needs compression \"mlfmm\""));
    CHECK_THROWS_AS(t.make(dense), std::invalid_argument);
    for (const bool d : {true, false}) {
        SimulationConfig c = cs;
        c.diagonal_preconditioner = d;
        CHECK_THROWS_WITH(t.make(c), ContainsSubstring("contradicts diagonal_preconditioner"));
    }
    for (const int bad : {-1, 3}) {
        SimulationConfig c = cs;
        c.block_levels_above_leaf = bad;
        CHECK_THROWS_WITH(t.make(c), ContainsSubstring("block_levels_above_leaf"));
    }
    // Never automatic: the "mlfmm" PML default stays the block preconditioner.
    SimulationConfig a = t.cfg;
    CHECK(t.make(a)->preconditioner_kind() == PreconditionerKind::BlockDiagonal);
    CHECK(std::string(to_string(PreconditionerKind::Schur)) == "schur");
}
