/// Tests of the MLFMM near field (WP19a): op::assemble_sparse, op::SparseOperator and
/// mlfmm::near_pattern / assemble_near (ADR 0008 §1).
///
/// Every stored entry is compared with the DenseStrategy matrix (bitwise: same blocks, same
/// summation order) and the stored (row, col) set with the near pairs computed independently from
/// the leaf ijk (Chebyshev distance <= 1). Meshes: icosphere n = 1 (80 triangles) and a small
/// closed rough box (uniform box at the top-face spacing), at lambda = 1 um with the cheap WP7
/// kernel options (the entries are compared with the same options; accuracy is not the point),
/// and octrees of 3 to 4 levels (6 elements per leaf, no leaf-size floor). The kernel options are
/// the WP7 scheme at the lowest positive-interior degrees (far 1, near and singular 2): the
/// comparisons are exact for any options, and the sanitizer run must stay within ~5 s per case.
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/region_sparse_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "assembler_test_support.hpp"

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;
using formulation::Kind;

namespace {

constexpr Real kLambda = 1e-6;

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = 1;
    opt.quad_degree_near = 2;
    opt.quad_degree_sing = 2;
    return opt;
}

geometry::TriangleMesh rough_box() {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = 0.3e-6;
    p.rms_roughness = 20e-9;
    p.correlation_length = 200e-9;
    p.mesh_size = 75e-9;
    p.seed = 20261009;
    p.box_depth = 0.15e-6;
    p.box_mesh_size = 75e-9;
    return geometry::make_rough_surface_mesh(p);
}

material::Material lossless() {
    return {Complex(2.25, 0.0), Complex(1.0, 0.0)};
}

/// Mesh, space, octree and a Problem without excitation (omega from kLambda).
struct NearCase {
    explicit NearCase(geometry::TriangleMesh m)
        : mesh(std::move(m)), space(mesh), tree(space, kLambda, mlfmm::OctreeParams{6, 12, 0.0}) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.kernel_options = cheap_options();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
    }
    void set(const material::Material& object, Kind kind) {
        problem.object = object;
        form = formulation::make_formulation(kind);
        problem.formulation = form.get();
    }
    /// near(m, n) from the leaf ijk (independent of the octree's near lists).
    [[nodiscard]] std::vector<std::array<Index, 3>> leaf_ijk() const {
        std::vector<std::array<Index, 3>> ijk(sz(space.size()));
        for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
            const mlfmm::Box& box = tree.boxes()[sz(b)];
            for (Index q = box.first_element; q < box.first_element + box.num_elements; ++q)
                ijk[sz(tree.permutation()[sz(q)])] = box.ijk;
        }
        return ijk;
    }

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    mlfmm::Octree tree;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
};

MatrixXc dense(const op::Problem& p) {
    return assembler_test::assemble(p);
}

/// Pattern and entries of Z_near against the dense matrix and the leaf ijk.
void check_near(NearCase& c, const std::string& label) {
    const Index N = c.space.size();
    REQUIRE(c.tree.levels() >= 3);
    const auto ijk = c.leaf_ijk();
    const auto near = [&](Index m, Index n) {
        Index d = 0;
        for (std::size_t a = 0; a < 3; ++a)
            d = std::max(d, std::abs(ijk[sz(m)][a] - ijk[sz(n)][a]));
        return d <= 1;
    };
    Index near_pairs = 0;
    for (Index m = 0; m < N; ++m) {
        for (Index n = 0; n < N; ++n) near_pairs += near(m, n) ? 1 : 0;
    }
    REQUIRE(near_pairs < N * N);  // the pattern is genuinely sparse

    auto t0 = std::chrono::steady_clock::now();
    const MatrixXc Zd = dense(c.problem);
    const Real t_dense = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    const std::shared_ptr<op::SparseOperator> S = mlfmm::assemble_near(c.problem, c.tree);
    const Real t_near = seconds_since(t0);
    WARN(label << ": N = " << N << ", levels " << c.tree.levels() << ", " << S->describe()
               << ", near " << t_near << " s, dense " << t_dense << " s");
    REQUIRE(S->rows() == 2 * N);
    REQUIRE(S->cols() == 2 * N);
    CHECK(S->nonzeros() == 4 * near_pairs);

    // Per-block scale of the dense matrix (the blocks carry different units).
    std::array<Real, 4> scale{};
    for (std::size_t b = 0; b < 4; ++b) {
        scale[b] = Zd.block(Index(b / 2) * N, Index(b % 2) * N, N, N).cwiseAbs().maxCoeff();
    }
    const op::SparseOperator::Matrix& Z = S->matrix();
    Index outside = 0, unsorted = 0, differ = 0, beyond_tol = 0;
    for (Index r = 0; r < 2 * N; ++r) {
        for (Index k = Z.outerIndexPtr()[r]; k < Z.outerIndexPtr()[r + 1]; ++k) {
            const Index col = Z.innerIndexPtr()[k];
            if (k > Z.outerIndexPtr()[r] && col <= Z.innerIndexPtr()[k - 1])
                ++unsorted;
            if (!near(r % N, col % N))
                ++outside;
            const Complex v = Z.valuePtr()[k];
            const Complex d = Zd(r, col);
            differ += v != d ? 1 : 0;
            const std::size_t blk = 2 * sz(r / N) + sz(col / N);
            beyond_tol += std::abs(v - d) <= 1e-12 * scale[blk] ? 0 : 1;
        }
    }
    CHECK(outside == 0);
    CHECK(unsorted == 0);
    CHECK(beyond_tol == 0);  // the docs/05-style criterion of the brief
    CHECK(differ == 0);      // identical summation order: bitwise
}

void check_material(NearCase& c, const material::Material& object, const std::string& label) {
    for (const Kind kind : {Kind::PMCHWT, Kind::ICTF}) {
        c.set(object, kind);
        check_near(c, label + (kind == Kind::PMCHWT ? " PMCHWT" : " ICTF"));
    }
}

/// The pre-WP19a DenseStrategy::build, serial: greedy colouring, test triangles by colour,
/// source triangles ascending, the pair_blocks arithmetic. Every entry receives its
/// contributions in the same order, so the result must match the library bitwise.
MatrixXc reference_dense(const op::Problem& p) {
    const basis::RwgSpace& space = *p.space;
    const Index N = space.size();
    const Index F = space.mesh().num_triangles();
    const auto region = [&](const material::Material& m) {
        return kernels::RegionParams{m.wavenumber(p.omega), m.wave_impedance(p.omega), p.omega,
                                     constants::eps0 * m.eps_r, constants::mu0 * m.mu_r};
    };
    const std::array<kernels::RegionParams, 2> reg = {region(p.exterior), region(p.object)};
    const formulation::Weights w = p.formulation->weights(reg[0].eta, reg[1].eta);
    const std::array<Complex, 2> ce = {w.a1 / reg[0].eta, w.a2 / reg[1].eta};
    const std::array<Complex, 2> ch = {w.b1 * reg[0].eta, w.b2 * reg[1].eta};
    const std::array<Complex, 2> cm = {w.b1 / reg[0].eta, w.b2 / reg[1].eta};
    const std::array<Real, 2> jump = {-0.5, 0.5};
    std::vector<int> colour(sz(F), -1);
    for (Index t = 0; t < F; ++t) {
        const auto sup = space.support(t);
        std::array<bool, 4> used{};
        for (int a = 0; a < sup.count; ++a) {
            const Index n = sup.n[a];
            const Index o =
                space.plus_triangle(n) == t ? space.minus_triangle(n) : space.plus_triangle(n);
            if (colour[sz(o)] >= 0)
                used[static_cast<std::size_t>(colour[sz(o)])] = true;
        }
        int c = 0;
        while (used[static_cast<std::size_t>(c)]) ++c;
        colour[sz(t)] = c;
    }
    MatrixXc Z = MatrixXc::Zero(2 * N, 2 * N);
    Eigen::Matrix<Complex, 3, 3> L, K, I;
    for (int c = 0; c < 4; ++c) {
        for (Index t = 0; t < F; ++t) {
            const auto st = space.support(t);
            if (colour[sz(t)] != c || st.count == 0)
                continue;
            for (Index s = 0; s < F; ++s) {
                const auto ss = space.support(s);
                if (ss.count == 0)
                    continue;
                std::array<Complex, 9> jj{}, jm{}, mj{}, mm{};
                if (t == s)
                    kernels::jump_block(space, t, I);
                for (std::size_t i = 0; i < 2; ++i) {
                    if (ce[i] == 0.0 && ch[i] == 0.0 && cm[i] == 0.0)
                        continue;
                    kernels::element_blocks(space, t, s, reg[i], p.kernel_options, L, K);
                    for (Eigen::Index a = 0; a < 3; ++a) {
                        for (Eigen::Index b = 0; b < 3; ++b) {
                            const auto k = static_cast<std::size_t>(3 * a + b);
                            const Complex kk = t == s ? K(a, b) + jump[i] * I(a, b) : K(a, b);
                            jj[k] += ce[i] * L(a, b);
                            jm[k] -= ce[i] * kk;
                            mj[k] += ch[i] * kk;
                            mm[k] += cm[i] * L(a, b);
                        }
                    }
                }
                for (int b = 0; b < ss.count; ++b) {
                    for (int a = 0; a < st.count; ++a) {
                        const Index m = st.n[a];
                        const Index n = ss.n[b];
                        const auto k = static_cast<std::size_t>(3 * a + b);
                        Z(m, n) += jj[k];
                        Z(m, N + n) += jm[k];
                        Z(N + m, n) += mj[k];
                        Z(N + m, N + n) += mm[k];
                    }
                }
            }
        }
    }
    return Z;
}

}  // namespace

TEST_CASE("near_field: icosphere entries and pattern, Si", "[near_field]") {
    NearCase c(geometry::make_icosphere(0.5e-6, 1));
    check_material(c, material::silicon_500nm(), "icosphere Si");
    // Single-region system: the jump terms of coincident triangles do not cancel.
    assembler_test::SingleRegion single(2);
    c.problem.formulation = &single;
    check_near(c, "icosphere Si region 2");
}

TEST_CASE("near_field: icosphere entries and pattern, Ag", "[near_field]") {
    NearCase c(geometry::make_icosphere(0.5e-6, 1));
    check_material(c, material::silver_500nm(), "icosphere Ag");
}

TEST_CASE("near_field: icosphere entries and pattern, lossless", "[near_field]") {
    NearCase c(geometry::make_icosphere(0.5e-6, 1));
    check_material(c, lossless(), "icosphere n = 1.5");
}

TEST_CASE("near_field: rough box entries and pattern, Si and PMCHWT", "[near_field]") {
    NearCase c(rough_box());
    c.set(material::silicon_500nm(), Kind::PMCHWT);
    check_near(c, "rough box Si PMCHWT");
}

TEST_CASE("near_field: rough box entries and pattern, Ag and ICTF", "[near_field]") {
    NearCase c(rough_box());
    c.set(material::silver_500nm(), Kind::ICTF);
    check_near(c, "rough box Ag ICTF");
}

TEST_CASE("near_field: rough box entries and pattern, lossless and ICTF", "[near_field]") {
    NearCase c(rough_box());
    c.set(lossless(), Kind::ICTF);
    check_near(c, "rough box n = 1.5 ICTF");
}

TEST_CASE("near_field: apply equals the masked dense product, any thread count", "[near_field]") {
    NearCase c(geometry::make_icosphere(0.5e-6, 1));
    c.set(material::silicon_500nm(), Kind::PMCHWT);
    const Index N = c.space.size();
    const MatrixXc Zd = dense(c.problem);
    const auto S = mlfmm::assemble_near(c.problem, c.tree);
    const auto ijk = c.leaf_ijk();
    MatrixXc masked = Zd;
    for (Index r = 0; r < 2 * N; ++r) {
        for (Index col = 0; col < 2 * N; ++col) {
            Index d = 0;
            for (std::size_t a = 0; a < 3; ++a)
                d = std::max(d, std::abs(ijk[sz(r % N)][a] - ijk[sz(col % N)][a]));
            if (d > 1)
                masked(r, col) = 0.0;
        }
    }
    std::mt19937_64 rng(20261009);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    VectorXc x(2 * N);
    for (Index i = 0; i < 2 * N; ++i) x(i) = Complex(u(rng), u(rng));
    const VectorXc ref = masked * x;
    VectorXc y;
    S->apply(x, y);
    REQUIRE(y.size() == 2 * N);
    for (const Index h : {Index{0}, N}) {  // E and H rows have different units
        CHECK((y.segment(h, N) - ref.segment(h, N)).norm() <= 1e-13 * ref.segment(h, N).norm());
    }
    VectorXc alias = x;
    S->apply(alias, alias);
    CHECK((alias.array() == y.array()).all());

#ifdef SPECKLEBEM_HAVE_OPENMP
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto S1 = mlfmm::assemble_near(c.problem, c.tree);
    VectorXc y1;
    S1->apply(x, y1);
    omp_set_num_threads(std::max(threads, 4));
    VectorXc y4;
    S->apply(x, y4);
    omp_set_num_threads(threads);
    const auto& A = S->matrix();
    const auto& B = S1->matrix();
    REQUIRE(A.nonZeros() == B.nonZeros());
    CHECK(std::equal(A.valuePtr(), A.valuePtr() + A.nonZeros(), B.valuePtr()));
    CHECK(std::equal(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros(), B.innerIndexPtr()));
    CHECK((y1.array() == y.array()).all());
    CHECK((y4.array() == y.array()).all());
#endif

    // describe / memory_bytes.
    const auto nnz = static_cast<std::size_t>(S->nonzeros());
    CHECK(S->memory_bytes() == nnz * 24u + (2u * sz(N) + 1u) * 8u);
    CHECK(S->describe().rfind("sparse 240x240, nnz = " + std::to_string(nnz), 0) == 0);
}

TEST_CASE("near_field: DenseStrategy matrix unchanged by the WP19a refactoring", "[near_field]") {
    // Icosphere n = 1 with PMCHWT; the serial reference is slow under the sanitizers, so MCTF
    // and the single-region system (non-cancelling jump terms) use the icosahedron.
    NearCase c(geometry::make_icosphere(0.5e-6, 1));
    c.set(material::silver_500nm(), Kind::PMCHWT);
    CHECK((dense(c.problem).array() == reference_dense(c.problem).array()).all());
    NearCase c0(geometry::make_icosphere(0.5e-6, 0));
    c0.set(material::silver_500nm(), Kind::MCTF);
    CHECK((dense(c0.problem).array() == reference_dense(c0.problem).array()).all());
    assembler_test::SingleRegion single(1);
    c0.problem.formulation = &single;
    CHECK((dense(c0.problem).array() == reference_dense(c0.problem).array()).all());
}

TEST_CASE("near_field: input errors", "[near_field]") {
    NearCase c(geometry::make_icosphere(0.5e-6, 1));
    c.set(material::silicon_500nm(), Kind::PMCHWT);
    const op::Problem good = c.problem;

    // Octree of another space: different size, or same size but shifted geometry.
    const geometry::TriangleMesh coarse = geometry::make_icosphere(0.5e-6, 0);
    const basis::RwgSpace coarse_space(coarse);
    const mlfmm::Octree coarse_tree(coarse_space, kLambda, mlfmm::OctreeParams{6, 12, 0.0});
    CHECK_THROWS_AS(mlfmm::assemble_near(good, coarse_tree), std::invalid_argument);
    const geometry::TriangleMesh shifted = geometry::make_icosphere(0.5e-6, 1, Vec3(1e-6, 0, 0));
    const basis::RwgSpace shifted_space(shifted);
    const mlfmm::Octree shifted_tree(shifted_space, kLambda, mlfmm::OctreeParams{6, 12, 0.0});
    CHECK_THROWS_AS(mlfmm::near_pattern(shifted_tree, c.space), std::invalid_argument);
    CHECK_NOTHROW(mlfmm::near_pattern(c.tree, c.space));

    // Invalid problems.
    op::Problem bad = good;
    bad.space = nullptr;
    CHECK_THROWS_AS(mlfmm::assemble_near(bad, c.tree), std::invalid_argument);
    bad = good;
    bad.formulation = nullptr;
    CHECK_THROWS_AS(mlfmm::assemble_near(bad, c.tree), std::invalid_argument);
    bad = good;
    bad.omega = 0.0;
    CHECK_THROWS_AS(mlfmm::assemble_near(bad, c.tree), std::invalid_argument);

    // Inconsistent patterns.
    const op::BasisPattern pat = mlfmm::near_pattern(c.tree, c.space);
    const auto throws = [&](const op::BasisPattern& q) {
        CHECK_THROWS_AS(op::assemble_sparse(good, q), std::invalid_argument);
    };
    op::BasisPattern q = pat;
    q.group_of_row.pop_back();
    throws(q);
    q = pat;
    q.group_of_row[0] = static_cast<Index>(q.col_ptr.size());
    throws(q);
    q = pat;
    q.cols.back() = c.space.size();
    throws(q);
    q = pat;
    std::swap(q.cols[0], q.cols[1]);
    throws(q);
    q = pat;
    q.col_ptr.back() += 1;
    throws(q);
    q = pat;
    q.col_ptr.clear();
    throws(q);
    // col_ptr overshooting cols in an early group, decreasing later (review W1: must throw, not
    // read past cols).
    q = pat;
    if (q.col_ptr.size() >= 3) {
        q.col_ptr[1] = static_cast<Index>(q.cols.size()) + 100;
        throws(q);
    }

    // An empty pattern gives an empty operator.
    op::BasisPattern empty;
    empty.group_of_row.assign(sz(c.space.size()), 0);
    empty.col_ptr = {0, 0};
    const auto E = op::assemble_sparse(good, empty);
    CHECK(E->nonzeros() == 0);
    VectorXc y;
    E->apply(VectorXc::Ones(2 * c.space.size()), y);
    CHECK(y.isZero(0.0));
    CHECK_THROWS_AS(E->apply(VectorXc::Ones(3), y), std::invalid_argument);
}

namespace {

/// The region-i part of Z on the pattern, rebuilt from the stored (L, K) and the weights.
MatrixXc region_matrix(const op::RegionSparseOperator& R, Index n) {
    // Columns of the identity through apply(): exact for the block structure.
    MatrixXc Z = MatrixXc::Zero(2 * n, 2 * n);
    VectorXc y;
    for (Index c = 0; c < 2 * n; ++c) {
        R.apply(VectorXc::Unit(2 * n, c), y);  // (GCC 16 -Wnull-dereference with e(c) = 1)
        Z.col(c) = y;
    }
    return Z;
}

}  // namespace

TEST_CASE("near_field: region (L, K) storage reproduces the dense matrix", "[near_field]") {
    // WP21 review: the MLFMM exact far part stores L_i and K_i per basis pair and applies the
    // block weights; R1 + R2 on the full pattern (jump terms included) equal the dense Z.
    NearCase c(geometry::make_icosphere(0.5 * kLambda, 1));
    const Index n = c.space.size();
    for (const Kind kind : {Kind::PMCHWT, Kind::ICTF}) {
        c.set(material::silver_500nm(), kind);
        const MatrixXc D = dense(c.problem);
        std::vector<Index> row_ptr(sz(n) + 1), cols;
        for (Index m = 0; m < n; ++m) {
            for (Index k = 0; k < n; ++k) cols.push_back(k);
            row_ptr[sz(m) + 1] = static_cast<Index>(cols.size());
        }
        const auto R1 = op::assemble_region_sparse(c.problem, 0, row_ptr, cols);
        const auto R2 = op::assemble_region_sparse(c.problem, 1, row_ptr, cols);
        CHECK(R1->pairs() == n * n);
        CHECK(R1->rows() == 2 * n);
        CHECK(R1->memory_bytes() == sz(n * n) * 40 + (sz(n) + 1) * sizeof(Index));
        CHECK(op::RegionSparseOperator::kBytesPerPair == 40);
        MatrixXc S = region_matrix(*R1, n);  // (GCC 16 -Wnull-dereference with a + b)
        S += region_matrix(*R2, n);
        const Real err = (S - D).cwiseAbs().maxCoeff() / D.cwiseAbs().maxCoeff();
        INFO("formulation " << static_cast<int>(kind) << ": max entry difference " << err);
        CHECK(err <= 1e-12);
        CHECK(R1->describe().find("basis pairs (L, K)") != std::string::npos);
    }
    // A sparse pattern (every third pair): R1 + R2 equal the dense entries on it, zero elsewhere;
    // x and y may alias.
    c.set(lossless(), Kind::PMCHWT);
    const MatrixXc D = dense(c.problem);
    std::vector<Index> row_ptr(sz(n) + 1), cols;
    for (Index m = 0; m < n; ++m) {
        for (Index k = m % 3; k < n; k += 3) cols.push_back(k);
        row_ptr[sz(m) + 1] = static_cast<Index>(cols.size());
    }
    const auto R1 = op::assemble_region_sparse(c.problem, 0, row_ptr, cols);
    const auto R2 = op::assemble_region_sparse(c.problem, 1, row_ptr, cols);
    MatrixXc masked = MatrixXc::Zero(2 * n, 2 * n);
    for (Index m = 0; m < n; ++m) {
        for (Index k = row_ptr[sz(m)]; k < row_ptr[sz(m) + 1]; ++k) {
            const Index col = cols[sz(k)];
            for (const Index r : {m, n + m})
                for (const Index q : {col, n + col}) masked(r, q) = D(r, q);
        }
    }
    MatrixXc S = region_matrix(*R1, n);  // (GCC 16 -Wnull-dereference with a + b)
    S += region_matrix(*R2, n);
    CHECK((S - masked).cwiseAbs().maxCoeff() <= 1e-12 * D.cwiseAbs().maxCoeff());
    VectorXc x = VectorXc::LinSpaced(2 * n, -1.0, 2.0);
    VectorXc y;
    R1->apply(x, y);
    R1->apply(x, x);
    CHECK((x.array() == y.array()).all());
    CHECK_THROWS_AS(R1->apply(VectorXc::Ones(3), y), std::invalid_argument);
}

TEST_CASE("near_field: region (L, K) storage input errors", "[near_field]") {
    NearCase c(geometry::make_icosphere(0.5 * kLambda, 1));
    c.set(lossless(), Kind::PMCHWT);
    const Index n = c.space.size();
    std::vector<Index> row_ptr(sz(n) + 1, 0), cols;
    CHECK_THROWS_AS(op::assemble_region_sparse(c.problem, 2, row_ptr, cols), std::invalid_argument);
    CHECK_THROWS_AS(op::assemble_region_sparse(c.problem, -1, row_ptr, cols),
                    std::invalid_argument);
    CHECK_THROWS_AS(op::assemble_region_sparse(c.problem, 0, std::vector<Index>(sz(n), 0), cols),
                    std::invalid_argument);
    std::vector<Index> bad_ptr(sz(n) + 1, 2);  // row 0 has columns {1, 0}: not ascending
    bad_ptr.front() = 0;
    CHECK_THROWS_AS(op::assemble_region_sparse(c.problem, 0, bad_ptr, {1, 0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(op::assemble_region_sparse(c.problem, 0, bad_ptr, {0, n}),
                    std::invalid_argument);
    // An empty pattern gives a zero operator.
    const auto E = op::assemble_region_sparse(c.problem, 1, row_ptr, cols);
    CHECK(E->pairs() == 0);
    VectorXc y;
    E->apply(VectorXc::Ones(2 * n), y);
    CHECK(y.isZero(0.0));
    // Constructor checks.
    const Complex one(1.0, 0.0);
    CHECK_THROWS_AS(op::RegionSparseOperator(2, {0, 1}, {0}, {one, one}, one, one, one),
                    std::invalid_argument);  // row_ptr size
    CHECK_THROWS_AS(op::RegionSparseOperator(2, {0, 1, 1}, {0}, {one}, one, one, one),
                    std::invalid_argument);  // one value per pair
    CHECK_THROWS_AS(op::RegionSparseOperator(2, {0, 1, 0}, {0}, {one, one}, one, one, one),
                    std::invalid_argument);  // decreasing / back mismatch
    CHECK_THROWS_AS(op::RegionSparseOperator(2, {0, 1, 1}, {2}, {one, one}, one, one, one),
                    std::invalid_argument);  // column out of range
    CHECK_THROWS_AS(op::RegionSparseOperator(-1, {0}, {}, {}, one, one, one),
                    std::invalid_argument);
}
