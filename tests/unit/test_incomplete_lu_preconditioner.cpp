// Unit tests of the incomplete LU (ILUT) of the near-field blocks A and D (WP-SC3, ADR 0012
// decision 6): solver::IncompleteLuPreconditioner (exactness without dropping, monotone error
// in the drop tolerance, the AMD ordering convention, bitwise thread-count independence, errors
// incl. the pivot guard and the budget), make_incomplete_lu_preconditioner on the small PML mesh
// of test_block_preconditioner.cpp, the ADR 0011 left-GMRES invariance, the Simulation option,
// and the zero-pivot guard of BlockDiagonalPreconditioner / the Schur blocks (ADR 0012 decision
// 8). Random matrices use fixed seeds (std::mt19937_64); the BEM cases use cheap kernel options
// (the preconditioner is compared with matrices of the same options, so accuracy does not
// matter).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/incomplete_lu_preconditioner.hpp"
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
#include <cstring>
#include <exception>
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
using solver::IlutOrdering;
using solver::IlutParams;
using solver::IncompleteLuPreconditioner;
using SparseMatrix = op::SparseOperator::Matrix;
using Sources = std::vector<const SparseMatrix*>;
using Triplet = Eigen::Triplet<Complex, Index>;

namespace {

constexpr Real kLambda = 500e-9;

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    VectorXc v(n);
    for (Index i = 0; i < n; ++i) v(i) = Complex(nd(rng), nd(rng));
    return v;
}

bool bitwise_equal(const VectorXc& a, const VectorXc& b) {
    return a.size() == b.size() && (a.array() == b.array()).all();
}

/// Random sparse N x N block: per row `per_row` off-diagonal entries at random columns, normal
/// complex values; diagonal = dominance x the row's absolute sum + 1 + 0.5 j (dominance >= 1:
/// strictly row diagonally dominant, so the LU without pivoting and its incomplete versions
/// exist and are stable).
SparseMatrix random_block(Index N, Index per_row, std::uint64_t seed, Real dominance = 1.0) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    std::uniform_int_distribution<Index> col(0, N - 1);
    std::vector<Triplet> t;
    for (Index i = 0; i < N; ++i) {
        std::set<Index> cols;
        while (static_cast<Index>(cols.size()) < std::min(per_row, N - 1)) {
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
        t.emplace_back(i, i, Complex(1.0 + dominance * sum, 0.5));
    }
    SparseMatrix A(N, N);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

/// diag(A, D) as one sparse 2N x 2N matrix.
SparseMatrix block_diag(const SparseMatrix& A, const SparseMatrix& D) {
    const Index N = A.rows();
    std::vector<Triplet> t;
    for (Index r = 0; r < N; ++r) {
        for (SparseMatrix::InnerIterator it(A, r); it; ++it)
            t.emplace_back(r, it.col(), it.value());
        for (SparseMatrix::InnerIterator it(D, r); it; ++it)
            t.emplace_back(N + r, N + it.col(), it.value());
    }
    SparseMatrix S(2 * N, 2 * N);
    S.setFromTriplets(t.begin(), t.end());
    S.makeCompressed();
    return S;
}

/// The dense matrix M^{-1} S (columns M^{-1} S e_j).
MatrixXc preconditioned(const solver::Preconditioner& P, const SparseMatrix& S) {
    const Index n = S.rows();
    const MatrixXc Sd = MatrixXc(S);
    MatrixXc out(n, n);
    VectorXc col;
    for (Index j = 0; j < n; ++j) {
        P.apply(Sd.col(j), col);
        out.col(j) = col;
    }
    return out;
}

/// Small sparse matrix from (row, col, value) triplets.
SparseMatrix small(Index n, const std::vector<Triplet>& t) {
    SparseMatrix S(n, n);
    S.setFromTriplets(t.begin(), t.end());
    S.makeCompressed();
    return S;
}

/// Byte-wise equality of two arrays of n trivially copyable values.
template <class T>
bool same_bits(const T* a, const T* b, std::size_t n) {
    return n == 0 || std::memcmp(a, b, n * sizeof(T)) == 0;
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && same_bits(a.data(), b.data(), a.size());
}

bool same_bits(const VectorXc& a, const VectorXc& b) {
    return a.size() == b.size() &&
           same_bits(a.data(), b.data(), static_cast<std::size_t>(a.size()));
}

/// Byte-wise equality of every array and every derived quantity of two block factorisations
/// (the timings excepted).
bool same_factors(const IncompleteLuPreconditioner::Factor& a,
                  const IncompleteLuPreconditioner::Factor& b) {
    return a.n == b.n && a.input_nnz == b.input_nnz && same_bits(a.perm, b.perm) &&
           same_bits(a.ptr, b.ptr) && same_bits(a.mid, b.mid) && same_bits(a.col, b.col) &&
           same_bits(a.val, b.val) && same_bits(a.diag, b.diag) &&
           same_bits(&a.min_pivot_ratio, &b.min_pivot_ratio, 1) &&
           same_bits(&a.norm_inf, &b.norm_inf, 1) && same_bits(&a.macs, &b.macs, 1) &&
           same_bits(&a.condest, &b.condest, 1);
}

#ifdef SPECKLEBEM_HAVE_OPENMP
/// Runs f on thread 0 of an outer team of two threads with max-active-levels = 1, so every
/// OpenMP region inside f, including one with a num_threads(2) clause, is inactive and runs with
/// a team of one thread. Returns the team size of a nested num_threads(2) probe region there: 1
/// as intended; 2 would mean that the outer region was not active (e.g. OMP_THREAD_LIMIT=1).
template <class F>
int run_with_teams_of_one(F&& f) {
    const int saved_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
    int probe = 0;
    std::exception_ptr err;
#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0) {
            try {
#pragma omp parallel num_threads(2)
                {
                    if (omp_get_thread_num() == 0)
                        probe = omp_get_num_threads();
                }
                f();
            } catch (...) {
                err = std::current_exception();
            }
        }
    }
    omp_set_max_active_levels(saved_levels);
    if (err)
        std::rethrow_exception(err);
    return probe;
}
#endif

/// No dropping and a fill limit that keeps every entry: the exact LU without pivoting.
IlutParams exact_params(Index n, IlutOrdering ordering) {
    IlutParams p;
    p.drop_tol = 0.0;
    p.fill_factor = static_cast<Real>(n) + 1.0;
    p.ordering = ordering;
    return p;
}

}  // namespace

TEST_CASE("ilut_preconditioner: without dropping it is the exact LU of diag(A, D)",
          "[ilut_preconditioner]") {
    const Index N = 60;
    const SparseMatrix A = random_block(N, 6, 11);
    const SparseMatrix D = random_block(N, 9, 12);
    SparseMatrix S = block_diag(A, D);
    const Eigen::PartialPivLU<MatrixXc> lu{MatrixXc(S)};
    const VectorXc r = random_vector(2 * N, 13);
    const VectorXc ref = lu.solve(r);
    for (const IlutOrdering ord : {IlutOrdering::Amd, IlutOrdering::Natural}) {
        const IncompleteLuPreconditioner P(A, D, exact_params(N, ord));
        INFO(P.describe());
        CHECK(P.size() == 2 * N);
        CHECK(P.half_size() == N);
        CHECK(P.name() == "ilut");
        CHECK(P.input_nonzeros_a() == A.nonZeros());
        CHECK(P.input_nonzeros_d() == D.nonZeros());
        CHECK(P.fill_ratio_a() > 1.0);  // the exact LU of a random sparse matrix fills in
        CHECK(P.fill_ratio_a() ==
              static_cast<Real>(P.nonzeros_a()) / static_cast<Real>(P.input_nonzeros_a()));
        CHECK(P.min_pivot_ratio_a() > 0.1);
        CHECK(P.memory_bytes() <= P.bound_bytes());
        VectorXc z;
        P.apply(r, z);
        const Real err = (z - ref).norm() / ref.norm();
        INFO("ordering " << solver::to_string(ord) << ": |z - z_LU| / |z_LU| = " << err);
        // Measured 4.1e-16 (AMD) / 4.3e-16 (natural), win-release: the same LU up to the
        // rounding of a different elimination order and pivoting.
        CHECK(err <= 1e-13);
        // In place (r and z the same vector).
        VectorXc rr = r;
        P.apply(rr, rr);
        CHECK(bitwise_equal(rr, z));
        // GMRES on diag(A, D) preconditioned by its exact LU: one iteration.
        const VectorXc b = random_vector(2 * N, 14);
        SparseMatrix copy = S;
        const op::SparseOperator op(std::move(copy));
        solver::GmresParams g;
        g.tolerance = 1e-10;
        g.max_iter = 50;
        g.verbose = false;
        const solver::GmresResult res = solver::gmres(op, b, P, g);
        CHECK(res.converged);
        CHECK(res.iterations == 1);
    }
}

TEST_CASE("ilut_preconditioner: AMD elimination order of an arrow matrix has no fill",
          "[ilut_preconditioner]") {
    // Arrow matrix with the hub at unknown 0 (dense row and column 0): in the natural order the
    // exact LU fills completely; AMD eliminates the hub last and has no fill, which checks the
    // convention perm[k] = original index of the k-th eliminated unknown (the inverse
    // permutation would place the hub early and fill in).
    const Index N = 40;
    std::vector<Triplet> t;
    for (Index i = 0; i < N; ++i) {
        t.emplace_back(i, i, Complex(static_cast<Real>(N) + 2.0, 0.0));
        if (i > 0) {
            t.emplace_back(0, i, Complex(1.0, 0.5));
            t.emplace_back(i, 0, Complex(1.0, -0.5));
        }
    }
    const SparseMatrix A = small(N, t);
    const IncompleteLuPreconditioner amd(A, A, exact_params(N, IlutOrdering::Amd));
    const IncompleteLuPreconditioner nat(A, A, exact_params(N, IlutOrdering::Natural));
    INFO(amd.describe() << "\n" << nat.describe());
    CHECK(amd.nonzeros_a() == A.nonZeros());
    CHECK(nat.nonzeros_a() == N * N);
    const VectorXc r = random_vector(2 * N, 21);
    VectorXc za, zn;
    amd.apply(r, za);
    nat.apply(r, zn);
    CHECK((za - zn).norm() <= 1e-13 * zn.norm());
}

TEST_CASE("ilut_preconditioner: the error decreases with the drop tolerance",
          "[ilut_preconditioner]") {
    // ||I - M^{-1} diag(A, D)||_F for drop 1e-2 -> 1e-3 -> 1e-4 at a fixed fill limit (20, high
    // enough that the drop tolerance decides) decreases monotonically, and the factors grow.
    const Index N = 200;
    const SparseMatrix A = random_block(N, 8, 31);
    const SparseMatrix D = random_block(N, 8, 32);
    const SparseMatrix S = block_diag(A, D);
    const MatrixXc I = MatrixXc::Identity(2 * N, 2 * N);
    std::vector<Real> err;
    std::vector<Index> nnz;
    std::string trace;
    for (const Real drop : {1e-2, 1e-3, 1e-4}) {
        IlutParams p;
        p.drop_tol = drop;
        p.fill_factor = 20.0;
        const IncompleteLuPreconditioner P(A, D, p);
        err.push_back((I - preconditioned(P, S)).norm());
        nnz.push_back(P.nonzeros_a() + P.nonzeros_d());
        trace += "drop " + std::to_string(drop) + ": error " + std::to_string(err.back()) +
                 ", nnz(L+U) " + std::to_string(nnz.back()) + "; ";
        CHECK(P.fill_ratio_a() <= 20.0 + 1e-12);
        CHECK(P.fill_ratio_d() <= 20.0 + 1e-12);
    }
    INFO(trace);
    // Measured (win-release): 0.51 / 0.059 / 0.0039 with nnz(L+U) 7 493 / 19 214 / 29 541.
    CHECK(err[1] < err[0]);
    CHECK(err[2] < err[1]);
    CHECK(nnz[1] > nnz[0]);
    CHECK(nnz[2] > nnz[1]);
    CHECK(err[2] < 0.05);
}

TEST_CASE("ilut_preconditioner: setup and apply are bitwise identical with one and two threads",
          "[ilut_preconditioner]") {
#ifndef SPECKLEBEM_HAVE_OPENMP
    SKIP("built without OpenMP");
#else
    // The OpenMP regions of setup and apply() carry num_threads(2), so omp_set_num_threads()
    // does not change their team size. Teams of two: top-level calls. Teams of one: the same
    // calls nested in an outer parallel region with max-active-levels = 1 (inactive regions).
    const Index N = 500;
    const SparseMatrix A = random_block(N, 10, 41);
    const SparseMatrix D = random_block(N, 12, 42);
    const VectorXc r = random_vector(2 * N, 43);
    IlutParams p;
    p.drop_tol = 1e-3;
    p.fill_factor = 3.0;
    int top_team = 0;
#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0)
            top_team = omp_get_num_threads();
    }
    const IncompleteLuPreconditioner P2(A, D, p);
    VectorXc z22;  // two-thread setup, two-thread apply
    P2.apply(r, z22);
    std::optional<IncompleteLuPreconditioner> P1;
    VectorXc z11, z21;  // one-thread setup + apply; two-thread setup, one-thread apply
    const int nested_team = run_with_teams_of_one([&] {
        P1.emplace(A, D, p);
        P1->apply(r, z11);
        P2.apply(r, z21);
    });
    REQUIRE(P1.has_value());
    VectorXc z12;  // one-thread setup, two-thread apply
    P1->apply(r, z12);
    INFO("team size of a num_threads(2) region: top level " << top_team << ", nested "
                                                            << nested_team);
    REQUIRE(top_team == 2);
    REQUIRE(nested_team == 1);
    CHECK(same_factors(P1->factors_a(), P2.factors_a()));
    CHECK(same_factors(P1->factors_d(), P2.factors_d()));
    CHECK(P1->memory_bytes() == P2.memory_bytes());
    CHECK(same_bits(z11, z22));
    CHECK(same_bits(z11, z12));
    CHECK(same_bits(z11, z21));
#endif
}

TEST_CASE("ilut_preconditioner: errors", "[ilut_preconditioner]") {
    const SparseMatrix A = random_block(10, 3, 51);
    const SparseMatrix B = random_block(12, 3, 52);
    // Sizes and shapes.
    CHECK_THROWS_WITH(IncompleteLuPreconditioner(A, B), ContainsSubstring("both must be N x N"));
    CHECK_THROWS_AS(IncompleteLuPreconditioner(A, B), std::invalid_argument);
    SparseMatrix rect(10, 12);
    CHECK_THROWS_WITH(IncompleteLuPreconditioner(rect, A), ContainsSubstring("square"));
    CHECK_THROWS_AS(IncompleteLuPreconditioner(A, rect), std::invalid_argument);
    const SparseMatrix empty;
    CHECK_THROWS_WITH(IncompleteLuPreconditioner(empty, empty), ContainsSubstring("empty"));
    // Parameters.
    const auto bad = [&](IlutParams p) {
        CHECK_THROWS_AS(IncompleteLuPreconditioner(A, A, p), std::invalid_argument);
        CHECK_THROWS_AS(solver::validate_ilut_params(p), std::invalid_argument);
    };
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    for (const Real v : {-1e-4, nan, inf}) {
        IlutParams p;
        p.drop_tol = v;
        bad(p);
    }
    for (const Real v : {0.0, -1.0, nan, inf}) {
        IlutParams p;
        p.fill_factor = v;
        bad(p);
    }
    for (const Real v : {-1e-3, nan, inf}) {
        IlutParams p;
        p.min_pivot_rel = v;
        bad(p);
    }
    {
        IlutParams p;
        p.budget_bytes = 0;
        bad(p);
        IlutParams q;
        q.ordering = static_cast<IlutOrdering>(7);
        bad(q);
    }
    CHECK_NOTHROW(solver::validate_ilut_params(IlutParams{}));
    // Non-finite entry.
    {
        SparseMatrix C = A;
        C.coeffRef(3, 3) = Complex(inf, 0.0);
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(A, C),
                          ContainsSubstring("block D: non-finite entry (3, 3)"));
        CHECK_THROWS_AS(IncompleteLuPreconditioner(A, C), std::invalid_argument);
    }
    // A zero row: singular block, refused.
    {
        SparseMatrix Z = A;
        Z.prune([](Index r, Index, const Complex&) { return r != 4; });
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(Z, A),
                          ContainsSubstring("block A: row 4 is zero"));
        CHECK_THROWS_AS(IncompleteLuPreconditioner(A, Z), std::runtime_error);
    }
    // A matrix whose exact LU (in any symmetric order) has a zero pivot: [1 1; 1 1]. No pivot
    // is replaced: the setup throws, with and without dropping, in both orderings.
    {
        const Complex one(1.0, 0.0);
        const SparseMatrix S = small(2, {{0, 0, one}, {0, 1, one}, {1, 0, one}, {1, 1, one}});
        const SparseMatrix I = small(2, {{0, 0, one}, {1, 1, one}});
        for (const IlutOrdering ord : {IlutOrdering::Amd, IlutOrdering::Natural}) {
            IlutParams p = exact_params(2, ord);
            CHECK_THROWS_WITH(IncompleteLuPreconditioner(I, S, p),
                              ContainsSubstring("block D: zero or tiny pivot in row"));
            CHECK_THROWS_AS(IncompleteLuPreconditioner(S, I, p), std::runtime_error);
            p.drop_tol = 1e-4;
            p.fill_factor = 3.0;
            CHECK_THROWS_AS(IncompleteLuPreconditioner(S, I, p), std::runtime_error);
        }
        // A tiny pivot: [1 1; 1 1 + 1e-13] has |u_22| / |row 2| = 7e-14, refused by the
        // default guard (1e-10), accepted with a lower one.
        const SparseMatrix T =
            small(2, {{0, 0, one}, {0, 1, one}, {1, 0, one}, {1, 1, Complex(1.0 + 1e-13, 0.0)}});
        IlutParams p = exact_params(2, IlutOrdering::Natural);
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(T, I, p),
                          ContainsSubstring("block A: zero or tiny pivot in row 1"));
        p.min_pivot_rel = 1e-15;
        CHECK_NOTHROW(IncompleteLuPreconditioner(T, I, p));
        // [0 1; 1 0] in the natural order: the first pivot is zero.
        const SparseMatrix X = small(2, {{0, 1, one}, {1, 0, one}});
        p.min_pivot_rel = 0.0;
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(X, I, p),
                          ContainsSubstring("zero or tiny pivot in row 0"));
    }
    // Budget: the pre-check against the bound of the fill limit (before any factorisation).
    // The post-check (actual factor memory) cannot fire after a passed pre-check, since the
    // factors never exceed the bound; it guards against a change of the accounting.
    {
        IlutParams p;
        p.budget_bytes = 100;
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(A, A, p),
                          ContainsSubstring("exceeds the budget 0.0 MB"));
        CHECK_THROWS_AS(IncompleteLuPreconditioner(A, A, p), std::runtime_error);
        const IncompleteLuPreconditioner ok(A, A);
        p.budget_bytes = ok.bound_bytes();
        const IncompleteLuPreconditioner exact(A, A, p);
        CHECK(exact.budget_bytes() == ok.bound_bytes());
        CHECK(exact.memory_bytes() <= exact.budget_bytes());
    }
    // The sources constructor.
    {
        const SparseMatrix Z = block_diag(A, A);
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(Sources{}, 10),
                          ContainsSubstring("no source"));
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(Sources{nullptr}, 10),
                          ContainsSubstring("null source"));
        CHECK_THROWS_WITH(IncompleteLuPreconditioner(Sources{&Z}, 9), ContainsSubstring("2N x 2N"));
        CHECK_THROWS_AS(IncompleteLuPreconditioner(Sources{&Z}, 0), std::invalid_argument);
    }
    // apply() size check.
    {
        const IncompleteLuPreconditioner P(A, A);
        VectorXc z;
        CHECK_THROWS_AS(P.apply(VectorXc::Ones(19), z), std::invalid_argument);
    }
}

TEST_CASE("ilut_preconditioner: the sources constructor reads A and D of the summed sources",
          "[ilut_preconditioner]") {
    // Z = [A B; C D] split into two disjoint sources (as Z_rr's near part and the ring): the
    // preconditioner ignores B and C and equals the one built from A and D directly; an entry
    // stored in both sources is summed.
    const Index N = 80;
    const SparseMatrix A = random_block(N, 7, 61);
    const SparseMatrix D = random_block(N, 7, 62);
    const SparseMatrix C = random_block(2 * N, 9, 63);  // supplies B and C (and is cut away)
    std::vector<Triplet> t1, t2;
    std::mt19937_64 rng(64);
    std::uniform_int_distribution<int> coin(0, 1);
    for (Index r = 0; r < 2 * N; ++r) {
        const SparseMatrix& blk = r < N ? A : D;
        const Index off = r < N ? 0 : N;
        for (SparseMatrix::InnerIterator it(blk, r - off); it; ++it)
            (coin(rng) == 0 ? t1 : t2).emplace_back(r, off + it.col(), it.value());
        for (SparseMatrix::InnerIterator it(C, r); it; ++it)
            if ((r < N) != (it.col() < N))
                (coin(rng) == 0 ? t1 : t2).emplace_back(r, it.col(), it.value());
    }
    const SparseMatrix S1 = small(2 * N, t1);
    const SparseMatrix S2 = small(2 * N, t2);
    IlutParams p;
    p.drop_tol = 1e-3;
    p.fill_factor = 2.5;
    const IncompleteLuPreconditioner P(Sources{&S1, &S2}, N, p);
    const IncompleteLuPreconditioner Q(A, D, p);
    CHECK(P.input_nonzeros_a() == A.nonZeros());
    CHECK(P.input_nonzeros_d() == D.nonZeros());
    CHECK(P.nonzeros_a() == Q.nonzeros_a());
    CHECK(P.nonzeros_d() == Q.nonzeros_d());
    const VectorXc v = random_vector(2 * N, 65);
    VectorXc zp, zq;
    P.apply(v, zp);
    Q.apply(v, zq);
    CHECK(bitwise_equal(zp, zq));
    // An A entry split over both sources is summed: half of every A entry in each source.
    std::vector<Triplet> h;
    for (Index r = 0; r < N; ++r)
        for (SparseMatrix::InnerIterator it(A, r); it; ++it)
            h.emplace_back(r, it.col(), 0.5 * it.value());
    for (Index r = 0; r < N; ++r)
        for (SparseMatrix::InnerIterator it(D, r); it; ++it)
            h.emplace_back(N + r, N + it.col(), it.value());
    const SparseMatrix H1 = small(2 * N, h);
    h.clear();
    for (Index r = 0; r < N; ++r)
        for (SparseMatrix::InnerIterator it(A, r); it; ++it)
            h.emplace_back(r, it.col(), 0.5 * it.value());
    const SparseMatrix H2 = small(2 * N, h);
    const IncompleteLuPreconditioner R(Sources{&H1, &H2}, N, p);
    VectorXc zr;
    R.apply(v, zr);
    CHECK((zr - zq).norm() <= 1e-14 * zq.norm());
}

namespace {

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

}  // namespace

TEST_CASE("ilut_preconditioner: PMCHWT, ICTF and MCTF give the same preconditioned system",
          "[ilut_preconditioner]") {
    // The ADR 0011 invariance: PMCHWT, ICTF and MCTF scale the J-test rows by one constant and
    // the M-test rows by another, i.e. A by alpha and D by beta. The ILUT's drop rule, fill
    // selection and pivot guard are relative to the row norm and AMD depends only on the
    // pattern, so Ã scales by alpha and D̃ by beta: M^{-1} Z is the same matrix up to rounding
    // and left GMRES runs the same iterates. Ag sphere R = 0.5 um, icosphere n = 2 at lambda =
    // 1 um (2N = 960), a forced 3-level tree (far pairs exact), ILUT fill 3, drop 1e-4.
    const Real lambda = 1e-6;
    const geometry::TriangleMesh mesh = geometry::make_icosphere(0.5e-6, 2);
    solver::GmresParams g;
    g.tolerance = 1e-10;
    g.max_iter = 500;
    g.verbose = false;
    std::vector<solver::GmresResult> res;
    std::vector<MatrixXc> pz;
    std::vector<Index> nnz;
    std::string desc;
#ifdef NDEBUG
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF, Kind::MCTF};
    const Index stride = 1;
#else
    // Sanitizer build: one pair and every 16th column (as the block and Schur tests).
    const std::vector<Kind> kinds = {Kind::PMCHWT, Kind::ICTF};
    const Index stride = 16;
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
        const Index n2 = Z.rows();
        const auto P = make_incomplete_lu_preconditioner(Z, s.problem);
        desc = P->describe();
        nnz.push_back(P->nonzeros_a() + P->nonzeros_d());
        res.push_back(solver::gmres(Z, b, *P, g));
        REQUIRE(res.back().converged);
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
        // The same drop and fill decisions (no tie flipped by the rounding of the scaled
        // entries): the same factor pattern.
        CHECK(nnz[f] == nnz[0]);
        // Measured 1.6e-15 (win-release, ICTF and MCTF vs PMCHWT, all columns; 40 iterations for
        // all three): only the rounding of the scaled entries differs. The bound leaves three
        // orders of magnitude.
        CHECK(dpz <= 1e-12);
        CHECK(std::abs(res[f].iterations - res[0].iterations) <= 1);
        const std::size_t m =
            std::min(res[f].residual_history.size(), res[0].residual_history.size());
        // Iterates equal up to rounding for the first 40 iterations (afterwards the rounding of
        // the Krylov process grows by ~10x per 3-4 iterations, as for the block, Schur and
        // Jacobi preconditioners).
        for (std::size_t k = 0; k < std::min<std::size_t>(m, 40); ++k) {
            CHECK(std::abs(res[f].residual_history[k] - res[0].residual_history[k]) <=
                  1e-9 * res[0].residual_history[k]);
        }
        CHECK((res[f].x - res[0].x).norm() <= 1e-8 * res[0].x.norm());
    }
}

TEST_CASE(
    "ilut_preconditioner: make_incomplete_lu_preconditioner on the PML route (Z_rr near + ring)",
    "[ilut_preconditioner]") {
    // The tiny rough PML surface with a forced 3-level tree: the factory reads A and D from Z_rr's
    // near part and the near ring and equals the preconditioner of the blocks of the summed
    // near-field matrix Z_n.
    PmlSetup c(true);
    const mlfmm::MlfmmOperator Z(c.problem, tree_params(3));
    REQUIRE(Z.ring_operator() != nullptr);
    const Index N = c.space.size();
    IlutParams p;
    p.drop_tol = 1e-3;
    p.fill_factor = 2.0;
    const auto P = make_incomplete_lu_preconditioner(Z, c.problem, p);
    REQUIRE(P != nullptr);
    INFO(P->describe());
    CHECK(P->size() == 2 * N);
    const SparseMatrix Zn = near_field_matrix(Z);
    const SparseMatrix A = Zn.topLeftCorner(N, N);
    const SparseMatrix D = Zn.bottomRightCorner(N, N);
    CHECK(P->input_nonzeros_a() == A.nonZeros());
    CHECK(P->input_nonzeros_d() == D.nonZeros());
    CHECK(P->fill_ratio_a() <= 2.0 + 1e-12);
    const IncompleteLuPreconditioner Q(A, D, p);
    CHECK(P->nonzeros_a() == Q.nonzeros_a());
    CHECK(P->nonzeros_d() == Q.nonzeros_d());
    const VectorXc r = random_vector(2 * N, 71);
    VectorXc zp, zq;
    P->apply(r, zp);
    Q.apply(r, zq);
    CHECK(bitwise_equal(zp, zq));
    // Errors: the dense route has no near-field matrix; wrong sizes.
    const MatrixXc I = MatrixXc::Identity(Z.rows(), Z.cols());
    const op::DenseOperator Dn(I);
    CHECK_THROWS_WITH(make_incomplete_lu_preconditioner(Dn, c.problem),
                      ContainsSubstring("not an MLFMM operator"));
    CHECK_THROWS_AS(make_incomplete_lu_preconditioner(Dn, c.problem), std::invalid_argument);
    const MatrixXc I4 = MatrixXc::Identity(4, 4);
    const op::DenseOperator wrong(I4);
    CHECK_THROWS_WITH(make_incomplete_lu_preconditioner(wrong, c.problem),
                      ContainsSubstring("not 2N x 2N"));
    IlutParams badp;
    badp.fill_factor = 0.0;
    CHECK_THROWS_AS(make_incomplete_lu_preconditioner(Z, c.problem, badp), std::invalid_argument);
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

TEST_CASE("ilut_preconditioner: Simulation option, PML solve reproduces the block solve",
          "[ilut_preconditioner]") {
    // Tiny flat PML surface (2N = 352), compression "mlfmm": the ILUT-preconditioned solve
    // reproduces the block-preconditioned one (leaf blocks; currents to the GMRES tolerance, R =
    // P_refl / P_inc to 1e-8). Measured (win-release): block 46 / ILUT 4 iterations, R =
    // 0.0350102 for both.
    const TinyPml t;
    SimulationConfig cb = t.cfg;
    cb.preconditioner = PreconditionerKind::BlockDiagonal;
    cb.block_levels_above_leaf = 0;
    const auto blk = t.make(cb);
    const solver::GmresResult rb = blk->solve();
    REQUIRE(rb.converged);
    SimulationConfig ci = t.cfg;
    ci.preconditioner = PreconditionerKind::IncompleteLu;
    ci.ilut.fill_factor = 2.0;
    ci.ilut.drop_tol = 1e-3;
    const auto ilu = t.make(ci);
    CHECK(ilu->preconditioner_kind() == PreconditionerKind::IncompleteLu);
    CHECK(ilu->incomplete_lu_preconditioner() == nullptr);
    CHECK_FALSE(ilu->diagonal_preconditioner());
    const solver::GmresResult ri = ilu->solve();
    REQUIRE(ilu->incomplete_lu_preconditioner() != nullptr);
    CHECK(ilu->block_preconditioner() == nullptr);
    CHECK(ilu->schur_preconditioner() == nullptr);
    const std::string rep = ilu->report();
    INFO("block " << rb.iterations << " / ILUT " << ri.iterations << " iterations\n" << rep);
    CHECK(ri.converged);
    CHECK(ilu->incomplete_lu_preconditioner()->params().fill_factor == 2.0);
    const VectorXc& xb = blk->solution().currents;
    CHECK((ilu->solution().currents - xb).norm() <= 1e-7 * xb.norm());
    const Real Rb = blk->reflected_power(18, 36) / blk->incident_power();
    const Real Ri = ilu->reflected_power(18, 36) / ilu->incident_power();
    INFO("R block " << Rb << ", ILUT " << Ri);
    CHECK(std::abs(Ri - Rb) <= 1e-8);
    CHECK(contains(rep,
                   "preconditioner: ILUT of the near-field blocks A, D (explicit), fill 2, "
                   "drop 0.001, ordering amd"));
    CHECK(contains(rep, "preconditioner setup:"));
    CHECK(contains(rep, "ILUT (fill 2, drop 1.0e-03, ordering amd)"));
    CHECK(contains(rep, "ILUT preconditioner"));
    const solver::IncompleteLuPreconditioner* p = ilu->incomplete_lu_preconditioner();
    ilu->assemble();  // idempotent
    CHECK(ilu->incomplete_lu_preconditioner() == p);

    // Validation: explicit only, compression "mlfmm", consistent with diagonal_preconditioner,
    // valid ILUT parameters (checked by the constructor, before any assembly).
    SimulationConfig dense = ci;
    dense.compression = "dense";
    CHECK_THROWS_WITH(t.make(dense), ContainsSubstring("\"ilut\" needs compression \"mlfmm\""));
    CHECK_THROWS_AS(t.make(dense), std::invalid_argument);
    for (const bool d : {true, false}) {
        SimulationConfig c = ci;
        c.diagonal_preconditioner = d;
        CHECK_THROWS_WITH(t.make(c), ContainsSubstring("contradicts diagonal_preconditioner"));
    }
    SimulationConfig badp = ci;
    badp.ilut.drop_tol = -1.0;
    CHECK_THROWS_WITH(t.make(badp), ContainsSubstring("drop_tol"));
    SimulationConfig other = t.cfg;
    other.preconditioner = PreconditionerKind::BlockDiagonal;
    other.ilut.drop_tol = -1.0;  // ignored unless IncompleteLu
    CHECK_NOTHROW(t.make(other));
    // Never automatic: the "mlfmm" PML default stays the block preconditioner.
    CHECK(t.make(t.cfg)->preconditioner_kind() == PreconditionerKind::BlockDiagonal);
    CHECK(std::string(to_string(PreconditionerKind::IncompleteLu)) == "ilut");
}

TEST_CASE("ilut_preconditioner: an exactly zero pivot is refused by the LU blocks (ADR 0012 d8)",
          "[ilut_preconditioner][block_preconditioner]") {
    // BlockDiagonalPreconditioner and the Schur blocks check the U diagonal of each block's LU
    // for an exact zero, whatever Eigen's rcond() estimate says (it divides by the zero pivot and
    // can return a finite value), and name the block.
    const Index n = 12;
    MatrixXc A = MatrixXc::Identity(n, n) * 4.0;
    for (Index i = 0; i + 1 < n; ++i) {
        A(i, i + 1) = Complex(1.0, 0.5);
        A(i + 1, i) = Complex(0.5, -1.0);
    }
    solver::BlockPartition part;
    part.blocks = {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}};
    part.ids = {3, 5};
    CHECK_NOTHROW(solver::BlockDiagonalPreconditioner(part, A));
    MatrixXc Z = A;
    Z.row(8).setZero();  // a zero row in block 5: its partial-pivoting LU has a zero pivot
    CHECK_THROWS_WITH(solver::BlockDiagonalPreconditioner(part, Z),
                      ContainsSubstring("block 5 (6 unknowns) is singular or ill-conditioned: "
                                        "exactly zero pivot on the U diagonal"));
    CHECK_THROWS_AS(solver::BlockDiagonalPreconditioner(part, Z), std::runtime_error);
    // Even with min_rcond = 0 (no estimate threshold at all).
    CHECK_THROWS_WITH(solver::BlockDiagonalPreconditioner(part, Z, 0.0),
                      ContainsSubstring("exactly zero pivot"));
    // The Schur complement's J blocks: A = [1 0; 0 0] (a zero row).
    const Complex one(1.0, 0.0);
    const SparseMatrix S = small(4, {{0, 0, one}, {2, 2, one}, {3, 3, one}, {1, 2, one}});
    solver::BlockPartition jp, mp;
    jp.blocks = {{0, 1}};
    mp.blocks = {{2, 3}};
    CHECK_THROWS_WITH(solver::SchurComplementPreconditioner(S, 2, jp, mp),
                      ContainsSubstring("J block 0 (2 unknowns) is singular or ill-conditioned: "
                                        "exactly zero pivot"));
    CHECK_THROWS_WITH(solver::SchurComplementPreconditioner(S, 2, jp, mp),
                      ContainsSubstring("estimated reciprocal condition number taken as 0"));
}
