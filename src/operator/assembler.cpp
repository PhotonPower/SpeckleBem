/// @file assembler.cpp
/// Dense assembly of the combined tangential system, its right-hand side and its diagonal
/// (docs/03_theory_sie.md, Fu et al. 2023 Eqs. 1-6 and Table 1).
///
/// Conventions (docs/06): exp(+jwt); n points out of R2 into R1; J = n x H, M = -n x E on the
/// R1 side, J_2 = -J, M_2 = -M; K X = int grad' G x X dS' (source-point gradient). Scattered
/// fields of region i (docs/03): E_i^s = -L_i J_i + K_i M_i, H_i^s = -K_i J_i - (1/eta_i^2)
/// L_i M_i. kernels::element_blocks returns the principal value K^PV, kernels::jump_block
/// I_mn = int f_m . (n x f_n) dS. The limit of K X on S is K^PV X + 1/2 n x X from the R1
/// side and K^PV X - 1/2 n x X from the R2 side.
///
/// Tangential equations with the jump terms worked in. For a tangential X, n x (n x X) = -X,
/// so the definitions give E_tan = n x M and H_tan = -n x J on S (both sides, tangential
/// continuity).
///
///  Region 1 (total field E_inc + E_1^s, J_1 = J, M_1 = M, limit from the R1 side):
///    n x M = E_inc,tan - L_1 J + K_1^PV M + 1/2 n x M
///      =>  E_inc,tan = L_1 J - (K_1^PV - 1/2 n x) M,
///    -n x J = H_inc,tan - K_1^PV J - 1/2 n x J - (1/eta_1^2) L_1 M
///      =>  H_inc,tan = (K_1^PV - 1/2 n x) J + (1/eta_1^2) L_1 M.
///  Region 2 (no incident field, J_2 = -J, M_2 = -M, limit from the R2 side):
///    n x M = L_2 J - K_2^PV M + 1/2 n x M
///      =>  0 = L_2 J - (K_2^PV + 1/2 n x) M,
///    -n x J = K_2^PV J - 1/2 n x J + (1/eta_2^2) L_2 M
///      =>  0 = (K_2^PV + 1/2 n x) J + (1/eta_2^2) L_2 M.
///
///  Sanity check: region 1 with M = 0 (PEC) gives H_inc,tan = K^PV J - 1/2 n x J; n x this is
///  n x H_inc = n x K^PV J + 1/2 J, Harrington's MFIE n x H_inc = 1/2 J + n x PV int grad' G x J
///  (= 1/2 J - n x PV int J x grad' G). Correct.
///
/// Galerkin testing with f_m (<f_m, n x f_n> = I_mn) turns the region operators of the
/// T-EFIE_i / T-MFIE_i of docs/03 into
///    K_1 = K_1^PV - 1/2 I,   K_2 = K_2^PV + 1/2 I.
/// The sign is opposite to the jump of K itself (+1/2 on the R1 side) because the definition
/// terms n x E = -M and n x H = J, moved to the left-hand side, contribute -n x M (+1/2 - 1 =
/// -1/2) in region 1 and +n x M in region 2 (-1/2 + 1 = +1/2). The per-region signs are
/// tested in tests/unit/test_assembler.cpp against exact Mie currents, for each region alone
/// (the combined formulations of Table 1 cancel the jump terms, see below).
///
/// Combination (docs/03 Eqs. 5-6) with (a1, a2, b1, b2) = formulation.weights(eta1, eta2):
///    Z = [ sum_i (a_i/eta_i) L_i      -sum_i (a_i/eta_i) K_i ]     x = [J; M]
///        [ sum_i  b_i eta_i  K_i       sum_i (b_i/eta_i) L_i ]
///    b = [ (a1/eta1) <f_m, E_inc> ;  b1 eta1 <f_m, H_inc> ].
/// Jump terms in Z: JM block 1/2 (a1/eta1 - a2/eta2) I, MJ block 1/2 (b2 eta2 - b1 eta1) I. For
/// PMCHWT (a_i/eta_i = 1, b_i eta_i = 1), ICTF (2/(eta1+eta2), (eta1+eta2)/2) and MCTF (1,
/// eta1 eta2) both coefficients are independent of i, so the jump terms cancel (up to the
/// rounding of a_i/eta_i) and only K^PV remains. The Galerkin L and K^PV are complex-symmetric
/// (L^T = L, (K^PV)^T = K^PV), so Z itself is block-antisymmetric in the K blocks
/// (Z_JM^T = -(a/eta)/(b eta) Z_MJ); the symmetric form is S = diag(b1 eta1 I, -(a1/eta1) I) Z,
/// for PMCHWT S = diag(I, -I) Z (the docs/01 / docs/05 symmetry criterion). For the same reason
/// each of the three is a row scaling of PMCHWT, Z = diag((a1/eta1) I, b1 eta1 I) Z_PMCHWT, with
/// b scaled alike: a direct solve gives the same currents, and the formulations differ only in
/// the conditioning an iterative solver sees (docs/03).
///
/// Parallelism (docs/07: deterministic results): the test triangles are coloured greedily so
/// that no two triangles of one colour share a basis function (edge-adjacent triangles differ;
/// at most 4 colours since a triangle has at most 3 neighbours). The colours are processed in
/// turn; within a colour the triangles are distributed with an OpenMP dynamic schedule, each
/// writing only the rows of its own basis functions (m and N + m). A basis function lives on two
/// triangles of different colours, so within a colour each row receives contributions from one
/// test triangle only, and every entry Z(m, n) receives its contributions in a fixed order
/// (colour of the test triangle, then source triangle index) for any thread count and any
/// assignment of triangles to threads: the matrix is bitwise identical with 1 or many threads
/// and free of data races. Load balance (WP-P2): the cost of a row of pair blocks varies by
/// more than x10 on graded meshes (coarse box triangles of rough surfaces need high near/far
/// degrees against every source triangle), and the static schedule of contiguous chunks gave one
/// thread all coarse rows (x2.5 the ideal time on 24 threads for the WP15 boxes). Within a
/// colour the triangles are therefore handed out one by one (schedule(dynamic, 1)) in the order
/// of decreasing longest edge (a cost proxy; largest first keeps the tail short), which does
/// not change any entry.
///
/// Sparse assembly (WP19a, assemble_sparse) shares this schedule (test_schedule,
/// for_each_test_triangle) and pair_blocks; per test triangle it visits only the source
/// triangles of its pattern pairs, in ascending order, and scatters only the pattern entries.
/// A stored entry thus receives exactly the dense contributions in the dense order: bitwise
/// equal to the DenseStrategy entry.
#include "specklebem/operator/assembler.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/region_sparse_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specklebem::op {

namespace {

using Block = Eigen::Matrix<Complex, 3, 3>;

constexpr Real kRelTol = 1e-12;

bool close(Complex a, Complex b) {
    return std::abs(a - b) <= kRelTol * std::max(std::abs(a), std::abs(b));
}

kernels::RegionParams region_params(const material::Material& m, Real omega) {
    return {m.wavenumber(omega), m.wave_impedance(omega), omega, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

/// Region parameters and the scalar factors of the region blocks in Z.
struct Setup {
    std::array<kernels::RegionParams, 2> region;
    std::array<Complex, 2> e;  ///< a_i / eta_i : +L_i in JJ, -K_i in JM, E part of b (i = 0)
    std::array<Complex, 2> h;  ///< b_i eta_i   : +K_i in MJ, H part of b (i = 0)
    std::array<Complex, 2> m;  ///< b_i / eta_i : +L_i in MM
    /// Region i contributes (some weight non-zero); element_blocks is skipped otherwise.
    std::array<bool, 2> active;
};

Setup make_setup(const Problem& p) {
    Setup s;
    s.region[0] = region_params(p.exterior, p.omega);
    s.region[1] = region_params(p.object, p.omega);
    const Complex eta1 = s.region[0].eta;
    const Complex eta2 = s.region[1].eta;
    const formulation::Weights w = p.formulation->weights(eta1, eta2);
    s.e = {w.a1 / eta1, w.a2 / eta2};
    s.h = {w.b1 * eta1, w.b2 * eta2};
    s.m = {w.b1 / eta1, w.b2 / eta2};
    const Complex zero(0.0, 0.0);
    for (std::size_t i = 0; i < 2; ++i) {
        s.active[i] = s.e[i] != zero || s.h[i] != zero || s.m[i] != zero;
    }
    return s;
}

/// Combined local blocks of one triangle pair (slots as in kernels::element_blocks), row-major
/// 3 x 3 arrays: entry (a, b) at index 3 a + b. Plain arrays and scalar loops keep the
/// per-pair overhead small also in the unoptimised sanitizer build.
struct PairBlocks {
    std::array<Complex, 9> jj, jm, mj, mm;
};

/// Jump coefficient of K_i in the Galerkin tangential equations: K_1 = K_1^PV - 1/2 I,
/// K_2 = K_2^PV + 1/2 I (file comment).
constexpr std::array<Real, 2> kJumpSign = {-0.5, 0.5};

void pair_blocks(const basis::RwgSpace& space, Index t, Index s, const Setup& setup,
                 const kernels::OperatorOptions& opt, PairBlocks& out) {
    out.jj.fill(Complex(0.0, 0.0));
    out.jm.fill(Complex(0.0, 0.0));
    out.mj.fill(Complex(0.0, 0.0));
    out.mm.fill(Complex(0.0, 0.0));
    Block L;
    Block K;
    Block I;
    const bool identical = t == s;
    if (identical) {
        kernels::jump_block(space, t, I);
    }
    for (std::size_t i = 0; i < 2; ++i) {
        if (!setup.active[i])
            continue;
        kernels::element_blocks(space, t, s, setup.region[i], opt, L, K);
        const Complex ce = setup.e[i];
        const Complex ch = setup.h[i];
        const Complex cm = setup.m[i];
        const Real jump = kJumpSign[i];
        for (Eigen::Index a = 0; a < 3; ++a) {
            for (Eigen::Index b = 0; b < 3; ++b) {
                const auto k = static_cast<std::size_t>(3 * a + b);
                const Complex l = L(a, b);
                const Complex kk = identical ? K(a, b) + jump * I(a, b) : K(a, b);
                out.jj[k] += ce * l;
                out.jm[k] -= ce * kk;
                out.mj[k] += ch * kk;
                out.mm[k] += cm * l;
            }
        }
    }
}

/// Greedy colouring of the triangles: triangles that share a basis function (i.e. are
/// adjacent across an interior edge) get different colours. Returns the triangles of each
/// colour in ascending order, and the colour of every triangle in `colour_of`.
std::vector<std::vector<Index>> colour_triangles(const basis::RwgSpace& space,
                                                 std::vector<int>& colour_of) {
    const Index F = space.mesh().num_triangles();
    colour_of.assign(static_cast<std::size_t>(F), -1);
    std::vector<std::vector<Index>> groups;
    for (Index t = 0; t < F; ++t) {
        const basis::RwgSpace::Support sup = space.support(t);
        std::array<bool, 4> used{};  // at most 3 neighbours => colours 0..3
        for (int a = 0; a < sup.count; ++a) {
            const Index n = sup.n[a];
            const Index other =
                space.plus_triangle(n) == t ? space.minus_triangle(n) : space.plus_triangle(n);
            const int c = colour_of[static_cast<std::size_t>(other)];
            if (c >= 0) {
                used[static_cast<std::size_t>(c)] = true;
            }
        }
        int c = 0;
        while (used[static_cast<std::size_t>(c)]) ++c;
        colour_of[static_cast<std::size_t>(t)] = c;
        if (static_cast<std::size_t>(c) >= groups.size()) {
            groups.resize(static_cast<std::size_t>(c) + 1);
        }
        groups[static_cast<std::size_t>(c)].push_back(t);
    }
    return groups;
}

/// Orders the triangles of every colour group by decreasing longest edge (ties: ascending index),
/// the processing order of DenseStrategy::build (file comment: largest, i.e. costliest, rows
/// first for the dynamic schedule). The order does not affect the matrix entries.
void order_by_cost(const geometry::TriangleMesh& mesh, std::vector<std::vector<Index>>& groups) {
    const Index F = mesh.num_triangles();
    std::vector<Real> edge(static_cast<std::size_t>(F));
    for (Index t = 0; t < F; ++t) {
        const auto v = [&](Eigen::Index i) -> Vec3 {
            return mesh.vertices().row(mesh.triangles()(t, i)).transpose();
        };
        edge[static_cast<std::size_t>(t)] =
            std::max({(v(1) - v(0)).squaredNorm(), (v(2) - v(1)).squaredNorm(),
                      (v(0) - v(2)).squaredNorm()});
    }
    for (std::vector<Index>& g : groups) {
        std::stable_sort(g.begin(), g.end(), [&](Index a, Index b) {
            return edge[static_cast<std::size_t>(a)] > edge[static_cast<std::size_t>(b)];
        });
    }
}

/// Slot of basis n in support(t) (-1 if not supported).
int slot_of(const basis::RwgSpace::Support& sup, Index n) {
    for (int a = 0; a < sup.count; ++a) {
        if (sup.n[a] == n)
            return a;
    }
    return -1;
}

/// Records the first exception thrown inside an OpenMP region (exceptions must not leave it).
class ExceptionSlot {
public:
    void capture() {
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp critical(specklebem_assembler_exception)
#endif
        {
            if (!ptr_)
                ptr_ = std::current_exception();
        }
    }
    void rethrow() const {
        if (ptr_)
            std::rethrow_exception(ptr_);
    }

private:
    std::exception_ptr ptr_;
};

/// Colour groups of the test triangles in processing order (colour_triangles, order_by_cost):
/// the test-triangle schedule shared by DenseStrategy::build and assemble_sparse.
std::vector<std::vector<Index>> test_schedule(const basis::RwgSpace& space) {
    std::vector<int> colour_of;
    std::vector<std::vector<Index>> groups = colour_triangles(space, colour_of);
    order_by_cost(space.mesh(), groups);
    return groups;
}

/// Calls body(t) for every test triangle t: colour by colour, within a colour OpenMP
/// schedule(dynamic, 1) in group order (file comment). body(t) may write only the rows of the
/// basis functions of t. The first exception is rethrown after the colour.
template <class Body>
void for_each_test_triangle(const std::vector<std::vector<Index>>& groups, const Body& body) {
    ExceptionSlot error;
    for (const std::vector<Index>& group : groups) {
        const auto ng = static_cast<Index>(group.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (Index g = 0; g < ng; ++g) {
            try {
                body(group[static_cast<std::size_t>(g)]);
            } catch (...) {
                error.capture();
            }
        }
        error.rethrow();
    }
}

/// Throws std::invalid_argument unless `pat` is a consistent N x N basis-pair pattern.
void check_pattern(const BasisPattern& pat, Index N) {
    const auto fail = [](const std::string& what) {
        throw std::invalid_argument("assemble_sparse: invalid BasisPattern: " + what);
    };
    if (static_cast<Index>(pat.group_of_row.size()) != N)
        fail("group_of_row has " + std::to_string(pat.group_of_row.size()) +
             " entries, N = " + std::to_string(N));
    if (pat.col_ptr.empty() || pat.col_ptr.front() != 0)
        fail("col_ptr must start with 0");
    const auto G = static_cast<Index>(pat.col_ptr.size()) - 1;
    if (pat.col_ptr.back() != static_cast<Index>(pat.cols.size()))
        fail("col_ptr.back() differs from cols.size()");
    for (const Index g : pat.group_of_row) {
        if (g < 0 || g >= G)
            fail("group index " + std::to_string(g) + " outside [0, " + std::to_string(G) + ")");
    }
    // Check all of col_ptr before reading cols (a malformed col_ptr must not index past cols).
    const auto ncols = static_cast<Index>(pat.cols.size());
    for (std::size_t g = 0; g + 1 < pat.col_ptr.size(); ++g) {
        if (pat.col_ptr[g + 1] < pat.col_ptr[g])
            fail("col_ptr decreases");
        if (pat.col_ptr[g + 1] > ncols)
            fail("col_ptr exceeds cols.size()");
    }
    for (std::size_t g = 0; g + 1 < pat.col_ptr.size(); ++g) {
        const Index b = pat.col_ptr[g];
        const Index e = pat.col_ptr[g + 1];
        for (Index k = b; k < e; ++k) {
            const Index c = pat.cols[static_cast<std::size_t>(k)];
            if (c < 0 || c >= N)
                fail("column " + std::to_string(c) + " outside [0, N)");
            if (k > b && c <= pat.cols[static_cast<std::size_t>(k - 1)])
                fail("columns of a group are not strictly ascending");
        }
    }
}

}  // namespace

void validate(const Problem& p) {
    if (p.space == nullptr) {
        throw std::invalid_argument("Problem: space is null");
    }
    if (p.formulation == nullptr) {
        throw std::invalid_argument("Problem: formulation is null");
    }
    if (!(std::isfinite(p.omega) && p.omega > 0.0)) {
        throw std::invalid_argument("Problem: omega must be finite and > 0");
    }
    const auto valid = [](Complex z) {
        return std::isfinite(z.real()) && std::isfinite(z.imag()) && z != Complex(0.0, 0.0);
    };
    if (!valid(p.exterior.eps_r) || !valid(p.exterior.mu_r)) {
        throw std::invalid_argument("Problem: exterior eps_r and mu_r must be finite and non-zero");
    }
    if (!valid(p.object.eps_r) || !valid(p.object.mu_r)) {
        throw std::invalid_argument("Problem: object eps_r and mu_r must be finite and non-zero");
    }
    if (p.excitation != nullptr) {
        const Real w = p.excitation->omega();
        if (!(std::abs(w - p.omega) <= kRelTol * p.omega)) {
            throw std::invalid_argument("Problem: omega differs from the excitation's omega");
        }
        const material::Material& bg = p.excitation->background();
        if (!close(bg.eps_r, p.exterior.eps_r) || !close(bg.mu_r, p.exterior.mu_r)) {
            throw std::invalid_argument(
                "Problem: the excitation's background differs from the exterior material");
        }
    }
    kernels::validate(p.kernel_options);
}

std::shared_ptr<LinearOperator> DenseStrategy::build(const Problem& p) const {
    validate(p);
    const basis::RwgSpace& space = *p.space;
    const Index N = space.size();
    if (2 * N > kMaxDenseUnknowns) {
        throw std::invalid_argument(
            "DenseStrategy: 2N = " + std::to_string(2 * N) +
            " unknowns exceed the dense limit of " + std::to_string(kMaxDenseUnknowns) +
            " (the matrix alone would need 16 (2N)^2 bytes = 160 GB at the limit); use a "
            "compressed operator");
    }
    const Index F = space.mesh().num_triangles();
    const Setup setup = make_setup(p);
    const std::vector<std::vector<Index>> groups = test_schedule(space);
    const Real bytes = 16.0 * static_cast<Real>(2 * N) * static_cast<Real>(2 * N);
    SBEM_INFO("dense assembly: N = {}, 2N = {}, {} triangles, {} colours, matrix {:.3f} GB", N,
              2 * N, F, groups.size(), bytes * 1e-9);
    const ScopedTimer timer("dense assembly (2N = " + std::to_string(2 * N) + ")");

    MatrixXc Z = MatrixXc::Zero(2 * N, 2 * N);
    Complex* const data = Z.data();
    const Index ld = Z.outerStride();
    for_each_test_triangle(groups, [&](Index t) {
        const basis::RwgSpace::Support st = space.support(t);
        if (st.count == 0)
            return;
        PairBlocks blk;
        for (Index s = 0; s < F; ++s) {
            const basis::RwgSpace::Support ss = space.support(s);
            if (ss.count == 0)
                continue;
            pair_blocks(space, t, s, setup, p.kernel_options, blk);
            for (int b = 0; b < ss.count; ++b) {
                // Column-major storage: columns n and N + n, rows m and N + m.
                Complex* col_j = data + ss.n[b] * ld;
                Complex* col_m = data + (N + ss.n[b]) * ld;
                for (int a = 0; a < st.count; ++a) {
                    const Index m = st.n[a];
                    const auto k = static_cast<std::size_t>(3 * a + b);
                    col_j[m] += blk.jj[k];
                    col_m[m] += blk.jm[k];
                    col_j[N + m] += blk.mj[k];
                    col_m[N + m] += blk.mm[k];
                }
            }
        }
    });
    return std::make_shared<DenseOperator>(std::move(Z));
}

std::shared_ptr<SparseOperator> assemble_sparse(const Problem& p, const BasisPattern& pattern) {
    validate(p);
    const basis::RwgSpace& space = *p.space;
    const Index N = space.size();
    check_pattern(pattern, N);
    const Setup setup = make_setup(p);
    const std::vector<std::vector<Index>> groups = test_schedule(space);

    // CSR layout: row m (< N) = [cols(m), N + cols(m)], row N + m the same; the J rows first.
    // begin[m] = offset of row m among the J rows (twice the pattern prefix sum).
    const auto row_cols = [&](Index m) {
        const auto g = static_cast<std::size_t>(pattern.group_of_row[static_cast<std::size_t>(m)]);
        return std::pair<const Index*, const Index*>(pattern.cols.data() + pattern.col_ptr[g],
                                                     pattern.cols.data() + pattern.col_ptr[g + 1]);
    };
    // Sizes in std::size_t from a clamped value (GCC 13 -O3 -Wnull-dereference false positive).
    const auto rows = static_cast<std::size_t>(std::max<Index>(N, 0));
    std::vector<Index> begin(rows + 1, 0);
    for (Index m = 0; m < N; ++m) {
        const auto [cb, ce] = row_cols(m);
        begin[static_cast<std::size_t>(m) + 1] = begin[static_cast<std::size_t>(m)] + 2 * (ce - cb);
    }
    const Index half = begin[static_cast<std::size_t>(N)];  // nnz of the J rows
    SparseOperator::Matrix Z(2 * N, 2 * N);
    Z.resizeNonZeros(2 * half);
    Index* const outer = Z.outerIndexPtr();
    Index* const inner = Z.innerIndexPtr();
    Complex* const val = Z.valuePtr();
    for (Index m = 0; m < N; ++m) {
        const auto [cb, ce] = row_cols(m);
        const Index len = ce - cb;
        const Index o = begin[static_cast<std::size_t>(m)];
        outer[m] = o;
        outer[N + m] = half + o;
        for (Index j = 0; j < len; ++j) {
            inner[o + j] = inner[half + o + j] = cb[j];
            inner[o + len + j] = inner[half + o + len + j] = N + cb[j];
        }
    }
    outer[2 * N] = 2 * half;
    std::fill(val, val + 2 * half, Complex(0.0, 0.0));
    const Real bytes =
        static_cast<Real>(2 * half) * static_cast<Real>(sizeof(Complex) + sizeof(Index));
    SBEM_INFO("sparse assembly: N = {}, 2N = {}, nnz = {} ({:.3g} per row), matrix {:.3f} MB", N,
              2 * N, 2 * half, N > 0 ? static_cast<Real>(half) / static_cast<Real>(N) : 0.0,
              bytes * 1e-6);
    const ScopedTimer timer("sparse assembly (2N = " + std::to_string(2 * N) +
                            ", nnz = " + std::to_string(2 * half) + ")");

    for_each_test_triangle(groups, [&](Index t) {
        const basis::RwgSpace::Support st = space.support(t);
        if (st.count == 0)
            return;
        // Source triangles of the pattern pairs of t's basis functions, ascending (the dense
        // order: every stored entry gets its contributions in the dense summation order).
        std::vector<Index> sources;
        for (int a = 0; a < st.count; ++a) {
            const auto [cb, ce] = row_cols(st.n[a]);
            for (const Index* c = cb; c != ce; ++c) {
                sources.push_back(space.plus_triangle(*c));
                sources.push_back(space.minus_triangle(*c));
            }
        }
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        PairBlocks blk;
        for (const Index s : sources) {
            const basis::RwgSpace::Support ss = space.support(s);
            pair_blocks(space, t, s, setup, p.kernel_options, blk);
            for (int a = 0; a < st.count; ++a) {
                const Index m = st.n[a];
                const auto [cb, ce] = row_cols(m);
                const Index len = ce - cb;
                Complex* const row_j = val + begin[static_cast<std::size_t>(m)];
                Complex* const row_m = row_j + half;
                for (int b = 0; b < ss.count; ++b) {
                    const Index* it = std::lower_bound(cb, ce, ss.n[b]);
                    if (it == ce || *it != ss.n[b])
                        continue;  // far pair
                    const Index j = it - cb;
                    const auto k = static_cast<std::size_t>(3 * a + b);
                    row_j[j] += blk.jj[k];
                    row_j[len + j] += blk.jm[k];
                    row_m[j] += blk.mj[k];
                    row_m[len + j] += blk.mm[k];
                }
            }
        }
    });
    return std::make_shared<SparseOperator>(std::move(Z));
}

std::shared_ptr<RegionSparseOperator> assemble_region_sparse(const Problem& p, int region,
                                                             std::vector<Index> row_ptr,
                                                             std::vector<Index> cols) {
    validate(p);
    if (region != 0 && region != 1)
        throw std::invalid_argument("assemble_region_sparse: region must be 0 or 1");
    const basis::RwgSpace& space = *p.space;
    const Index N = space.size();
    const auto fail = [](const std::string& what) {
        throw std::invalid_argument("assemble_region_sparse: invalid pattern: " + what);
    };
    const auto rows = static_cast<std::size_t>(std::max<Index>(N, 0));
    if (row_ptr.size() != rows + 1 || row_ptr.front() != 0 ||
        row_ptr.back() != static_cast<Index>(cols.size()))
        fail("row_ptr must have N + 1 entries from 0 to cols.size()");
    for (std::size_t r = 0; r < rows; ++r) {
        if (row_ptr[r + 1] < row_ptr[r])
            fail("row_ptr decreases");
        for (Index k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            const Index c = cols[static_cast<std::size_t>(k)];
            if (c < 0 || c >= N || (k > row_ptr[r] && c <= cols[static_cast<std::size_t>(k - 1)]))
                fail("columns out of range or not strictly ascending");
        }
    }
    const Setup setup = make_setup(p);
    const auto ri = static_cast<std::size_t>(region);
    std::vector<Complex> values(2 * cols.size(), Complex(0.0, 0.0));
    SBEM_INFO("region sparse assembly: region R{}, N = {}, {} basis pairs, {:.3f} MB", region + 1,
              N, cols.size(),
              static_cast<Real>(cols.size() * RegionSparseOperator::kBytesPerPair) * 1e-6);
    if (setup.active[ri] && !cols.empty()) {
        const ScopedTimer timer("region sparse assembly (R" + std::to_string(region + 1) +
                                ", pairs = " + std::to_string(cols.size()) + ")");
        const std::vector<std::vector<Index>> groups = test_schedule(space);
        const Real jump = kJumpSign[ri];
        for_each_test_triangle(groups, [&](Index t) {
            const basis::RwgSpace::Support st = space.support(t);
            if (st.count == 0)
                return;
            std::vector<Index> sources;
            for (int a = 0; a < st.count; ++a) {
                const auto m = static_cast<std::size_t>(st.n[a]);
                for (Index k = row_ptr[m]; k < row_ptr[m + 1]; ++k) {
                    const Index c = cols[static_cast<std::size_t>(k)];
                    sources.push_back(space.plus_triangle(c));
                    sources.push_back(space.minus_triangle(c));
                }
            }
            std::sort(sources.begin(), sources.end());
            sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
            Block L;
            Block K;
            Block I;
            for (const Index s : sources) {
                const basis::RwgSpace::Support ss = space.support(s);
                kernels::element_blocks(space, t, s, setup.region[ri], p.kernel_options, L, K);
                if (s == t) {
                    kernels::jump_block(space, t, I);
                    K += jump * I;
                }
                for (int a = 0; a < st.count; ++a) {
                    const auto m = static_cast<std::size_t>(st.n[a]);
                    const Index* const cb = cols.data() + row_ptr[m];
                    const Index* const ce = cols.data() + row_ptr[m + 1];
                    for (int b = 0; b < ss.count; ++b) {
                        const Index* it = std::lower_bound(cb, ce, ss.n[b]);
                        if (it == ce || *it != ss.n[b])
                            continue;
                        const auto k = static_cast<std::size_t>(it - cols.data());
                        values[2 * k] += L(a, b);
                        values[2 * k + 1] += K(a, b);
                    }
                }
            }
        });
    }
    return std::make_shared<RegionSparseOperator>(N, std::move(row_ptr), std::move(cols),
                                                  std::move(values), setup.e[ri], setup.h[ri],
                                                  setup.m[ri]);
}

VectorXc assemble_rhs(const Problem& p) {
    validate(p);
    if (p.excitation == nullptr) {
        throw std::invalid_argument("assemble_rhs: Problem has no excitation");
    }
    const basis::RwgSpace& space = *p.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const excitation::Excitation& exc = *p.excitation;
    const Index N = space.size();
    const Index F = mesh.num_triangles();
    const Setup setup = make_setup(p);
    // kernels::validate (in validate) guarantees a positive-interior rule.
    const kernels::TriangleRule& rule = kernels::triangle_rule(p.kernel_options.quad_degree_rhs);

    // Per-triangle moments <f_a, E_inc>_t and <f_a, H_inc>_t (slot a of support(t)), computed in
    // parallel (each triangle writes its own row), then scattered in triangle order.
    Eigen::Matrix<Complex, Eigen::Dynamic, 3, Eigen::RowMajor> me =
        Eigen::Matrix<Complex, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(F, 3);
    Eigen::Matrix<Complex, Eigen::Dynamic, 3, Eigen::RowMajor> mh = me;
    ExceptionSlot error;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Index t = 0; t < F; ++t) {
        try {
            const basis::RwgSpace::Support sup = space.support(t);
            if (sup.count == 0)
                continue;
            const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
            const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
            const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
            const Real area = mesh.area(t);
            for (std::size_t q = 0; q < rule.weights.size(); ++q) {
                const Vec3& l = rule.barycentric[q];
                const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
                const Real w = rule.weights[q] * area;
                const Vec3c E = exc.electric_field(r);
                const Vec3c H = exc.magnetic_field(r);
                for (int a = 0; a < sup.count; ++a) {
                    const Vec3 f = space.value(sup.n[a], t, r);
                    // f is real: plain (non-conjugating) dot product.
                    me(t, a) += w * (f(0) * E(0) + f(1) * E(1) + f(2) * E(2));
                    mh(t, a) += w * (f(0) * H(0) + f(1) * H(1) + f(2) * H(2));
                }
            }
        } catch (...) {
            error.capture();
        }
    }
    error.rethrow();

    VectorXc b = VectorXc::Zero(2 * N);
    for (Index t = 0; t < F; ++t) {
        const basis::RwgSpace::Support sup = space.support(t);
        for (int a = 0; a < sup.count; ++a) {
            b(sup.n[a]) += me(t, a);
            b(N + sup.n[a]) += mh(t, a);
        }
    }
    b.head(N) *= setup.e[0];
    b.tail(N) *= setup.h[0];
    if (!b.allFinite()) {
        throw std::invalid_argument("assemble_rhs: non-finite incident field");
    }
    return b;
}

VectorXc assemble_diagonal(const Problem& p) {
    validate(p);
    const basis::RwgSpace& space = *p.space;
    const Index N = space.size();
    const Setup setup = make_setup(p);
    // Same summation order as DenseStrategy::build (test triangles by colour, then index;
    // source triangles by index), so the result matches Z.diagonal() to the last bit in
    // practice. The jump terms enter only the off-diagonal K blocks (and I_mm = 0).
    std::vector<int> colour_of;
    (void)colour_triangles(space, colour_of);
    const auto before = [&](Index t1, Index t2) {
        const int c1 = colour_of[static_cast<std::size_t>(t1)];
        const int c2 = colour_of[static_cast<std::size_t>(t2)];
        return c1 != c2 ? c1 < c2 : t1 < t2;
    };

    VectorXc d = VectorXc::Zero(2 * N);
    ExceptionSlot error;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Index m = 0; m < N; ++m) {
        try {
            const Index tp = space.plus_triangle(m);
            const Index tm = space.minus_triangle(m);
            const std::array<Index, 2> tests =
                before(tp, tm) ? std::array<Index, 2>{tp, tm} : std::array<Index, 2>{tm, tp};
            const std::array<Index, 2> sources = {std::min(tp, tm), std::max(tp, tm)};
            PairBlocks blk;
            for (const Index t : tests) {
                const int a = slot_of(space.support(t), m);
                for (const Index s : sources) {
                    const int b = slot_of(space.support(s), m);
                    pair_blocks(space, t, s, setup, p.kernel_options, blk);
                    const auto k = static_cast<std::size_t>(3 * a + b);
                    d(m) += blk.jj[k];
                    d(N + m) += blk.mm[k];
                }
            }
        } catch (...) {
            error.capture();
        }
    }
    error.rethrow();
    return d;
}

}  // namespace specklebem::op
