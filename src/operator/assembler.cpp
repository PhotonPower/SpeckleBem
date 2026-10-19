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
/// Stretched pairs (PML, WP-PML2c). With a non-identity Problem::stretch the blocks of a pair
/// with a stretched triangle cost 1e5-1e7 kernel evaluations, and kernels::element_blocks
/// computes them for the canonical ordering and transposes for the other: B(s, t) = B(t, s)^T
/// bitwise for each of the combined blocks jj, jm, mj, mm (L and K^PV are complex symmetric; the
/// jump term enters identical pairs only). Write Z_X (X one of the four N x N sub-blocks) as
/// U_X + D_X + C_X, the contributions of the unstretched pairs and the stretched identical pairs
/// (U + D), of the evaluated ordering of each unordered stretched pair {t, s}, t != s (A), and of
/// the other ordering (C). C_X(m, n) = sum over the other orderings (t, s) of B_X(t, s)(a, b) =
/// sum over the evaluated (s, t) of B_X(s, t)(b, a) = A_X(n, m), so Z_X = A_X + A_X^T + (U + D)_X:
///  pass 1   the schedule above over the evaluated orderings only (each unordered stretched pair
///           once, half of each row's pairs, see evaluated_ordering), giving A in Z;
///  transpose Z_X <- Z_X + Z_X^T per sub-block (add_block_transposes: one writer per entry);
///  pass 2   the schedule above over the unstretched pairs and the stretched identical pairs.
/// Every pass is deterministic and race-free as above, so the matrix does not depend on the
/// thread count. Entries whose two basis functions have no stretched support triangle receive
/// only pass-2 contributions, in the order without a stretch (bitwise the unstretched entry); the
/// others differ from a pair-by-pair sum in the summation order only (tested to 1e-13 of the sum
/// of |contributions|). Without a stretch (or with the identity) the loop above is unchanged.
/// assemble_diagonal follows the same order.
///
/// Sparse assembly (WP19a, assemble_sparse) shares this schedule (test_schedule,
/// for_each_test_triangle) and pair_blocks; per test triangle it visits only the source
/// triangles of its pattern pairs, in ascending order, and scatters only the pattern entries.
/// A stored entry thus receives exactly the dense contributions in the dense order: bitwise
/// equal to the DenseStrategy entry. With a stretch (assemble_sparse_stretched, WP-PML5c) the
/// blocks of the stretched pairs come from a cache of canonical blocks (each needed unordered
/// pair {t, s} evaluated once as (min, max) in a parallel pass before the scatter; the ordering
/// (t, s) with t > s takes the transpose, bitwise the block pair_blocks gives for it, see
/// "Stretched pairs"); the scatter order is unchanged, so the result is bitwise that of the
/// per-ordering evaluation of WP-PML5a. The default also evaluates the touching pairs of the PML
/// strips once per translation class (translation_representatives, keyed by the kernels layer:
/// kernels::stretched_touching_key; bitwise, see assembler.hpp).
#include "specklebem/operator/assembler.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/region_sparse_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
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

/// Region parameters and the scalar factors of the region blocks in Z (BlockWeights).
using Setup = BlockWeights;

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

/// Combined local blocks of one triangle pair (TrianglePairBlocks: slots as in
/// kernels::element_blocks, row-major 3 x 3 arrays, entry (a, b) at index 3 a + b). Plain arrays
/// and scalar loops keep the per-pair overhead small also in the unoptimised sanitizer build.
using PairBlocks = TrianglePairBlocks;

/// Jump coefficient of K_i in the Galerkin tangential equations: K_1 = K_1^PV - 1/2 I,
/// K_2 = K_2^PV + 1/2 I (file comment).
constexpr std::array<Real, 2> kJumpSign = {-0.5, 0.5};

void pair_blocks(const basis::RwgSpace& space, Index t, Index s, const Setup& setup,
                 const kernels::OperatorOptions& opt, const geometry::CoordinateStretch* stretch,
                 PairBlocks& out) {
    out.jj.fill(Complex(0.0, 0.0));
    out.jm.fill(Complex(0.0, 0.0));
    out.mj.fill(Complex(0.0, 0.0));
    out.mm.fill(Complex(0.0, 0.0));
    std::array<Block, 2> Ls;
    std::array<Block, 2> Ks;
    Block I;
    const bool identical = t == s;
    if (identical) {
        kernels::jump_block(space, t, I);
    }
    // Both regions in one call (bitwise the per-region element_blocks; a stretched pair shares
    // its geometry and, where the rules agree, its point loop between the regions).
    kernels::element_blocks_both_regions(space, t, s, setup.region, setup.active, opt, stretch, Ls,
                                         Ks);
    for (std::size_t i = 0; i < 2; ++i) {
        if (!setup.active[i])
            continue;
        const Block& L = Ls[i];
        const Block& K = Ks[i];
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

/// Per triangle: 1 if it takes the stretched path of `stretch` (kernels::is_stretched_triangle).
std::vector<char> stretched_triangles(const geometry::TriangleMesh& mesh,
                                      const geometry::CoordinateStretch& stretch) {
    std::vector<char> out(static_cast<std::size_t>(mesh.num_triangles()), 0);
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        out[static_cast<std::size_t>(t)] = kernels::is_stretched_triangle(stretch, mesh, t) ? 1 : 0;
    }
    return out;
}

std::vector<char> stretched_triangles(const Problem& p) {
    return stretched_triangles(p.space->mesh(), *p.stretch);
}

/// The ordering (t, s) of an unordered pair {t, s} of distinct triangles that pass 1 of the PML
/// assembly evaluates (exactly one of (t, s) and (s, t); file comment "Stretched pairs"): t < s
/// for even t + s, t > s for odd t + s, so that every test triangle evaluates about half of its
/// pairs (the plain (min, max) choice would give the first rows all and the last none).
bool evaluated_ordering(Index t, Index s) {
    return ((t + s) % 2 == 0) == (t < s);
}

/// Z_X <- Z_X + Z_X^T for each of the four N x N sub-blocks X of the 2N x 2N matrix Z (file
/// comment "Stretched pairs"); both entries of a pair get the same sum (a + b = b + a). Tiles of
/// the upper sub-block triangle are distributed over the threads; each entry is written by one
/// thread, so the result does not depend on the thread count.
void add_block_transposes(MatrixXc& Z, Index N) {
    constexpr Index kTile = 64;
    const Index nt = (N + kTile - 1) / kTile;
    for (Index bi = 0; bi < 2; ++bi) {
        for (Index bj = 0; bj < 2; ++bj) {
            const Index r0 = bi * N;
            const Index c0 = bj * N;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (Index I = 0; I < nt; ++I) {
                for (Index J = I; J < nt; ++J) {
                    const Index m_end = std::min(N, (I + 1) * kTile);
                    const Index n_end = std::min(N, (J + 1) * kTile);
                    for (Index n = J * kTile; n < n_end; ++n) {
                        for (Index m = I * kTile; m < std::min(m_end, n + 1); ++m) {
                            if (m == n) {
                                Z(r0 + m, c0 + m) = Z(r0 + m, c0 + m) + Z(r0 + m, c0 + m);
                            } else {
                                const Complex v = Z(r0 + m, c0 + n) + Z(r0 + n, c0 + m);
                                Z(r0 + m, c0 + n) = v;
                                Z(r0 + n, c0 + m) = v;
                            }
                        }
                    }
                }
            }
        }
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

bool has_stretch(const Problem& p) {
    return p.stretch != nullptr && !p.stretch->is_identity();
}

void require_unstretched(const Problem& p, const std::string& who) {
    if (has_stretch(p)) {
        throw std::invalid_argument(
            who +
            ": a non-identity PML coordinate stretch (Problem::stretch, ADR 0009) is supported "
            "only by DenseStrategy, assemble_diagonal, assemble_rhs, assemble_sparse_stretched, "
            "mlfmm::MlfmmOperator (PML ring) and post::far_field with a current window");
    }
}

std::vector<std::uint8_t> pml_masked_functions(const basis::RwgSpace& space,
                                               const geometry::CoordinateStretch& stretch) {
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index F = mesh.num_triangles();
    const Vertices& V = mesh.vertices();
    const Triangles& T = mesh.triangles();
    // Triangles with a vertex strictly inside the stretched zone.
    std::vector<std::uint8_t> tri_pml(static_cast<std::size_t>(F), 0);
    for (Index t = 0; t < F; ++t) {
        for (Index c = 0; c < 3; ++c) {
            if (stretch.in_pml(V.row(T(t, c)).transpose())) {
                tri_pml[static_cast<std::size_t>(t)] = 1;
                break;
            }
        }
    }
    const Index N = space.size();
    std::vector<std::uint8_t> masked(static_cast<std::size_t>(N), 0);
    for (Index n = 0; n < N; ++n) {
        const bool m = tri_pml[static_cast<std::size_t>(space.plus_triangle(n))] != 0 ||
                       tri_pml[static_cast<std::size_t>(space.minus_triangle(n))] != 0;
        masked[static_cast<std::size_t>(n)] = m ? 1 : 0;
    }
    return masked;
}

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
    // Adds the combined blocks of the pair (t, s) to the rows of t's basis functions.
    const auto scatter = [&](const basis::RwgSpace::Support& st, const basis::RwgSpace::Support& ss,
                             const PairBlocks& blk) {
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
    };
    if (!has_stretch(p)) {
        for_each_test_triangle(groups, [&](Index t) {
            const basis::RwgSpace::Support st = space.support(t);
            if (st.count == 0)
                return;
            PairBlocks blk;
            for (Index s = 0; s < F; ++s) {
                const basis::RwgSpace::Support ss = space.support(s);
                if (ss.count == 0)
                    continue;
                pair_blocks(space, t, s, setup, p.kernel_options, p.stretch, blk);
                scatter(st, ss, blk);
            }
        });
        return std::make_shared<DenseOperator>(std::move(Z));
    }
    // PML (file comment "Stretched pairs"): pass 1 the evaluated ordering of every unordered
    // stretched pair of distinct triangles, the sub-block transposes, pass 2 the rest.
    const std::vector<char> str = stretched_triangles(p);
    const auto stretched_pair = [&](Index t, Index s) {
        return str[static_cast<std::size_t>(t)] != 0 || str[static_cast<std::size_t>(s)] != 0;
    };
    for_each_test_triangle(groups, [&](Index t) {
        const basis::RwgSpace::Support st = space.support(t);
        if (st.count == 0)
            return;
        PairBlocks blk;
        for (Index s = 0; s < F; ++s) {
            if (s == t || !stretched_pair(t, s) || !evaluated_ordering(t, s))
                continue;
            const basis::RwgSpace::Support ss = space.support(s);
            if (ss.count == 0)
                continue;
            pair_blocks(space, t, s, setup, p.kernel_options, p.stretch, blk);
            scatter(st, ss, blk);
        }
    });
    add_block_transposes(Z, N);
    for_each_test_triangle(groups, [&](Index t) {
        const basis::RwgSpace::Support st = space.support(t);
        if (st.count == 0)
            return;
        PairBlocks blk;
        for (Index s = 0; s < F; ++s) {
            if (s != t && stretched_pair(t, s))
                continue;
            const basis::RwgSpace::Support ss = space.support(s);
            if (ss.count == 0)
                continue;
            pair_blocks(space, t, s, setup, p.kernel_options, p.stretch, blk);
            scatter(st, ss, blk);
        }
    });
    return std::make_shared<DenseOperator>(std::move(Z));
}

namespace {

/// Columns of row m of a (checked) pattern.
std::pair<const Index*, const Index*> pattern_row(const BasisPattern& pattern, Index m) {
    const auto g = static_cast<std::size_t>(pattern.group_of_row[static_cast<std::size_t>(m)]);
    return {pattern.cols.data() + pattern.col_ptr[g], pattern.cols.data() + pattern.col_ptr[g + 1]};
}

/// Source triangles of the pattern pairs of t's basis functions, ascending and unique (the
/// dense order: every stored entry gets its contributions in the dense summation order).
void pattern_sources(const basis::RwgSpace& space, const BasisPattern& pattern,
                     const basis::RwgSpace::Support& st, std::vector<Index>& sources) {
    sources.clear();
    for (int a = 0; a < st.count; ++a) {
        const auto [cb, ce] = pattern_row(pattern, st.n[a]);
        for (const Index* c = cb; c != ce; ++c) {
            sources.push_back(space.plus_triangle(*c));
            sources.push_back(space.minus_triangle(*c));
        }
    }
    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
}

/// True if triangles t and s share a vertex index (identical, shared edge or shared vertex).
bool shares_vertex(const Triangles& T, Index t, Index s) {
    for (Eigen::Index i = 0; i < 3; ++i) {
        for (Eigen::Index j = 0; j < 3; ++j) {
            if (T(t, i) == T(s, j))
                return true;
        }
    }
    return false;
}

/// The blocks of (t, s) from the canonical block c of (min(t, s), max(t, s)): c itself for
/// t <= s, its transpose (entry (a, b) = c(b, a)) for t > s. Bitwise the blocks pair_blocks gives
/// for (t, s) when the pair is stretched (assemble_sparse_stretched, "Stretched pairs").
void oriented_blocks(const PairBlocks& c, bool transpose, PairBlocks& out) {
    if (!transpose) {
        out = c;
        return;
    }
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
            out.jj[3 * a + b] = c.jj[3 * b + a];
            out.jm[3 * a + b] = c.jm[3 * b + a];
            out.mj[3 * a + b] = c.mj[3 * b + a];
            out.mm[3 * a + b] = c.mm[3 * b + a];
        }
    }
}

/// Canonical blocks of the stretched triangle pairs of a pattern (WP-PML5c): per triangle a the
/// partners b >= a of the unordered stretched pairs {a, b} the pattern needs (in some ordering),
/// ascending, and per pair the blocks of the ordering (a, b).
struct CanonicalPairCache {
    std::vector<Index> ptr;      ///< size F + 1: partners of a in partner[ptr[a], ptr[a + 1])
    std::vector<Index> partner;  ///< ascending per a
    std::vector<PairBlocks> blocks;

    /// Slot of the unordered pair {t, s}; -1 if it is not cached.
    [[nodiscard]] Index find(Index t, Index s) const {
        const Index a = std::min(t, s);
        const Index b = std::max(t, s);
        const Index* pb = partner.data() + ptr[static_cast<std::size_t>(a)];
        const Index* pe = partner.data() + ptr[static_cast<std::size_t>(a) + 1];
        const Index* it = std::lower_bound(pb, pe, b);
        return it != pe && *it == b ? static_cast<Index>(it - partner.data()) : -1;
    }
};

/// Index of the canonical-pair cache (WP-PML5c): per triangle a the partners b >= a of the
/// unordered stretched pairs {a, b} that `pattern` needs (in some ordering), ascending; `first`
/// holds the a of every slot. `str`: per triangle, 1 if stretched.
void canonical_index(const basis::RwgSpace& space, const BasisPattern& pattern,
                     const std::vector<char>& str, std::vector<Index>& ptr,
                     std::vector<Index>& partner, std::vector<Index>& first) {
    const Index F = space.mesh().num_triangles();
    const auto stretched_pair = [&](Index t, Index s) {
        return str[static_cast<std::size_t>(t)] != 0 || str[static_cast<std::size_t>(s)] != 0;
    };
    // Partners b >= a of every needed stretched pair {a, b} (from the sources of each test
    // triangle; a pattern needs (t, s) iff s is a source of t).
    std::vector<std::vector<Index>> partners(static_cast<std::size_t>(F));
    {
        std::vector<Index> sources;
        for (Index t = 0; t < F; ++t) {
            const basis::RwgSpace::Support st = space.support(t);
            if (st.count == 0)
                continue;
            pattern_sources(space, pattern, st, sources);
            for (const Index s : sources) {
                if (stretched_pair(t, s))
                    partners[static_cast<std::size_t>(std::min(t, s))].push_back(std::max(t, s));
            }
        }
    }
    ptr.assign(static_cast<std::size_t>(F) + 1, 0);
    for (Index a = 0; a < F; ++a) {
        std::vector<Index>& v = partners[static_cast<std::size_t>(a)];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        ptr[static_cast<std::size_t>(a) + 1] =
            ptr[static_cast<std::size_t>(a)] + static_cast<Index>(v.size());
    }
    const Index npairs = ptr.back();
    partner.resize(static_cast<std::size_t>(npairs));
    first.resize(static_cast<std::size_t>(npairs));
    for (Index a = 0; a < F; ++a) {
        std::vector<Index>& v = partners[static_cast<std::size_t>(a)];
        const Index o = ptr[static_cast<std::size_t>(a)];
        for (std::size_t i = 0; i < v.size(); ++i) {
            partner[static_cast<std::size_t>(o) + i] = v[i];
            first[static_cast<std::size_t>(o) + i] = a;
        }
        std::vector<Index>().swap(v);
    }
}

/// Free-vertex slot map of a translation class member (kernels::StretchedTouchingKey::slot_vertex
/// of the canonical ordering, as int8).
using SlotMap = std::array<std::array<std::int8_t, 3>, 2>;

/// Bytes of a std::map<std::vector<std::uint64_t>, Index> node with a key of `words` words
/// (allocation of the node and of the key vector; a model, libstdc++ / libc++ within ~16 bytes).
constexpr std::size_t kKeyNodeBytes = 48 + sizeof(std::vector<std::uint64_t>) + sizeof(Index);

/// Transient bytes of the canonical-pair cache of assemble_sparse_stretched (SparseStretchedStats::
/// cache_bytes, estimate_sparse_stretched_cache): blocks, index (ptr, partner, first), processing
/// order and, for translation classes, the representatives, slot maps, the touching list and the
/// key map (`classes` keys of at most `key_words` words).
std::size_t cache_bytes_model(Index F, Index npairs, Index touching, Index classes,
                              std::size_t key_words, bool with_classes) {
    const auto n = static_cast<std::size_t>(npairs);
    std::size_t b = n * sizeof(PairBlocks) + (static_cast<std::size_t>(F) + 1) * sizeof(Index) +
                    3 * n * sizeof(Index);  // blocks, ptr, partner, first, order
    if (with_classes) {
        b +=
            n * (sizeof(Index) + sizeof(SlotMap)) +
            static_cast<std::size_t>(touching) * sizeof(Index) +
            static_cast<std::size_t>(classes) * (kKeyNodeBytes + key_words * sizeof(std::uint64_t));
    }
    return b;
}

/// Translation classes of the touching stretched pairs (WP-PML5c; assemble_sparse_stretched,
/// "Translation classes"). For every slot k of `touching` (canonical pairs {first[k],
/// partner[k]}, first < partner, sharing a vertex) the slot of its class representative, or -1
/// if k is evaluated itself. The key is kernels::stretched_touching_key (built in the kernels
/// layer from the code paths of the stretched touching rule, bit for bit, including the
/// fastmath decision, the shared-index pattern and the strip margin); pairs without a key
/// (identical, corner, strip boundary) are evaluated themselves. A copy is therefore bitwise the
/// blocks its own evaluation gives (the kernels guarantee). The representative is the first slot
/// of its key (ascending slot order: deterministic). slot_map[k] receives the free-vertex slot
/// map of every keyed slot (translated_blocks).
std::vector<Index> translation_representatives(
    const basis::RwgSpace& space, const Setup& setup, const kernels::OperatorOptions& opt,
    const geometry::CoordinateStretch& stretch, const std::vector<Index>& first,
    const std::vector<Index>& partner, const std::vector<Index>& touching, Index& classes,
    std::vector<SlotMap>& slot_map, std::size_t& key_words) {
    std::vector<Index> rep(first.size(), -1);
    slot_map.assign(first.size(), SlotMap{});
    classes = 0;
    key_words = 0;
    std::map<std::vector<std::uint64_t>, Index> first_of_key;
    kernels::StretchedTouchingKey key;
    for (const Index k : touching) {
        const auto kk = static_cast<std::size_t>(k);
        if (!kernels::stretched_touching_key(space, first[kk], partner[kk], setup.region,
                                             setup.active, opt, stretch, key))
            continue;  // identical, corner or strip-boundary pair: evaluated itself
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t a = 0; a < 3; ++a)
                slot_map[kk][i][a] = static_cast<std::int8_t>(key.slot_vertex[i][a]);
        }
        const auto [it, inserted] = first_of_key.emplace(key.words, k);
        if (inserted) {
            ++classes;
            key_words = std::max(key_words, key.words.size());
        } else {
            rep[kk] = it->second;
        }
    }
    return rep;
}

/// Blocks of the member pair (a, b) from those of its translation representative (ar, br)
/// (translation_representatives): slots mapped by the local index of their free vertex (the
/// kernels' slot_vertex maps of both pairs; the key holds the per-vertex slot presence, so the
/// map is a bijection), an exact copy.
void translated_blocks(const basis::RwgSpace& space, Index a, Index b, const SlotMap& m,
                       const SlotMap& mr, const PairBlocks& rep, PairBlocks& out) {
    const basis::RwgSpace::Support sa = space.support(a), sb = space.support(b);
    const auto map_slot = [&](std::size_t tri, int slot) {
        const std::int8_t j = m[tri][static_cast<std::size_t>(slot)];
        for (std::size_t x = 0; x < 3; ++x) {
            if (mr[tri][x] == j)
                return static_cast<int>(x);
        }
        throw std::logic_error("assemble_sparse: translation class with different supports");
    };
    out.jj.fill(Complex(0.0, 0.0));
    out.jm.fill(Complex(0.0, 0.0));
    out.mj.fill(Complex(0.0, 0.0));
    out.mm.fill(Complex(0.0, 0.0));
    for (int x = 0; x < sa.count; ++x) {
        const int xr = map_slot(0, x);
        for (int y = 0; y < sb.count; ++y) {
            const int yr = map_slot(1, y);
            const auto k = static_cast<std::size_t>(3 * x + y);
            const auto kr = static_cast<std::size_t>(3 * xr + yr);
            out.jj[k] = rep.jj[kr];
            out.jm[k] = rep.jm[kr];
            out.mj[k] = rep.mj[kr];
            out.mm[k] = rep.mm[kr];
        }
    }
}

/// assemble_sparse with the pair blocks of `stretch` (nullptr: the unstretched kernels; the
/// stretched path of element_blocks for pairs with a stretched triangle otherwise). The caller
/// has validated p. `options` (stretch only): the evaluation of the stretched pairs and stats.
std::shared_ptr<SparseOperator> assemble_sparse_impl(const Problem& p, const BasisPattern& pattern,
                                                     const geometry::CoordinateStretch* stretch,
                                                     const SparseStretchedOptions& options = {}) {
    const basis::RwgSpace& space = *p.space;
    const Index N = space.size();
    check_pattern(pattern, N);
    const Setup setup = make_setup(p);
    const std::vector<std::vector<Index>> groups = test_schedule(space);

    // CSR layout: row m (< N) = [cols(m), N + cols(m)], row N + m the same; the J rows first.
    // begin[m] = offset of row m among the J rows (twice the pattern prefix sum).
    const auto row_cols = [&](Index m) { return pattern_row(pattern, m); };
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

    // Stretched pairs (WP-PML5c, assemble_sparse_stretched): the canonical blocks of every needed
    // unordered stretched pair, evaluated once in a parallel pass before the scatter.
    const bool use_cache =
        stretch != nullptr && options.evaluation != StretchedPairEvaluation::per_ordering;
    const bool use_classes =
        stretch != nullptr && options.evaluation == StretchedPairEvaluation::translation_classes;
    const Index F = space.mesh().num_triangles();
    // Per triangle: takes the stretched path.
    const std::vector<char> str =
        stretch != nullptr ? stretched_triangles(space.mesh(), *stretch) : std::vector<char>{};
    const auto stretched_pair = [&](Index t, Index s) {
        return !str.empty() &&
               (str[static_cast<std::size_t>(t)] != 0 || str[static_cast<std::size_t>(s)] != 0);
    };
    SparseStretchedStats stats;
    CanonicalPairCache cache;
    if (use_cache) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Index> first;  // the a of every slot
        canonical_index(space, pattern, str, cache.ptr, cache.partner, first);
        const Index npairs = cache.ptr.back();
        // Processing order: touching pairs (the expensive relative-coordinate rules) first, then
        // the rest, each in slot order; it does not change any block. One shares_vertex test per
        // pair (review S5).
        const Triangles& T = space.mesh().triangles();
        std::vector<Index> order;
        order.reserve(static_cast<std::size_t>(npairs));
        std::vector<Index> rest;
        for (Index k = 0; k < npairs; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            (shares_vertex(T, first[kk], cache.partner[kk]) ? order : rest).push_back(k);
        }
        stats.touching_pairs = static_cast<Index>(order.size());
        order.insert(order.end(), rest.begin(), rest.end());
        std::vector<Index>().swap(rest);
        // Translation classes of the touching pairs (one evaluation per class, the members copied).
        std::vector<Index> rep;
        std::vector<SlotMap> slot_map;
        std::size_t key_words = 0;
        if (use_classes) {
            const std::vector<Index> touching(
                order.begin(), order.begin() + static_cast<std::ptrdiff_t>(stats.touching_pairs));
            rep = translation_representatives(space, setup, p.kernel_options, *stretch, first,
                                              cache.partner, touching, stats.translation_classes,
                                              slot_map, key_words);
            for (const Index r : rep) stats.translated_pairs += r >= 0 ? 1 : 0;
        }
        cache.blocks.resize(static_cast<std::size_t>(npairs));
        stats.canonical_pairs = npairs;
        stats.cache_bytes = cache_bytes_model(F, npairs, stats.touching_pairs,
                                              stats.translation_classes, key_words, use_classes);
        SBEM_INFO(
            "sparse assembly (PML): {} unordered stretched triangle pairs ({} touching, {} of "
            "them copied from {} translation classes), cache {:.1f} MB",
            npairs, stats.touching_pairs, stats.translated_pairs, stats.translation_classes,
            static_cast<Real>(stats.cache_bytes) * 1e-6);
        ExceptionSlot error;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (Index i = 0; i < npairs; ++i) {
            try {
                const auto k = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                if (!rep.empty() && rep[k] >= 0)
                    continue;  // copied from its class representative below
                pair_blocks(space, first[k], cache.partner[k], setup, p.kernel_options, stretch,
                            cache.blocks[k]);
            } catch (...) {
                error.capture();
            }
        }
        error.rethrow();
        if (!rep.empty()) {
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (Index k = 0; k < npairs; ++k) {
                try {
                    const auto kk = static_cast<std::size_t>(k);
                    const Index r = rep[kk];
                    if (r < 0)
                        continue;
                    const auto rr = static_cast<std::size_t>(r);
                    translated_blocks(space, first[kk], cache.partner[kk], slot_map[kk],
                                      slot_map[rr], cache.blocks[rr], cache.blocks[kk]);
                } catch (...) {
                    error.capture();
                }
            }
            error.rethrow();
        }
        stats.cache_seconds =
            std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
    }

    const auto t_scatter = std::chrono::steady_clock::now();
    std::vector<Index> pairs_of(static_cast<std::size_t>(F), 0), stretched_of(pairs_of);
    for_each_test_triangle(groups, [&](Index t) {
        const basis::RwgSpace::Support st = space.support(t);
        if (st.count == 0)
            return;
        std::vector<Index> sources;
        pattern_sources(space, pattern, st, sources);
        PairBlocks blk;
        for (const Index s : sources) {
            const basis::RwgSpace::Support ss = space.support(s);
            const bool sp = stretched_pair(t, s);
            if (use_cache && sp) {
                const Index k = cache.find(t, s);
                if (k < 0)
                    throw std::logic_error("assemble_sparse: stretched pair missing in the cache");
                oriented_blocks(cache.blocks[static_cast<std::size_t>(k)], t > s, blk);
            } else {
                pair_blocks(space, t, s, setup, p.kernel_options, stretch, blk);
            }
            pairs_of[static_cast<std::size_t>(t)] += 1;
            stretched_of[static_cast<std::size_t>(t)] += sp ? 1 : 0;
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
    if (options.stats != nullptr) {
        for (Index t = 0; t < F; ++t) {
            stats.ordered_pairs += pairs_of[static_cast<std::size_t>(t)];
            stats.stretched_pairs += stretched_of[static_cast<std::size_t>(t)];
        }
        stats.scatter_seconds =
            std::chrono::duration<Real>(std::chrono::steady_clock::now() - t_scatter).count();
        *options.stats = stats;
    }
    return std::make_shared<SparseOperator>(std::move(Z));
}

}  // namespace

std::shared_ptr<SparseOperator> assemble_sparse(const Problem& p, const BasisPattern& pattern) {
    validate(p);
    require_unstretched(p, "assemble_sparse (MLFMM near field)");
    return assemble_sparse_impl(p, pattern, nullptr);
}

std::shared_ptr<SparseOperator> assemble_sparse_stretched(const Problem& p,
                                                          const BasisPattern& pattern,
                                                          const SparseStretchedOptions& options) {
    validate(p);
    return assemble_sparse_impl(p, pattern, has_stretch(p) ? p.stretch : nullptr, options);
}

SparseStretchedCacheEstimate estimate_sparse_stretched_cache(
    const basis::RwgSpace& space, const geometry::CoordinateStretch& stretch,
    const BasisPattern& pattern, StretchedPairEvaluation evaluation) {
    const Index N = space.size();
    check_pattern(pattern, N);
    SparseStretchedCacheEstimate e;
    if (stretch.is_identity() || evaluation == StretchedPairEvaluation::per_ordering)
        return e;
    const geometry::TriangleMesh& mesh = space.mesh();
    const std::vector<char> str = stretched_triangles(mesh, stretch);
    std::vector<Index> ptr, partner, first;
    canonical_index(space, pattern, str, ptr, partner, first);
    e.canonical_pairs = ptr.back();
    const Triangles& T = mesh.triangles();
    for (std::size_t k = 0; k < partner.size(); ++k)
        e.touching_pairs += shares_vertex(T, first[k], partner[k]) ? 1 : 0;
    const bool classes = evaluation == StretchedPairEvaluation::translation_classes;
    // At most one class per touching pair, keys of at most kMaxStretchedTouchingKeyWords words.
    e.bytes = cache_bytes_model(mesh.num_triangles(), e.canonical_pairs, e.touching_pairs,
                                classes ? e.touching_pairs : 0,
                                kernels::kMaxStretchedTouchingKeyWords, classes);
    return e;
}

PairBlockEvaluator::PairBlockEvaluator(const Problem& p) {
    validate(p);
    space_ = p.space;
    options_ = p.kernel_options;
    stretch_ = has_stretch(p) ? p.stretch : nullptr;
    weights_ = make_setup(p);
}

void PairBlockEvaluator::evaluate(Index t_test, Index t_src, TrianglePairBlocks& out) const {
    pair_blocks(*space_, t_test, t_src, weights_, options_, stretch_, out);
}

std::shared_ptr<RegionSparseOperator> assemble_region_sparse(const Problem& p, int region,
                                                             std::vector<Index> row_ptr,
                                                             std::vector<Index> cols) {
    validate(p);
    require_unstretched(p, "assemble_region_sparse (MLFMM exact far part)");
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
    // PML (ADR 0009 decision 4): masked test functions get exactly 0; only triangles carrying an
    // unmasked function are integrated (never a triangle with a vertex inside the PML). Without
    // a stretch both vectors stay empty and the loop below is the unstretched one.
    const bool use_mask = has_stretch(p);
    std::vector<std::uint8_t> masked;
    std::vector<std::uint8_t> active;  // per triangle: carries an unmasked function
    if (use_mask) {
        masked = pml_masked_functions(space, *p.stretch);
        active.assign(static_cast<std::size_t>(F), 0);
        for (Index t = 0; t < F; ++t) {
            const basis::RwgSpace::Support sup = space.support(t);
            for (int a = 0; a < sup.count; ++a) {
                if (masked[static_cast<std::size_t>(sup.n[a])] == 0)
                    active[static_cast<std::size_t>(t)] = 1;
            }
        }
    }
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
            if (sup.count == 0 || (use_mask && active[static_cast<std::size_t>(t)] == 0))
                continue;
            const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
            const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
            const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
            const Real area = mesh.area(t);
            for (std::size_t q = 0; q < rule.weights.size(); ++q) {
                const Vec3& l = rule.barycentric[q];
                const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
                const Real w = rule.weights[q] * area;
                const auto [E, H] = exc.fields(r);  // one pass for AngularSpectrumBeam
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
    for (Index n = 0; n < static_cast<Index>(masked.size()); ++n) {
        if (masked[static_cast<std::size_t>(n)] != 0) {
            b(n) = Complex(0.0, 0.0);
            b(N + n) = Complex(0.0, 0.0);
        }
    }
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

    // With a PML the dense path sums in a different order (DenseStrategy::build, file comment
    // "Stretched pairs"): first twice the evaluated ordering of the stretched pair (tp, tm), then
    // the remaining pairs in the order above.
    const bool pml = has_stretch(p);
    const std::vector<char> str = pml ? stretched_triangles(p) : std::vector<char>{};
    const auto stretched_pair = [&](Index t, Index s) {
        return pml &&
               (str[static_cast<std::size_t>(t)] != 0 || str[static_cast<std::size_t>(s)] != 0);
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
            if (stretched_pair(tp, tm)) {
                const bool fwd = evaluated_ordering(tp, tm);
                const Index t = fwd ? tp : tm;
                const Index s = fwd ? tm : tp;
                pair_blocks(space, t, s, setup, p.kernel_options, p.stretch, blk);
                const auto k = static_cast<std::size_t>(3 * slot_of(space.support(t), m) +
                                                        slot_of(space.support(s), m));
                d(m) += blk.jj[k];
                d(N + m) += blk.mm[k];
                d(m) = d(m) + d(m);
                d(N + m) = d(N + m) + d(N + m);
            }
            for (const Index t : tests) {
                const int a = slot_of(space.support(t), m);
                for (const Index s : sources) {
                    if (s != t && stretched_pair(t, s))
                        continue;
                    const int b = slot_of(space.support(s), m);
                    pair_blocks(space, t, s, setup, p.kernel_options, p.stretch, blk);
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
