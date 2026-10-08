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
/// turn; within a colour the triangles are distributed with an OpenMP static schedule, each
/// writing only the rows of its own basis functions (m and N + m). Every entry Z(m, n)
/// therefore receives its contributions in a fixed order (colour of the test triangle, then
/// test triangle index, then source triangle index) for any thread count: the matrix is
/// bitwise identical with 1 or many threads and free of data races.
#include "specklebem/operator/assembler.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/operator/dense_operator.hpp"

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

/// Largest dense system: 2N = 1e5 unknowns are 160 GB of matrix storage (16 (2N)^2 bytes);
/// beyond that a dense matrix makes no sense (compressed operators are the tool, docs/01).
constexpr Index kMaxDenseUnknowns = 100000;
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
    std::vector<int> colour_of;
    const std::vector<std::vector<Index>> groups = colour_triangles(space, colour_of);
    const Real bytes = 16.0 * static_cast<Real>(2 * N) * static_cast<Real>(2 * N);
    SBEM_INFO("dense assembly: N = {}, 2N = {}, {} triangles, {} colours, matrix {:.3f} GB", N,
              2 * N, F, groups.size(), bytes * 1e-9);
    const ScopedTimer timer("dense assembly (2N = " + std::to_string(2 * N) + ")");

    MatrixXc Z = MatrixXc::Zero(2 * N, 2 * N);
    Complex* const data = Z.data();
    const Index ld = Z.outerStride();
    ExceptionSlot error;
    for (const std::vector<Index>& group : groups) {
        const auto ng = static_cast<Index>(group.size());
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (Index g = 0; g < ng; ++g) {
            try {
                const Index t = group[static_cast<std::size_t>(g)];
                const basis::RwgSpace::Support st = space.support(t);
                if (st.count == 0)
                    continue;
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
            } catch (...) {
                error.capture();
            }
        }
        error.rethrow();
    }
    return std::make_shared<DenseOperator>(std::move(Z));
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
    const kernels::TriangleRule& rule = kernels::triangle_rule(p.kernel_options.quad_degree_near);

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
