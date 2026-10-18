#pragma once
/// @file assembler.hpp
/// Builds the system operator and right-hand side for a given
/// (mesh, materials, formulation, excitation) from an interchangeable
/// compression strategy. The unknown vector is x = [J; M] with 2N entries.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/linear_operator.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specklebem::op {

class SparseOperator;
class RegionSparseOperator;

/// Region parameters of R1, R2 and the scalar factors of the region blocks in Z (DenseStrategy):
/// e_i = a_i / eta_i (+L_i in JJ, -K_i in JM, E part of b for i = 0), h_i = b_i eta_i (+K_i in
/// MJ, H part of b for i = 0), m_i = b_i / eta_i (+L_i in MM); active[i]: region i contributes
/// (some weight non-zero), its element_blocks are skipped otherwise.
struct BlockWeights {
    std::array<kernels::RegionParams, 2> region;
    std::array<Complex, 2> e;
    std::array<Complex, 2> h;
    std::array<Complex, 2> m;
    std::array<bool, 2> active;
};

/// The combined contribution of one triangle pair (t_test, t_src) to the four N x N sub-blocks
/// of Z: entry (a, b) at index 3 a + b, a the slot of support(t_test), b the slot of
/// support(t_src) (kernels::element_blocks); unused slots are zero.
struct TrianglePairBlocks {
    std::array<Complex, 9> jj, jm, mj, mm;
};

struct Problem {
    const basis::RwgSpace* space = nullptr;
    material::Material exterior;  ///< region R1
    material::Material object;    ///< region R2
    const formulation::Formulation* formulation = nullptr;
    const excitation::Excitation* excitation = nullptr;
    kernels::OperatorOptions kernel_options;
    /// Angular frequency [rad/s]: the single source of the frequency for op:: and post::.
    /// Must be > 0 and equal to excitation->omega() when an excitation is set (relative
    /// tolerance 1e-12).
    Real omega = 0;
    /// Complex coordinate stretch of the PML (ADR 0009, WP-PML2), non-owning; nullptr or an
    /// identity stretch: none (every result bitwise as without the field). A non-identity stretch
    /// is supported by DenseStrategy and assemble_diagonal (kernels::element_blocks with the
    /// stretch, both regions), by assemble_rhs (masked PML right-hand side, WP-PML3a) and by
    /// post::far_field with a current window (WP-PML3a), by assemble_sparse_stretched and
    /// PairBlockEvaluator and by mlfmm::MlfmmOperator (fast PML ring, WP-PML5a); assemble_sparse,
    /// assemble_region_sparse, mlfmm::MlfmmFarOperator and post::scattered_field / total_field
    /// refuse it (require_unstretched).
    const geometry::CoordinateStretch* stretch = nullptr;
};

/// True if p carries a stretch that is not the identity (Problem::stretch).
[[nodiscard]] bool has_stretch(const Problem& p);

/// @throws std::invalid_argument naming `who` if has_stretch(p): the caller does not support the
///         PML stretch (Problem::stretch).
void require_unstretched(const Problem& p, const std::string& who);

/// PML mask of the test functions (ADR 0009 decision 4, implementation WP-PML3a): entry n is 1
/// iff a support triangle of basis n (T_n^+ or T_n^-) has a vertex strictly inside the stretched
/// zone (stretch.in_pml(): |x| > x.start or |y| > y.start), else 0. Masking rule: an RWG function
/// straddling the Gap/Pml line (one triangle in the gap, one in the PML) is masked; a function
/// whose triangles only touch the PML start line (vertices on it, none beyond) is not. The
/// identity stretch masks nothing; a stretch with sigma_max = 0 still masks by the geometric
/// zone (in_pml). Size N = space.size(); O(N).
[[nodiscard]] std::vector<std::uint8_t> pml_masked_functions(
    const basis::RwgSpace& space, const geometry::CoordinateStretch& stretch);

/// Checks a Problem before assembly (WP9): space and formulation non-null, omega finite and
/// > 0, eps_r and mu_r of both regions finite and non-zero, omega equal to excitation->omega()
/// (relative 1e-12) if an excitation is set, the excitation's background equal to `exterior` (the
/// incident field must live in R1), and the kernel options (kernels::validate). O(1).
/// DenseStrategy::build, assemble_rhs and assemble_diagonal call it first.
/// @throws std::invalid_argument if any check fails.
void validate(const Problem& p);

/// Strategy interface: a compression scheme turns a Problem into a LinearOperator.
class CompressionStrategy {
public:
    virtual ~CompressionStrategy() = default;
    [[nodiscard]] virtual std::shared_ptr<LinearOperator> build(const Problem& p) const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

/// Largest dense system DenseStrategy builds: 2N = 1e5 unknowns are 160 GB of matrix storage
/// (16 (2N)^2 bytes); beyond that a dense matrix makes no sense (compressed operators are the
/// tool, docs/01).
inline constexpr Index kMaxDenseUnknowns = 100000;

/// Dense reference: every interaction integrated explicitly.
///
/// Z (2N x 2N, x = [J; M]) is the Galerkin matrix of the combined tangential equations
/// (docs/03; derivation and jump-term signs in src/operator/assembler.cpp):
///   Z = [ sum_i (a_i/eta_i) L_i     -sum_i (a_i/eta_i) K_i ]
///       [ sum_i  b_i eta_i  K_i      sum_i (b_i/eta_i) L_i ]
/// with K_1 = K_1^PV - 1/2 I and K_2 = K_2^PV + 1/2 I (I_mn = <f_m, n x f_n>, n into R1).
/// With a non-identity Problem::stretch every pair with a stretched triangle uses the stretched
/// kernels (kernels::element_blocks overload; jump terms unchanged), all others bitwise as without.
/// Memory 16 (2N)^2 bytes; time O(F^2) element_blocks calls per region. OpenMP over the test
/// triangles of a greedy triangle colouring: the result is bitwise identical for any thread
/// count. A region whose weights are all zero is skipped. @throws std::invalid_argument for an
/// invalid Problem (validate) or 2N > kMaxDenseUnknowns.
class DenseStrategy final : public CompressionStrategy {
public:
    [[nodiscard]] std::shared_ptr<LinearOperator> build(const Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "dense"; }
};

/// Pattern of basis pairs (m, n), m, n in [0, N), shared by all four blocks of Z. Rows are
/// grouped: the rows m with group_of_row[m] = g share the column list
/// cols[col_ptr[g] .. col_ptr[g + 1]) (strictly ascending). Grouping keeps the pattern O(N)
/// plus one list per group when many rows have the same columns (the rows of one octree leaf
/// for the MLFMM near field, mlfmm::near_pattern); one group per row is plain CSR.
struct BasisPattern {
    std::vector<Index> group_of_row;  ///< size N, entries in [0, G)
    std::vector<Index> col_ptr;       ///< size G + 1, col_ptr[0] = 0, non-decreasing
    std::vector<Index> cols;          ///< size col_ptr[G], each group ascending in [0, N)
};

/// Exact Galerkin entries of Z (the DenseStrategy matrix, same formula, weights and jump terms)
/// on a basis-pair pattern: entry (r, c) of the 2N x 2N system is stored iff
/// (r mod N, c mod N) is in the pattern, i.e. each pattern pair gives its JJ, JM, MJ and MM
/// entries. Row r holds the columns of row r mod N, then the same columns + N.
///
/// Every triangle pair (t, s) with t in supp(m), s in supp(n) for some pattern pair (m, n) is
/// integrated once per active region (element_blocks); only the pattern entries of its 3 x 3
/// blocks are scattered. Test triangles run colour by colour with the dense scheduling, source
/// triangles in ascending order, so every stored entry receives the same contributions in the
/// same order as in DenseStrategy::build and is bitwise equal to the dense entry. The result
/// is bitwise identical for any thread count. Memory 4 nnz (16 + 8) bytes for nnz pattern
/// pairs; time O(number of triangle pairs touched by the pattern).
/// @throws std::invalid_argument for an invalid Problem (validate), a non-identity stretch
///         (require_unstretched) or an inconsistent pattern (sizes, group indices, columns out of
///         range or not strictly ascending).
std::shared_ptr<SparseOperator> assemble_sparse(const Problem& p, const BasisPattern& pattern);

/// assemble_sparse with the PML stretch of p (WP-PML5a, the near part of the fast PML ring,
/// mlfmm::PmlRingOperator): every triangle pair with a stretched triangle uses the stretched
/// kernels (kernels::element_blocks overload, jump terms unchanged), every other pair the
/// unstretched ones, so the entries of basis pairs without a stretched support triangle are
/// bitwise those of assemble_sparse and of the unstretched DenseStrategy, and the others equal
/// the DenseStrategy entries with the stretch up to the summation order (the dense assembler sums
/// a stretched entry as A + A^T + rest, assembler.cpp "Stretched pairs"; here every triangle pair
/// of a pattern pair is evaluated in the ordering (test, source) of its row, i.e. an unordered
/// stretched pair whose both orderings are in the pattern is evaluated twice). Without a stretch
/// (or with the identity) bitwise assemble_sparse. Schedule, determinism and memory as
/// assemble_sparse.
/// @throws std::invalid_argument for an invalid Problem (validate) or an inconsistent pattern.
std::shared_ptr<SparseOperator> assemble_sparse_stretched(const Problem& p,
                                                          const BasisPattern& pattern);

/// The per-triangle-pair blocks of the dense assembler (WP-PML5a, the far part of the fast PML
/// ring): evaluate(t, s) gives the combined blocks of Z that DenseStrategy adds for the pair
/// (t, s) (both regions with the weights of p.formulation, K with its jump term on identical
/// triangles, the stretched kernels with p.stretch for pairs with a stretched triangle; bitwise
/// the dense contribution of that ordering). Holds a pointer to p.space (must outlive the
/// evaluator) and copies of the kernel options, the stretch pointer and the weights.
/// evaluate() is allocation-free and safe to call concurrently.
class PairBlockEvaluator {
public:
    /// @throws std::invalid_argument for an invalid Problem (validate).
    explicit PairBlockEvaluator(const Problem& p);
    /// @throws as kernels::element_blocks (triangle indices out of range, non-finite blocks).
    void evaluate(Index t_test, Index t_src, TrianglePairBlocks& out) const;
    [[nodiscard]] const BlockWeights& weights() const { return weights_; }

private:
    const basis::RwgSpace* space_ = nullptr;
    kernels::OperatorOptions options_;
    const geometry::CoordinateStretch* stretch_ = nullptr;
    BlockWeights weights_;
};

/// Region i's Galerkin entries L_i and K_i (K_i with its jump term on coincident triangles, as in
/// DenseStrategy) on an N x N basis-pair pattern in CSR form, stored per pair as (L_i, K_i) with
/// the region's weights e_i, h_i, m_i of p.formulation (op::RegionSparseOperator): the region-i
/// part of Z on the pattern with 40 instead of 96 bytes per pair (WP21 review, MLFMM exact far
/// part). Same test-triangle schedule as assemble_sparse (element_blocks once per triangle pair
/// touched by the pattern; bitwise identical for any thread count). Each region-i entry of Z on
/// the pattern equals weight x stored value up to round-off (the dense assembler applies the
/// weights per triangle pair). Memory 40 bytes per pair + 8 (N + 1) bytes.
/// @param region 0 = R1, 1 = R2
/// @param row_ptr, cols the pattern, taken over: row_ptr has N + 1 entries starting with 0,
///        cols strictly ascending per row in [0, N)
/// @throws std::invalid_argument for an invalid Problem (validate), a non-identity stretch
///         (require_unstretched), a region other than 0 / 1 or an inconsistent pattern.
std::shared_ptr<RegionSparseOperator> assemble_region_sparse(const Problem& p, int region,
                                                             std::vector<Index> row_ptr,
                                                             std::vector<Index> cols);

/// Right-hand side  b = [ (a1/eta1) <f, E_inc>_tan ; b1 eta1 <f, H_inc>_tan ].
/// <f_m, E_inc> = sum over the two support triangles of int f_m . E_inc dS with the Dunavant
/// rule of degree p.kernel_options.quad_degree_rhs (positive-interior; WP7c, before:
/// quad_degree_near).
///
/// PML (non-identity Problem::stretch, ADR 0009 decision 4, WP-PML3a): the entries of every
/// test function masked by pml_masked_functions() are exactly 0 in both the J and the M rows
/// (the continued incident field would grow like exp(+k Im x~) in the PML; with a dark band it is
/// below the PML target there anyway). All other entries are bitwise equal to the unstretched
/// right-hand side: the stretch is the identity on their support, so the incident field is
/// evaluated at the real r. The incident field is evaluated only on triangles that carry at
/// least one unmasked function (never inside the PML). Without a stretch or with an identity
/// stretch the result is bitwise unchanged.
/// @throws std::invalid_argument for an invalid Problem (validate) or a null excitation.
VectorXc assemble_rhs(const Problem& p);

/// Diagonal of Z (for the left Jacobi preconditioner used in the paper), computed without
/// forming Z: for each basis m the four pairs (t, s) of its support triangles, both regions.
/// Bitwise equal to the diagonal of DenseStrategy().build(p): same blocks, same summation
/// order (test triangles by colour, then index; source triangles by index), also with a
/// stretch.
/// @throws std::invalid_argument for an invalid Problem (validate).
VectorXc assemble_diagonal(const Problem& p);

}  // namespace specklebem::op
