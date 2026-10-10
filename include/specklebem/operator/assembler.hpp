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

#include <memory>
#include <vector>

namespace specklebem::op {

class SparseOperator;
class RegionSparseOperator;

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
};

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
/// @throws std::invalid_argument for an invalid Problem (validate) or an inconsistent pattern
///         (sizes, group indices, columns out of range or not strictly ascending).
std::shared_ptr<SparseOperator> assemble_sparse(const Problem& p, const BasisPattern& pattern);

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
/// @throws std::invalid_argument for an invalid Problem (validate), a region other than 0 / 1
///         or an inconsistent pattern.
std::shared_ptr<RegionSparseOperator> assemble_region_sparse(const Problem& p, int region,
                                                             std::vector<Index> row_ptr,
                                                             std::vector<Index> cols);

/// Right-hand side  b = [ (a1/eta1) <f, E_inc>_tan ; b1 eta1 <f, H_inc>_tan ].
/// <f_m, E_inc> = sum over the two support triangles of int f_m . E_inc dS with the Dunavant
/// rule of degree p.kernel_options.quad_degree_rhs (positive-interior; WP7c, before:
/// quad_degree_near).
/// @throws std::invalid_argument for an invalid Problem (validate) or a null excitation.
VectorXc assemble_rhs(const Problem& p);

/// Diagonal of Z (for the left Jacobi preconditioner used in the paper), computed without
/// forming Z: for each basis m the four pairs (t, s) of its support triangles, both regions.
/// Bitwise equal to the diagonal of DenseStrategy().build(p): same blocks, same summation
/// order (test triangles by colour, then index; source triangles by index).
/// @throws std::invalid_argument for an invalid Problem (validate).
VectorXc assemble_diagonal(const Problem& p);

}  // namespace specklebem::op
