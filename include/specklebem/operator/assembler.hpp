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

namespace specklebem::op {

struct Problem {
    const basis::RwgSpace* space = nullptr;
    material::Material exterior;  ///< region R1
    material::Material object;    ///< region R2
    const formulation::Formulation* formulation = nullptr;
    const excitation::Excitation* excitation = nullptr;
    kernels::OperatorOptions kernel_options;
    /// Angular frequency [rad/s]; must be > 0 and equal to excitation->omega() when an
    /// excitation is set (relative tolerance 1e-12). Additive member (WP9).
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
/// invalid Problem (validate) or 2N > 1e5 (160 GB of matrix storage).
class DenseStrategy final : public CompressionStrategy {
public:
    [[nodiscard]] std::shared_ptr<LinearOperator> build(const Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "dense"; }
};

/// Right-hand side  b = [ (a1/eta1) <f, E_inc>_tan ; b1 eta1 <f, H_inc>_tan ].
/// <f_m, E_inc> = sum over the two support triangles of int f_m . E_inc dS with the Dunavant
/// rule of degree p.kernel_options.quad_degree_near (positive-interior).
/// @throws std::invalid_argument for an invalid Problem (validate) or a null excitation.
VectorXc assemble_rhs(const Problem& p);

/// Diagonal of Z (for the left Jacobi preconditioner used in the paper), computed without
/// forming Z: for each basis m the four pairs (t, s) of its support triangles, both regions.
/// Bitwise equal to the diagonal of DenseStrategy().build(p): same blocks, same summation
/// order (test triangles by colour, then index; source triangles by index).
/// @throws std::invalid_argument for an invalid Problem (validate).
VectorXc assemble_diagonal(const Problem& p);

}  // namespace specklebem::op
