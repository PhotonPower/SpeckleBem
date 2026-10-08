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
};

/// Strategy interface: a compression scheme turns a Problem into a LinearOperator.
class CompressionStrategy {
public:
    virtual ~CompressionStrategy() = default;
    [[nodiscard]] virtual std::shared_ptr<LinearOperator> build(const Problem& p) const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

/// Dense reference: every interaction integrated explicitly.
class DenseStrategy final : public CompressionStrategy {
public:
    [[nodiscard]] std::shared_ptr<LinearOperator> build(const Problem& p) const override;
    [[nodiscard]] std::string name() const override { return "dense"; }
};

/// Right-hand side  b = [ (a1/eta1) <f, E_inc>_tan ; b1 eta1 <f, H_inc>_tan ].
VectorXc assemble_rhs(const Problem& p);

/// Diagonal of Z (for the left Jacobi preconditioner used in the paper).
VectorXc assemble_diagonal(const Problem& p);

}  // namespace specklebem::op
