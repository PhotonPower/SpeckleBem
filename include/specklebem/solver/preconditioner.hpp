#pragma once
/// @file preconditioner.hpp
/// Left preconditioners M^{-1} for the iterative solver.
#include "specklebem/operator/linear_operator.hpp"

#include <memory>

namespace specklebem::solver {

class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void apply(const VectorXc& r, VectorXc& z) const = 0;  ///< z = M^{-1} r
    [[nodiscard]] virtual std::string name() const = 0;
};

class IdentityPreconditioner final : public Preconditioner {
public:
    void apply(const VectorXc& r, VectorXc& z) const override { z = r; }
    [[nodiscard]] std::string name() const override { return "none"; }
};

/// Diagonal (Jacobi) preconditioner: the one that gave the fastest convergence
/// for Ag surfaces in the paper.
class DiagonalPreconditioner final : public Preconditioner {
public:
    explicit DiagonalPreconditioner(VectorXc diag);
    void apply(const VectorXc& r, VectorXc& z) const override;
    [[nodiscard]] std::string name() const override { return "diagonal"; }

private:
    VectorXc inv_diag_;
};

/// Block-diagonal preconditioner built from the near-field blocks of the
/// octree leaves (planned; see docs/01_project_plan.md, Phase 4).
class BlockDiagonalPreconditioner;

/// Schur-complement preconditioner (Ergül, Malas & Gürel 2011) for metals
/// (planned; Phase 8).
class SchurComplementPreconditioner;

}  // namespace specklebem::solver
