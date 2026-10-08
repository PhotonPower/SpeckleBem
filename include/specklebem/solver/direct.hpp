#pragma once
/// @file direct.hpp
/// Dense LU solve (LAPACK zgesv) for reference solutions of small problems.
#include "specklebem/operator/dense_operator.hpp"

namespace specklebem::solver {

VectorXc solve_direct(const op::DenseOperator& Z, const VectorXc& b);

}  // namespace specklebem::solver
