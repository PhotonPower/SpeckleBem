#pragma once
/// @file gmres.hpp
/// Complex GMRES (Saad & Schultz) with optional restart. The paper reports
/// that restarts slow convergence for these problems, so the default is
/// full GMRES with a residual tolerance of 1e-3.
#include "specklebem/operator/linear_operator.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <functional>
#include <vector>

namespace specklebem::solver {

struct GmresParams {
    Real tolerance = 1e-3;  ///< relative residual ||r||/||b||
    int max_iter = 2000;
    int restart = 0;  ///< 0 => no restart (full GMRES)
    bool verbose = true;
};

struct GmresResult {
    VectorXc x;
    std::vector<Real> residual_history;
    int iterations = 0;
    bool converged = false;
    double wall_seconds = 0;
};

using IterationCallback = std::function<void(int iter, Real residual)>;

GmresResult gmres(const op::LinearOperator& A, const VectorXc& b, const Preconditioner& M,
                  const GmresParams& p, const IterationCallback& cb = {});

}  // namespace specklebem::solver
