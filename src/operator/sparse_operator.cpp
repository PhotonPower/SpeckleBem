/// @file sparse_operator.cpp
/// Row-parallel CSR matrix-vector product of the stored near field (WP19a).
#include "specklebem/operator/sparse_operator.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace specklebem::op {

SparseOperator::SparseOperator(Matrix Z) {
    // Swap instead of Z_(std::move(Z)): Eigen 3.4.0's SparseMatrix has no move constructor, so the
    // move would copy the matrix (a third copy of the near field during the MLFMM setup).
    Z_.swap(Z);
    Z_.makeCompressed();
}

void SparseOperator::apply(const VectorXc& x, VectorXc& y) const {
    if (x.size() != cols()) {
        throw std::invalid_argument("SparseOperator::apply: x has " + std::to_string(x.size()) +
                                    " entries, expected " + std::to_string(cols()));
    }
    if (&x == &y) {
        const VectorXc copy = x;
        apply(copy, y);
        return;
    }
    y.resize(rows());
    const Index* const outer = Z_.outerIndexPtr();
    const Index* const inner = Z_.innerIndexPtr();
    const Complex* const val = Z_.valuePtr();
    const Complex* const xp = x.data();
    Complex* const yp = y.data();
    const Index n = rows();
    // Rows differ in length (leaf occupancy): small dynamic chunks; each row sum is serial.
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (Index r = 0; r < n; ++r) {
        Complex acc(0.0, 0.0);
        for (Index k = outer[r]; k < outer[r + 1]; ++k) acc += val[k] * xp[inner[k]];
        yp[r] = acc;
    }
}

std::string SparseOperator::describe() const {
    const Real dense = static_cast<Real>(rows()) * static_cast<Real>(cols());
    const Real fill = dense > 0.0 ? 100.0 * static_cast<Real>(nonzeros()) / dense : 0.0;
    char buf[96];
    std::snprintf(buf, sizeof(buf), " (%.3g %% of dense), %.3f MB", fill,
                  static_cast<Real>(memory_bytes()) * 1e-6);
    return "sparse " + std::to_string(rows()) + "x" + std::to_string(cols()) +
           ", nnz = " + std::to_string(nonzeros()) + buf;
}

std::size_t SparseOperator::memory_bytes() const {
    const auto nnz = static_cast<std::size_t>(nonzeros());
    const auto outer = static_cast<std::size_t>(Z_.outerSize()) + 1;
    return nnz * (sizeof(Complex) + sizeof(Index)) + outer * sizeof(Index);
}

}  // namespace specklebem::op
