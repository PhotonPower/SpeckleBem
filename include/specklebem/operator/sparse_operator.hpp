#pragma once
/// @file sparse_operator.hpp
/// Block-sparse near-field part Z_near shared by all compression schemes.
#include "specklebem/operator/linear_operator.hpp"

#include <Eigen/SparseCore>

namespace specklebem::op {

/// Explicitly stored sparse matrix (row-major CSR, Index indices). Built by
/// op::assemble_sparse, e.g. for the MLFMM near field (mlfmm::assemble_near, ADR 0008 §1).
class SparseOperator final : public LinearOperator {
public:
    using Matrix = Eigen::SparseMatrix<Complex, Eigen::RowMajor, Index>;
    /// Takes the matrix and compresses it (no-op if already compressed).
    explicit SparseOperator(Matrix Z);
    [[nodiscard]] Index rows() const override { return Z_.rows(); }
    [[nodiscard]] Index cols() const override { return Z_.cols(); }
    /// y = Z x: OpenMP over rows, each row summed in storage (column) order, so y is bitwise
    /// identical for any thread count. x and y may alias. O(nnz).
    /// @throws std::invalid_argument if x.size() != cols().
    void apply(const VectorXc& x, VectorXc& y) const override;
    /// "sparse RxC, nnz = K (f % of dense), M MB".
    [[nodiscard]] std::string describe() const override;
    /// nnz (sizeof(Complex) + sizeof(Index)) + (rows + 1) sizeof(Index).
    [[nodiscard]] std::size_t memory_bytes() const override;
    [[nodiscard]] Index nonzeros() const { return Z_.nonZeros(); }
    [[nodiscard]] const Matrix& matrix() const { return Z_; }

private:
    Matrix Z_;
};

}  // namespace specklebem::op
