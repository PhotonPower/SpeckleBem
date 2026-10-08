#pragma once
/// @file sparse_operator.hpp
/// Block-sparse near-field part Z_near shared by all compression schemes.
#include "specklebem/operator/linear_operator.hpp"

#include <Eigen/SparseCore>

namespace specklebem::op {

class SparseOperator final : public LinearOperator {
public:
    using Matrix = Eigen::SparseMatrix<Complex, Eigen::RowMajor, Index>;
    explicit SparseOperator(Matrix Z);
    [[nodiscard]] Index rows() const override { return Z_.rows(); }
    [[nodiscard]] Index cols() const override { return Z_.cols(); }
    void apply(const VectorXc& x, VectorXc& y) const override;
    [[nodiscard]] std::string describe() const override;
    [[nodiscard]] std::size_t memory_bytes() const override;
    [[nodiscard]] const Matrix& matrix() const { return Z_; }

private:
    Matrix Z_;
};

}  // namespace specklebem::op
