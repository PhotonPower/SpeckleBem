#pragma once
/// @file dense_operator.hpp
/// Explicitly stored impedance matrix. Reference implementation used for
/// validation of all compressed operators and for small problems (N < ~1e5).
#include "specklebem/operator/linear_operator.hpp"

namespace specklebem::op {

class DenseOperator final : public LinearOperator {
public:
    explicit DenseOperator(MatrixXc Z);
    [[nodiscard]] Index rows() const override { return Z_.rows(); }
    [[nodiscard]] Index cols() const override { return Z_.cols(); }
    void apply(const VectorXc& x, VectorXc& y) const override;
    [[nodiscard]] std::string describe() const override;
    [[nodiscard]] std::size_t memory_bytes() const override;
    [[nodiscard]] const MatrixXc& matrix() const { return Z_; }
    MatrixXc& matrix() { return Z_; }

private:
    MatrixXc Z_;
};

}  // namespace specklebem::op
