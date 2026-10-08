#include "specklebem/operator/dense_operator.hpp"

#include "specklebem/backend/backend.hpp"

namespace specklebem::op {

DenseOperator::DenseOperator(MatrixXc Z) : Z_(std::move(Z)) {}

void DenseOperator::apply(const VectorXc& x, VectorXc& y) const {
    y.resize(rows());
    backend::default_backend()->gemv(Z_, x, y);
}

std::string DenseOperator::describe() const {
    return "dense " + std::to_string(rows()) + "x" + std::to_string(cols());
}

std::size_t DenseOperator::memory_bytes() const {
    return static_cast<std::size_t>(Z_.size()) * sizeof(Complex);
}

}  // namespace specklebem::op
