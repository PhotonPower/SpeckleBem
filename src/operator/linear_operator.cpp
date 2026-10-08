#include "specklebem/operator/linear_operator.hpp"

#include <stdexcept>

namespace specklebem::op {

SumOperator::SumOperator(std::shared_ptr<const LinearOperator> a,
                         std::shared_ptr<const LinearOperator> b)
    : a_(std::move(a)), b_(std::move(b)) {
    if (a_->rows() != b_->rows() || a_->cols() != b_->cols()) {
        throw std::invalid_argument("SumOperator: dimension mismatch");
    }
}

Index SumOperator::rows() const {
    return a_->rows();
}
Index SumOperator::cols() const {
    return a_->cols();
}

void SumOperator::apply(const VectorXc& x, VectorXc& y) const {
    a_->apply(x, y);
    VectorXc tmp(rows());
    b_->apply(x, tmp);
    y += tmp;
}

std::string SumOperator::describe() const {
    return "(" + a_->describe() + ") + (" + b_->describe() + ")";
}
std::size_t SumOperator::memory_bytes() const {
    return a_->memory_bytes() + b_->memory_bytes();
}

}  // namespace specklebem::op
