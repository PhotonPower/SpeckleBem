#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/linear_operator.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace specklebem;

TEST_CASE("dense operator applies and sums", "[operator]") {
    MatrixXc A = MatrixXc::Random(8, 8), B = MatrixXc::Random(8, 8);
    VectorXc x = VectorXc::Random(8);
    auto opA = std::make_shared<op::DenseOperator>(A);
    auto opB = std::make_shared<op::DenseOperator>(B);
    op::SumOperator sum(opA, opB);
    const VectorXc y = sum * x;
    CHECK((y - (A + B) * x).norm() < 1e-12);
    CHECK(sum.memory_bytes() == 2 * 64 * sizeof(Complex));
}
