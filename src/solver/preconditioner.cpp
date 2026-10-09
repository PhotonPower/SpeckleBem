#include "specklebem/solver/preconditioner.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace specklebem::solver {

DiagonalPreconditioner::DiagonalPreconditioner(VectorXc diag) : inv_diag_(std::move(diag)) {
    if (inv_diag_.size() == 0) {
        throw std::invalid_argument("DiagonalPreconditioner: empty diagonal");
    }
    for (Index i = 0; i < inv_diag_.size(); ++i) {
        const Complex d = inv_diag_(i);
        if (!std::isfinite(d.real()) || !std::isfinite(d.imag())) {
            throw std::invalid_argument("DiagonalPreconditioner: non-finite diagonal entry " +
                                        std::to_string(i));
        }
        if (d == Complex(0.0, 0.0)) {
            throw std::invalid_argument("DiagonalPreconditioner: zero diagonal entry " +
                                        std::to_string(i));
        }
        const Complex inv = Complex(1.0, 0.0) / d;
        if (!std::isfinite(inv.real()) || !std::isfinite(inv.imag())) {
            throw std::invalid_argument("DiagonalPreconditioner: diagonal entry " +
                                        std::to_string(i) + " has no finite reciprocal");
        }
        inv_diag_(i) = inv;
    }
}

void DiagonalPreconditioner::apply(const VectorXc& r, VectorXc& z) const {
    if (r.size() != inv_diag_.size()) {
        throw std::invalid_argument("DiagonalPreconditioner::apply: vector size " +
                                    std::to_string(r.size()) +
                                    " != " + std::to_string(inv_diag_.size()));
    }
    z = inv_diag_.cwiseProduct(r);
}

}  // namespace specklebem::solver
