#pragma once
/// @file linear_operator.hpp
/// The central abstraction of SpeckleBem (ADR 0002).
///
/// Every representation of the impedance matrix Z (dense, MLFMM, H-matrix,
/// ACA, GPU-resident, ...) implements this interface. Iterative solvers only
/// ever call apply(). The matrix is never required to exist explicitly.
#include "specklebem/core/types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace specklebem::op {

class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    [[nodiscard]] virtual Index rows() const = 0;
    [[nodiscard]] virtual Index cols() const = 0;

    /// y = A x
    ///
    /// Thread safety: apply() must be safe to call concurrently from several threads on the
    /// same const operator (distinct x, y). Implementations with scratch buffers (e.g. MLFMM
    /// aggregation/disaggregation storage) keep them per call or per thread, never as shared
    /// mutable members. The Python binding (LinearOperator.matvec) calls apply() without any
    /// lock and relies on this.
    virtual void apply(const VectorXc& x, VectorXc& y) const = 0;

    /// Human-readable description (type, levels, memory, ...).
    [[nodiscard]] virtual std::string describe() const = 0;
    /// Approximate memory footprint in bytes (for reports and planning).
    [[nodiscard]] virtual std::size_t memory_bytes() const = 0;

    [[nodiscard]] VectorXc operator*(const VectorXc& x) const {
        VectorXc y(rows());
        apply(x, y);
        return y;
    }
};

/// A + B, used e.g. for Z = Z_near + Z_far.
class SumOperator final : public LinearOperator {
public:
    SumOperator(std::shared_ptr<const LinearOperator> a, std::shared_ptr<const LinearOperator> b);
    [[nodiscard]] Index rows() const override;
    [[nodiscard]] Index cols() const override;
    void apply(const VectorXc& x, VectorXc& y) const override;
    [[nodiscard]] std::string describe() const override;
    [[nodiscard]] std::size_t memory_bytes() const override;

private:
    std::shared_ptr<const LinearOperator> a_, b_;
};

}  // namespace specklebem::op
