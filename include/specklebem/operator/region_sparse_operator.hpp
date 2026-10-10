#pragma once
/// @file region_sparse_operator.hpp
/// Sparse single-region part of Z stored as the region's operator entries L_i and K_i per basis
/// pair (WP21 review: the MLFMM exact far part of ADR 0008 §6).
///
/// Region i contributes to the 2N x 2N system (docs/03, operator/assembler.hpp) with e_i =
/// a_i / eta_i, h_i = b_i eta_i, m_i = b_i / eta_i:
///   y_J = e_i (L_i x_J - K_i x_M),   y_M = h_i K_i x_J + m_i L_i x_M,
/// with K_i including its jump term (K_1 - 1/2 I, K_2 + 1/2 I) on coincident triangles. The four
/// blocks share the two N x N matrices, so storing (L_i, K_i) per basis pair and combining them
/// with the weights in apply() needs 2 complex values + 1 column index per pair (40 bytes)
/// instead of the 4 (16 + 8) = 96 bytes of the four combined entries in op::SparseOperator.
#include "specklebem/operator/linear_operator.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace specklebem::op {

class RegionSparseOperator final : public LinearOperator {
public:
    /// N x N basis-pair pattern in CSR form with the region's entries per pair.
    /// @param n number of basis functions N
    /// @param row_ptr size N + 1, row_ptr[0] = 0, non-decreasing, row_ptr[N] = cols.size()
    /// @param cols column (basis) indices, strictly ascending per row, in [0, N)
    /// @param values size 2 cols.size(): (L_i, K_i) of pair k at 2 k and 2 k + 1
    /// @param e, h, m the region's block weights (file comment)
    /// @throws std::invalid_argument for inconsistent sizes or indices.
    RegionSparseOperator(Index n, std::vector<Index> row_ptr, std::vector<Index> cols,
                         std::vector<Complex> values, Complex e, Complex h, Complex m);

    [[nodiscard]] Index rows() const override { return 2 * n_; }
    [[nodiscard]] Index cols() const override { return 2 * n_; }
    /// y = region block x (file comment): OpenMP over basis rows, each row summed in storage
    /// order, so y is bitwise identical for any thread count. x and y may alias. O(pairs).
    /// @throws std::invalid_argument if x.size() != cols().
    void apply(const VectorXc& x, VectorXc& y) const override;
    /// "region sparse 2Nx2N, P basis pairs (f % of N^2), M MB".
    [[nodiscard]] std::string describe() const override;
    /// pairs (2 sizeof(Complex) + sizeof(Index)) + (N + 1) sizeof(Index).
    [[nodiscard]] std::size_t memory_bytes() const override;
    /// Stored basis pairs (each gives four entries of the 2N x 2N system).
    [[nodiscard]] Index pairs() const { return static_cast<Index>(cols_.size()); }
    [[nodiscard]] const std::vector<Index>& row_ptr() const { return row_ptr_; }
    [[nodiscard]] const std::vector<Index>& col_indices() const { return cols_; }
    /// (L_i, K_i) interleaved per stored pair.
    [[nodiscard]] const std::vector<Complex>& values() const { return values_; }

    /// Memory per stored basis pair [bytes].
    static constexpr std::size_t kBytesPerPair = 2 * sizeof(Complex) + sizeof(Index);

private:
    Index n_;
    std::vector<Index> row_ptr_;
    std::vector<Index> cols_;
    std::vector<Complex> values_;
    Complex e_, h_, m_;
};

}  // namespace specklebem::op
