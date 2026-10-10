/// @file region_sparse_operator.cpp
/// Single-region sparse part of Z stored as (L_i, K_i) per basis pair (WP21 review).
#include "specklebem/operator/region_sparse_operator.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace specklebem::op {

RegionSparseOperator::RegionSparseOperator(Index n, std::vector<Index> row_ptr,
                                           std::vector<Index> cols, std::vector<Complex> values,
                                           Complex e, Complex h, Complex m)
    : n_(n),
      row_ptr_(std::move(row_ptr)),
      cols_(std::move(cols)),
      values_(std::move(values)),
      e_(e),
      h_(h),
      m_(m) {
    const auto fail = [](const std::string& what) {
        throw std::invalid_argument("RegionSparseOperator: " + what);
    };
    if (n_ < 0)
        fail("negative N");
    if (row_ptr_.size() != static_cast<std::size_t>(n_) + 1 || row_ptr_.front() != 0)
        fail("row_ptr must have N + 1 entries starting with 0");
    if (row_ptr_.back() != static_cast<Index>(cols_.size()))
        fail("row_ptr[N] differs from the number of columns");
    if (values_.size() != 2 * cols_.size())
        fail("values must hold two entries (L, K) per column");
    for (std::size_t r = 0; r + 1 < row_ptr_.size(); ++r) {
        if (row_ptr_[r + 1] < row_ptr_[r])
            fail("row_ptr decreases");
    }
    for (std::size_t r = 0; r + 1 < row_ptr_.size(); ++r) {
        for (Index k = row_ptr_[r]; k < row_ptr_[r + 1]; ++k) {
            const Index c = cols_[static_cast<std::size_t>(k)];
            if (c < 0 || c >= n_)
                fail("column " + std::to_string(c) + " outside [0, N)");
            if (k > row_ptr_[r] && c <= cols_[static_cast<std::size_t>(k - 1)])
                fail("columns of a row are not strictly ascending");
        }
    }
}

void RegionSparseOperator::apply(const VectorXc& x, VectorXc& y) const {
    if (x.size() != cols()) {
        throw std::invalid_argument("RegionSparseOperator::apply: x has " +
                                    std::to_string(x.size()) + " entries, expected " +
                                    std::to_string(cols()));
    }
    if (&x == &y) {
        const VectorXc copy = x;
        apply(copy, y);
        return;
    }
    y.resize(rows());
    const Index* const ptr = row_ptr_.data();
    const Index* const col = cols_.data();
    const Complex* const val = values_.data();
    const Complex* const xj = x.data();
    const Complex* const xm = x.data() + n_;
    Complex* const yp = y.data();
    const Index n = n_;
    const Complex e = e_, h = h_, m = m_;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (Index r = 0; r < n; ++r) {
        Complex lj(0.0, 0.0), km(0.0, 0.0), kj(0.0, 0.0), lm(0.0, 0.0);
        for (Index k = ptr[r]; k < ptr[r + 1]; ++k) {
            const Complex l = val[2 * k];
            const Complex kk = val[2 * k + 1];
            const Index c = col[k];
            lj += l * xj[c];
            km += kk * xm[c];
            kj += kk * xj[c];
            lm += l * xm[c];
        }
        yp[r] = e * (lj - km);
        yp[n + r] = h * kj + m * lm;
    }
}

std::string RegionSparseOperator::describe() const {
    const Real full = static_cast<Real>(n_) * static_cast<Real>(n_);
    const Real fill = full > 0.0 ? 100.0 * static_cast<Real>(pairs()) / full : 0.0;
    char buf[96];
    std::snprintf(buf, sizeof(buf), " (%.3g %% of N^2), %.3f MB", fill,
                  static_cast<Real>(memory_bytes()) * 1e-6);
    return "region sparse " + std::to_string(rows()) + "x" + std::to_string(cols()) + ", " +
           std::to_string(pairs()) + " basis pairs (L, K)" + buf;
}

std::size_t RegionSparseOperator::memory_bytes() const {
    return cols_.size() * kBytesPerPair + row_ptr_.size() * sizeof(Index);
}

}  // namespace specklebem::op
