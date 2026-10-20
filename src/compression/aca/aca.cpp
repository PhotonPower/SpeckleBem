/// @file aca.cpp
/// Partially pivoted ACA (aca.hpp).
#include "specklebem/compression/aca/aca.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace specklebem::aca {

namespace {

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// Index of the largest |x_i| over the entries with used[i] == 0 (first on ties); -1 if none, and
/// best = that modulus (0 if none).
Index argmax_unused(const VectorXc& x, const std::vector<std::uint8_t>& used, Real& best) {
    Index arg = -1;
    best = -1.0;
    for (Index i = 0; i < x.size(); ++i) {
        if (used[sz(i)] != 0)
            continue;
        const Real a = std::abs(x(i));
        if (a > best) {
            best = a;
            arg = i;
        }
    }
    if (arg < 0)
        best = 0.0;
    return arg;
}

/// The unused index farthest (in index distance) from every used one, the first on ties; -1 if
/// all are used. With spatially sorted columns (octree order) the column least like the pivots.
Index farthest_unused(const std::vector<std::uint8_t>& used) {
    const auto n = static_cast<Index>(used.size());
    std::vector<Index> dist(sz(n), n);  // distance to the nearest used index
    Index last = -1;
    for (Index c = 0; c < n; ++c) {
        if (used[sz(c)] != 0)
            last = c;
        if (last >= 0)
            dist[sz(c)] = c - last;
    }
    last = -1;
    for (Index c = n - 1; c >= 0; --c) {
        if (used[sz(c)] != 0)
            last = c;
        if (last >= 0)
            dist[sz(c)] = std::min(dist[sz(c)], last - c);
    }
    Index arg = -1;
    for (Index c = 0; c < n; ++c) {
        if (used[sz(c)] == 0 && (arg < 0 || dist[sz(c)] > dist[sz(arg)]))
            arg = c;
    }
    return arg;
}

void require_finite(const VectorXc& x, const char* what) {
    if (!x.allFinite())
        throw std::runtime_error(
            std::string("aca::compress: the generator returned a non-finite ") + what);
}

}  // namespace

const char* to_string(StopReason r) {
    switch (r) {
        case StopReason::converged:
            return "converged";
        case StopReason::zero_rows:
            return "zero_rows";
        case StopReason::zero_block:
            return "zero_block";
        case StopReason::exhausted:
            return "exhausted";
        case StopReason::rank_cap:
            return "rank_cap";
    }
    return "unknown";
}

std::size_t LowRankBlock::memory_bytes() const {
    return sz(rank()) * sz(rows + cols) * sizeof(Complex);
}

LowRankBlock compress(Index rows, Index cols, const EntryGenerator& gen,
                      const CompressParams& params) {
    if (rows < 1 || cols < 1)
        throw std::invalid_argument("aca::compress: rows and cols must be >= 1");
    if (!(params.epsilon > 0.0 && params.epsilon < 1.0))
        throw std::invalid_argument("aca::compress: epsilon must lie in (0, 1)");
    if (params.max_rank < 0)
        throw std::invalid_argument("aca::compress: max_rank must be >= 0");
    if (params.max_zero_rows < 1)
        throw std::invalid_argument("aca::compress: max_zero_rows must be >= 1");
    if (params.confirmations < 1)
        throw std::invalid_argument("aca::compress: confirmations must be >= 1");
    const Index full = std::min(rows, cols);
    const Index kmax = params.max_rank > 0 ? std::min(params.max_rank, full) : full;

    LowRankBlock out;
    out.rows = rows;
    out.cols = cols;
    Index cap = std::min<Index>(kmax, 16);
    MatrixXc U(rows, cap), V(cols, cap);
    Index k = 0;
    std::vector<std::uint8_t> used_row(sz(rows), 0), used_col(sz(cols), 0);
    std::size_t next_pref = 0;  // next entry of preferred_rows to try
    Index next_seq = 0;         // next row of the ascending fallback
    const auto next_unused_row = [&]() -> Index {
        while (next_pref < params.preferred_rows.size()) {
            const Index i = params.preferred_rows[next_pref++];
            if (i >= 0 && i < rows && used_row[sz(i)] == 0)
                return i;
        }
        for (Index t = 0; t < rows; ++t) {
            const Index i = (next_seq + t) % rows;
            if (used_row[sz(i)] == 0) {
                next_seq = (i + 1) % rows;
                return i;
            }
        }
        return -1;
    };
    VectorXc r(cols), u(rows);
    Real norm2 = 0.0;
    Index zero_run = 0;
    Index hits = 0;         // consecutive steps meeting the stopping rule
    bool probed = false;    // k = 0 probe columns tried
    bool probed_k = false;  // k > 0 probe column tried
    Index i = next_unused_row();
    out.reason = StopReason::exhausted;
    while (k < kmax) {
        if (i < 0) {  // every row used: the approximation reproduces all of them
            out.reason = StopReason::exhausted;
            break;
        }
        gen.row(i, r.data());
        ++out.generated_rows;
        used_row[sz(i)] = 1;
        require_finite(r, "row");
        if (k > 0)
            r.noalias() -= V.leftCols(k) * U.row(i).leftCols(k).transpose();
        Real piv_abs = 0.0;
        const Index j = argmax_unused(r, used_col, piv_abs);
        if (j < 0 || piv_abs == 0.0) {  // zero residual row: try another one
            ++out.zero_rows;
            ++zero_run;
            if (zero_run >= params.max_zero_rows) {
                if (k > 0) {
                    // Probe one unused column (the farthest from the pivot columns) before
                    // accepting (WP-PML5b review suggestion): the zero rows may be rows the
                    // approximation reproduces exactly while other rows are not yet
                    // represented. Its residual restarts the iteration at its largest
                    // unused-row entry if, taken as typical for all columns, it would leave a
                    // residual above the tolerance: ||u|| sqrt(cols) > epsilon ||S_k||_F.
                    Index pc = -1;
                    if (!probed_k) {
                        probed_k = true;
                        pc = farthest_unused(used_col);
                    }
                    Index restart = -1;
                    if (pc >= 0) {
                        gen.column(pc, u.data());
                        ++out.generated_columns;
                        require_finite(u, "column");
                        u.noalias() -= U.leftCols(k) * V.row(pc).leftCols(k).transpose();
                        Real best = 0.0;
                        const Index ri = argmax_unused(u, used_row, best);
                        const Real residual = u.norm() * std::sqrt(static_cast<Real>(cols));
                        if (ri >= 0 && best > 0.0 && residual > params.epsilon * std::sqrt(norm2))
                            restart = ri;
                    }
                    if (restart < 0) {
                        out.reason = StopReason::zero_rows;
                        break;
                    }
                    zero_run = 0;
                    i = restart;
                    continue;
                }
                // k = 0: probe columns before declaring the block zero.
                Index restart = -1;
                if (!probed) {
                    probed = true;
                    std::vector<Index> probes;
                    for (const Index c : params.preferred_columns)
                        if (c >= 0 && c < cols && used_col[sz(c)] == 0)
                            probes.push_back(c);
                    if (probes.empty())
                        probes.push_back(0);
                    for (const Index c : probes) {
                        gen.column(c, u.data());
                        ++out.generated_columns;
                        require_finite(u, "column");
                        Real best = 0.0;
                        const Index ri = argmax_unused(u, used_row, best);
                        if (ri >= 0 && best > 0.0) {
                            restart = ri;
                            break;
                        }
                    }
                }
                if (restart < 0) {
                    out.reason = StopReason::zero_block;
                    break;
                }
                zero_run = 0;
                i = restart;
                continue;
            }
            i = next_unused_row();
            continue;
        }
        zero_run = 0;
        const Complex piv = r(j);
        used_col[sz(j)] = 1;
        gen.column(j, u.data());
        ++out.generated_columns;
        require_finite(u, "column");
        if (k > 0)
            u.noalias() -= U.leftCols(k) * V.row(j).leftCols(k).transpose();
        if (k == cap) {
            cap = std::min(kmax, 2 * cap);
            U.conservativeResize(Eigen::NoChange, cap);
            V.conservativeResize(Eigen::NoChange, cap);
        }
        V.col(k) = r / piv;
        U.col(k) = u;
        const Real nu = u.norm();
        const Real nv = V.col(k).norm();
        Complex cross(0.0, 0.0);
        if (k > 0) {
            const VectorXc au = U.leftCols(k).adjoint() * u;
            const VectorXc av = V.leftCols(k).adjoint() * V.col(k);
            for (Index l = 0; l < k; ++l) cross += au(l) * av(l);
        }
        norm2 += 2.0 * cross.real() + nu * nu * nv * nv;
        norm2 = std::max(norm2, 0.0);
        ++k;
        out.last_increment = nu * nv;
        if (out.last_increment <= params.epsilon * std::sqrt(norm2)) {
            if (++hits >= params.confirmations) {
                out.reason = StopReason::converged;
                break;
            }
        } else {
            hits = 0;
        }
        if (k == kmax) {
            out.reason = kmax == full ? StopReason::exhausted : StopReason::rank_cap;
            break;
        }
        Real best = 0.0;
        i = argmax_unused(u, used_row, best);
        if (i < 0 || best == 0.0)
            i = next_unused_row();
    }
    out.U = U.leftCols(k);
    out.V = V.leftCols(k);
    out.norm = std::sqrt(norm2);
    return out;
}

Real frobenius_error(const MatrixXc& A, const LowRankBlock& b) {
    if (A.rows() != b.rows || A.cols() != b.cols)
        throw std::invalid_argument("aca::frobenius_error: the sizes differ");
    if (b.rank() == 0)
        return A.norm();
    return (A - b.U * b.V.transpose()).norm();
}

}  // namespace specklebem::aca
