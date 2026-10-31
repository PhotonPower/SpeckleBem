#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace specklebem::solver {

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Index block_id(const BlockPartition& p, std::size_t k) {
    return p.ids.empty() ? static_cast<Index>(k) : p.ids[k];
}

/// Position of every unknown in the partition: block[i] and local[i] (index within its block).
struct PartitionMap {
    std::vector<Index> block;
    std::vector<Index> local;
};

/// Validates the partition of [0, n) and returns the map.
PartitionMap map_partition(const BlockPartition& p, Index n, const char* who) {
    if (!p.ids.empty() && p.ids.size() != p.blocks.size()) {
        throw std::invalid_argument(std::string(who) + ": " + std::to_string(p.ids.size()) +
                                    " block ids for " + std::to_string(p.blocks.size()) +
                                    " blocks");
    }
    PartitionMap m;
    m.block.assign(static_cast<std::size_t>(n), -1);
    m.local.assign(static_cast<std::size_t>(n), -1);
    Index count = 0;
    for (std::size_t k = 0; k < p.blocks.size(); ++k) {
        const std::vector<Index>& b = p.blocks[k];
        if (b.empty()) {
            throw std::invalid_argument(std::string(who) + ": block " +
                                        std::to_string(block_id(p, k)) + " is empty");
        }
        for (std::size_t a = 0; a < b.size(); ++a) {
            const Index i = b[a];
            if (i < 0 || i >= n) {
                throw std::invalid_argument(std::string(who) + ": index " + std::to_string(i) +
                                            " of block " + std::to_string(block_id(p, k)) +
                                            " outside [0, " + std::to_string(n) + ")");
            }
            if (m.block[static_cast<std::size_t>(i)] >= 0) {
                throw std::invalid_argument(std::string(who) + ": index " + std::to_string(i) +
                                            " belongs to more than one block (or twice to block " +
                                            std::to_string(block_id(p, k)) + ")");
            }
            m.block[static_cast<std::size_t>(i)] = static_cast<Index>(k);
            m.local[static_cast<std::size_t>(i)] = static_cast<Index>(a);
        }
        count += static_cast<Index>(b.size());
    }
    if (count != n) {
        throw std::invalid_argument(std::string(who) + ": the blocks hold " +
                                    std::to_string(count) + " of the " + std::to_string(n) +
                                    " unknowns (every unknown must belong to exactly one block)");
    }
    return m;
}

}  // namespace

std::vector<MatrixXc> extract_diagonal_blocks(
    const BlockPartition& partition,
    const std::vector<const op::SparseOperator::Matrix*>& sources) {
    return extract_diagonal_blocks(partition, sources, MissingBlockEntries::Throw);
}

std::vector<MatrixXc> extract_diagonal_blocks(
    const BlockPartition& partition, const std::vector<const op::SparseOperator::Matrix*>& sources,
    MissingBlockEntries missing, Index* missing_count) {
    const char* who = "extract_diagonal_blocks";
    if (missing != MissingBlockEntries::Throw && missing != MissingBlockEntries::Zero)
        throw std::invalid_argument(std::string(who) + ": invalid MissingBlockEntries value");
    const bool allow_missing = missing == MissingBlockEntries::Zero;
    if (sources.empty())
        throw std::invalid_argument(std::string(who) + ": no source matrix");
    for (const op::SparseOperator::Matrix* s : sources) {
        if (s == nullptr)
            throw std::invalid_argument(std::string(who) + ": null source matrix");
        if (s->rows() != s->cols() || s->rows() != sources.front()->rows()) {
            throw std::invalid_argument(std::string(who) +
                                        ": the source matrices must be square and of one size");
        }
        if (!s->isCompressed())
            throw std::invalid_argument(std::string(who) + ": source matrix not compressed");
    }
    const Index n = sources.front()->rows();
    const PartitionMap map = map_partition(partition, n, who);
    const std::size_t K = partition.blocks.size();
    std::vector<MatrixXc> out(K);
    // Errors inside the parallel loop are recorded per block and thrown afterwards for the
    // lowest failing block (deterministic message).
    std::vector<std::string> error(K);
    std::vector<Index> absent(K, 0);
    const auto nk = static_cast<std::int64_t>(K);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t kk = 0; kk < nk; ++kk) {
        const auto k = static_cast<std::size_t>(kk);
        const std::vector<Index>& rows = partition.blocks[k];
        const auto m = static_cast<Index>(rows.size());
        MatrixXc D = MatrixXc::Zero(m, m);
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(m * m), 0);
        for (const op::SparseOperator::Matrix* s : sources) {
            const Index* outer = s->outerIndexPtr();
            const Index* inner = s->innerIndexPtr();
            const Complex* val = s->valuePtr();
            for (Index a = 0; a < m && error[k].empty(); ++a) {
                const Index r = rows[static_cast<std::size_t>(a)];
                for (Index q = outer[r]; q < outer[r + 1]; ++q) {
                    const Index c = inner[q];
                    if (map.block[static_cast<std::size_t>(c)] != static_cast<Index>(k))
                        continue;
                    const Index b = map.local[static_cast<std::size_t>(c)];
                    std::uint8_t& f = seen[static_cast<std::size_t>(b * m + a)];
                    if (f != 0) {
                        error[k] = "entry (" + std::to_string(r) + ", " + std::to_string(c) +
                                   ") of block " + std::to_string(block_id(partition, k)) +
                                   " is stored in more than one source (it would be counted "
                                   "twice)";
                        break;
                    }
                    f = 1;
                    D(a, b) = val[q];
                }
            }
        }
        if (allow_missing && error[k].empty()) {
            absent[k] = static_cast<Index>(std::count(seen.begin(), seen.end(), std::uint8_t{0}));
            out[k] = std::move(D);
            continue;
        }
        for (Index b = 0; b < m && error[k].empty(); ++b) {
            for (Index a = 0; a < m; ++a) {
                if (seen[static_cast<std::size_t>(b * m + a)] == 0) {
                    error[k] = "entry (" + std::to_string(rows[static_cast<std::size_t>(a)]) +
                               ", " + std::to_string(rows[static_cast<std::size_t>(b)]) +
                               ") of block " + std::to_string(block_id(partition, k)) +
                               " is stored in no source (the sources must hold every same-block "
                               "entry)";
                    break;
                }
            }
        }
        if (error[k].empty())
            out[k] = std::move(D);
    }
    for (std::size_t k = 0; k < K; ++k) {
        if (!error[k].empty())
            throw std::invalid_argument(std::string(who) + ": " + error[k]);
    }
    if (missing_count != nullptr) {
        Index total = 0;
        for (const Index a : absent) total += a;
        *missing_count = total;
    }
    return out;
}

std::vector<MatrixXc> extract_diagonal_blocks(const BlockPartition& partition, const MatrixXc& A) {
    const char* who = "extract_diagonal_blocks";
    if (A.rows() != A.cols())
        throw std::invalid_argument(std::string(who) + ": the matrix must be square");
    (void)map_partition(partition, A.rows(), who);
    const std::size_t K = partition.blocks.size();
    std::vector<MatrixXc> out(K);
    const auto nk = static_cast<std::int64_t>(K);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t kk = 0; kk < nk; ++kk) {
        const auto k = static_cast<std::size_t>(kk);
        const std::vector<Index>& idx = partition.blocks[k];
        const auto m = static_cast<Index>(idx.size());
        MatrixXc D(m, m);
        for (Index b = 0; b < m; ++b) {
            for (Index a = 0; a < m; ++a)
                D(a, b) = A(idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(b)]);
        }
        out[k] = std::move(D);
    }
    return out;
}

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(BlockPartition partition,
                                                         std::vector<MatrixXc> blocks,
                                                         Real min_rcond)
    : partition_(std::move(partition)) {
    factorize(std::move(blocks), min_rcond);
}

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(
    BlockPartition partition, const std::vector<const op::SparseOperator::Matrix*>& sources,
    Real min_rcond)
    : partition_(std::move(partition)) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<MatrixXc> blocks = extract_diagonal_blocks(partition_, sources);
    extract_s_ = seconds_since(t0);
    factorize(std::move(blocks), min_rcond);
}

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(
    BlockPartition partition, const std::vector<const op::SparseOperator::Matrix*>& sources,
    MissingBlockEntries missing, Real min_rcond)
    : partition_(std::move(partition)) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<MatrixXc> blocks = extract_diagonal_blocks(partition_, sources, missing, &missing_);
    extract_s_ = seconds_since(t0);
    factorize(std::move(blocks), min_rcond);
}

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(BlockPartition partition,
                                                         const MatrixXc& A, Real min_rcond)
    : partition_(std::move(partition)) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<MatrixXc> blocks = extract_diagonal_blocks(partition_, A);
    extract_s_ = seconds_since(t0);
    factorize(std::move(blocks), min_rcond);
}

void BlockDiagonalPreconditioner::factorize(std::vector<MatrixXc> blocks, Real min_rcond) {
    const char* who = "BlockDiagonalPreconditioner";
    if (!(std::isfinite(min_rcond) && min_rcond >= 0.0))
        throw std::invalid_argument(std::string(who) + ": min_rcond must be finite and >= 0");
    Index n = 0;
    for (const std::vector<Index>& b : partition_.blocks) n += static_cast<Index>(b.size());
    (void)map_partition(partition_, n, who);
    const std::size_t K = partition_.blocks.size();
    if (K == 0)
        throw std::invalid_argument(std::string(who) + ": the partition has no blocks");
    if (blocks.size() != K) {
        throw std::invalid_argument(std::string(who) + ": " + std::to_string(blocks.size()) +
                                    " block matrices for " + std::to_string(K) + " blocks");
    }
    for (std::size_t k = 0; k < K; ++k) {
        const auto m = static_cast<Index>(partition_.blocks[k].size());
        if (blocks[k].rows() != m || blocks[k].cols() != m) {
            throw std::invalid_argument(
                std::string(who) + ": block " + std::to_string(block_id(partition_, k)) + " is " +
                std::to_string(blocks[k].rows()) + " x " + std::to_string(blocks[k].cols()) +
                ", expected " + std::to_string(m) + " x " + std::to_string(m));
        }
    }
    n_ = n;
    const auto t0 = std::chrono::steady_clock::now();
    lu_.resize(K);
    std::vector<Real> rcond(K, 0.0);
    std::vector<std::uint8_t> finite(K, 1);
    std::vector<Index> zero_pivot(K, -1);  // first exactly zero U diagonal entry, -1 if none
    const auto nk = static_cast<std::int64_t>(K);
    // Largest blocks first would balance better, but the dynamic schedule of single blocks is
    // enough for the near-field partitions (blocks of similar size). With EIGEN_USE_BLAS (the
    // BLAS/LAPACK builds) the blocked LU's trailing updates call zgemm / ztrsm from inside this
    // parallel region: an OpenMP-built OpenBLAS (the MSYS2 and the usual Linux packages) runs
    // them single-threaded there, and the bitwise thread-count independence of the factors
    // (tested) relies on that; a pthreads-built BLAS may start its own threads per call (set
    // OPENBLAS_NUM_THREADS=1 for such builds). Without BLAS, Eigen's own kernels are serial here.
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t kk = 0; kk < nk; ++kk) {
        const auto k = static_cast<std::size_t>(kk);
        if (!blocks[k].allFinite()) {
            finite[k] = 0;
            continue;
        }
        lu_[k].compute(blocks[k]);
        MatrixXc().swap(blocks[k]);  // release the copy early
        // ADR 0012 decision 8: an exactly zero pivot makes Eigen's rcond() estimate meaningless
        // (it divides by zero and can return a finite value), so the U diagonal is checked
        // directly and such a block is refused whatever the estimate says.
        const auto d = lu_[k].matrixLU().diagonal();
        for (Index i = 0; i < d.size(); ++i) {
            if (d(i) == Complex(0.0, 0.0)) {
                zero_pivot[k] = i;
                break;
            }
        }
        rcond[k] = zero_pivot[k] >= 0 ? 0.0 : lu_[k].rcond();
    }
    factor_s_ = seconds_since(t0);
    min_block_ = std::numeric_limits<Index>::max();
    max_block_ = 0;
    min_rcond_seen_ = std::numeric_limits<Real>::infinity();
    for (std::size_t k = 0; k < K; ++k) {
        const auto m = static_cast<Index>(partition_.blocks[k].size());
        min_block_ = std::min(min_block_, m);
        max_block_ = std::max(max_block_, m);
        if (finite[k] == 0) {
            throw std::runtime_error(std::string(who) + ": block " +
                                     std::to_string(block_id(partition_, k)) + " (" +
                                     std::to_string(m) + " unknowns) has non-finite entries");
        }
        if (zero_pivot[k] >= 0) {
            throw std::runtime_error(std::string(who) + ": block " +
                                     std::to_string(block_id(partition_, k)) + " (" +
                                     std::to_string(m) +
                                     " unknowns) is singular or ill-conditioned: exactly zero "
                                     "pivot on the U diagonal of its LU (position " +
                                     std::to_string(zero_pivot[k]) + ")");
        }
        // !(rcond >= min) also catches a NaN estimate.
        if (!(rcond[k] >= min_rcond) || rcond[k] == 0.0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.3e", rcond[k]);
            char lim[64];
            std::snprintf(lim, sizeof(lim), "%.1e", min_rcond);
            throw std::runtime_error(std::string(who) + ": block " +
                                     std::to_string(block_id(partition_, k)) + " (" +
                                     std::to_string(m) +
                                     " unknowns) is singular or ill-conditioned: estimated "
                                     "reciprocal condition number " +
                                     buf + " < " + lim);
        }
        if (rcond[k] < min_rcond_seen_) {
            min_rcond_seen_ = rcond[k];
            worst_id_ = block_id(partition_, k);
        }
    }
}

const Eigen::PartialPivLU<MatrixXc>& BlockDiagonalPreconditioner::block_lu(Index k) const {
    if (k < 0 || k >= num_blocks())
        throw std::out_of_range("BlockDiagonalPreconditioner::block_lu: block out of range");
    return lu_[static_cast<std::size_t>(k)];
}

void BlockDiagonalPreconditioner::apply(const VectorXc& r, VectorXc& z) const {
    if (r.size() != n_) {
        throw std::invalid_argument("BlockDiagonalPreconditioner::apply: vector size " +
                                    std::to_string(r.size()) + " != " + std::to_string(n_));
    }
    if (z.size() != n_)
        z.resize(n_);  // never taken when z aliases r (same size)
    const auto nk = static_cast<std::int64_t>(lu_.size());
    const Complex* const rp = r.data();
    Complex* const zp = z.data();
#pragma omp parallel
    {
        std::vector<Complex> w(static_cast<std::size_t>(max_block_));
#pragma omp for schedule(dynamic, 1)
        for (std::int64_t kk = 0; kk < nk; ++kk) {
            const auto k = static_cast<std::size_t>(kk);
            const std::vector<Index>& idx = partition_.blocks[k];
            const auto m = static_cast<Index>(idx.size());
            const MatrixXc& LU = lu_[k].matrixLU();
            const auto& perm = lu_[k].permutationP().indices();
            // A = P^{-1} L U: w = P r(I_k), then L w' = w (unit lower), U y = w'.
            for (Index a = 0; a < m; ++a)
                w[static_cast<std::size_t>(perm(a))] = rp[idx[static_cast<std::size_t>(a)]];
            const Complex* const lu = LU.data();
            for (Index j = 0; j < m; ++j) {
                const Complex wj = w[static_cast<std::size_t>(j)];
                const Complex* col = lu + j * m;
                for (Index i = j + 1; i < m; ++i) w[static_cast<std::size_t>(i)] -= col[i] * wj;
            }
            for (Index j = m - 1; j >= 0; --j) {
                const Complex* col = lu + j * m;
                const Complex wj = w[static_cast<std::size_t>(j)] / col[j];
                w[static_cast<std::size_t>(j)] = wj;
                for (Index i = 0; i < j; ++i) w[static_cast<std::size_t>(i)] -= col[i] * wj;
            }
            for (Index a = 0; a < m; ++a)
                zp[idx[static_cast<std::size_t>(a)]] = w[static_cast<std::size_t>(a)];
        }
    }
}

std::size_t BlockDiagonalPreconditioner::memory_bytes() const {
    std::size_t b = sizeof(*this);
    for (std::size_t k = 0; k < lu_.size(); ++k) {
        const auto m = static_cast<std::size_t>(partition_.blocks[k].size());
        // LU factors, the permutation and the row transpositions (int each), the index list.
        b += m * m * sizeof(Complex) + 2 * m * sizeof(int) + m * sizeof(Index) +
             sizeof(Eigen::PartialPivLU<MatrixXc>) + sizeof(std::vector<Index>);
    }
    b += partition_.ids.size() * sizeof(Index);
    return b;
}

std::string BlockDiagonalPreconditioner::describe() const {
    const Real mean = lu_.empty() ? 0.0 : static_cast<Real>(n_) / static_cast<Real>(lu_.size());
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "block-diagonal: %lld blocks of %lld / %.1f / %lld unknowns (min / mean / max), "
                  "%.1f MB, setup %.3f s (extraction %.3f s, LU %.3f s), min rcond %.2e (block "
                  "%lld)",
                  static_cast<long long>(lu_.size()), static_cast<long long>(min_block_), mean,
                  static_cast<long long>(max_block_), static_cast<double>(memory_bytes()) / 1e6,
                  setup_seconds(), extract_s_, factor_s_, min_rcond_seen_,
                  static_cast<long long>(worst_id_));
    std::string s = buf;
    if (missing_ > 0) {
        std::snprintf(buf, sizeof(buf), ", %lld of the %lld block entries in no source (zero)",
                      static_cast<long long>(missing_), static_cast<long long>(block_entries()));
        s += buf;
    }
    return s;
}

Index BlockDiagonalPreconditioner::block_entries() const {
    Index e = 0;
    for (const std::vector<Index>& b : partition_.blocks) {
        const auto m = static_cast<Index>(b.size());
        e += m * m;
    }
    return e;
}

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
