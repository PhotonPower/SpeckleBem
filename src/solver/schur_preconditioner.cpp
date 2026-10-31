/// @file schur_preconditioner.cpp
/// solver::SchurComplementPreconditioner (WP-SC2, ADR 0012 decision 2).
#include "specklebem/solver/schur_preconditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace specklebem::solver {

namespace {

using Matrix = SchurComplementPreconditioner::Matrix;
constexpr const char* kWho = "SchurComplementPreconditioner";

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

Index block_id(const BlockPartition& p, std::size_t k) {
    return p.ids.empty() ? static_cast<Index>(k) : p.ids[k];
}

/// Block and position (within the block) of every unknown of one half, indexed by the local
/// index (global - first index of the half).
struct HalfMap {
    std::vector<Index> block;
    std::vector<Index> local;
};

/// Validates a partition of the half [lo, lo + N) of [0, 2N) and maps it.
HalfMap map_half(const BlockPartition& p, Index N, Index lo, const char* what) {
    const std::string who = std::string(kWho) + ": " + what + " blocks: ";
    if (!p.ids.empty() && p.ids.size() != p.blocks.size()) {
        throw std::invalid_argument(who + std::to_string(p.ids.size()) + " block ids for " +
                                    std::to_string(p.blocks.size()) + " blocks");
    }
    if (p.blocks.empty())
        throw std::invalid_argument(who + "no block");
    HalfMap m;
    m.block.assign(sz(N), -1);
    m.local.assign(sz(N), -1);
    Index count = 0;
    const char* other = lo == 0 ? "an M unknown" : "a J unknown";
    for (std::size_t k = 0; k < p.blocks.size(); ++k) {
        const std::vector<Index>& b = p.blocks[k];
        const std::string id = std::to_string(block_id(p, k));
        if (b.empty())
            throw std::invalid_argument(who + "block " + id + " is empty");
        for (std::size_t a = 0; a < b.size(); ++a) {
            const Index i = b[a];
            if (i < 0 || i >= 2 * N) {
                throw std::invalid_argument(who + "index " + std::to_string(i) + " of block " + id +
                                            " outside [0, " + std::to_string(2 * N) + ")");
            }
            if (i < lo || i >= lo + N) {
                throw std::invalid_argument(
                    who + "block " + id + " holds index " + std::to_string(i) + ", " + other +
                    " (expected [" + std::to_string(lo) + ", " + std::to_string(lo + N) + "))");
            }
            const std::size_t l = sz(i - lo);
            if (m.block[l] >= 0) {
                throw std::invalid_argument(who + "index " + std::to_string(i) +
                                            " belongs to more than one block (or twice to block " +
                                            id + ")");
            }
            m.block[l] = static_cast<Index>(k);
            m.local[l] = static_cast<Index>(a);
        }
        count += static_cast<Index>(b.size());
    }
    if (count != N) {
        throw std::invalid_argument(who + "the blocks hold " + std::to_string(count) + " of the " +
                                    std::to_string(N) +
                                    " unknowns (every unknown must belong to exactly one block)");
    }
    return m;
}

/// z(I_k) = LU_k^{-1} r(I_k) for every block (indices local to the half; `in` and `out` may be
/// the same array): permutation and forward / back substitution with explicit loops in a fixed
/// order, one block per task (as BlockDiagonalPreconditioner::apply), so the result is bitwise
/// independent of the thread count.
void solve_blocks(const std::vector<Eigen::PartialPivLU<MatrixXc>>& lu,
                  const std::vector<std::vector<Index>>& blocks, Index max_block, const Complex* in,
                  Complex* out) {
    const auto nk = static_cast<std::int64_t>(lu.size());
#pragma omp parallel
    {
        std::vector<Complex> w(sz(max_block));
#pragma omp for schedule(dynamic, 1)
        for (std::int64_t kk = 0; kk < nk; ++kk) {
            const auto k = static_cast<std::size_t>(kk);
            const std::vector<Index>& idx = blocks[k];
            const auto m = static_cast<Index>(idx.size());
            const MatrixXc& LU = lu[k].matrixLU();
            const auto& perm = lu[k].permutationP().indices();
            for (Index a = 0; a < m; ++a) w[sz(perm(a))] = in[idx[sz(a)]];
            const Complex* const f = LU.data();
            for (Index j = 0; j < m; ++j) {
                const Complex wj = w[sz(j)];
                const Complex* col = f + j * m;
                for (Index i = j + 1; i < m; ++i) w[sz(i)] -= col[i] * wj;
            }
            for (Index j = m - 1; j >= 0; --j) {
                const Complex* col = f + j * m;
                const Complex wj = w[sz(j)] / col[j];
                w[sz(j)] = wj;
                for (Index i = 0; i < j; ++i) w[sz(i)] -= col[i] * wj;
            }
            for (Index a = 0; a < m; ++a) out[idx[sz(a)]] = w[sz(a)];
        }
    }
}

/// Factorises the blocks (released one by one) in parallel; errors (non-finite entries,
/// rcond below the limit) are thrown for the lowest failing block. Returns the smallest rcond
/// and its block id through the arguments.
void factorize_blocks(std::vector<MatrixXc>& blocks, const BlockPartition& p, Real min_rcond,
                      const char* what, std::vector<Eigen::PartialPivLU<MatrixXc>>& lu,
                      Real& min_seen, Index& worst_id) {
    const std::size_t K = blocks.size();
    lu.resize(K);
    std::vector<Real> rcond(K, 0.0);
    std::vector<std::uint8_t> finite(K, 1);
    std::vector<std::uint8_t> zero(K, 0);  // an exactly zero pivot on the U diagonal
    const auto nk = static_cast<std::int64_t>(K);
    // As BlockDiagonalPreconditioner::factorize: the BLAS calls of the blocked LU nested in this
    // parallel region run single-threaded with an OpenMP-built OpenBLAS (bitwise thread-count
    // independence of the factors, tested).
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t kk = 0; kk < nk; ++kk) {
        const auto k = static_cast<std::size_t>(kk);
        if (!blocks[k].allFinite()) {
            finite[k] = 0;
            continue;
        }
        lu[k].compute(blocks[k]);
        MatrixXc().swap(blocks[k]);
        // An exactly zero pivot makes Eigen's estimate meaningless (it divides by zero and can
        // return a finite value): such a block is singular.
        const auto d = lu[k].matrixLU().diagonal();
        const bool zero_pivot = (d.array() == Complex(0.0, 0.0)).any();
        zero[k] = zero_pivot ? 1 : 0;
        rcond[k] = zero_pivot ? 0.0 : lu[k].rcond();
    }
    min_seen = std::numeric_limits<Real>::infinity();
    worst_id = -1;
    for (std::size_t k = 0; k < K; ++k) {
        const std::string head = std::string(kWho) + ": " + what + " block " +
                                 std::to_string(block_id(p, k)) + " (" +
                                 std::to_string(p.blocks[k].size()) + " unknowns)";
        if (finite[k] == 0)
            throw std::runtime_error(head + " has non-finite entries");
        if (zero[k] != 0) {
            throw std::runtime_error(
                head +
                " is singular or ill-conditioned: exactly zero pivot on the U diagonal of its LU "
                "(estimated reciprocal condition number taken as 0: Eigen's estimate is "
                "meaningless for a zero pivot; ADR 0012 decision 8)");
        }
        if (!(rcond[k] >= min_rcond) || rcond[k] == 0.0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.3e", rcond[k]);
            char lim[64];
            std::snprintf(lim, sizeof(lim), "%.1e", min_rcond);
            throw std::runtime_error(head +
                                     " is singular or ill-conditioned: estimated reciprocal "
                                     "condition number " +
                                     buf + " < " + lim);
        }
        if (rcond[k] < min_seen) {
            min_seen = rcond[k];
            worst_id = block_id(p, k);
        }
    }
}

/// Min and max block size.
std::pair<Index, Index> block_size_range(const BlockPartition& p) {
    Index lo = std::numeric_limits<Index>::max();
    Index hi = 0;
    for (const std::vector<Index>& b : p.blocks) {
        lo = std::min(lo, static_cast<Index>(b.size()));
        hi = std::max(hi, static_cast<Index>(b.size()));
    }
    return {lo, hi};
}

/// One stored entry of C(R_b, I_k) or B(I_k, R_b) during the formation of S̃_b: the J block k,
/// the position in the M block (row of C / column of B), the position in the J block, the
/// value.
struct Coupling {
    Index k;
    Index m_pos;
    Index j_pos;
    Complex value;
};

/// Per-row offsets of the column N in every row of A (2N x 2N, compressed RowMajor), checking
/// that the column indices of each row are sorted. Counts the entries right of the split in the
/// first N rows (B) and left of it in the last N rows (C).
std::vector<Index> row_splits(const Matrix& A, Index N, Index& nnz_b, Index& nnz_c) {
    const Index n2 = 2 * N;
    std::vector<Index> split(sz(n2), 0);
    const Index* outer = A.outerIndexPtr();
    const Index* inner = A.innerIndexPtr();
    Index unsorted = n2;  // the lowest row with unsorted columns (n2: none)
    Index b = 0;
    Index c = 0;
#pragma omp parallel for schedule(static) reduction(+ : b, c) reduction(min : unsorted)
    for (Index r = 0; r < n2; ++r) {
        Index s = outer[r + 1];
        for (Index q = outer[r]; q < outer[r + 1]; ++q) {
            if (q > outer[r] && inner[q] <= inner[q - 1])
                unsorted = std::min(unsorted, r);
            if (inner[q] >= N && s == outer[r + 1])
                s = q;
        }
        split[sz(r)] = s;
        if (r < N)
            b += outer[r + 1] - s;
        else
            c += s - outer[r];
    }
    if (unsorted < n2) {
        throw std::invalid_argument(std::string(kWho) + ": row " + std::to_string(unsorted) +
                                    " of a source has unsorted or repeated column indices");
    }
    nnz_b += b;
    nnz_c += c;
    return split;
}

std::string megabytes(std::size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / 1e6);
    return buf;
}

}  // namespace

SchurComplementPreconditioner::SchurComplementPreconditioner(Matrix Zn, Index N,
                                                             BlockPartition j_blocks,
                                                             BlockPartition m_blocks,
                                                             SchurParams params)
    : owned_(std::move(Zn)) {
    if (N < 1)
        throw std::invalid_argument(std::string(kWho) + ": N must be >= 1");
    if (owned_.rows() != 2 * N || owned_.cols() != 2 * N) {
        throw std::invalid_argument(
            std::string(kWho) + ": the matrix is " + std::to_string(owned_.rows()) + " x " +
            std::to_string(owned_.cols()) + ", expected 2N x 2N = " + std::to_string(2 * N) +
            " x " + std::to_string(2 * N));
    }
    owned_.makeCompressed();
    setup({&owned_}, std::move(j_blocks), std::move(m_blocks), params);
    // Keep only B and C (apply() never reads A and D), and recompute the row offsets.
    const Index n = N;
    owned_.prune([n](Index row, Index col, const Complex&) { return (row < n) != (col < n); });
    owned_.makeCompressed();
    Index nb = 0;
    Index nc = 0;
    split_.assign(1, row_splits(owned_, n_, nb, nc));
}

SchurComplementPreconditioner::SchurComplementPreconditioner(std::vector<const Matrix*> sources,
                                                             Index N, BlockPartition j_blocks,
                                                             BlockPartition m_blocks,
                                                             SchurParams params) {
    if (N < 1)
        throw std::invalid_argument(std::string(kWho) + ": N must be >= 1");
    if (sources.empty())
        throw std::invalid_argument(std::string(kWho) + ": no source matrix");
    for (const Matrix* s : sources) {
        if (s == nullptr)
            throw std::invalid_argument(std::string(kWho) + ": null source matrix");
        if (s->rows() != 2 * N || s->cols() != 2 * N) {
            throw std::invalid_argument(
                std::string(kWho) + ": a source is " + std::to_string(s->rows()) + " x " +
                std::to_string(s->cols()) + ", expected 2N x 2N = " + std::to_string(2 * N) +
                " x " + std::to_string(2 * N));
        }
        if (!s->isCompressed())
            throw std::invalid_argument(std::string(kWho) + ": source matrix not compressed");
    }
    setup(std::move(sources), std::move(j_blocks), std::move(m_blocks), params);
}

void SchurComplementPreconditioner::setup(std::vector<const Matrix*> sources,
                                          BlockPartition j_blocks, BlockPartition m_blocks,
                                          const SchurParams& params) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!(std::isfinite(params.min_rcond) && params.min_rcond >= 0.0))
        throw std::invalid_argument(std::string(kWho) + ": min_rcond must be finite and >= 0");
    const Index N = sources.front()->rows() / 2;
    n_ = N;
    const HalfMap jm = map_half(j_blocks, N, 0, "J");
    const HalfMap mm = map_half(m_blocks, N, N, "M");
    j_part_ = std::move(j_blocks);
    m_part_ = std::move(m_blocks);
    m_local_ = m_part_.blocks;
    for (std::vector<Index>& b : m_local_)
        for (Index& i : b) i -= N;
    std::tie(min_j_, max_j_) = block_size_range(j_part_);
    std::tie(min_m_, max_m_) = block_size_range(m_part_);
    sources_ = std::move(sources);
    split_.clear();
    nnz_b_ = 0;
    nnz_c_ = 0;
    for (const Matrix* s : sources_) split_.push_back(row_splits(*s, N, nnz_b_, nnz_c_));
    prep_s_ = seconds_since(t0);

    // --- Ã: the J blocks of A, extracted (entries stored in several sources summed) and
    // factorised.
    const auto t1 = std::chrono::steady_clock::now();
    const std::size_t KJ = j_part_.blocks.size();
    {
        std::vector<MatrixXc> blocks(KJ);
        std::vector<Index> missing(KJ, 0);
        const auto nk = static_cast<std::int64_t>(KJ);
#pragma omp parallel for schedule(dynamic, 1)
        for (std::int64_t kk = 0; kk < nk; ++kk) {
            const auto k = static_cast<std::size_t>(kk);
            const std::vector<Index>& rows = j_part_.blocks[k];
            const auto m = static_cast<Index>(rows.size());
            MatrixXc A = MatrixXc::Zero(m, m);
            std::vector<std::uint8_t> seen(sz(m * m), 0);
            for (std::size_t s = 0; s < sources_.size(); ++s) {
                const Index* outer = sources_[s]->outerIndexPtr();
                const Index* inner = sources_[s]->innerIndexPtr();
                const Complex* val = sources_[s]->valuePtr();
                const std::vector<Index>& split = split_[s];
                for (Index a = 0; a < m; ++a) {
                    const Index r = rows[sz(a)];
                    for (Index q = outer[r]; q < split[sz(r)]; ++q) {
                        const Index c = inner[q];
                        if (jm.block[sz(c)] != static_cast<Index>(k))
                            continue;
                        const Index b = jm.local[sz(c)];
                        A(a, b) += val[q];
                        seen[sz(b * m + a)] = 1;
                    }
                }
            }
            missing[k] = static_cast<Index>(std::count(seen.begin(), seen.end(), std::uint8_t{0}));
            blocks[k] = std::move(A);
        }
        missing_a_ = 0;
        for (const Index x : missing) missing_a_ += x;
        factorize_blocks(blocks, j_part_, params.min_rcond, "J", lu_a_, min_rcond_a_, worst_a_id_);
    }
    a_s_ = seconds_since(t1);

    // --- S̃_b: the M blocks of D - C Ã^{-1} B on the pattern of D.
    const auto t2 = std::chrono::steady_clock::now();
    const std::size_t KM = m_local_.size();
    std::vector<MatrixXc> sblocks(KM);
    {
        // B column by column (local M column j: its J rows and values), filled in the order
        // (source, row, entry): deterministic.
        std::vector<Index> bt_ptr(sz(N + 1), 0);
        for (std::size_t s = 0; s < sources_.size(); ++s) {
            const Index* outer = sources_[s]->outerIndexPtr();
            const Index* inner = sources_[s]->innerIndexPtr();
            for (Index r = 0; r < N; ++r)
                for (Index q = split_[s][sz(r)]; q < outer[r + 1]; ++q)
                    ++bt_ptr[sz(inner[q] - N + 1)];
        }
        for (Index j = 0; j < N; ++j) bt_ptr[sz(j + 1)] += bt_ptr[sz(j)];
        std::vector<Index> bt_row(sz(bt_ptr[sz(N)]));
        std::vector<Complex> bt_val(sz(bt_ptr[sz(N)]));
        {
            std::vector<Index> cursor(bt_ptr.begin(), bt_ptr.end() - 1);
            for (std::size_t s = 0; s < sources_.size(); ++s) {
                const Index* outer = sources_[s]->outerIndexPtr();
                const Index* inner = sources_[s]->innerIndexPtr();
                const Complex* val = sources_[s]->valuePtr();
                for (Index r = 0; r < N; ++r) {
                    for (Index q = split_[s][sz(r)]; q < outer[r + 1]; ++q) {
                        const std::size_t pos = sz(cursor[sz(inner[q] - N)]++);
                        bt_row[pos] = r;
                        bt_val[pos] = val[q];
                    }
                }
            }
        }

        std::vector<Index> masked(KM, 0);
        std::vector<Index> pairs(KM, 0);
        std::vector<double> macs(KM, 0.0);
        const auto nb = static_cast<std::int64_t>(KM);
#pragma omp parallel
        {
            std::vector<Coupling> cc;
            std::vector<Coupling> bb;
            std::vector<Index> rpos(sz(max_m_), -1);
            std::vector<Index> qpos(sz(max_m_), -1);
            std::vector<Index> rlist;
            std::vector<Index> qlist;
            MatrixXc Ck, Bk, T, G;
#pragma omp for schedule(dynamic, 1)
            for (std::int64_t bi = 0; bi < nb; ++bi) {
                const auto b = static_cast<std::size_t>(bi);
                const std::vector<Index>& rows = m_local_[b];
                const auto m = static_cast<Index>(rows.size());
                MatrixXc S = MatrixXc::Zero(m, m);
                std::vector<std::uint8_t> mask(sz(m * m), 0);
                cc.clear();
                bb.clear();
                // D(R_b, R_b) with its pattern, and the stored entries of C(R_b, :).
                for (std::size_t s = 0; s < sources_.size(); ++s) {
                    const Index* outer = sources_[s]->outerIndexPtr();
                    const Index* inner = sources_[s]->innerIndexPtr();
                    const Complex* val = sources_[s]->valuePtr();
                    const std::vector<Index>& split = split_[s];
                    for (Index a = 0; a < m; ++a) {
                        const Index g = N + rows[sz(a)];
                        for (Index q = outer[g]; q < split[sz(g)]; ++q) {
                            const Index c = inner[q];
                            cc.push_back({jm.block[sz(c)], a, jm.local[sz(c)], val[q]});
                        }
                        for (Index q = split[sz(g)]; q < outer[g + 1]; ++q) {
                            const Index c = inner[q] - N;
                            if (mm.block[sz(c)] != static_cast<Index>(b))
                                continue;
                            const Index col = mm.local[sz(c)];
                            S(a, col) += val[q];
                            mask[sz(col * m + a)] = 1;
                        }
                    }
                }
                masked[b] =
                    static_cast<Index>(std::count(mask.begin(), mask.end(), std::uint8_t{0}));
                // The stored entries of B(:, R_b).
                for (Index qq = 0; qq < m; ++qq) {
                    const Index j = rows[sz(qq)];
                    for (Index e = bt_ptr[sz(j)]; e < bt_ptr[sz(j + 1)]; ++e) {
                        const Index r = bt_row[sz(e)];
                        bb.push_back({jm.block[sz(r)], qq, jm.local[sz(r)], bt_val[sz(e)]});
                    }
                }
                const auto by_block = [](const Coupling& x, const Coupling& y) {
                    return x.k < y.k;
                };
                std::stable_sort(cc.begin(), cc.end(), by_block);
                std::stable_sort(bb.begin(), bb.end(), by_block);
                // Every J block coupled both ways, in increasing k.
                std::size_t ic = 0;
                std::size_t ib = 0;
                while (ic < cc.size() && ib < bb.size()) {
                    const Index kc = cc[ic].k;
                    const Index kb = bb[ib].k;
                    if (kc < kb) {
                        while (ic < cc.size() && cc[ic].k == kc) ++ic;
                        continue;
                    }
                    if (kb < kc) {
                        while (ib < bb.size() && bb[ib].k == kb) ++ib;
                        continue;
                    }
                    const Index k = kc;
                    const std::size_t c0 = ic;
                    const std::size_t b0 = ib;
                    while (ic < cc.size() && cc[ic].k == k) ++ic;
                    while (ib < bb.size() && bb[ib].k == k) ++ib;
                    // The rows of C_kb and the columns of B_kb in ascending order within the M
                    // block: independent of the order of the sources and of their entries, so
                    // the products (and S̃) are too.
                    rlist.clear();
                    qlist.clear();
                    for (std::size_t e = c0; e < ic; ++e) {
                        Index& p = rpos[sz(cc[e].m_pos)];
                        if (p < 0) {
                            p = 0;
                            rlist.push_back(cc[e].m_pos);
                        }
                    }
                    for (std::size_t e = b0; e < ib; ++e) {
                        Index& p = qpos[sz(bb[e].m_pos)];
                        if (p < 0) {
                            p = 0;
                            qlist.push_back(bb[e].m_pos);
                        }
                    }
                    std::sort(rlist.begin(), rlist.end());
                    std::sort(qlist.begin(), qlist.end());
                    for (std::size_t x = 0; x < rlist.size(); ++x)
                        rpos[sz(rlist[x])] = static_cast<Index>(x);
                    for (std::size_t y = 0; y < qlist.size(); ++y)
                        qpos[sz(qlist[y])] = static_cast<Index>(y);
                    const auto nk = static_cast<Index>(j_part_.blocks[sz(k)].size());
                    const auto nr = static_cast<Index>(rlist.size());
                    const auto nq = static_cast<Index>(qlist.size());
                    Ck.setZero(nr, nk);
                    Bk.setZero(nk, nq);
                    for (std::size_t e = c0; e < ic; ++e)
                        Ck(rpos[sz(cc[e].m_pos)], cc[e].j_pos) += cc[e].value;
                    for (std::size_t e = b0; e < ib; ++e)
                        Bk(bb[e].j_pos, qpos[sz(bb[e].m_pos)]) += bb[e].value;
                    T = lu_a_[sz(k)].solve(Bk);
                    G.resize(nr, nq);
                    G.noalias() = Ck * T;
                    for (Index y = 0; y < nq; ++y) {
                        const Index col = qlist[sz(y)];
                        for (Index x = 0; x < nr; ++x) {
                            const Index row = rlist[sz(x)];
                            if (mask[sz(col * m + row)] != 0)
                                S(row, col) -= G(x, y);
                        }
                    }
                    ++pairs[b];
                    macs[b] +=
                        static_cast<double>(nk) * static_cast<double>(nk) *
                            static_cast<double>(nq) +
                        static_cast<double>(nr) * static_cast<double>(nk) * static_cast<double>(nq);
                    for (const Index a : rlist) rpos[sz(a)] = -1;
                    for (const Index a : qlist) qpos[sz(a)] = -1;
                }
                sblocks[b] = std::move(S);
            }
        }
        masked_s_ = 0;
        pairs_ = 0;
        macs_ = 0;
        for (std::size_t b = 0; b < KM; ++b) {
            masked_s_ += masked[b];
            pairs_ += pairs[b];
            macs_ += macs[b];
        }
    }
    form_s_ = seconds_since(t2);

    // --- S̃_b factorised.
    const auto t3 = std::chrono::steady_clock::now();
    factorize_blocks(sblocks, m_part_, params.min_rcond, "Schur-complement M", lu_s_, min_rcond_s_,
                     worst_s_id_);
    s_s_ = seconds_since(t3);
}

const Eigen::PartialPivLU<MatrixXc>& SchurComplementPreconditioner::j_block_lu(Index k) const {
    if (k < 0 || k >= num_j_blocks())
        throw std::out_of_range(std::string(kWho) + "::j_block_lu: block out of range");
    return lu_a_[sz(k)];
}

const Eigen::PartialPivLU<MatrixXc>& SchurComplementPreconditioner::m_block_lu(Index b) const {
    if (b < 0 || b >= num_m_blocks())
        throw std::out_of_range(std::string(kWho) + "::m_block_lu: block out of range");
    return lu_s_[sz(b)];
}

void SchurComplementPreconditioner::apply(const VectorXc& r, VectorXc& z) const {
    const Index N = n_;
    if (r.size() != 2 * N) {
        throw std::invalid_argument(std::string(kWho) + "::apply: vector size " +
                                    std::to_string(r.size()) + " != " + std::to_string(2 * N));
    }
    const Complex* const rp = r.data();
    VectorXc y1(N), y2(N), t(N);
    Complex* const tp = t.data();
    // y1 = Ã^{-1} r1.
    solve_blocks(lu_a_, j_part_.blocks, max_j_, rp, y1.data());
    // t = r2 - C y1, row by row (sources in order).
    {
        const Complex* const y = y1.data();
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < N; ++i) {
            const Index g = N + i;
            Complex acc(0.0, 0.0);
            for (std::size_t s = 0; s < sources_.size(); ++s) {
                const Index* outer = sources_[s]->outerIndexPtr();
                const Index* inner = sources_[s]->innerIndexPtr();
                const Complex* val = sources_[s]->valuePtr();
                const Index end = split_[s][sz(g)];
                for (Index q = outer[g]; q < end; ++q) acc += val[q] * y[inner[q]];
            }
            tp[i] = rp[g] - acc;
        }
    }
    // y2 = S̃_b^{-1} t.
    solve_blocks(lu_s_, m_local_, max_m_, tp, y2.data());
    // t = B y2.
    {
        const Complex* const y = y2.data();
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < N; ++i) {
            Complex acc(0.0, 0.0);
            for (std::size_t s = 0; s < sources_.size(); ++s) {
                const Index* outer = sources_[s]->outerIndexPtr();
                const Index* inner = sources_[s]->innerIndexPtr();
                const Complex* val = sources_[s]->valuePtr();
                for (Index q = split_[s][sz(i)]; q < outer[i + 1]; ++q)
                    acc += val[q] * y[inner[q] - N];
            }
            tp[i] = acc;
        }
    }
    // x1 = y1 - Ã^{-1} t; x2 = y2 (r is no longer read: z may alias it).
    solve_blocks(lu_a_, j_part_.blocks, max_j_, tp, tp);
    if (z.size() != 2 * N)
        z.resize(2 * N);
    z.head(N) = y1 - t;
    z.tail(N) = y2;
}

Index SchurComplementPreconditioner::j_block_entries() const {
    Index e = 0;
    for (const std::vector<Index>& b : j_part_.blocks) {
        const auto m = static_cast<Index>(b.size());
        e += m * m;
    }
    return e;
}

Index SchurComplementPreconditioner::m_block_entries() const {
    Index e = 0;
    for (const std::vector<Index>& b : m_part_.blocks) {
        const auto m = static_cast<Index>(b.size());
        e += m * m;
    }
    return e;
}

std::size_t SchurComplementPreconditioner::memory_bytes() const {
    std::size_t bytes = sizeof(*this);
    const auto lu_bytes = [](const std::vector<Eigen::PartialPivLU<MatrixXc>>& lu,
                             const std::vector<std::vector<Index>>& blocks) {
        std::size_t s = 0;
        for (std::size_t k = 0; k < lu.size(); ++k) {
            const std::size_t m = blocks[k].size();
            // LU factors, the permutation and the row transpositions (int each), the index list.
            s += m * m * sizeof(Complex) + 2 * m * sizeof(int) + m * sizeof(Index) +
                 sizeof(Eigen::PartialPivLU<MatrixXc>) + sizeof(std::vector<Index>);
        }
        return s;
    };
    bytes += lu_bytes(lu_a_, j_part_.blocks);
    bytes += lu_bytes(lu_s_, m_part_.blocks);
    for (const std::vector<Index>& b : m_local_) bytes += b.size() * sizeof(Index);
    bytes += (j_part_.ids.size() + m_part_.ids.size()) * sizeof(Index);
    for (const std::vector<Index>& s : split_) bytes += s.size() * sizeof(Index);
    if (!borrows_sources()) {
        bytes += static_cast<std::size_t>(owned_.nonZeros()) * (sizeof(Complex) + sizeof(Index)) +
                 static_cast<std::size_t>(owned_.outerSize() + 1) * sizeof(Index);
    }
    return bytes;
}

std::string SchurComplementPreconditioner::describe() const {
    const auto mean = [](Index n, std::size_t k) {
        return k == 0 ? 0.0 : static_cast<double>(n) / static_cast<double>(k);
    };
    char buf[800];
    std::snprintf(
        buf, sizeof(buf),
        "Schur complement: J blocks %lld (%lld / %.1f / %lld), M blocks %lld (%lld / %.1f / %lld) "
        "unknowns (min / mean / max), nnz(B) %lld, nnz(C) %lld (%s), %lld coupling pairs (%.1f "
        "GFlop), %s, setup %.3f s (A LU %.3f s, Schur formation %.3f s, Schur LU %.3f s), min "
        "rcond A %.2e (block %lld), Schur %.2e (block %lld)",
        static_cast<long long>(lu_a_.size()), static_cast<long long>(min_j_),
        mean(n_, lu_a_.size()), static_cast<long long>(max_j_),
        static_cast<long long>(lu_s_.size()), static_cast<long long>(min_m_),
        mean(n_, lu_s_.size()), static_cast<long long>(max_m_), static_cast<long long>(nnz_b_),
        static_cast<long long>(nnz_c_), borrows_sources() ? "borrowed" : "owned",
        static_cast<long long>(pairs_), 8.0 * macs_ / 1e9, megabytes(memory_bytes()).c_str(),
        setup_seconds(), a_s_, form_s_, s_s_, min_rcond_a_, static_cast<long long>(worst_a_id_),
        min_rcond_s_, static_cast<long long>(worst_s_id_));
    std::string s = buf;
    if (missing_a_ > 0) {
        std::snprintf(
            buf, sizeof(buf), ", %lld of the %lld J-block entries of A in no source (zero)",
            static_cast<long long>(missing_a_), static_cast<long long>(j_block_entries()));
        s += buf;
    }
    if (masked_s_ > 0) {
        std::snprintf(buf, sizeof(buf),
                      ", %lld of the %lld M-block entries of the Schur complement outside the "
                      "pattern of D (zero)",
                      static_cast<long long>(masked_s_), static_cast<long long>(m_block_entries()));
        s += buf;
    }
    return s;
}

}  // namespace specklebem::solver
