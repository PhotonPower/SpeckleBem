/// @file incomplete_lu_preconditioner.cpp
/// solver::IncompleteLuPreconditioner (WP-SC3, ADR 0012 decision 6): own dual-threshold ILUT
/// (Saad 1994) of the near-field blocks A and D; see the header for why not Eigen::IncompleteLUT.
#include "specklebem/solver/incomplete_lu_preconditioner.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/system_memory.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace specklebem::solver {

const char* to_string(IlutOrdering ordering) {
    switch (ordering) {
        case IlutOrdering::Amd:
            return "amd";
        case IlutOrdering::Natural:
            return "natural";
    }
    return "invalid";
}

namespace {

using Matrix = IncompleteLuPreconditioner::Matrix;
using Factor = IncompleteLuPreconditioner::Factor;
constexpr const char* kWho = "IncompleteLuPreconditioner";

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

std::string megabytes(std::size_t bytes) {
    if (bytes == std::numeric_limits<std::size_t>::max())
        return "unlimited";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / 1e6);
    return buf;
}

std::string sci(Real x) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3e", x);
    return buf;
}

/// One block in compressed row storage: rows with sorted, unique columns.
struct Csr {
    Index n = 0;
    std::vector<Index> ptr;
    std::vector<Index> col;
    std::vector<Complex> val;
};

/// The block [off, off + N)^2 of the sum of the sources (row r of the block = row off + r of the
/// sources, column c = column off + c). Entries of one position are summed in source order.
/// @throws std::invalid_argument for a non-finite entry.
Csr extract_block(const std::vector<const Matrix*>& sources, Index off, Index N, const char* name) {
    Csr b;
    b.n = N;
    b.ptr.assign(sz(N) + 1, 0);
    Index total = 0;
    for (Index r = 0; r < N; ++r) {
        for (const Matrix* s : sources) {
            for (Matrix::InnerIterator it(*s, off + r); it; ++it)
                if (it.col() >= off && it.col() < off + N)
                    ++total;
        }
    }
    b.col.resize(sz(total));
    b.val.resize(sz(total));
    std::vector<std::pair<Index, Complex>> row;
    Index out = 0;
    for (Index r = 0; r < N; ++r) {
        row.clear();
        for (const Matrix* s : sources) {
            for (Matrix::InnerIterator it(*s, off + r); it; ++it) {
                const Index c = it.col();
                if (c < off || c >= off + N)
                    continue;
                const Complex v = it.value();
                if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) {
                    throw std::invalid_argument(std::string(kWho) + ": block " + name +
                                                ": non-finite entry (" + std::to_string(r) + ", " +
                                                std::to_string(c - off) + ")");
                }
                row.emplace_back(c - off, v);
            }
        }
        // Stable: duplicates are summed in source order (deterministic rounding).
        std::stable_sort(row.begin(), row.end(),
                         [](const auto& x, const auto& y) { return x.first < y.first; });
        for (std::size_t e = 0; e < row.size(); ++e) {
            if (out > b.ptr[sz(r)] && b.col[sz(out - 1)] == row[e].first) {
                b.val[sz(out - 1)] += row[e].second;
            } else {
                b.col[sz(out)] = row[e].first;
                b.val[sz(out)] = row[e].second;
                ++out;
            }
        }
        b.ptr[sz(r) + 1] = out;
    }
    b.col.resize(sz(out));
    b.val.resize(sz(out));
    return b;
}

/// Elimination order: perm[k] = the original index of the k-th unknown.
std::vector<Index> make_ordering(const Csr& b, IlutOrdering ordering) {
    std::vector<Index> perm(sz(b.n));
    if (ordering == IlutOrdering::Natural || b.n == 1) {
        std::iota(perm.begin(), perm.end(), Index{0});
        return perm;
    }
    // AMD on the pattern of B + B^T. The CSR arrays of B are the CSC arrays of B^T.
    // Eigen::AMDOrdering symmetrises its input again (A^T + A); the explicit B + B^T is kept for
    // clarity (and so as not to change the recorded orderings).
    using Pattern = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;
    const auto nnz = static_cast<Index>(b.col.size());
    Pattern S;
    {
        const std::vector<double> ones(b.col.size(), 1.0);
        const Eigen::Map<const Pattern> bt(b.n, b.n, nnz, b.ptr.data(), b.col.data(), ones.data());
        S = bt;
        const Pattern t = S.transpose();
        S = S + t;  // the union of the two patterns (a sparse sum keeps every stored position)
    }
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> P;
    Eigen::AMDOrdering<Index> amd;
    amd(S, P);
    if (P.size() != b.n)
        throw std::logic_error(std::string(kWho) + ": AMD returned a permutation of wrong size");
    // Eigen's AMD (CSparse's cs_amd) returns P with P.indices()[k] = the original index of the
    // k-th unknown of the elimination order (Eigen's IncompleteLUT permutes by P^{-1}, i.e.
    // moves original row P(k) to position k).
    for (Index k = 0; k < b.n; ++k) perm[sz(k)] = P.indices()(k);
    return perm;
}

/// Off-diagonal entries row k may keep: max(1, round(fill x nnz_row)) - 1, at most n - 1.
Index off_diagonal_cap(Index row_nnz, Real fill, Index n) {
    const Real c = std::max(1.0, std::round(fill * static_cast<Real>(row_nnz)));
    const Real off = std::min(c - 1.0, static_cast<Real>(n - 1));
    return static_cast<Index>(off);
}

/// Sum over the rows of off_diagonal_cap.
Index off_diagonal_bound(const Csr& b, Real fill) {
    Index total = 0;
    for (Index r = 0; r < b.n; ++r)
        total += off_diagonal_cap(b.ptr[sz(r) + 1] - b.ptr[sz(r)], fill, b.n);
    return total;
}

/// Factor memory for n rows and `off` off-diagonal entries (as Factor::memory_bytes()).
std::size_t factor_bytes(Index n, Index off) {
    return sz(off) * IncompleteLuPreconditioner::kBytesPerEntry + sz(n) * sizeof(Complex) +
           (2 * sz(n) + 1) * sizeof(Index) + sz(n) * sizeof(Index);
}

/// An L or U candidate of the current row: column (elimination numbering), stored value (l_ik for
/// L, u_ij for U) and the magnitude that ranks it (|w_j| before the division by the pivot).
struct Candidate {
    Index col;
    Complex value;
    Real mag;
};

/// ILUT of the block b in the order perm (see the class doc). @throws std::runtime_error for a
/// zero row or a zero, tiny or non-finite pivot.
void factorize(const Csr& b, std::vector<Index> perm, const IlutParams& p, const char* name,
               Factor& f) {
    const Index n = b.n;
    f.n = n;
    f.input_nnz = static_cast<Index>(b.col.size());
    std::vector<Index> iperm(sz(n));
    for (Index k = 0; k < n; ++k) iperm[sz(perm[sz(k)])] = k;
    f.perm = std::move(perm);
    const Index bound = off_diagonal_bound(b, p.fill_factor);
    f.col.clear();
    f.val.clear();
    f.col.reserve(sz(bound));
    f.val.reserve(sz(bound));
    f.ptr.assign(sz(n) + 1, 0);
    f.mid.assign(sz(n), 0);
    f.diag.assign(sz(n), Complex(0.0, 0.0));

    std::vector<Complex> w(sz(n), Complex(0.0, 0.0));
    std::vector<std::uint8_t> in(sz(n), 0);
    std::vector<Index> touched;
    std::vector<Index> heap;  // min-heap of the lower-part columns still to eliminate
    std::vector<Candidate> cand;
    const std::greater<Index> later;
    Real min_ratio = std::numeric_limits<Real>::infinity();
    Real norm_inf = 0.0;  // max_i sum_j |b_ij|
    Index macs = 0;       // complex multiply-adds of the row updates
    for (Index k = 0; k < n; ++k) {
        const Index r = f.perm[sz(k)];
        touched.clear();
        heap.clear();
        cand.clear();
        Real norm2 = 0.0;
        Real norm1 = 0.0;
        const Index row_nnz = b.ptr[sz(r) + 1] - b.ptr[sz(r)];
        for (Index e = b.ptr[sz(r)]; e < b.ptr[sz(r) + 1]; ++e) {
            const Index j = iperm[sz(b.col[sz(e)])];
            w[sz(j)] = b.val[sz(e)];
            in[sz(j)] = 1;
            touched.push_back(j);
            if (j < k)
                heap.push_back(j);
            norm2 += std::norm(b.val[sz(e)]);
            norm1 += std::abs(b.val[sz(e)]);
        }
        if (!(norm2 > 0.0)) {
            throw std::runtime_error(std::string(kWho) + ": block " + name + ": row " +
                                     std::to_string(r) + " is zero (singular block)");
        }
        norm_inf = std::max(norm_inf, norm1);
        std::make_heap(heap.begin(), heap.end(), later);
        if (in[sz(k)] == 0) {
            in[sz(k)] = 1;
            touched.push_back(k);
        }
        const Real rn = std::sqrt(norm2);
        const Real thr = p.drop_tol * rn;
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), later);
            const Index q = heap.back();
            heap.pop_back();
            const Complex wq = w[sz(q)];
            const Real mag = std::abs(wq);
            if (!(mag > thr)) {
                w[sz(q)] = Complex(0.0, 0.0);
                continue;
            }
            const Complex l = wq / f.diag[sz(q)];
            cand.push_back({q, l, mag});
            macs += f.ptr[sz(q) + 1] - f.mid[sz(q)];
            for (Index e = f.mid[sz(q)]; e < f.ptr[sz(q) + 1]; ++e) {
                const Index j = f.col[sz(e)];
                const Complex upd = l * f.val[sz(e)];
                if (in[sz(j)] == 0) {
                    in[sz(j)] = 1;
                    w[sz(j)] = -upd;
                    touched.push_back(j);
                    if (j < k) {
                        heap.push_back(j);
                        std::push_heap(heap.begin(), heap.end(), later);
                    }
                } else {
                    w[sz(j)] -= upd;
                }
            }
        }
        const Complex piv = w[sz(k)];
        const Real ratio = std::abs(piv) / rn;
        if (!(ratio > p.min_pivot_rel) || !std::isfinite(ratio)) {
            char lim[32];
            std::snprintf(lim, sizeof(lim), "%.1e", p.min_pivot_rel);
            throw std::runtime_error(std::string(kWho) + ": block " + name +
                                     ": zero or tiny pivot in row " + std::to_string(r) +
                                     " (elimination step " + std::to_string(k) + "): |u_ii| / " +
                                     "||row||_2 = " + sci(ratio) + " <= min_pivot_rel " + lim +
                                     " (ILUT breakdown; no pivot is replaced)");
        }
        min_ratio = std::min(min_ratio, ratio);
        for (const Index j : touched) {
            if (j <= k)
                continue;
            const Real mag = std::abs(w[sz(j)]);
            if (mag > thr)
                cand.push_back({j, w[sz(j)], mag});
        }
        const auto cap = sz(off_diagonal_cap(row_nnz, p.fill_factor, n));
        if (cand.size() > cap) {
            const auto larger = [](const Candidate& x, const Candidate& y) {
                return x.mag > y.mag || (x.mag == y.mag && x.col < y.col);
            };
            std::nth_element(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(cap),
                             cand.end(), larger);
            cand.resize(cap);
        }
        std::sort(cand.begin(), cand.end(),
                  [](const Candidate& x, const Candidate& y) { return x.col < y.col; });
        // ptr[k] was set at the end of row k - 1 (ptr[0] = 0); ptr[k + 1] closes this row now,
        // since the elimination of row k + 1 reads the U part of row k.
        std::size_t c = 0;
        for (; c < cand.size() && cand[c].col < k; ++c) {
            f.col.push_back(cand[c].col);
            f.val.push_back(cand[c].value);
        }
        f.mid[sz(k)] = static_cast<Index>(f.col.size());
        for (; c < cand.size(); ++c) {
            f.col.push_back(cand[c].col);
            f.val.push_back(cand[c].value);
        }
        f.ptr[sz(k) + 1] = static_cast<Index>(f.col.size());
        f.diag[sz(k)] = piv;
        for (const Index j : touched) {
            w[sz(j)] = Complex(0.0, 0.0);
            in[sz(j)] = 0;
        }
    }

    f.col.shrink_to_fit();
    f.val.shrink_to_fit();
    f.min_pivot_ratio = min_ratio;
    f.norm_inf = norm_inf;
    f.macs = static_cast<double>(macs);
}

/// z = P^T U^{-1} L^{-1} P r for one block (r, z of length n; may not alias).
void solve_block(const Factor& f, const Complex* r, Complex* z) {
    const Index n = f.n;
    std::vector<Complex> y(sz(n));
    for (Index k = 0; k < n; ++k) y[sz(k)] = r[f.perm[sz(k)]];
    for (Index k = 0; k < n; ++k) {
        Complex s = y[sz(k)];
        for (Index e = f.ptr[sz(k)]; e < f.mid[sz(k)]; ++e) s -= f.val[sz(e)] * y[sz(f.col[sz(e)])];
        y[sz(k)] = s;
    }
    for (Index k = n - 1; k >= 0; --k) {
        Complex s = y[sz(k)];
        for (Index e = f.mid[sz(k)]; e < f.ptr[sz(k) + 1]; ++e)
            s -= f.val[sz(e)] * y[sz(f.col[sz(e)])];
        y[sz(k)] = s / f.diag[sz(k)];
    }
    for (Index k = 0; k < n; ++k) z[f.perm[sz(k)]] = y[sz(k)];
}

}  // namespace

void validate_ilut_params(const IlutParams& p) {
    if (!(std::isfinite(p.drop_tol) && p.drop_tol >= 0.0))
        throw std::invalid_argument(std::string(kWho) + ": drop_tol must be finite and >= 0");
    if (!(std::isfinite(p.fill_factor) && p.fill_factor > 0.0))
        throw std::invalid_argument(std::string(kWho) + ": fill_factor must be finite and > 0");
    if (!(std::isfinite(p.min_pivot_rel) && p.min_pivot_rel >= 0.0))
        throw std::invalid_argument(std::string(kWho) + ": min_pivot_rel must be finite and >= 0");
    if (p.budget_bytes && *p.budget_bytes == 0)
        throw std::invalid_argument(std::string(kWho) + ": budget_bytes must be > 0 when set");
    if (p.ordering != IlutOrdering::Amd && p.ordering != IlutOrdering::Natural)
        throw std::invalid_argument(std::string(kWho) + ": invalid IlutOrdering value");
}

namespace {

void check_block(const Matrix& M, const char* name) {
    if (M.rows() == 0 || M.cols() == 0)
        throw std::invalid_argument(std::string(kWho) + ": block " + name + " is empty");
    if (M.rows() != M.cols()) {
        throw std::invalid_argument(std::string(kWho) + ": block " + name + " is " +
                                    std::to_string(M.rows()) + " x " + std::to_string(M.cols()) +
                                    ", expected a square matrix");
    }
}

}  // namespace

Real IncompleteLuPreconditioner::Factor::fill_ratio() const {
    return input_nnz > 0 ? static_cast<Real>(factor_nnz()) / static_cast<Real>(input_nnz) : 0.0;
}

std::size_t IncompleteLuPreconditioner::Factor::memory_bytes() const {
    return col.capacity() * sizeof(Index) + val.capacity() * sizeof(Complex) +
           diag.capacity() * sizeof(Complex) + ptr.capacity() * sizeof(Index) +
           mid.capacity() * sizeof(Index) + perm.capacity() * sizeof(Index);
}

IncompleteLuPreconditioner::IncompleteLuPreconditioner(const Matrix& A, const Matrix& D,
                                                       const IlutParams& params)
    : params_(params) {
    check_block(A, "A");
    check_block(D, "D");
    if (A.rows() != D.rows()) {
        throw std::invalid_argument(std::string(kWho) + ": A is " + std::to_string(A.rows()) +
                                    " x " + std::to_string(A.rows()) + ", D is " +
                                    std::to_string(D.rows()) + " x " + std::to_string(D.rows()) +
                                    " (both must be N x N)");
    }
    setup({&A}, 0, {&D}, 0, A.rows());
}

IncompleteLuPreconditioner::IncompleteLuPreconditioner(const std::vector<const Matrix*>& sources,
                                                       Index N, const IlutParams& params)
    : params_(params) {
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
    }
    setup(sources, 0, sources, N, N);
}

void IncompleteLuPreconditioner::setup(const std::vector<const Matrix*>& sources_a, Index offset_a,
                                       const std::vector<const Matrix*>& sources_d, Index offset_d,
                                       Index N) {
    const auto t0 = std::chrono::steady_clock::now();
    validate_ilut_params(params_);
    n_ = N;
    if (params_.budget_bytes) {
        budget_ = *params_.budget_bytes;
    } else {
        const std::size_t avail = core::available_memory_bytes();
        if (avail == 0) {
            SBEM_WARN("{}: the available memory is unknown, the factor memory is not limited",
                      kWho);
            budget_ = std::numeric_limits<std::size_t>::max();
        } else {
            budget_ = avail / 2;
        }
    }

    // Extraction of both blocks (parallel; each sequential).
    Csr blocks[2];
    Factor* factors[2] = {&a_, &d_};
    const char* names[2] = {"A", "D"};
    const std::vector<const Matrix*>* srcs[2] = {&sources_a, &sources_d};
    const Index offs[2] = {offset_a, offset_d};
    std::exception_ptr err[2];
#pragma omp parallel for schedule(static, 1) num_threads(2)
    for (int bi = 0; bi < 2; ++bi) {
        try {
            const auto te = std::chrono::steady_clock::now();
            blocks[bi] = extract_block(*srcs[bi], offs[bi], N, names[bi]);
            factors[bi]->extract_s = seconds_since(te);
        } catch (...) {
            err[bi] = std::current_exception();
        }
    }
    for (const std::exception_ptr& e : err)
        if (e)
            std::rethrow_exception(e);

    // Budget pre-check against the bound of the fill limit.
    bound_ = 0;
    for (const Csr& b : blocks)
        bound_ += factor_bytes(b.n, off_diagonal_bound(b, params_.fill_factor));
    if (bound_ > budget_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3g", params_.fill_factor);
        throw std::runtime_error(
            std::string(kWho) + ": the factor memory bound " + megabytes(bound_) +
            " (fill factor " + buf + " x nnz(A) " + std::to_string(blocks[0].col.size()) +
            " + nnz(D) " + std::to_string(blocks[1].col.size()) + " entries x " +
            std::to_string(kBytesPerEntry) + " B + per-row arrays) exceeds the budget " +
            megabytes(budget_) + "; not factorised");
    }

    // Ordering and factorisation (parallel over A and D; each sequential and deterministic).
#pragma omp parallel for schedule(static, 1) num_threads(2)
    for (int bi = 0; bi < 2; ++bi) {
        try {
            const auto to = std::chrono::steady_clock::now();
            std::vector<Index> perm = make_ordering(blocks[bi], params_.ordering);
            factors[bi]->order_s = seconds_since(to);
            const auto tf = std::chrono::steady_clock::now();
            factorize(blocks[bi], std::move(perm), params_, names[bi], *factors[bi]);
            factors[bi]->factor_s = seconds_since(tf);
            // Stability indicator of the triangular factors (Chow & Saad 1997), scale-free:
            // condest = ||(L U)^{-1} e||_inf ||B||_inf with e = (1, ..., 1). Unstable factors of
            // an indefinite block can overflow even with every pivot passing the guard.
            const std::vector<Complex> ones(sz(N), Complex(1.0, 0.0));
            std::vector<Complex> z(sz(N));
            solve_block(*factors[bi], ones.data(), z.data());
            Real m = 0.0;
            bool finite = true;
            for (const Complex& v : z) {
                const Real a = std::abs(v);
                finite = finite && std::isfinite(a);
                m = std::max(m, a);
            }
            factors[bi]->condest =
                finite ? m * factors[bi]->norm_inf : std::numeric_limits<Real>::infinity();
            if (!finite) {
                throw std::runtime_error(
                    std::string(kWho) + ": block " + names[bi] +
                    ": unstable triangular factors, ||(L U)^{-1} e||_inf is not finite (ILUT "
                    "breakdown; a smaller drop_tol, a larger fill_factor or another ordering may "
                    "help)");
            }
        } catch (...) {
            err[bi] = std::current_exception();
        }
        blocks[bi] = Csr{};  // release the copy of the block
    }
    for (const std::exception_ptr& e : err)
        if (e)
            std::rethrow_exception(e);
    setup_s_ = seconds_since(t0);
    if (memory_bytes() > budget_) {
        throw std::runtime_error(std::string(kWho) + ": the factor memory " +
                                 megabytes(memory_bytes()) + " exceeds the budget " +
                                 megabytes(budget_) + " (bound was " + megabytes(bound_) +
                                 "); factors released");
    }
}

void IncompleteLuPreconditioner::apply(const VectorXc& r, VectorXc& z) const {
    if (r.size() != size()) {
        throw std::invalid_argument("IncompleteLuPreconditioner::apply: vector size " +
                                    std::to_string(r.size()) + " != " + std::to_string(size()));
    }
    // A separate result vector: r and z may be the same object.
    VectorXc out(size());
    const Factor* factors[2] = {&a_, &d_};
#pragma omp parallel for schedule(static, 1) num_threads(2)
    for (int bi = 0; bi < 2; ++bi)
        solve_block(*factors[bi], r.data() + bi * n_, out.data() + bi * n_);
    z.swap(out);
}

std::size_t IncompleteLuPreconditioner::memory_bytes() const {
    return a_.memory_bytes() + d_.memory_bytes();
}

std::string IncompleteLuPreconditioner::describe() const {
    char buf[900];
    std::snprintf(
        buf, sizeof(buf),
        "ILUT (fill %.3g, drop %.1e, ordering %s): 2 x %lld unknowns, A nnz %lld -> L+U %lld "
        "(fill %.2f, min pivot %.2e, condest %.2e), D nnz %lld -> L+U %lld (fill %.2f, min pivot "
        "%.2e, condest %.2e), %s "
        "(bound %s, budget %s), setup %.2f s (A %.2f s incl. ordering %.2f s, D %.2f s incl. "
        "ordering %.2f s)",
        params_.fill_factor, params_.drop_tol, to_string(params_.ordering),
        static_cast<long long>(n_), static_cast<long long>(a_.input_nnz),
        static_cast<long long>(nonzeros_a()), fill_ratio_a(), a_.min_pivot_ratio, a_.condest,
        static_cast<long long>(d_.input_nnz), static_cast<long long>(nonzeros_d()), fill_ratio_d(),
        d_.min_pivot_ratio, d_.condest, megabytes(memory_bytes()).c_str(),
        megabytes(bound_).c_str(), megabytes(budget_).c_str(), setup_s_, setup_seconds_a(),
        a_.order_s, setup_seconds_d(), d_.order_s);
    return buf;
}

}  // namespace specklebem::solver
