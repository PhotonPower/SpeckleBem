// GMRES orthogonalisation cost (WP-G1): time per iteration versus the Krylov dimension k for the
// Gram-Schmidt variants of solver::gmres (serial MGS with the 0.7 re-orthogonalisation criterion,
// parallel CGS2) on a synthetic operator with a cheap O(n) matvec, so that the orthogonalisation
// dominates as in the MLFMM solves of WP22b1 (42-66 % of the Ag solve time with MGS).
//
// Operator: y = d .* x + U (W^H x), d uniform in the disk |z - 1| < --radius, U, W random n x 2
// (fixed seeds). GMRES runs exactly --iters iterations (tolerance 1e-300, no restart), so every
// variant builds a basis of the same size; the time per iteration at k is the mean over the
// iterations k - 9 .. k (callback timestamps). Optionally (--blas-reference) the same two
// products per pass are timed with Eigen GEMV on one contiguous n x k matrix (dispatched to
// BLAS when the build defines EIGEN_USE_BLAS).
//
// Memory: 16 n (iters + 1) bytes for the basis (n = 2e5, 1000 iterations: 3.2 GB), twice that
// with --blas-reference. Record: benchmarks/results/gmres_orthogonalization.md.
//
// Usage: specklebem_gmres_orthogonalization [--n 200000] [--iters 1000] [--radius 0.95]
//            [--variants mgs,cgs2] [--threads 0] [--blas-reference]
//        --threads takes a comma-separated list of OpenMP thread counts for CGS2 (0 = the
//        default); MGS is serial and runs once.
#include "specklebem/core/logging.hpp"
#include "specklebem/operator/linear_operator.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

namespace {

using namespace specklebem;
using Clock = std::chrono::steady_clock;

MatrixXc random_matrix(Index rows, Index cols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    MatrixXc A(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) A(i, j) = Complex(nd(rng), nd(rng));
    }
    return A;
}

class DiagonalPlusLowRank final : public op::LinearOperator {
public:
    DiagonalPlusLowRank(Index n, Real radius)
        : d_(n), U_(random_matrix(n, 2, 2)), W_(random_matrix(n, 2, 3)) {
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<Real> u(0.0, 1.0);
        for (Index i = 0; i < n; ++i) {
            d_(i) = 1.0 + std::polar(radius * std::sqrt(u(rng)), 2.0 * constants::pi * u(rng));
        }
        U_ /= static_cast<Real>(n);
    }
    [[nodiscard]] Index rows() const override { return d_.size(); }
    [[nodiscard]] Index cols() const override { return d_.size(); }
    void apply(const VectorXc& x, VectorXc& y) const override {
        const Complex c0 = W_.col(0).dot(x);
        const Complex c1 = W_.col(1).dot(x);
        y = d_.cwiseProduct(x) + c0 * U_.col(0) + c1 * U_.col(1);
    }
    [[nodiscard]] std::string describe() const override { return "DiagonalPlusLowRank"; }
    [[nodiscard]] std::size_t memory_bytes() const override { return 0; }

private:
    VectorXc d_;
    MatrixXc U_;
    MatrixXc W_;
};

struct Options {
    Index n = 200000;
    int iters = 1000;
    Real radius = 0.95;
    std::vector<std::string> variants{"mgs", "cgs2"};
    std::vector<int> threads{0};
    bool blas_reference = false;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--n") {
            o.n = static_cast<Index>(std::stod(value()));
        } else if (a == "--iters") {
            o.iters = std::stoi(value());
        } else if (a == "--radius") {
            o.radius = std::stod(value());
        } else if (a == "--variants") {
            o.variants = split(value());
        } else if (a == "--threads") {
            o.threads.clear();
            for (const auto& t : split(value())) o.threads.push_back(std::stoi(t));
        } else if (a == "--blas-reference") {
            o.blas_reference = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (o.n < 1 || o.iters < 10)
        throw std::invalid_argument("--n must be >= 1 and --iters >= 10");
    return o;
}

int default_threads() {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void set_threads(int t) {
#ifdef SPECKLEBEM_HAVE_OPENMP
    omp_set_num_threads(t);
#else
    (void)t;
#endif
}

/// Mean seconds per iteration over the iterations k - 9 .. k (stamps[k] = end of iteration k).
double per_iteration(const std::vector<double>& stamps, int k) {
    return (stamps[static_cast<std::size_t>(k)] - stamps[static_cast<std::size_t>(k - 10)]) / 10.0;
}

std::vector<int> sample_points(int iters) {
    std::vector<int> ks;
    for (const int k : {10, 100, 250, 500, 750, 1000, 1500, 2000}) {
        if (k <= iters)
            ks.push_back(k);
    }
    if (ks.back() != iters)
        ks.push_back(iters);
    return ks;
}

void run_variant(const Options& o, const op::LinearOperator& A, const VectorXc& b,
                 const std::string& variant, int threads, double matvec_s) {
    solver::GmresParams p;
    p.tolerance = 1e-300;
    p.max_iter = o.iters;
    p.verbose = false;
    if (variant == "mgs") {
        p.orthogonalization = solver::Orthogonalization::MGS;
    } else if (variant == "cgs2") {
        p.orthogonalization = solver::Orthogonalization::CGS2;
    } else {
        throw std::invalid_argument("unknown variant " + variant);
    }
    set_threads(threads);
    std::vector<double> stamps;
    stamps.reserve(static_cast<std::size_t>(o.iters) + 1);
    const solver::IdentityPreconditioner none;
    Clock::time_point t0 = Clock::now();
    const auto cb = [&](int, Real) {
        stamps.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
    };
    stamps.push_back(0.0);
    t0 = Clock::now();
    const solver::GmresResult r = solver::gmres(A, b, none, p, cb);
    std::printf("| %-11s | %7d | %8.2f | %8.2f | %5.1f %% | %5d | %9.2e |", variant.c_str(),
                threads, r.wall_seconds, r.orthogonalization_seconds,
                100.0 * r.orthogonalization_seconds / r.wall_seconds, r.reorthogonalizations,
                r.residual_history.back());
    for (const int k : sample_points(o.iters))
        std::printf(" %7.2f |", 1e3 * per_iteration(stamps, k));
    std::printf(" %.2f |\n", 1e3 * matvec_s);
    std::fflush(stdout);
}

/// Times one orthogonalisation pass (h = V^H w; w -= V h) with Eigen GEMV on a contiguous basis.
void blas_reference(const Options& o, const VectorXc& w0) {
    std::printf(
        "\nEigen GEMV pair (h = V^H w; w -= V h) on a contiguous n x k basis, %d threads:\n",
        default_threads());
    std::printf("| k | seconds per pass |\n|---|---|\n");
    const MatrixXc V = random_matrix(o.n, o.iters, 4);
    for (const int k : sample_points(o.iters)) {
        double best = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            VectorXc w = w0;
            const auto t = Clock::now();
            const VectorXc h = V.leftCols(k).adjoint() * w;
            w.noalias() -= V.leftCols(k) * h;
            best = std::min(best, std::chrono::duration<double>(Clock::now() - t).count());
        }
        std::printf("| %d | %.4f |\n", k, best);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        log::set_level(log::Level::warn);
        const int t_default = default_threads();
        std::printf(
            "GMRES orthogonalisation benchmark: n = %lld, %d iterations, disk radius %.3f, "
            "default OpenMP threads %d, basis %.2f GB\n",
            static_cast<long long>(o.n), o.iters, o.radius, t_default,
            16.0 * static_cast<double>(o.n) * (o.iters + 1) / 1e9);
        const DiagonalPlusLowRank A(o.n, o.radius);
        const VectorXc b = random_matrix(o.n, 1, 5).col(0);
        VectorXc y;
        const auto tm = Clock::now();
        for (int i = 0; i < 20; ++i) A.apply(b, y);
        const double matvec_s = std::chrono::duration<double>(Clock::now() - tm).count() / 20;

        std::printf("\n| variant | threads | total s | orth s | share | reorth | final res |");
        for (const int k : sample_points(o.iters)) std::printf(" ms/it k=%d |", k);
        std::printf(" matvec ms |\n|---|---|---|---|---|---|---|");
        for (std::size_t i = 0; i < sample_points(o.iters).size(); ++i) std::printf("---|");
        std::printf("---|\n");
        for (const auto& v : o.variants) {
            if (v == "mgs") {
                run_variant(o, A, b, v, t_default, matvec_s);
                continue;
            }
            for (const int t : o.threads) run_variant(o, A, b, v, t > 0 ? t : t_default, matvec_s);
        }
        set_threads(t_default);
        if (o.blas_reference)
            blas_reference(o, b);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gmres_orthogonalization: %s\n", e.what());
        return 1;
    }
}
