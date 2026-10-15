// Dense LU timing (WP-LU1): factorisation and solve time of solver::DenseLu with the LAPACK
// backend (zgetrf/zgetrs, multithreaded through the BLAS) and the Eigen backend (PartialPivLU)
// on random complex n x n matrices (normal entries, fixed seeds) or on a matrix read from a file
// (--matrix). For every matrix, backend and repetition: the factorisation (equilibrated copy, LU,
// Hager/Higham rcond estimate), optionally the bare LU of a plain copy (--raw: zgetrf or Eigen
// PartialPivLU, nothing else), one solve with one right-hand side (LU solve + up to two
// refinement steps, i.e. 3 matvecs with Z), one solve with --nrhs right-hand sides, the relative
// residuals, and the relative difference to the first backend's solution.
//
// --matrix FILE.bin: int64 n, then the n x n complex<double> entries column-major (16 n^2 bytes,
// native little-endian), as dumped from a Simulation's dense operator; replaces the random sizes.
// Memory: Z plus the LU copy, 2 x 16 n^2 bytes (n = 11 440: 4.2 GB), one more copy with --raw.
// Threads: OMP_NUM_THREADS (the OpenBLAS of MSYS2 uses the same OpenMP pool). Record:
// benchmarks/results/dense_lu.md.
//
// Usage: specklebem_dense_lu [--n 2000,5000,11440] [--matrix FILE.bin] [--nrhs 8]
//                            [--backends lapack,eigen] [--repeat 1] [--raw 0|1]
//        A backend that is not available in the build (LAPACK) is reported and skipped.
#include "specklebem/solver/direct.hpp"

#include <Eigen/LU>

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

#ifdef SPECKLEBEM_HAVE_LAPACK
extern "C" void zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda,
                        int* ipiv, int* info);
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

/// Reads the --matrix format (int64 n, then 16 n^2 bytes column-major).
MatrixXc read_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::invalid_argument("cannot open " + path);
    std::int64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof n);
    const auto bytes = static_cast<std::uintmax_t>(std::filesystem::file_size(path));
    const auto un = static_cast<std::uintmax_t>(n);
    if (!in || n <= 0 || bytes != sizeof n + 16 * un * un)
        throw std::invalid_argument(path + ": not an int64 n + n x n complex<double> file");
    MatrixXc Z(n, n);
    in.read(reinterpret_cast<char*>(Z.data()), static_cast<std::streamsize>(16 * un * un));
    if (!in)
        throw std::runtime_error(path + ": short read");
    return Z;
}

/// Wall time of the bare LU of a copy of Z (no equilibration, no rcond).
Real raw_lu_seconds(const MatrixXc& Z, solver::LuBackend backend) {
    MatrixXc A = Z;
    const auto t0 = Clock::now();
    if (backend == solver::LuBackend::Lapack) {
#ifdef SPECKLEBEM_HAVE_LAPACK
        const int n = static_cast<int>(A.rows());
        std::vector<int> ipiv(static_cast<std::size_t>(n));
        int info = 0;
        zgetrf_(&n, &n, A.data(), &n, ipiv.data(), &info);
        if (info != 0)
            throw std::runtime_error("zgetrf info = " + std::to_string(info));
#endif
    } else {
        const Eigen::PartialPivLU<Eigen::Ref<MatrixXc>> lu(A);
    }
    return std::chrono::duration<Real>(Clock::now() - t0).count();
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

Real secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<Real>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<Index> sizes{2000, 5000, 11440};
        std::string matrix_file;
        Index nrhs = 8;
        int repeat = 1;
        bool raw = false;
        std::vector<std::string> backends{"lapack", "eigen"};
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + a);
            const std::string v = argv[++i];
            if (a == "--n") {
                sizes.clear();
                for (const std::string& s : split(v)) sizes.push_back(std::stoll(s));
            } else if (a == "--matrix") {
                matrix_file = v;
            } else if (a == "--nrhs") {
                nrhs = std::stoll(v);
            } else if (a == "--backends") {
                backends = split(v);
            } else if (a == "--repeat") {
                repeat = std::stoi(v);
            } else if (a == "--raw") {
                raw = std::stoi(v) != 0;
            } else {
                throw std::invalid_argument("unknown option " + a);
            }
        }
        int threads = 1;
#ifdef SPECKLEBEM_HAVE_OPENMP
        threads = omp_get_max_threads();
#endif
        const char* ob_env = std::getenv("OPENBLAS_NUM_THREADS");
        std::printf(
            "# dense LU timing: lapack_available = %d, OpenMP threads = %d, "
            "OPENBLAS_NUM_THREADS = %s, nrhs = %lld\n",
            solver::lapack_available() ? 1 : 0, threads, ob_env ? ob_env : "unset",
            static_cast<long long>(nrhs));
        std::printf("# %7s %7s %3s %9s %9s %9s %9s %10s %10s %10s %10s\n", "n", "backend", "rep",
                    "factor_s", "raw_lu_s", "solve1_s", "solveM_s", "rcond", "residual", "res_M",
                    "diff");
        const std::size_t cases = matrix_file.empty() ? sizes.size() : 1;
        for (std::size_t c = 0; c < cases; ++c) {
            const MatrixXc Z =
                matrix_file.empty()
                    ? random_matrix(sizes[c], sizes[c], 1000 + static_cast<std::uint64_t>(sizes[c]))
                    : read_matrix(matrix_file);
            const Index n = Z.rows();
            const VectorXc b = random_matrix(n, 1, 7).col(0);
            const MatrixXc B = random_matrix(n, nrhs, 8);
            VectorXc x_ref;
            for (const std::string& name : backends) {
                solver::LuBackend backend = solver::LuBackend::Eigen;
                if (name == "lapack") {
                    if (!solver::lapack_available()) {
                        std::printf("  %7lld %7s   (not available in this build)\n",
                                    static_cast<long long>(n), name.c_str());
                        continue;
                    }
                    backend = solver::LuBackend::Lapack;
                } else if (name != "eigen") {
                    throw std::invalid_argument("unknown backend " + name);
                }
                for (int rep = 0; rep < repeat; ++rep) {
                    const Real raw_s = raw ? raw_lu_seconds(Z, backend) : 0.0;
                    const auto t0 = Clock::now();
                    const solver::DenseLu lu(Z, backend);
                    const auto t1 = Clock::now();
                    solver::DirectSolveInfo info;
                    const VectorXc x = lu.solve(b, &info);
                    const auto t2 = Clock::now();
                    solver::DirectSolveInfo info_m;
                    const MatrixXc X = lu.solve(B, &info_m);
                    const auto t3 = Clock::now();
                    Real diff = 0;
                    if (x_ref.size() == 0) {
                        x_ref = x;
                    } else {
                        diff = (x - x_ref).norm() / x_ref.norm();
                    }
                    std::printf(
                        "  %7lld %7s %3d %9.2f %9.2f %9.3f %9.3f %10.3e %10.3e %10.3e %10.3e\n",
                        static_cast<long long>(n), name.c_str(), rep, secs(t0, t1), raw_s,
                        secs(t1, t2), secs(t2, t3), info.rcond, info.residual, info_m.residual,
                        diff);
                    std::fflush(stdout);
                    (void)X;
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
