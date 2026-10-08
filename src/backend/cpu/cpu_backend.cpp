#include "specklebem/backend/cpu/cpu_backend.hpp"

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

namespace specklebem::backend {

std::string CpuBackend::name() const {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return "cpu(eigen, openmp)";
#else
    return "cpu(eigen)";
#endif
}

int CpuBackend::num_threads() const {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void CpuBackend::gemv(const MatrixXc& A, const VectorXc& x, VectorXc& y, Complex alpha,
                      Complex beta) const {
    // Eigen dispatches to BLAS zgemv when EIGEN_USE_BLAS is defined.
    if (beta == Complex(0)) {
        y.noalias() = alpha * (A * x);
    } else {
        y = alpha * (A * x) + beta * y;
    }
}

}  // namespace specklebem::backend
