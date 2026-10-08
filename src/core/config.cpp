#include "specklebem/core/config.hpp"

namespace specklebem {

bool has_openmp() {
#ifdef SPECKLEBEM_HAVE_OPENMP
    return true;
#else
    return false;
#endif
}
bool has_cuda() {
#ifdef SPECKLEBEM_HAVE_CUDA
    return true;
#else
    return false;
#endif
}
bool has_hdf5() {
#ifdef SPECKLEBEM_HAVE_HDF5
    return true;
#else
    return false;
#endif
}
bool has_blas_lapack() {
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
    return true;
#else
    return false;
#endif
}

}  // namespace specklebem
