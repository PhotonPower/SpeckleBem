# ---------------------------------------------------------------------------
# Third-party dependencies.
#
# Policy: prefer system / package-manager installations (find_package).
# Header-only or small dependencies fall back to FetchContent so that a fresh
# clone builds without manual steps. Heavy dependencies (MKL, CUDA)
# are always found, never fetched.
# ---------------------------------------------------------------------------
include(FetchContent)

# --- Eigen (dense linear algebra, used throughout) ------------------------
# NO_CMAKE_PACKAGE_REGISTRY: a previously fetched Eigen build tree (e.g. from another
# preset) registers itself in ~/.cmake/packages; never pick that up.
find_package(Eigen3 3.4 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system - fetching 3.4.0")
    # SYSTEM (CMake >= 3.25): the fetched headers are system headers for every consumer, so the
    # project warning set (and -Werror in the debug presets) applies to our code, not to Eigen
    # internals. Setting INTERFACE_SYSTEM_INCLUDE_DIRECTORIES on Eigen's target by hand instead
    # breaks Eigen's install(EXPORT) with CMake 4 ("prefixed in the build directory").
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
        SYSTEM)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)  # Eigen's export(PACKAGE) must not pollute ~/.cmake
    FetchContent_MakeAvailable(eigen)
    if(NOT TARGET Eigen3::Eigen)
        # Older Eigen trees do not export the imported target when added as a subdirectory.
        # Include directories of imported targets are system include directories by default.
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
    endif()
endif()

# --- BLAS / LAPACK ---------------------------------------------------------
if(SPECKLEBEM_USE_MKL)
    set(BLA_VENDOR Intel10_64lp)
endif()
# 32-bit (LP64) BLAS/LAPACK integers only: the dense LU and Eigen's BLAS calls pass int. With
# BLA_SIZEOF_INTEGER (CMake >= 3.22) FindBLAS/FindLAPACK search only the LP64 variants of the
# vendors they know (e.g. MKL lp64, openblas rather than openblas64). An ILP64 LAPACK exporting the
# unsuffixed zgetrf_ would otherwise pass the link probe below and then read 64-bit integers from
# 32-bit arguments (memory corruption); an ILP64 build installed under an LP64 library name is
# still not detected.
set(BLA_SIZEOF_INTEGER 4)
find_package(BLAS)
find_package(LAPACK)
if(BLAS_FOUND AND LAPACK_FOUND)
    set(SPECKLEBEM_HAVE_BLAS_LAPACK ON)
else()
    message(WARNING "BLAS/LAPACK not found - dense reference solver will use Eigen only (slow)")
    set(SPECKLEBEM_HAVE_BLAS_LAPACK OFF)
endif()

# The dense LU (solver::DenseLu, WP-LU1) calls the LAPACK routines zgetrf_ and zgetrs_ through
# the Fortran interface with 32-bit (LP64) integers. FindLAPACK checks one routine only (cheev),
# so link a probe that calls exactly these symbols; SPECKLEBEM_HAVE_LAPACK selects the LAPACK
# path in src/solver/direct.cpp (Eigen's PartialPivLU otherwise, reported in
# solver::DirectSolveInfo::backend).
set(SPECKLEBEM_HAVE_LAPACK OFF)
if(SPECKLEBEM_HAVE_BLAS_LAPACK)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    set(CMAKE_REQUIRED_QUIET ON)
    check_cxx_source_compiles([=[
#include <complex>
#include <cstddef>
extern "C" {
void zgetrf_(const int*, const int*, std::complex<double>*, const int*, int*, int*);
void zgetrs_(const char*, const int*, const int*, const std::complex<double>*, const int*,
             const int*, std::complex<double>*, const int*, int*, std::size_t);
}
int main() {
    int n = 1, info = 0, ipiv = 0;
    std::complex<double> a(2.0, 0.0), b(1.0, 0.0);
    zgetrf_(&n, &n, &a, &n, &ipiv, &info);
    zgetrs_("N", &n, &n, &a, &n, &ipiv, &b, &n, &info, 1);
    return info;
}
]=] SPECKLEBEM_LAPACK_GETRF_GETRS_LINK)
    cmake_pop_check_state()
    if(SPECKLEBEM_LAPACK_GETRF_GETRS_LINK)
        set(SPECKLEBEM_HAVE_LAPACK ON)
        message(STATUS "LAPACK zgetrf_/zgetrs_ link: the dense LU uses LAPACK")
    else()
        message(WARNING "LAPACK found but zgetrf_/zgetrs_ do not link - the dense LU uses Eigen")
    endif()
endif()

# --- OpenMP ----------------------------------------------------------------
if(SPECKLEBEM_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# --- CUDA (optional, experimental) -----------------------------------------
if(SPECKLEBEM_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    find_package(CUDAToolkit REQUIRED)
endif()

# --- spdlog (logging) ------------------------------------------------------
find_package(spdlog QUIET CONFIG)
if(NOT spdlog_FOUND)
    # v1.15.3 bundles fmt 11.2: the fmt 10.2 of spdlog 1.14 fails to compile with Clang 22
    # (FMT_STRING consteval check "subexpression not valid in a constant expression").
    # SYSTEM: warnings inside the spdlog / fmt headers are not ours (-Werror in debug presets).
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.15.3
        GIT_SHALLOW    TRUE
        SYSTEM)
    set(SPDLOG_INSTALL ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)
endif()

# --- Catch2 (tests) --------------------------------------------------------
if(SPECKLEBEM_BUILD_TESTS)
    find_package(Catch2 3 QUIET CONFIG)
    if(NOT Catch2_FOUND)
        # >= v3.14: Clang 22 flags the __COUNTER__ in Catch2's TEST_CASE / SECTION macros as a
        # C2y extension (-Wc2y-extensions under -Wpedantic, at the expansion site in our code);
        # Catch2 3.14 suppresses that itself. SYSTEM, as a package-manager Catch2 would be.
        FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.16.1
            GIT_SHALLOW    TRUE
            SYSTEM)
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    endif()
endif()

# --- pybind11 (Python bindings) -------------------------------------------
if(SPECKLEBEM_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 QUIET CONFIG)
    if(NOT pybind11_FOUND)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    TRUE
            SYSTEM)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()
