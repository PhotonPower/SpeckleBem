cmake_minimum_required(VERSION 3.24)

project(SpeckleBem
    VERSION 0.1.0
    DESCRIPTION "Rigorous electromagnetic scattering from rough surfaces via SIE + fast BEM"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # needed for the Python module
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SPECKLEBEM_BUILD_TESTS       "Build unit and validation tests"        ON)
option(SPECKLEBEM_BUILD_PYTHON      "Build the Python bindings (pybind11)"   ON)
option(SPECKLEBEM_BUILD_EXAMPLES    "Build C++ examples"                     ON)
option(SPECKLEBEM_BUILD_BENCHMARKS  "Build benchmarks"                       OFF)
option(SPECKLEBEM_ENABLE_OPENMP     "Enable OpenMP shared-memory parallelism" ON)
option(SPECKLEBEM_ENABLE_CUDA       "Enable the CUDA backend (experimental)"  OFF)
option(SPECKLEBEM_ENABLE_HDF5       "Enable HDF5 result output"              OFF)
option(SPECKLEBEM_USE_MKL           "Prefer Intel MKL as BLAS/LAPACK"        OFF)
option(SPECKLEBEM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"     OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Dependencies)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Optional components
# ---------------------------------------------------------------------------
if(SPECKLEBEM_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(SPECKLEBEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SPECKLEBEM_BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

if(SPECKLEBEM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS specklebem specklebem_warnings EXPORT SpeckleBemTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(EXPORT SpeckleBemTargets
    NAMESPACE SpeckleBem::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpeckleBem)

configure_package_config_file(
    cmake/SpeckleBemConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SpeckleBemConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpeckleBem)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SpeckleBemConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SpeckleBemConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SpeckleBemConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpeckleBem)
