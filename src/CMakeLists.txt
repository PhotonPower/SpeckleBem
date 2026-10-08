# Core library. Source files are listed explicitly (no GLOB) so that adding a
# file is a conscious, reviewable act.
set(SPECKLEBEM_SOURCES
    core/config.cpp
    core/logging.cpp
    core/timer.cpp
    material/material.cpp
    formulation/formulation.cpp
    excitation/excitation.cpp
    geometry/mesh.cpp
    geometry/sphere.cpp
    geometry/rough_surface.cpp
    geometry/mesh_io.cpp
    basis/rwg.cpp
    kernels/green.cpp
    kernels/quadrature.cpp
    kernels/singularity.cpp
    kernels/operators.cpp
    operator/dense_operator.cpp
    operator/linear_operator.cpp
    operator/assembler.cpp
    backend/backend.cpp
    backend/cpu/cpu_backend.cpp
    solver/direct.cpp
    postprocessing/fields.cpp
    postprocessing/scattering.cpp
    # --- to be added in the respective phase (see docs/01_project_plan.md) ---
    # operator/sparse_operator.cpp
    # compression/mlfmm/octree.cpp compression/mlfmm/mlfmm_operator.cpp
    # solver/gmres.cpp solver/preconditioner.cpp
    # postprocessing/speckle_statistics.cpp
    reference/mie.cpp
    # io/result_writer.cpp
    # simulation.cpp
)

add_library(specklebem ${SPECKLEBEM_SOURCES})
add_library(SpeckleBem::specklebem ALIAS specklebem)

target_include_directories(specklebem
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(specklebem
    PUBLIC  Eigen3::Eigen spdlog::spdlog
    PRIVATE SpeckleBem::warnings)

if(SPECKLEBEM_ENABLE_OPENMP)
    target_link_libraries(specklebem PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(specklebem PUBLIC SPECKLEBEM_HAVE_OPENMP=1)
endif()

if(SPECKLEBEM_HAVE_BLAS_LAPACK)
    target_link_libraries(specklebem PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_compile_definitions(specklebem PUBLIC SPECKLEBEM_HAVE_BLAS_LAPACK=1 EIGEN_USE_BLAS=1)
endif()

if(SPECKLEBEM_ENABLE_CUDA)
    target_sources(specklebem PRIVATE backend/gpu/cuda_backend.cu)
    target_link_libraries(specklebem PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(specklebem PUBLIC SPECKLEBEM_HAVE_CUDA=1)
    set_target_properties(specklebem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(SPECKLEBEM_ENABLE_HDF5)
    target_link_libraries(specklebem PUBLIC ${HDF5_CXX_LIBRARIES})
    target_include_directories(specklebem PUBLIC ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(specklebem PUBLIC SPECKLEBEM_HAVE_HDF5=1)
endif()

set_target_properties(specklebem PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
