include(Catch)

# Shared compiled helpers (tests/support/system_memory.cpp): target specklebem_test_support.
add_subdirectory(support)

# Catch2 returns 4 when every test case of an invocation was skipped (SKIP()). Catch2 >= 3.5 sets
# SKIP_RETURN_CODE itself; Ubuntu 24.04 ships 3.4.0, whose Catch.cmake does not, so set it here.

add_executable(specklebem_unit_tests
    unit/test_material.cpp
    unit/test_formulation.cpp
    unit/test_green.cpp
    unit/test_dense_operator.cpp
    unit/test_mesh.cpp
    unit/test_mie.cpp
    unit/test_rwg.cpp
    unit/test_quadrature.cpp
    unit/test_rough_surface.cpp
    unit/test_mesh_io.cpp
    unit/test_singularity.cpp
    unit/test_excitation.cpp
    unit/test_angular_spectrum_beam.cpp
    unit/test_direct.cpp
    unit/test_fields.cpp
    unit/test_operators.cpp
    unit/test_assembler.cpp
    unit/test_gmres.cpp
    unit/test_result_writer.cpp
    unit/test_simulation.cpp
    unit/test_path.cpp
    unit/test_octree.cpp
    unit/test_near_field.cpp
    unit/test_plane_wave.cpp
    unit/test_fast_math.cpp
    unit/test_patterns.cpp
    unit/test_mlfmm.cpp
    unit/test_mlfmm_operator.cpp
    unit/test_mlfmm_policy.cpp
    unit/test_system_memory.cpp
    unit/test_mlfmm_scaling.cpp
    # unit/test_speckle_statistics.cpp
)
target_link_libraries(specklebem_unit_tests PRIVATE SpeckleBem::specklebem specklebem_test_support Catch2::Catch2WithMain SpeckleBem::warnings)
# Shared test helpers (tests/support), included without relative paths.
target_include_directories(specklebem_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
# Reference data and fixtures (tests/data) read by regression and I/O tests.
target_compile_definitions(specklebem_unit_tests PRIVATE
    SPECKLEBEM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
catch_discover_tests(specklebem_unit_tests PROPERTIES SKIP_RETURN_CODE 4)
# Slow studies (hidden Catch2 cases, WP7c): label `slow`, minutes in release, manual only:
# `ctest --preset release -L slow`. Standard runs and CI exclude them
# (`-LE "validation|slow"`, `-LE "validation-large|slow"`). They SKIP in unoptimised builds.
add_test(NAME "kernels: fold sweep (slow)" COMMAND specklebem_unit_tests "[fold]")
add_test(NAME "assembler: rhs degree study (slow)" COMMAND specklebem_unit_tests "[rhsstudy]")
add_test(NAME "mlfmm: plane-wave accuracy sweep (slow)" COMMAND specklebem_unit_tests "[mlfmm_sweep]")
add_test(NAME "mlfmm: single-level FMM accuracy sweep (slow)" COMMAND specklebem_unit_tests "[patterns_sweep]")
add_test(NAME "mlfmm: far operator accuracy sweep (slow)" COMMAND specklebem_unit_tests "[mlfmm_far_sweep]")
add_test(NAME "mlfmm: far operator apply timing (slow)" COMMAND specklebem_unit_tests "[mlfmm_apply_timing]")
add_test(NAME "mlfmm: region policy sweep (slow)" COMMAND specklebem_unit_tests "[mlfmm_policy_sweep]")
set_tests_properties("kernels: fold sweep (slow)" "assembler: rhs degree study (slow)"
    "mlfmm: plane-wave accuracy sweep (slow)" "mlfmm: single-level FMM accuracy sweep (slow)"
    "mlfmm: far operator accuracy sweep (slow)" "mlfmm: far operator apply timing (slow)"
    "mlfmm: region policy sweep (slow)"
    PROPERTIES LABELS slow SKIP_RETURN_CODE 4)

# Zero-allocation check of the element kernels. Separate executable because it replaces the
# global operator new / delete (which would disable ASan's new/delete mismatch detection for
# every other unit test).
add_executable(specklebem_alloc_tests unit/test_operators_alloc.cpp)
target_link_libraries(specklebem_alloc_tests PRIVATE SpeckleBem::specklebem Catch2::Catch2WithMain SpeckleBem::warnings)
catch_discover_tests(specklebem_alloc_tests PROPERTIES SKIP_RETURN_CODE 4)

# Writes a fixed dataset with io::open_npy_directory into the directory given as argument; run
# by tests/python/test_result_writer.py (NumPy round trip), which finds it as
# <build>/tests/specklebem_npy_fixture next to <build>/python. Not a CTest test itself.
add_executable(specklebem_npy_fixture support/npy_fixture_main.cpp)
target_link_libraries(specklebem_npy_fixture PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)
if(WIN32)
    target_link_libraries(specklebem_npy_fixture PRIVATE shell32)  # CommandLineToArgvW
endif()

# Validation tests compare against analytic references (Mie, flat-interface
# Fresnel) and against the dense operator. They are slower and labelled so that
# `ctest -L validation` runs them separately.
add_executable(specklebem_validation_tests
    validation/test_placeholder.cpp
    validation/test_mie_reference.cpp
    validation/test_post_mie_projection.cpp
    validation/test_assembler_mie.cpp
    validation/test_mie_sphere_dense.cpp
    validation/test_simulation_mie.cpp
    validation/test_mlfmm_simulation.cpp
    # validation/test_mie_sphere_mlfmm.cpp
    # validation/test_mlfmm_vs_dense.cpp
    # validation/test_flat_interface_fresnel.cpp
)
target_link_libraries(specklebem_validation_tests PRIVATE SpeckleBem::specklebem specklebem_test_support Catch2::Catch2WithMain SpeckleBem::warnings)
target_include_directories(specklebem_validation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
catch_discover_tests(specklebem_validation_tests PROPERTIES SKIP_RETURN_CODE 4 LABELS validation)

# Large validation cases (dense solves with 2N >= 15 k unknowns: minutes to hours and several
# GB of memory). Label `validation-large` only: never run in CI (which runs -L '^validation$'),
# nightly / manual with `ctest --preset release -L validation-large`. The cases SKIP in
# unoptimised builds and when the dense solve would exceed 60 % of the physical memory.
add_executable(specklebem_validation_large_tests
    validation_large/test_mie_sphere_dense_large.cpp
    validation_large/test_gmres_vs_lu_large.cpp
    validation_large/test_mlfmm_vs_dense_large.cpp
    validation_large/test_ag_sphere_mlfmm_large.cpp
)
target_link_libraries(specklebem_validation_large_tests PRIVATE SpeckleBem::specklebem specklebem_test_support Catch2::Catch2WithMain SpeckleBem::warnings)
target_include_directories(specklebem_validation_large_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
# RUN_SERIAL: each n = 4 dense case needs ~4 GB; running two at once was OOM-killed on a 15 GB machine.
catch_discover_tests(specklebem_validation_large_tests PROPERTIES SKIP_RETURN_CODE 4 LABELS validation-large TIMEOUT 21600 RUN_SERIAL TRUE)
