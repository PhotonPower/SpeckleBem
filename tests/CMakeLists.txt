include(Catch)

add_executable(specklebem_unit_tests
    unit/test_material.cpp
    unit/test_formulation.cpp
    unit/test_green.cpp
    unit/test_dense_operator.cpp
    # unit/test_mesh.cpp unit/test_rwg.cpp unit/test_quadrature.cpp unit/test_singularity.cpp
    # unit/test_octree.cpp unit/test_gmres.cpp unit/test_speckle_statistics.cpp
)
target_link_libraries(specklebem_unit_tests PRIVATE SpeckleBem::specklebem Catch2::Catch2WithMain SpeckleBem::warnings)
catch_discover_tests(specklebem_unit_tests)

# Validation tests compare against analytic references (Mie, flat-interface
# Fresnel) and against the dense operator. They are slower and labelled so that
# `ctest -L validation` runs them separately.
add_executable(specklebem_validation_tests
    validation/test_placeholder.cpp
    # validation/test_mie_sphere_dense.cpp
    # validation/test_mie_sphere_mlfmm.cpp
    # validation/test_mlfmm_vs_dense.cpp
    # validation/test_flat_interface_fresnel.cpp
)
target_link_libraries(specklebem_validation_tests PRIVATE SpeckleBem::specklebem Catch2::Catch2WithMain)
catch_discover_tests(specklebem_validation_tests PROPERTIES LABELS validation)
