add_executable(example_materials materials.cpp)
target_link_libraries(example_materials PRIVATE SpeckleBem::specklebem)
# Phase 2+: example_mie_sphere, example_rough_surface_speckle
