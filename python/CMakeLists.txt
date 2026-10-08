pybind11_add_module(_specklebem MODULE src/bindings.cpp)
target_link_libraries(_specklebem PRIVATE SpeckleBem::specklebem)
target_compile_definitions(_specklebem PRIVATE VERSION_INFO="${PROJECT_VERSION}")
set_target_properties(_specklebem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/specklebem)
install(TARGETS _specklebem LIBRARY DESTINATION specklebem)
