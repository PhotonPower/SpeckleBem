pybind11_add_module(_specklebem MODULE src/bindings.cpp)
target_link_libraries(_specklebem PRIVATE SpeckleBem::specklebem)
target_compile_definitions(_specklebem PRIVATE VERSION_INFO="${PROJECT_VERSION}")
set_target_properties(_specklebem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/specklebem)

# Mirror the pure-Python part of the package next to the built extension so that
# `PYTHONPATH=<build>/python python3 -c "import specklebem"` works without installing
# (CI and local tests). Listed explicitly, like the C++ sources.
set(SPECKLEBEM_PYTHON_FILES
    specklebem/__init__.py)
foreach(_py IN LISTS SPECKLEBEM_PYTHON_FILES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_py} ${CMAKE_CURRENT_BINARY_DIR}/${_py} COPYONLY)
endforeach()
install(TARGETS _specklebem LIBRARY DESTINATION specklebem)
