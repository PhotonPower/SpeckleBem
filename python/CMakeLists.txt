set(SPECKLEBEM_PYTHON_SOURCES
    src/bindings.cpp
    src/bind_excitation.cpp
    src/bind_geometry.cpp
    src/bind_io.cpp
    src/bind_simulation.cpp)
pybind11_add_module(_specklebem MODULE ${SPECKLEBEM_PYTHON_SOURCES})
target_link_libraries(_specklebem PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)
# GCC's flow-sensitive warnings fire on pybind11 / Eigen / libstdc++ code inlined at -O2 (false
# positives inside system headers, e.g. pybind11::array::ndim(), std::copy of Eigen storage).
# Source-level options follow the project warning set on the command line.
# Re-test removing this suppression after each pybind11 or GCC upgrade.
set_property(SOURCE ${SPECKLEBEM_PYTHON_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:GNU>:-Wno-null-dereference$<SEMICOLON>-Wno-maybe-uninitialized>)
target_compile_definitions(_specklebem PRIVATE VERSION_INFO="${PROJECT_VERSION}")
set_target_properties(_specklebem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/specklebem)

# Mirror the pure-Python part of the package next to the built extension so that
# `PYTHONPATH=<build>/python python3 -c "import specklebem"` works without installing
# (CI and local tests). Listed explicitly, like the C++ sources.
set(SPECKLEBEM_PYTHON_FILES
    specklebem/__init__.py
    specklebem/rough_surface.py)
foreach(_py IN LISTS SPECKLEBEM_PYTHON_FILES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_py} ${CMAKE_CURRENT_BINARY_DIR}/${_py} COPYONLY)
endforeach()
install(TARGETS _specklebem LIBRARY DESTINATION specklebem)
