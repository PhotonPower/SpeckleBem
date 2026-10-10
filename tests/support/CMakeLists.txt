# Compiled test helpers shared by the test executables (tests/CMakeLists.txt) and the study
# executables (benchmarks/CMakeLists.txt, which adds this directory itself when the tests are not
# built). Linking specklebem_test_support also provides the include path of tests/support.
add_library(specklebem_test_support STATIC system_memory.cpp)
target_include_directories(specklebem_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(specklebem_test_support PUBLIC SpeckleBem::specklebem PRIVATE SpeckleBem::warnings)
