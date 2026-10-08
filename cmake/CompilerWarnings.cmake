# Interface target carrying the project-wide warning set.
add_library(specklebem_warnings INTERFACE)
add_library(SpeckleBem::warnings ALIAS specklebem_warnings)

if(MSVC)
    target_compile_options(specklebem_warnings INTERFACE /W4 /permissive-)
    if(SPECKLEBEM_WARNINGS_AS_ERRORS)
        target_compile_options(specklebem_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(specklebem_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
        -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion)
    if(SPECKLEBEM_WARNINGS_AS_ERRORS)
        target_compile_options(specklebem_warnings INTERFACE -Werror)
    endif()
endif()
