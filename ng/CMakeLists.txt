cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/cmake")
set(SOFA_WITH_EXPERIMENTAL_FEATURES_ 1)
set(SOFA_HAVE_GLEW 1)

add_subdirectory(extlibs)
add_subdirectory(kernel)
#add_subdirectory(framework)
#add_subdirectory(plugins)

