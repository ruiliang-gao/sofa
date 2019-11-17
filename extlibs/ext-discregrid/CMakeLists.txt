cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

project(Discregrid)

set(DISCREGRID_MAJOR_VERSION 0)
set(DISCREGRID_MINOR_VERSION 1)
set(DISCREGRID_VERSION ${DISCREGRID_MAJOR_VERSION}.${DISCREGRID_MINOR_VERSION})

# Visual studio solution directories.
set_property(GLOBAL PROPERTY USE_FOLDERS on)

# Require C++11 compiler
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif(UNIX)

# Enable simultaneous compilation of source files.
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4250")
endif(WIN32)

set(CMAKE_DEBUG_POSTFIX "_d")

# Put all executables and libraries into a common directory.
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH    "${PROJECT_BINARY_DIR}/bin")

add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/ext-discregrid/discregrid ${CMAKE_BINARY_DIR}/extlibs/Discregrid/discregrid)

option(DISCREGRID_BUILD_CMD_EXECUTABLE "Build command line executable" ON)
if(DISCREGRID_BUILD_CMD_EXECUTABLE)
        add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/ext-discregrid/cmd ${CMAKE_BINARY_DIR}/extlibs/Discregrid/cmd)
endif(DISCREGRID_BUILD_CMD_EXECUTABLE)

