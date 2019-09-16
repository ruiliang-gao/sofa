project(zycutil)

find_package(CUDA REQUIRED)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "zycutil")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "zycutil helper library.")
set(CPACK_SET_DESTDIR TRUE)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Fabian Aichele")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "5")
include(CPack)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(include)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH lib)

set(cutil_headers
    include/cutil/bank_checker.h
    include/cutil/cutil.h
    include/cutil/cutil_inline_runtime.h
    include/cutil/exception.h
    include/cutil/stopwatch_base.inl
    include/cutil/cmd_arg_reader.h
    include/cutil/cutil_inline_bankchecker.h
    include/cutil/cutil_math.h
    include/cutil/helper_string.h
    include/cutil/stopwatch.h
    include/cutil/cutil_gl_error.h
    include/cutil/cutil_inline_drvapi.h
    include/cutil/drvapi_error_string.h
    include/cutil/multithreading.h
    include/cutil/cutil_gl_inline.h
    include/cutil/cutil_inline.h
    include/cutil/error_checker.h
    include/cutil/stopwatch_base.h
)

set(cutil_src
  src/bank_checker.cpp
  src/cmd_arg_reader.cpp
  src/cutil.cpp
  src/stopwatch.cpp
  src/multithreading.cpp)

if (UNIX)
    list(APPEND cutil_headers include/cutil/stopwatch_linux.h)
    set(cutil_src ${cutil_src} src/stopwatch_linux.cpp)
endif(UNIX)

if (WIN32)
    list(APPEND cutil_headers include/cutil/stopwatch_win.h)
    include_directories(${CMAKE_SOURCE_DIR}/trulibs/cutil/include/cutil)
	set(cutil_src ${cutil_src} src/stopwatch_win.cpp)
    add_definitions(-DBUILD_DLL)
endif(WIN32)

add_library(${PROJECT_NAME} SHARED ${cutil_headers} ${cutil_src})
if (UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin
                                ARCHIVE DESTINATION lib
                                LIBRARY DESTINATION lib)
install(FILES
    include/cutil/cutil.h
    include/cutil/cutil_inline_bankchecker.h
    include/cutil/cutil_inline_drvapi.h
    include/cutil/cutil_inline.h
    include/cutil/cutil_inline_runtime.h
    include/cutil/cutil_gl_error.h
    include/cutil/cutil_gl_inline.h
    include/cutil/cutil_math.h
    include/cutil/drvapi_error_string.h
    include/cutil/helper_string.h
    DESTINATION
    include/${PROJECT_NAME})

sofa_create_package(${PROJECT_NAME} "1.0.5" ${PROJECT_NAME} ${PROJECT_NAME})
