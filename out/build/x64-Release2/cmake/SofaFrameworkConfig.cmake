# CMake package configuration file for SofaFramework


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SofaFrameworkConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

get_filename_component(SOFA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# Add CMAKE_CURRENT_LIST_DIR to CMAKE_MODULE_PATH (if not already done)
# Needed by: include(SofaMacros)
list(FIND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}" HAS_SOFAFRAMEWORK_CMAKE_MODULE_PATH)
if(HAS_SOFAFRAMEWORK_CMAKE_MODULE_PATH EQUAL -1)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
endif()

list(FIND CMAKE_PREFIX_PATH "${SOFA_ROOT}/plugins" HAS_PLUGINS_CMAKE_PREFIX_PATH)
if(HAS_PLUGINS_CMAKE_PREFIX_PATH EQUAL -1)
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../plugins")
endif()

list(APPEND CMAKE_INCLUDE_PATH "${SOFA_ROOT}/include/extlibs/WinDepPack")
list(APPEND CMAKE_MODULE_PATH "${SOFA_ROOT}/cmake/Modules")

include(SofaMacros)

set(SOFAHELPER_HAVE_BOOST "1")
set(SOFAHELPER_HAVE_BOOST_SYSTEM "1")
set(SOFAHELPER_HAVE_BOOST_FILESYSTEM "1")
set(SOFAHELPER_HAVE_BOOST_PROGRAM_OPTIONS "1")
set(SOFAHELPER_HAVE_BOOST_THREAD "1")
set(SOFAHELPER_HAVE_OPENGL "1")
set(SOFAHELPER_HAVE_GLEW "1")
set(SOFAHELPER_HAVE_GTEST "1")

set(SOFA_NO_OPENGL "OFF")
set(SOFA_USE_MASK "OFF")

set(SOFA_WITH_DEVTOOLS "ON")
set(SOFA_WITH_THREADING "ON")
set(SOFA_WITH_DEPRECATED_COMPONENTS "ON")

# Find dependencies
find_package(Boost QUIET REQUIRED system filesystem program_options)
if(SOFAHELPER_HAVE_BOOST_THREAD)
    find_package(Boost QUIET REQUIRED thread)
endif()
if(SOFAHELPER_HAVE_OPENGL)
    find_package(OpenGL QUIET REQUIRED)
endif()
if(SOFAHELPER_HAVE_GLEW)
    find_package(GLEW QUIET REQUIRED)
endif()
if(SOFAHELPER_HAVE_GTEST)
    find_package(GTest CONFIG QUIET REQUIRED)
endif()

foreach(target SofaHelper SofaDefaultType SofaCore)
    if(NOT TARGET ${target})
        include("${CMAKE_CURRENT_LIST_DIR}/SofaFrameworkTargets.cmake")
        break()
    endif()
endforeach()

check_required_components(SofaHelper SofaDefaultType SofaCore)
