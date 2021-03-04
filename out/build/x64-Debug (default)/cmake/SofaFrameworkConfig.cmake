# CMake package configuration file for SofaFramework
cmake_minimum_required(VERSION 3.12)

### Expanded from @PACKAGE_GUARD@ by SofaMacrosInstall.cmake ###
include_guard()
################################################################

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
list(APPEND CMAKE_MODULE_PATH "${SOFA_ROOT}/lib/cmake/Modules")

# Help RELOCATABLE plugins to resolve their dependencies.
# See SofaMacrosInstall.cmake for usage of this property.
define_property(TARGET
    PROPERTY "RELOCATABLE_INSTALL_DIR"
    BRIEF_DOCS "Install directory of RELOCATABLE target"
    FULL_DOCS "Install directory of RELOCATABLE target"
    )

include(SofaMacros)

set(SOFAFRAMEWORK_TARGETS SofaCore;SofaDefaultType;SofaHelper;SofaSimulationCore)
set(Sofa_VERSION 20.12.00)

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

# Eigen3 is required by SofaDefaultType and SofaHelper
find_package(Eigen3 QUIET REQUIRED)

foreach(target ${SOFAFRAMEWORK_TARGETS})
    if(NOT TARGET ${target})
        include("${CMAKE_CURRENT_LIST_DIR}/SofaFrameworkTargets.cmake")
        break()
    endif()
endforeach()
if(NOT TARGET SofaFramework)
    include("${CMAKE_CURRENT_LIST_DIR}/SofaFrameworkTargets.cmake")
endif()
check_required_components(SofaFramework)
