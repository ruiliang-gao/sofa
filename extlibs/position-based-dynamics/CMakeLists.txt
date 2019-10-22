cmake_minimum_required(VERSION 2.8)

project(PositionBasedDynamics)

set(POSITION_BASED_DYNAMICS_MAJOR_VERSION 0)
set(POSITION_BASED_DYNAMICS_MINOR_VERSION 1)
set(POSITION_BASED_DYNAMICS_VERSION ${POSITION_BASED_DYNAMICS_MAJOR_VERSION}.${POSITION_BASED_DYNAMICS_MINOR_VERSION})

set(PROJECT_PATH ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/CMake;${CMAKE_MODULE_PATH}")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/CMake/Common.cmake)

add_definitions(-DPBD_DATA_PATH="../data")

#if (DEFINED PBD_EXTERNALINSTALLDIR)
#        set(ExternalInstallDir "${PBD_EXTERNALINSTALLDIR}" CACHE INTERNAL "External install directory")
#else()
#        set(ExternalInstallDir "${PROJECT_BINARY_DIR}" CACHE INTERNAL "External install directory")
#endif()

include_directories(${CMAKE_SOURCE_DIR}/extlibs/Discregrid)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/GenericParameters)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/extern/Miniball)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils)

add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics)
add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/Simulation)
add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/Utils)

# if (NOT PBD_NO_DEMOS)
#	if (WIN32)
#		add_subdirectory(extern/freeglut)
#	endif()
#	add_subdirectory(extern/AntTweakBar)
#	add_subdirectory(extern/glew)
#	add_subdirectory(extern/md5)
#	add_subdirectory(Demos)
#endif()

#include(ExternalProject)
#set(EigenDir "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen")

#set(EXT_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
#if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#        set(EXT_CMAKE_BUILD_TYPE "Release")
#endif()

## Discregrid
#ExternalProject_Add(
#   Ext_Discregrid
#   PREFIX "${ExternalInstallDir}/Discregrid"
#   GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/Discregrid.git
#   GIT_TAG "c0fb5aeac4c8a83e9f37c720315f13a834409b81"
#   INSTALL_DIR ${ExternalInstallDir}/Discregrid
#   CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=${EXT_CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/Discregrid -DBUILD_CMD_EXECUTABLE:BOOL=0 -DEIGEN3_INCLUDE_DIR:PATH=${EigenDir}
#)

## GenericParameters
#ExternalProject_Add(
#   Ext_GenericParameters
#   PREFIX "${ExternalInstallDir}/GenericParameters"
#   GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/GenericParameters.git
#   GIT_TAG "b1ad669fac8d106515f6aa8514a03598d5766a36"
#   INSTALL_DIR ${ExternalInstallDir}/GenericParameters
#   CMAKE_ARGS -DCMAKE_BUILD_TYPE=${EXT_CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/GenericParameters -DGENERICPARAMETERS_NO_TESTS:BOOL=1
#)

#install(DIRECTORY ./Common
#DESTINATION include
#FILES_MATCHING PATTERN "*.h")

#install(DIRECTORY ./data
#DESTINATION include
#FILES_MATCHING PATTERN "*.glsl")

#install(DIRECTORY ./Demos/Visualization
#DESTINATION include
#FILES_MATCHING PATTERN "*.h")

