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

include_directories(${CMAKE_SOURCE_DIR}/extlibs/Discregrid)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/GenericParameters)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/GenericParameters/GenericParameters)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/extern/Miniball)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation)
include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils)

add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/position-based-dynamics/PositionBasedDynamics ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics)
add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/position-based-dynamics/Simulation ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/Simulation)
add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/position-based-dynamics/Utils ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/Utils)

# sofa_generate_package(${PROJECT_NAME} ${POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
