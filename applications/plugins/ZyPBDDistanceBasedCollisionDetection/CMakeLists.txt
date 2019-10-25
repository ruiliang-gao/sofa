cmake_minimum_required(VERSION 3.1)
project(ZyPBDDistanceBasedCollisionDetection)

add_definitions("-DSOFA_BUILD_ZY_PDB_DISTANCE_BASED_COLLISION_DETECTION")

set(ZY_POSITION_BASED_DYNAMICS_MAJOR_VERSION 0)
set(ZY_POSITION_BASED_DYNAMICS_MINOR_VERSION 1)
set(ZY_POSITION_BASED_DYNAMICS_VERSION ${ZY_POSITION_BASED_DYNAMICS_MAJOR_VERSION}.${ZY_POSITION_BASED_DYNAMICS_MINOR_VERSION})

find_package(SofaFramework REQUIRED)

## Eigen
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/applications/plugins/ZyPositionBasedDynamics)

include_directories(${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics
                    ${CMAKE_SOURCE_DIR}/extlibs/GenericParameters/GenericParameters
                    ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics
                    ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Common
                    ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils
                    ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics
                    ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation)

include_directories(${CMAKE_SOURCE_DIR}/SofaKernel
                    ${CMAKE_SOURCE_DIR}/SofaKernel/modules
                    ${EIGEN3_INCLUDE_DIR}
                    ${EIGEN3_INCLUDE_DIR}/unsupported)

set(HEADER_FILES initZyPBDDistanceBasedCollisionDetectionPlugin.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CollisionDetection.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/Aabb.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/BoundingSphere.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/BoundingSphereHierarchy.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/DistanceFieldCollisionDetection.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CubicSDFCollisionDetection.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/kdTree.h
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/kdTree.inl
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/extern/Miniball/MiniBall.h
                 )

set(SOURCE_FILES initZyPBDDistanceBasedCollisionDetectionPlugin.cpp
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CollisionDetection.cpp
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/BoundingSphereHierarchy.cpp
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/DistanceFieldCollisionDetection.cpp
                 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CubicSDFCollisionDetection.cpp
                 )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationCore
   Discregrid PositionBasedDynamics Simulation Utils)

sofa_create_package(${PROJECT_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
