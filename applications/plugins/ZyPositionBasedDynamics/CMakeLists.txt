cmake_minimum_required(VERSION 3.1)
project(ZyPositionBasedDynamics)

add_definitions("-DSOFA_BUILD_ZY_POSITION_BASED_DYNAMICS_PLUGIN")

set(ZY_POSITION_BASED_DYNAMICS_MAJOR_VERSION 0)
set(ZY_POSITION_BASED_DYNAMICS_MINOR_VERSION 1)
set(ZY_POSITION_BASED_DYNAMICS_VERSION ${ZY_POSITION_BASED_DYNAMICS_MAJOR_VERSION}.${ZY_POSITION_BASED_DYNAMICS_MINOR_VERSION})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(SofaFramework REQUIRED)

## Eigen
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/PBDCommon
                    ${CMAKE_CURRENT_SOURCE_DIR}/PBDSimulation
                    ${CMAKE_SOURCE_DIR}/applications/plugins/ZyPBDDistanceBasedCollisionDetection)

include_directories(${CMAKE_SOURCE_DIR}/SofaKernel
                    ${CMAKE_SOURCE_DIR}/SofaKernel/modules
                    ${EIGEN3_INCLUDE_DIR}
                    ${EIGEN3_INCLUDE_DIR}/unsupported)

set(HEADER_FILES initZyPositionBasedDynamicsPlugin.h
                 PBDAnimationLoop.h
                 PBDTimeStep.h
                 PBDSimulation.h
                 PBDSimulationModel.h

                 PBDCommon/PBDCommon.h
                 PBDCommon/PBDParameterObject.h
                 PBDCommon/IdFactory.h
                 PBDCommon/PBDMathFunctions.h

                 PBDUtils/PBDVolumeIntegration.h
                 PBDUtils/PBDIndexedFaceMesh.h
                 PBDUtils/PBDIndexedTetrahedronMesh.h
                 PBDUtils/PBDTimeStepController.h
                 PBDUtils/PBDTimeManager.h

                 PBDSimulation/PBDParticleData.h
                 PBDSimulation/PBDRigidBodyGeometry.h
                 PBDSimulation/PBDRigidBody.h
                 PBDSimulation/PBDLineModel.h
                 PBDSimulation/PBDTriangleModel.h
                 PBDSimulation/PBDTetrahedronModel.h

                 PBDynamics/PBDConstraints.h
                 PBDynamics/PositionBasedDynamics.h
                 PBDynamics/PositionBasedRigidBodyDynamics.h
                 PBDynamics/PositionBasedElasticRods.h
                 PBDynamics/PBDTimeIntegration.h
                 )

set(SOURCE_FILES initZyPositionBasedDynamicsPlugin.cpp
                 PBDAnimationLoop.cpp
                 PBDTimeStep.cpp
                 PBDSimulation.cpp
                 PBDSimulationModel.cpp

                 PBDCommon/PBDMathFunctions.cpp

                 PBDUtils/PBDVolumeIntegration.cpp
                 PBDUtils/PBDIndexedFaceMesh.cpp
                 PBDUtils/PBDIndexedTetrahedronMesh.cpp
                 PBDUtils/PBDTimeStepController.cpp
                 PBDUtils/PBDTimeManager.cpp

                 PBDSimulation/PBDRigidBodyGeometry.cpp
                 PBDSimulation/PBDLineModel.cpp
                 PBDSimulation/PBDTriangleModel.cpp
                 PBDSimulation/PBDTetrahedronModel.cpp

                 PBDynamics/PBDConstraints.cpp
                 PBDynamics/PositionBasedDynamics.cpp
                 PBDynamics/PositionBasedRigidBodyDynamics.cpp
                 PBDynamics/PositionBasedElasticRods.cpp
                 PBDynamics/PBDTimeIntegration.cpp
                 )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore)

sofa_create_package(${PROJECT_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
