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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

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

set(COMMON_UTIL_HEADER_FILES
    PBDCommon/PBDCommon.h
    PBDCommon/PBDParameterObject.h
    PBDCommon/IdFactory.h
    PBDCommon/PBDMathFunctions.h

    PBDUtils/PBDVolumeIntegration.h
    PBDUtils/PBDIndexedFaceMesh.h
    PBDUtils/PBDIndexedTetrahedronMesh.h
    PBDUtils/PBDTimeManager.h)

set(COMMON_UTIL_SOURCE_FILES
    PBDCommon/IDFactory.cpp
    PBDCommon/PBDMathFunctions.cpp

    PBDUtils/PBDVolumeIntegration.cpp
    PBDUtils/PBDIndexedFaceMesh.cpp
    PBDUtils/PBDIndexedTetrahedronMesh.cpp
    PBDUtils/PBDTimeManager.cpp)

set(MODELS_HEADER_FILES
    PBDModels/PBDLineModel.h
    PBDModels/PBDTriangleModel.h
    PBDModels/PBDTetrahedronModel.h
    PBDModels/PBDSimulationModel.h
    PBDModels/SofaPBDLineModel.h
    PBDModels/SofaPBDTriangleModel.h)

set(MODELS_SOURCE_FILES
    PBDModels/PBDLineModel.cpp
    PBDModels/PBDTriangleModel.cpp
    PBDModels/PBDTetrahedronModel.cpp
    PBDModels/PBDSimulationModel.cpp
    PBDModels/SofaPBDLineModel.cpp
    PBDModels/SofaPBDTriangleModel.cpp)

set(CONSTRAINTS_HEADER_FILES
    PBDConstraints/PBDConstraintBase.h
    PBDConstraints/PBDConstraints.h
    PBDConstraints/PBDFixedConstraint.h)

set(CONSTRAINTS_SOURCE_FILES
    PBDConstraints/PBDConstraints.cpp
    PBDConstraints/PBDFixedConstraint.cpp)

set(SIMULATION_HEADER_FILES
    PBDSimulation/PBDParticleData.h
    PBDSimulation/PBDRigidBodyGeometry.h
    PBDSimulation/PBDRigidBody.h
    )

set(SIMULATION_SOURCE_FILES
    PBDSimulation/PBDRigidBodyGeometry.cpp
    )

set(DYNAMICS_HEADER_FILES
    PBDynamics/PositionBasedDynamics.h
    PBDynamics/PositionBasedRigidBodyDynamics.h
    PBDynamics/PositionBasedElasticRods.h
    PBDynamics/PBDTimeIntegration.h)

set(DYNAMICS_SOURCE_FILES
    PBDynamics/PositionBasedDynamics.cpp
    PBDynamics/PositionBasedRigidBodyDynamics.cpp
    PBDynamics/PositionBasedElasticRods.cpp
    PBDynamics/PBDTimeIntegration.cpp)

set(HEADER_FILES initZyPositionBasedDynamicsPlugin.h
                 PBDMain/SofaPBDAnimationLoop.h
                 PBDMain/SofaPBDTimeStep.h
                 PBDMain/SofaPBDSimulation.h
                 # PBDMain/SofaPBDTimeStepController.h

                 PBDIntegration/GeometryConversion.h
                 PBDIntegration/MechObjConversion.h

                 )

set(SOURCE_FILES initZyPositionBasedDynamicsPlugin.cpp
                 PBDMain/SofaPBDAnimationLoop.cpp
                 PBDMain/SofaPBDTimeStep.cpp
                 PBDMain/SofaPBDSimulation.cpp
                 # PBDMain/SofaPBDTimeStepController.cpp

                 PBDIntegration/GeometryConversion.cpp
                 PBDIntegration/MechObjConversion.cpp
                 )

add_library(${PROJECT_NAME}_CommonUtils STATIC ${COMMON_UTIL_HEADER_FILES} ${COMMON_UTIL_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_CommonUtils SofaCore SofaSimulationCore SofaBaseMechanics)

add_library(${PROJECT_NAME}_Constraints STATIC ${CONSTRAINTS_HEADER_FILES} ${CONSTRAINTS_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Constraints SofaCore SofaSimulationCore SofaBaseMechanics ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Models)

add_library(${PROJECT_NAME}_Models STATIC ${MODELS_HEADER_FILES} ${MODELS_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Models SofaCore SofaSimulationCore SofaBaseMechanics ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Constraints)

add_library(${PROJECT_NAME}_Simulation STATIC ${SIMULATION_HEADER_FILES} ${SIMULATION_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Simulation SofaCore SofaSimulationCore SofaBaseMechanics ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Models)

add_library(${PROJECT_NAME}_Dynamics STATIC ${DYNAMICS_HEADER_FILES} ${DYNAMICS_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Dynamics SofaCore SofaSimulationCore SofaBaseMechanics ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Models)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationCore SofaBaseMechanics ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Models ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Dynamics ${PROJECT_NAME}_Simulation ZyPBDDistanceBasedCollisionDetection)

sofa_create_package(${PROJECT_NAME}_CommonUtils ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_CommonUtils)
sofa_create_package(${PROJECT_NAME}_Models ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Models ${PROJECT_NAME}_Models)
sofa_create_package(${PROJECT_NAME}_Constraints ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Constraints)
sofa_create_package(${PROJECT_NAME}_Simulation ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Simulation ${PROJECT_NAME}_Simulation)
sofa_create_package(${PROJECT_NAME}_Dynamics ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Dynamics ${PROJECT_NAME}_Dynamics)
sofa_create_package(${PROJECT_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
