cmake_minimum_required(VERSION 3.1)
project(ZyPositionBasedDynamics)

set(CMAKE_VERBOSE_MAKEFILE On)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++14")

find_package(SofaFramework REQUIRED)
find_package(ZyPipeline REQUIRED)

## Eigen
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/PBDCommon
                    ${CMAKE_CURRENT_SOURCE_DIR}/PBDSimulation
                    ${CMAKE_SOURCE_DIR}/applications/plugins/ZyPBDDistanceBasedCollisionDetection)

include_directories(${CMAKE_SOURCE_DIR}/SofaKernel
                    ${CMAKE_SOURCE_DIR}/SofaKernel/modules
                    ${CMAKE_SOURCE_DIR}/modules
                    ${EIGEN3_INCLUDE_DIR}
                    ${EIGEN3_INCLUDE_DIR}/unsupported)

include_directories(${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics
                    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator
                    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable
                    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/extern/gl3w)

if (ZyPipeline_FOUND)
    message(STATUS "Found ZyPipeline package.")
    include_directories(${CMAKE_SOURCE_DIR}/applications/plugins/ZyPipeline)
else()
    message(FATAL "Failed to find ZyPipeline package!")
endif()

#if (SOFA_BUILD_IMPULSE_BASED_DYNAMICS)
#    set(IPD_SIMULATOR_LIBRARY_NAME IPDSimulator CACHE INTERNAL "" FORCE)

#    set (IPD_SIMULATOR_HEADERS
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/Config.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/Simulator.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/SimulationClock.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/SimulationData.hpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/ParticleState.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/RigidBodyState.hpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ContactConstraints.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/Constraints.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ConstraintBase.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/DistanceConstraint.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ParticleConstraint.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyBallJoint.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyParticleJoint.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/TetrahedralConstraints.hpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/Forces.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/ForceBase.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/BodyForce.hpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSIntegrator.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate.hpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_ContactConstraints.inl
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_DistanceConstraint.inl
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_Joints.inl
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_TetConstraints.inl

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/ConstraintDataAbstractionUtilities.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/MatrixUtilities.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/SolverUtilities.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/StorageUtilities.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/Timer.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/TypeListUtilities.hpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/Utility.hpp
#        )

#    set (IPD_SIMULATOR_SOURCES
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/ParticleState.cpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/RigidBodyState.cpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ContactConstraints.cpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/DistanceConstraint.cpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyBallJoint.cpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyParticleJoint.cpp
#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/TetrahedralConstraints.cpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/BodyForce.cpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSIntegrator.cpp

#        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/Timer.cpp
#        )

#    add_library(${IPD_SIMULATOR_LIBRARY_NAME} STATIC
#            ${IPD_SIMULATOR_HEADERS}
#            ${IPD_SIMULATOR_SOURCES}
#            )

#    target_link_libraries(${IPD_SIMULATOR_LIBRARY_NAME} gl3w)

#    set(IPD_DYNAMICS_LIBRARY_NAME IPDDynamics CACHE INTERNAL "" FORCE)

#    set(IPD_DYNAMICS_BODY_HEADERS
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBody.hpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBodyBuilder.hpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBody.hpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBodyBuilder.hpp
#    )
#    set(IPD_DYNAMICS_BODY_SOURCES
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBody.cpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBodyBuilder.cpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBody.cpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBodyBuilder.cpp
#    )

#    set(IPD_DYNAMICS_COLLISION_HEADERS
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/AABB.hpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionDetection.hpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionShapes.hpp
#    )
#    set(IPD_DYNAMICS_COLLISION_SOURCES
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionDetection.cpp
#            ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionShapes.cpp
#    )

#    add_library(${IPD_DYNAMICS_LIBRARY_NAME} STATIC
#                ${IPD_DYNAMICS_BODY_HEADERS}
#                ${IPD_DYNAMICS_BODY_SOURCES}
#                ${IPD_DYNAMICS_COLLISION_HEADERS}
#                ${IPD_DYNAMICS_COLLISION_SOURCES})

#    target_link_libraries(${IPD_DYNAMICS_LIBRARY_NAME} gl3w)
#endif()

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
    PBDModels/initZyPositionBasedDynamicsModels.h
    PBDModels/PBDLineModel.h
    PBDModels/PBDTriangleModel.h
    PBDModels/PBDTetrahedronModel.h
    PBDModels/PBDSimulationModel.h
    )

set(MODELS_SOURCE_FILES
    PBDModels/initZyPositionBasedDynamicsModels.cpp
    PBDModels/PBDLineModel.cpp
    PBDModels/PBDTriangleModel.cpp
    PBDModels/PBDTetrahedronModel.cpp
    PBDModels/PBDSimulationModel.cpp
    )

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
                 PBDMain/SofaPBDTimeStepInterface.h
                 PBDMain/SofaPBDTimeStep.h
                 PBDMain/SofaIPDTimeStep.h
                 PBDMain/SofaPBDSimulation.h

                 PBDModels/SofaPBDModelBase.h
                 PBDModels/SofaPBDLineModel.h
                 PBDModels/SofaPBDRigidBodyModel.h
                 PBDModels/SofaPBDTriangleMeshModel.h
                 PBDModels/SofaPBDTetrahedronMeshModel.h

                 PBDIntegration/GeometryAdapterBase.h
                 PBDIntegration/MechanicalObjectAdapterBase.h

                 PBDIntegration/SofaPBDPipeline.h
                 PBDIntegration/SofaPBDCollisionVisitor.h
                 PBDIntegration/SofaPBDSearchNodeVisitor.h

                 PBDIntegration/SofaPBDCollisionIntersectorInterface.h

                 PBDIntegration/SofaPBDNarrowPhaseCollisionIntersectors.h
                 PBDIntegration/SofaPBDNarrowPhaseCollisionIntersectors.inl

                 PBDIntegration/SofaPBDPointCollisionModel.h
                 PBDIntegration/SofaPBDPointCollisionModel.inl
                 PBDIntegration/SofaPBDLineCollisionModel.h
                 PBDIntegration/SofaPBDLineCollisionModel.inl
                 PBDIntegration/SofaPBDTriangleCollisionModel.h
                 PBDIntegration/SofaPBDTriangleCollisionModel.inl

                 PBDIntegration/SofaPBDCollisionDetectionOutput.h
                 PBDIntegration/SofaPBDBruteForceDetection.h
                 PBDIntegration/PBDCollisionModelsIntersection.h
                 )

set(SOURCE_FILES initZyPositionBasedDynamicsPlugin.cpp
                 PBDMain/SofaPBDAnimationLoop.cpp
                 PBDMain/SofaIPDTimeStep.cpp
                 PBDMain/SofaPBDTimeStep.cpp
                 PBDMain/SofaPBDSimulation.cpp

                 PBDModels/SofaPBDModelBase.cpp
                 PBDModels/SofaPBDLineModel.cpp
                 PBDModels/SofaPBDRigidBodyModel.cpp
                 PBDModels/SofaPBDTriangleMeshModel.cpp
                 PBDModels/SofaPBDTetrahedronMeshModel.cpp

                 PBDIntegration/GeometryAdapterBase.cpp
                 PBDIntegration/MechanicalObjectAdapterBase.cpp

                 PBDIntegration/SofaPBDPipeline.cpp
                 PBDIntegration/SofaPBDCollisionVisitor.cpp

                 PBDIntegration/SofaPBDNarrowPhaseCollisionIntersectors.cpp

                 PBDIntegration/SofaPBDPointCollisionModel.cpp
                 PBDIntegration/SofaPBDLineCollisionModel.cpp
                 PBDIntegration/SofaPBDTriangleCollisionModel.cpp

                 PBDIntegration/SofaPBDBruteForceDetection.cpp
                 PBDIntegration/PBDCollisionModelsIntersection.cpp
                 )

add_library(${PROJECT_NAME}_CommonUtils STATIC ${COMMON_UTIL_HEADER_FILES} ${COMMON_UTIL_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_CommonUtils SofaCore SofaSimulationCore SofaBaseMechanics SofaBaseCollision)

add_library(${PROJECT_NAME}_Constraints STATIC ${CONSTRAINTS_HEADER_FILES} ${CONSTRAINTS_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Constraints SofaCore SofaSimulationCore SofaBaseMechanics SofaBaseCollision SofaLoader ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Models)

add_library(${PROJECT_NAME}_Models STATIC ${MODELS_HEADER_FILES} ${MODELS_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Models SofaCore SofaSimulationCore SofaBaseMechanics SofaBaseCollision ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Constraints)

add_library(${PROJECT_NAME}_Simulation STATIC ${SIMULATION_HEADER_FILES} ${SIMULATION_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Simulation SofaCore SofaSimulationCore SofaBaseMechanics SofaBaseCollision ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Models)

add_library(${PROJECT_NAME}_Dynamics STATIC ${DYNAMICS_HEADER_FILES} ${DYNAMICS_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_Dynamics SofaCore SofaSimulationCore SofaBaseMechanics SofaBaseCollision ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Models)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore PUBLIC SofaSimulationCore PUBLIC SofaBaseMechanics PUBLIC SofaBaseCollision PUBLIC ${PROJECT_NAME}_CommonUtils PUBLIC ${PROJECT_NAME}_Models PUBLIC ${PROJECT_NAME}_Constraints PUBLIC ${PROJECT_NAME}_Dynamics PUBLIC ${PROJECT_NAME}_Simulation PUBLIC ZyPBDDistanceBasedCollisionDetection)

if (ZyPipeline_FOUND)
    message(STATUS "Found ZyPipeline package.")
    target_link_libraries(${PROJECT_NAME} PUBLIC ZyPipeline)
endif()

if (SOFA_BUILD_IMPULSE_BASED_DYNAMICS)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${IPD_SIMULATOR_LIBRARY_NAME} PRIVATE ${IPD_DYNAMICS_LIBRARY_NAME})
endif()

if (SOFA_BUILD_IMPULSE_BASED_DYNAMICS)
    message(STATUS "Creating packages for ${IPD_SIMULATOR_LIBRARY_NAME} and ${IPD_DYNAMICS_LIBRARY_NAME} libraries.")
    sofa_create_package(${IPD_SIMULATOR_LIBRARY_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
    sofa_create_package(${IPD_DYNAMICS_LIBRARY_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
endif()

sofa_create_package(${PROJECT_NAME}_CommonUtils ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_CommonUtils ${PROJECT_NAME}_CommonUtils)
sofa_create_package(${PROJECT_NAME}_Models ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Models ${PROJECT_NAME}_Models)
sofa_create_package(${PROJECT_NAME}_Constraints ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Constraints ${PROJECT_NAME}_Constraints)
sofa_create_package(${PROJECT_NAME}_Simulation ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Simulation ${PROJECT_NAME}_Simulation)
sofa_create_package(${PROJECT_NAME}_Dynamics ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME}_Dynamics ${PROJECT_NAME}_Dynamics)
sofa_create_package(${PROJECT_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
