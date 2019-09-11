cmake_minimum_required(VERSION 3.1)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++14")

# find_package(SofaFramework REQUIRED)

## Eigen
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR}
                    ${EIGEN3_INCLUDE_DIR}/unsupported)

include_directories(${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics
                    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator
                    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable
                    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/extern/gl3w)

set(IPD_SIMULATOR_LIBRARY_NAME IPDSimulator CACHE INTERNAL "" FORCE)

set (IPD_SIMULATOR_HEADERS
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/Config.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/Simulator.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/SimulationClock.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/SimulationData.hpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/ParticleState.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/RigidBodyState.hpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ContactConstraints.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/Constraints.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ConstraintBase.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/DistanceConstraint.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ParticleConstraint.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyBallJoint.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyParticleJoint.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/TetrahedralConstraints.hpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/Forces.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/ForceBase.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/BodyForce.hpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSIntegrator.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate.hpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_ContactConstraints.inl
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_DistanceConstraint.inl
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_Joints.inl
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSUpdate_TetConstraints.inl

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/ConstraintDataAbstractionUtilities.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/MatrixUtilities.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/SolverUtilities.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/StorageUtilities.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/Timer.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/TypeListUtilities.hpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/Utility.hpp
    )

set (IPD_SIMULATOR_SOURCES
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/ParticleState.cpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/state/RigidBodyState.cpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/ContactConstraints.cpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/DistanceConstraint.cpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyBallJoint.cpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/RigidBodyParticleJoint.cpp
    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/constraints/TetrahedralConstraints.cpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/forces/BodyForce.cpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/integrator/XIBDSIntegrator.cpp

    ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/simulator/utility/Timer.cpp
    )

add_library(${IPD_SIMULATOR_LIBRARY_NAME} STATIC
        ${IPD_SIMULATOR_HEADERS}
        ${IPD_SIMULATOR_SOURCES}
        )

target_link_libraries(${IPD_SIMULATOR_LIBRARY_NAME} gl3w)

set(IPD_DYNAMICS_LIBRARY_NAME IPDDynamics CACHE INTERNAL "" FORCE)

set(IPD_DYNAMICS_BODY_HEADERS
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBody.hpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBodyBuilder.hpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBody.hpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBodyBuilder.hpp
)
set(IPD_DYNAMICS_BODY_SOURCES
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBody.cpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/DeformableBodyBuilder.cpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBody.cpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/bodies/RigidBodyBuilder.cpp
)

set(IPD_DYNAMICS_COLLISION_HEADERS
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/AABB.hpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionDetection.hpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionShapes.hpp
)
set(IPD_DYNAMICS_COLLISION_SOURCES
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionDetection.cpp
        ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics/executable/collision/CollisionShapes.cpp
)

add_library(${IPD_DYNAMICS_LIBRARY_NAME} STATIC
            ${IPD_DYNAMICS_BODY_HEADERS}
            ${IPD_DYNAMICS_BODY_SOURCES}
            ${IPD_DYNAMICS_COLLISION_HEADERS}
            ${IPD_DYNAMICS_COLLISION_SOURCES})

target_link_libraries(${IPD_DYNAMICS_LIBRARY_NAME} gl3w)

#message(STATUS "Creating packages for ${IPD_SIMULATOR_LIBRARY_NAME} and ${IPD_DYNAMICS_LIBRARY_NAME} libraries.")
#sofa_create_package(${IPD_SIMULATOR_LIBRARY_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
#sofa_create_package(${IPD_DYNAMICS_LIBRARY_NAME} ${ZY_POSITION_BASED_DYNAMICS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
