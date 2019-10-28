add_library(Simulation
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/AABB.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CollisionDetection.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CollisionDetection.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/Constraints.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/Constraints.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CubicSDFCollisionDetection.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/CubicSDFCollisionDetection.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/DistanceFieldCollisionDetection.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/DistanceFieldCollisionDetection.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/IDFactory.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/IDFactory.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/LineModel.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/LineModel.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/NeighborhoodSearchSpatialHashing.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/NeighborhoodSearchSpatialHashing.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/ParticleData.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/RigidBody.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/RigidBodyGeometry.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/RigidBodyGeometry.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/Simulation.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/Simulation.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/SimulationModel.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/SimulationModel.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TetModel.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TetModel.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TimeManager.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TimeManager.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TimeStep.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TimeStep.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TimeStepController.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TimeStepController.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TriangleModel.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/TriangleModel.h
		
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/BoundingSphere.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/BoundingSphereHierarchy.cpp
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/BoundingSphereHierarchy.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/kdTree.h
		${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Simulation/kdTree.inl

		CMakeLists.txt
)

############################################################
# Discregrid
############################################################
include_directories(${ExternalInstallDir}/Discregrid/include)
set(SIMULATION_DEPENDENCIES ${SIMULATION_DEPENDENCIES} Discregrid)

############################################################
# GenericParameters
############################################################
include_directories(${ExternalInstallDir}/GenericParameters/include)
set(SIMULATION_DEPENDENCIES ${SIMULATION_DEPENDENCIES} GenericParameters)

add_dependencies(Simulation ${SIMULATION_DEPENDENCIES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(/${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/extern/Miniball)

target_link_libraries(Simulation PositionBasedDynamics)


install(TARGETS Simulation
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY .
DESTINATION include/Simulation
FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
