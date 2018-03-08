cmake_minimum_required(VERSION 3.1)
project(Sofa.Simulation)

find_package(Sofa.Core)

set(HEADER_FILES
        src/sofa/simulation/AnimateBeginEvent.h
        src/sofa/simulation/AnimateEndEvent.h
        src/sofa/simulation/AnimateVisitor.h
        src/sofa/simulation/BehaviorUpdatePositionVisitor.h
        src/sofa/simulation/CactusStackStorage.h
        src/sofa/simulation/ClassSystem.h
        src/sofa/simulation/CleanupVisitor.h
        src/sofa/simulation/CollisionAnimationLoop.h
        src/sofa/simulation/CollisionBeginEvent.h
        src/sofa/simulation/CollisionEndEvent.h
        src/sofa/simulation/CollisionVisitor.h
        src/sofa/simulation/Colors.h
        src/sofa/simulation/CopyAspectVisitor.h
        src/sofa/simulation/DeactivatedNodeVisitor.h
        src/sofa/simulation/DefaultAnimationLoop.h
        src/sofa/simulation/DefaultVisualManagerLoop.h
        src/sofa/simulation/DeleteVisitor.h
        src/sofa/simulation/ExportGnuplotVisitor.h
        src/sofa/simulation/ExportOBJVisitor.h
        src/sofa/simulation/InitVisitor.h
        src/sofa/simulation/IntegrateBeginEvent.h
        src/sofa/simulation/IntegrateEndEvent.h
        src/sofa/simulation/LocalStorage.h
        src/sofa/simulation/MechanicalComputeEnergyVisitor.h
        src/sofa/simulation/MechanicalMatrixVisitor.h
        src/sofa/simulation/MechanicalOperations.h
        src/sofa/simulation/MechanicalVPrintVisitor.h
        src/sofa/simulation/MechanicalVisitor.h
        src/sofa/simulation/MutationListener.h
        src/sofa/simulation/Node.h
        src/sofa/simulation/Node.inl
        src/sofa/simulation/ParallelVisitorScheduler.h
        src/sofa/simulation/PauseEvent.h
        src/sofa/simulation/PipelineImpl.h
        src/sofa/simulation/PositionEvent.h
        src/sofa/simulation/PrintVisitor.h
        src/sofa/simulation/PropagateEventVisitor.h
        src/sofa/simulation/ReleaseAspectVisitor.h
        src/sofa/simulation/ResetVisitor.h
        src/sofa/simulation/SceneLoaderFactory.h
        src/sofa/simulation/Simulation.h
        src/sofa/simulation/SolveVisitor.h
        src/sofa/simulation/StateChangeVisitor.h
        src/sofa/simulation/TopologyChangeVisitor.h
        src/sofa/simulation/UpdateBoundingBoxVisitor.h
        src/sofa/simulation/UpdateContextVisitor.h
        src/sofa/simulation/UpdateLinksVisitor.h
        src/sofa/simulation/UpdateMappingEndEvent.h
        src/sofa/simulation/UpdateMappingVisitor.h
        src/sofa/simulation/VectorOperations.h
        src/sofa/simulation/VelocityThresholdVisitor.h
        src/sofa/simulation/Visitor.h
        src/sofa/simulation/VisitorExecuteFunc.h
        src/sofa/simulation/VisitorScheduler.h
        src/sofa/simulation/VisualVisitor.h
        src/sofa/simulation/WriteStateVisitor.h
        src/sofa/simulation/XMLPrintVisitor.h
        src/sofa/simulation/init.h
        src/sofa/simulation/config.h
        src/sofa/simulation/MechanicalGetMomentumVisitor.h
        src/sofa/simulation/BaseSimulationExporter.h
)

set(SOURCE_FILES
        src/sofa/simulation/AnimateBeginEvent.cpp
        src/sofa/simulation/AnimateEndEvent.cpp
        src/sofa/simulation/AnimateVisitor.cpp
        src/sofa/simulation/BehaviorUpdatePositionVisitor.cpp
        src/sofa/simulation/CactusStackStorage.cpp
        src/sofa/simulation/CleanupVisitor.cpp
        src/sofa/simulation/CollisionAnimationLoop.cpp
        src/sofa/simulation/CollisionBeginEvent.cpp
        src/sofa/simulation/CollisionEndEvent.cpp
        src/sofa/simulation/CollisionVisitor.cpp
        src/sofa/simulation/CopyAspectVisitor.cpp
        src/sofa/simulation/DeactivatedNodeVisitor.cpp
        src/sofa/simulation/DefaultAnimationLoop.cpp
        src/sofa/simulation/DefaultVisualManagerLoop.cpp
        src/sofa/simulation/DeleteVisitor.cpp
        src/sofa/simulation/ExportGnuplotVisitor.cpp
        src/sofa/simulation/ExportOBJVisitor.cpp
        src/sofa/simulation/InitVisitor.cpp
        src/sofa/simulation/IntegrateBeginEvent.cpp
        src/sofa/simulation/IntegrateEndEvent.cpp
        src/sofa/simulation/MechanicalMatrixVisitor.cpp
        src/sofa/simulation/MechanicalOperations.cpp
        src/sofa/simulation/MechanicalVPrintVisitor.cpp
        src/sofa/simulation/MechanicalVisitor.cpp
        src/sofa/simulation/MutationListener.cpp
        src/sofa/simulation/Node.cpp
        src/sofa/simulation/ParallelVisitorScheduler.cpp
        src/sofa/simulation/PauseEvent.cpp
        src/sofa/simulation/PipelineImpl.cpp
        src/sofa/simulation/PositionEvent.cpp
        src/sofa/simulation/PrintVisitor.cpp
        src/sofa/simulation/PropagateEventVisitor.cpp
        src/sofa/simulation/ReleaseAspectVisitor.cpp
        src/sofa/simulation/ResetVisitor.cpp
        src/sofa/simulation/SceneLoaderFactory.cpp
        src/sofa/simulation/Simulation.cpp
        src/sofa/simulation/SolveVisitor.cpp
        src/sofa/simulation/StateChangeVisitor.cpp
        src/sofa/simulation/TopologyChangeVisitor.cpp
        src/sofa/simulation/UpdateBoundingBoxVisitor.cpp
        src/sofa/simulation/UpdateContextVisitor.cpp
        src/sofa/simulation/UpdateLinksVisitor.cpp
        src/sofa/simulation/UpdateMappingEndEvent.cpp
        src/sofa/simulation/UpdateMappingVisitor.cpp
        src/sofa/simulation/VectorOperations.cpp
        src/sofa/simulation/VelocityThresholdVisitor.cpp
        src/sofa/simulation/Visitor.cpp
        src/sofa/simulation/VisitorScheduler.cpp
        src/sofa/simulation/VisualVisitor.cpp
        src/sofa/simulation/WriteStateVisitor.cpp
        src/sofa/simulation/XMLPrintVisitor.cpp
        src/sofa/simulation/init.cpp
        src/sofa/simulation/MechanicalComputeEnergyVisitor.cpp
        src/sofa/simulation/BaseSimulationExporter.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DBUILD_TARGET_SOFA_SIMULATION")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/sofa/simulation"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
