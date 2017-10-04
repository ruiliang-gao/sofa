cmake_minimum_required(VERSION 3.1)
project(SofaSpecificBoundaryConditions)

set(SOFASPECIFICBOUNDARYCONDITIONS_VERSION 1.0)

set(HEADER_FILES
    SofaSpecificBoundaryConditions.h
    initSofaSpecificBoundaryConditions.h

    AffineMovementConstraint.h
    AffineMovementConstraint.inl
    ConicalForceField.h
    ConicalForceField.inl
    DiagonalVelocityDampingForceField.h
    DiagonalVelocityDampingForceField.inl
    EdgePressureForceField.h
    EdgePressureForceField.inl
    EllipsoidForceField.h
    EllipsoidForceField.inl 
    FixedPlaneConstraint.h
    FixedPlaneConstraint.inl
    FixedRotationConstraint.h
    FixedRotationConstraint.inl
    FixedTranslationConstraint.h
    FixedTranslationConstraint.inl
    HermiteSplineConstraint.h
    HermiteSplineConstraint.inl
    LinearForceField.h
    LinearForceField.inl
    LinearMovementConstraint.h
    LinearMovementConstraint.inl
    LinearVelocityConstraint.h
    LinearVelocityConstraint.inl
    OscillatingTorsionPressureForceField.h
    OscillatingTorsionPressureForceField.inl
    OscillatorConstraint.h
    OscillatorConstraint.inl
    ParabolicConstraint.h
    ParabolicConstraint.inl
	PartialLinearMovementConstraint.h
    PartialLinearMovementConstraint.inl
    PatchTestMovementConstraint.h
    PatchTestMovementConstraint.inl
    PointConstraint.h
    PointConstraint.inl
    PositionBasedDynamicsConstraint.h
    PositionBasedDynamicsConstraint.inl
    QuadPressureForceField.h
    QuadPressureForceField.inl
    SkeletalMotionConstraint.h
    SkeletalMotionConstraint.inl
    SphereForceField.h
    SphereForceField.inl
    TorsionForceField.h
	TorsionForceField.inl
    ProjectToLineConstraint.h
    ProjectToLineConstraint.inl
    ProjectToPlaneConstraint.h
    ProjectToPlaneConstraint.inl
    ProjectToPointConstraint.h
    ProjectToPointConstraint.inl
    ProjectDirectionConstraint.h
    ProjectDirectionConstraint.inl
)

set(SOURCE_FILES
    initSofaSpecificBoundaryConditions.cpp

    AffineMovementConstraint.cpp
    ConicalForceField.cpp
    DiagonalVelocityDampingForceField.cpp
    EdgePressureForceField.cpp
    EllipsoidForceField.cpp
    FixedPlaneConstraint.cpp
    FixedRotationConstraint.cpp
    FixedTranslationConstraint.cpp
    HermiteSplineConstraint.cpp
    LinearForceField.cpp
    LinearMovementConstraint.cpp
    LinearVelocityConstraint.cpp
    OscillatingTorsionPressureForceField.cpp
    OscillatorConstraint.cpp
    ParabolicConstraint.cpp
    PartialLinearMovementConstraint.cpp
    PatchTestMovementConstraint.cpp
	PointConstraint.cpp
    PositionBasedDynamicsConstraint.cpp
    QuadPressureForceField.cpp
    SkeletalMotionConstraint.cpp
    SphereForceField.cpp
    TorsionForceField.cpp
    ProjectToLineConstraint.cpp
    ProjectToPlaneConstraint.cpp
    ProjectToPointConstraint.cpp
    ProjectDirectionConstraint.cpp
)

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(SofaSpecificBoundaryConditions_test)
    endif()
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBoundaryCondition)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFA_SPECIFIC_BOUNDARY_CONDITIONS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
sofa_create_package(SofaSpecificBoundaryConditions ${SOFASPECIFICBOUNDARYCONDITIONS_VERSION} ${PROJECT_NAME} SofaSpecificBoundaryConditions)
