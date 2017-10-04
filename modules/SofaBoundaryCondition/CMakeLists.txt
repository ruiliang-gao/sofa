cmake_minimum_required(VERSION 3.1)
project(SofaBoundaryCondition)

message(STATUS "SofaBoundaryCondition:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initBoundaryCondition.h
)

set(SOURCE_FILES
    initBoundaryCondition.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        ConstantForceField.h
        ConstantForceField.inl
        
        FixedConstraint.h
        FixedConstraint.inl
        PartialFixedConstraint.h
        PartialFixedConstraint.inl
        PlaneForceField.h
        PlaneForceField.inl 
        SurfacePressureForceField.h
        SurfacePressureForceField.inl
        TaitSurfacePressureForceField.h
        TaitSurfacePressureForceField.inl
        TrianglePressureForceField.h
        TrianglePressureForceField.inl

        )

    list(APPEND SOURCE_FILES
        ConstantForceField.cpp
        FixedConstraint.cpp
        PartialFixedConstraint.cpp
        PlaneForceField.cpp
        SurfacePressureForceField.cpp
        TaitSurfacePressureForceField.cpp
        TrianglePressureForceField.cpp
        )
    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    list(APPEND HEADER_FILES
        BuoyantForceField.h
        BuoyantForceField.inl
        UniformVelocityDampingForceField.h
        UniformVelocityDampingForceField.inl
        VaccumSphereForceField.h
        VaccumSphereForceField.inl
        )
    list(APPEND SOURCE_FILES
        BuoyantForceField.cpp
        UniformVelocityDampingForceField.cpp
        VaccumSphereForceField.cpp
        )
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaEigen2Solver)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  "${SOFAGENERAL_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BOUNDARY_CONDITION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
