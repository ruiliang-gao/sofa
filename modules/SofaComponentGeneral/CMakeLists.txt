cmake_minimum_required(VERSION 3.1)
project(SofaComponentGeneral)

set(HEADER_FILES
    config.h
    initComponentGeneral.h
    )

set(SOURCE_FILES
    initComponentGeneral.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC
    SofaGeneralAnimationLoop SofaGeneralDeformable SofaGeneralExplicitOdeSolver SofaGeneralImplicitOdeSolver SofaGeneralLinearSolver SofaGeneralLoader
    SofaGeneralMeshCollision SofaGeneralObjectInteraction SofaGeneralRigid SofaGeneralSimpleFem SofaGeneralTopology  SofaGeneralVisual SofaBoundaryCondition
    SofaConstraint SofaGeneralEngine SofaExporter SofaGraphComponent SofaHaptics SofaPreconditioner SofaSparseSolver SofaTopologyMapping SofaUserInteraction SofaValidation)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_GENERAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if(SOFA_HAVE_NEWMAT)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
endif()
if(NOT SOFA_NO_OPENGL)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaOpenglVisual)
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}" SOFA_LIBRARY_INCLUDE_FLAGS "")
else()
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}_d" SOFA_LIBRARY_INCLUDE_FLAGS "")
endif()

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
