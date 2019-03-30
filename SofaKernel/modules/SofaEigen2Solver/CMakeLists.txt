cmake_minimum_required(VERSION 3.1)
project(SofaEigen2Solver)

set(HEADER_FILES
    EigenBaseSparseMatrix.h
    EigenBaseSparseMatrix_MT.h
    EigenMatrixManipulator.h
    EigenSparseMatrix.h
    # EigenSparseSquareMatrix.h
    EigenVector.h
    EigenVectorWrapper.h
    SVDLinearSolver.h
    config.h
    initEigen2Solver.h
)

set(SOURCE_FILES
    EigenMatrixManipulator.cpp
    EigenVector.cpp
    SVDLinearSolver.cpp
    initEigen2Solver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EIGEN2_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}" SOFA_LIBRARY_INCLUDE_FLAGS "")
else()
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}_d" SOFA_LIBRARY_INCLUDE_FLAGS "")
endif()

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
