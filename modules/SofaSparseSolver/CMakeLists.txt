cmake_minimum_required(VERSION 3.1)
project(SofaSparseSolver)

set(HEADER_FILES
    PrecomputedLinearSolver.h
    PrecomputedLinearSolver.inl
    SparseCholeskySolver.h
    SparseLUSolver.h
    config.h
    initSparseSolver.h
)

set(SOURCE_FILES
    PrecomputedLinearSolver.cpp
    SparseCholeskySolver.cpp
    SparseLUSolver.cpp
    initSparseSolver.cpp
)

if(SOFA_HAVE_METIS)
    list(APPEND HEADER_FILES SparseLDLSolver.h SparseLDLSolver.inl SparseLDLSolverImpl.h)
    list(APPEND SOURCE_FILES SparseLDLSolver.h SparseLDLSolver.cpp)
    message(STATUS "SofaSparseSolver: metis was found, SparseLDLSolver will be built")
else()
    message(STATUS "SofaSparseSolver: metis was not found, SparseLDLSolver won't be built")
endif()

if(SOFA_HAVE_CSPARSE)
    list(APPEND HEADER_FILES SparseCholeskySolver.h)
    list(APPEND HEADER_FILES SparseLUSolver.h)
    list(APPEND SOURCE_FILES SparseCholeskySolver.cpp)
    list(APPEND SOURCE_FILES SparseLUSolver.cpp)
    message(STATUS "SofaSparseSolver: csparse was found, SparseLUSolver and SparseCholeskySolver will be built")
else()
    message(STATUS "SofaSparseSolver: csparse was not found, SparseLUSolver and SparseCholeskySolver won't be built")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaImplicitOdeSolver SofaSimpleFem)
if(Metis_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC metis)
endif()
if(SOFA_HAVE_CSPARSE)
    target_link_libraries(${PROJECT_NAME} PUBLIC csparse)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SPARSE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}" SOFA_LIBRARY_INCLUDE_FLAGS "")
else()
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}_d" SOFA_LIBRARY_INCLUDE_FLAGS "")
endif()

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
