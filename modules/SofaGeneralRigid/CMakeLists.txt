cmake_minimum_required(VERSION 3.1)
project(SofaGeneralRigid)

set(HEADER_FILES
    config.h
    initGeneralRigid.h
    )

set(SOURCE_FILES
    initGeneralRigid.cpp
    )

list(APPEND HEADER_FILES
    ArticulatedHierarchyContainer.h
    ArticulatedHierarchyContainer.inl
    ArticulatedSystemMapping.h
    ArticulatedSystemMapping.inl
    LineSetSkinningMapping.h
    LineSetSkinningMapping.inl
    SkinningMapping.h
    SkinningMapping.inl
    )
list(APPEND SOURCE_FILES
    ArticulatedHierarchyContainer.cpp
    ArticulatedSystemMapping.cpp
    LineSetSkinningMapping.cpp
    SkinningMapping.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_RIGID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}" SOFA_LIBRARY_INCLUDE_FLAGS "")
else()
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}_d" SOFA_LIBRARY_INCLUDE_FLAGS "")
endif()

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
