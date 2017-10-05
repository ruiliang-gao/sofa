cmake_minimum_required(VERSION 3.1)
project(SofaSpecificNonUniformFem)

set(SOFASPECIFICNONUNIFORMFEM_VERSION 1.0)

# Dependencies
find_package(SofaDenseSolver)


message(STATUS "SofaMisc:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    SofaSpecificNonUniformFem.h
    initSofaSpecificNonUniformFem.h
)

set(SOURCE_FILES
    initSofaSpecificNonUniformFem.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
       HexahedronCompositeFEMMapping.h
        HexahedronCompositeFEMMapping.inl
        NonUniformHexahedralFEMForceFieldAndMass.h
        NonUniformHexahedralFEMForceFieldAndMass.inl
        NonUniformHexahedronFEMForceFieldAndMass.h
        NonUniformHexahedronFEMForceFieldAndMass.inl
        )
    list(APPEND SOURCE_FILES
        HexahedronCompositeFEMMapping.cpp
        NonUniformHexahedralFEMForceFieldAndMass.cpp
        NonUniformHexahedronFEMForceFieldAndMass.cpp
        )

    if(${SofaDenseSolver_FOUND})
        list(APPEND HEADER_FILES
             HexahedronCompositeFEMForceFieldAndMass.h
             HexahedronCompositeFEMForceFieldAndMass.inl
        )
        list(APPEND SOURCE_FILES
             HexahedronCompositeFEMForceFieldAndMass.cpp
        )
    endif()

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")

endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
if(${SofaDenseSolver_FOUND})
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC SofaNonUniformFem)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFA_SPECIFIC_NON_UNIFORM_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
sofa_create_package(SofaSpecificNonUniformFem ${SOFASPECIFICNONUNIFORMFEM_VERSION} ${PROJECT_NAME} SofaSpecificNonUniformFem)
