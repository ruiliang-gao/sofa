cmake_minimum_required(VERSION 3.1)
project(SofaNonUniformFem)


message(STATUS "SofaNonUniformFem:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initNonUniformFEM.h
)

set(SOURCE_FILES
    initNonUniformFEM.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        DynamicSparseGridGeometryAlgorithms.h
        DynamicSparseGridGeometryAlgorithms.inl
        DynamicSparseGridTopologyAlgorithms.h
        DynamicSparseGridTopologyContainer.h
        DynamicSparseGridTopologyModifier.h
        MultilevelHexahedronSetTopologyContainer.h
        SparseGridMultipleTopology.h
        SparseGridRamificationTopology.h
        )
    list(APPEND SOURCE_FILES
        DynamicSparseGridGeometryAlgorithms.cpp
        DynamicSparseGridTopologyAlgorithms.cpp
        DynamicSparseGridTopologyContainer.cpp
        DynamicSparseGridTopologyModifier.cpp
        MultilevelHexahedronSetTopologyContainer.cpp
        SparseGridMultipleTopology.cpp
        SparseGridRamificationTopology.cpp
        )

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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaGeneralSimpleFem SofaBaseTopology SofaBaseMechanics)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_NON_UNIFORM_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaAdvanced ${PROJECT_NAME} ${PROJECT_NAME})
