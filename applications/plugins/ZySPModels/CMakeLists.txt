project(SPModels)
add_definitions("-DSOFA_BUILD_SPMODELS_PLUGIN")

set(HEADER_FILES
    initSPModels.h
    PointHull_Computation.h
    PointHull_Container.h
    #PointHull.h
    #PointHull.inl
    PointHullModel.h

    Morton.h
    Octree_Container.h
    Octree_Computation.h

    BVHDrawHelpers.h
)

set(SOURCE_FILES
    initSPModels.cpp
    PointHull_Computation.cpp
    PointHull_Container.cpp
    #PointHull.cpp
    PointHullModel.cpp

    Morton.cpp
    Octree_Container.cpp
    Octree_Computation.cpp

    BVHDrawHelpers.cpp
)

if(SOFA-EXTERNAL_GLEW)
    add_definitions("-DSOFA_HAVE_GLEW")
endif()

include_directories(${CMAKE_SOURCE_DIR}/trulibs
                    ${CMAKE_SOURCE_DIR}/trulibs/PQP/include
                    ${CMAKE_SOURCE_DIR}/extlibs/tinyxml
                    ${CMAKE_SOURCE_DIR}/trulibs/BVHModels
                    ${CMAKE_SOURCE_DIR}/framework
                    ${CMAKE_SOURCE_DIR}/modules
                    ${CMAKE_BINARY_DIR}/include
                    ${EIGEN3_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} BVHModels PQP)
target_link_libraries(${PROJECT_NAME} SofaDefaultType SofaBaseCollision SofaCore SofaHelper)

if (TRUPHYSICS_INSTALLER)
    sofa_install_library(LIBRARY_NAME ${PROJECT_NAME} LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_HEADER_FILES ${HEADER_FILES})
endif()
