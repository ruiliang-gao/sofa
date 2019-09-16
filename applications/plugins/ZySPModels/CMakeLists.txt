project(ZySPModelsPlugin)
add_definitions("-DSOFA_BUILD_ZYSPMODELS_PLUGIN")

set(ZY_SPMODELS_MAJOR_VERSION 0)
set(ZY_SPMODELS_MINOR_VERSION 2)
set(ZY_SPMODELS_VERSION ${ZY_SPMODELS_MAJOR_VERSION}.${ZY_SPMODELS_MINOR_VERSION})

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

include_directories(${CMAKE_SOURCE_DIR}/extlibs
                    ${CMAKE_SOURCE_DIR}/extlibs/ZyPQP/include
                    ${CMAKE_SOURCE_DIR}/extlibs/tinyxml
                    ${CMAKE_SOURCE_DIR}/extlibs/ZyBVHModels
                    ${CMAKE_SOURCE_DIR}/framework
                    ${CMAKE_SOURCE_DIR}/modules
                    ${CMAKE_BINARY_DIR}/include
                    ${EIGEN3_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})
# target_link_libraries(${PROJECT_NAME} ZyBVHModels ZyPQP)
target_link_libraries(${PROJECT_NAME} SofaDefaultType SofaBaseCollision SofaCore SofaHelper)

sofa_create_package(${PROJECT_NAME} ${ZY_SPMODELS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
