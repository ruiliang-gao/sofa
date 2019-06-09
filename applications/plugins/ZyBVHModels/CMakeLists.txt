project(ZyBVHModelsPlugin)

add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/ZyPQP ${CMAKE_BINARY_DIR}/extlibs/ZyPQP)

add_definitions("-DSOFA_BUILD_BVHMODELSPLUGIN")

set(ZY_BVHMODELS_MAJOR_VERSION 0)
set(ZY_BVHMODELS_MINOR_VERSION 2)
set(ZY_BVHMODELS_VERSION ${ZY_BVHMODELS_MAJOR_VERSION}.${ZY_BVHMODELS_MINOR_VERSION})

find_package(Boost)

set(HEADER_FILES
    initBVHModelsPlugin.h

    PQPModel.h
    PQPModel.inl
    ObbTree.h
    ObbTree.inl
    ObbTreeCPUCollisionModel.h
    ObbTreeCPUCollisionModel.inl
    BVHDrawHelpers.h
    TriangleIntersection.h

    BVHModelsIntersection.h

    BVHTopDownConstruction.h
    BVHTopDownConstruction.inl
)

set(SOURCE_FILES
    initBVHModelsPlugin.cpp

    PQPModel.cpp
    ObbTree.cpp
    ObbTreeCPUCollisionModel.cpp
    BVHDrawHelpers.cpp
    TriangleIntersection.cpp

    BVHModelsIntersection.cpp
    BVHModelsContact.cpp

    BVHTopDownConstruction.cpp
)

if(SOFA-EXTERNAL_GLEW)
    add_definitions("-DSOFA_HAVE_GLEW")
endif()

include_directories(${CMAKE_SOURCE_DIR}/extlibs/ZyPQP/include
                    ${CMAKE_SOURCE_DIR}/framework
                    ${CMAKE_SOURCE_DIR}/modules
                    ${CMAKE_BINARY_DIR}/include
                    ${CMAKE_SOURCE_DIR}/applications/plugins/SofaDistanceGrid
                    ${CMAKE_SOURCE_DIR}/applications/plugins
                    ${EIGEN3_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${Boost_INCLUDE_DIR} # without this, boost/static_assert.hpp can't be found
                    )
					
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(${PROJECT_NAME} ZyPQP tinyxml)
target_link_libraries(${PROJECT_NAME} SofaHelper SofaCore SofaBaseCollision SofaConstraint SofaLoader)

sofa_create_package(${PROJECT_NAME} ${ZY_BVHMODELS_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
