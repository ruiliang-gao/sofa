cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper.BVH)

find_package(Sofa.DefaultType)
find_package(Sofa.Helper.Rendering.GL)

set(HEADER_FILES
    src/sofa/helper/io/bvh/BVHChannels.h
    src/sofa/helper/io/bvh/BVHJoint.h
    src/sofa/helper/io/bvh/BVHLoader.h
    src/sofa/helper/io/bvh/BVHMotion.h
    src/sofa/helper/io/bvh/BVHOffset.h
)

set(SOURCE_FILES
    src/sofa/helper/io/bvh/BVHJoint.cpp
    src/sofa/helper/io/bvh/BVHLoader.cpp
    src/sofa/helper/io/bvh/BVHMotion.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC tinyxml Sofa.Helper.Rendering.GL)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER_BVH")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
