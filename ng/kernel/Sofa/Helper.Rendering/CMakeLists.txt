cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper.Rendering)

find_package(Sofa.Helper.Rendering.GL)
#find_package(Sofa.Helper.None)

set(HEADER_FILES
    src/sofa/helper/rendering/Transformation.h
    src/sofa/helper/rendering/FrameBufferObject.h
)

set(SOURCE_FILES
    src/sofa/helper/rendering/Transformation.cpp
    src/sofa/helper/rendering/FrameBufferObject.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper.Rendering.GL)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_GL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
