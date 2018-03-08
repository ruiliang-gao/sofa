cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper.Image)

find_package(Sofa.DefaultType)

set(HEADER_FILES
    src/sofa/helper/io/Image.h
    src/sofa/helper/io/ImageDDS.h
    src/sofa/helper/io/ImageRAW.h
)

set(SOURCE_FILES
    src/sofa/helper/io/Image.cpp
    src/sofa/helper/io/ImageDDS.cpp
    src/sofa/helper/io/ImageRAW.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC tinyxml Sofa.DefaultType)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER_IMAGE")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
