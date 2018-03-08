cmake_minimum_required(VERSION 3.1)

project(tinyxml VERSION 2.6.2)

set(SOURCE_FILES
    src/tinystr.cpp
    src/tinyxml.cpp
    src/tinyxmlerror.cpp
    src/tinyxmlparser.cpp
)

set(HEADER_FILES
    src/tinystr.h
    src/tinyxml.h
)


add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Findtinyxml)
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
