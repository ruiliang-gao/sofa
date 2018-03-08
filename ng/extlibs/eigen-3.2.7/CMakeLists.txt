cmake_minimum_required(VERSION 3.1)


project(Eigen)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Eigen
        COMPONENT Eigen_headers
        DESTINATION include
        PATTERN "*.in" EXCLUDE
        PATTERN "*.txt" EXCLUDE
        PATTERN "*.cpp" EXCLUDE)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen
        COMPONENT Eigen_headers
        DESTINATION include
        PATTERN "*.in" EXCLUDE
        PATTERN "*.txt" EXCLUDE
        PATTERN "*.cpp" EXCLUDE)
