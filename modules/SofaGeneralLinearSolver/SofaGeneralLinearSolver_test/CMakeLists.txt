cmake_minimum_required(VERSION 3.12)

project(SofaGeneralLinearSolver_test)

set(SOURCE_FILES
)

find_package(SofaGeneralLinearSolver REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaGeneralLinearSolver)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})


