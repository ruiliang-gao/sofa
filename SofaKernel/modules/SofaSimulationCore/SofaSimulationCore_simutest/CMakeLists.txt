cmake_minimum_required(VERSION 3.12)

project(SofaSimulationCore_simutest)

# Do not find dependencies explicitely, let CMake decide the appropriate time to do it
#sofa_find_package(SofaBase REQUIRED)

set(SOURCE_FILES
    DefaultAnimationLoop_test.cpp
    NodeContext_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaBase)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
