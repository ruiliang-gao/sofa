cmake_minimum_required(VERSION 3.12)

project(SofaEngine_test)


set(HEADER_FILES
        )

set(SOURCE_FILES
    BoxROI_test.cpp
    )
    
sofa_find_package(SofaEngine REQUIRED)
sofa_find_package(SofaBaseUtils REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)
sofa_find_package(SofaBaseTopology REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaEngine SofaBaseUtils SofaBaseMechanics SofaBaseTopology)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
