cmake_minimum_required(VERSION 3.12)
project(SofaGeneralExplicitOdeSolver LANGUAGES CXX)

set(SOFAGENERALEXPLICITODESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/config.h.in
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/initSofaGeneralExplicitOdeSolver.h
    )

set(SOURCE_FILES
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/initSofaGeneralExplicitOdeSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.ODESolver.Forward instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.ODESolver.Forward REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.ODESolver.Forward)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
