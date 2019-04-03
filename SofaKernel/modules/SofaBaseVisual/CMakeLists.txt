cmake_minimum_required(VERSION 3.1)
project(SofaBaseVisual)

set(HEADER_FILES
    BaseCamera.h
    BackgroundSetting.h
    Camera.h
    InteractiveCamera.h
    VisualModelImpl.h
    VisualStyle.h
    config.h
    initBaseVisual.h
)

set(SOURCE_FILES
    BaseCamera.cpp
    BackgroundSetting.cpp
    Camera.cpp
    InteractiveCamera.cpp
    VisualModelImpl.cpp
    VisualStyle.cpp
    initBaseVisual.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_VISUAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}" SOFA_LIBRARY_INCLUDE_FLAGS "")
else()
    sofa_generate_pkgconfig_file(SOFA_LIBRARY_NAME ${PROJECT_NAME} SOFA_LIBRARY_DESCRIPTION "The ${PROJECT_NAME} library." SOFA_LIBRARY_SO_NAME "${PROJECT_NAME}_d" SOFA_LIBRARY_INCLUDE_FLAGS "")
endif()

sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
