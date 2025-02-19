cmake_minimum_required(VERSION 3.12)
project(SofaAssimp VERSION 0.2)


set(HEADER_FILES
    config.h
    AssimpLoader.h
    SceneColladaLoader.h
)

set(SOURCE_FILES
    initAssimpPlugin.cpp
    AssimpLoader.cpp
    SceneColladaLoader.cpp
)

set(README_FILES README_AssimpPlugin.txt)


## Assimp dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

### Dependencies
sofa_find_package(Assimp REQUIRED)
sofa_find_package(SofaLoader REQUIRED)
sofa_find_package(SofaOpenglVisual REQUIRED)
sofa_find_package(SofaBoundaryCondition REQUIRED)
sofa_find_package(SofaGeneralRigid REQUIRED)
sofa_find_package(SofaMeshCollision REQUIRED)
sofa_find_package(image QUIET)
sofa_find_package(Flexible QUIET)

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAASSIMP")

target_link_libraries(${PROJECT_NAME} PUBLIC ${ASSIMP_LIBRARY} SofaLoader SofaOpenglVisual SofaBoundaryCondition SofaGeneralRigid SofaMeshCollision)
if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC image)
endif()
if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Flexible)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/..")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${ASSIMP_INCLUDE_DIR}>")


if(ASSIMP_DLL)
    sofa_install_libraries(PATHS ${ASSIMP_DLL})
endif()


#TODO
if(SOFA_BUILD_TESTS)
   find_package(SofaTest QUIET)
#   add_subdirectory(CGALPlugin_test)
endif()

## Install rules for the library; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
