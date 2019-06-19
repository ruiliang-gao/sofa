cmake_minimum_required(VERSION 2.8.12)

project(ZyCollisionManager)

set(ZY_COLLISIONMANAGER_MAJOR_VERSION 0)
set(ZY_COLLISIONMANAGER_MINOR_VERSION 1)
set(ZY_COLLISIONMANAGER_VERSION ${ZY_COLLISIONMANAGER_MAJOR_VERSION}.${ZY_COLLISIONMANAGER_MINOR_VERSION})

set(HEADER_FILES
	ZyCollisionManager.h
	config_collisionManager.h
	)
	
set(SOURCE_FILES
	ZyCollisionManager.cpp
	)
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaCore SofaHelper SofaDefaultType SofaSimulationCommon)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DBUILD_ZY_COLLISIONMANAGER")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${ZY_COLLISIONMANAGER_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(ZyCollisionManager ${ZY_COLLISIONMANAGER_VERSION} ZyCollisionManager ZyCollisionManager)
