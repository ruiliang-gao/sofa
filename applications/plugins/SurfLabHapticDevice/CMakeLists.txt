cmake_minimum_required(VERSION 3.1)
project(SurfLabHapticDevice)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

find_package(OpenHaptics)

if( NOT OPENHAPTICS_FOUND )
    message( SEND_ERROR "OpenHaptics library not found, GeomagicDriver cannot compile")
endif()


set(HEADER_FILES
    SurfLabHapticDeviceDriver.h    
	SurfLabHapticInstruments.h
)

set(SOURCE_FILES
    SurfLabHapticDeviceDriver.cpp
    initSurfLabHapticDevice.cpp
	SurfLabHapticInstruments.cpp
)

set(README_FILES PluginSurfLabHapticDevice.txt)

find_package(SofaGeneral REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(${OPENHAPTICS_INCLUDE_DIR})
include_directories(${OPENHAPTICS_UTIL_INCLUDE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/.."
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaGeneralDeformable/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaLoader/src
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} )
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SURFLABHAPTICDEVICE")
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulation SofaHelper SofaHaptics SofaOpenglVisual Sensable ${OPENHAPTICS_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
        COMPONENT SurfLabHapticDevice_libraries
        EXPORT SurfLabHapticDeviceTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})