cmake_minimum_required(VERSION 3.12)
project(SurfLabHaptic)

#find_package(OpenHaptics)
find_package(SofaGeneral REQUIRED)
if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(HEADER_FILES
	initSurfLabHaptic.h
    GraspingForceFeedback.h
	HapticManager.h
	)

set(SOURCE_FILES
	initSurfLabHaptic.cpp
    GraspingForceFeedback.cpp
	HapticManager.cpp	
	)

set(README_FILES SurfLabHaptic.txt)


include_directories(${OPENHAPTICS_INCLUDE_DIR})
include_directories(${OPENHAPTICS_UTIL_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SURFLABHAPTIC")

target_link_libraries(${PROJECT_NAME} SofaOpenglVisual SofaGeneralDeformable SofaConstraint SofaHaptics SofaCore SofaGuiCommon SofaSimulationGraph SofaPython SofaGuiQt SurfLabHapticDevice SofaGuiCommon SofaGuiQt SofaOpenglVisual ${OPENHAPTICS_LIBRARIES} ${Boost_LIBRARIES} winmm.lib)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/.."

${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaGuiQt/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaGuiCommon/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaHaptics/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaGeneralDeformable/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaLoader/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaConstraint/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaOpenglVisual/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaGuiQt/src/sof/gui/qt
)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

