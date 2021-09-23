cmake_minimum_required(VERSION 3.12)
project(SurfLabConnectingTissue)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(HEADER_FILES
	initConnectingTissue.h
    ConnectingTissue.h
	)

set(SOURCE_FILES
	initConnectingTissue.cpp
    ConnectingTissue.cpp
	)

set(README_FILES ConnectingTissue.txt)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CONNECTINGTISSUE")
target_link_libraries(${PROJECT_NAME} SofaCore SofaOpenglVisual SofaGeneralDeformable SofaConstraint
SofaHelper SofaDefaultType SofaBaseVisual SofaUserInteraction SofaBaseMechanics SofaSimulationGraph)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/.."
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaCore/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaHelper/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaDefaultType/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaSimulationCore/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaBaseMechanics/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaOpenglVisual/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaUserInteraction/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaConstraint/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../modules/SofaGeneralDeformable/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaFramework/src
${CMAKE_CURRENT_SOURCE_DIR}/../../../SofaKernel/modules/SofaHelper/src

	)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

