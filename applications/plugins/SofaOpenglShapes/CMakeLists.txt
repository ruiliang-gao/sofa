cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(SofaOpenglShapes)

set(OPENGGLSHAPES_MAJOR_VERSION 1)
set(OPENGGLSHAPES_MINOR_VERSION 0)
set(OPENGGLSHAPES_VERSION ${OPENGGLSHAPES_MAJOR_VERSION}.${OPENGGLSHAPES_MINOR_VERSION})

#set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(SofaFramework REQUIRED)
find_package(SofaGeneral REQUIRED)

include_directories(../${CMAKE_CURRENT_SOURCE_DIR})
  
set(HEADER_FILES
	config.h
	OglCylinderModel.h
	OglGrid.h
	OglLabel.h
	OglLineAxis.h
	OglRenderingSRGB.h
	OglSceneFrame.h
 )

set(SOURCE_FILES
	initSofaOpenglShapes.cpp
	OglCylinderModel.cpp
	OglGrid.cpp
	OglLabel.cpp
	OglLineAxis.cpp
	OglRenderingSRGB.cpp
	OglSceneFrame.cpp
 )

set(README_FILE
	readme.txt
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${OPENGGLSHAPES_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OPENGL_SHAPES")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_link_libraries(${PROJECT_NAME} ${GENERAL_DEPENDENCIES} SofaComponentBase SofaOpenglVisual SofaGraphComponent)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")


install(TARGETS ${PROJECT_NAME}
        COMPONENT OPENGGLSHAPES_libraries
        EXPORT OPENGGLSHAPESTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
