cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(SofaOpenglAdvanced)

set(OPENGGLADVANCED_MAJOR_VERSION 1)
set(OPENGGLADVANCED_MINOR_VERSION 0)
set(OPENGGLADVANCED_VERSION ${OPENGGLADVANCED_MAJOR_VERSION}.${OPENGGLADVANCED_MINOR_VERSION})

#set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(SofaFramework REQUIRED)
find_package(SofaGeneral REQUIRED)

include_directories(../${CMAKE_CURRENT_SOURCE_DIR})
  
set(HEADER_FILES
	config.h
    CompositingVisualLoop.h
    Light.h
	LightManager.h
	OglOITShader.h
    OglAttribute.h
    OglAttribute.inl
    OglShader.h
    OglShaderMacro.h
    OglShaderVisualModel.h
    OglShadowShader.h
    OglTexture.h
    OglTexturePointer.h
    OglVariable.h
    OglVariable.inl
    OglViewport.h
	OrderIndependentTransparencyManager.h
    PostProcessManager.h
    SlicedVolumetricModel.h
    VisualManagerPass.h
    VisualManagerSecondaryPass.h
 )

set(SOURCE_FILES
	initSofaOpenglAdvanced.cpp
	CompositingVisualLoop.cpp
	Light.cpp
	LightManager.cpp
	OglOITShader.cpp
    OglAttribute.cpp
    OglShader.cpp
    OglShaderMacro.cpp
    OglShaderVisualModel.cpp
    OglShadowShader.cpp
    OglTexture.cpp
    OglTexturePointer.cpp
    OglVariable.cpp
    OglViewport.cpp
	OrderIndependentTransparencyManager.cpp
    PostProcessManager.cpp
    SlicedVolumetricModel.cpp
    VisualManagerPass.cpp
    VisualManagerSecondaryPass.cpp
 )

set(README_FILE
	readme.txt
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${OPENGGLADVANCED_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OPENGL_ADVANCED")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_link_libraries(${PROJECT_NAME} ${GENERAL_DEPENDENCIES} SofaComponentBase SofaOpenglVisual)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")


install(TARGETS ${PROJECT_NAME}
        COMPONENT OPENGGLADVANCED_libraries
        EXPORT OPENGGLADVANCEDTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
