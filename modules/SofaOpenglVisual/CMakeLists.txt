cmake_minimum_required(VERSION 3.1)
project(SofaOpenglVisual)

if(${SOFA_NO_OPENGL})
    message(SEND_ERROR "Warning: CMake flag SOFA_NO_OPENGL is active.")
    message("Flag SOFA_NO_OPENGL incompatible with the SofaOpenglVisual module.")    
endif()

message(STATUS "SofaOpenglVisual:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initOpenGLVisual.h
)

set(SOURCE_FILES
    initOpenGLVisual.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        ClipPlane.h
        OglColorMap.h
        DataDisplay.h
        OglModel.h
        MergeVisualModels.h
        )
    list(APPEND SOURCE_FILES
        ClipPlane.cpp
        OglColorMap.cpp
        DataDisplay.cpp
        OglModel.cpp
        MergeVisualModels.cpp
        )

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
    list(APPEND HEADER_FILES
        PointSplatModel.h
    )
    list(APPEND SOURCE_FILES
        PointSplatModel.cpp
    )
else()
    message(STATUS "   Without deprecated components.")
endif()



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseVisual SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OPENGL_VISUAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
