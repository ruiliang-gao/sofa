cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper.GL)

find_package(Sofa.DefaultType)
find_package(Sofa.Helper.Image)

set(HEADER_FILES
    src/sofa/helper/gl/Trackball.h
    src/sofa/helper/gl/Transformation.h
)

set(SOURCE_FILES
    src/sofa/helper/gl/Trackball.cpp
    src/sofa/helper/gl/Transformation.cpp
)

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES
            src/sofa/helper/gl/config.h
            src/sofa/helper/gl/Capture.h
            src/sofa/helper/gl/Color.h
            src/sofa/helper/gl/RAII.h
            src/sofa/helper/gl/glfont.h
            src/sofa/helper/gl/template.h
            src/sofa/helper/gl/Axis.h
            src/sofa/helper/gl/BasicShapes.h
            src/sofa/helper/gl/BasicShapesGL.h
            src/sofa/helper/gl/BasicShapesGL.inl
            src/sofa/helper/gl/Cylinder.h
            src/sofa/helper/gl/Texture.h
            src/sofa/helper/gl/Trackball.h
            src/sofa/helper/gl/Transformation.h
            src/sofa/helper/gl/glText.h
            src/sofa/helper/gl/glText.inl
    )
    list(APPEND SOURCE_FILES
            src/sofa/helper/gl/Axis.cpp
            src/sofa/helper/gl/BasicShapesGL.cpp
            src/sofa/helper/gl/Cylinder.cpp
            src/sofa/helper/gl/glText.cpp
            src/sofa/helper/gl/Capture.cpp
            src/sofa/helper/gl/Texture.cpp
            src/sofa/helper/gl/Color.cpp
            src/sofa/helper/gl/glfont.cpp
    )

    if(SOFA_HAVE_GLUT)
        list(APPEND HEADER_FILES
            system/glut.h
        )
    endif()

    if(SOFA_HAVE_GLEW)
        list(APPEND HEADER_FILES
                src/sofa/helper/gl/FrameBufferObject.h
                src/sofa/helper/gl/GLSLShader.h
                src/sofa/helper/gl/Texture.h
        )
        list(APPEND SOURCE_FILES
                src/sofa/helper/gl/FrameBufferObject.cpp
                src/sofa/helper/gl/GLSLShader.cpp
        )
        list(APPEND SHADER_FILES
                src/sofa/helper/gl/shaders/generateSphere.cppglsl
        )
    endif()

endif()


 if(SOFA-EXTERNAL_FFMPEG)
    list(APPEND HEADER_FILES gl/VideoRecorder.h)

    list(APPEND SOURCE_FILES "gl/VideoRecorder.cpp")
        include_directories(${SOFA-EXTERNAL_FFMPEG_PATH}/include)
endif()



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.DefaultType Sofa.Helper.Image)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_GL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")

