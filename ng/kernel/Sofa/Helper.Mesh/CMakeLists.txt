cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper.Mesh)

find_package(tinyxml)

set(HEADER_FILES
    src/sofa/helper/io/MassSpringLoader.h
    src/sofa/helper/io/MeshOBJ.h
    src/sofa/helper/io/MeshSTL.h
    src/sofa/helper/io/MeshTrian.h
    src/sofa/helper/io/MeshVTK.h
    src/sofa/helper/io/Mesh.h
    src/sofa/helper/io/MeshTopologyLoader.h
    src/sofa/helper/io/SphereLoader.h
    src/sofa/helper/io/TriangleLoader.h
)

set(SOURCE_FILES
    src/sofa/helper/io/MassSpringLoader.cpp
    src/sofa/helper/io/MeshOBJ.cpp
    src/sofa/helper/io/MeshSTL.cpp
    src/sofa/helper/io/MeshTrian.cpp
    src/sofa/helper/io/MeshVTK.cpp
    src/sofa/helper/io/Mesh.cpp
    src/sofa/helper/io/MeshTopologyLoader.cpp
    src/sofa/helper/io/SphereLoader.cpp
    src/sofa/helper/io/TriangleLoader.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC tinyxml Sofa.DefaultType)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER_MESH")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")


