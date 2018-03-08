cmake_minimum_required(VERSION 3.1)
 
project(Sofa.DefaultType)
string(TOUPPER ${PROJECT_NAME} PROJECT_UNAME)
string(MAKE_C_IDENTIFIER ${PROJECT_UNAME} PROJECT_CNAME)

find_package(Sofa.Config)
find_package(Sofa.Messaging)
find_package(Sofa.Helper)

set(HEADER_FILES
    src/sofa/defaulttype/BaseMatrix.h
    src/sofa/defaulttype/BaseVector.h
    src/sofa/defaulttype/BoundingBox.h
    src/sofa/defaulttype/Frame.h
    src/sofa/defaulttype/MapMapSparseMatrix.h
    src/sofa/defaulttype/Mat.h
    src/sofa/defaulttype/MatSym.h
    src/sofa/defaulttype/Mat_solve_Cholesky.h
    src/sofa/defaulttype/Mat_solve_LU.h
    src/sofa/defaulttype/Mat_solve_SVD.h
    src/sofa/defaulttype/RigidTypes.h
    src/sofa/defaulttype/RigidVec6Types.h
    src/sofa/defaulttype/SolidTypes.h
    src/sofa/defaulttype/SolidTypes.inl
    src/sofa/defaulttype/TemplatesAliases.h
    src/sofa/defaulttype/Vec.h
    src/sofa/defaulttype/Vec3Types.h
    src/sofa/defaulttype/VecTypes.h
    src/sofa/defaulttype/config.h
    src/sofa/defaulttype/Color.h
    src/sofa/defaulttype/StateMask.h
)

if(EIGEN3_FOUND OR Eigen3_FOUND)
    list(APPEND HEADER_FILES
         src/sofa/defaulttype/MapMapSparseMatrixEigenUtils.h
        )
endif()

set(SOURCE_FILES
    src/sofa/defaulttype/BaseMatrix.cpp
    src/sofa/defaulttype/BoundingBox.cpp
    src/sofa/defaulttype/Frame.cpp
    src/sofa/defaulttype/SolidTypes.cpp
    src/sofa/defaulttype/TemplatesAliases.cpp
    src/sofa/defaulttype/StateMask.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Messaging Sofa.Helper)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-D${PROJECT_CNAME}_BUILD_TARGET")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
