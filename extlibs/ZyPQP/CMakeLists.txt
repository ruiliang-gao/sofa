set(PQP_baseDir ${CMAKE_SOURCE_DIR}/extlibs/ZyPQP)
project(ZyPQP)

find_package(Boost QUIET REQUIRED)

set(HEADER_FILES ${PQP_baseDir}/include/BV.h
               ${PQP_baseDir}/include/PQP_Compile.h
               ${PQP_baseDir}/include/PQP.h
               ${PQP_baseDir}/include/PQP_Internal.h
               ${PQP_baseDir}/include/Tri.h
               ${PQP_baseDir}/src/Build.h
               ${PQP_baseDir}/src/BV.h
               ${PQP_baseDir}/src/BVTQ.h
               ${PQP_baseDir}/src/GetTime.h
               ${PQP_baseDir}/src/MatVec.h
               ${PQP_baseDir}/src/OBB_Disjoint.h
               ${PQP_baseDir}/src/PQP_Compile.h
               ${PQP_baseDir}/src/PQP_Internal.h
               ${PQP_baseDir}/src/PQP.h
               ${PQP_baseDir}/src/RectDist.h
               ${PQP_baseDir}/src/TriDist.h
               ${PQP_baseDir}/src/Tri.h)

set(SOURCE_FILES ${PQP_baseDir}/src/Build.cpp
                ${PQP_baseDir}/src/BV.cpp
                ${PQP_baseDir}/src/PQP.cpp
                ${PQP_baseDir}/src/TriDist.cpp)

include_directories(${CMAKE_SOURCE_DIR}/framework ${CMAKE_BINARY_DIR}/include)
include_directories(${Boost_INCLUDE_DIR})

if (WIN32)
    add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
else (WIN32)
    add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
endif (WIN32)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "TruExtlibs")


# Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} "0.2" ${PROJECT_NAME} ${PROJECT_NAME})
