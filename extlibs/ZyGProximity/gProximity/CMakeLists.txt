set(gProximity_baseDir ${CMAKE_CURRENT_SOURCE_DIR})
project(gProximity)

set(CMAKE_VERBOSE_MAKEFILE On)

find_package(CUDA REQUIRED)

set(HEADER_FILES
           ${gProximity_baseDir}/bvh_kernels.h
           ${gProximity_baseDir}/cpu_bvh_constru.h
           ${gProximity_baseDir}/cuda_aabb.h
           ${gProximity_baseDir}/cuda_aabbnode.h
           ${gProximity_baseDir}/cuda_aabbsplit.h
           #${gProximity_baseDir}/cuda_bvh_constru.h
           ${gProximity_baseDir}/cuda_bvhexports.h
           ${gProximity_baseDir}/cuda_collision.h
           ${gProximity_baseDir}/cuda_debug_info.h
           ${gProximity_baseDir}/cuda_defs.h
           ${gProximity_baseDir}/cuda_intersect_nodes.h
           ${gProximity_baseDir}/cuda_intersect_tree.h
           ${gProximity_baseDir}/cuda_intersect_tritri.h
           ${gProximity_baseDir}/cuda_obb.h
           ${gProximity_baseDir}/cuda_prefix.h
           #${gProximity_baseDir}/cuda_rss_constru.h
           #${gProximity_baseDir}/cuda_rss.h
           ${gProximity_baseDir}/cuda_timer.h
           ${gProximity_baseDir}/cuda_vectors.h
           ${gProximity_baseDir}/cuda_vertex.h
           ${gProximity_baseDir}/cuda_workqueue.h
           ${gProximity_baseDir}/geometry.h
           ${gProximity_baseDir}/importer_utility.h
           ${gProximity_baseDir}/lbvh.h
           ${gProximity_baseDir}/mathdef.h
           ${gProximity_baseDir}/Matrix.h
           ${gProximity_baseDir}/plyfunctions.h
           ${gProximity_baseDir}/ply.h
           ${gProximity_baseDir}/rgb.h
           ${gProximity_baseDir}/scene.h
           ${gProximity_baseDir}/split_compaction.h
           ${gProximity_baseDir}/split_sort.h
           ${gProximity_baseDir}/transform.h
           ${gProximity_baseDir}/Triangle.h
           ${gProximity_baseDir}/Utility.h
           ${gProximity_baseDir}/Vector2.h
           ${gProximity_baseDir}/Vector3.h
)

set(SOURCE_FILES
           ${gProximity_baseDir}/cpu_bvh_constru.cpp
           ${gProximity_baseDir}/cuda_make_grid.cpp
           ${gProximity_baseDir}/cuda_timer.cpp
           ${gProximity_baseDir}/geometry.cpp
           ${gProximity_baseDir}/importer_utility.cpp
           ${gProximity_baseDir}/Matrix.cpp
           ${gProximity_baseDir}/plyfile.c
           ${gProximity_baseDir}/plyfunctions.cpp
           ${gProximity_baseDir}/transform.cpp
           ${gProximity_baseDir}/scene.cpp
)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/extlibs/ZyPQP/include
        ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cutil/include ### is in the wrong path
        ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/CyberX3D/include
)

cuda_include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/extlibs/ZyPQP/include
)

set(SOURCE_FILES_CUDPP
    ${gProximity_baseDir}/radixsort.h
    ${gProximity_baseDir}/radixsort.cpp
)

set(CUDA_SOURCE_FILES_CUDPP
	${gProximity_baseDir}/bvh_kernels.cu
	${gProximity_baseDir}/cuda_collision.cu
    ${gProximity_baseDir}/lbvh.cu
    ${gProximity_baseDir}/cuda_bvh_constru.cu
    ${gProximity_baseDir}/cuda_bvh_constru_state.cu
    ${gProximity_baseDir}/cuda_rss_constru.cu
    ${gProximity_baseDir}/cuda_rss_constru_state.cu
    ${gProximity_baseDir}/cudasort.cu
)

# if (CUDPP_FOUND)
#     include_directories(${CUDPP_INCLUDE_DIR})
# else (CUDPP_FOUND)
#     include_directories(${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp/include ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp/src)
# endif (CUDPP_FOUND)

# Fine, always use the local version of CudPP
include_directories(${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp/include ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp/src)

if (NOT WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -fPIC")
endif (NOT WIN32)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_61 -Xcompiler")

# GCC versions > 5.x need this
if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
endif()

if (WIN32)
    cuda_add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${SOURCE_FILES_CUDPP} ${CUDA_SOURCE_FILES_CUDPP} ${GPROXIMITY_README_FILES} OPTIONS ${VERBOSE_PTXAS})
else (WIN32)
    cuda_add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${SOURCE_FILES_CUDPP} ${CUDA_SOURCE_FILES_CUDPP} ${GPROXIMITY_README_FILES} OPTIONS ${VERBOSE_PTXAS})
endif (WIN32)

if (CUDPP_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CUDPP_LIBRARIES})
else (CUDPP_FOUND)
    target_link_libraries(${PROJECT_NAME} cudpp)
endif (CUDPP_FOUND)

target_link_libraries(${PROJECT_NAME} cutil CyberX3D ZyPQP tinyxml)

if (NOT WIN32)
  target_link_libraries(${PROJECT_NAME} GL GLU GLEW)
endif()

if(WIN32)
    if(SOFA-EXTERNAL_GLEW)
        add_definitions("-DSOFA_HAVE_GLEW")
    endif()
	include_directories(${CMAKE_SOURCE_DIR}/include/Win32)
	cuda_include_directories(${CMAKE_SOURCE_DIR}/include/Win32)
    #target_link_libraries(${PROJECT_NAME} glew32)
	link_directories(${CMAKE_SOURCE_DIR}/lib/Win32)
	link_directories(${CMAKE_SOURCE_DIR}/lib/Win64)
endif(WIN32)

# Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} "1.0.0" ${PROJECT_NAME} ${PROJECT_NAME})
