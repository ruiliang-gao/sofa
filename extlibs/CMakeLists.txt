cmake_minimum_required(VERSION 3.1)

set(EXTLIBS_TARGETS
    csparse
    newmat
    tinyxml
)

## Necessary ones
add_subdirectory(SuiteSparse)
add_subdirectory(csparse)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/eigen-3.2.7 eigen)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/newmat newmat)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/tinyxml tinyxml)
add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/json json)


## Optional ones
if (SOFAGUI_ENABLE_QWT)
    message("Qwt activated")
    add_subdirectory(qwt-6.1.2/src qwt)
endif()

option(SOFA_BUILD_MINIFLOWVR "Build the miniFlowVR library distributed in the extlibs/ directory." OFF)
if (SOFA_BUILD_MINIFLOWVR)
    add_subdirectory(miniFlowVR)
    list(APPEND EXTLIBS_TARGETS miniFlowVR)
endif()

option(SOFA_BUILD_ARTRACK
"Build the ARTrack library distributed in the extlibs/ directory.  This library
is used only by the ARTrack plugin."  OFF)
if(SOFA_BUILD_ARTRACK)
    add_subdirectory(ARTrack ARTrackLib)
    list(APPEND EXTLIBS_TARGETS ARTrackLib)
endif()

option(SOFA_BUILD_METIS
"Build the metis library distributed in the extlibs/ directory.  It is used only
by the SparseLDLSolver component.  Note that this library is usually available
on Linux distributions." OFF)
if(SOFA_BUILD_METIS)
    add_subdirectory(metis-5.1.0 "metis")
    list(APPEND EXTLIBS_TARGETS metis)
endif()

option(SOFA_BUILD_ZY_WORKER_THREADS
"Build the ZyWorkerThreads library." ON)
if (SOFA_BUILD_ZY_WORKER_THREADS)
    add_subdirectory(ZyWorkerThreads ${CMAKE_BINARY_DIR}/extlibs/ZyWorkerThreads)
    list(APPEND EXTLIBS_TARGETS ZyWorkerThreads)
endif()

option(SOFA_BUILD_ZY_BVH_MODELS
"Build the ZyBVHModels library." ON)
if (SOFA_BUILD_ZY_BVH_MODELS)
    add_subdirectory(ZyBVHModels ${CMAKE_BINARY_DIR}/extlibs/ZyBVHModels)
    list(APPEND EXTLIBS_TARGETS ZyBVHModels)
endif()

option(SOFA_BUILD_ZY_SP_MODELS
"Build the ZyBVHModels library." ON)
if (SOFA_BUILD_ZY_SP_MODELS)
    add_subdirectory(ZySPModels ${CMAKE_BINARY_DIR}/extlibs/ZySPModels)
    list(APPEND EXTLIBS_TARGETS ZySPModels)
endif()

foreach(target ${EXTLIBS_TARGETS})
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES FOLDER "SofaExtlibs")
endforeach()

option(SOFA_BUILD_PROTOSPHERE_EXTENDED "Build the ProtoSphere helper library for IST." OFF)
if (SOFA_BUILD_PROTOSPHERE_EXTENDED)
    if(EXISTS ${CMAKE_SOURCE_DIR}/extlibs/ProtoSphereExtended)
        if (SOFA_BUILD_PROTOSPHERE_EXTENDED)
            add_subdirectory(ProtoSphereExtended ${CMAKE_BINARY_DIR}/extlibs/ProtoSphereExtended)
        endif()
    endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/extlibs/CollDet)
    option(SOFA_BUILD_COLLDET "Build the CollDet library." ON)

    if (SOFA_BUILD_COLLDET)
        add_subdirectory(CollDet ${CMAKE_BINARY_DIR}/extlibs/CollDet)
    endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/extlibs/position-based-dynamics)
    if (EXISTS ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics)
        option(SOFA_BUILD_POSITION_BASED_DYNAMICS "Build the PositionBasedDynamics library." ON)

        if (SOFA_BUILD_POSITION_BASED_DYNAMICS)
            set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/CMake;${CMAKE_MODULE_PATH}")
            option(PBD_NO_DEMOS "Build the PositionBasedDynamics demos." OFF)
            # add_subdirectory(GenericParameters ${CMAKE_BINARY_DIR}/extlibs/GenericParameters)
            add_subdirectory(ext-discregrid)
            add_subdirectory(position-based-dynamics ${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics)
        endif()
    endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/extlibs/impulse-based-dynamics)
    if (EXISTS ${CMAKE_SOURCE_DIR}/extlibs/ImpulseBasedDynamics)
        option(SOFA_BUILD_IMPULSE_BASED_DYNAMICS "Build the ImpulseBasedDynamics library." ON)

        if (SOFA_BUILD_IMPULSE_BASED_DYNAMICS)
            add_subdirectory(impulse-based-dynamics/extern/gl3w ${CMAKE_BINARY_DIR}/extlibs/impulse-based-dynamics/extern/gl3w)
            add_subdirectory(ImpulseBasedDynamics ${CMAKE_BINARY_DIR}/extlibs/ImpulseBasedDynamics)
        endif()
    endif()
endif()
