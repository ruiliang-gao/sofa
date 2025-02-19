cmake_minimum_required(VERSION 3.12)
project(SofaCUDA)

set(SOFACUDA_MAJOR_VERSION 0)
set(SOFACUDA_MINOR_VERSION 1)
set(SOFACUDA_VERSION ${SOFACUDA_MAJOR_VERSION}.${SOFACUDA_MINOR_VERSION})

# quick and dirty fix for nvcc compatibility with -fno-partial-inlining flag
if(NOT WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

set(HEADER_FILES
    config.h.in
    
    ### Common
    sofa/gpu/cuda/CudaBaseVector.h
    sofa/gpu/cuda/CudaCommon.h
    sofa/gpu/cuda/CudaMath.h
    sofa/gpu/cuda/CudaMath.inl
    sofa/gpu/cuda/CudaMathRigid.h
    sofa/gpu/cuda/CudaMathRigid.inl
    sofa/gpu/cuda/CudaMatrix.h
    sofa/gpu/cuda/CudaMemoryManager.h
    sofa/gpu/cuda/CudaScan.h
    sofa/gpu/cuda/CudaSort.h
    sofa/gpu/cuda/CudaTypes.h
    sofa/gpu/cuda/mycuda.h

    ### Mechanical
    sofa/gpu/cuda/CudaMechanicalObject.h
    sofa/gpu/cuda/CudaMechanicalObject.inl
    sofa/gpu/cuda/CudaParticleSource.h
    sofa/gpu/cuda/CudaParticleSource.inl

    ### Mappings
    sofa/gpu/cuda/CudaBarycentricMapping.h
    sofa/gpu/cuda/CudaBarycentricMapping.inl
    sofa/gpu/cuda/CudaBarycentricMappingRigid.h
    sofa/gpu/cuda/CudaIdentityMapping.h
    sofa/gpu/cuda/CudaIdentityMapping.inl
    sofa/gpu/cuda/CudaRigidMapping.h
    sofa/gpu/cuda/CudaRigidMapping.inl
    sofa/gpu/cuda/CudaSubsetMapping.h
    sofa/gpu/cuda/CudaSubsetMapping.inl


    ### Mass
    sofa/gpu/cuda/CudaDiagonalMass.h
    sofa/gpu/cuda/CudaDiagonalMass.inl
    sofa/gpu/cuda/CudaMeshMatrixMass.h
    sofa/gpu/cuda/CudaMeshMatrixMass.inl
    sofa/gpu/cuda/CudaUniformMass.h
    sofa/gpu/cuda/CudaUniformMass.inl
    

    ### FEM
    sofa/gpu/cuda/CudaHexahedronFEMForceField.h
    sofa/gpu/cuda/CudaHexahedronFEMForceField.inl
    sofa/gpu/cuda/CudaHexahedronTLEDForceField.h
    sofa/gpu/cuda/CudaStandardTetrahedralFEMForceField.h
    sofa/gpu/cuda/CudaStandardTetrahedralFEMForceField.inl
    sofa/gpu/cuda/CudaTetrahedralTensorMassForceField.h
    sofa/gpu/cuda/CudaTetrahedralTensorMassForceField.inl
    sofa/gpu/cuda/CudaTetrahedronFEMForceField.h
    sofa/gpu/cuda/CudaTetrahedronFEMForceField.inl
    #sofa/gpu/cuda/CudaTetrahedronTLEDForceField.h
    #sofa/gpu/cuda/CudaTetrahedronTLEDForceField.inl
    sofa/gpu/cuda/CudaTriangularFEMForceFieldOptim.h
    sofa/gpu/cuda/CudaTriangularFEMForceFieldOptim.inl

    ### ForceFields
    sofa/gpu/cuda/CudaEllipsoidForceField.h
    sofa/gpu/cuda/CudaEllipsoidForceField.inl
    sofa/gpu/cuda/CudaLinearForceField.h
    sofa/gpu/cuda/CudaLinearForceField.inl
    sofa/gpu/cuda/CudaPlaneForceField.h
    sofa/gpu/cuda/CudaPlaneForceField.inl
    sofa/gpu/cuda/CudaSphereForceField.h
    sofa/gpu/cuda/CudaSphereForceField.inl
    sofa/gpu/cuda/CudaSpringForceField.h
    sofa/gpu/cuda/CudaSpringForceField.inl

    

    ### Collisions
    sofa/gpu/cuda/CudaLineModel.h
    sofa/gpu/cuda/CudaPointModel.h
    sofa/gpu/cuda/CudaSphereModel.h
    sofa/gpu/cuda/CudaTriangleModel.h

    ### Constraints
    sofa/gpu/cuda/CudaFixedConstraint.h
    sofa/gpu/cuda/CudaFixedConstraint.inl
    sofa/gpu/cuda/CudaLinearMovementConstraint.h
    sofa/gpu/cuda/CudaLinearMovementConstraint.inl
    sofa/gpu/cuda/CudaPenalityContactForceField.h
    sofa/gpu/cuda/CudaPenalityContactForceField.inl
  
)

set(SOURCE_FILES
    ### Common
    main.cpp
    sofa/gpu/cuda/CudaBaseVector.cpp
    sofa/gpu/cuda/CudaTypes.cpp
    sofa/gpu/cuda/mycuda.cpp

    ### Mechanical
    sofa/gpu/cuda/CudaMechanicalObject.cpp
    sofa/gpu/cuda/CudaSetTopology.cpp

    ### Mappings
    sofa/gpu/cuda/CudaBarycentricMapping-3f.cpp
    sofa/gpu/cuda/CudaBarycentricMapping-3f1-3f.cpp
    sofa/gpu/cuda/CudaBarycentricMapping-3f1-d.cpp
    sofa/gpu/cuda/CudaBarycentricMapping-3f1-f.cpp
    sofa/gpu/cuda/CudaBarycentricMapping-3f1.cpp
    sofa/gpu/cuda/CudaBarycentricMapping-f.cpp
    sofa/gpu/cuda/CudaBarycentricMapping.cpp
    sofa/gpu/cuda/CudaBarycentricMappingRigid.cpp
    sofa/gpu/cuda/CudaIdentityMapping.cpp
    sofa/gpu/cuda/CudaMultiMapping.cpp
    sofa/gpu/cuda/CudaRigidMapping.cpp
    sofa/gpu/cuda/CudaSubsetMapping.cpp
    sofa/gpu/cuda/CudaSubsetMultiMapping.cpp

    ### Mass
    sofa/gpu/cuda/CudaDiagonalMass.cpp
    sofa/gpu/cuda/CudaMeshMatrixMass.cpp
    sofa/gpu/cuda/CudaUniformMass.cpp

    ### FEM
    sofa/gpu/cuda/CudaHexahedronFEMForceField.cpp
    sofa/gpu/cuda/CudaHexahedronTLEDForceField.cpp
    sofa/gpu/cuda/CudaStandardTetrahedralFEMForceField.cpp
    sofa/gpu/cuda/CudaTetrahedralTensorMassForceField.cpp
    sofa/gpu/cuda/CudaTetrahedronFEMForceField.cpp
    #sofa/gpu/cuda/CudaTetrahedronTLEDForceField.cpp
    sofa/gpu/cuda/CudaTriangularFEMForceFieldOptim.cpp

    ### ForceFields
    sofa/gpu/cuda/CudaConstantForceField.cpp
    sofa/gpu/cuda/CudaEllipsoidForceField.cpp
    sofa/gpu/cuda/CudaLinearForceField.cpp
    sofa/gpu/cuda/CudaPenalityContactForceField.cpp
    sofa/gpu/cuda/CudaPlaneForceField.cpp
    sofa/gpu/cuda/CudaRestShapeSpringsForceField.cpp
    sofa/gpu/cuda/CudaSphereForceField.cpp
    sofa/gpu/cuda/CudaSpringForceField.cpp
    

    ### Collisions
    sofa/gpu/cuda/CudaCollision.cpp
    sofa/gpu/cuda/CudaLineModel.cpp
    sofa/gpu/cuda/CudaPointModel.cpp
    sofa/gpu/cuda/CudaSphereModel.cpp
    sofa/gpu/cuda/CudaTriangleModel.cpp

    ### Constraints
    sofa/gpu/cuda/CudaFixedConstraint.cpp
    sofa/gpu/cuda/CudaFixedTranslationConstraint.cpp
    sofa/gpu/cuda/CudaLinearMovementConstraint.cpp
    sofa/gpu/cuda/CudaLinearVelocityConstraint.cpp

    sofa/gpu/cuda/CudaBeamLinearMapping.cpp
    sofa/gpu/cuda/CudaBoxROI.cpp
    sofa/gpu/cuda/CudaNearestPointROI.cpp
    sofa/gpu/cuda/CudaSphereROI.cpp

    sofa/gpu/cuda/CudaExtraMonitor.cpp
    sofa/gpu/cuda/CudaIndexValueMapper.cpp
)

set(CUDA_SOURCES
    ### Common
    sofa/gpu/cuda/mycuda.cu
    sofa/gpu/cuda/CudaBaseVector.cu
    sofa/gpu/cuda/CudaScan.cu
    sofa/gpu/cuda/CudaSort.cu

    ### Mechanical
    sofa/gpu/cuda/CudaMechanicalObject.cu

    ### Mappings
    sofa/gpu/cuda/CudaBarycentricMapping.cu
    sofa/gpu/cuda/CudaRigidMapping.cu
    sofa/gpu/cuda/CudaSubsetMapping.cu

    ### Mass
    sofa/gpu/cuda/CudaDiagonalMass.cu
    sofa/gpu/cuda/CudaMeshMatrixMass.cu
    sofa/gpu/cuda/CudaUniformMass.cu

    ### FEM
    sofa/gpu/cuda/CudaHexahedronFEMForceField.cu
    sofa/gpu/cuda/CudaHexahedronTLEDForceField.cu
    sofa/gpu/cuda/CudaStandardTetrahedralFEMForceField.cu
    sofa/gpu/cuda/CudaTetrahedralTensorMassForceField.cu
    sofa/gpu/cuda/CudaTetrahedronFEMForceField.cu
    sofa/gpu/cuda/CudaTetrahedronTLEDForceField.cu
    sofa/gpu/cuda/CudaTriangularFEMForceFieldOptim.cu

    ### ForceFields
    sofa/gpu/cuda/CudaEllipsoidForceField.cu
    sofa/gpu/cuda/CudaLinearForceField.cu
    sofa/gpu/cuda/CudaPenalityContactForceField.cu
    sofa/gpu/cuda/CudaPlaneForceField.cu
    sofa/gpu/cuda/CudaSphereForceField.cu
    sofa/gpu/cuda/CudaSpringForceField.cu

    ### Collisions
    
    ### Constraints
    sofa/gpu/cuda/CudaFixedConstraint.cu
    sofa/gpu/cuda/CudaLinearMovementConstraint.cu
    
    
)

sofa_find_package(Sofa.GL QUIET)
if(NOT Sofa.GL_FOUND)
    message(WARNING "Sofa.GL is not active.")
    message("OpenGL-related code wont be enabled (Sharing OpenGL buffers with CUDA)")
else()
    list(APPEND HEADER_FILES
        sofa/gpu/cuda/CudaVisualModel.h
        sofa/gpu/cuda/CudaVisualModel.inl
    )
    list(APPEND SOURCE_FILES
        sofa/gpu/cuda/CudaVisualModel.cpp
    )
    list(APPEND CUDA_SOURCES
        sofa/gpu/cuda/CudaVisualModel.cu
    )
endif()

set(README_FILES README.md)

find_package(CUDA REQUIRED)

find_package(SofaGeneralEngine REQUIRED)
find_package(SofaGeneralDeformable REQUIRED)
find_package(SofaEngine REQUIRED)
find_package(SofaUserInteraction REQUIRED)
find_package(SofaNonUniformFem REQUIRED)
find_package(SofaMiscMapping REQUIRED)
find_package(SofaMiscFem REQUIRED)
find_package(SofaValidation REQUIRED)
find_package(SofaMiscForceField REQUIRED)
find_package(SofaGeneralObjectInteraction REQUIRED)

find_package(VolumetricRendering QUIET)

if(VolumetricRendering_FOUND)
    list(APPEND HEADER_FILES sofa/gpu/cuda/CudaTetrahedralVisualModel.h)
    list(APPEND HEADER_FILES sofa/gpu/cuda/CudaTetrahedralVisualModel.inl)
    list(APPEND SOURCE_FILES sofa/gpu/cuda/CudaTetrahedralVisualModel.cpp)
    add_definitions("-DSOFACUDA_ENABLE_VOLUMETRICRENDERING")
else()
    message(STATUS "SofaCUDA: Plugin VolumetricRendering was not enabled/found, therefore CudaTetrahedralVisualModel will not be compiled.")
endif()

sofa_find_package(SofaGuiQt QUIET)
if(SofaGuiQt_FOUND)
    list(APPEND HEADER_FILES sofa/gpu/gui/CudaDataWidget.h)
    list(APPEND SOURCE_FILES sofa/gpu/gui/CudaDataWidget.cpp)
endif()

find_package(SofaDistanceGrid QUIET)
sofa_find_package(MiniFlowVR QUIET)

if(SofaDistanceGrid_FOUND)
    if(MiniFlowVR_FOUND)
        message(STATUS "SofaCUDA: MiniFlowVR enabled.")
    else()
        message(STATUS "SofaCUDA: MiniFlowVR was not enabled, therefore some feature of CudaDistanceGridCollisionModel will not be compiled.")
    endif()

    list(APPEND HEADER_FILES
        sofa/gpu/cuda/CudaDistanceGridCollisionModel.h
        sofa/gpu/cuda/CudaContactMapper.h
        sofa/gpu/cuda/CudaCollisionDetection.h)

    list(APPEND SOURCE_FILES
        sofa/gpu/cuda/CudaDistanceGridCollisionModel.cpp
        sofa/gpu/cuda/CudaCollisionDistanceGrid.cpp
        sofa/gpu/cuda/CudaCollisionDetection.cpp)

    list(APPEND CUDA_SOURCES
        sofa/gpu/cuda/CudaContactMapper.cu
        sofa/gpu/cuda/CudaCollisionDetection.cu)
else()
    message(STATUS "SofaCUDA: SofaDistanceGrid was not enabled, therefore CudaDistanceGridCollisionModel will not be compiled.")
endif()

if (PLUGIN_SOFASPHFLUID)
    find_package(SofaSphFluid QUIET)
    if(SofaSphFluid_FOUND)
        list(APPEND HEADER_FILES
            sofa/gpu/cuda/CudaParticleSource.h
            sofa/gpu/cuda/CudaParticleSource.inl
            sofa/gpu/cuda/CudaSPHFluidForceField.h
            sofa/gpu/cuda/CudaSPHFluidForceField.inl
            sofa/gpu/cuda/CudaParticlesRepulsionForceField.h
            sofa/gpu/cuda/CudaParticlesRepulsionForceField.inl
            sofa/gpu/cuda/CudaSpatialGridContainer.h
            sofa/gpu/cuda/CudaSpatialGridContainer.inl
            )
        list(APPEND SOURCE_FILES
        sofa/gpu/cuda/CudaParticleSource.cpp
        sofa/gpu/cuda/CudaSPHFluidForceField.cpp
        sofa/gpu/cuda/CudaParticlesRepulsionForceField.cpp
        sofa/gpu/cuda/CudaSpatialGridContainer.cpp
        )
        list(APPEND CUDA_SOURCES
        sofa/gpu/cuda/CudaParticleSource.cu
        sofa/gpu/cuda/CudaSPHFluidForceField.cu
        sofa/gpu/cuda/CudaParticlesRepulsionForceField.cu
        sofa/gpu/cuda/CudaSpatialGridContainer.cu
        )
        message(STATUS "SofaCUDA: optional dependency to SofaSphFluid found. ")
    else()
        message(STATUS "SofaCUDA: optional dependency SofaSphFluid not found. ")
    endif()
else()
    message(STATUS "SofaCUDA: optional dependency SofaSphFluid not enabled. ")
endif()


option(SOFACUDA_VERBOSE_PTXAS "???" OFF)
if(SOFACUDA_VERBOSE_PTXAS)
    set(VERBOSE_PTXAS --ptxas-options=-v)
endif()

option(SOFACUDA_CUBLAS "Activate cublas support in CUDA (requires SOFACUDA_DOUBLE)." OFF)
if(SOFACUDA_CUBLAS)
    set(SOFA_GPU_CUBLAS 1)       # #cmakedefine
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(CUDASparse REQUIRED)
endif()

option(SOFACUDA_CUDPP "Activate CUDPP (for RadixSort)." OFF)
if(SOFACUDA_CUDPP)
    set(SOFA_GPU_CUDPP 1)       # #cmakedefine
endif()

# Note: THRUST is included in CUDA SDK 4.0+, it is recommended to use it if available
option(SOFACUDA_THRUST "Activate THRUST (for RadixSort)." ON)
if(SOFACUDA_THRUST)
    set(SOFA_GPU_THRUST 1)       # #cmakedefine
endif()

option(SOFACUDA_DOUBLE "Activate double-precision support in CUDA (requires GT200+ GPU and -arch sm_13 flag." OFF)
if(SOFACUDA_DOUBLE)
    set(SOFA_GPU_CUDA_DOUBLE 1)       # #cmakedefine
endif()


option(SOFACUDA_DOUBLE_PRECISE "Enable double-precision for sqrt/div... (requires compute capability
>= 2 and CUDA_VERSION > 3.0)" OFF)
# Note: with SOFA_GPU_CUDA_PRECISE and SOFA_GPU_CUDA_DOUBLE you get IEEE
# 754-compliant floating point operations for addition and multiplication only.
if(SOFACUDA_DOUBLE_PRECISE)
    set(SOFA_GPU_CUDA_DOUBLE_PRECISE 1)       # #cmakedefine
endif()

option(SOFACUDA_PRECISE "Use IEEE 754-compliant floating point operations." OFF)

include(SofaCUDANvccFlags.cmake)

# nvcc uses a "host code compiler" to compile CPU code, specified by CUDA_HOST_COMPILER.
# With some versions of CMake, CUDA_HOST_COMPILER defaults to CMAKE_C_COMPILER,
# but few host compilers are actually supported. Workarounds should go here.
if (${CUDA_HOST_COMPILER} MATCHES "ccache$")
    message(STATUS "SofaCUDA: CUDA host compiler was set to ccache, changing to g++")
    set(CUDA_HOST_COMPILER "g++" CACHE STRING "Host side compiler used by NVCC" FORCE)
endif()

# quick and dirty fix for nvcc compatibility with -fno-partial-inlining flag
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(WIN32)
    add_definitions("-DWIN32")
    set(WINDOWSOPTIONS "/MD")
endif()

# Make the compiler definitions available to nvcc
add_definitions("-DSOFA_BUILD_SOFACUDA")
# Give include directories to nvcc compiler.
# Note: keep cuda_include_directories() before cuda_add_library().
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
cuda_include_directories("${CMAKE_BINARY_DIR}/include")
cuda_add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES} ${README_FILES} OPTIONS ${VERBOSE_PTXAS} ${WINDOWSOPTIONS})


target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACUDA_COMPILE_DEFINITIONS}")
target_link_libraries(${PROJECT_NAME}
    SofaGeneralEngine
    SofaGeneralDeformable
    SofaEngine
    SofaUserInteraction
    SofaNonUniformFem
    SofaMiscMapping
    SofaMiscFem
    SofaValidation
    SofaMiscForceField
    SofaGeneralObjectInteraction
    )

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(SofaOpenglVisual_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaOpenglVisual)
endif()

if(SofaDistanceGrid_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaDistanceGrid)
endif()

if(MiniFlowVR_FOUND)
    target_link_libraries(${PROJECT_NAME} miniFlowVR)
endif()

if(SofaGuiQt_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaGuiQt)
endif()


if(SOFACUDA_CUBLAS)
    cuda_add_cublas_to_target(${PROJECT_NAME})
    target_link_libraries(${PROJECT_NAME} ${CUDA_SPARSE_LIBRARY})
endif()
if(SOFACUDA_CUDPP)
    target_link_libraries(${PROJECT_NAME} cudpp)
endif()
if(VolumetricRendering_FOUND)
    target_link_libraries(${PROJECT_NAME} VolumetricRendering)
endif()
if(SofaSphFluid_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaSphFluid)
endif()


## SofaCUDANvccFlags.cmake
# Build tree
configure_file(SofaCUDANvccFlags.cmake ${CMAKE_BINARY_DIR}/lib/cmake/SofaCUDANvccFlags.cmake COPYONLY)
# Install tree
install(FILES SofaCUDANvccFlags.cmake DESTINATION lib/cmake/SofaCUDA)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFACUDA_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
