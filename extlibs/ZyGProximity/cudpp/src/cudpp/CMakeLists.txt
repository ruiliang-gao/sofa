##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/app
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/cta
  )

set(CCFILES
  cudpp.cpp
  cudpp_plan.cpp
  cudpp_manager.cpp
  )

set (HFILES
  cudpp_manager.h
  cudpp_plan.h
  cuda_util.h
  cudpp_util.h
  cudpp_globals.h
  cudpp_compact.h
  cudpp_compress.h
  cudpp_listrank.h
  cudpp_mergesort.h
  cudpp_radixsort.h
  cudpp_rand.h
  cudpp_reduce.h
  cudpp_stringsort.h
  cudpp_scan.h
  cudpp_segscan.h
  cudpp_spmvmult.h
  sharedmem.h
  )

set (CUHFILES
  cta/compress_cta.cuh
  cta/mergesort_cta.cuh
  cta/radixsort_cta.cuh  
  cta/rand_cta.cuh
  cta/scan_cta.cuh
  cta/segmented_scan_cta.cuh
  cta/stringsort_cta.cuh  
  kernel/compact_kernel.cuh
  kernel/compress_kernel.cuh
  kernel/listrank_kernel.cuh
  kernel/mergesort_kernel.cuh
  kernel/radixsort_kernel.cuh
  kernel/rand_kernel.cuh
  kernel/reduce_kernel.cuh
  kernel/segmented_scan_kernel.cuh
  kernel/spmvmult_kernel.cuh
  kernel/stringsort_kernel.cuh
  kernel/vector_kernel.cuh
  kernel/tridiagonal_kernel.cuh
  )

set(CUFILES
  app/reduce_app.cu
  app/compact_app.cu
  app/compress_app.cu
  app/listrank_app.cu
  app/mergesort_app.cu
  app/scan_app.cu
  app/segmented_scan_app.cu
  app/spmvmult_app.cu
  app/stringsort_app.cu
  app/radixsort_app.cu
  app/rand_app.cu 
  app/tridiagonal_app.cu
  )

set(HFILES_PUBLIC
  ../../include/cudpp.h
  ../../include/cudpp_config.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_61 -gencode=arch=compute_35,code=compute_61)

option(CUDPP_GENCODE_SM61
       "ON to generate code for Compute Capability 6.1"
       ON)

if (CUDPP_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(CUDPP_GENCODE_SM61)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

# Todo: Cuda architecture flag
set(TEMP_CUDA_FLAGS "-arch sm_61 -Xcompiler")
if(NOT (${CUDA_NVCC_FLAGS} MATCHES ${TEMP_CUDA_FLAGS}))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${TEMP_CUDA_FLAGS}")
        if(NOT WIN32)
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -fPIC")
        endif()
endif()
set(CUDA_NVCC_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG")
unset(TEMP_CUDA_FLAGS)

cuda_add_library(${PROJECT_NAME} ${LIB_TYPE} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES} 
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
  )
  
sofa_create_package(${PROJECT_NAME} "${cudpp_VERSION_MAJOR}.${cudpp_VERSION_MINOR}.${cudpp_VERSION_PATCH}" ${PROJECT_NAME} ${PROJECT_NAME})
