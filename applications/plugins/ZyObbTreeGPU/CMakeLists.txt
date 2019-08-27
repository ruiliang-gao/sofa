project(ZyObbTreeGPU)

add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity ${CMAKE_BINARY_DIR}/extlibs/ZyGProximity)

set(ZY_OBBTREEGPU_MAJOR_VERSION 0)
set(ZY_OBBTREEGPU_MINOR_VERSION 1)
set(ZY_OBBTREEGPU_VERSION ${ZY_OBBTREEGPU_MAJOR_VERSION}.${ZY_OBBTREEGPU_MINOR_VERSION})

add_definitions("-DSOFA_BUILD_OBBTREEGPU -DSOFA_BUILD_OBBTREEGPUPLUGIN")

find_package(CUDA REQUIRED)
find_package(cudpp)
find_package(Boost COMPONENTS system chrono thread regex REQUIRED)

add_definitions("-DSOFA_HAVE_PLUGIN_OBBTREEGPU")

set(HEADER_FILES
    config.h
    ObbTreeGPUCollisionModel.h
    ObbTreeGPUCollisionModel_cuda.h
    ObbTreeGPUCollisionModel.inl
    ObbTreeGPUIntersection.h
    ObbTreeGPUCollisionDetection_cuda.h
    ObbTreeGPUCollisionDetection_rss_cuda.h
    ObbTreeGPUCollisionDetection_cuda.inl
    ObbTreeGPUCollisionDetection_rss_cuda.inl
    ObbTreeGPUTriangleCollision_cuda.h
    ObbTreeGPU_CudaDataStructures.h
    BVHDrawHelpers.h
    ObbTreeGPUFrictionContact.h
    ObbTreeGPUFrictionContact.inl

    ObbTreeGPU_MultiThread_Tasks.h
    ObbTreeGPU_MultiThread_Tasks.inl
    ObbTreeGPU_MultiThread_CPU_Tasks.h
    ObbTreeGPUCollisionDetection_Threaded.h

    ObbTreeGPU_MultiThread_Scheduler.h
)

set(SOURCE_FILES

    initObbTreeGpuPlugin.cpp
    ObbTreeGPUCollisionModel.cpp
    ObbTreeGPUIntersection.cpp
    BVHDrawHelpers.cpp
    ObbTreeGPUFrictionContact.cpp

    ObbTreeGPU_MultiThread_Tasks.cpp
    ObbTreeGPU_MultiThread_CPU_Tasks.cpp
    ObbTreeGPUCollisionDetection_Threaded.cpp
    
    ObbTreeGPU_MultiThread_Scheduler.cpp
)

if (WIN32)
    list(APPEND HEADER_FILES obbtree_rulevisualisation.h)
    list(APPEND SOURCE_FILES obbtree_rulevisualisation.cpp)

    set(MOC_HEADER_FILES obbtree_rulevisualisation.h)

    set(UI_FILES
         obbtree_rulevisualisation.ui
    )
endif()


set(QT5_NO_LINK_QTMAIN 1)

find_package(Qt5 QUIET COMPONENTS Core Gui Widgets)
if(Qt5Core_FOUND)
    message(STATUS "${PROJECT_NAME}: Using Qt5")
    include_directories(${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
else()
    message(STATUS "${PROJECT_NAME}: Qt5 not found, using Qt4")

    set(QT4_NO_LINK_QTMAIN 1)
    #find_package(Qt4 QUIET COMPONENTS qtcore qtgui qtwidgets REQUIRED)
    find_package(Qt4 QUIET COMPONENTS qtcore qtgui REQUIRED)

    qt4_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")
    qt4_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")
endif()

set(CUDA_SOURCES
    ObbTreeGPUCollisionDetection.cu
    ObbTreeGPUCollisionModel.cu
    ObbTreeGPUTriangleCollision.cu
    ObbTreeGPU_CudaDataStructures.cu
)

if(Qt5Core_FOUND)
    qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")
    qt5_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")
else()
    qt4_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")
    qt4_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/framework
    ${CMAKE_SOURCE_DIR}/applications/plugins
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/SofaKernel/framework
    ${CMAKE_SOURCE_DIR}/extlibs/ZyWorkerThreads
    ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity
    ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/CyberX3D/include
    ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cutil/include
    ${CMAKE_SOURCE_DIR}/extlibs/ZyPQP/include
    ${CMAKE_SOURCE_DIR}/extlibs/ZyBVHModels
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

cuda_include_directories(
    ${CMAKE_SOURCE_DIR}/framework
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/extlibs/ZyWorkerThreads
    ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity
    ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cutil/include
    ${CMAKE_SOURCE_DIR}/extlibs/ZyPQP/include
    ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/CyberX3D/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIR} # without this, boost/chrono.hpp can't be found
    ${CMAKE_SOURCE_DIR}/SofaKernel/framework
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_61 -Xcompiler")

if(SOFA-EXTERNAL_GLEW)
    add_definitions("-DSOFA_HAVE_GLEW")
endif()

if(WIN32)
    add_definitions("-DSOFA_BUILD_CONSTRAINT")
    add_definitions("-DBUILD_TRU_MULTITHREADING")  # without this the compiler complains that "definition of dllimport function not allowed" for TaskSchedulerBoostPool.inl functions
endif(WIN32)

# ZyMultiThreading
include_directories(${CMAKE_SOURCE_DIR}/applications/plugins/ZyMultiThreading)

# ZyCollisionManager
include_directories(${CMAKE_SOURCE_DIR}/applications/plugins/ZyCollisionManager)
add_definitions(-DBUILD_ZY_COLLISIONMANAGER)

#if (CUDPP_FOUND)
#    message(STATUS "CudPP library found. Include directory: ${CUDPP_INCLUDE_DIR}")
#        include_directories(${CUDPP_INCLUDE_DIR})
#else (CUDPP_FOUND)
#        include_directories(${CMAKE_SOURCE_DIR}/trulibs/cudpp/include ${SOFA_EXTLIBS_DIR}/cudpp ${SOFA_EXTLIBS_DIR}/cudpp/src)
#endif (CUDPP_FOUND)

# Fine, always use the local version of CudPP
include_directories(${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp/include ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp ${CMAKE_SOURCE_DIR}/extlibs/ZyGProximity/cudpp/src)


link_directories(${Boost_LIBRARY_DIRS})
link_directories(${CMAKE_BINARY_DIR}/lib/Debug ${CMAKE_BINARY_DIR}/lib/RelWithDebInfo ${CMAKE_BINARY_DIR}/lib/Release)

# GCC versions > 5.x need this
if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
endif()

cuda_add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${MOC_FILES} ${HEADER_FILES} ${FORM_FILES} ${CUDA_SOURCES} ${README_FILES})

target_link_libraries(${PROJECT_NAME} gProximity CyberX3D ZyPQP SofaDefaultType SofaBaseCollision SofaCore SofaLoader SofaHelper SofaGeneralAnimationLoop SofaConstraint)

if (NOT WIN32)
    target_link_libraries(${PROJECT_NAME} gomp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif (NOT WIN32)

target_link_libraries(${PROJECT_NAME} cudpp ${Boost_THREAD_LIBRARY} ${Boost_REGEX_LIBRARY} ${QT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ZyWorkerThreads ZyBVHModels cutil)

target_link_libraries(${PROJECT_NAME} SofaOpenglVisual)

# ZyCollisionManager
target_link_libraries(${PROJECT_NAME} ZyCollisionManager)

if(Qt5Core_FOUND)
    target_link_libraries(${PROJECT_NAME} ${Qt5Core_LIBRARIES} ${Qt5Widgets_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME} Qt4::QtCore Qt4::QtGui)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${ZY_OBBTREEGPU_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})

