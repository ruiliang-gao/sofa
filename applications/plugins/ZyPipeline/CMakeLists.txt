cmake_minimum_required(VERSION 2.8.12)
project(ZyPipeline)

set(ZY_PIPELINE_MAJOR_VERSION 0)
set(ZY_PIPELINE_MINOR_VERSION 2)
set(ZY_PIPELINE_VERSION ${ZY_PIPELINE_MAJOR_VERSION}.${ZY_PIPELINE_MINOR_VERSION})


find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system chrono thread REQUIRED)

set(HEADER_FILES initZyPipeline.h
                 ZyPipelineInterface.h
                 ZyBasePipeline.h
		 ZyDefaultPipeline.h
                 ZyParallelNarrowPhase.h
                 ZyMultiThreadedCollisionPipeline.h
                 ZySingleThreadedCollisionPipeline.h
		 )
				 
set(SOURCE_FILES initZyPipeline.cpp
                 ZyPipelineInterface.cpp
                 ZyBasePipeline.cpp
                 ZyDefaultPipeline.cpp
                 ZyParallelNarrowPhase.cpp
                 ZySingleThreadedCollisionPipeline.cpp
                 ZyMultiThreadedCollisionPipeline.cpp
		 )
				 
set(TASK_HEADER_FILES Zy_MultiThread_Tasks_BVHTraversal.h
	              Zy_MultiThread_Tasks_TriangleTests.h
                      config_zypipeline_tasks.h)
set(TASK_SOURCE_FILES Zy_MultiThread_Tasks_BVHTraversal.cpp
		      Zy_MultiThread_Tasks_TriangleTests.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/extlibs/ZyWorkerThreads)

add_definitions("-DBUILD_ZY_PIPELINE")
# Needed for template classes from ZyWorkerThreads
add_definitions("-DBUILD_ZY_MULTITHREADING")

find_package(Boost COMPONENTS system chrono thread REQUIRED QUIET)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

add_definitions("-DBUILD_ZY_PIPELINE_TASKS")
add_library(${PROJECT_NAME}_Tasks SHARED ${TASK_HEADER_FILES} ${TASK_SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}_Tasks SofaBaseCollision SofaMiscCollision ZyWorkerThreads)
target_link_libraries(${PROJECT_NAME}_Tasks ${Boost_DATE_TIME_LIBRARY})
set_target_properties(${PROJECT_NAME}_Tasks PROPERTIES VERSION ${ZY_PIPELINE_VERSION})
set_target_properties(${PROJECT_NAME}_Tasks PROPERTIES PUBLIC_HEADER "${TASK_HEADER_FILES}")

target_link_libraries(${PROJECT_NAME} ZyWorkerThreads ZyPipeline_Tasks)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${ZY_PIPELINE_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

set_target_properties(${PROJECT_NAME}_Tasks PROPERTIES FOLDER "TruLibs")
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "TruLibs")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${ZY_PIPELINE_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
sofa_create_package(${PROJECT_NAME}_Tasks ${ZY_PIPELINE_VERSION} ${PROJECT_NAME}_Tasks ${PROJECT_NAME}_Tasks)
