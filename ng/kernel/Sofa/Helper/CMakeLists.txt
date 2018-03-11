cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper)

find_package(nlohmann_json)
find_package(Eigen)
find_package(Sofa.Config)
find_package(Sofa.Messaging)
find_package(Sofa.Helper.Types)

set(HEADER_FILES
    src/sofa/helper/ArgumentParser.h
    src/sofa/helper/AdvancedTimer.h
    src/sofa/helper/BackTrace.h
    src/sofa/helper/cast.h
    src/sofa/helper/DualQuat.h                  # Compatibility layer
    src/sofa/helper/Factory.h
    src/sofa/helper/Factory.inl
    src/sofa/helper/FnDispatcher.h
    src/sofa/helper/FnDispatcher.inl
    src/sofa/helper/IndexOpenMP.h
    src/sofa/helper/LCPSolver.h
    src/sofa/helper/LCPSolver.inl
    src/sofa/helper/LCPcalc.h
    src/sofa/helper/MatEigen.h
    src/sofa/helper/OwnershipSPtr.h
    src/sofa/helper/StateMask.h
    #src/sofa/helper/Polynomial_LD.h            # Compatibility layer
    #src/sofa/helper/Quater.h                   # Compatibility layer
    src/sofa/helper/RandomGenerator.h
    #src/sofa/helper/SVector.h
    src/sofa/helper/SimpleTimer.h
    src/sofa/helper/SortedPermutation.h
    src/sofa/helper/StringUtils.h
    src/sofa/helper/TagFactory.h
    src/sofa/helper/Utils.h
    src/sofa/helper/config.h
    src/sofa/helper/decompose.h
    src/sofa/helper/deprecatedcomponents.h
    #src/sofa/helper/hash.h
    src/sofa/helper/helper.h
    src/sofa/helper/init.h
    #src/sofa/helper/integer_id.h
    src/sofa/helper/io/BaseFileAccess.h
    src/sofa/helper/io/FileAccess.h
    src/sofa/helper/io/File.h
    src/sofa/helper/json.h
    #src/sofa/helper/list.h
    #src/sofa/helper/map.h
    #src/sofa/helper/map_ptr_stable_compare.h
    src/sofa/helper/polygon_cube_intersection/polygon_cube_intersection.h
    src/sofa/helper/polygon_cube_intersection/vec.h
    #src/sofa/helper/stable_vector.h
    src/sofa/helper/system/DynamicLibrary.h
    src/sofa/helper/system/FileSystem.h
    src/sofa/helper/system/Locale.h
    src/sofa/helper/system/PipeProcess.h
    src/sofa/helper/system/PluginManager.h
    src/sofa/helper/system/SetDirectory.h
    src/sofa/helper/system/SofaOStream.h
    src/sofa/helper/system/atomic.h
    src/sofa/helper/system/config.h
    src/sofa/helper/system/console.h
    src/sofa/helper/system/gl.h
    src/sofa/helper/system/glu.h
    src/sofa/helper/system/thread/CTime.h
    src/sofa/helper/system/thread/CircularQueue.h
    src/sofa/helper/system/thread/CircularQueue.inl
    src/sofa/helper/system/thread/debug.h
    src/sofa/helper/system/thread/thread_specific_ptr.h
    src/sofa/helper/system/FileMonitor.h
    src/sofa/helper/system/FileRepository.h
    #src/sofa/helper/vectorData.h
    #src/sofa/helper/vectorLinks.h
    #src/sofa/helper/vector_algebra.h
    src/sofa/helper/vector_device.h
    src/sofa/helper/logging/Messaging.h
    src/sofa/helper/logging/Message.h
    src/sofa/helper/logging/ComponentInfo.h
    src/sofa/helper/logging/FileInfo.h
    src/sofa/helper/logging/MessageDispatcher.h
    src/sofa/helper/logging/MessageHandler.h
    src/sofa/helper/logging/ConsoleMessageHandler.h
    src/sofa/helper/logging/SilentMessageHandler.h
    src/sofa/helper/logging/FileMessageHandler.h
    src/sofa/helper/logging/CountingMessageHandler.h
    src/sofa/helper/logging/LoggingMessageHandler.h
    src/sofa/helper/logging/RoutingMessageHandler.h
    src/sofa/helper/logging/MessageHandler.h
    src/sofa/helper/logging/MessageFormatter.h
    src/sofa/helper/logging/ClangMessageHandler.h
    src/sofa/helper/logging/ClangStyleMessageFormatter.h
    src/sofa/helper/logging/DefaultStyleMessageFormatter.h
    src/sofa/helper/logging/ExceptionMessageHandler.h
    src/sofa/helper/system/thread/TimeoutWatchdog.h
    src/sofa/helper/system/FileRepository.h
    #src/sofa/helper/types/Material.h
)

set(SOURCE_FILES
    src/sofa/helper/ArgumentParser.cpp
    src/sofa/helper/AdvancedTimer.cpp
    src/sofa/helper/BackTrace.cpp
    src/sofa/helper/Factory.cpp
    src/sofa/helper/LCPcalc.cpp
    #src/sofa/helper/SVector.cpp
    src/sofa/helper/RandomGenerator.cpp
    src/sofa/helper/TagFactory.cpp
    src/sofa/helper/Utils.cpp
    src/sofa/helper/deprecatedcomponents.cpp
    src/sofa/helper/init.cpp
    src/sofa/helper/io/BaseFileAccess.cpp
    src/sofa/helper/io/FileAccess.cpp
    src/sofa/helper/io/File.cpp
    #src/sofa/helper/types/Material.cpp
    src/sofa/helper/polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    src/sofa/helper/polygon_cube_intersection/polygon_cube_intersection.cpp
    src/sofa/helper/system/DynamicLibrary.cpp
    src/sofa/helper/system/FileSystem.cpp
    src/sofa/helper/system/Locale.cpp
    src/sofa/helper/system/PipeProcess.cpp
    src/sofa/helper/system/PluginManager.cpp
    src/sofa/helper/system/SetDirectory.cpp
    src/sofa/helper/system/SofaOStream.cpp
    src/sofa/helper/system/console.cpp
    src/sofa/helper/system/thread/CTime.cpp
    src/sofa/helper/system/thread/CircularQueue.cpp
    src/sofa/helper/system/thread/debug.cpp
    src/sofa/helper/system/FileRepository.cpp
    src/sofa/helper/system/thread/TimeoutWatchdog.cpp
    src/sofa/helper/system/FileRepository.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    list(APPEND SOURCE_FILES  src/sofa/helper/system/FileMonitor_linux.cpp)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    list(APPEND SOURCE_FILES  src/sofa/helper/system/FileMonitor_macosx.cpp)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    list(APPEND SOURCE_FILES  src/sofa/helper/system/FileMonitor_windows.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Messaging Sofa.Helper.Types nlohmann_json Eigen)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
