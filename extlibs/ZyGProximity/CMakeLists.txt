list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_subdirectory(zycutil)

find_package(CudPP REQUIRED)
if (NOT CUDPP_FOUND)
    add_subdirectory(zycudpp)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cudpp/include")
endif()

add_subdirectory(CyberX3D)
add_subdirectory(gProximity)
