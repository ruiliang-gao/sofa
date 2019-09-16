project(ZySPModels)

set(HEADER_FILES
    SPModelBase.h
)

set(SOURCE_FILES
    SPModelBase.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaHelper)

# Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} "0.0.2" ${PROJECT_NAME} ${PROJECT_NAME})
