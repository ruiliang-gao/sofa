

### CMakeLists.txt are generated from our lovely sofa-pck-manager tools 
### In case you need to customize this CMakeLists.txt please copy our template from
### sofa-pck-packages/templates/CMakeLists.tpl and drop it into you current directory
### then launch sofa-pck-manager regenerate CMakeLists.txt
cmake_minimum_required(VERSION 3.1)
project(name)


set(HEADER_FILES
)

set(SOURCE_FILES
    Sofa/Helper/Types/src/sofa/helper/types/OptionsGroup.h
    Sofa/Helper/Types/src/sofa/helper/types/OptionsGroup.cpp
    Sofa/Helper/Types/src/sofa/helper/types/vector.h
    Sofa/Helper/Types/src/sofa/helper/types/vector.cpp
    Sofa/Helper/Types/src/sofa/helper/types/deque.h
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Messaging)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER_TYPES")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deprecated_layout>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")

