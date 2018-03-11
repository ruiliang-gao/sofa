cmake_minimum_required(VERSION 3.1)
project(Sofa.Helper.Types)

find_package(Sofa.Config)
find_package(Sofa.Messaging)

set(HEADER_FILES   
    src/sofa/helper/types/cast.h
    src/sofa/helper/types/config.h
    src/sofa/helper/types/deque.h
    src/sofa/helper/types/fixed_array.h
    src/sofa/helper/types/integer_id.h
    src/sofa/helper/types/list.h
    src/sofa/helper/types/map.h
    src/sofa/helper/types/map_ptr_stable_compare.h
    src/sofa/helper/types/Material.h
    src/sofa/helper/types/MemoryManager.h
    src/sofa/helper/types/OptionsGroup.h
    src/sofa/helper/types/pair.h
    src/sofa/helper/types/RGBAColor.h
    src/sofa/helper/types/set.h
    src/sofa/helper/types/stable_vector.h
    src/sofa/helper/types/typeinfos.h
    src/sofa/helper/types/vector.h
)

set(DEPRECATED_LAYOUT_FILES
    deprecated_layout/sofa/helper/fixed_array.h
    deprecated_layout/sofa/helper/map.h
    deprecated_layout/sofa/helper/map_ptr_stable_compare.h
    deprecated_layout/sofa/helper/pair.h
    deprecated_layout/sofa/helper/set.h
    deprecated_layout/sofa/helper/vector.h
)


set(SOURCE_FILES
    src/sofa/helper/types/fixed_array.cpp
    src/sofa/helper/types/Material.cpp
    src/sofa/helper/types/OptionsGroup.cpp
    src/sofa/helper/types/RGBAColor.cpp
    src/sofa/helper/types/typeinfos.cpp
    src/sofa/helper/types/vector.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES} ${DEPRECATED_LAYOUT_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Messaging)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_TARGET_SOFA_HELPER_TYPES")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deprecated_layout>")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT Find${PROJET_NAME})
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
