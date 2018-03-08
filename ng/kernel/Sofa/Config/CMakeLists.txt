cmake_minimum_required(VERSION 3.5)


project(Sofa.Config VERSION 1.0 LANGUAGES CXX)


set(HEADER_FILES
    src/sofa/config/basemacros.h
)

set(SOURCE_FILES
    src/sofa/config/config.cpp
)

## Generate & install the different files that contains the defines associated with
## the state of the build option.
set(SOFA_BUILD_OPTIONS_SRC
        config/sharedlibrary_defines.h
        config/build_option_dump_visitor.h
        config/build_option_experimental_features.h
        config/build_option_opengl.h
        config/build_option_threading.h
        config/build_option_sreal_definition.h
    )

foreach(NAME ${SOFA_BUILD_OPTIONS_SRC})
    configure_file("${NAME}.in" "${CMAKE_BINARY_DIR}/include/sofa/${NAME}")
    install(FILES "${CMAKE_BINARY_DIR}/include/sofa/${NAME}" DESTINATION "include/sofa/config")
endforeach()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

configure_file("config.h.in" "${CMAKE_BINARY_DIR}/include/sofa/config.h")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config.h" DESTINATION "include/sofa/")
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR} EXPORT FindSofa.Config)
export(TARGETS ${PROJECT_NAME} FILE "${CMAKE_BINARY_DIR}/cmake/Find${PROJECT_NAME}.cmake")
