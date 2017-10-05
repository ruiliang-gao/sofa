cmake_minimum_required(VERSION 3.1)
project(SofaSpecificUserInteraction)

set(SOFASPECIFICUSERINTERACTION_VERSION 1.0)


message(STATUS "SofaMisc:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    SofaSpecificUserInteraction.h
    initSofaSpecificUserInteraction.h
)

set(SOURCE_FILES
    initSofaSpecificUserInteraction.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        AddRecordedCameraPerformer.h
        FixParticlePerformer.h
        FixParticlePerformer.inl
        InciseAlongPathPerformer.h
        RemovePrimitivePerformer.h
        RemovePrimitivePerformer.inl
        StartNavigationPerformer.h
        SuturePointPerformer.h
        SuturePointPerformer.inl
        )
    list(APPEND SOURCE_FILES
        AddRecordedCameraPerformer.cpp
        FixParticlePerformer.cpp
        InciseAlongPathPerformer.cpp
        RemovePrimitivePerformer.cpp
        StartNavigationPerformer.cpp
        SuturePointPerformer.cpp
        )

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")

endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFA_SPECIFIC_USER_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
sofa_create_package(SofaSpecificUserInteraction ${SOFASPECIFICUSERINTERACTION_VERSION} ${PROJECT_NAME} SofaSpecificUserInteraction)
