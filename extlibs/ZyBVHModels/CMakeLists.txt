project(ZyBVHModels)

set(libPrimitives_HEADER_FILES
    Primitives/AxisAlignedBox3.h  
    Primitives/AxisAlignedBox3.inl  
    Primitives/Box3.h  
    Primitives/Box3.inl  
    Primitives/Triangle2.h
    Primitives/Triangle2.inl
    Primitives/Triangle3.h
    Primitives/Triangle3.inl
    Primitives/Rectangle3.h
    Primitives/Rectangle3.inl
    Primitives/Segment2.h
    Primitives/Segment2.inl
    Primitives/Segment3.h
    Primitives/Segment3.inl
    Primitives/Line2.h
    Primitives/Line2.inl
    Primitives/Line3.h
    Primitives/Line3.inl
    Primitives/Plane3.h
    Primitives/Plane3.inl
    Primitives/Point3.h
    Primitives/Point3.inl

    Primitives/Utils_2D.h
    Primitives/Utils_2D.inl
)

set(libPrimitives_SOURCE_FILES
    Primitives/AxisAlignedBox3.cpp  
    Primitives/Box3.cpp  
    Primitives/Triangle3.cpp
    Primitives/Rectangle3.cpp
    Primitives/Segment3.cpp
    Primitives/Line2.cpp
    Primitives/Line3.cpp
    Primitives/Plane3.cpp
    Primitives/Triangle2.cpp
    Primitives/Segment2.cpp
    Primitives/Point3.cpp

    Primitives/Utils_2D.cpp
)

set(HEADER_FILES
    Intersectable.h
    DistanceComputable.h
    Math/MathUtils.h
    Math/MathUtils.inl
    Math/IntervalUtils.h
    Query/Query.h
    Query/Query2.h
    Query/Query2.inl
    ${libPrimitives_HEADER_FILES}
)

set(SOURCE_FILES
    Intersectable.cpp
    DistanceComputable.cpp
    Math/MathUtils.cpp
    Math/IntervalUtils.cpp
    Query/Query.cpp
    Query/Query2.cpp
    ${libPrimitives_SOURCE_FILES}
)

if(SOFA-EXTERNAL_GLEW)
    add_definitions("-DSOFA_HAVE_GLEW")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (WIN32)
        add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
else (WIN32)
        add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
endif (WIN32)

target_link_libraries(${PROJECT_NAME} SofaHelper)

# Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} "1.0.0" ${PROJECT_NAME} ${PROJECT_NAME})
