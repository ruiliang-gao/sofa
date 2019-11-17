set(DISCREGRID_LIBRARY_MAJOR_VERSION 0)
set(DISCREGRID_LIBRARY_MINOR_VERSION 1)
set(DISCREGRID_LIBRARY_VERSION ${DISCREGRID_LIBRARY_MAJOR_VERSION}.${DISCREGRID_LIBRARY_MINOR_VERSION})

set(HEADERS
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/discrete_grid.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/cubic_lagrange_discrete_grid.hpp
)

set(HEADERS_ACCELERATION
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/acceleration/bounding_sphere_hierarchy.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/acceleration/bounding_sphere.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/acceleration/kd_tree.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/acceleration/kd_tree.inl
)

set(HEADERS_MESH
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/mesh/triangle_mesh.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/mesh/entity_containers.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/mesh/entity_iterators.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/mesh/halfedge.hpp
)

set(HEADERS_GEOMETRY
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/geometry/mesh_distance.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/geometry/point_triangle_distance.hpp
)

set(HEADERS_UTILITY
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/utility/serialize.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid/utility/lru_cache.hpp

	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/utility/timing.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/utility/spinlock.hpp
)

set(SOURCES
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/discrete_grid.cpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/cubic_lagrange_discrete_grid.cpp
)

set(SOURCES_DATA
)

set(SOURCES_ACCELERATION
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/acceleration/bounding_sphere_hierarchy.cpp
)

set(SOURCES_MESH
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/mesh/entity_containers.cpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/mesh/entity_iterators.cpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/mesh/triangle_mesh.cpp
)

set(SOURCES_GEOMETRY
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/geometry/mesh_distance.cpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/geometry/point_triangle_distance.cpp
)

set(SOURCES_UTILITY
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/src/utility/timing.cpp
)

macro(SOURCEGROUP name)
	string(TOLOWER ${name} name_lower)
	string(SUBSTRING ${name_lower} 0 1 FIRST_LETTER)
	string(TOUPPER ${FIRST_LETTER} FIRST_LETTER)
	string(REGEX REPLACE "^.(.*)" "${FIRST_LETTER}\\1" dname "${name_lower}")
	source_group("Source Files ${dname}" FILES ${SOURCES_${name}})
	source_group("Header Files ${dname}" FILES ${HEADERS_${name}})
endmacro()
SOURCEGROUP(MESH)
SOURCEGROUP(ACCELERATION)
SOURCEGROUP(DATA)
SOURCEGROUP(GEOMETRY)
SOURCEGROUP(UTILITY)

# OpenMP support.
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen library.
find_package(Eigen3 REQUIRED)

# Set include directories.
include_directories(
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include/Discregrid
	${EIGEN3_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/../extern
)

# Disable stupid MSVC compiler warnings.
if(WIN32)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_USE_MATH_DEFINES)
endif(WIN32)

add_library(Discregrid SHARED
	${HEADERS}
	${SOURCES}
	${HEADERS_ACCELERATION}
	${SOURCES_ACCELERATION}
	${SOURCES_DATA}
	${HEADERS_MESH}
	${SOURCES_MESH}
	${HEADERS_GEOMETRY}
	${SOURCES_GEOMETRY}
	${HEADERS_UTILITY}
	${SOURCES_UTILITY}
)
install(TARGETS Discregrid
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib)

install(DIRECTORY include
DESTINATION .
PATTERN "*.hpp")

sofa_create_package("Discregrid" ${DISCREGRID_LIBRARY_VERSION} "Discregrid" "Discregrid")
