# Eigen library.
find_package(Eigen3 REQUIRED)

# Set include directories.
include_directories(
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/extern
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/discregrid/include
	${EIGEN3_INCLUDE_DIR}
)

if(WIN32)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
endif(WIN32)

add_executable(GenerateDensityMap
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/cmd/generate_density_map/main.cpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/cmd/generate_density_map/gauss_quadrature.hpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/cmd/generate_density_map/gauss_quadrature.cpp
	${CMAKE_SOURCE_DIR}/extlibs/Discregrid/cmd/generate_density_map/sph_kernel.hpp
)

# OpenMP support.
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_dependencies(GenerateDensityMap
	Discregrid
)

target_link_libraries(GenerateDensityMap
	Discregrid
)

set_target_properties(GenerateDensityMap PROPERTIES FOLDER Cmd)
