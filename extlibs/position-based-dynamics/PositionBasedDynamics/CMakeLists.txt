set(PBD_POSITION_BASED_DYNAMICS_LIBRARY_MAJOR_VERSION 0)
set(PBD_POSITION_BASED_DYNAMICS_LIBRARY_MINOR_VERSION 1)
set(PBD_POSITION_BASED_DYNAMICS_LIBRARY_VERSION ${PBD_POSITION_BASED_DYNAMICS_LIBRARY_MAJOR_VERSION}.${PBD_POSITION_BASED_DYNAMICS_LIBRARY_MINOR_VERSION})

add_library(PositionBasedDynamics SHARED
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Common/Common.h
		
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/DirectPositionBasedSolverForStiffRodsInterface.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/MathFunctions.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/MathFunctions.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedDynamics.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedDynamics.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedElasticRods.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedElasticRods.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedFluids.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedFluids.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedRigidBodyDynamics.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedRigidBodyDynamics.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/PositionBasedGenericConstraints.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/SPHKernels.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/SPHKernels.h
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/TimeIntegration.cpp
		 ${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/PositionBasedDynamics/TimeIntegration.h
		
		 CMakeLists.txt
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

install(TARGETS PositionBasedDynamics
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY .
DESTINATION include/PositionBasedDynamics
FILES_MATCHING PATTERN "*.h")

sofa_create_package("PositionBasedDynamics" ${PBD_POSITION_BASED_DYNAMICS_LIBRARY_VERSION} "PositionBasedDynamics" "PositionBasedDynamics")
