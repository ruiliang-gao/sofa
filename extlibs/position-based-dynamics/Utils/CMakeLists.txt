set(PBD_UTILS_LIBRARY_MAJOR_VERSION 0)
set(PBD_UTILS_LIBRARY_MINOR_VERSION 1)
set(PBD_UTILS_LIBRARY_VERSION ${PBD_UTILS_LIBRARY_MAJOR_VERSION}.${PBD_UTILS_LIBRARY_MINOR_VERSION})

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_LOCAL_CHANGES)
if (${GIT_LOCAL_CHANGES} STREQUAL "DIRTY")
	set(COMPILER_MESSAGE "#pragma WARNING(Local changes not committed.)")
endif()
configure_file("${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/Version.h.in" "${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/Utils/Version.h" @ONLY)

add_library(Utils SHARED
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/FileSystem.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/Hashmap.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/IndexedFaceMesh.cpp
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/IndexedFaceMesh.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/IndexedTetMesh.cpp
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/IndexedTetMesh.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/Logger.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/OBJLoader.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/SceneLoader.cpp
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/SceneLoader.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/StringTools.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/SystemInfo.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/TetGenLoader.cpp
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/TetGenLoader.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/Timing.h
	${CMAKE_BINARY_DIR}/extlibs/PositionBasedDynamics/Utils/Version.h
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/VolumeIntegration.cpp
	${CMAKE_SOURCE_DIR}/extlibs/PositionBasedDynamics/Utils/VolumeIntegration.h

	CMakeLists.txt
)

############################################################
# GenericParameters
############################################################
include_directories(${ExternalInstallDir}/GenericParameters/include)
# add_dependencies(Utils GenericParameters)

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

install(TARGETS Utils
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY .
DESTINATION include/Utils
FILES_MATCHING PATTERN "*.h")

sofa_create_package("Utils" ${PBD_UTILS_LIBRARY_VERSION} "Utils" "Utils")
