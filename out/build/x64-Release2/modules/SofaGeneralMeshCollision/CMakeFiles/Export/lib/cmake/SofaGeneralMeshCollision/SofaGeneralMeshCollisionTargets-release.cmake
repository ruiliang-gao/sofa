#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralMeshCollision" for configuration "Release"
set_property(TARGET SofaGeneralMeshCollision APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaGeneralMeshCollision PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaGeneralMeshCollision.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaGeneralMeshCollision.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralMeshCollision )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralMeshCollision "${_IMPORT_PREFIX}/lib/SofaGeneralMeshCollision.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralMeshCollision.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
