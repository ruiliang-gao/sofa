#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMeshCollision" for configuration "Release"
set_property(TARGET SofaMeshCollision APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaMeshCollision PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaMeshCollision.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaMeshCollision.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMeshCollision )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMeshCollision "${_IMPORT_PREFIX}/lib/SofaMeshCollision.lib" "${_IMPORT_PREFIX}/bin/SofaMeshCollision.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
