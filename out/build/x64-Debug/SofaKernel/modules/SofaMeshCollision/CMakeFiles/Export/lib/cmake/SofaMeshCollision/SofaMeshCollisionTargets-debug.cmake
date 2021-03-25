#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMeshCollision" for configuration "Debug"
set_property(TARGET SofaMeshCollision APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMeshCollision PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMeshCollision_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMeshCollision_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMeshCollision )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMeshCollision "${_IMPORT_PREFIX}/lib/SofaMeshCollision_d.lib" "${_IMPORT_PREFIX}/bin/SofaMeshCollision_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
