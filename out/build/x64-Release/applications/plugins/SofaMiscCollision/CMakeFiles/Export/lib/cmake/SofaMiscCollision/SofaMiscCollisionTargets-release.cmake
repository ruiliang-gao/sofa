#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscCollision" for configuration "Release"
set_property(TARGET SofaMiscCollision APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaMiscCollision PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaMiscCollision.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaMiscCollision.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscCollision )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscCollision "${_IMPORT_PREFIX}/lib/SofaMiscCollision.lib" "${_IMPORT_PREFIX}/bin/SofaMiscCollision.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
