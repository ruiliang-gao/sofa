#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaBaseCollision" for configuration "Debug"
set_property(TARGET SofaBaseCollision APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaBaseCollision PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaBaseCollision_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaBaseCollision_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaBaseCollision )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaBaseCollision "${_IMPORT_PREFIX}/lib/SofaBaseCollision_d.lib" "${_IMPORT_PREFIX}/bin/SofaBaseCollision_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
