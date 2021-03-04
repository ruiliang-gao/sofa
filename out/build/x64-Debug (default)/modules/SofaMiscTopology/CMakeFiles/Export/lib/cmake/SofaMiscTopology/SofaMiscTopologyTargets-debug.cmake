#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscTopology" for configuration "Debug"
set_property(TARGET SofaMiscTopology APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMiscTopology PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMiscTopology_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMiscTopology_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscTopology )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscTopology "${_IMPORT_PREFIX}/lib/SofaMiscTopology_d.lib" "${_IMPORT_PREFIX}/bin/SofaMiscTopology_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
