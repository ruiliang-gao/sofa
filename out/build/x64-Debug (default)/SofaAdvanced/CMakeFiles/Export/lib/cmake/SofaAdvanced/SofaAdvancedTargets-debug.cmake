#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaAdvanced" for configuration "Debug"
set_property(TARGET SofaAdvanced APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaAdvanced PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaAdvanced_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaAdvanced_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaAdvanced )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaAdvanced "${_IMPORT_PREFIX}/lib/SofaAdvanced_d.lib" "${_IMPORT_PREFIX}/bin/SofaAdvanced_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
