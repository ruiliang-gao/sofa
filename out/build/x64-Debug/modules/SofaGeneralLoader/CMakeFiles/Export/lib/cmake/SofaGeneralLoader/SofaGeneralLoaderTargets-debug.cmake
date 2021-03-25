#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralLoader" for configuration "Debug"
set_property(TARGET SofaGeneralLoader APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralLoader PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralLoader_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralLoader_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralLoader )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralLoader "${_IMPORT_PREFIX}/lib/SofaGeneralLoader_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralLoader_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
