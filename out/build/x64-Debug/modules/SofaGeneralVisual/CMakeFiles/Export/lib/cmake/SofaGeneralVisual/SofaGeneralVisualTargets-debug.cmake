#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralVisual" for configuration "Debug"
set_property(TARGET SofaGeneralVisual APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralVisual PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralVisual_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralVisual_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralVisual )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralVisual "${_IMPORT_PREFIX}/lib/SofaGeneralVisual_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralVisual_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
