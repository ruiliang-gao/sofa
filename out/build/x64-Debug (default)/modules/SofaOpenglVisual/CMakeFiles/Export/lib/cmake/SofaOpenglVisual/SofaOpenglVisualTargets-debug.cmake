#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaOpenglVisual" for configuration "Debug"
set_property(TARGET SofaOpenglVisual APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaOpenglVisual PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaOpenglVisual_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaOpenglVisual_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaOpenglVisual )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaOpenglVisual "${_IMPORT_PREFIX}/lib/SofaOpenglVisual_d.lib" "${_IMPORT_PREFIX}/bin/SofaOpenglVisual_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
