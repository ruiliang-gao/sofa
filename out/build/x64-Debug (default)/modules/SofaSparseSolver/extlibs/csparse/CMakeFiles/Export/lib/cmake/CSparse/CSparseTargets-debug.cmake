#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "csparse" for configuration "Debug"
set_property(TARGET csparse APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(csparse PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/csparse_d.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS csparse )
list(APPEND _IMPORT_CHECK_FILES_FOR_csparse "${_IMPORT_PREFIX}/lib/csparse_d.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
