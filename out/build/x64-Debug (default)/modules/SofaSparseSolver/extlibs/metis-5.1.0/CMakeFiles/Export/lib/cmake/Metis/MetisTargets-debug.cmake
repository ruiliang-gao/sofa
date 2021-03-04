#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metis" for configuration "Debug"
set_property(TARGET metis APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metis PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/metis_d.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS metis )
list(APPEND _IMPORT_CHECK_FILES_FOR_metis "${_IMPORT_PREFIX}/lib/metis_d.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
