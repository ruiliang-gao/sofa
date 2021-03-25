#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "newmat" for configuration "Debug"
set_property(TARGET newmat APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(newmat PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/newmat_d.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS newmat )
list(APPEND _IMPORT_CHECK_FILES_FOR_newmat "${_IMPORT_PREFIX}/lib/newmat_d.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
