#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaLoader" for configuration "Debug"
set_property(TARGET SofaLoader APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaLoader PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaLoader_d.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "tinyxml"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaLoader_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaLoader )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaLoader "${_IMPORT_PREFIX}/lib/SofaLoader_d.lib" "${_IMPORT_PREFIX}/bin/SofaLoader_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
