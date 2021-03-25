#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaTopologyMapping" for configuration "Debug"
set_property(TARGET SofaTopologyMapping APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaTopologyMapping PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaTopologyMapping_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaTopologyMapping_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaTopologyMapping )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaTopologyMapping "${_IMPORT_PREFIX}/lib/SofaTopologyMapping_d.lib" "${_IMPORT_PREFIX}/bin/SofaTopologyMapping_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
