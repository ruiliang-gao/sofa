#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaTopologyMapping" for configuration "Release"
set_property(TARGET SofaTopologyMapping APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaTopologyMapping PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaTopologyMapping.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaTopologyMapping.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaTopologyMapping )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaTopologyMapping "${_IMPORT_PREFIX}/lib/SofaTopologyMapping.lib" "${_IMPORT_PREFIX}/bin/SofaTopologyMapping.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
