#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralTopology" for configuration "Release"
set_property(TARGET SofaGeneralTopology APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaGeneralTopology PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaGeneralTopology.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaGeneralTopology.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralTopology )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralTopology "${_IMPORT_PREFIX}/lib/SofaGeneralTopology.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralTopology.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
