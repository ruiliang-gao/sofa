#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralLoader" for configuration "Release"
set_property(TARGET SofaGeneralLoader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaGeneralLoader PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaGeneralLoader.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaGeneralLoader.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralLoader )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralLoader "${_IMPORT_PREFIX}/lib/SofaGeneralLoader.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralLoader.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
