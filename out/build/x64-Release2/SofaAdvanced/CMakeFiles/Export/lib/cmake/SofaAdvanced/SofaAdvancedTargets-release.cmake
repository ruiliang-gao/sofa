#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaAdvanced" for configuration "Release"
set_property(TARGET SofaAdvanced APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaAdvanced PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaAdvanced.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaAdvanced.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaAdvanced )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaAdvanced "${_IMPORT_PREFIX}/lib/SofaAdvanced.lib" "${_IMPORT_PREFIX}/bin/SofaAdvanced.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
