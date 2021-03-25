#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaCommon" for configuration "Release"
set_property(TARGET SofaCommon APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaCommon PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaCommon.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaCommon.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaCommon )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaCommon "${_IMPORT_PREFIX}/lib/SofaCommon.lib" "${_IMPORT_PREFIX}/bin/SofaCommon.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
