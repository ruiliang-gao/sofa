#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaHaptics" for configuration "Release"
set_property(TARGET SofaHaptics APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaHaptics PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaHaptics.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaHaptics.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaHaptics )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaHaptics "${_IMPORT_PREFIX}/lib/SofaHaptics.lib" "${_IMPORT_PREFIX}/bin/SofaHaptics.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
