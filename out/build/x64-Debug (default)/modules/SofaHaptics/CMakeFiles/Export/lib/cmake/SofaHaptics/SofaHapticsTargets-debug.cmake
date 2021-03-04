#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaHaptics" for configuration "Debug"
set_property(TARGET SofaHaptics APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaHaptics PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaHaptics_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaHaptics_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaHaptics )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaHaptics "${_IMPORT_PREFIX}/lib/SofaHaptics_d.lib" "${_IMPORT_PREFIX}/bin/SofaHaptics_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
