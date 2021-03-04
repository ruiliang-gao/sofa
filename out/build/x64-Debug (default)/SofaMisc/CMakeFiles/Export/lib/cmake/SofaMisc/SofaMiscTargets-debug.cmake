#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMisc" for configuration "Debug"
set_property(TARGET SofaMisc APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMisc PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMisc_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMisc_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMisc )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMisc "${_IMPORT_PREFIX}/lib/SofaMisc_d.lib" "${_IMPORT_PREFIX}/bin/SofaMisc_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
