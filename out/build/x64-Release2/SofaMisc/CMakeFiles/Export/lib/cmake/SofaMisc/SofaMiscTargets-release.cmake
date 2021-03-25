#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMisc" for configuration "Release"
set_property(TARGET SofaMisc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaMisc PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaMisc.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaMisc.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMisc )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMisc "${_IMPORT_PREFIX}/lib/SofaMisc.lib" "${_IMPORT_PREFIX}/bin/SofaMisc.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
