#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscEngine" for configuration "Debug"
set_property(TARGET SofaMiscEngine APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMiscEngine PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMiscEngine_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMiscEngine_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscEngine )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscEngine "${_IMPORT_PREFIX}/lib/SofaMiscEngine_d.lib" "${_IMPORT_PREFIX}/bin/SofaMiscEngine_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
