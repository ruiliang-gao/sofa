#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscFem" for configuration "Debug"
set_property(TARGET SofaMiscFem APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMiscFem PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMiscFem_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMiscFem_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscFem )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscFem "${_IMPORT_PREFIX}/lib/SofaMiscFem_d.lib" "${_IMPORT_PREFIX}/bin/SofaMiscFem_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
