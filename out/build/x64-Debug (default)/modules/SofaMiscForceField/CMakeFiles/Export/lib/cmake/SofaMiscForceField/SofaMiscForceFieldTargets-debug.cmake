#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscForceField" for configuration "Debug"
set_property(TARGET SofaMiscForceField APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMiscForceField PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMiscForceField_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMiscForceField_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscForceField )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscForceField "${_IMPORT_PREFIX}/lib/SofaMiscForceField_d.lib" "${_IMPORT_PREFIX}/bin/SofaMiscForceField_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
