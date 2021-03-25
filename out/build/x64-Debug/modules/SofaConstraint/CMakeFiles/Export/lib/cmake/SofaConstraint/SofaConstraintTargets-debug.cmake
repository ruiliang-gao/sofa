#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaConstraint" for configuration "Debug"
set_property(TARGET SofaConstraint APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaConstraint PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaConstraint_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaConstraint_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaConstraint )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaConstraint "${_IMPORT_PREFIX}/lib/SofaConstraint_d.lib" "${_IMPORT_PREFIX}/bin/SofaConstraint_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
