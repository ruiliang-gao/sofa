#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaValidation" for configuration "Debug"
set_property(TARGET SofaValidation APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaValidation PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaValidation_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaValidation_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaValidation )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaValidation "${_IMPORT_PREFIX}/lib/SofaValidation_d.lib" "${_IMPORT_PREFIX}/bin/SofaValidation_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
