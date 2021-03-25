#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaValidation" for configuration "Release"
set_property(TARGET SofaValidation APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaValidation PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaValidation.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaValidation.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaValidation )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaValidation "${_IMPORT_PREFIX}/lib/SofaValidation.lib" "${_IMPORT_PREFIX}/bin/SofaValidation.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
