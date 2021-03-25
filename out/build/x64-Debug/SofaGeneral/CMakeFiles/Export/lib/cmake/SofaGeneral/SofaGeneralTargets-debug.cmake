#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneral" for configuration "Debug"
set_property(TARGET SofaGeneral APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneral PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneral_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneral_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneral )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneral "${_IMPORT_PREFIX}/lib/SofaGeneral_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneral_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
