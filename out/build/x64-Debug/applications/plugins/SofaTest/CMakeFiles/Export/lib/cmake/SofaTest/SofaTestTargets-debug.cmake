#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaTest" for configuration "Debug"
set_property(TARGET SofaTest APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaTest PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaTest_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaTest_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaTest )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaTest "${_IMPORT_PREFIX}/lib/SofaTest_d.lib" "${_IMPORT_PREFIX}/bin/SofaTest_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
