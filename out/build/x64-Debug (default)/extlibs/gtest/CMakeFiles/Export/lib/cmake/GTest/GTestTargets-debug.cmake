#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gtest" for configuration "Debug"
set_property(TARGET gtest APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gtest PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/gtest_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/gtest_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS gtest )
list(APPEND _IMPORT_CHECK_FILES_FOR_gtest "${_IMPORT_PREFIX}/lib/gtest_d.lib" "${_IMPORT_PREFIX}/bin/gtest_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
