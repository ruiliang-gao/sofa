#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TestPlugin" for configuration "Debug"
set_property(TARGET TestPlugin APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(TestPlugin PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/TestPlugin_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/TestPlugin_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS TestPlugin )
list(APPEND _IMPORT_CHECK_FILES_FOR_TestPlugin "${_IMPORT_PREFIX}/lib/TestPlugin_d.lib" "${_IMPORT_PREFIX}/bin/TestPlugin_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
