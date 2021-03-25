#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaComponentAll" for configuration "Debug"
set_property(TARGET SofaComponentAll APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaComponentAll PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaComponentAll_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaComponentAll_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaComponentAll )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaComponentAll "${_IMPORT_PREFIX}/lib/SofaComponentAll_d.lib" "${_IMPORT_PREFIX}/bin/SofaComponentAll_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
