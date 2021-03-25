#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaComponentAll" for configuration "Release"
set_property(TARGET SofaComponentAll APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaComponentAll PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaComponentAll.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaComponentAll.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaComponentAll )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaComponentAll "${_IMPORT_PREFIX}/lib/SofaComponentAll.lib" "${_IMPORT_PREFIX}/bin/SofaComponentAll.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
