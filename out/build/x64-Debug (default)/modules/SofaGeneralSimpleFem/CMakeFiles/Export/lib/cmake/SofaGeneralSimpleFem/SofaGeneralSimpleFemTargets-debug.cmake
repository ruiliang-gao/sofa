#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralSimpleFem" for configuration "Debug"
set_property(TARGET SofaGeneralSimpleFem APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralSimpleFem PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralSimpleFem_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralSimpleFem_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralSimpleFem )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralSimpleFem "${_IMPORT_PREFIX}/lib/SofaGeneralSimpleFem_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralSimpleFem_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
