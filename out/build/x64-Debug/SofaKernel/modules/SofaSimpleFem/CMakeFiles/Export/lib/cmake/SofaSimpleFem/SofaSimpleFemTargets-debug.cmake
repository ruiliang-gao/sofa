#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaSimpleFem" for configuration "Debug"
set_property(TARGET SofaSimpleFem APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSimpleFem PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSimpleFem_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSimpleFem_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimpleFem )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimpleFem "${_IMPORT_PREFIX}/lib/SofaSimpleFem_d.lib" "${_IMPORT_PREFIX}/bin/SofaSimpleFem_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
