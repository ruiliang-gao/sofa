#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralSimpleFem" for configuration "Release"
set_property(TARGET SofaGeneralSimpleFem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaGeneralSimpleFem PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaGeneralSimpleFem.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaGeneralSimpleFem.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralSimpleFem )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralSimpleFem "${_IMPORT_PREFIX}/lib/SofaGeneralSimpleFem.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralSimpleFem.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
