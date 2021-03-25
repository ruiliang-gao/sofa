#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPreconditioner" for configuration "Debug"
set_property(TARGET SofaPreconditioner APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaPreconditioner PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaPreconditioner_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaPreconditioner_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaPreconditioner )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaPreconditioner "${_IMPORT_PREFIX}/lib/SofaPreconditioner_d.lib" "${_IMPORT_PREFIX}/bin/SofaPreconditioner_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
