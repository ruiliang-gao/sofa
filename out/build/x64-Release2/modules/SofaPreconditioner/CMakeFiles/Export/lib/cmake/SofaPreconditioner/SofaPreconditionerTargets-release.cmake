#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPreconditioner" for configuration "Release"
set_property(TARGET SofaPreconditioner APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaPreconditioner PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaPreconditioner.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaPreconditioner.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaPreconditioner )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaPreconditioner "${_IMPORT_PREFIX}/lib/SofaPreconditioner.lib" "${_IMPORT_PREFIX}/bin/SofaPreconditioner.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
