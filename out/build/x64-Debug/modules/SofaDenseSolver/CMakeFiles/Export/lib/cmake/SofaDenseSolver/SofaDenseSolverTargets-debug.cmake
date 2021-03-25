#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaDenseSolver" for configuration "Debug"
set_property(TARGET SofaDenseSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaDenseSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaDenseSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaDenseSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaDenseSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaDenseSolver "${_IMPORT_PREFIX}/lib/SofaDenseSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaDenseSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
