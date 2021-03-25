#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaEigen2Solver" for configuration "Debug"
set_property(TARGET SofaEigen2Solver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaEigen2Solver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaEigen2Solver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaEigen2Solver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaEigen2Solver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaEigen2Solver "${_IMPORT_PREFIX}/lib/SofaEigen2Solver_d.lib" "${_IMPORT_PREFIX}/bin/SofaEigen2Solver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
