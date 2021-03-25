#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaBaseLinearSolver" for configuration "Debug"
set_property(TARGET SofaBaseLinearSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaBaseLinearSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaBaseLinearSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaBaseLinearSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaBaseLinearSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaBaseLinearSolver "${_IMPORT_PREFIX}/lib/SofaBaseLinearSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaBaseLinearSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
