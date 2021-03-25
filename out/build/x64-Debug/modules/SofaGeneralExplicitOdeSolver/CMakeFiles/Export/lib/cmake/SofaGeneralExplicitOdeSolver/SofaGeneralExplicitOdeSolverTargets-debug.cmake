#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralExplicitOdeSolver" for configuration "Debug"
set_property(TARGET SofaGeneralExplicitOdeSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralExplicitOdeSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralExplicitOdeSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralExplicitOdeSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralExplicitOdeSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralExplicitOdeSolver "${_IMPORT_PREFIX}/lib/SofaGeneralExplicitOdeSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralExplicitOdeSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
