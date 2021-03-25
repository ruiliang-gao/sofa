#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaExplicitOdeSolver" for configuration "Debug"
set_property(TARGET SofaExplicitOdeSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaExplicitOdeSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaExplicitOdeSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaExplicitOdeSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaExplicitOdeSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaExplicitOdeSolver "${_IMPORT_PREFIX}/lib/SofaExplicitOdeSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaExplicitOdeSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
