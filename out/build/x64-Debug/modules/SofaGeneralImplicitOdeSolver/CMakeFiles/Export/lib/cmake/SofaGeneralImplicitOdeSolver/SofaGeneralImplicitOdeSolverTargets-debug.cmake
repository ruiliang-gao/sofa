#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralImplicitOdeSolver" for configuration "Debug"
set_property(TARGET SofaGeneralImplicitOdeSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralImplicitOdeSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralImplicitOdeSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralImplicitOdeSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralImplicitOdeSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralImplicitOdeSolver "${_IMPORT_PREFIX}/lib/SofaGeneralImplicitOdeSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralImplicitOdeSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
