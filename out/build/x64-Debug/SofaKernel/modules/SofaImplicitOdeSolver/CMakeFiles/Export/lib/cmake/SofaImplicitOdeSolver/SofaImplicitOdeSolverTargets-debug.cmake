#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaImplicitOdeSolver" for configuration "Debug"
set_property(TARGET SofaImplicitOdeSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaImplicitOdeSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaImplicitOdeSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaImplicitOdeSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaImplicitOdeSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaImplicitOdeSolver "${_IMPORT_PREFIX}/lib/SofaImplicitOdeSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaImplicitOdeSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
