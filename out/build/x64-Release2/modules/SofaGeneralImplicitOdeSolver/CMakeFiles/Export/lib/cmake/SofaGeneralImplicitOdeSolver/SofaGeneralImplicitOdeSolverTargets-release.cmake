#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralImplicitOdeSolver" for configuration "Release"
set_property(TARGET SofaGeneralImplicitOdeSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaGeneralImplicitOdeSolver PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaGeneralImplicitOdeSolver.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaGeneralImplicitOdeSolver.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralImplicitOdeSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralImplicitOdeSolver "${_IMPORT_PREFIX}/lib/SofaGeneralImplicitOdeSolver.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralImplicitOdeSolver.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
