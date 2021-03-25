#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaImplicitOdeSolver" for configuration "Release"
set_property(TARGET SofaImplicitOdeSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaImplicitOdeSolver PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaImplicitOdeSolver.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaImplicitOdeSolver.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaImplicitOdeSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaImplicitOdeSolver "${_IMPORT_PREFIX}/lib/SofaImplicitOdeSolver.lib" "${_IMPORT_PREFIX}/bin/SofaImplicitOdeSolver.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
