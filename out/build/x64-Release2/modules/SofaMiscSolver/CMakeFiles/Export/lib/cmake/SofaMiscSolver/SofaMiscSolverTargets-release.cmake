#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscSolver" for configuration "Release"
set_property(TARGET SofaMiscSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaMiscSolver PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaMiscSolver.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaMiscSolver.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscSolver "${_IMPORT_PREFIX}/lib/SofaMiscSolver.lib" "${_IMPORT_PREFIX}/bin/SofaMiscSolver.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
