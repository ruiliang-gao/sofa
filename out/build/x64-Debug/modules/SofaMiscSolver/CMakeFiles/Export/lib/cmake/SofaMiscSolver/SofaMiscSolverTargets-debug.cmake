#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMiscSolver" for configuration "Debug"
set_property(TARGET SofaMiscSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaMiscSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaMiscSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaMiscSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaMiscSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaMiscSolver "${_IMPORT_PREFIX}/lib/SofaMiscSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaMiscSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
