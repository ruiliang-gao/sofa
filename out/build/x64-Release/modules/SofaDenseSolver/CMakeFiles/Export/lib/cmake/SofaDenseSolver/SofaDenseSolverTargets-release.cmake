#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaDenseSolver" for configuration "Release"
set_property(TARGET SofaDenseSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaDenseSolver PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaDenseSolver.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaDenseSolver.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaDenseSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaDenseSolver "${_IMPORT_PREFIX}/lib/SofaDenseSolver.lib" "${_IMPORT_PREFIX}/bin/SofaDenseSolver.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
