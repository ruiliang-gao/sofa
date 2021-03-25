#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaSparseSolver" for configuration "Debug"
set_property(TARGET SofaSparseSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSparseSolver PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSparseSolver_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSparseSolver_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSparseSolver )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSparseSolver "${_IMPORT_PREFIX}/lib/SofaSparseSolver_d.lib" "${_IMPORT_PREFIX}/bin/SofaSparseSolver_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
